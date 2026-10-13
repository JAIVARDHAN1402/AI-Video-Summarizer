"""Large-language-model client for the understanding layer.

The local summarizers (BART, flan-t5) can condense text but cannot *teach* — fed
a spoken, often machine-translated transcript they produce disjointed,
quote-fragment prose. Producing an explanation a reader can learn from without
watching the video needs a capable instruction-following model.

This module is that seam. It speaks to Gemini over the REST API (no SDK
dependency — ``requests`` is already required for the optional Q&A backend) and
exposes one method, :meth:`LLMClient.generate_json`, that returns parsed JSON
constrained to a caller-supplied schema. Swapping in OpenAI or a local
Ollama-served model later means adding a sibling client with the same method;
nothing above this file knows which provider answered.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Gemini Flash accepts ~1M input tokens, but latency and cost still scale with
# the prompt, and a lecture transcript is nowhere near that. Cap the assembled
# context well below the ceiling; callers trim to fit and note when they did.
MAX_CONTEXT_CHARS = 600_000

# The ceiling a retry may grow the output cap to after a reply was cut off.
# Above the model's own limit the request is rejected outright, so this
# stays within what the free tiers advertise.
MAX_GROWN_OUTPUT_TOKENS = 32_768

# (caption, jpeg_bytes) pair for a keyframe sent to a multimodal model.
ImageRef = tuple[str, bytes]


def _retry_delay_sec(response) -> float:
    """The wait Gemini asks for in a 429 body, if it gave one.

    It arrives as a RetryInfo detail, e.g. {"retryDelay": "31s"}.
    """
    try:
        for detail in (response.json().get("error", {}).get("details") or []):
            raw = detail.get("retryDelay")
            if raw:
                return min(float(str(raw).rstrip("s")), 120.0)
    except Exception:
        pass
    return 0.0


def _quota_is_exhausted(response) -> bool:
    """Whether a 429 means the quota is spent rather than momentarily busy.

    The two are not the same failure and must not be treated alike. A
    per-minute limit clears in under a minute and is worth waiting for; a
    per-day quota does not clear today, and waiting for it is pure loss.

    Gemini distinguishes them in the error body: a QuotaFailure violation
    names the quota it belongs to, and the daily ones say so in the id. An
    11-hour video is 28 calls, and treating a spent daily quota as transient
    cost 27 x 3 retries x 120 s of sleeping before the run gave up — close to
    three hours of doing nothing.
    """
    try:
        body = response.json()
        error = body.get("error") or {}
        for detail in error.get("details") or []:
            for violation in detail.get("violations") or []:
                quota_id = str(violation.get("quotaId") or "")
                if "PerDay" in quota_id:
                    return True

        # OpenRouter says it differently: a daily limit source, and a headers
        # block whose remaining count is zero. Missing this cost twelve
        # requests against a quota that was already spent, each preceded by a
        # wait for it to clear.
        meta = error.get("metadata") or {}
        if "daily" in str(meta.get("limit_source") or "").lower():
            return True
        headers = meta.get("headers") or {}
        if str(headers.get("X-RateLimit-Remaining", "")).strip() == "0":
            return True

        text = response.text.lower()
        # Fallback on wording when no structured detail is present.
        if "per-day" in text or "per day" in text:
            return True
        return "quota" in text and "billing" in text
    except Exception:
        return False


def quota_resets_at(response) -> Optional[str]:
    """When a spent quota comes back, as the provider reports it.

    Worth carrying to the screen: "try again later" is not actionable, and
    the reset is a plain epoch in the headers the gateway echoes back.
    """
    try:
        headers = ((response.json().get("error") or {}).get("metadata") or {}).get("headers") or {}
        raw = str(headers.get("X-RateLimit-Reset") or "").strip()
        if not raw.isdigit():
            return None
        seconds = int(raw) / 1000 if len(raw) > 11 else int(raw)
        return time.strftime("%H:%M on %d %b", time.localtime(seconds))
    except Exception:
        return None


class LLMError(RuntimeError):
    """Raised when the model call fails or returns something unusable."""


class LLMUnavailable(LLMError):
    """Raised when no provider is configured — callers fall back to local models."""


def _request_exceeds_rate_limit(response) -> bool:
    """Whether a 429 says this one request can never fit the rate limit.

    Free tiers meter tokens per minute as well as per day, and a single window
    of a long transcript can be larger than the whole per-minute allowance.
    Waiting does not help here: the request is too big, not too early.
    """
    try:
        text = response.text.lower()
    except Exception:
        return False
    return "request too large" in text or (
        "tokens per minute" in text and "requested" in text)


def _refused_the_images(response) -> bool:
    """Whether a rejection is about the attached keyframes specifically.

    Worth separating from a malformed request: the same prompt without the
    images will succeed, so the run can continue with a smaller explanation
    rather than failing.
    """
    if response.status_code not in (400, 404, 415, 422):
        return False
    try:
        text = response.text.lower()
    except Exception:
        return False
    return any(s in text for s in (
        "image", "vision", "multimodal", "modality", "image_url"))


class LLMQuotaExceeded(LLMError):
    """Raised when the provider's quota is spent for the period.

    Separate from LLMError because the right response differs: a caller making
    many calls in sequence should abandon the rest rather than attempt each
    one, and a caller with an expensive local fallback should weigh whether
    running it is worth more than failing quickly.
    """


@dataclass
class LLMResult:
    data: dict[str, Any]
    model: str
    finish_reason: str
    prompt_chars: int
    elapsed_sec: float
    truncated_input: bool


class LLMClient:
    """Provider-agnostic entry point. Today it only wires up Gemini."""

    def __init__(self) -> None:
        self._api_key = config.GEMINI_API_KEY
        self._model = config.UNDERSTANDING_MODEL

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model(self) -> str:
        return self._model

    def generate_json(
        self,
        system_instruction: str,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        thinking_budget: int = 0,
        images: Optional[list["ImageRef"]] = None,
        retries: int = 3,
    ) -> LLMResult:
        """Return the model's reply parsed as JSON.

        Args:
            system_instruction: The persona / task framing. Kept separate from
                the user prompt so the model weights it as instruction, not data.
            prompt: The material to work from (transcript, scene list, ...).
            images: Keyframes to send alongside the text, as (caption, jpeg_bytes)
                pairs. The model reads slides, diagrams, handwriting and code
                that OCR alone misses. Each is ~260+ tokens, so callers cap
                the count.
            schema: An optional JSON Schema the reply must conform to. Gemini
                enforces this server-side, which removes almost all of the
                "model returned prose around the JSON" failure mode.
            temperature: Low by default — this is summarisation, not ideation.
            max_output_tokens: Hard ceiling on the reply length.
            retries: Attempts for transient (429 / 5xx) failures.

        Raises:
            LLMUnavailable: no API key configured.
            LLMError: the call failed, was safety-blocked, or returned non-JSON.
        """
        if not self.available:
            raise LLMUnavailable("no GEMINI_API_KEY configured")

        import requests

        truncated = len(prompt) > MAX_CONTEXT_CHARS
        if truncated:
            logger.warning(
                "prompt is %d chars; trimming to %d before sending",
                len(prompt), MAX_CONTEXT_CHARS,
            )
            prompt = prompt[:MAX_CONTEXT_CHARS]

        # Gemini 3.x Flash is a reasoning model. Left unconstrained it burns
        # the whole output budget on hidden "thinking" and returns an empty
        # reply; thinkingBudget=0 gives a direct answer, a small positive
        # budget lets it plan structure first.
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": max(thinking_budget, 0)},
        }
        if schema is not None:
            generation_config["responseSchema"] = schema

        user_parts: list[dict[str, Any]] = [{"text": prompt}]
        for caption, jpeg in images or []:
            user_parts.append({"text": f"\n[Image — {caption}]"})
            user_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                }
            })

        body = {
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"role": "user", "parts": user_parts}],
            "generationConfig": generation_config,
            # Loosen the default safety filters: lecture and meeting audio
            # legitimately discusses topics (security exploits, medical detail,
            # history) that the strict defaults sometimes block.
            "safetySettings": [
                {"category": c, "threshold": "BLOCK_ONLY_HIGH"}
                for c in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                )
            ],
        }

        url = _GEMINI_ENDPOINT.format(model=self._model)
        last_error: Optional[str] = None
        started = time.perf_counter()

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(
                    url, params={"key": self._api_key}, json=body, timeout=180
                )
            except requests.RequestException as exc:
                last_error = f"network error: {exc}"
                logger.warning("LLM attempt %d/%d failed: %s", attempt, retries, last_error)
                time.sleep(2 * attempt)
                continue

            if response.status_code == 200:
                return self._parse(response.json(), len(prompt), time.perf_counter() - started, truncated)

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                # A rate limit is not a transient blip to back off from in
                # seconds: the quota is per minute, so an exponential ramp that
                # tops out around six seconds just burns the retries and fails.
                # Gemini says how long to wait in the error body; honour it.
                if response.status_code == 429 and _quota_is_exhausted(response):
                    # Nothing to wait for. Sleeping here does not get the
                    # quota back; it only delays telling the caller.
                    raise LLMQuotaExceeded(
                        "the provider's quota is exhausted for this period "
                        f"(HTTP 429): {response.text[:200]}"
                    )
                if response.status_code == 429:
                    delay = _retry_delay_sec(response) or min(30 * attempt, 90)
                else:
                    delay = min(2 ** attempt, 30)
                logger.warning("LLM attempt %d/%d transient (%s); waiting %.0fs",
                               attempt, retries, last_error[:120], delay)
                time.sleep(delay)
                continue

            # 400 and friends are our fault (bad key, bad schema); don't retry.
            raise LLMError(f"Gemini rejected the request (HTTP {response.status_code}): "
                           f"{response.text[:400]}")

        raise LLMError(f"Gemini call failed after {retries} attempts. Last error: {last_error}")

    # --- response handling --------------------------------------------------

    def _parse(
        self, payload: dict[str, Any], prompt_chars: int, elapsed: float, truncated: bool
    ) -> LLMResult:
        block_reason = (payload.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise LLMError(f"prompt was blocked by the safety filter ({block_reason})")

        candidates = payload.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates")

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason", "STOP")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts).strip()

        if finish_reason == "SAFETY":
            raise LLMError("the reply was blocked by the safety filter")
        if finish_reason in ("MAX_TOKENS", "MALFORMED_FUNCTION_CALL") and not text:
            raise LLMError(
                "the model produced no text before hitting the token limit "
                "(raise max_output_tokens or lower thinking_budget)"
            )
        if not text:
            raise LLMError(f"Gemini returned an empty reply (finishReason={finish_reason})")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # With responseMimeType=application/json this should not happen, but
            # a MAX_TOKENS cut can truncate valid JSON mid-structure.
            raise LLMError(
                f"could not parse the reply as JSON (finishReason={finish_reason}): {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMError(f"expected a JSON object, got {type(data).__name__}")

        return LLMResult(
            data=data,
            model=self._model,
            finish_reason=finish_reason,
            prompt_chars=prompt_chars,
            elapsed_sec=round(elapsed, 2),
            truncated_input=truncated,
        )


_client: Optional[LLMClient] = None


class ChainedClient:
    """Two providers, the second used only once the first is spent.

    Both free tiers are generous enough for a day of work and neither is
    enough on its own: an eleven-hour video is seventeen calls, and the
    accounts are metered per day. Keeping one configured as a spare is what
    turns "come back tomorrow" into "this one takes a little longer".

    Only a spent quota hands over. A model that returned bad JSON, or an
    endpoint having a bad minute, is the first provider's own business and it
    retries that itself; switching on those would quietly spend the spare on
    problems that were about to resolve.
    """

    def __init__(self, primary, spare) -> None:
        self._primary = primary
        self._spare = spare
        self._primary_spent = False

    @property
    def available(self) -> bool:
        return self._primary.available or self._spare.available

    @property
    def model(self) -> str:
        return self._current.model

    @property
    def _current(self):
        return self._spare if self._primary_spent else self._primary

    def generate_json(self, *args, **kwargs) -> LLMResult:
        if not self._primary_spent:
            try:
                return self._primary.generate_json(*args, **kwargs)
            except LLMQuotaExceeded as exc:
                if not self._spare.available:
                    raise
                logger.warning(
                    "%s is out of quota (%s); the rest of this run uses %s",
                    self._primary.model, str(exc)[:160], self._spare.model,
                )
                self._primary_spent = True
        return self._spare.generate_json(*args, **kwargs)


def get_client():
    """Process-wide singleton for whichever providers are configured.

    An explicitly configured OpenAI-compatible endpoint wins over Gemini,
    because setting one is a deliberate act. When both are configured the
    other becomes the spare rather than being ignored, since running out of
    one of them part-way through a long video is the normal case, not an
    exceptional one.
    """
    global _client
    if _client is None:
        openai_compatible = OpenAICompatibleClient()
        gemini = LLMClient()
        if openai_compatible.available and gemini.available:
            _client = ChainedClient(openai_compatible, gemini)
            logger.info("LLM provider: %s, falling back to %s when its quota is spent",
                        openai_compatible.model, gemini.model)
        else:
            _client = openai_compatible if openai_compatible.available else gemini
            logger.info("LLM provider: %s (%s)",
                        type(_client).__name__,
                        _client.model if _client.available else "unconfigured")
    return _client


def reset_client() -> None:
    """Drop the cached client so a configuration change takes effect."""
    global _client
    _client = None


# =============================================================================
# OpenAI-compatible providers
# =============================================================================
#
# Groq, OpenRouter, Cerebras, Together, Mistral and a locally served Ollama
# all speak the same chat-completions shape, so one client reaches all of
# them. That matters because several have a usable free tier with no card,
# and the alternative when a quota runs out is the local BART summariser —
# which on a long video is both far worse and far slower than a smaller
# hosted model would be.
#
# Two things differ from Gemini and are handled here rather than pushed up:
#
#   Schema. Gemini enforces a JSON Schema server-side. Most of these accept
#   only {"type": "json_object"}, so the schema is described in the system
#   message instead and the reply is validated on arrival. Weaker, but the
#   callers already treat a malformed reply as a failure.
#
#   Images. The free text models are not multimodal. Keyframes are dropped
#   with one warning rather than silently, because losing them changes what
#   the explanation can say about slides and diagrams.


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-shaped endpoint."""

    def __init__(self) -> None:
        self._api_key = config.LLM_API_KEY
        self._base_url = (config.LLM_BASE_URL or "").rstrip("/")
        # A comma-separated VS_LLM_MODEL is a preference order, not a pool.
        # Free endpoints go down for stretches — ten of thirteen windows of one
        # video failed on a model that had been answering fine an hour before —
        # and naming a second one turns that from a failed run into a slower
        # one. The first is used until it stops answering.
        self._models = [m.strip() for m in (config.LLM_MODEL or "").split(",") if m.strip()]
        self._model = self._models[0] if self._models else ""
        self._warned_images = False

    @property
    def available(self) -> bool:
        return bool(self._api_key and self._base_url and self._models)

    @property
    def model(self) -> str:
        return self._model

    def _reasoning(self, thinking_budget: int) -> Optional[dict[str, Any]]:
        """The provider switch for hidden reasoning, when it takes one.

        Reasoning is switched off here whatever budget the caller asked for,
        which is not what thinking_budget means on the Gemini client. There it
        buys a separate allowance to plan with. Here the thinking is charged
        against the same max_tokens as the answer, and the budget is advisory:
        asking for 4096 was measured spending 6385 and leaving an unparseable
        reply, so a caller that wanted a better answer got no answer at all.
        VS_LLM_REASONING=1 turns it back on for anyone who wants it, and then
        the budget is added to the cap rather than taken out of it.

        Only OpenRouter documents the field, so it is not sent blind to an
        endpoint that might reject an unknown key.
        """
        forced = config.LLM_REASONING
        if forced is None and "openrouter.ai" not in self._base_url:
            return None
        if forced in ("1", "true", "True") and thinking_budget > 0:
            return {"max_tokens": thinking_budget}
        return {"enabled": False}

    def _output_cap(self, max_output_tokens: int, thinking_budget: int) -> int:
        """How many tokens to allow, given the reply has to fit in them.

        When reasoning is on the model spends this budget twice over, so the
        requested thinking is added rather than silently eaten out of the
        answer.
        """
        cap = min(max_output_tokens, config.LLM_MAX_OUTPUT_TOKENS)
        reasoning = self._reasoning(thinking_budget)
        if reasoning and "max_tokens" in reasoning:
            cap += int(reasoning["max_tokens"])
        return cap

    @staticmethod
    def _user_content(prompt: str, images: list["ImageRef"]) -> Any:
        """The user message, as plain text or as vision content parts.

        A model that takes images wants a list of parts; one that does not
        wants a bare string, and sending it a single-element list is a needless
        difference. Each keyframe is captioned first so the model can tie it to
        a timestamp, matching how the Gemini client sends them.
        """
        if not images:
            return prompt
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for caption, jpeg in images:
            encoded = base64.b64encode(jpeg).decode("ascii")
            parts.append({"type": "text", "text": f"\n[Image — {caption}]"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        return parts

    def generate_json(
        self,
        system_instruction: str,
        prompt: str,
        *,
        schema: Optional[dict[str, Any]] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        thinking_budget: int = 0,
        images: Optional[list["ImageRef"]] = None,
        retries: int = 3,
    ) -> LLMResult:
        if not self.available:
            raise LLMUnavailable("no OpenAI-compatible provider configured")

        import requests

        images = list(images or [])
        if images and not config.LLM_SEND_IMAGES:
            if not self._warned_images:
                logger.warning(
                    "VS_LLM_SEND_IMAGES is off, so %d keyframes are not being "
                    "sent; the explanation can only describe what is on screen "
                    "through the OCR text", len(images),
                )
                self._warned_images = True
            images = []

        truncated = len(prompt) > MAX_CONTEXT_CHARS
        if truncated:
            logger.warning("prompt is %d chars; trimming to %d before sending",
                           len(prompt), MAX_CONTEXT_CHARS)
            prompt = prompt[:MAX_CONTEXT_CHARS]

        system = system_instruction
        if schema:
            system += (
                "\n\nReply with a single JSON object and nothing else — no "
                "prose, no code fence. It must match this JSON Schema:\n"
                + json.dumps(schema, separators=(",", ":"))
            )

        def build(with_images: list["ImageRef"]) -> dict[str, Any]:
            body: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",
                     "content": self._user_content(prompt, with_images)},
                ],
                "temperature": temperature,
                "max_tokens": self._output_cap(max_output_tokens, thinking_budget),
            }
            if schema:
                body["response_format"] = {"type": "json_object"}
            reasoning = self._reasoning(thinking_budget)
            if reasoning is not None:
                body["reasoning"] = reasoning
            return body

        body = build(images)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        started = time.perf_counter()
        failures: list[str] = []
        for model_index, model_name in enumerate(self._models):
            self._model = model_name
            body["model"] = model_name
            try:
                return self._send(
                    url, headers, body, build, images, prompt, truncated,
                    started, retries,
                )
            except (LLMQuotaExceeded, LLMUnavailable):
                raise
            except LLMError as exc:
                failures.append(f"{model_name}: {exc}")
                if model_index + 1 < len(self._models):
                    logger.warning(
                        "%s is not answering (%s); falling back to %s",
                        model_name, str(exc)[:140], self._models[model_index + 1],
                    )

        if len(failures) == 1:
            raise LLMError(failures[0].split(": ", 1)[1])
        raise LLMError("every configured model failed — " + " | ".join(failures))

    def _send(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
        build, images: list["ImageRef"], prompt: str, truncated: bool,
        started: float, retries: int,
    ) -> LLMResult:
        """One model's attempts: retries, growing the cap, dropping images."""
        import requests

        last_error = ""
        grown = False
        for attempt in range(1, retries + 2):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=600)
            except Exception as exc:                      # network-level
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("LLM attempt %d/%d failed to connect (%s)",
                               attempt, retries, last_error[:120])
                time.sleep(min(2 ** attempt, 30))
                continue

            if response.status_code == 200:
                payload = response.json()
                upstream = self._upstream_failed(payload)
                if upstream and attempt <= retries:
                    last_error = f"HTTP 200 but {upstream}"
                    delay = min(2 ** attempt, 30)
                    logger.warning(
                        "LLM attempt %d/%d: %s; waiting %.0fs",
                        attempt, retries, upstream, delay,
                    )
                    time.sleep(delay)
                    continue
                # How much a window needs varies with how dense it is: the
                # same cap that holds one part of a video comfortably cuts the
                # next one off mid-object, and that window is then lost from
                # the explanation entirely. Ask again with room rather than
                # making the cap large for every call.
                if self._was_cut_off(payload) and not grown:
                    grown = True
                    body["max_tokens"] = min(body["max_tokens"] * 4,
                                             MAX_GROWN_OUTPUT_TOKENS)
                    logger.warning(
                        "%s hit the output limit; retrying this call with "
                        "max_tokens=%d", self._model, body["max_tokens"],
                    )
                    continue
                return self._parse(
                    payload, len(prompt),
                    time.perf_counter() - started, truncated,
                )

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code == 429 and _quota_is_exhausted(response):
                    resets = quota_resets_at(response)
                    raise LLMQuotaExceeded(
                        "the provider's quota is spent for this period"
                        + (f"; it resets at {resets}" if resets else "")
                        + f" (HTTP 429): {response.text[:200]}"
                    )
                if response.status_code == 429 and _request_exceeds_rate_limit(response):
                    raise LLMError(
                        "this request is larger than the provider's per-minute "
                        "token allowance, so retrying cannot help. Lower "
                        "VS_UNDERSTANDING_WINDOW_MIN so each window sends less, "
                        f"or use a provider with a higher limit: {response.text[:200]}"
                    )
                delay = _retry_delay_sec(response) or min(2 ** attempt, 30)
                logger.warning("LLM attempt %d/%d transient (%s); waiting %.0fs",
                               attempt, retries, last_error[:120], delay)
                time.sleep(delay)
                continue

            # A text-only model rejects the image parts outright. Which free
            # models take images changes, so rather than making the user find
            # out from a stack trace, drop them and ask again as text.
            if images and _refused_the_images(response):
                logger.warning(
                    "%s will not accept keyframes (HTTP %d), so the rest of "
                    "this run sends transcript and OCR text only: %s",
                    self._model, response.status_code, response.text[:200],
                )
                images, body = [], build([])
                self._warned_images = True
                continue

            raise LLMError(
                f"{self._base_url} rejected the request "
                f"(HTTP {response.status_code}): {response.text[:400]}"
            )

        raise LLMError(f"call failed after {retries} attempts. Last error: {last_error}")

    @staticmethod
    def _upstream_failed(payload: dict[str, Any]) -> str:
        """Why a 200 carries no usable answer, or "" when it does.

        A gateway in front of several providers reports the provider's trouble
        in the body of an otherwise successful response: no choices at all, or
        a choice whose finish_reason is "error". Both were fatal on the first
        occurrence, which is the wrong reading — they are the same transient
        upstream hiccup that a 502 would be, and the next attempt usually
        works.
        """
        if payload.get("error"):
            return str(payload["error"])[:200]
        choices = payload.get("choices") or []
        if not choices:
            return "the provider returned no choices"
        if choices[0].get("finish_reason") == "error":
            return "the provider stopped with an error part-way through"
        return ""

    @staticmethod
    def _was_cut_off(payload: dict[str, Any]) -> bool:
        """Whether the reply stopped because it ran out of output budget."""
        choices = payload.get("choices") or []
        if not choices:
            return False
        if choices[0].get("finish_reason") != "length":
            return False
        text = ((choices[0].get("message") or {}).get("content") or "").strip()
        if not text:
            return True
        try:
            json.loads(text)
        except json.JSONDecodeError:
            return True
        return False      # it finished the object and then ran out; usable

    def _parse(
        self, payload: dict[str, Any], prompt_chars: int, elapsed: float, truncated: bool
    ) -> LLMResult:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("the provider returned no choices")
        choice = choices[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        finish = choice.get("finish_reason") or ""
        if not text:
            raise LLMError(f"the provider returned an empty reply (finish_reason={finish})")

        # Some models wrap JSON in a fence despite being asked not to.
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # A truncated reply is the usual cause, and says the output cap is
            # too low for what was asked — worth saying so plainly.
            hint = (
                " The reply hit the output limit, so it was cut mid-JSON; "
                "lower VS_UNDERSTANDING_MAX_TOKENS or raise the window count."
                if finish == "length" else ""
            )
            raise LLMError(
                f"the provider did not return valid JSON (finish_reason={finish}).{hint} "
                f"First 200 chars: {text[:200]}"
            ) from exc

        if not isinstance(data, dict):
            raise LLMError("the provider returned JSON that is not an object")

        return LLMResult(
            data=data,
            model=payload.get("model") or self._model,
            finish_reason=finish,
            prompt_chars=prompt_chars,
            elapsed_sec=elapsed,
            truncated_input=truncated,
        )
