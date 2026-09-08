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


class LLMError(RuntimeError):
    """Raised when the model call fails or returns something unusable."""


class LLMUnavailable(LLMError):
    """Raised when no provider is configured — callers fall back to local models."""


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


def get_client() -> LLMClient:
    """Process-wide singleton."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
