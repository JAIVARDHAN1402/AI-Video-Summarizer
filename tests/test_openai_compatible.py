"""The OpenAI-compatible provider path.

Groq, OpenRouter, Cerebras, Together, Mistral and a local Ollama all speak
this shape, and several have a free tier with no card. The parsing is what
matters: unlike Gemini these endpoints do not enforce a JSON Schema
server-side, so the reply arrives however the model felt like sending it.
"""

from __future__ import annotations

import pytest

import config
from core import llm


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "some-free-model")
    return llm.OpenAICompatibleClient()


def reply(content: str, finish: str = "stop") -> dict:
    return {
        "model": "some-free-model",
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
    }


def test_it_is_available_only_when_fully_configured(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", None)
    monkeypatch.setattr(llm.config, "LLM_API_KEY", None)
    monkeypatch.setattr(llm.config, "LLM_MODEL", None)
    assert llm.OpenAICompatibleClient().available is False

    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "m")
    assert llm.OpenAICompatibleClient().available is True


def test_a_plain_json_reply_is_parsed(client):
    result = client._parse(reply('{"overview": "a talk about trees"}'), 100, 1.0, False)
    assert result.data == {"overview": "a talk about trees"}
    assert result.model == "some-free-model"


def test_a_fenced_reply_is_unwrapped(client):
    """Models wrap JSON in a code fence despite being told not to."""
    fenced = '```json\n{"overview": "fenced anyway"}\n```'
    assert client._parse(reply(fenced), 100, 1.0, False).data == {"overview": "fenced anyway"}


def test_a_reply_cut_off_by_the_output_cap_says_so(client):
    """finish_reason=length is the usual cause of broken JSON, and the
    message should point at the cap rather than blaming the model."""
    with pytest.raises(llm.LLMError) as exc:
        client._parse(reply('{"overview": "it was cut off mid', "length"), 100, 1.0, False)
    assert "output limit" in str(exc.value)


def test_non_json_prose_is_an_error_not_a_silent_empty(client):
    with pytest.raises(llm.LLMError):
        client._parse(reply("Sure! Here is your summary:"), 100, 1.0, False)


def test_a_json_array_is_rejected(client):
    """Callers index the reply by key; a list would fail further away."""
    with pytest.raises(llm.LLMError):
        client._parse(reply('[{"overview": "wrong shape"}]'), 100, 1.0, False)


def test_an_empty_reply_is_an_error(client):
    with pytest.raises(llm.LLMError):
        client._parse(reply(""), 100, 1.0, False)


def test_no_choices_is_an_error(client):
    with pytest.raises(llm.LLMError):
        client._parse({"choices": []}, 100, 1.0, False)


def test_a_configured_endpoint_wins_over_gemini(monkeypatch):
    """Setting one is deliberate — usually because the Gemini quota ran out."""
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm.config, "GEMINI_API_KEY", "also-set")
    llm.reset_client()
    try:
        assert isinstance(llm.get_client(), llm.OpenAICompatibleClient)
    finally:
        llm.reset_client()


def test_gemini_is_used_when_no_endpoint_is_configured(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", None)
    monkeypatch.setattr(llm.config, "LLM_API_KEY", None)
    monkeypatch.setattr(llm.config, "LLM_MODEL", None)
    monkeypatch.setattr(llm.config, "GEMINI_API_KEY", "gemini-key")
    llm.reset_client()
    try:
        assert isinstance(llm.get_client(), llm.LLMClient)
    finally:
        llm.reset_client()


class RateLimitResponse:
    """A 429 whose body says the request itself is too big."""

    def __init__(self, text: str):
        self.text = text
        self.status_code = 429

    def json(self):
        return {"error": {"message": self.text, "code": "rate_limit_exceeded"}}


def test_an_oversized_request_is_not_retried():
    """A window of a long transcript can exceed the whole per-minute token
    allowance. That is permanent, so waiting three times is wasted."""
    body = ("Request too large for model `some-free-model` on tokens per "
            "minute (TPM): Limit 8000, Requested 11500.")
    assert llm._request_exceeds_rate_limit(RateLimitResponse(body)) is True


def test_a_plain_busy_429_is_still_retried():
    assert llm._request_exceeds_rate_limit(
        RateLimitResponse("Rate limit reached, please slow down")) is False


def test_an_oversized_request_is_not_mistaken_for_a_spent_quota():
    """The two need different handling: one narrows the window, the other
    waits for tomorrow."""
    body = "Request too large for model on tokens per minute (TPM): Requested 11500."
    assert llm._quota_is_exhausted(RateLimitResponse(body)) is False


def test_the_completion_request_is_clamped_to_the_provider_ceiling(client, monkeypatch):
    """UNDERSTANDING_MAX_OUTPUT_TOKENS is sized for Gemini's 32k. Sending that
    to a model whose ceiling is 8192 is a 400, not a shorter reply."""
    monkeypatch.setattr(llm.config, "LLM_MAX_OUTPUT_TOKENS", 8192)
    sent = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "ok"}')

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return FakeResponse()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    client.generate_json("sys", "prompt", max_output_tokens=32768)
    assert sent["max_tokens"] == 8192

    client.generate_json("sys", "prompt", max_output_tokens=2048)
    assert sent["max_tokens"] == 2048, "a smaller ask must not be raised"


# --- keyframes ------------------------------------------------------------
# Several of OpenRouter's free models accept images, so the multimodal path
# matters here too; the complication is that which ones do changes over time.

JPEG = b"\xff\xd8\xff\xe0 not really a jpeg \xff\xd9"


def test_a_text_only_prompt_stays_a_plain_string():
    """Wrapping one text part in a list is a needless difference from what
    every provider documents."""
    assert llm.OpenAICompatibleClient._user_content("hello", []) == "hello"


def test_keyframes_become_captioned_vision_parts():
    parts = llm.OpenAICompatibleClient._user_content(
        "explain this", [("scene at 01:23", JPEG)])
    assert parts[0] == {"type": "text", "text": "explain this"}
    assert "01:23" in parts[1]["text"], "the caption ties the frame to a time"
    assert parts[2]["type"] == "image_url"
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_images_can_be_turned_off_for_a_text_only_model(client, monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_SEND_IMAGES", False)
    sent = {}

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "ok"}')

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        (sent.update(json), Ok())[1])

    client.generate_json("sys", "prompt", images=[("scene at 00:10", JPEG)])
    assert sent["messages"][1]["content"] == "prompt", "no image parts"


def test_a_model_that_refuses_images_is_retried_without_them(client, monkeypatch):
    """The run should lose the visual detail, not fail outright."""
    bodies = []

    class Refusal:
        status_code = 400
        text = "This model does not support image input"

        def json(self):
            return {"error": {"message": Refusal.text}}

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "ok"}')

    def fake_post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return Refusal() if len(bodies) == 1 else Ok()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    result = client.generate_json("sys", "prompt", images=[("scene at 00:10", JPEG)])
    assert result.data == {"overview": "ok"}
    assert isinstance(bodies[0]["messages"][1]["content"], list), "images first"
    assert bodies[1]["messages"][1]["content"] == "prompt", "then text only"


def test_an_unrelated_400_is_still_an_error(client, monkeypatch):
    """Dropping the images must not swallow a genuinely bad request."""
    class BadRequest:
        status_code = 400
        text = "model `some-free-model` does not exist"

        def json(self):
            return {"error": {"message": BadRequest.text}}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None: BadRequest())

    with pytest.raises(llm.LLMError, match="does not exist"):
        client.generate_json("sys", "prompt", images=[("scene at 00:10", JPEG)])


# --- reasoning ------------------------------------------------------------
# A reasoning model spends the output budget thinking before it writes. On a
# schema-shaped reply that is not a slower answer, it is JSON cut mid-object:
# measured at 145 reasoning tokens against 18 of actual content.

@pytest.fixture
def openrouter(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "some/model:free")
    monkeypatch.setattr(llm.config, "LLM_REASONING", None)
    return llm.OpenAICompatibleClient()


def test_openrouter_is_told_not_to_think_by_default(openrouter):
    assert openrouter._reasoning(0) == {"enabled": False}


def test_a_thinking_budget_does_not_switch_reasoning_on(openrouter):
    """thinking_budget means something different here than on Gemini: the
    thinking comes out of the same max_tokens as the answer, and the budget is
    advisory. A coding video asking for 4096 was measured spending 6385 and
    returning an unparseable reply, so every window of a 3-hour video failed."""
    assert openrouter._reasoning(4096) == {"enabled": False}


def test_reasoning_can_be_turned_back_on_with_its_own_headroom(openrouter, monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_REASONING", "1")
    monkeypatch.setattr(llm.config, "LLM_MAX_OUTPUT_TOKENS", 8192)
    assert openrouter._reasoning(4096) == {"max_tokens": 4096}
    assert openrouter._output_cap(32768, 4096) == 8192 + 4096,         "the thinking must be added to the cap, not taken out of the answer"


def test_the_cap_is_not_inflated_when_reasoning_is_off(openrouter):
    assert openrouter._output_cap(32768, 4096) == config.LLM_MAX_OUTPUT_TOKENS


def test_other_providers_are_not_sent_an_openrouter_extension(monkeypatch):
    """A strict endpoint rejects a field it does not know, so the switch is
    not sent blind."""
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm.config, "LLM_REASONING", None)
    assert llm.OpenAICompatibleClient()._reasoning(0) is None


def test_the_switch_can_be_forced_on_for_any_provider(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "m")
    monkeypatch.setattr(llm.config, "LLM_REASONING", "0")
    assert llm.OpenAICompatibleClient()._reasoning(4096) == {"enabled": False}, \
        "forcing it off must win over a requested budget"


# --- a reply that was cut off --------------------------------------------
# How much a window needs varies with how dense it is. One part of the SQL
# video was lost to the cap while the next part finished in 1,614 tokens.

def test_a_truncated_reply_is_recognised():
    cut = {"choices": [{"message": {"content": '{"explanation":[{"heading":"Set'},
                        "finish_reason": "length"}]}
    assert llm.OpenAICompatibleClient._was_cut_off(cut) is True


def test_an_empty_reply_at_the_cap_is_recognised():
    assert llm.OpenAICompatibleClient._was_cut_off(
        {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}) is True


def test_a_complete_object_at_the_cap_is_not_retried():
    """It said 'length' but the JSON closed, so the answer is usable and a
    second call would spend a request for nothing."""
    assert llm.OpenAICompatibleClient._was_cut_off(
        {"choices": [{"message": {"content": '{"ok":true}'},
                      "finish_reason": "length"}]}) is False


def test_a_normal_reply_is_not_retried():
    assert llm.OpenAICompatibleClient._was_cut_off(
        {"choices": [{"message": {"content": '{"ok":true}'},
                      "finish_reason": "stop"}]}) is False


def test_a_cut_off_window_is_asked_again_with_room(client, monkeypatch):
    caps = []

    class Cut:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"explanation":[{"head'},
                                 "finish_reason": "length"}]}

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "second time lucky"}')

    def fake_post(url, headers=None, json=None, timeout=None):
        caps.append(json["max_tokens"])
        return Cut() if len(caps) == 1 else Ok()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(llm.config, "LLM_MAX_OUTPUT_TOKENS", 8192)

    result = client.generate_json("sys", "prompt", max_output_tokens=8192)
    assert result.data == {"overview": "second time lucky"}
    assert caps == [8192, 32768], "it should grow once, then stop growing"


def test_it_does_not_grow_for_ever(client, monkeypatch):
    """Two cut-off replies in a row is a prompt problem, not a cap problem."""
    calls = []

    class Cut:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": '{"explanation":[{"head'},
                                 "finish_reason": "length"}]}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        (calls.append(json["max_tokens"]), Cut())[1])

    with pytest.raises(llm.LLMError, match="output limit"):
        client.generate_json("sys", "prompt", max_output_tokens=8192)
    assert len(calls) == 2, "one grow, then report it"


# --- an upstream failure dressed as success -------------------------------
# A gateway in front of several providers answers 200 and reports the
# provider's trouble in the body. Both of these were fatal on first sight,
# which cost a five-hour video every window it had already paid for.

def test_no_choices_is_an_upstream_failure():
    assert llm.OpenAICompatibleClient._upstream_failed({"choices": []})


def test_finish_reason_error_is_an_upstream_failure():
    assert llm.OpenAICompatibleClient._upstream_failed(
        {"choices": [{"message": {"content": '{"partial":'}, "finish_reason": "error"}]})


def test_an_error_body_is_an_upstream_failure():
    assert "overloaded" in llm.OpenAICompatibleClient._upstream_failed(
        {"error": {"message": "upstream overloaded"}, "choices": []})


def test_a_good_reply_is_not_an_upstream_failure():
    assert llm.OpenAICompatibleClient._upstream_failed(
        {"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]}) == ""


def test_an_upstream_failure_is_retried_not_raised(client, monkeypatch):
    calls = []

    class Broken:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [], "error": {"message": "upstream overloaded"}}

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "worked on the retry"}')

    import requests
    monkeypatch.setattr(time := __import__("time"), "sleep", lambda s: None)
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        (calls.append(1), Broken() if len(calls) < 3 else Ok())[1])

    result = client.generate_json("sys", "prompt")
    assert result.data == {"overview": "worked on the retry"}
    assert len(calls) == 3, "it should have kept trying"


def test_an_upstream_failure_that_never_clears_is_reported(client, monkeypatch):
    class Broken:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": []}

    import requests
    monkeypatch.setattr(__import__("time"), "sleep", lambda s: None)
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None: Broken())

    with pytest.raises(llm.LLMError, match="no choices"):
        client.generate_json("sys", "prompt")


# --- a second model, for when the first stops answering -------------------
# Ten of thirteen windows of one video failed on a model that had been
# answering fine an hour earlier. Naming a fallback turns that from a failed
# run into a slower one.

@pytest.fixture
def two_models(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setattr(llm.config, "LLM_API_KEY", "k")
    monkeypatch.setattr(llm.config, "LLM_MODEL", "first/model:free, second/model:free")
    monkeypatch.setattr(llm.config, "LLM_REASONING", None)
    monkeypatch.setattr(__import__("time"), "sleep", lambda s: None)
    return llm.OpenAICompatibleClient()


def test_a_comma_separated_setting_is_a_preference_order(two_models):
    assert two_models._models == ["first/model:free", "second/model:free"]
    assert two_models.model == "first/model:free", "the first is used until it fails"


def test_the_fallback_is_used_when_the_first_stops_answering(two_models, monkeypatch):
    tried = []

    class Broken:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": []}

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "the second one answered"}')

    def fake_post(url, headers=None, json=None, timeout=None):
        tried.append(json["model"])
        return Ok() if json["model"] == "second/model:free" else Broken()

    import requests
    monkeypatch.setattr(requests, "post", fake_post)

    result = two_models.generate_json("sys", "prompt")
    assert result.data == {"overview": "the second one answered"}
    assert tried[0] == "first/model:free"
    assert "second/model:free" in tried


def test_the_first_model_is_not_abandoned_while_it_works(two_models, monkeypatch):
    tried = []

    class Ok:
        status_code = 200
        text = ""

        def json(self):
            return reply('{"overview": "fine"}')

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None:
                        (tried.append(json["model"]), Ok())[1])

    two_models.generate_json("sys", "prompt")
    assert tried == ["first/model:free"], "no need to spend a request on the spare"


def test_a_spent_quota_is_not_worked_around_by_switching_model(two_models, monkeypatch):
    """A per-day quota belongs to the account, not the model, so the second
    one would fail identically — and the caller wants to keep partial work."""
    class Spent:
        status_code = 429
        text = '{"error":{"details":[{"violations":[{"quotaId":"GenerateRequestsPerDay"}]}]}}'

        def json(self):
            return {"error": {"details": [
                {"violations": [{"quotaId": "GenerateRequestsPerDayPerProject"}]}]}}

    import requests
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None: Spent())

    with pytest.raises(llm.LLMQuotaExceeded):
        two_models.generate_json("sys", "prompt")


def test_one_model_still_reports_its_own_error_plainly(client, monkeypatch):
    class Broken:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": []}

    import requests
    monkeypatch.setattr(__import__("time"), "sleep", lambda s: None)
    monkeypatch.setattr(requests, "post",
                        lambda url, headers=None, json=None, timeout=None: Broken())

    with pytest.raises(llm.LLMError) as exc:
        client.generate_json("sys", "prompt")
    assert "every configured model" not in str(exc.value)
