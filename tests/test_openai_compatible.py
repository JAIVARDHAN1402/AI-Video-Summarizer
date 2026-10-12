"""The OpenAI-compatible provider path.

Groq, OpenRouter, Cerebras, Together, Mistral and a local Ollama all speak
this shape, and several have a free tier with no card. The parsing is what
matters: unlike Gemini these endpoints do not enforce a JSON Schema
server-side, so the reply arrives however the model felt like sending it.
"""

from __future__ import annotations

import pytest

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
