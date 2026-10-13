"""What the reader is told when the model cannot answer.

The local path is extractive: it returns the passages that matched, lightly
tidied. Asked "how do I find the second highest salary", it returned an
unrelated stretch of a SQL lecture about average unit price — labelled high
confidence. That reads as the system answering badly rather than as it being
unable to answer, which is the thing a reader most needs to tell apart.
"""

from __future__ import annotations

import pytest

from core import llm, qa


@pytest.fixture
def retrieval(monkeypatch):
    """A video with passages, so the only variable is the model."""
    chunks = [
        {"start": 10.0, "end": 40.0, "text": "a passage about unit prices", "score": 0.9},
    ]
    monkeypatch.setattr(qa, "_retrieve", lambda *a, **k: (chunks, [0.9]), raising=False)
    return chunks


def test_a_spent_quota_is_reported_not_disguised(monkeypatch, retrieval):
    class Spent:
        available = True
        model = "some-model"

        def generate_json(self, *a, **k):
            raise llm.LLMQuotaExceeded("spent until midnight")

    monkeypatch.setattr(llm, "get_client", lambda: Spent())
    out = qa.answer("vid", "How do I find the second highest salary?")

    assert out["model_unavailable"] is True
    assert out["confidence"] == "low", "an excerpt is not a confident answer"
    assert "out of quota" in out["answer"]
    assert out["sources"], "the passages are still worth citing"
    assert "unit prices" not in out["answer"], "the excerpt must not pose as the answer"


def test_an_ordinary_model_failure_still_falls_back(monkeypatch, retrieval):
    """A bad reply is not the same as no model: the extractive path is a
    reasonable consolation there, and the caller keeps its old behaviour."""
    class Broken:
        available = True
        model = "some-model"

        def generate_json(self, *a, **k):
            raise llm.LLMError("returned no choices")

    monkeypatch.setattr(llm, "get_client", lambda: Broken())
    out = qa.answer("vid", "What is covered here?")
    assert out.get("model_unavailable") is not True
