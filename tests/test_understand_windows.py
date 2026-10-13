"""Windowing and quota handling for long videos.

These cover the failure that made long videos useless: an 11-hour recording
needed 28 model calls, exhausted a free-tier quota part-way through, and then
spent close to three hours retrying a quota that was never coming back before
falling through to a local summariser that produced text nobody could read.
"""

from __future__ import annotations

import config
from core import llm, understand


class FakeResponse:
    """Just enough of requests.Response for the quota check."""

    def __init__(self, payload: dict, text: str = ""):
        self._payload = payload
        self.text = text or str(payload)

    def json(self):
        return self._payload


def _quota_body(quota_id: str) -> dict:
    return {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota",
            "details": [{"violations": [{"quotaId": quota_id}]}],
        }
    }


def test_a_daily_quota_is_recognised_as_spent():
    """A per-day quota does not clear today, so waiting for it is pure loss."""
    response = FakeResponse(_quota_body("GenerateRequestsPerDayPerProjectPerModel"))
    assert llm._quota_is_exhausted(response) is True


def test_a_per_minute_limit_is_not_treated_as_spent():
    """A per-minute limit clears in under a minute and is worth waiting for."""
    response = FakeResponse(_quota_body("GenerateRequestsPerMinutePerProjectPerModel"))
    assert llm._quota_is_exhausted(response) is False


def test_an_unstructured_429_is_not_assumed_fatal():
    """Without a violation to read, prefer retrying over giving up."""
    response = FakeResponse({"error": {"code": 429, "message": "Too Many Requests"}},
                            text="Too Many Requests")
    assert llm._quota_is_exhausted(response) is False


def test_quota_exceeded_is_distinguishable_from_other_failures():
    """Callers branch on it, so it must not be caught only as a generic error."""
    assert issubclass(llm.LLMQuotaExceeded, llm.LLMError)
    assert not issubclass(llm.LLMError, llm.LLMQuotaExceeded)


def test_windows_are_equal_sized_with_no_stub_at_the_end():
    bounds = understand._windows(70 * 60, 25)
    assert len(bounds) == 3
    lengths = [round((e - s) / 60, 1) for s, e in bounds]
    assert lengths == [23.3, 23.3, 23.3], lengths
    assert bounds[0][0] == 0 and bounds[-1][1] == 70 * 60


def test_window_count_is_capped_so_one_video_cannot_drain_the_quota():
    """Every window is a request. An 11-hour video at 25 minutes is 27 of
    them plus a reduce, which is enough to exhaust a free tier on one video."""
    eleven_hours = 665 * 60
    uncapped = understand._windows(eleven_hours, 25)
    capped = understand._windows(eleven_hours, 25, max_windows=16)

    assert len(uncapped) == 27
    assert len(capped) == 16
    # Widening the windows is the trade: fewer calls, less density.
    assert (capped[0][1] - capped[0][0]) > (uncapped[0][1] - uncapped[0][0])
    # The whole video is still covered, with no gap and no overrun.
    assert capped[0][0] == 0
    assert capped[-1][1] == eleven_hours
    for (_, end), (start, _) in zip(capped, capped[1:]):
        assert abs(end - start) < 1e-6


def test_a_short_video_is_unaffected_by_the_cap():
    bounds = understand._windows(40 * 60, 25, max_windows=16)
    assert len(bounds) == 2


def test_the_configured_cap_is_applied():
    """The default must actually bound an 11-hour video, not just exist."""
    bounds = understand._windows(
        665 * 60, config.UNDERSTANDING_WINDOW_MIN, config.UNDERSTANDING_MAX_WINDOWS
    )
    assert len(bounds) <= config.UNDERSTANDING_MAX_WINDOWS
    assert len(bounds) < 27, "the cap should bite on a video this long"


# --- the overall summary is one more call, and it can fail ----------------

def test_a_failed_summary_leaves_the_sections_readable():
    """Three windows of a five-hour video were thrown away because the
    summary-of-summaries failed after them. A document without a written
    overview is worth having; a failed run is not."""
    sections = [
        {"heading": "Joins", "body": "...", "start_sec": 0},
        {"heading": "Window functions", "body": "...", "start_sec": 600},
    ]
    chapters = [{"title": "Joins", "start": 0}, {"title": "Window functions", "start": 600}]
    out = understand._fallback_overview({"title": "SQL in 5 hours"}, sections, chapters)

    assert out["overview_is_fallback"] is True
    assert "Joins" in out["overview"] and "Window functions" in out["overview"]
    assert "could not be written" in out["overview"], "it must not pass as the real thing"
    assert out["key_points"] == ["Joins", "Window functions"]


def test_the_fallback_works_without_chapters():
    out = understand._fallback_overview(
        {}, [{"heading": "Only section", "body": "x", "start_sec": 0}], [])
    assert "Only section" in out["overview"]


# --- the other gateway's way of saying the day is over --------------------
# Missing this cost twelve requests against a quota that was already spent,
# each one preceded by a wait for it to clear.

class OpenRouterResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.text = str(payload)
        self.status_code = 429

    def json(self):
        return self._payload


def _daily_limit(remaining: str = "0", reset: str = "1791936000000") -> dict:
    return {"error": {
        "message": "Rate limit exceeded: free-models-per-day. Add 10 credits "
                   "to unlock 1000 free model requests per day",
        "code": 429,
        "metadata": {
            "headers": {"X-RateLimit-Limit": "50",
                        "X-RateLimit-Remaining": remaining,
                        "X-RateLimit-Reset": reset},
            "limit_source": "openrouter_free_tier_daily",
        },
    }}


def test_a_daily_free_tier_limit_is_recognised_as_spent():
    assert llm._quota_is_exhausted(OpenRouterResponse(_daily_limit())) is True


def test_a_burst_limit_with_requests_left_is_still_worth_waiting_for():
    body = {"error": {"message": "Rate limit exceeded: requests per minute",
                      "code": 429,
                      "metadata": {"headers": {"X-RateLimit-Remaining": "12"},
                                   "limit_source": "openrouter_per_minute"}}}
    assert llm._quota_is_exhausted(OpenRouterResponse(body)) is False


def test_the_reset_time_is_read_back_for_the_reader():
    """"Try again later" is not actionable; the gateway states the hour."""
    when = llm.quota_resets_at(OpenRouterResponse(_daily_limit()))
    assert when and ":" in when, when


def test_a_missing_reset_time_is_not_an_error():
    assert llm.quota_resets_at(
        OpenRouterResponse({"error": {"message": "slow down", "metadata": {}}})) is None


def test_the_document_admits_when_it_is_incomplete():
    """A missing part and a stand-in overview are not the same document as a
    complete one, and the reader is owed that. Both were being dropped by the
    whitelist that assembles the final document."""
    import inspect
    source = inspect.getsource(understand.generate)
    assert '"windows_failed"' in source
    assert '"overview_is_fallback"' in source
