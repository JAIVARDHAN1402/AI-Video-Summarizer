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
