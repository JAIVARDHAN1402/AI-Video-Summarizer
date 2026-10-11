"""Unit tests for core.cache. No ML dependencies required."""

from __future__ import annotations

import pytest

import config
from core import cache


@pytest.fixture(autouse=True)
def isolated_cache_root(tmp_path, monkeypatch):
    """Point config.CACHE_ROOT at a throwaway directory for every test."""
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path / "cache")
    yield


def test_get_video_id_is_deterministic_and_short():
    vid_a = cache.get_video_id("https://youtube.com/watch?v=abc123")
    vid_b = cache.get_video_id("https://youtube.com/watch?v=abc123")
    vid_c = cache.get_video_id("https://youtube.com/watch?v=different")

    assert vid_a == vid_b
    assert vid_a != vid_c
    assert len(vid_a) == 16


def test_get_cache_dir_creates_directory():
    video_id = "testvid0000000a"
    cache_dir = cache.get_cache_dir(video_id)

    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert cache_dir.name == video_id


def test_write_and_read_json_round_trip():
    video_id = "testvid0000000b"
    data = {"video_id": video_id, "duration_sec": 123.4, "nested": {"a": [1, 2, 3]}}

    cache.write_json(video_id, "meta", data)
    result = cache.read_json(video_id, "meta")

    assert result == data


def test_read_json_returns_none_when_missing():
    assert cache.read_json("nonexistent0000", "meta") is None


def test_read_json_returns_none_on_corrupt_file():
    video_id = "testvid0000000c"
    cache_dir = cache.get_cache_dir(video_id)
    (cache_dir / "meta.json").write_text("{not valid json", encoding="utf-8")

    assert cache.read_json(video_id, "meta") is None


def test_artifact_exists_reflects_presence():
    video_id = "testvid0000000d"
    assert cache.artifact_exists(video_id, "transcript") is False

    cache.write_json(video_id, "transcript", {"language": "en"})
    assert cache.artifact_exists(video_id, "transcript") is True


def test_state_defaults_when_unset():
    state = cache.get_state("testvid0000000e")
    assert state["stage"] == "not_started"
    assert state["progress"] == 0.0
    assert state["error"] is None


def test_set_state_then_get_state_round_trip():
    video_id = "testvid0000000f"
    cache.set_state(video_id, stage="transcribe", progress=0.42,
                     message="Transcribing audio...")

    state = cache.get_state(video_id)
    assert state["stage"] == "transcribe"
    assert state["progress"] == 0.42
    assert state["message"] == "Transcribing audio..."
    assert state["error"] is None
    assert "updated_at" in state


def test_set_state_with_error():
    video_id = "testvid0000000g"
    cache.set_state(video_id, stage="error", progress=0.5,
                     message="failed", error="boom")

    state = cache.get_state(video_id)
    assert state["stage"] == "error"
    assert state["error"] == "boom"


def test_clear_cache_removes_directory():
    video_id = "testvid0000000h"
    cache_dir = cache.get_cache_dir(video_id)
    cache.write_json(video_id, "meta", {"video_id": video_id})
    assert cache_dir.exists()

    cache.clear_cache(video_id)
    assert not cache_dir.exists()


def test_clear_cache_on_missing_video_is_noop():
    cache.clear_cache("never-existed-0")  # should not raise


def test_list_cached_videos_orders_most_recent_first(monkeypatch):
    import time

    cache.get_cache_dir("older0000000000")
    time.sleep(0.01)
    cache.get_cache_dir("newer0000000000")

    videos = cache.list_cached_videos()
    assert videos[0] == "newer0000000000"
    assert "older0000000000" in videos


def test_list_cached_videos_empty_when_no_cache_root():
    assert cache.list_cached_videos() == []


def test_a_later_state_write_keeps_the_finished_run_s_record():
    """Progress reported after a run must not erase what the run cost.

    state.json holds two different kinds of thing: where processing is right
    now, which every tick replaces, and the timings and plan of the finished
    run, written once at the end. Generating an audio overview reports
    progress long after the pipeline is done, and that used to overwrite the
    file wholesale — leaving the run report with nothing to show for videos
    that had been processed perfectly well.
    """
    vid = "state-merge"
    cache.write_state(
        vid, "done", 1.0, "finished",
        extra={"timings": {"ingest": 2.0, "total": 58.6}, "plan": {"asr_model": "small"}},
    )

    # A terminal write always lands, so this is not swallowed by the tick
    # rate-limiter the way an intermediate progress tick would be.
    cache.write_state(vid, "audio", 1.0, "Audio overview ready")

    state = cache.get_state(vid)
    assert state["stage"] == "audio", "the new stage should still be recorded"
    assert state["timings"] == {"ingest": 2.0, "total": 58.6}
    assert state["plan"] == {"asr_model": "small"}


def test_a_rerun_can_still_replace_the_timings():
    """Carrying the old record forward must not pin it there for ever."""
    vid = "state-replace"
    cache.write_state(vid, "done", 1.0, "first", extra={"timings": {"total": 10.0}})
    cache.write_state(vid, "done", 1.0, "second", extra={"timings": {"total": 20.0}})
    assert cache.get_state(vid)["timings"] == {"total": 20.0}


def test_reading_an_unknown_video_creates_nothing():
    """A lookup must not leave a directory behind.

    _artifact_path used to create the cache directory for whoever asked, so a
    404 on an unknown id — or a read of an artefact that happened not to exist
    — left an empty directory that then appeared in list_cached_videos() as a
    video with nothing in it.
    """
    assert cache.read_json("never-processed", "meta") is None
    assert not (config.CACHE_ROOT / "never-processed").exists()
    assert "never-processed" not in cache.list_cached_videos()


def test_writing_still_creates_what_it_needs():
    cache.write_json("fresh-video", "meta", {"title": "x"})
    assert cache.read_json("fresh-video", "meta") == {"title": "x"}
