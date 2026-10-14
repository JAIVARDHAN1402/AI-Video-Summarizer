"""Resuming must be a choice, and a resumed run must say so.

Submitting a link that had been processed here before silently reused the
cache, and the run that followed reported seventeen seconds for a three-hour
video. That number is true of that run and worthless as a measurement, and
the report went on to call it "624x faster than watching".
"""

from __future__ import annotations

import json

import pytest

import config
from core import cache


@pytest.fixture
def video(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path)
    vid = cache.get_video_id("https://youtu.be/example")
    (tmp_path / vid).mkdir(parents=True)
    return vid, tmp_path / vid


def write(dirpath, name, payload=None):
    path = dirpath / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"ok": True}), encoding="utf-8")


def test_a_link_never_seen_has_no_finished_stages(video):
    vid, _ = video
    assert not any(cache.completed_stages(vid).values())


def test_a_part_finished_run_reports_exactly_what_it_has(video):
    """This is the shape of the real failure: the transcription was paid for,
    the explanation was not."""
    vid, d = video
    write(d, "meta")
    write(d, "transcript")
    write(d, "scenes")
    write(d, "fused")

    stages = cache.completed_stages(vid)
    assert stages == {"ingest": True, "analyze": True, "fuse": True,
                      "summarize": False, "index": False}


def test_a_half_finished_stage_does_not_count_as_finished(video):
    """analyze produces two artefacts; one of them is not the stage."""
    vid, d = video
    write(d, "meta")
    write(d, "transcript")
    assert cache.completed_stages(vid)["analyze"] is False


def test_the_index_stage_is_read_from_its_own_directory(video):
    vid, d = video
    assert cache.completed_stages(vid)["index"] is False
    write(d / "qa_index", "chunks")
    assert cache.completed_stages(vid)["index"] is True


def test_every_pipeline_stage_can_be_asked_about():
    """A stage missing from the map would silently never be reported as
    reusable, which is the bug this exists to prevent."""
    assert set(cache.STAGE_ARTIFACTS) == set(config.STAGE_WEIGHTS)


# --- a job nobody is running --------------------------------------------
# The API was killed mid-transcription when the terminal tab it was tied to
# closed. No exception, no crash log — the row simply stayed RUNNING, and the
# page kept showing a progress bar that would never move again.

def test_a_restart_fails_the_jobs_it_interrupted(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from server import config as server_config
    from server import db as server_db
    from server import jobs

    engine = create_engine(f"sqlite:///{tmp_path / 'orphans.db'}", future=True)
    server_db.Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)
    monkeypatch.setattr(jobs, "session", maker)
    monkeypatch.setattr(server_config, "QUEUE_BACKEND", "inprocess")

    with maker() as s:
        s.add_all([
            server_db.Job(source_ref="a", status=server_db.JobStatus.RUNNING, stage="analyze"),
            server_db.Job(source_ref="b", status=server_db.JobStatus.QUEUED, stage="queued"),
            server_db.Job(source_ref="c", status=server_db.JobStatus.DONE, stage="done"),
        ])
        s.commit()

    assert jobs.release_orphans() == 2

    with maker() as s:
        by_ref = {j.source_ref: j for j in s.query(server_db.Job).all()}
        assert by_ref["a"].status is server_db.JobStatus.FAILED
        assert by_ref["b"].status is server_db.JobStatus.FAILED
        assert "restarted" in (by_ref["a"].error or "")
        assert "resume" in (by_ref["a"].error or ""), "it should say what to do next"
        assert by_ref["c"].status is server_db.JobStatus.DONE, "a finished job is untouched"


def test_redis_workers_are_left_alone(monkeypatch):
    """There the workers are separate processes, and a job that was running
    probably still is."""
    from server import config as server_config
    from server import jobs

    monkeypatch.setattr(server_config, "QUEUE_BACKEND", "redis")
    assert jobs.release_orphans() == 0
