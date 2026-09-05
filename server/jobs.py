"""Job lifecycle: create, run, and update rows in the jobs table.

The worker (in-process pool or an RQ process) calls :func:`run_job`. It drives
``core.pipeline.process`` and mirrors every progress tick into the job row, so
the API can report status by reading one row — no message bus needed for a
single instance.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from typing import Optional

from server import config
from server.db import Job, JobStatus, session

logger = logging.getLogger("server.jobs")

_PROFILE_BUDGET = {"fast": 150.0, "balanced": None, "accurate": None}
_PROFILE_QUALITY_FLOOR = {"fast": False, "balanced": False, "accurate": True}

# Progress writes are cheap (one row) but still rate-limited so a chatty stage
# doesn't hammer the database; terminal transitions always write through.
_MIN_WRITE_INTERVAL = 0.4


def create_job(
    *, source_ref: str, is_upload: bool, video_type: str, topic: str,
    profile: str, force: bool,
) -> Job:
    with session() as db:
        job = Job(
            source_ref=source_ref, is_upload=is_upload, video_type=video_type,
            topic=topic, profile=profile, force=force,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        logger.info("created job %s (%s, %s)", job.id, video_type, profile)
        return job


def get_job(job_id: str) -> Optional[Job]:
    with session() as db:
        return db.get(Job, job_id)


def list_jobs(limit: int = 50) -> list[Job]:
    with session() as db:
        return list(
            db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        )


def run_job(job_id: str) -> None:
    """Execute one job to completion. Safe to call from a worker thread/process."""
    from core import pipeline  # deferred: pulls in torch etc.

    with session() as db:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("run_job: %s not found", job_id)
            return
        job.status = JobStatus.RUNNING
        job.stage = "starting"
        job.started_at = dt.datetime.now(dt.timezone.utc)
        job.worker_class = config.WORKER_CLASS or _autodetect_worker_class()
        db.commit()
        params = (
            job.source_ref, job.is_upload, job.video_type, job.topic,
            job.profile, job.force,
        )

    source_ref, is_upload, video_type, topic, profile, force = params
    last_write = [0.0]

    def on_progress(stage: str, progress: float, message: str) -> None:
        now = time.monotonic()
        terminal = stage in ("done", "error") or progress >= 1.0
        if not terminal and now - last_write[0] < _MIN_WRITE_INTERVAL:
            return
        last_write[0] = now
        with session() as db:
            row = db.get(Job, job_id)
            if row is None:
                return
            row.stage = stage
            row.progress = min(max(progress, 0.0), 1.0)
            row.message = message
            state = pipeline.get_state(pipeline.video_id_for(source_ref))
            if state.get("timings"):
                row.timings = state["timings"]
            db.commit()

    if _PROFILE_BUDGET.get(profile) is not None:
        import config as core_config

        core_config.TARGET_PIPELINE_SEC = _PROFILE_BUDGET[profile]

    try:
        result = pipeline.process(
            source_ref, is_upload=is_upload, force=force,
            progress_cb=on_progress,
            quality_floor=_PROFILE_QUALITY_FLOOR.get(profile, False),
            video_type=video_type, topic=topic,
        )
    except Exception as exc:  # noqa: BLE001 - recorded on the row, re-raised for the worker log
        from core.models import describe_exception

        reason = describe_exception(exc)
        logger.exception("job %s failed", job_id)
        with session() as db:
            row = db.get(Job, job_id)
            if row is not None:
                row.status = JobStatus.FAILED
                row.stage = "error"
                row.error = reason
                row.finished_at = dt.datetime.now(dt.timezone.utc)
                db.commit()
        return

    with session() as db:
        row = db.get(Job, job_id)
        if row is not None:
            row.status = JobStatus.DONE
            row.stage = "done"
            row.progress = 1.0
            row.message = "Complete."
            row.video_id = result["video_id"]
            row.title = result["meta"].get("title")
            row.duration_sec = result["meta"].get("duration_sec")
            row.timings = result.get("timings", {})
            row.finished_at = dt.datetime.now(dt.timezone.utc)
            db.commit()
    logger.info("job %s done -> video %s", job_id, result["video_id"])


def _autodetect_worker_class() -> str:
    """A coarse hardware label for the scheduler's per-class cost model."""
    try:
        from core import perf

        p = perf.get_profile()
        return f"{'gpu' if p.has_cuda else 'cpu'}-{p.cpu_count}core"
    except Exception:
        return "unknown"


# --- in-process worker pool ---------------------------------------------------
#
# The zero-infrastructure queue: a bounded thread pool inside the API process.
# Good for a single box and for demonstrating the pipeline; swap in RQ for a
# real fleet (server/config.QUEUE_BACKEND).

_pool_lock = threading.Lock()
_pool: Optional["_InProcessPool"] = None


class _InProcessPool:
    def __init__(self, workers: int) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self._ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="job")

    def submit(self, job_id: str) -> None:
        self._ex.submit(self._guarded, job_id)

    @staticmethod
    def _guarded(job_id: str) -> None:
        try:
            run_job(job_id)
        except Exception:  # never let a worker thread die silently
            logger.exception("unhandled error running job %s", job_id)


def enqueue(job_id: str) -> None:
    """Hand a job to whichever queue backend is configured."""
    if config.QUEUE_BACKEND == "redis":
        _enqueue_redis(job_id)
        return
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = _InProcessPool(config.WORKER_CONCURRENCY)
    _pool.submit(job_id)


def _enqueue_redis(job_id: str) -> None:
    from redis import Redis  # type: ignore
    from rq import Queue  # type: ignore

    queue = Queue("video", connection=Redis.from_url(config.REDIS_URL))
    queue.enqueue("server.jobs.run_job", job_id, job_timeout=3600)
