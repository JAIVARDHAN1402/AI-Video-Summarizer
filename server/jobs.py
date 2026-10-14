"""Job lifecycle: create, run, and update rows in the jobs table.

The worker (in-process pool or an RQ process) calls :func:`run_job`. It drives
``core.pipeline.process`` and mirrors every progress tick into the job row, so
the API can report status by reading one row — no message bus needed for a
single instance.
"""

from __future__ import annotations

import datetime as dt
import logging
import queue as _queue
import threading
import time
from pathlib import Path
from typing import Optional

from server import config
from server.db import Job, JobStatus, session
from server import scheduler, storage

logger = logging.getLogger("server.jobs")

def _profile_budgets() -> dict[str, Optional[float]]:
    """Wall-clock budget per profile. "accurate" gets a large but real budget
    rather than none, so the planner still refuses configurations that can't
    finish — see config.ACCURATE_PROFILE_BUDGET_SEC."""
    import config as core_config

    return {
        "fast": 150.0,
        "balanced": None,
        "accurate": core_config.ACCURATE_PROFILE_BUDGET_SEC,
    }


_PROFILE_BUDGET = _profile_budgets()
# quality_floor pins every ladder to its top rung and disables sampling; it is
# unbounded by design and now used only as the scheduler's static baseline.
_PROFILE_QUALITY_FLOOR = {"fast": False, "balanced": False, "accurate": False}

# Progress writes are cheap (one row) but still rate-limited so a chatty stage
# doesn't hammer the database; terminal transitions always write through.
_MIN_WRITE_INTERVAL = 0.4

# One heavy pipeline run at a time per process: the local fleet shares a single
# set of warmed models (Whisper, YOLO, CLIP), and running two ``pipeline.process``
# calls through them concurrently is not safe. Class routing still happens
# independently — only the compute is serialised — and the emulated slow-down
# sleep is taken outside this lock so per-class wall-clocks still diverge.
# Harmless for RQ workers: each is its own process, so the lock is a no-op there.
_EXEC_LOCK = threading.Lock()


def create_job(
    *, source_ref: str, is_upload: bool, video_type: str, topic: str,
    profile: str, force: bool, sla_sec: Optional[float] = None,
) -> Job:
    with session() as db:
        job = Job(
            source_ref=source_ref, is_upload=is_upload, video_type=video_type,
            topic=topic, profile=profile, force=force,
        )
        db.add(job)
        db.commit()
        db.refresh(job)

    # Route it: pick the worker class for this job against its SLA. Best-effort
    # — a routing failure must not stop the job being queued.
    try:
        preview = _preview_duration(source_ref, is_upload)
        sla = sla_sec or _PROFILE_BUDGET.get(profile) or config.DEFAULT_BUDGET_SEC
        decision = scheduler.route(preview, video_type, sla)
        with session() as db:
            row = db.get(Job, job.id)
            row.routed_worker_class = decision.worker_class
            row.predicted_sec = round(decision.predicted_sec, 1)
            row.sla_target_sec = sla
            db.commit()
            db.refresh(row)
            job = row
        logger.info(
            "job %s routed -> %s (predicted %.0fs / SLA %.0fs)",
            job.id, decision.worker_class, decision.predicted_sec, sla,
        )
    except Exception:
        logger.warning("routing failed for job %s; queuing unrouted", job.id, exc_info=True)

    return job


def _preview_duration(source_ref: str, is_upload: bool = False) -> float:
    """Cheap duration lookup for routing, before the job runs.

    Uses a cached meta.json if the video was seen before, else ffprobe for an
    uploaded file or a yt-dlp metadata probe for a URL. Falls back to a neutral
    15 minutes so routing still has a number to work with.
    """
    from core import cache, ingest

    try:
        cached = cache.read_json(cache.get_video_id(source_ref), "meta")
        if cached and cached.get("duration_sec"):
            return float(cached["duration_sec"])
        if is_upload:
            return float(ingest._probe(Path(source_ref)).get("duration_sec") or 900.0)
        return float(ingest.probe_remote(source_ref).get("duration_sec") or 900.0)
    except Exception:
        return 900.0


class JobCancelled(Exception):
    """Raised out of the progress callback to unwind a cancelled run."""


def get_job(job_id: str) -> Optional[Job]:
    with session() as db:
        return db.get(Job, job_id)


def cancel_job(job_id: str) -> Optional[Job]:
    """Stop a queued or running job.

    Flips the row to CANCELLED, which is what the UI reads and what a running
    worker checks on its next progress tick. A job wedged by a killed worker
    has no one left to notice, and flipping the row is exactly what clears it
    from the queue view. Terminal jobs are returned unchanged.
    """
    with session() as db:
        row = db.get(Job, job_id)
        if row is None:
            return None
        if row.status.is_terminal:
            return row
        was = row.status
        row.status = JobStatus.CANCELLED
        row.stage = "cancelled"
        row.message = "Stopped."
        row.finished_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(row)
        logger.info("job %s cancelled (was %s)", job_id, was.value)
        return row


def list_jobs(limit: int = 50) -> list[Job]:
    with session() as db:
        return list(
            db.query(Job).order_by(Job.created_at.desc()).limit(limit).all()
        )


def run_job(
    job_id: str,
    *,
    worker_class: Optional[str] = None,
    worker_speed: Optional[float] = None,
) -> None:
    """Execute one job to completion. Safe to call from a worker thread/process.

    ``worker_class`` / ``worker_speed`` let the in-process fleet run each thread
    as a distinct class; an RQ worker process leaves them ``None`` and the
    values come from its environment (``VS_WORKER_CLASS`` / ``VS_WORKER_SPEED``).
    """
    from core import pipeline  # deferred: pulls in torch etc.

    wc = worker_class or config.WORKER_CLASS or _autodetect_worker_class()
    speed = worker_speed if worker_speed is not None else config.WORKER_SPEED

    with session() as db:
        job = db.get(Job, job_id)
        if job is None:
            logger.warning("run_job: %s not found", job_id)
            return
        if job.status == JobStatus.CANCELLED:
            logger.info("run_job: %s was cancelled before it started", job_id)
            return
        job.status = JobStatus.RUNNING
        job.stage = "starting"
        job.started_at = dt.datetime.now(dt.timezone.utc)
        job.worker_class = wc
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
            # The cancel endpoint only flips the row; this is where a worker
            # notices. Stages are coarse, so a cancel lands at the next stage
            # boundary rather than instantly.
            if row.status == JobStatus.CANCELLED:
                raise JobCancelled(job_id)
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

    proc_start = time.monotonic()
    try:
        with _EXEC_LOCK:
            result = pipeline.process(
                source_ref, is_upload=is_upload, force=force,
                progress_cb=on_progress,
                quality_floor=_PROFILE_QUALITY_FLOOR.get(profile, False),
                video_type=video_type, topic=topic,
            )
    except JobCancelled:
        logger.info("job %s cancelled mid-run", job_id)
        with session() as db:
            row = db.get(Job, job_id)
            if row is not None:
                row.stage = "cancelled"
                row.message = "Stopped."
                row.finished_at = dt.datetime.now(dt.timezone.utc)
                db.commit()
        return
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

    timings = _apply_emulated_speed(dict(result.get("timings", {})),
                                    time.monotonic() - proc_start, speed)

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
            row.timings = timings
            row.finished_at = dt.datetime.now(dt.timezone.utc)
            row.sla_met = timings.get("total", 0) <= (row.sla_target_sec or 1e9)
            db.commit()
            outcome_row = row
        else:
            outcome_row = None
    try:
        if outcome_row is not None:
            scheduler.record_outcome(outcome_row)
    except Exception:
        logger.warning("could not record scheduler outcome for %s", job_id, exc_info=True)

    if config.MIRROR_ARTIFACTS:
        try:
            import config as core_config

            storage.get_storage().mirror_artifacts(
                result["video_id"], core_config.CACHE_ROOT / result["video_id"]
            )
        except Exception:
            logger.warning("artifact mirror failed for %s", result["video_id"], exc_info=True)

    logger.info(
        "job %s done on %s (speed %.2f) -> video %s",
        job_id, wc, speed, result["video_id"],
    )


def _apply_emulated_speed(
    timings: dict[str, float], proc_wall_sec: float, speed: float,
) -> dict[str, float]:
    """Emulate a worker slower than the real hardware.

    ``speed`` 0.4 means "runs at 40% of this box's speed": sleep the extra
    wall-clock the slower class would have taken, and scale the recorded
    timings by 1/speed so the scheduler's per-class cost model learns the
    slower figure. ``speed >= 1`` is a no-op — we never fake being faster
    than the hardware actually is.
    """
    if not speed or speed >= 1.0:
        return timings
    factor = 1.0 / speed
    extra = min(proc_wall_sec * (factor - 1.0), config.MAX_EMULATED_DELAY_SEC)
    if extra > 0:
        time.sleep(extra)
    return {k: round(v * factor, 3) for k, v in timings.items()}


def _autodetect_worker_class() -> str:
    """A coarse hardware label for the scheduler's per-class cost model."""
    try:
        from core import perf

        p = perf.get_profile()
        return f"{'gpu' if p.has_cuda else 'cpu'}-{p.cpu_count}core"
    except Exception:
        return "unknown"


# --- local fleet (zero-infra multi-worker) ----------------------------------
#
# One in-memory queue per worker class, plus a shared "any" queue for unrouted
# jobs. Each fleet member is a thread that drains its own class queue first,
# then the shared one. This is what lets the SLA scheduler route across classes
# on a single box with no external services — set VS_LOCAL_FLEET to define the
# classes. For separate processes / real CPU throttling instead, use
# VS_QUEUE_BACKEND=redis (server/worker.py) or the docker-compose fleet.

_fleet_lock = threading.Lock()
_fleet: Optional["_LocalFleet"] = None


class _LocalFleet:
    def __init__(self, members: list[dict]) -> None:
        self._class_q: dict[str, _queue.Queue] = {}
        self._any_q: _queue.Queue = _queue.Queue()
        self._threads: list[threading.Thread] = []
        for m in members:
            cls = m["class"]
            self._class_q.setdefault(cls, _queue.Queue())
            for n in range(m.get("concurrency", 1)):
                t = threading.Thread(
                    target=self._run, args=(cls, float(m.get("speed", 1.0))),
                    name=f"worker[{cls}]#{n}", daemon=True,
                )
                t.start()
                self._threads.append(t)
        logger.info(
            "local fleet up: %s",
            ", ".join(
                f"{m['class']}x{m.get('concurrency', 1)}@speed{m.get('speed', 1.0)}"
                for m in members
            ),
        )

    def submit(self, job_id: str, routed_class: Optional[str]) -> None:
        q = self._class_q.get(routed_class) if routed_class else None
        (q or self._any_q).put(job_id)

    def _run(self, worker_class: str, speed: float) -> None:
        own = self._class_q[worker_class]
        while True:
            job_id = self._next(own)
            if job_id is None:
                continue
            try:
                run_job(job_id, worker_class=worker_class, worker_speed=speed)
            except Exception:  # never let a worker thread die silently
                logger.exception("unhandled error running job %s", job_id)

    def _next(self, own: _queue.Queue) -> Optional[str]:
        try:
            return own.get(timeout=0.5)
        except _queue.Empty:
            pass
        try:
            return self._any_q.get(timeout=0.5)
        except _queue.Empty:
            return None


def _fleet_members() -> list[dict]:
    if config.LOCAL_FLEET:
        return config.LOCAL_FLEET
    return [{
        "class": config.WORKER_CLASS or _autodetect_worker_class(),
        "speed": config.WORKER_SPEED,
        "concurrency": config.WORKER_CONCURRENCY,
    }]


def release_orphans() -> int:
    """Fail jobs left mid-flight by a server that is no longer running.

    With the in-process backend the work lives inside this process, so a
    restart means every job that was running died with it. The row stays
    RUNNING for ever otherwise, and the page shows a progress bar that will
    never move again — which reads as the pipeline being stuck rather than as
    the server having been restarted under it.

    Redis is left alone: there the workers are separate processes and a job
    that was running probably still is.
    """
    if config.QUEUE_BACKEND == "redis":
        return 0

    released = 0
    with session() as db:
        stale = (
            db.query(Job)
            .filter(Job.status.in_([JobStatus.RUNNING, JobStatus.QUEUED]))
            .all()
        )
        for job in stale:
            job.status = JobStatus.FAILED
            job.stage = "error"
            job.error = (
                "The server restarted while this job was running, so it was "
                "interrupted. Everything finished before that point is cached: "
                "submitting the same link again offers to resume from there."
            )
            job.message = "Interrupted by a server restart."
            released += 1
        if released:
            db.commit()
    if released:
        logger.warning("released %d job(s) left running by a previous process", released)
    return released


def fleet_status() -> dict:
    """What the local fleet is running as — surfaced on /api/health."""
    return {"backend": config.QUEUE_BACKEND, "members": _fleet_members()}


def enqueue(job_id: str, routed_class: Optional[str] = None) -> None:
    """Hand a job to whichever queue backend is configured, on the queue for
    its routed worker class."""
    if config.QUEUE_BACKEND == "redis":
        _enqueue_redis(job_id, routed_class)
        return
    global _fleet
    with _fleet_lock:
        if _fleet is None:
            _fleet = _LocalFleet(_fleet_members())
    _fleet.submit(job_id, routed_class)


def _enqueue_redis(job_id: str, routed_class: Optional[str]) -> None:
    from redis import Redis  # type: ignore
    from rq import Queue  # type: ignore

    name = (
        f"{config.QUEUE_NAMESPACE}.{routed_class}" if routed_class
        else f"{config.QUEUE_NAMESPACE}.any"
    )
    queue = Queue(name, connection=Redis.from_url(config.REDIS_URL))
    queue.enqueue("server.jobs.run_job", job_id, job_timeout=3600)
