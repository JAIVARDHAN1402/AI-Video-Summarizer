"""FastAPI application: the multi-user tier over the same ``core`` pipeline.

Endpoints
    POST   /api/jobs                 submit a URL for processing
    GET    /api/jobs                 recent jobs
    GET    /api/jobs/{id}            one job's status + progress
    GET    /api/jobs/{id}/events     Server-Sent Events stream of progress
    GET    /api/videos/{video_id}    the finished result (understanding, scenes, ...)
    GET    /api/videos/{video_id}/keyframes/{name}   a keyframe image
    POST   /api/videos/{video_id}/ask                grounded Q&A
    GET    /api/health

Auth is not wired here — a real deployment puts a JWT / Clerk middleware in
front. Everything the browser needs to render a result is JSON from
``/api/videos/{video_id}``; long work happens on the worker, and the SSE
stream carries progress so the client never blocks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from server import auth, config, jobs, schemas, scheduler, storage
from server.db import JobStatus, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("server")

app = FastAPI(
    title=config.API_TITLE, version=config.API_VERSION,
    dependencies=[Depends(auth.require_auth)],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # Warm the models in the background so the first job doesn't eat ~70s of
    # cold loading (same rationale as the Streamlit app).
    try:
        from core import warm

        warm.start()
    except Exception:
        logger.warning("model warm-up could not start", exc_info=True)
    storage.get_storage()  # log the backend, fail fast on a bad S3 config
    logger.info("auth: %s", "enabled" if config.AUTH_ENABLED else "disabled (open)")
    if config.QUEUE_BACKEND == "redis":
        logger.info("API ready — queue backend: redis (%s); workers run separately", config.REDIS_URL)
    else:
        logger.info("API ready — %s", jobs.fleet_status())


# --- Jobs -------------------------------------------------------------------


@app.post("/api/uploads", response_model=schemas.UploadResponse, status_code=201)
async def upload_file(file: UploadFile = File(...)) -> schemas.UploadResponse:
    """Store a video file and return an ``upload_id`` to pass to POST /api/jobs."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.UPLOAD_ALLOWED_EXTS:
        raise HTTPException(415, f"unsupported type {ext!r}; allowed: {sorted(config.UPLOAD_ALLOWED_EXTS)}")
    try:
        info = await asyncio.to_thread(
            storage.get_storage().save_upload, file.file, file.filename or f"upload{ext}"
        )
    except storage.StorageError as e:
        raise HTTPException(413, str(e)) from e
    return schemas.UploadResponse(upload_id=info["upload_id"], filename=info["filename"], bytes=info["bytes"])


@app.post("/api/jobs", response_model=schemas.JobRef, status_code=202)
def create_job(req: schemas.CreateJobRequest) -> schemas.JobRef:
    if req.upload_id:
        try:
            source_ref = str(storage.get_storage().upload_path(req.upload_id.strip()))
        except storage.StorageError as e:
            raise HTTPException(404, str(e)) from e
        is_upload = True
    else:
        source_ref, is_upload = req.url.strip(), False

    job = jobs.create_job(
        source_ref=source_ref, is_upload=is_upload, video_type=req.video_type,
        topic=req.topic.strip(), profile=req.profile, force=req.force,
        sla_sec=req.sla_sec,
    )
    jobs.enqueue(job.id, job.routed_worker_class)
    return schemas.JobRef(id=job.id, status=job.status.value)


@app.get("/api/jobs", response_model=list[schemas.JobView])
def recent_jobs(limit: int = 50) -> list[schemas.JobView]:
    return [schemas.JobView(**j.to_dict()) for j in jobs.list_jobs(min(limit, 200))]


@app.get("/api/jobs/{job_id}", response_model=schemas.JobView)
def job_status(job_id: str) -> schemas.JobView:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return schemas.JobView(**job.to_dict())


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> EventSourceResponse:
    """Stream job progress as SSE until the job reaches a terminal state.

    Polls the job row rather than subscribing to a bus — fine for one
    instance, and it means the worker only has to write, never publish.
    """
    if jobs.get_job(job_id) is None:
        raise HTTPException(404, "job not found")

    async def stream() -> AsyncIterator[dict]:
        last = None
        while True:
            job = await asyncio.to_thread(jobs.get_job, job_id)
            if job is None:
                break
            payload = job.to_dict()
            snapshot = (payload["status"], payload["stage"], round(payload["progress"], 3), payload["message"])
            if snapshot != last:
                last = snapshot
                yield {"event": "progress", "data": json.dumps(payload)}
            if job.status in (JobStatus.DONE, JobStatus.FAILED):
                yield {"event": "end", "data": json.dumps(payload)}
                break
            await asyncio.sleep(1.0)

    return EventSourceResponse(stream())


# --- Results ---------------------------------------------------------------


def _keyframe_urls(video_id: str, data: dict | None) -> dict | None:
    """Rewrite absolute keyframe_path values to API URLs.

    The frontend should not see server filesystem paths, and it should not have
    to know the naming scheme (scene_NNN vs sample_NNN) — it just needs a URL
    it can put in an <img src>.
    """
    if not data:
        return data
    base = f"/api/videos/{video_id}/keyframes/"
    for scene in data.get("scenes", []) or []:
        p = scene.get("keyframe_path")
        if p:
            scene["keyframe_url"] = base + Path(p).name
    for note in data.get("scene_notes", []) or []:
        p = note.get("keyframe_path")
        if p:
            note["keyframe_url"] = base + Path(p).name
    return data


def _ensure_local_artifacts(video_id: str) -> None:
    """With S3 storage, pull this video's cache tree down if it isn't local
    (e.g. the job ran on another worker). No-op for local storage."""
    import config as core_config

    cache_dir = core_config.CACHE_ROOT / video_id
    try:
        storage.get_storage().fetch_artifacts(video_id, cache_dir)
    except Exception:
        logger.warning("could not fetch artifacts for %s from storage", video_id, exc_info=True)


@app.get("/api/videos/{video_id}", response_model=schemas.VideoResult)
def video_result(video_id: str) -> schemas.VideoResult:
    from core import pipeline

    _ensure_local_artifacts(video_id)
    meta = pipeline.get_meta(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    return schemas.VideoResult(
        video_id=video_id,
        meta=meta,
        understanding=_keyframe_urls(video_id, pipeline.get_understanding(video_id)),
        summary=pipeline.get_summary(video_id),
        scenes=_keyframe_urls(video_id, pipeline.get_scenes(video_id)),
        transcript=pipeline.get_transcript(video_id),
        state=pipeline.get_state(video_id),
    )


@app.get("/api/videos/{video_id}/keyframes/{name}")
def keyframe(video_id: str, name: str) -> FileResponse:
    import config as core_config

    # Guard against path traversal — only a bare filename under this video's dir.
    safe = Path(name).name
    path = core_config.CACHE_ROOT / video_id / "keyframes" / safe
    if not path.exists():
        _ensure_local_artifacts(video_id)
    if not path.exists() or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(404, "keyframe not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/videos/{video_id}/ask", response_model=schemas.AskResponse)
def ask(video_id: str, req: schemas.AskRequest) -> schemas.AskResponse:
    from core import pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    result = pipeline.answer_question(video_id, req.question.strip())
    return schemas.AskResponse(
        answer=result["answer"],
        confidence=result.get("confidence", "low"),
        sources=result.get("sources", []),
    )


@app.get("/api/scheduler/classes")
def scheduler_classes() -> list[dict]:
    """Registered worker classes and their current cost-per-video-minute."""
    return scheduler.classes_view()


@app.get("/api/scheduler/sla")
def scheduler_sla() -> dict:
    """SLA adherence: how often routing held the budget, and how a fixed
    'always accurate models' assignment would have compared."""
    return scheduler.sla_stats()


@app.post("/api/scheduler/preview")
def scheduler_preview(req: schemas.CreateJobRequest) -> dict:
    """Show the routing decision for a request without queuing it."""
    from core import ingest

    try:
        duration = float(ingest.probe_remote(req.url).get("duration_sec") or 900.0)
    except Exception:
        duration = 900.0
    sla = req.sla_sec or jobs._PROFILE_BUDGET.get(req.profile) or config.DEFAULT_BUDGET_SEC
    return scheduler.route(duration, req.video_type, sla).to_dict()


@app.get("/api/health")
def health() -> dict:
    from core import perf, understand

    return {
        "status": "ok",
        "queue": config.QUEUE_BACKEND,
        "fleet": jobs.fleet_status(),
        "llm_understanding": understand.is_available(),
        "hardware": perf.get_profile().__dict__ if hasattr(perf.get_profile(), "__dict__") else {},
    }
