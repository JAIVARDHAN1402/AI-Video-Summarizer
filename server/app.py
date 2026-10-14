"""FastAPI application: the multi-user tier over the same ``core`` pipeline.

Endpoints
    POST   /api/jobs                 submit a URL for processing
    GET    /api/jobs                 recent jobs
    GET    /api/jobs/{id}            one job's status + progress
    GET    /api/jobs/{id}/events     Server-Sent Events stream of progress
    GET    /api/videos               every processed video in the cache
    DELETE /api/videos/{video_id}    delete one processed video
    DELETE /api/videos?confirm=DELETE  empty the library
    GET    /api/chats?scope=...      saved conversations for a video or notebook
    GET    /api/notebooks            sets of videos asked as one
    POST   /api/notebooks/{id}/ask   one question across a notebook
    GET    /api/videos/{video_id}    the finished result (understanding, scenes, ...)
    GET    /api/videos/{video_id}/keyframes/{name}   a keyframe image
    POST   /api/videos/{video_id}/ask                grounded Q&A
    GET    /api/videos/{video_id}/audio-overview     two-host audio conversation
    GET    /api/videos/{video_id}/quiz               the cached quiz, if any
    POST   /api/videos/{video_id}/quiz               generate one
    GET    /api/preflight                            what already exists for a link
    GET    /api/auth/methods                         which ways in this server offers
    POST   /api/auth/otp/request | /verify           sign in with a one-time code
    GET    /api/auth/google/url  | POST /callback    sign in with Google
    GET    /api/auth/me                              the account and its allowance
    GET    /api/billing/plan | POST /order,/confirm  upgrading to premium
    GET    /api/videos/{video_id}/problems           problems the video solves
    POST   /api/videos/{video_id}/problems           extract them
    POST   /api/videos/{video_id}/problems/{n}/solutions   other languages
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
from fastapi.responses import FileResponse, Response
from sse_starlette.sse import EventSourceResponse

from server import (auth, billing, config, db, identity, jobs, quota,
                    schemas, scheduler, storage)
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
    # Anything still marked running belongs to a process that no longer
    # exists, and a progress bar that will never move reads as a stuck
    # pipeline rather than as a restart.
    try:
        jobs.release_orphans()
    except Exception:
        logger.warning("could not release orphaned jobs", exc_info=True)
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
def create_job(req: schemas.CreateJobRequest,
               principal: dict = Depends(auth.require_auth)) -> schemas.JobRef:
    if req.upload_id:
        try:
            source_ref = str(storage.get_storage().upload_path(req.upload_id.strip()))
        except storage.StorageError as e:
            raise HTTPException(404, str(e)) from e
        is_upload = True
    else:
        source_ref, is_upload = req.url.strip(), False

    from core import pipeline

    # What this costs is decided before anything is downloaded. The duration
    # comes from the source's own metadata, so a video too long for the
    # reader's plan is refused in a second rather than after an hour of work.
    session = db.session()
    try:
        user = identity.user_from_principal(session, principal)
        cost = 0
        try:
            preview = pipeline.preview_source(source_ref, is_upload)
            cost = quota.check(session, user, float(preview.get("duration_sec") or 0.0))
        except quota.QuotaError as exc:
            raise HTTPException(402, exc.to_dict()) from exc
        except HTTPException:
            raise
        except Exception:
            # A source we cannot measure yet is not a reason to refuse; the
            # run itself will fail with something the reader can act on.
            logger.warning("could not measure %s before queueing", source_ref, exc_info=True)

        job = jobs.create_job(
            source_ref=source_ref, is_upload=is_upload, video_type=req.video_type,
            topic=req.topic.strip(), profile=req.profile, force=req.force,
            sla_sec=req.sla_sec,
        )
        quota.charge(session, user, cost, job_id=job.id, kind="process")
    finally:
        session.close()

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


@app.post("/api/jobs/{job_id}/cancel", response_model=schemas.JobView)
def cancel_job(job_id: str) -> schemas.JobView:
    """Stop a queued or running job.

    A worker mid-stage notices at its next progress tick, so a long stage
    finishes before the run unwinds; the row goes to "cancelled" immediately
    either way, which is what clears a job wedged by a dead worker.
    """
    job = jobs.cancel_job(job_id)
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
            if job.status.is_terminal:
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


@app.get("/api/videos")
def list_videos() -> list[dict]:
    """Every processed video in the cache, newest first.

    The library used to be assembled from the jobs table alone, which hid any
    video processed outside that flow — through the Streamlit app, before a
    database reset, or after the cache moved. The cache is the ground truth
    for what has actually been processed, so the listing is taken from there
    and the job row only supplies extras when one happens to exist.

    Only small files are read per video (meta and state); the transcript and
    understanding documents are left on disk, and their presence is reported
    as a flag rather than their contents.
    """
    import config as core_config
    from core import pipeline

    out: list[dict] = []
    for vid in pipeline.list_cached_videos():
        root = core_config.CACHE_ROOT / vid
        if not (root / "summary.json").exists():
            continue  # ingested or part-processed, nothing to show yet
        meta = pipeline.get_meta(vid) or {}
        state = pipeline.get_state(vid) or {}
        timings = state.get("timings") or {}
        out.append(
            {
                "video_id": vid,
                "title": meta.get("title") or vid,
                "duration_sec": meta.get("duration_sec"),
                "source_ref": meta.get("source_ref"),
                "video_type": meta.get("video_type"),
                "processed_sec": timings.get("total"),
                "has_understanding": (root / "understanding.json").exists(),
                "has_highlight": (root / "highlight.mp4").exists(),
                "mtime": root.stat().st_mtime,
            }
        )
    return out


def _forget_video(video_id: str) -> None:
    """Remove a video and everything that points at it.

    Deleting only the cache directory leaves two kinds of ghost behind: chat
    threads scoped to a video that no longer exists, and notebooks listing it
    as a missing source. Both are cleaned here so "delete" means deleted.
    """
    from core import pipeline
    from server.db import ChatThread, Notebook, session

    pipeline.clear_cache(video_id)

    with session() as db:
        db.query(ChatThread).filter(ChatThread.scope == f"video:{video_id}").delete()
        for nb in db.query(Notebook).all():
            ids = list(nb.video_ids or [])
            if video_id in ids:
                nb.video_ids = [v for v in ids if v != video_id]
        db.commit()


@app.delete("/api/videos/{video_id}", status_code=204, response_class=Response)
def delete_video(video_id: str) -> Response:
    """Delete one processed video, irreversibly."""
    from core import pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    _forget_video(video_id)
    return Response(status_code=204)


@app.delete("/api/videos")
def clear_library(confirm: str = "") -> dict:
    """Delete every processed video.

    ``confirm`` must be the literal string "DELETE". A destructive endpoint
    reachable by a bare DELETE to a collection URL is one stray fetch away
    from erasing the library, and this costs the caller one query parameter.
    """
    from core import pipeline

    if confirm != "DELETE":
        raise HTTPException(400, 'pass ?confirm=DELETE to empty the library')

    removed = []
    for vid in pipeline.list_cached_videos():
        try:
            _forget_video(vid)
            removed.append(vid)
        except Exception:
            logger.warning("could not remove %s", vid, exc_info=True)
    logger.info("library cleared: %d videos removed", len(removed))
    return {"removed": len(removed), "video_ids": removed}


@app.get("/api/videos/{video_id}", response_model=schemas.VideoResult)
def video_result(video_id: str) -> schemas.VideoResult:
    from core import pipeline

    _ensure_local_artifacts(video_id)
    meta = pipeline.get_meta(video_id)
    if meta is None:
        raise HTTPException(404, "video not found")
    scenes = _keyframe_urls(video_id, pipeline.get_scenes(video_id))
    highlight = None
    hl = (scenes or {}).get("highlight") if scenes else None
    if hl and Path(hl.get("path", "")).exists():
        highlight = {
            "url": f"/api/videos/{video_id}/highlight",
            "seconds": hl.get("seconds"),
            "scene_ids": hl.get("scene_ids", []),
        }
    return schemas.VideoResult(
        video_id=video_id,
        meta=meta,
        understanding=_keyframe_urls(video_id, pipeline.get_understanding(video_id)),
        summary=pipeline.get_summary(video_id),
        scenes=scenes,
        transcript=pipeline.get_transcript(video_id),
        state=pipeline.get_state(video_id),
        highlight=highlight,
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


@app.get("/api/videos/{video_id}/highlight")
def highlight_reel(video_id: str) -> FileResponse:
    """The M7 highlight reel — built on demand if a checkpoint is configured."""
    from core import pipeline

    _ensure_local_artifacts(video_id)
    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    path = pipeline.get_highlight(video_id, build=True)
    if not path or not Path(path).exists():
        raise HTTPException(
            404, "no highlight reel — the importance model isn't trained/enabled for this run"
        )
    return FileResponse(path, media_type="video/mp4", filename=f"{video_id}-highlight.mp4")


@app.post("/api/videos/{video_id}/ask", response_model=schemas.AskResponse)
def ask(video_id: str, req: schemas.AskRequest) -> schemas.AskResponse:
    from core import pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    result = pipeline.answer_question(
        video_id, req.question.strip(),
        [{"q": t.q, "a": t.a} for t in req.history],
    )
    return schemas.AskResponse(
        answer=result["answer"],
        confidence=result.get("confidence", "low"),
        sources=result.get("sources", []),
        beyond_video=bool(result.get("beyond_video")),
        off_topic=bool(result.get("off_topic")),
    )


@app.get("/api/videos/{video_id}/quiz")
def get_quiz(video_id: str) -> dict:
    """The cached quiz, or 204 if none has been generated for this video yet.

    Deliberately not a 404: "no quiz yet" is the normal state, and the client
    distinguishes it from "this video does not exist".
    """
    from core import pipeline, quiz

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    cached = quiz.get(video_id)
    return cached or {"questions": [], "available": quiz.is_available()}


@app.post("/api/videos/{video_id}/quiz")
def make_quiz(video_id: str, count: int = 10, force: bool = False) -> dict:
    """Generate a quiz from the video's understanding document.

    Runs inline rather than on the worker queue: it is one LLM call over a
    document that is already summarised, so it finishes in seconds, and the
    reader is sitting in front of it waiting.
    """
    from core import llm, pipeline, quiz

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    try:
        return quiz.generate(video_id, count=count, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except llm.LLMUnavailable as exc:
        raise HTTPException(503, "no language model is configured on the server") from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"quiz generation failed: {exc}") from exc


@app.get("/api/preflight")
def preflight(url: str = "", upload_id: str = "") -> dict:
    """What already exists for this source, before a job is created.

    Submitting a link that has been here before silently resumed from the
    cache, and the run that followed reported seventeen seconds for a
    three-hour video — true of that run, and worthless as a measurement. The
    client asks this first so the choice between resuming and starting over
    is the reader's, and so the answer can say which it was.
    """
    from core import cache, pipeline

    source_ref = (url or upload_id or "").strip()
    if not source_ref:
        raise HTTPException(400, "give a url or an upload_id")

    video_id = cache.get_video_id(source_ref)
    stages = cache.completed_stages(video_id)
    meta = pipeline.get_meta(video_id) or {}
    state = cache.read_json(video_id, "state") or {}
    previous = (state.get("timings") or {}).get("total")

    return {
        "video_id": video_id,
        "source_ref": source_ref,
        "known": any(stages.values()),
        "stages": stages,
        "stages_done": sum(1 for v in stages.values() if v),
        "stages_total": len(stages),
        "complete": all(stages.values()),
        "title": meta.get("title"),
        "duration_sec": meta.get("duration_sec"),
        "last_stage": state.get("stage"),
        "last_error": state.get("error"),
        "previous_total_sec": previous,
        "previously_reused": state.get("reused_stages") or [],
    }


# --- accounts ----------------------------------------------------------------


def _session_user(principal: dict):
    """The signed-in account for a request, or None. Caller closes the session."""
    session = db.session()
    return session, identity.user_from_principal(session, principal)


@app.get("/api/auth/methods")
def auth_methods() -> dict:
    """Which ways in this deployment actually offers.

    The client asks rather than assuming, so a server without SMS credit
    simply does not show a phone tab instead of showing one that fails.
    """
    sms_ready = (
        (config.SMS_PROVIDER == "twilio" and bool(
            config.SMS_API_KEY and config.SMS_TWILIO_SID and config.SMS_TWILIO_FROM
        ))
        or (config.SMS_PROVIDER == "msg91" and bool(
            config.SMS_API_KEY and config.SMS_SENDER
        ))
    )
    return {
        "auth_required": config.AUTH_ENABLED,
        # A code can be delivered without a JWT secret, but it cannot produce
        # a session the API will accept. Do not advertise a dead sign-in path
        # to a deployment configured only with static API keys.
        "google": bool(config.JWT_SECRET) and identity.google_configured(),
        "email": bool(config.JWT_SECRET) and bool(config.SMTP_HOST or config.DEV_OTP_ECHO),
        "phone": bool(config.JWT_SECRET) and (sms_ready or config.DEV_OTP_ECHO),
        "dev_echo": config.DEV_OTP_ECHO,
        "payments": billing.configured(),
    }


@app.post("/api/auth/otp/request")
def request_otp(body: schemas.OtpRequest) -> dict:
    session = db.session()
    try:
        return identity.request_code(session, body.channel, body.destination)
    except identity.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        session.close()


@app.post("/api/auth/otp/verify")
def verify_otp(body: schemas.OtpVerify) -> dict:
    session = db.session()
    try:
        user = identity.verify_code(session, body.channel, body.destination, body.code)
        return {"token": identity.issue_token(user), "user": user.to_dict(),
                "quota": quota.summary(session, user)}
    except identity.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        session.close()


@app.get("/api/auth/google/url")
def google_url(state: str = "") -> dict:
    try:
        return {"url": identity.google_auth_url(state)}
    except identity.IdentityError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/auth/google/callback")
def google_callback(body: schemas.GoogleCallback) -> dict:
    session = db.session()
    try:
        user = identity.google_sign_in(session, body.code)
        return {"token": identity.issue_token(user), "user": user.to_dict(),
                "quota": quota.summary(session, user)}
    except identity.IdentityError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        session.close()


@app.get("/api/auth/me")
def whoami(principal: dict = Depends(auth.require_auth)) -> dict:
    session, user = _session_user(principal)
    try:
        return {
            "signed_in": user is not None,
            "user": user.to_dict() if user else None,
            "quota": quota.summary(session, user),
        }
    finally:
        session.close()


@app.get("/api/billing/plan")
def billing_plan() -> dict:
    return billing.plan()


@app.post("/api/billing/order")
def billing_order(principal: dict = Depends(auth.require_auth)) -> dict:
    session, user = _session_user(principal)
    try:
        if user is None:
            raise HTTPException(401, "sign in before upgrading")
        return billing.create_order(session, user)
    except billing.BillingError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        session.close()


@app.post("/api/billing/confirm")
def billing_confirm(body: schemas.PaymentConfirm,
                    principal: dict = Depends(auth.require_auth)) -> dict:
    session, user = _session_user(principal)
    try:
        if user is None:
            raise HTTPException(401, "sign in before upgrading")
        user = billing.confirm(session, user, body.order_id, body.payment_id,
                               body.signature)
        return {"user": user.to_dict(), "quota": quota.summary(session, user)}
    except billing.BillingError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        session.close()


# --- problem list: what a coding video actually works through ----------------


@app.get("/api/videos/{video_id}/problems")
def get_problems(video_id: str) -> dict:
    """The cached problem list, or an empty one if it has not been built.

    As with the quiz, "not extracted yet" is the normal state rather than an
    error, and the client offers the button instead of showing a failure.
    """
    from core import pipeline, problems

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    cached = problems.get(video_id)
    if cached:
        return cached
    return {
        "problems": [],
        "available": problems.is_available(),
        "families": problems.FAMILIES,
    }


@app.post("/api/videos/{video_id}/problems")
def make_problems(video_id: str, force: bool = False) -> dict:
    """Read the whole video and write down every problem it solves.

    Unlike the quiz this is several calls — one per part of the video — so a
    three-hour practice session takes a few minutes rather than seconds. It
    still runs inline: the reader asked for it and is waiting, and the result
    is cached, so it is paid for once.
    """
    from core import llm, pipeline, problems

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    try:
        return problems.generate(video_id, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except llm.LLMUnavailable as exc:
        raise HTTPException(503, "no language model is configured on the server") from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"could not extract the problems: {exc}") from exc


@app.post("/api/videos/{video_id}/problems/{index}/solutions")
def translate_problem(
    video_id: str,
    index: int,
    language: str = "",
    force: bool = False,
) -> dict:
    """Fill in one problem's solution in the other languages of its family.

    With no ``language`` every language the video did not use is filled in at
    once, which is one request rather than one per dialect — the difference
    between a free tier lasting a day and lasting an hour.
    """
    from core import llm, pipeline, problems

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    wanted = [language] if language else None
    try:
        return problems.translate(video_id, index, languages=wanted, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except IndexError as exc:
        raise HTTPException(404, str(exc)) from exc
    except llm.LLMUnavailable as exc:
        raise HTTPException(503, "no language model is configured on the server") from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"translation failed: {exc}") from exc


# --- chat threads: conversations that outlive the page -----------------------


@app.get("/api/chats")
def list_chats(scope: str) -> list[dict]:
    """Threads for one scope, newest first, without their turns.

    The list is a sidebar; loading every message of every past conversation to
    render a list of titles would be the expensive way to draw nothing.
    """
    from server.db import ChatThread, session

    with session() as db:
        rows = (
            db.query(ChatThread)
            .filter(ChatThread.scope == scope)
            .order_by(ChatThread.updated_at.desc())
            .all()
        )
        return [t.to_dict(with_turns=False) for t in rows]


@app.get("/api/chats/{thread_id}")
def get_chat(thread_id: str) -> dict:
    from server.db import ChatThread, session

    with session() as db:
        t = db.get(ChatThread, thread_id)
        if t is None:
            raise HTTPException(404, "chat not found")
        return t.to_dict()


@app.post("/api/chats", status_code=201)
def create_chat(req: schemas.ChatThreadRequest) -> dict:
    from server.db import ChatThread, session

    if not req.scope:
        raise HTTPException(400, "a chat needs a scope")
    with session() as db:
        t = ChatThread(
            scope=req.scope,
            title=(req.title or "New chat").strip()[:200],
            turns=req.turns or [],
        )
        db.add(t)
        db.commit()
        db.refresh(t)
        return t.to_dict()


@app.put("/api/chats/{thread_id}")
def save_chat(thread_id: str, req: schemas.ChatThreadRequest) -> dict:
    """Replace a thread's turns. Called after each exchange, so a reload or a
    tab switch never loses the conversation."""
    import datetime as _dt

    from server.db import ChatThread, session

    with session() as db:
        t = db.get(ChatThread, thread_id)
        if t is None:
            raise HTTPException(404, "chat not found")
        if req.turns is not None:
            t.turns = req.turns
        if req.title is not None:
            t.title = req.title.strip()[:200] or "New chat"
        t.updated_at = _dt.datetime.now(_dt.timezone.utc)
        db.commit()
        db.refresh(t)
        return t.to_dict()


@app.delete("/api/chats/{thread_id}", status_code=204, response_class=Response)
def delete_chat(thread_id: str) -> Response:
    from server.db import ChatThread, session

    with session() as db:
        t = db.get(ChatThread, thread_id)
        if t is None:
            raise HTTPException(404, "chat not found")
        db.delete(t)
        db.commit()
    return Response(status_code=204)


# --- notebooks: several videos asked as one body of material -----------------


def _notebook_view(nb) -> dict:
    """A notebook plus enough about each member for the sidebar to render.

    Members whose cache has been cleared are reported rather than hidden, so
    a notebook never silently shrinks.
    """
    from core import pipeline

    videos = []
    for vid in nb.video_ids or []:
        meta = pipeline.get_meta(vid)
        videos.append(
            {
                "video_id": vid,
                "title": (meta or {}).get("title") or vid,
                "duration_sec": (meta or {}).get("duration_sec"),
                "source_ref": (meta or {}).get("source_ref"),
                "missing": meta is None,
            }
        )
    return {**nb.to_dict(), "videos": videos}


@app.get("/api/notebooks")
def list_notebooks() -> list[dict]:
    from server.db import Notebook, session

    with session() as db:
        rows = db.query(Notebook).order_by(Notebook.updated_at.desc()).all()
        return [_notebook_view(nb) for nb in rows]


@app.post("/api/notebooks", status_code=201)
def create_notebook(req: schemas.NotebookRequest) -> dict:
    from server.db import Notebook, session

    with session() as db:
        nb = Notebook(
            title=(req.title or "Untitled notebook").strip()[:200],
            video_ids=list(dict.fromkeys(req.video_ids or [])),
        )
        db.add(nb)
        db.commit()
        db.refresh(nb)
        return _notebook_view(nb)


@app.get("/api/notebooks/{notebook_id}")
def get_notebook(notebook_id: str) -> dict:
    from server.db import Notebook, session

    with session() as db:
        nb = db.get(Notebook, notebook_id)
        if nb is None:
            raise HTTPException(404, "notebook not found")
        return _notebook_view(nb)


@app.put("/api/notebooks/{notebook_id}")
def update_notebook(notebook_id: str, req: schemas.NotebookRequest) -> dict:
    import datetime as _dt

    from server.db import Notebook, session

    with session() as db:
        nb = db.get(Notebook, notebook_id)
        if nb is None:
            raise HTTPException(404, "notebook not found")
        if req.title is not None:
            nb.title = req.title.strip()[:200] or "Untitled notebook"
        if req.video_ids is not None:
            # dict.fromkeys keeps the caller's ordering while dropping repeats.
            nb.video_ids = list(dict.fromkeys(req.video_ids))
        nb.updated_at = _dt.datetime.now(_dt.timezone.utc)
        db.commit()
        db.refresh(nb)
        return _notebook_view(nb)


@app.delete("/api/notebooks/{notebook_id}", status_code=204, response_class=Response)
def delete_notebook(notebook_id: str) -> Response:
    """Removes the notebook only. The videos it named stay in the cache."""
    from server.db import Notebook, session

    with session() as db:
        nb = db.get(Notebook, notebook_id)
        if nb is None:
            raise HTTPException(404, "notebook not found")
        db.delete(nb)
        db.commit()
    return Response(status_code=204)


@app.post("/api/notebooks/{notebook_id}/ask", response_model=schemas.AskResponse)
def ask_notebook(notebook_id: str, req: schemas.AskRequest) -> schemas.AskResponse:
    """One question against every video in the notebook."""
    from core import collection
    from server.db import Notebook, session

    with session() as db:
        nb = db.get(Notebook, notebook_id)
        if nb is None:
            raise HTTPException(404, "notebook not found")
        video_ids = list(nb.video_ids or [])

    result = collection.answer_across(
        video_ids, req.question.strip(), [{"q": t.q, "a": t.a} for t in req.history]
    )
    return schemas.AskResponse(
        answer=result["answer"],
        confidence=result.get("confidence", "low"),
        sources=result.get("sources", []),
        beyond_video=bool(result.get("beyond_video")),
        off_topic=bool(result.get("off_topic")),
    )


@app.get("/api/videos/{video_id}/audio-overview")
def get_audio_overview(video_id: str) -> dict:
    """The cached overview's script and timings, or an empty shell.

    As with the quiz, "not made yet" is the normal state and is reported as
    data rather than as a 404.
    """
    from core import audio_overview, pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    doc = audio_overview.get(video_id)
    if doc:
        return {**doc, "url": f"/api/videos/{video_id}/audio-overview/audio"}
    return {"turns": [], "available": audio_overview.is_available()}


@app.post("/api/videos/{video_id}/audio-overview")
def make_audio_overview(video_id: str, force: bool = False) -> dict:
    """Write, speak and assemble a two-host conversation about the video.

    Runs inline. It takes the better part of a minute — one model call for the
    script, then a few seconds of concurrent speech synthesis per turn — which
    is long for a request but far simpler than a second queue, and the reader
    is waiting on it with nothing else to do.
    """
    from core import audio_overview, llm, pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    try:
        doc = audio_overview.generate(video_id, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except llm.LLMUnavailable as exc:
        raise HTTPException(503, "no language model is configured on the server") from exc
    except (llm.LLMError, RuntimeError) as exc:
        raise HTTPException(502, f"audio overview failed: {exc}") from exc
    return {**doc, "url": f"/api/videos/{video_id}/audio-overview/audio"}


@app.get("/api/videos/{video_id}/audio-overview/audio")
def audio_overview_file(video_id: str) -> FileResponse:
    from core import audio_overview

    path = audio_overview.audio_path(video_id)
    if not path.exists():
        raise HTTPException(404, "no audio overview for this video")
    return FileResponse(path, media_type="audio/mpeg", filename="audio-overview.mp3")


@app.get("/api/videos/{video_id}/notes")
def list_notes(video_id: str) -> list[dict]:
    from core import notes, pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    return notes.list_notes(video_id)


@app.post("/api/videos/{video_id}/notes", status_code=201)
def add_note(video_id: str, req: schemas.NoteRequest) -> dict:
    from core import notes, pipeline

    if pipeline.get_meta(video_id) is None:
        raise HTTPException(404, "video not found")
    try:
        return notes.add(
            video_id, req.text, start_sec=req.start_sec, source=req.source
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.put("/api/videos/{video_id}/notes/{note_id}")
def edit_note(video_id: str, note_id: str, req: schemas.NoteRequest) -> dict:
    from core import notes

    try:
        return notes.update(video_id, note_id, req.text)
    except KeyError as exc:
        raise HTTPException(404, "note not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# response_class is explicit because the default JSONResponse always writes a
# body, and FastAPI asserts that a 204 has none.
@app.delete("/api/videos/{video_id}/notes/{note_id}", status_code=204,
            response_class=Response)
def remove_note(video_id: str, note_id: str) -> Response:
    from core import notes

    try:
        notes.delete(video_id, note_id)
    except KeyError as exc:
        raise HTTPException(404, "note not found") from exc
    return Response(status_code=204)


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
