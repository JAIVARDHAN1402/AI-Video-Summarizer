"""Orchestrator: runs M1 -> M6 in order and tracks weighted overall progress
across the whole pipeline, on top of the fine-grained per-stage progress
each module already writes to state.json while it runs.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import config
from core import cache, fuse, ingest, qa, summarize, transcribe, vision

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]


def process(
    source_ref: str,
    is_upload: bool = False,
    force: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> dict[str, Any]:
    """Run the full M1 -> M6 pipeline for one video.

    Args:
        source_ref: A YouTube (or yt-dlp-supported) URL, or a local file path
            when ``is_upload`` is True.
        is_upload: True if ``source_ref`` is a local file path.
        force: Re-run every stage even if cached artifacts already exist.
        progress_cb: Optional ``callback(stage, cumulative_progress, message)``,
            invoked at the start and end of every stage.

    Returns:
        ``{"video_id": str, "meta": dict, "summary": dict}``
    """
    video_id = cache.get_video_id(source_ref)
    cumulative = 0.0

    def report(stage: str, message: str) -> None:
        cache.set_state(video_id, stage, cumulative, message)
        if progress_cb is not None:
            progress_cb(stage, cumulative, message)

    def run_stage(stage: str, message: str, fn, *args, **kwargs):
        nonlocal cumulative
        report(stage, f"Starting: {message}")
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            error_message = f"{message} failed: {exc}"
            cache.set_state(video_id, "error", cumulative, error_message, error=str(exc))
            if progress_cb is not None:
                progress_cb("error", cumulative, error_message)
            logger.exception("pipeline stage '%s' failed for %s", stage, video_id)
            raise
        cumulative += config.STAGE_WEIGHTS[stage]
        report(stage, f"Done: {message}")
        return result

    meta = run_stage("ingest", "Ingesting video", ingest.ingest, source_ref, is_upload, force)
    run_stage("transcribe", "Transcribing audio", transcribe.transcribe, video_id, force)
    run_stage("vision", "Analyzing visual content", vision.analyze_visual, video_id, force)
    run_stage("fuse", "Fusing transcript and visuals", fuse.fuse, video_id, force)
    summary = run_stage(
        "summarize", "Generating summary and chapters", summarize.summarize, video_id, force
    )
    run_stage("index", "Building Q&A index", qa.build_index, video_id, force)

    cache.set_state(video_id, "done", 1.0, "Pipeline complete.")
    if progress_cb is not None:
        progress_cb("done", 1.0, "Pipeline complete.")

    return {"video_id": video_id, "meta": meta, "summary": summary}


# --- UI façade -----------------------------------------------------------------------
#
# app.py is restricted to importing core.pipeline and core.models only (see
# BUILD_PLAN.md section 2), so every read/action the Streamlit UI needs —
# beyond running the pipeline itself — is exposed here as a thin wrapper.


def answer_question(video_id: str, question: str) -> dict[str, Any]:
    """Answer a question about an already-processed video (see core.qa.answer).

    build_index() is idempotent (it no-ops if a cached index already exists),
    so calling it here makes this self-healing for videos whose pipeline run
    was interrupted before the index stage completed, instead of surfacing
    a raw "no index" error to the UI.
    """
    qa.build_index(video_id)
    return qa.answer(video_id, question)


def get_state(video_id: str) -> dict[str, Any]:
    return cache.get_state(video_id)


def get_meta(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "meta")


def get_summary(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "summary")


def get_scenes(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "scenes")


def clear_cache(video_id: str) -> None:
    cache.clear_cache(video_id)


def list_cached_videos() -> list[str]:
    return cache.list_cached_videos()
