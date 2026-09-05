"""Orchestrator: runs M1 -> M6 and tracks weighted overall progress across the
whole pipeline, on top of the fine-grained per-stage progress each module
already reports while it runs.

The ordering here is the other half of the performance story (the first half
being what each module was made to stop doing). Transcription and visual
analysis have no data dependency on each other — fusion is the first stage that
needs both — so they are run **concurrently** rather than back to back. Visual
analysis is the cheaper of the two after its keyframe cap, so in practice it
disappears entirely into transcription's wall time.

The two stages contend for the same cores, so torch's thread pool is narrowed
while they overlap and restored before summarization, which runs alone and
wants every core it can get.
"""

from __future__ import annotations

import logging
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, Optional

import config
from core import cache, fuse, ingest, perf, qa, summarize, transcribe, understand, vision, warm
from core.models import describe_exception

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float, str], None]

# Relative weight of each concurrent stage inside the merged "analyze" phase.
# Transcription dominates, so it drives most of the bar's movement.
_ANALYZE_WEIGHTS = {"transcribe": 0.75, "vision": 0.25}


def _understand_or_summarize(
    video_id: str, force: bool, plan, video_type: str, topic: str,
    progress_cb=None,
) -> dict[str, Any]:
    """Produce the explanation/summary: an LLM when one is configured, the
    local BART/flan-t5 map-reduce otherwise.

    Both write ``summary.json`` in a shape the UI and exports understand; the
    LLM path additionally writes the richer ``understanding.json``. Falling
    back on LLM failure keeps a transient API problem from failing the whole
    run (unless config.UNDERSTANDING_FALLBACK_TO_LOCAL is off).
    """
    if understand.is_available():
        try:
            result = understand.generate(
                video_id, video_type=video_type, topic=topic, force=force
            )
            summary = _understanding_to_summary(result)
            cache.write_json(video_id, "summary", summary)
            return summary
        except understand.llm.LLMError as exc:
            if not config.UNDERSTANDING_FALLBACK_TO_LOCAL:
                raise
            logger.warning(
                "LLM understanding failed (%s); falling back to local summarizer",
                understand.llm.LLMError.__name__, exc_info=True,
            )
    return summarize.summarize(video_id, force, plan)


def _understanding_to_summary(u: dict[str, Any]) -> dict[str, Any]:
    """Project understanding.json onto the legacy summary.json shape.

    The explanation sections become the section-by-section walkthrough, the
    recap becomes the overall summary, and chapters carry through. Everything
    richer (glossary, coding, meeting, scene notes) stays only in
    understanding.json, which the new UI reads directly.
    """
    return {
        "overall_summary": u.get("summary") or u.get("overview", ""),
        "chunk_summaries": [
            {"start": s["start_sec"], "end": s["start_sec"], "text": s["body"]}
            for s in u.get("explanation", [])
        ],
        "chapters": u.get("chapters", []),
        "model": u.get("model", "llm"),
        "understanding": True,
    }

def _set_torch_threads(n: int) -> None:
    """Resize torch's intra-op thread pool, if torch is loaded at all."""
    try:
        import torch

        torch.set_num_threads(max(1, n))
        logger.debug("torch threads set to %d", max(1, n))
    except Exception:
        logger.debug("could not set torch thread count (non-fatal)", exc_info=True)


class _AnalyzeProgress:
    """Collects progress from the two concurrent stages, for the main thread.

    Each stage keeps calling ``cache.set_state`` exactly as it does when run
    alone; this sink intercepts those calls and records the latest fraction
    and message per stage. Without it the two would overwrite each other's
    state.json and the bar would jump backwards.

    Crucially it **does not publish anything itself**. These calls arrive on
    worker threads, and the progress callback ends in Streamlit widget calls,
    which raise ``NoSessionContext`` off the script-runner thread. So the
    workers only ever record; :func:`process` polls :meth:`snapshot` from the
    main thread and reports from there.
    """

    def __init__(self, base_progress: float, span: float) -> None:
        self._base = base_progress
        self._span = span
        self._lock = threading.Lock()
        self._fractions: dict[str, float] = {"transcribe": 0.0, "vision": 0.0}
        self._messages: dict[str, str] = {}

    def __call__(self, video_id: str, stage: str, progress: float,
                 message: str, error: Optional[str]) -> None:
        with self._lock:
            if error is not None:
                # Persisting is safe from any thread; reporting is not.
                cache.write_state(video_id, "error", self._current(), message, error)
                return
            if stage in self._fractions:
                self._fractions[stage] = min(max(progress, 0.0), 1.0)
                self._messages[stage] = message

    def snapshot(self) -> tuple[float, str]:
        """Current merged (progress, message). Safe to call from any thread."""
        with self._lock:
            return self._current(), self._combined_message()

    def _current(self) -> float:
        weighted = sum(self._fractions[s] * w for s, w in _ANALYZE_WEIGHTS.items())
        return self._base + self._span * weighted

    def _combined_message(self) -> str:
        # Transcription is the long pole, so its message is the informative
        # one to show; vision's only surfaces before transcription starts.
        return (
            self._messages.get("transcribe")
            or self._messages.get("vision")
            or "Analyzing..."
        )

def process(
    source_ref: str,
    is_upload: bool = False,
    force: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
    quality_floor: bool = False,
    video_type: str = "lecture",
    topic: str = "",
) -> dict[str, Any]:
    """Run the full M1 -> M6 pipeline for one video.

    Args:
        source_ref: A YouTube (or yt-dlp-supported) URL, or a local file path
            when ``is_upload`` is True.
        is_upload: True if ``source_ref`` is a local file path.
        force: Re-run every stage even if cached artifacts already exist.
        progress_cb: Optional ``callback(stage, cumulative_progress, message)``,
            invoked at the start and end of every stage.
        quality_floor: Pin the accurate models and keep all optional analysis,
            trading the time budget away for accuracy.

    Returns:
        ``{"video_id", "meta", "summary", "timings", "elapsed_sec"}``
    """
    video_id = cache.get_video_id(source_ref)
    cumulative = 0.0
    timings: dict[str, float] = {}
    pipeline_started = time.perf_counter()

    def report(stage: str, progress: float, message: str) -> None:
        cache.write_state(video_id, stage, progress, message, extra={"timings": timings})
        if progress_cb is not None:
            progress_cb(stage, progress, message)

    def fail(stage: str, message: str, exc: BaseException) -> None:
        error_message = f"{message} failed: {describe_exception(exc)}"
        cache.write_state(video_id, "error", cumulative, error_message, error=str(exc))
        if progress_cb is not None:
            progress_cb("error", cumulative, error_message)
        logger.exception("pipeline stage '%s' failed for %s", stage, video_id)

    def run_stage(stage: str, message: str, fn, *args, **kwargs):
        nonlocal cumulative
        report(stage, cumulative, f"{message}...")
        started = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            fail(stage, message, exc)
            raise
        timings[stage] = round(time.perf_counter() - started, 2)
        cumulative += config.STAGE_WEIGHTS[stage]
        report(stage, cumulative, f"{message} complete.")
        return result

    # --- M1 ---------------------------------------------------------------
    meta = run_stage(
        "ingest", "Fetching and preparing video",
        ingest.ingest, source_ref, is_upload, force, quality_floor,
    )
    plan = perf.plan_from_dict(meta["plan"]) if meta.get("plan") else perf.plan_for(
        float(meta.get("duration_sec", 0.0)), quality_floor=quality_floor
    )
    logger.info("running %s with plan: %s", video_id, plan.describe())

    # --- M2 + M3, concurrently -------------------------------------------
    profile = perf.get_profile()
    analyze_base, analyze_span = cumulative, config.STAGE_WEIGHTS["analyze"]
    sink = _AnalyzeProgress(analyze_base, analyze_span)

    report("analyze", analyze_base, "Transcribing audio and analyzing visuals...")
    analyze_started = time.perf_counter()
    # Transcription saturates the cores through CTranslate2; leaving torch
    # wide open here would have the two stages fighting over them.
    _set_torch_threads(max(2, profile.cpu_count // 4))
    cache.set_progress_sink(sink)
    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="analyze") as pool:
            jobs = {
                pool.submit(transcribe.transcribe, video_id, force, plan):
                    ("transcribe", "Transcribing audio"),
                pool.submit(vision.analyze_visual, video_id, force, plan):
                    ("vision", "Analyzing visual content"),
            }
            # Progress is published from *this* thread, never from the workers.
            # progress_cb ends in Streamlit widget calls, which raise
            # NoSessionContext when invoked off the script-runner thread — so
            # the workers only record into the sink and the main thread polls
            # it while it waits.
            pending = set(jobs)
            while pending:
                _done, pending = wait(
                    pending, timeout=config.PROGRESS_POLL_SEC, return_when=FIRST_COMPLETED
                )
                progress, message = sink.snapshot()
                report("analyze", progress, message)

        # Both futures have settled by here, so no worker is left running
        # behind a raised exception. Report transcription first: it is the
        # stage a user is most likely to be able to act on.
        for future, (stage, label) in jobs.items():
            exc = future.exception()
            if exc is not None:
                fail(stage, label, exc)
                raise exc
    finally:
        cache.set_progress_sink(None)
        _set_torch_threads(profile.cpu_count)

    timings["analyze"] = round(time.perf_counter() - analyze_started, 2)
    cumulative = analyze_base + analyze_span
    report("analyze", cumulative, "Transcription and visual analysis complete.")

    # --- M4 -> M6 ---------------------------------------------------------
    run_stage("fuse", "Aligning transcript with visuals", fuse.fuse, video_id, force)
    summary = run_stage(
        "summarize", "Generating summary and chapters",
        summarize.summarize, video_id, force, plan,
    )
    run_stage("index", "Building Q&A index", qa.build_index, video_id, force)

    # Optional M7 highlight reel. Best-effort: a missing checkpoint or ffmpeg
    # hiccup must not fail an otherwise-complete run.
    highlight_path: Optional[str] = None
    if config.HIGHLIGHT_AUTO_BUILD and getattr(config, "HIGHLIGHT_MODEL_PATH", None):
        try:
            from core import highlight as _highlight

            report("highlight", cumulative, "Building highlight reel...")
            hp = _highlight.build_highlight(video_id, force=force)
            highlight_path = str(hp) if hp else None
        except Exception:
            logger.warning("highlight reel step failed for %s", video_id, exc_info=True)

    elapsed = round(time.perf_counter() - pipeline_started, 2)
    timings["total"] = elapsed
    cache.write_state(
        video_id, "done", 1.0,
        f"Completed in {elapsed / 60:.1f} min.",
        extra={"timings": timings, "plan": perf.plan_to_dict(plan)},
    )
    if progress_cb is not None:
        progress_cb("done", 1.0, f"Completed in {elapsed / 60:.1f} min.")
    logger.info(
        "pipeline finished %s in %.1f s (%.2fx realtime): %s",
        video_id, elapsed, elapsed / max(float(meta.get("duration_sec", 1.0)), 1.0), timings,
    )

    return {
        "video_id": video_id,
        "meta": meta,
        "summary": summary,
        "timings": timings,
        "elapsed_sec": elapsed,
        "highlight_path": highlight_path,
    }


# --- UI façade -----------------------------------------------------------------------
#
# app.py is restricted to importing core.pipeline and core.models only, so every
# read/action the Streamlit UI needs — beyond running the pipeline itself — is
# exposed here as a thin wrapper.


def answer_question(video_id: str, question: str) -> dict[str, Any]:
    """Answer a question about an already-processed video (see core.qa.answer).

    build_index() is idempotent (it no-ops when a cached index exists), so
    calling it here makes this self-healing for videos whose pipeline run was
    interrupted before the index stage completed, instead of surfacing a raw
    "no index" error to the UI.
    """
    qa.build_index(video_id)
    return qa.answer(video_id, question)


def preview_source(source_ref: str, is_upload: bool = False) -> dict[str, Any]:
    """Resolve title/duration and the plan for a source without processing it.

    Lets the UI show what it is about to do — and how long it expects to take —
    before the user commits to a run.
    """
    if is_upload:
        from pathlib import Path

        probe = ingest._probe(Path(source_ref))  # noqa: SLF001 - module-internal by design
        duration, title = probe["duration_sec"], Path(source_ref).name
    else:
        remote = ingest.probe_remote(source_ref)
        duration, title = remote["duration_sec"], remote["title"]

    plan = perf.plan_for(duration)
    return {
        "title": title,
        "duration_sec": duration,
        "video_id": cache.get_video_id(source_ref),
        "plan": perf.plan_to_dict(plan),
        "predicted_sec": plan.predicted_total_sec,
        "cached": cache.artifact_exists(cache.get_video_id(source_ref), "summary"),
    }


def start_warmup() -> None:
    """Begin loading models in the background (see core.warm)."""
    warm.start()


def warmup_status() -> dict[str, Any]:
    return warm.status()


def video_id_for(source_ref: str) -> str:
    """The cache key a given source will be stored under.

    Exposed so the UI can watch a run's state before ``process()`` returns,
    without reaching past this façade into core.cache.
    """
    return cache.get_video_id(source_ref)


def get_state(video_id: str) -> dict[str, Any]:
    return cache.get_state(video_id)


def get_meta(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "meta")


def get_summary(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "summary")


def get_understanding(video_id: str) -> Optional[dict[str, Any]]:
    """The rich LLM output (explanation sections, glossary, coding, scene notes)."""
    return cache.read_json(video_id, "understanding")


def get_scenes(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "scenes")


def get_transcript(video_id: str) -> Optional[dict[str, Any]]:
    return cache.read_json(video_id, "transcript")


def get_highlight(video_id: str, *, build: bool = False) -> Optional[str]:
    """Path to the highlight reel, optionally building it on demand."""
    from core import highlight as _highlight

    p = _highlight.get_highlight_path(video_id)
    if p is None and build:
        p = _highlight.build_highlight(video_id)
    return str(p) if p else None


def get_hardware_profile() -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(perf.get_profile())


def clear_cache(video_id: str) -> None:
    cache.clear_cache(video_id)


def list_cached_videos() -> list[str]:
    return cache.list_cached_videos()
