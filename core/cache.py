"""Cache-directory management, job state, and JSON read/write helpers.

Every other module depends on this one. The cache directory is the sole
persistence layer for the project (no database) and is what makes the
pipeline resumable: each stage checks :func:`artifact_exists` for its own
output and returns the cached result immediately unless ``force=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import config

logger = logging.getLogger(__name__)

# Retries for the tmp->final rename in write_json. On Windows, a file
# indexer, antivirus, or (very commonly, if the project lives inside a
# OneDrive/Dropbox folder) the sync client can briefly hold a lock on the
# destination file right after it's written, making an otherwise-atomic
# os.replace() fail with WinError 5 (Access is denied). state.json in
# particular is rewritten on every pipeline progress tick, so it hits this
# far more often than other artifacts. A few short retries ride out the lock
# without changing write semantics.
_REPLACE_MAX_RETRIES = 8
_REPLACE_RETRY_DELAY_SEC = 0.15

# Serialises writes to the same artifact. transcribe and vision run
# concurrently and both report progress, so without this they race on
# state.json — and, before the temp filename was made unique below, raced on
# the same scratch file and failed outright with a PermissionError.
_WRITE_LOCK = threading.Lock()


def get_video_id(source_ref: str) -> str:
    """Derive a stable, filesystem-safe id from a URL or filename."""
    return hashlib.sha1(source_ref.encode("utf-8")).hexdigest()[:16]


# What each pipeline stage leaves behind. A stage is cache-first, so an
# artefact that was already on disk before a run started is one that run did
# not produce — which is the only honest way to read a resumed run's timings.
STAGE_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "ingest": ("meta",),
    "analyze": ("transcript", "scenes"),
    "fuse": ("fused",),
    "summarize": ("summary",),
    "index": ("qa_index/chunks",),
}


# Below this, a recorded time cannot describe a full run: even reading a
# video's own metadata takes longer than a hundredth of its length. Runs that
# resumed from the cache land here, and so do ones whose state was last
# written by a standalone step.
MIN_PLAUSIBLE_REALTIME_RATIO = 0.01


def run_was_partial(duration_sec: float, processed_sec: Optional[float],
                    reused_stages: Optional[list] = None) -> bool:
    """Whether a recorded time is too short to be a full run of this video.

    Resuming is the normal case after a failure and is not a problem; a
    resumed run quietly reporting "17 seconds" for a three-hour video is,
    because the number then gets read — and repeated — as the pipeline's
    speed. Runs recorded before reused_stages existed are judged by whether
    the time is physically possible at all.
    """
    if reused_stages:
        return True
    if not processed_sec or not duration_sec:
        return True
    return (processed_sec / duration_sec) < MIN_PLAUSIBLE_REALTIME_RATIO


def completed_stages(video_id: str) -> dict[str, bool]:
    """Which stages this video already has results for, in pipeline order."""
    done: dict[str, bool] = {}
    for stage, names in STAGE_ARTIFACTS.items():
        done[stage] = all(
            (config.CACHE_ROOT / video_id / f"{n}.json").exists() for n in names
        )
    return done


def get_cache_dir(video_id: str) -> Path:
    """Return ``cache/<video_id>/``, creating it (and the cache root) if absent."""
    cache_dir = config.CACHE_ROOT / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _artifact_path(video_id: str, name: str, *, create: bool = False) -> Path:
    """Where an artefact lives. Only writers ask for the directory.

    This used to create the directory unconditionally, so merely *looking* for
    a video left an empty one behind — every 404 on an unknown id, and every
    read of an artefact that happened not to exist. Those empties then showed
    up in list_cached_videos() as videos with nothing in them.
    """
    base = get_cache_dir(video_id) if create else config.CACHE_ROOT / video_id
    return base / f"{name}.json"


def write_json(
    video_id: str, name: str, data: dict[str, Any], required: bool = True
) -> None:
    """Write ``data`` to ``cache/<video_id>/<name>.json`` atomically.

    ``required=False`` downgrades a write that could not survive the retries
    from an exception to a warning. Progress state is the case that matters:
    it is rewritten on every tick, so it collides with a sync client far more
    often than any real artifact — and losing one progress update is nothing,
    while raising kills a run that was otherwise fine. Real artifacts stay
    required, because silently losing a transcript is not recoverable.

    The scratch file is named per-writer rather than per-artifact: two threads
    writing the same artifact would otherwise open one shared ``.tmp`` path at
    once, which fails on Windows with a PermissionError rather than merely
    interleaving. The lock then makes the write-then-rename pair atomic against
    other writers in this process, and the retry loop rides out the external
    locks a sync client or antivirus scanner briefly takes on the destination.
    """
    path = _artifact_path(video_id, name, create=True)
    payload = json.dumps(data, indent=2)
    tmp_path = path.with_suffix(f".json.{os.getpid()}.{threading.get_ident():x}.tmp")

    with _WRITE_LOCK:
        try:
            for attempt in range(1, _REPLACE_MAX_RETRIES + 1):
                try:
                    tmp_path.write_text(payload, encoding="utf-8")
                    tmp_path.replace(path)
                    break
                except OSError as e:
                    if attempt == _REPLACE_MAX_RETRIES:
                        if required:
                            raise
                        logger.warning(
                            "could not write %s after %d attempts (%s); continuing — "
                            "if this repeats, exclude the cache directory from your "
                            "file-sync client or set VS_CACHE_DIR outside it",
                            path.name, _REPLACE_MAX_RETRIES, e.__class__.__name__,
                        )
                        break
                    logger.debug(
                        "retrying write of %s (attempt %d/%d) after transient lock",
                        path, attempt, _REPLACE_MAX_RETRIES,
                    )
                    time.sleep(_REPLACE_RETRY_DELAY_SEC * attempt)
        finally:
            tmp_path.unlink(missing_ok=True)
    logger.debug("wrote %s", path)


def read_json(video_id: str, name: str) -> Optional[dict[str, Any]]:
    """Read ``cache/<video_id>/<name>.json``, or return None if absent/corrupt."""
    path = _artifact_path(video_id, name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("corrupt cache artifact %s; treating as absent", path)
        return None


def artifact_exists(video_id: str, name: str) -> bool:
    """Check whether ``cache/<video_id>/<name>.json`` exists and is valid JSON."""
    return read_json(video_id, name) is not None


# When two stages run concurrently (transcribe and vision), each still reports
# its own fine-grained progress. Letting both write state.json directly would
# make the two overwrite each other, so the orchestrator installs a sink and
# publishes a single merged view instead. See core.pipeline._parallel_progress.
_progress_sink: Optional[Callable[[str, str, float, str, Optional[str]], None]] = None


def set_progress_sink(sink: Optional[Callable[[str, str, float, str, Optional[str]], None]]) -> None:
    """Route :func:`set_state` through ``sink`` instead of writing state.json.

    Passing None restores direct writes. The sink is process-global, matching
    the fact that a single pipeline run owns the process while it executes.
    """
    global _progress_sink
    _progress_sink = sink


def set_state(video_id: str, stage: str, progress: float, message: str,
              error: Optional[str] = None) -> None:
    """Persist current pipeline stage/progress for the given video."""
    if _progress_sink is not None:
        _progress_sink(video_id, stage, progress, message, error)
        return
    write_state(video_id, stage, progress, message, error)


# Progress ticks arrive far faster than any consumer needs them, and each one
# is a file write — expensive when the cache lives in a synced folder, and a
# source of lock contention. Ticks are therefore coalesced to this interval.
# Terminal states always write through, so the final state is never lost.
_STATE_WRITE_MIN_INTERVAL_SEC = 0.3
_TERMINAL_STAGES = frozenset({"done", "error"})
_last_state_write: dict[str, float] = {}


def write_state(video_id: str, stage: str, progress: float, message: str,
                error: Optional[str] = None, extra: Optional[dict[str, Any]] = None) -> None:
    """Write state.json unconditionally, bypassing any installed sink.

    Intermediate progress ticks are rate-limited; terminal states, errors, and
    stage boundaries are always persisted.
    """
    important = (
        stage in _TERMINAL_STAGES or error is not None
        or progress <= 0.0 or progress >= 1.0
    )
    if not important:
        now = time.monotonic()
        if now - _last_state_write.get(video_id, 0.0) < _STATE_WRITE_MIN_INTERVAL_SEC:
            return
        _last_state_write[video_id] = now
    else:
        _last_state_write[video_id] = time.monotonic()

    payload = {
        "stage": stage,
        "progress": progress,
        "message": message,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # The run's record survives later writes. state.json holds two different
    # kinds of thing: where processing is right now, which every tick
    # replaces, and what the finished run cost — its timings and the plan it
    # chose — which is written once at the end and must outlive anything that
    # reports progress afterwards. Audio-overview generation does exactly
    # that, and without this it silently erased the timings of every video it
    # was run on, leaving the run report with nothing to show.
    for key in ("timings", "plan"):
        if extra and key in extra:
            continue
        previous = (read_json(video_id, "state") or {}).get(key)
        if previous:
            payload[key] = previous

    if extra:
        payload.update(extra)
    write_json(video_id, "state", payload, required=False)


def get_state(video_id: str) -> dict[str, Any]:
    """Return the current job state, defaulting to an unstarted state."""
    state = read_json(video_id, "state")
    if state is None:
        state = {
            "stage": "not_started",
            "progress": 0.0,
            "message": "",
            "error": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return state


def clear_cache(video_id: str) -> None:
    """Delete the entire cache directory for a video (irreversible)."""
    cache_dir = config.CACHE_ROOT / video_id
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("cleared cache for %s", video_id)


def list_cached_videos() -> list[str]:
    """Return video_ids present under the cache root (used by the UI sidebar)."""
    if not config.CACHE_ROOT.exists():
        return []
    return sorted(
        (p.name for p in config.CACHE_ROOT.iterdir() if p.is_dir()),
        key=lambda vid: (config.CACHE_ROOT / vid).stat().st_mtime,
        reverse=True,
    )
