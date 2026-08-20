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
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
_REPLACE_MAX_RETRIES = 5
_REPLACE_RETRY_DELAY_SEC = 0.1


def get_video_id(source_ref: str) -> str:
    """Derive a stable, filesystem-safe id from a URL or filename."""
    return hashlib.sha1(source_ref.encode("utf-8")).hexdigest()[:16]


def get_cache_dir(video_id: str) -> Path:
    """Return ``cache/<video_id>/``, creating it (and the cache root) if absent."""
    cache_dir = config.CACHE_ROOT / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _artifact_path(video_id: str, name: str) -> Path:
    return get_cache_dir(video_id) / f"{name}.json"


def write_json(video_id: str, name: str, data: dict[str, Any]) -> None:
    """Write ``data`` to ``cache/<video_id>/<name>.json`` atomically."""
    path = _artifact_path(video_id, name)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    for attempt in range(1, _REPLACE_MAX_RETRIES + 1):
        try:
            tmp_path.replace(path)
            break
        except OSError:
            if attempt == _REPLACE_MAX_RETRIES:
                raise
            logger.debug(
                "retrying replace of %s (attempt %d/%d) after transient lock",
                path, attempt, _REPLACE_MAX_RETRIES,
            )
            time.sleep(_REPLACE_RETRY_DELAY_SEC * attempt)
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


def set_state(video_id: str, stage: str, progress: float, message: str,
              error: Optional[str] = None) -> None:
    """Persist current pipeline stage/progress for the given video."""
    write_json(video_id, "state", {
        "stage": stage,
        "progress": progress,
        "message": message,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


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
