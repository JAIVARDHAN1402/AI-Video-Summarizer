"""Build a short highlight reel from the M7 scene-importance scores.

Greedy-picks whole scenes in descending ``importance`` until a time budget is
hit, puts them back in chronological order, and stitches them with one ffmpeg
``select`` pass (no temp files). Needs ``scenes.json`` to carry ``importance``
(i.e. a trained highlight checkpoint was active during the vision stage) and
the source ``video.mp4`` in the cache dir.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import config
from core import cache

logger = logging.getLogger("core.highlight")

DEFAULT_MAX_SECONDS = float(getattr(config, "HIGHLIGHT_MAX_SECONDS", 90.0))
MIN_SCENE_SECONDS = 1.5


def _pick_scenes(scenes: list[dict[str, Any]], max_seconds: float) -> list[dict[str, Any]]:
    scored = [s for s in scenes if "importance" in s and (s.get("end", 0) - s.get("start", 0)) >= MIN_SCENE_SECONDS]
    if not scored:
        return []
    chosen: list[dict[str, Any]] = []
    total = 0.0
    for s in sorted(scored, key=lambda s: s["importance"], reverse=True):
        length = float(s["end"]) - float(s["start"])
        if total + length > max_seconds and chosen:
            continue
        chosen.append(s)
        total += length
        if total >= max_seconds:
            break
    chosen.sort(key=lambda s: s["start"])
    return chosen


def _ffmpeg_select_expr(spans: list[tuple[float, float]]) -> str:
    return "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in spans)


def build_highlight(
    video_id: str, *, max_seconds: float = DEFAULT_MAX_SECONDS, force: bool = False,
) -> Optional[Path]:
    """Write ``cache/<video_id>/highlight.mp4`` and return its path, or ``None``
    if there's nothing to build (no importance scores, no video, ffmpeg error)."""
    cache_dir = cache.get_cache_dir(video_id)
    out_path = cache_dir / "highlight.mp4"
    if out_path.exists() and not force:
        return out_path

    scenes_data = cache.read_json(video_id, "scenes")
    video_path = cache_dir / "video.mp4"
    if not scenes_data or not video_path.exists():
        logger.info("highlight: missing scenes.json or video.mp4 for %s", video_id)
        return None

    picks = _pick_scenes(scenes_data.get("scenes") or [], max_seconds)
    if not picks:
        logger.info("highlight: no importance-scored scenes for %s (train the M7 model)", video_id)
        return None

    spans = [(float(s["start"]), float(s["end"])) for s in picks]
    vexpr = _ffmpeg_select_expr(spans)
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vf", f"select='{vexpr}',setpts=N/FRAME_RATE/TB",
        "-af", f"aselect='{vexpr}',asetpts=N/SR/TB",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_path.exists():
        logger.warning("highlight: ffmpeg failed for %s: %s", video_id, result.stderr.strip()[-500:])
        return None

    total = sum(b - a for a, b in spans)
    logger.info("highlight: %s -> %d scenes, %.0fs", video_id, len(picks), total)
    scenes_data["highlight"] = {
        "path": str(out_path),
        "scene_ids": [s["id"] for s in picks],
        "spans": [[round(a, 2), round(b, 2)] for a, b in spans],
        "seconds": round(total, 1),
    }
    cache.write_json(video_id, "scenes", scenes_data)
    return out_path


def get_highlight_path(video_id: str) -> Optional[Path]:
    p = cache.get_cache_dir(video_id) / "highlight.mp4"
    return p if p.exists() else None
