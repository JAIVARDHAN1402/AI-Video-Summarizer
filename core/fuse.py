"""M4 — Fusion: align transcript segments to visual scenes, producing
timestamped blocks that carry both spoken text and visual tags.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core import cache

logger = logging.getLogger(__name__)

# A scene with no overlapping speech is only dropped if it's this short —
# a longer silent scene is still meaningful visual information to keep.
_MIN_BLOCK_LEN_TO_KEEP_EMPTY_SEC = 5.0


def _format_mmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _fuse_segments_and_scenes(
    segments: list[dict[str, Any]], scenes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    next_id = 0
    n_scenes = len(scenes)

    for idx, scene in enumerate(scenes):
        s_start, s_end = scene["start"], scene["end"]
        is_last = idx == n_scenes - 1

        matched = []
        for seg in segments:
            midpoint = (seg["start"] + seg["end"]) / 2.0
            # Half-open [s_start, s_end) so contiguous scenes never double-claim
            # a segment; the last scene is closed at the end so nothing near
            # the video's final timestamp is dropped.
            if s_start <= midpoint < s_end or (is_last and midpoint == s_end):
                matched.append(seg)

        text = " ".join(seg["text"] for seg in matched).strip()

        if not text and (s_end - s_start) < _MIN_BLOCK_LEN_TO_KEEP_EMPTY_SEC:
            continue

        visual_tags = sorted({obj["label"] for obj in scene.get("objects", [])})
        tags_str = ", ".join(visual_tags)
        fused_text = (
            f"[{_format_mmss(s_start)}-{_format_mmss(s_end)}] "
            f"(visual: {tags_str}) {text}"
        ).strip()

        blocks.append({
            "id": next_id,
            "start": s_start,
            "end": s_end,
            "scene_ids": [scene["id"]],
            "visual_tags": visual_tags,
            "text": text,
            "fused_text": fused_text,
        })
        next_id += 1

    return blocks


def fuse(video_id: str, force: bool = False) -> dict[str, Any]:
    """Align transcript segments to scenes into fused.json.

    Args:
        video_id: The cache key for this video.
        force: Re-run even if a cached fused.json already exists.

    Returns:
        The fused.json dict.
    """
    if not force and cache.artifact_exists(video_id, "fused"):
        logger.info("using cached fused.json for %s", video_id)
        return cache.read_json(video_id, "fused")  # type: ignore[return-value]

    transcript = cache.read_json(video_id, "transcript")
    scenes_data = cache.read_json(video_id, "scenes")
    if transcript is None or scenes_data is None:
        raise RuntimeError(
            f"transcript.json and scenes.json are both required to fuse {video_id}"
        )

    cache.set_state(video_id, "fuse", 0.0, "Fusing transcript with visual scenes...")
    blocks = _fuse_segments_and_scenes(transcript["segments"], scenes_data["scenes"])

    result = {"blocks": blocks}
    cache.write_json(video_id, "fused", result)
    cache.set_state(video_id, "fuse", 1.0, "Fusion complete.")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M4 fusion in isolation.")
    parser.add_argument("video_id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = fuse(args.video_id, force=args.force)
    print(json.dumps(result, indent=2)[:2000])
