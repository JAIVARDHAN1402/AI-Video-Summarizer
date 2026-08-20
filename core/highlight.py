"""M7 — Highlight video generation. FUTURE SCOPE, not implemented.

Stitching the top-ranked scenes into a short highlight clip (via MoviePy) is
explicitly out of scope for this prototype (see report Section 6, item O6).
This stub exists solely so the architecture diagram in the report is honest
about what is and isn't built.
"""

from __future__ import annotations


def build_highlight(video_id: str, target_duration_sec: int = 120) -> str:
    """FUTURE SCOPE: stitch top-ranked scenes into a short clip via MoviePy.

    Args:
        video_id: The cache key for the source video.
        target_duration_sec: Desired length of the generated highlight clip.

    Returns:
        Path to the generated highlight video (not implemented).
    """
    raise NotImplementedError("Highlight video is future scope (see report Section 6, O6).")
