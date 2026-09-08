"""Formatting and export helpers for the UI layer.

Pure functions over already-computed pipeline output — no I/O, no models — so
they are cheap to call inside a Streamlit rerun and easy to test.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Optional

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/|v/))([A-Za-z0-9_-]{11})"
)


# --- Time ---------------------------------------------------------------------


def hhmmss(seconds: float) -> str:
    """``1:03:07`` for hour-long videos, ``3:07`` for shorter ones."""
    total = max(int(round(seconds)), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def srt_timestamp(seconds: float) -> str:
    """``HH:MM:SS,mmm`` — the SubRip wire format."""
    total = max(seconds, 0.0)
    hours, remainder = divmod(int(total), 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(round((total - int(total)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def human_duration(seconds: float) -> str:
    """``4 min 12 s`` / ``1 h 22 min`` — for durations shown as prose."""
    seconds = max(seconds, 0)
    if seconds < 60:
        return f"{seconds:.0f} sec"
    if seconds < 3600:
        minutes, secs = divmod(int(round(seconds)), 60)
        return f"{minutes} min {secs} sec" if secs else f"{minutes} min"
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes = remainder // 60
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


# --- Links --------------------------------------------------------------------


def youtube_id(url: str) -> Optional[str]:
    match = _YOUTUBE_ID_RE.search(url or "")
    return match.group(1) if match else None


def deep_link(source_ref: str, start_sec: float) -> str:
    """A URL that opens the source at ``start_sec``."""
    separator = "&" if "?" in source_ref else "?"
    return f"{source_ref}{separator}t={int(start_sec)}s"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text or "").strip("-").lower()
    return slug[:60] or "video"


# --- Text ---------------------------------------------------------------------


def highlight(text: str, query: str) -> str:
    """HTML-escape ``text`` and wrap case-insensitive matches of ``query`` in
    ``<mark>``. Escaping happens first so the query can never inject markup."""
    escaped = html.escape(text or "")
    query = (query or "").strip()
    if not query:
        return escaped
    return re.sub(f"({re.escape(html.escape(query))})", r"<mark>\1</mark>", escaped, flags=re.I)


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


# --- Exports ------------------------------------------------------------------


def to_markdown(meta: dict[str, Any], summary: dict[str, Any]) -> str:
    """A self-contained Markdown report: walkthrough first, recap and chapters after."""
    lines = [
        f"# {meta.get('title', 'Untitled video')}",
        "",
        f"*Source: {meta.get('source_ref', 'n/a')}*  ",
        f"*Duration: {human_duration(meta.get('duration_sec', 0))}*",
        "",
    ]
    if summary.get("warning"):
        lines += [f"> **Note:** {summary['warning'].replace('_', ' ')}", ""]

    detail = [d for d in (summary.get("chunk_summaries") or []) if d.get("text")]
    overall = summary.get("overall_summary", "") or ""

    if detail:
        lines += ["## Detailed walkthrough", ""]
        for i, entry in enumerate(detail, start=1):
            lines += [f"### {i}. [{hhmmss(entry['start'])}]", "", entry["text"], ""]
        if overall:
            lines += ["## In summary", "", overall, ""]
    else:
        lines += ["## Summary", "", overall or "_No summary produced._", ""]

    chapters = summary.get("chapters") or []
    if chapters:
        lines += ["## Chapters", ""]
        for chapter in chapters:
            lines.append(f"- **[{hhmmss(chapter['start'])}] {chapter['title']}** — {chapter.get('summary', '')}")
        lines.append("")

    return "\n".join(lines)


def to_srt(transcript: dict[str, Any]) -> str:
    """Transcript segments as a SubRip (.srt) subtitle file."""
    blocks = []
    for i, segment in enumerate(transcript.get("segments") or [], start=1):
        blocks.append(
            f"{i}\n{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}\n"
            f"{segment['text'].strip()}\n"
        )
    return "\n".join(blocks)


def to_json_export(
    meta: dict[str, Any], summary: dict[str, Any],
    scenes: Optional[dict[str, Any]], transcript: Optional[dict[str, Any]],
) -> str:
    """Everything the pipeline produced, as one machine-readable document."""
    return json.dumps(
        {
            "video": {
                "title": meta.get("title"),
                "source": meta.get("source"),
                "source_ref": meta.get("source_ref"),
                "duration_sec": meta.get("duration_sec"),
                "processed_at": meta.get("created_at"),
            },
            "summary": summary.get("overall_summary"),
            "chapters": summary.get("chapters", []),
            "section_summaries": summary.get("chunk_summaries", []),
            "scenes": [
                {
                    "start": s["start"], "end": s["end"],
                    "objects": [o["label"] for o in s.get("objects", [])],
                    "on_screen_text": s.get("ocr_text", ""),
                }
                for s in ((scenes or {}).get("scenes") or [])
            ],
            "transcript": (transcript or {}).get("segments", []),
        },
        indent=2,
        ensure_ascii=False,
    )
