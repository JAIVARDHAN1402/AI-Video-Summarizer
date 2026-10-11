"""Reader-authored notes attached to a video.

Everything else in the cache is derived — the pipeline can rebuild it from the
source at any time. Notes are the one thing it cannot: they are the reader's
own words, so they are never touched by a re-run and never cleared by a cache
invalidation of the derived artefacts.

A note may carry a timestamp, which is what makes it worth keeping next to the
video rather than in a separate app: a thought recorded at 14:02 links back to
the moment that provoked it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from core import cache

MAX_NOTES = 500
MAX_CHARS = 8000


def _load(video_id: str) -> list[dict[str, Any]]:
    doc = cache.read_json(video_id, "notes") or {}
    notes = doc.get("notes")
    return notes if isinstance(notes, list) else []


def _save(video_id: str, notes: list[dict[str, Any]]) -> None:
    cache.write_json(video_id, "notes", {"notes": notes})


def list_notes(video_id: str) -> list[dict[str, Any]]:
    """Newest first, which is the order a reader adds and reviews them in."""
    return sorted(_load(video_id), key=lambda n: n.get("created_at", 0), reverse=True)


def add(
    video_id: str,
    text: str,
    *,
    start_sec: Optional[float] = None,
    source: str = "manual",
) -> dict[str, Any]:
    """Append a note. ``source`` records where it came from — typed by the
    reader, or saved from a chat answer — so the UI can mark the difference.

    Raises:
        ValueError: empty text, or the per-video ceiling reached.
    """
    body = (text or "").strip()
    if not body:
        raise ValueError("a note needs some text")
    if len(body) > MAX_CHARS:
        body = body[:MAX_CHARS]

    notes = _load(video_id)
    if len(notes) >= MAX_NOTES:
        raise ValueError(f"this video already has the maximum of {MAX_NOTES} notes")

    note = {
        "id": uuid.uuid4().hex[:12],
        "text": body,
        "created_at": time.time(),
        "start_sec": float(start_sec) if start_sec is not None else None,
        "source": source if source in {"manual", "chat", "explanation"} else "manual",
    }
    notes.append(note)
    _save(video_id, notes)
    return note


def update(video_id: str, note_id: str, text: str) -> dict[str, Any]:
    """Rewrite one note's text, leaving its timestamp and origin alone."""
    body = (text or "").strip()
    if not body:
        raise ValueError("a note needs some text")

    notes = _load(video_id)
    for n in notes:
        if n.get("id") == note_id:
            n["text"] = body[:MAX_CHARS]
            n["edited_at"] = time.time()
            _save(video_id, notes)
            return n
    raise KeyError(note_id)


def delete(video_id: str, note_id: str) -> None:
    notes = _load(video_id)
    remaining = [n for n in notes if n.get("id") != note_id]
    if len(remaining) == len(notes):
        raise KeyError(note_id)
    _save(video_id, remaining)
