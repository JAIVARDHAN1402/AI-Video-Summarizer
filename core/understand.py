"""M5' — Understanding: turn the transcript, scenes, and on-screen text into a
teachable explanation using a capable LLM.

This replaces the BART/flan-t5 map-reduce (``core.summarize``) whenever an LLM
provider is configured. The difference is not incremental: the local models can
only compress text, so on a spoken — often machine-translated — transcript they
produce disjointed fragments. A capable model, given the whole transcript at
once (no lossy chunking), can produce a self-contained explanation that a
reader learns the concept from without watching the video, filling gaps from
general knowledge and marking clearly which parts came from the video.

The output shape adapts to what kind of video it is: a coding tutorial also
gets the algorithm and multi-language implementations; a meeting also gets
decisions and action items; every type gets per-scene notes on what each
slide/diagram shows.

``core.summarize`` remains the fallback when no provider is available.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, Optional

import config
from pathlib import Path

from core import cache, llm, perf

logger = logging.getLogger(__name__)


class VideoType(str, Enum):
    """What the video is, which shapes the prompt and the output sections."""

    LECTURE = "lecture"          # study / educational explainer
    CODING = "coding"            # programming, DSA, algorithms
    MEETING = "meeting"          # meeting / standup / call recording
    INTERVIEW = "interview"      # interview, podcast, panel
    TALK = "talk"               # conference talk, keynote, seminar
    NEWS = "news"               # news, current affairs
    DOCUMENTARY = "documentary"  # documentary, long-form explainer
    OTHER = "other"

    @classmethod
    def coerce(cls, value: Optional[str]) -> "VideoType":
        if not value:
            return cls.LECTURE
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.OTHER


# --- Personas -------------------------------------------------------------------
#
# The system instruction is where the "act as a tutor who completes the
# explanation" behaviour lives. Each type extends a shared base.

_BASE_PERSONA = """You are an expert explainer. You are given the full transcript of a video, \
a list of its visual scenes with any on-screen text, and the video's stated topic. \
Your job is to produce an explanation and summary so complete and clear that a reader \
understands the material fully WITHOUT watching the video.

Rules:
- Write in plain, well-structured prose. Short paragraphs. No filler, no "in this video".
- The transcript is the primary source, but it may be rough, out of order, or a \
machine translation. Reconstruct the intended meaning. Fix obvious transcription errors.
- Where the video is incomplete, unclear, or skips a step, fill the gap with correct \
standard knowledge so the explanation stands on its own. Set "from_video": false for \
any explanation section that is mostly your added context rather than the video's content.
- Every explanation section and chapter must carry the start time (in seconds) of where \
that material begins in the video, taken from the transcript timestamps.
- Be accurate. Do not invent facts, names, numbers, or quotes that aren't supported by \
the transcript or by well-established general knowledge.
- Output ONLY valid JSON matching the given schema."""

_PERSONA_BY_TYPE: dict[VideoType, str] = {
    VideoType.LECTURE: _BASE_PERSONA + """

This is an educational video. Teach the concept step by step, as a good tutor would: \
define terms before using them, give the intuition before the formalism, and include \
a short worked example where it helps. The reader should be able to explain the topic \
back to someone else after reading.""",
    VideoType.CODING: _BASE_PERSONA + """

This is a programming / algorithms video. In addition to the explanation, populate the \
"coding" object: name the algorithm or technique, give the intuition, list the steps, \
state time and space complexity, and provide clean, idiomatic, correct implementations \
in Python, Java, C++, and JavaScript. The code must actually run and solve the problem \
discussed — derive it from the approach explained in the video, corrected and completed \
with your own knowledge if the video's code is partial or buggy.""",
    VideoType.MEETING: _BASE_PERSONA + """

This is a meeting recording. The "explanation" should be a readable narrative of what \
was discussed and why. Also populate the "meeting" object: every decision reached, and \
every action item with its owner and due date if stated. Keep it factual — do not \
soften or editorialise.""",
    VideoType.INTERVIEW: _BASE_PERSONA + """

This is an interview or podcast. Organise the explanation around the questions and the \
substantive points each speaker made. Attribute claims to the speaker where the \
transcript makes the speaker clear.""",
    VideoType.TALK: _BASE_PERSONA + """

This is a conference talk or seminar. Capture the speaker's thesis, the argument or \
evidence for it, and the practical takeaways.""",
    VideoType.NEWS: _BASE_PERSONA + """

This is a news segment. Lead with what happened, then context and implications. \
Separate reported fact from opinion or speculation.""",
    VideoType.DOCUMENTARY: _BASE_PERSONA + """

This is a documentary or long-form explainer. Follow its narrative arc and draw out \
the through-line, while keeping the explanation self-contained.""",
    VideoType.OTHER: _BASE_PERSONA,
}


# --- Output schema -----------------------------------------------------------------
#
# Gemini enforces this server-side, which removes the "prose wrapped around the
# JSON" failure mode almost entirely.

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overview": {
            "type": "string",
            "description": "2-4 sentences: what the video covers and what the reader will learn.",
        },
        "explanation": {
            "type": "array",
            "description": "The full teaching walkthrough, in reading order.",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "start_sec": {"type": "number"},
                    "body": {"type": "string", "description": "One or more full paragraphs."},
                    "from_video": {
                        "type": "boolean",
                        "description": "true if mostly from the video; false if mostly added context.",
                    },
                },
                "required": ["heading", "start_sec", "body", "from_video"],
            },
        },
        "summary": {
            "type": "string",
            "description": "A tight recap of the whole thing in one short paragraph.",
        },
        "key_points": {"type": "array", "items": {"type": "string"}},
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
                "required": ["term", "definition"],
            },
        },
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["start_sec", "title", "summary"],
            },
        },
        "scene_notes": {
            "type": "array",
            "description": "For scenes that show a slide, diagram, code, or board: what it shows.",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "what_it_shows": {"type": "string"},
                },
                "required": ["start_sec", "what_it_shows"],
            },
        },
        "coding": {
            "type": "object",
            "properties": {
                "algorithm_name": {"type": "string"},
                "intuition": {"type": "string"},
                "steps": {"type": "array", "items": {"type": "string"}},
                "time_complexity": {"type": "string"},
                "space_complexity": {"type": "string"},
                "implementations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "code": {"type": "string"},
                        },
                        "required": ["language", "code"],
                    },
                },
            },
        },
        "meeting": {
            "type": "object",
            "properties": {
                "decisions": {"type": "array", "items": {"type": "string"}},
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "owner": {"type": "string"},
                            "task": {"type": "string"},
                            "due": {"type": "string"},
                        },
                        "required": ["task"],
                    },
                },
            },
        },
    },
    "required": ["overview", "explanation", "summary", "key_points", "chapters"],
}


# --- Context assembly ------------------------------------------------------------


def _mmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    return f"{total // 60:02d}:{total % 60:02d}"


def _format_transcript(segments: list[dict[str, Any]]) -> str:
    """Timestamped transcript, one line per segment.

    The timestamps are what lets the model anchor each explanation section and
    chapter to a real point in the video.
    """
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"[{_mmss(seg['start'])}] {text}")
    return "\n".join(lines)


def _format_scenes(scenes: list[dict[str, Any]]) -> str:
    """Scene list with detected objects and on-screen text.

    Only scenes that carry information (readable text, or a distinctive object
    set) are worth the tokens; a plain talking-head scene adds nothing.
    """
    # A scene whose only detected object is "person" and that has no readable
    # text is a talking-head shot — it carries nothing the explanation needs.
    _NOISE = {"person", "tie", "chair"}
    lines = []
    for scene in scenes:
        ocr = (scene.get("ocr_text") or "").strip()
        labels = [o["label"] for o in scene.get("objects", [])[:5]]
        informative = [l for l in labels if l not in _NOISE]
        if not ocr and not informative:
            continue
        objects = ", ".join(labels)
        parts = [f"[{_mmss(scene['start'])}]"]
        if objects:
            parts.append(f"shows: {objects}")
        if ocr:
            parts.append(f'on screen: "{ocr}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _select_scene_images(
    scenes: list[dict[str, Any]], limit: int
) -> list[llm.ImageRef]:
    """Pick the keyframes worth sending to a multimodal model and encode them.

    Only scenes that show real material (readable text, or a non-talking-head
    object set) are candidates; sending a webcam shot wastes ~260 tokens for
    nothing. When there are more candidates than the cap: if the M7 importance
    model has scored the scenes, keep the highest-importance ones (then restore
    time order); otherwise sample evenly across the runtime. Each frame is
    downscaled and JPEG-compressed to keep the request small.
    """
    try:
        import cv2
    except Exception:
        logger.warning("opencv unavailable; sending no scene images", exc_info=True)
        return []

    noise = {"person", "tie", "chair"}
    candidates = []
    for sc in scenes:
        path = sc.get("keyframe_path")
        if not path or not Path(path).exists():
            continue
        ocr = (sc.get("ocr_text") or "").strip()
        informative = [o["label"] for o in sc.get("objects", []) if o["label"] not in noise]
        if ocr or informative:
            candidates.append(sc)

    if not candidates:
        return []
    if len(candidates) > limit:
        if any("importance" in sc for sc in candidates):
            candidates = sorted(
                candidates, key=lambda sc: sc.get("importance", 0.0), reverse=True
            )[:limit]
            candidates.sort(key=lambda sc: sc.get("start", 0.0))
        else:
            step = len(candidates) / limit
            candidates = [candidates[int(i * step)] for i in range(limit)]

    images: list[llm.ImageRef] = []
    for sc in candidates:
        img = cv2.imread(str(sc["keyframe_path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        target_w = config.UNDERSTANDING_SCENE_IMAGE_WIDTH
        if w > target_w:
            img = cv2.resize(img, (target_w, int(h * target_w / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            images.append((f"scene at {_mmss(sc['start'])}", buf.tobytes()))
    logger.info("attaching %d scene images to the understanding prompt", len(images))
    return images

def _build_prompt(
    meta: dict[str, Any], transcript: dict[str, Any], scenes_data: Optional[dict[str, Any]],
    video_type: VideoType, topic: str,
) -> str:
    duration_min = meta.get("duration_sec", 0) / 60
    segments = transcript.get("segments") or []
    scenes = (scenes_data or {}).get("scenes") or []

    header = [
        f"VIDEO TITLE: {meta.get('title', 'Untitled')}",
        f"VIDEO TYPE: {video_type.value}",
        f"STATED TOPIC: {topic or '(not given — infer it from the content)'}",
        f"DURATION: {duration_min:.0f} minutes",
        f"TRANSCRIPT LANGUAGE: {transcript.get('language', 'en')}"
        + (" (translated to English)" if transcript.get("translated") else ""),
        "",
        "=== TRANSCRIPT (timestamped) ===",
        _format_transcript(segments) or "(no speech detected)",
    ]
    scene_block = _format_scenes(scenes)
    if scene_block:
        header += ["", "=== VISUAL SCENES (objects + on-screen text) ===", scene_block]

    header += [
        "",
        "=== TASK ===",
        "Produce the JSON described by the schema. The 'explanation' array is the "
        "most important field: it must be a complete, self-contained teaching of "
        "the material. Anchor every section and chapter to a transcript timestamp "
        "(start_sec) — use DISTINCT timestamps, not the same one repeated. "
        "Populate 'scene_notes' ONLY for scenes that show a slide, diagram, board, "
        "code editor, or document — skip plain talking-head or webcam shots. "
        "Keyframe images for the more informative scenes are attached below; use "
        "them to read diagrams, handwriting, and layout that the OCR text misses, "
        "and to make the scene_notes and explanation more precise.",
    ]
    if video_type is VideoType.CODING:
        header.append(
            "This is a coding video: also fill 'coding' with the algorithm, "
            "complexity, and working Python / Java / C++ / JavaScript implementations."
        )
    if video_type is VideoType.MEETING:
        header.append("This is a meeting: also fill 'meeting' with decisions and action items.")

    return "\n".join(header)


# --- Public API ------------------------------------------------------------------


def is_available() -> bool:
    """True when an LLM provider is configured (so the pipeline should use this)."""
    return llm.get_client().available


def generate(
    video_id: str,
    *,
    video_type: str | VideoType = VideoType.LECTURE,
    topic: str = "",
    force: bool = False,
) -> dict[str, Any]:
    """Produce understanding.json for a processed video.

    Args:
        video_id: The cache key for this video.
        video_type: One of :class:`VideoType` (or its string value).
        topic: The user-supplied topic, if any — steers extraction.
        force: Re-run even if a cached understanding.json exists.

    Returns:
        The understanding.json dict.

    Raises:
        llm.LLMUnavailable: no provider configured (caller should fall back).
        llm.LLMError: the model call failed or returned something unusable.
        RuntimeError: required upstream artifacts are missing.
    """
    vtype = VideoType.coerce(video_type if isinstance(video_type, str) else video_type.value)

    if not force and cache.artifact_exists(video_id, "understanding"):
        logger.info("using cached understanding.json for %s", video_id)
        return cache.read_json(video_id, "understanding")  # type: ignore[return-value]

    meta = cache.read_json(video_id, "meta")
    transcript = cache.read_json(video_id, "transcript")
    if meta is None or transcript is None:
        raise RuntimeError(f"meta.json and transcript.json are required for {video_id}")
    scenes_data = cache.read_json(video_id, "scenes")

    client = llm.get_client()
    persona = _PERSONA_BY_TYPE[vtype]
    prompt = _build_prompt(meta, transcript, scenes_data, vtype, topic)
    images = (
        _select_scene_images(
            (scenes_data or {}).get("scenes") or [],
            config.UNDERSTANDING_MAX_SCENE_IMAGES,
        )
        if config.UNDERSTANDING_SEND_SCENE_IMAGES
        else []
    )

    cache.set_state(video_id, "summarize", 0.2, f"Understanding the video ({client.model})...")
    started = time.perf_counter()
    thinking = (
        config.UNDERSTANDING_THINKING_BUDGET_CODING
        if vtype is VideoType.CODING
        else config.UNDERSTANDING_THINKING_BUDGET
    )
    result = client.generate_json(
        persona, prompt,
        schema=_SCHEMA,
        temperature=config.UNDERSTANDING_TEMPERATURE,
        max_output_tokens=config.UNDERSTANDING_MAX_OUTPUT_TOKENS,
        thinking_budget=thinking,
        images=images,
    )
    elapsed = time.perf_counter() - started
    perf.record_observation(
        "understand:per_min",
        elapsed / max(meta.get("duration_sec", 60) / 60, 1),
    )

    data = _normalise(result.data, transcript, scenes_data)
    understanding = {
        "video_type": vtype.value,
        "topic": topic,
        "model": result.model,
        "generated_in_sec": round(elapsed, 1),
        "input_truncated": result.truncated_input,
        "finish_reason": result.finish_reason,
        **data,
    }
    cache.write_json(video_id, "understanding", understanding)
    logger.info(
        "understanding for %s: %d explanation sections, %d chapters, %d scene notes in %.1fs",
        video_id, len(data["explanation"]), len(data["chapters"]),
        len(data.get("scene_notes", [])), elapsed,
    )
    return understanding


def _normalise(
    data: dict[str, Any], transcript: dict[str, Any], scenes_data: Optional[dict[str, Any]]
) -> dict[str, Any]:
    """Coerce the model's reply into the shape the rest of the app expects.

    The schema is enforced server-side so this is light: clamp timestamps to
    the real duration, drop empty sections, and attach each scene note to its
    keyframe path so the UI can show the image beside the explanation.
    """
    segments = transcript.get("segments") or []
    max_time = segments[-1]["end"] if segments else 0.0

    def clamp(t: Any) -> float:
        try:
            return max(0.0, min(float(t), max_time))
        except (TypeError, ValueError):
            return 0.0

    explanation = [
        {
            "heading": s.get("heading", "").strip() or "Section",
            "start_sec": clamp(s.get("start_sec")),
            "body": s.get("body", "").strip(),
            "from_video": bool(s.get("from_video", True)),
        }
        for s in (data.get("explanation") or [])
        if s.get("body", "").strip()
    ]
    explanation.sort(key=lambda s: s["start_sec"])

    chapters = [
        {
            "start": clamp(c.get("start_sec")),
            "title": c.get("title", "").strip() or "Chapter",
            "summary": c.get("summary", "").strip(),
        }
        for c in (data.get("chapters") or [])
        if c.get("title", "").strip()
    ]
    chapters.sort(key=lambda c: c["start"])

    # Attach keyframe paths to scene notes by nearest scene start.
    scenes = (scenes_data or {}).get("scenes") or []
    scene_notes = []
    for note in data.get("scene_notes") or []:
        if not note.get("what_it_shows", "").strip():
            continue
        t = clamp(note.get("start_sec"))
        nearest = min(scenes, key=lambda sc: abs(sc["start"] - t), default=None)
        scene_notes.append({
            "start": t,
            "what_it_shows": note["what_it_shows"].strip(),
            "keyframe_path": (nearest or {}).get("keyframe_path"),
            "on_screen_text": (nearest or {}).get("ocr_text", ""),
        })
    scene_notes.sort(key=lambda n: n["start"])

    out: dict[str, Any] = {
        "overview": data.get("overview", "").strip(),
        "explanation": explanation,
        "summary": data.get("summary", "").strip(),
        "key_points": [p.strip() for p in (data.get("key_points") or []) if p.strip()],
        "glossary": [
            {"term": g["term"].strip(), "definition": g["definition"].strip()}
            for g in (data.get("glossary") or [])
            if g.get("term", "").strip() and g.get("definition", "").strip()
        ],
        "chapters": chapters,
        "scene_notes": scene_notes,
    }

    coding = data.get("coding") or {}
    impls = [
        {"language": i["language"].strip(), "code": i["code"].strip()}
        for i in (coding.get("implementations") or [])
        if i.get("language", "").strip() and i.get("code", "").strip()
    ]
    if coding.get("algorithm_name") or impls:
        out["coding"] = {
            "algorithm_name": coding.get("algorithm_name", "").strip(),
            "intuition": coding.get("intuition", "").strip(),
            "steps": [s.strip() for s in (coding.get("steps") or []) if s.strip()],
            "time_complexity": coding.get("time_complexity", "").strip(),
            "space_complexity": coding.get("space_complexity", "").strip(),
            "implementations": impls,
        }

    meeting = data.get("meeting") or {}
    decisions = [d.strip() for d in (meeting.get("decisions") or []) if d.strip()]
    actions = [
        {
            "owner": a.get("owner", "").strip(),
            "task": a["task"].strip(),
            "due": a.get("due", "").strip(),
        }
        for a in (meeting.get("action_items") or [])
        if a.get("task", "").strip()
    ]
    if decisions or actions:
        out["meeting"] = {"decisions": decisions, "action_items": actions}

    return out
