"""Quiz generation from a video's understanding document.

The explanation stage has already read the whole video and written a structured
document: an overview, a section-by-section walkthrough, key points and a
glossary. Generating questions from *that* rather than from the raw transcript
is both cheaper and better — the hard work of deciding what the video actually
teaches has been done, and the document is a fraction of the transcript's size.

Questions carry the timestamp of the section they came from, so a wrong answer
can send the reader back to the exact moment that covers it.

Like the understanding layer this needs a provider; unlike it there is no local
fallback, because a BART-class model cannot write a plausible distractor. With
no key configured the feature is simply unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import config
from core import cache, llm

logger = logging.getLogger(__name__)

DEFAULT_COUNT = 10
MAX_COUNT = 25

# The understanding document for a 12-hour video is large. Questions are drawn
# from what the video teaches rather than from every sentence of it, so the
# source is capped — the overview, key points and glossary always survive, and
# the walkthrough is sampled evenly across the video's length so the quiz does
# not end up about the first twenty minutes.
_MAX_SOURCE_CHARS = 60_000
_MAX_BODY_CHARS = 900

_PERSONA = """You write exam questions for a student who has just studied this \
material and wants to find out what they have not understood.

Rules you must follow:
- Every question must be answerable from the material given. Do not test \
trivia, phrasing, or anything the material does not state.
- Exactly four options. Exactly one is correct.
- Wrong options must be plausible to someone who half-understands the topic — \
a common misconception, a neighbouring concept, an off-by-one, a swapped \
cause and effect. Never filler, never obviously absurd, never "none of the \
above".
- Options must be similar in length and specificity. A noticeably longer or \
more detailed option gives the answer away.
- The explanation says why the right answer is right AND why the most \
tempting wrong option is wrong.
- start_sec is in SECONDS. Copy the number from the [t=N seconds] marker on \
the section the question came from. It is never a minute count.
- Spread the questions across the whole of the material, not just its opening.
- Mix difficulty: roughly a third easy (recall), a half medium (apply), the \
rest hard (connect two ideas)."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Exactly four options.",
                    },
                    "correct_index": {
                        "type": "integer",
                        "description": "0-based index into options.",
                    },
                    "explanation": {"type": "string"},
                    "start_sec": {
                        "type": "number",
                        "description": (
                            "Seconds from the start of the video: copy the "
                            "number from the [t=N seconds] marker on the "
                            "section this question came from. Never minutes."
                        ),
                    },
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": [
                    "question",
                    "options",
                    "correct_index",
                    "explanation",
                    "start_sec",
                    "difficulty",
                ],
            },
        }
    },
    "required": ["questions"],
}


def is_available() -> bool:
    """Whether a quiz can be generated at all."""
    return llm.get_client().available


def _sample_sections(sections: list[dict], budget: int) -> list[dict]:
    """Take sections spread across the video until the budget is spent.

    Truncating from the front would make every quiz about the introduction, so
    this walks the list at a stride that lands roughly within budget and keeps
    the ordering.
    """
    if not sections:
        return []
    total = sum(min(len(s.get("body") or ""), _MAX_BODY_CHARS) for s in sections)
    if total <= budget:
        return sections
    stride = max(1, round(total / budget))
    return sections[::stride]


def _source_text(u: dict[str, Any]) -> str:
    parts: list[str] = []

    if u.get("overview"):
        parts.append(f"OVERVIEW\n{u['overview']}")

    points = u.get("key_points") or []
    if points:
        parts.append("KEY POINTS\n" + "\n".join(f"- {p}" for p in points))

    glossary = u.get("glossary") or []
    if glossary:
        parts.append(
            "GLOSSARY\n"
            + "\n".join(f"- {g.get('term')}: {g.get('definition')}" for g in glossary)
        )

    used = sum(len(p) for p in parts)
    sections = _sample_sections(u.get("explanation") or [], max(_MAX_SOURCE_CHARS - used, 2000))
    if sections:
        body = []
        for s in sections:
            text = (s.get("body") or "")[:_MAX_BODY_CHARS]
            body.append(
                f"[t={int(s.get('start_sec') or 0)} seconds] "
                f"{s.get('heading') or ''}\n{text}"
            )
        parts.append("WALKTHROUGH\n" + "\n\n".join(body))

    return "\n\n".join(parts)[:_MAX_SOURCE_CHARS]


def _clean(raw: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Keep only questions that are actually answerable as posed.

    A model that returns three options, or an index outside them, has produced
    something the UI cannot score — dropping those is better than rendering a
    question with no right answer.
    """
    out: list[dict[str, Any]] = []
    for q in raw.get("questions") or []:
        options = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        if len(options) != 4:
            continue
        try:
            idx = int(q.get("correct_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < 4:
            continue
        question = str(q.get("question") or "").strip()
        if not question:
            continue
        if len(set(o.lower() for o in options)) != 4:
            continue  # duplicated options make two answers correct
        out.append(
            {
                "question": question,
                "options": options,
                "correct_index": idx,
                "explanation": str(q.get("explanation") or "").strip(),
                "start_sec": float(q.get("start_sec") or 0),
                "difficulty": q.get("difficulty")
                if q.get("difficulty") in {"easy", "medium", "hard"}
                else "medium",
            }
        )
        if len(out) >= count:
            break
    return out


def generate(
    video_id: str,
    *,
    count: int = DEFAULT_COUNT,
    force: bool = False,
) -> dict[str, Any]:
    """Produce (and cache) a multiple-choice quiz for one video.

    Raises:
        llm.LLMUnavailable: no provider configured.
        llm.LLMError: the model call failed, or returned nothing usable.
        FileNotFoundError: the video has not been through the understanding
            stage, so there is nothing to ask questions about.
    """
    count = max(3, min(int(count), MAX_COUNT))

    if not force:
        cached = cache.read_json(video_id, "quiz")
        # A cached quiz of a different length is still useful; only regenerate
        # when the reader explicitly asks for a fresh one.
        if cached and cached.get("questions"):
            return cached

    u = cache.read_json(video_id, "understanding")
    if not u:
        raise FileNotFoundError(
            "this video has no understanding document — quizzes are generated "
            "from the explanation, not from the raw transcript"
        )

    client = llm.get_client()
    if not client.available:
        raise llm.LLMUnavailable("no LLM provider configured")

    source = _source_text(u)
    if len(source) < 200:
        raise llm.LLMError("the understanding document is too thin to build a quiz from")

    started = time.perf_counter()
    result = client.generate_json(
        _PERSONA,
        f"Write {count} multiple-choice questions on the following material.\n\n{source}",
        schema=_SCHEMA,
        temperature=0.6,  # some spread, so a regenerated quiz is not the same one
        max_output_tokens=config.UNDERSTANDING_MAX_OUTPUT_TOKENS,
    )
    elapsed = time.perf_counter() - started

    questions = _clean(result.data, count)
    if not questions:
        # Name the model: when the primary provider is out of quota the chain
        # falls through to the next one, and "the model" alone leaves the
        # reader guessing which of them actually answered.
        raise llm.LLMError(
            f"{result.model} returned no usable questions")

    quiz = {
        "video_id": video_id,
        "model": result.model,
        "generated_in_sec": round(elapsed, 1),
        "requested": count,
        "questions": questions,
    }
    cache.write_json(video_id, "quiz", quiz)
    logger.info(
        "quiz for %s: %d questions (asked for %d) in %.1fs",
        video_id, len(questions), count, elapsed,
    )
    return quiz


def get(video_id: str) -> Optional[dict[str, Any]]:
    """The cached quiz, if one has been generated."""
    return cache.read_json(video_id, "quiz")
