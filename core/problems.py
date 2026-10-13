"""The problems a coding video actually works through, and their solutions.

A three-hour SQL practice video solves seventy questions. The explanation
tells you what the video teaches; it does not hand you the seventy questions
as a list you can revise from, and it certainly does not give you each answer
in the dialect you happen to use. That is what this builds.

Two things are separated on purpose:

*Extraction* reads the understanding document plus the text that was on screen
and writes down every problem the video solves, with the solution in the
language the video used. This is one pass over the video and is done once.

*Translation* rewrites one problem's solution into the other languages of the
same family — MySQL into PostgreSQL, Oracle, SQL Server; Java into Python,
C++. This is deliberately on demand and cached per problem, because a free
tier allows tens of requests a day and seventy problems times five dialects
would spend a month of them to produce text nobody asked to read.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Iterable, Optional

import config
from core import cache, llm

logger = logging.getLogger(__name__)

# Both writers rewrite the whole document, so two translations running at
# once would each save a copy made before the other started and one of
# them would simply vanish. The model call is deliberately outside this:
# only the read-modify-write is serialised.
_write_lock = threading.Lock()

# A family is the set of languages it is meaningful to translate between. SQL
# dialects differ in syntax but express the same query; asking for a SQL
# question "in Python" is a different exercise, not a translation, so the
# families do not mix.
FAMILIES: dict[str, list[str]] = {
    "sql": ["MySQL", "PostgreSQL", "Oracle", "SQL Server", "SQLite"],
    "programming": ["Python", "Java", "C++", "JavaScript", "C#", "Go"],
}

# Aliases the model reaches for, mapped onto the canonical names above, so a
# solution labelled "Postgres" does not become a sixth language.
_ALIASES = {
    "postgres": "PostgreSQL", "postgresql": "PostgreSQL", "psql": "PostgreSQL",
    "mysql": "MySQL", "mariadb": "MySQL",
    "oracle": "Oracle", "plsql": "Oracle", "pl/sql": "Oracle", "oracle sql": "Oracle",
    "sql server": "SQL Server", "mssql": "SQL Server", "t-sql": "SQL Server",
    "tsql": "SQL Server", "transact-sql": "SQL Server",
    "sqlite": "SQLite", "sql": "MySQL", "ansi sql": "MySQL",
    "python": "Python", "python3": "Python", "py": "Python",
    "java": "Java", "c++": "C++", "cpp": "C++", "c plus plus": "C++",
    "javascript": "JavaScript", "js": "JavaScript", "typescript": "JavaScript",
    "node": "JavaScript", "c#": "C#", "csharp": "C#", "go": "Go", "golang": "Go",
}

# How much source text one extraction call is given. The reply carries whole
# solutions, so a window that is comfortable to read is already large to write
# back: this is sized so the answer fits the output cap rather than so the
# question fits the context.
_WINDOW_CHARS = 9_000
_MAX_BODY_CHARS = 1_400
_MAX_OCR_CHARS = 2_500

_EXTRACT_PERSONA = """You are reading a transcript and the text that was on \
screen during a programming video, and writing down every problem the video \
actually solves so a student can revise from the list without rewatching.

Rules:
- Only problems the video genuinely works through. A concept being explained \
is not a problem. If this part of the video solves nothing, return an empty \
list — that is a correct answer, not a failure.
- statement must stand on its own. "Find the second highest salary from the \
Employee table" — not "the next question" or "this one".
- solution is the working code or query as the video arrives at it, cleaned up \
to run: complete, correctly indented, no commentary inside it, no ellipses.
- language is the one the video used. For databases name the dialect if the \
video makes it clear (MySQL, PostgreSQL, Oracle, SQL Server, SQLite); \
otherwise name the closest one.
- approach is one or two sentences on the idea that makes it work — the thing \
a student would want to recall before looking at the code.
- start_sec is in SECONDS. Copy it from the [t=N seconds] marker on the \
material the problem came from. Never a minute count.
- Keep the video's own order."""

_TRANSLATE_PERSONA = """You rewrite a working solution into other languages of \
the same family. The logic must stay the same; only the syntax and the \
idioms change.

Rules:
- Each translation must be complete and runnable on its own, not a diff and \
not a note about what to change.
- Use what is idiomatic in the target. In Oracle that means FETCH FIRST or \
ROWNUM rather than LIMIT; in SQL Server, TOP or OFFSET/FETCH; in Python, \
the standard library rather than an invented helper.
- Where a target genuinely cannot express the original the same way, write the \
closest working equivalent and say what changed in one short comment inside \
the code.
- Return one entry per language asked for, and no others."""

_PROBLEM_PROPERTIES: dict[str, Any] = {
    "title": {"type": "string", "description": "A short name, 3-8 words."},
    "statement": {"type": "string", "description": "The question, standing alone."},
    "approach": {"type": "string"},
    "language": {"type": "string"},
    "solution": {"type": "string", "description": "Complete, runnable code."},
    "start_sec": {
        "type": "number",
        "description": (
            "Seconds from the start of the video: copy the number from the "
            "[t=N seconds] marker. Never minutes."
        ),
    },
    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
    "topics": {
        "type": "array",
        "items": {"type": "string"},
        "description": "1-3 short tags, e.g. 'joins', 'window functions'.",
    },
}

_EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "family": {
            "type": "string",
            "enum": ["sql", "programming", "none"],
            "description": "'none' when this part solves no problems.",
        },
        "problems": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _PROBLEM_PROPERTIES,
                "required": ["title", "statement", "solution", "language", "start_sec"],
            },
        },
    },
    "required": ["family", "problems"],
}

_TRANSLATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "solutions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"},
                    "note": {
                        "type": "string",
                        "description": "Only when the target needed a real change.",
                    },
                },
                "required": ["language", "code"],
            },
        }
    },
    "required": ["solutions"],
}


def is_available() -> bool:
    """Whether problems can be extracted at all."""
    return llm.get_client().available


def canonical_language(name: str) -> Optional[str]:
    """The canonical name for a language, or None if it is not one we know."""
    key = re.sub(r"\s+", " ", str(name or "")).strip().lower()
    if not key:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    for langs in FAMILIES.values():
        for lang in langs:
            if lang.lower() == key:
                return lang
    return None


def family_of(language: str) -> Optional[str]:
    for family, langs in FAMILIES.items():
        if language in langs:
            return family
    return None


def _ocr_by_window(scenes_doc: Optional[dict[str, Any]]) -> list[tuple[float, str]]:
    """On-screen text with the time it appeared.

    In a SQL video the query is on the screen and not in the speech: without
    this the extractor has the question and no answer.
    """
    out: list[tuple[float, str]] = []
    for sc in (scenes_doc or {}).get("scenes") or []:
        text = (sc.get("ocr_text") or "").strip()
        if len(text) > 12:
            out.append((float(sc.get("start") or 0.0), text))
    return out


def _windows(u: dict[str, Any], scenes_doc: Optional[dict[str, Any]]) -> list[str]:
    """Cut the video's material into pieces one call can answer about.

    The cut follows the explanation's own sections so a problem is not split
    across two calls, and each piece carries the on-screen text from the same
    stretch of video.
    """
    sections = u.get("explanation") or []
    if not sections:
        return []
    ocr = _ocr_by_window(scenes_doc)

    pieces: list[str] = []
    current: list[str] = []
    size = 0
    bounds: list[float] = []

    def flush() -> None:
        nonlocal current, size, bounds
        if not current:
            return
        text = "\n\n".join(current)
        if bounds and ocr:
            lo, hi = bounds[0], bounds[-1]
            shown = [t for start, t in ocr if lo - 30 <= start <= hi + 180]
            if shown:
                text += ("\n\nON SCREEN DURING THIS PART\n"
                         + "\n---\n".join(shown)[:_MAX_OCR_CHARS])
        pieces.append(text)
        current, size, bounds = [], 0, []

    for s in sections:
        start = float(s.get("start_sec") or 0.0)
        body = (s.get("body") or "")[:_MAX_BODY_CHARS]
        block = f"[t={int(start)} seconds] {s.get('heading') or ''}\n{body}"
        if size and size + len(block) > _WINDOW_CHARS:
            flush()
        current.append(block)
        bounds.append(start)
        size += len(block)
    flush()

    if len(pieces) > config.PROBLEMS_MAX_WINDOWS:
        # Rather than drop the end of the video, take an even spread: a list
        # that covers all three hours thinly beats one that stops at ninety
        # minutes.
        cap = config.PROBLEMS_MAX_WINDOWS
        stride = len(pieces) / cap
        pieces = [pieces[int(i * stride)] for i in range(cap)]
    return pieces


def _clean_problem(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    statement = str(raw.get("statement") or "").strip()
    solution = str(raw.get("solution") or "").strip()
    if len(statement) < 12 or len(solution) < 8:
        return None
    language = canonical_language(raw.get("language") or "") or "Unknown"
    topics = [str(t).strip() for t in (raw.get("topics") or []) if str(t).strip()][:3]
    difficulty = raw.get("difficulty")
    return {
        "title": str(raw.get("title") or statement[:60]).strip(),
        "statement": statement,
        "approach": str(raw.get("approach") or "").strip(),
        "language": language,
        "solutions": {language: _strip_fence(solution)} if language != "Unknown" else {},
        "original_solution": _strip_fence(solution),
        "start_sec": float(raw.get("start_sec") or 0.0),
        "difficulty": difficulty if difficulty in {"easy", "medium", "hard"} else "medium",
        "topics": topics,
    }


def _strip_fence(code: str) -> str:
    """Models fence code even when told the field is already code."""
    text = code.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _key(statement: str) -> str:
    """What makes two statements the same question.

    Used both to drop a repeat within one reading and to recognise a problem
    again after a re-reading, which is what lets translations survive it.
    """
    return " ".join(re.findall(r"[a-z0-9]+", statement.lower()))[:120]


def _dedupe(problems: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the same question arriving twice.

    Windows overlap at their edges in what they discuss, and a video that
    revisits a question states it twice. Matching on the statement's words
    catches both without needing them worded identically.
    """
    seen: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for p in problems:
        key = _key(p["statement"])
        if key in seen:
            continue
        seen[key] = p
        out.append(p)
    out.sort(key=lambda p: p["start_sec"])
    for i, p in enumerate(out, 1):
        p["n"] = i
    return out


def generate(video_id: str, *, force: bool = False) -> dict[str, Any]:
    """Extract (and cache) every problem the video solves.

    Raises:
        llm.LLMUnavailable: no provider configured.
        llm.LLMError: every window failed, so there is nothing to show.
        FileNotFoundError: the video has not been through the understanding
            stage, so there is nothing to read.
    """
    if not force:
        cached = cache.read_json(video_id, "problems")
        if cached and cached.get("problems"):
            return cached

    u = cache.read_json(video_id, "understanding")
    if not u:
        raise FileNotFoundError(
            "this video has no understanding document — the problem list is "
            "read from the explanation, not from the raw transcript"
        )

    client = llm.get_client()
    if not client.available:
        raise llm.LLMUnavailable("no LLM provider configured")

    scenes_doc = cache.read_json(video_id, "scenes")
    pieces = _windows(u, scenes_doc)
    if not pieces:
        raise llm.LLMError("the understanding document has no walkthrough to read")

    started = time.perf_counter()
    collected: list[dict[str, Any]] = []
    families: list[str] = []
    failed = 0
    model = client.model

    for i, piece in enumerate(pieces, 1):
        cache.set_state(
            video_id, "problems", i / len(pieces),
            f"Reading part {i} of {len(pieces)} for solved problems...",
        )
        try:
            result = client.generate_json(
                _EXTRACT_PERSONA,
                "List every problem solved in this part of the video.\n\n" + piece,
                schema=_EXTRACT_SCHEMA,
                temperature=0.2,          # this is extraction, not writing
                max_output_tokens=config.UNDERSTANDING_MAX_OUTPUT_TOKENS,
            )
        except llm.LLMQuotaExceeded:
            logger.warning(
                "%s: quota exhausted after %d/%d parts; keeping %d problems",
                video_id, i - 1, len(pieces), len(collected),
            )
            break
        except llm.LLMError as exc:
            failed += 1
            logger.warning("  part %d/%d failed (%s); continuing", i, len(pieces), exc)
            continue

        model = result.model
        family = str(result.data.get("family") or "").lower()
        if family in FAMILIES:
            families.append(family)
        for raw in result.data.get("problems") or []:
            cleaned = _clean_problem(raw)
            if cleaned:
                collected.append(cleaned)

    if failed == len(pieces):
        raise llm.LLMError(f"every one of the {len(pieces)} parts failed; no problem list")

    problems = _dedupe(collected)
    _carry_over_translations(cache.read_json(video_id, "problems"), problems)
    family = max(set(families), key=families.count) if families else _guess_family(problems)
    primary = problems[0]["language"] if problems else ""
    doc = {
        "video_id": video_id,
        "model": model,
        "generated_in_sec": round(time.perf_counter() - started, 1),
        "family": family,
        "primary_language": primary,
        "targets": FAMILIES.get(family, []),
        "parts": len(pieces),
        "parts_failed": failed,
        "problems": problems,
    }
    cache.write_json(video_id, "problems", doc)
    cache.set_state(video_id, "problems", 1.0,
                    f"{len(problems)} problems extracted")
    logger.info(
        "problems for %s: %d found across %d parts (%d failed) in %.1fs",
        video_id, len(problems), len(pieces), failed, doc["generated_in_sec"],
    )
    return doc


def _carry_over_translations(
    previous: Optional[dict[str, Any]], fresh: list[dict[str, Any]]
) -> None:
    """Move already-paid-for translations onto a freshly read list.

    Re-reading is a different model run, so the list comes back slightly
    different each time. Without this, pressing "re-read" silently throws away
    every translation the reader has generated, which is both their work and
    their quota. A translation is only carried over when the solution it was
    made from is unchanged, since otherwise it would describe different code.
    """
    if not previous:
        return
    old = {_key(p.get("statement") or ""): p for p in previous.get("problems") or []}
    kept = 0
    for p in fresh:
        before = old.get(_key(p["statement"]))
        if not before:
            continue
        if (before.get("original_solution") or "") != (p.get("original_solution") or ""):
            continue
        for lang, code in (before.get("solutions") or {}).items():
            if lang not in p["solutions"]:
                p["solutions"][lang] = code
                kept += 1
        if before.get("notes"):
            p["notes"] = {**(before.get("notes") or {}), **(p.get("notes") or {})}
    if kept:
        logger.info("kept %d translations from the previous reading", kept)


def _guess_family(problems: list[dict[str, Any]]) -> str:
    for p in problems:
        fam = family_of(p.get("language") or "")
        if fam:
            return fam
    return "programming"


def translate(
    video_id: str,
    index: int,
    *,
    languages: Optional[list[str]] = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fill in one problem's solution in the other languages of its family.

    Translating a whole problem in one call rather than one language at a time
    is what makes this affordable: five dialects cost one request, not five.

    Returns the updated problem.
    """
    doc = cache.read_json(video_id, "problems")
    if not doc or not doc.get("problems"):
        raise FileNotFoundError("no problem list has been extracted for this video")
    problems = doc["problems"]
    if not 1 <= index <= len(problems):
        raise IndexError(f"problem {index} does not exist (1-{len(problems)})")
    problem = problems[index - 1]
    key = _key(problem.get("statement") or "")

    family = doc.get("family") or _guess_family(problems)
    targets = FAMILIES.get(family, [])
    wanted = [canonical_language(x) for x in (languages or targets)]
    wanted = [w for w in wanted if w and w in targets]
    if not force:
        wanted = [w for w in wanted if w not in (problem.get("solutions") or {})]
    if not wanted:
        return problem

    client = llm.get_client()
    if not client.available:
        raise llm.LLMUnavailable("no LLM provider configured")

    source_lang = problem.get("language") or "the original language"
    prompt = (
        f"PROBLEM\n{problem['statement']}\n\n"
        f"WORKING SOLUTION ({source_lang})\n{problem.get('original_solution') or ''}\n\n"
        f"Rewrite it in: {', '.join(wanted)}."
    )
    result = client.generate_json(
        _TRANSLATE_PERSONA, prompt,
        schema=_TRANSLATE_SCHEMA,
        temperature=0.2,
        max_output_tokens=config.UNDERSTANDING_MAX_OUTPUT_TOKENS,
    )

    fresh: dict[str, str] = {}
    notes: dict[str, str] = {}
    for entry in result.data.get("solutions") or []:
        lang = canonical_language(entry.get("language") or "")
        code = _strip_fence(str(entry.get("code") or ""))
        if not lang or lang not in targets or len(code) < 4:
            continue
        fresh[lang] = code
        note = str(entry.get("note") or "").strip()
        if note:
            notes[lang] = note

    # Re-read inside the lock: another translation may have finished while the
    # model was answering this one, and saving the copy read before it started
    # would undo it.
    with _write_lock:
        doc = cache.read_json(video_id, "problems") or doc
        problem = next(
            (x for x in doc["problems"] if _key(x.get("statement") or "") == key),
            problem,
        )
        problem["solutions"] = {**(problem.get("solutions") or {}), **fresh}
        if notes:
            problem["notes"] = {**(problem.get("notes") or {}), **notes}
        cache.write_json(video_id, "problems", doc)
    logger.info("problem %d of %s: now has %d languages",
                index, video_id, len(solutions))
    return problem


def get(video_id: str) -> Optional[dict[str, Any]]:
    """The cached problem list, if one has been extracted."""
    return cache.read_json(video_id, "problems")
