"""Questions answered across several videos at once.

One video at a time is the wrong unit for how people actually study: a topic
is spread over a playlist, a module, a term's worth of lectures. This module
treats a set of processed videos as one body of material — retrieving from
every member's index, ranking the passages together, and answering with
citations that say which video each came from.

Ranking across indexes is sound because every index is built by the same
embedding model and FAISS searches normalised vectors, so the inner products
are cosine similarities on one scale. Scores from different videos are
directly comparable; no per-video normalisation is needed.

What makes this worth more than asking each video separately is that the
answer can relate them: say which video covers something best, and say plainly
when two of them disagree.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import config
from core import cache, llm, qa

logger = logging.getLogger(__name__)

# Passages kept after merging every source's hits. Larger than the per-video
# top-k because the pool is larger and the point is to find the few videos
# that actually address the question.
TOP_K_ACROSS = int(config.RAG_TOP_K) * 3

# Understanding documents are long; only the best-matching videos contribute
# one, or a six-video notebook would spend its whole context on preambles.
MAX_UNDERSTANDING_SOURCES = 3

_PERSONA = """You are a tutor answering a question using several videos at once.

How to answer:
- Answer the question first, in plain language. Teach it; do not summarise \
what each source says in turn.
- Attribute by name, the way a person would: "the DFIR lecture covers this". \
Never write a video_id or a timestamp in the answer text — those go in the \
cited list, and the reader is shown them separately. An id in the prose reads \
as a database leak.
- If several videos cover the same ground, say which one covers it best and \
why, rather than repeating it.
- If they disagree, say so explicitly and give both positions. Do not \
silently pick one or average them.
- If only one video touches the question, say that too — a reader should know \
how well supported an answer is.
- You may go beyond the videos when they fall short, but set beyond_video and \
make clear in the text which part was added.
- Set off_topic when none of the material relates to the question."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited": {
            "type": "array",
            "description": "Moments actually used, as {video_id, sec} pairs.",
            "items": {
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "sec": {"type": "number"},
                },
                "required": ["video_id", "sec"],
            },
        },
        "beyond_video": {"type": "boolean"},
        "off_topic": {"type": "boolean"},
    },
    "required": ["answer", "cited", "beyond_video", "off_topic"],
}


def title_of(video_id: str) -> str:
    meta = cache.read_json(video_id, "meta") or {}
    return meta.get("title") or video_id


def retrieve_across(
    video_ids: list[str], question: str, k: int = TOP_K_ACROSS
) -> list[dict[str, Any]]:
    """Best passages from across the set, each tagged with where it came from.

    A video whose index is missing or unreadable is skipped rather than
    failing the question: a notebook with one broken member should still
    answer from the rest.
    """
    pool: list[dict[str, Any]] = []
    for vid in video_ids:
        try:
            chunks, scores = qa._retrieve(vid, question)  # noqa: SLF001 - internal by design
        except Exception:
            logger.warning("retrieval failed for %s; skipping it", vid, exc_info=True)
            continue
        for chunk, score in zip(chunks, scores):
            pool.append({**chunk, "video_id": vid, "score": float(score)})

    pool.sort(key=lambda c: -c["score"])
    return pool[:k]


def _context_blocks(
    video_ids: list[str], question: str, retrieved: list[dict[str, Any]]
) -> list[str]:
    blocks: list[str] = []

    # Which videos actually matched, best first — the written explanations are
    # pulled for those only.
    ranked: list[str] = []
    for chunk in retrieved:
        if chunk["video_id"] not in ranked:
            ranked.append(chunk["video_id"])

    titles = {vid: title_of(vid) for vid in video_ids}
    blocks.append(
        "=== THE VIDEOS IN THIS NOTEBOOK ===\n"
        + "\n".join(f"- {vid}: {titles[vid]}" for vid in video_ids)
    )

    for vid in ranked[:MAX_UNDERSTANDING_SOURCES]:
        written = qa._understanding_context(vid, question)  # noqa: SLF001
        if written:
            blocks.append(
                f"=== WRITTEN EXPLANATION OF \"{titles.get(vid, vid)}\" "
                f"(video_id {vid}) ===\n{written}"
            )

    # Passages stay grouped by video so the model can attribute without having
    # to infer which source a line belongs to.
    by_video: dict[str, list[dict[str, Any]]] = {}
    for chunk in retrieved:
        by_video.setdefault(chunk["video_id"], []).append(chunk)
    for vid, chunks in by_video.items():
        lines = [
            f"[{qa._format_mmss(c['start'])}] {c['text'].strip()}"  # noqa: SLF001
            for c in chunks
        ]
        blocks.append(
            f"=== TRANSCRIPT PASSAGES FROM \"{titles.get(vid, vid)}\" "
            f"(video_id {vid}) ===\n" + "\n\n".join(lines)
        )

    return blocks


def answer_across(
    video_ids: list[str],
    question: str,
    history: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Answer one question using every video in the set.

    Returns the same shape as :func:`core.qa.answer`, except that each source
    also carries ``video_id`` and ``video_title`` so the interface can say
    which video a citation points into.
    """
    video_ids = [v for v in video_ids if v]
    if not video_ids:
        return {
            "answer": "This notebook has no videos in it yet.",
            "sources": [],
            "confidence": "low",
            "off_topic": True,
        }
    if len(video_ids) == 1:
        # One source is just the single-video tutor, which already has a
        # better prompt for that case.
        result = qa.answer(video_ids[0], question, history)
        for s in result.get("sources", []):
            s["video_id"] = video_ids[0]
            s["video_title"] = title_of(video_ids[0])
        return result

    retrieved = retrieve_across(video_ids, question)

    client = llm.get_client()
    if not client.available:
        # No provider: answer from the single best-matching video rather than
        # pretending to reason across them, and say which one it was.
        if not retrieved:
            return {
                "answer": "This is not discussed in any of these videos.",
                "sources": [],
                "confidence": "low",
            }
        best = retrieved[0]["video_id"]
        result = qa.answer(best, question, history)
        for s in result.get("sources", []):
            s["video_id"] = best
            s["video_title"] = title_of(best)
        result["answer"] = (
            f"Answering from \"{title_of(best)}\" alone — reasoning across "
            f"several videos needs a language model.\n\n" + result["answer"]
        )
        return result

    blocks = _context_blocks(video_ids, question, retrieved)
    for turn in (history or [])[-config.QA_HISTORY_TURNS:]:
        blocks.append(f"EARLIER — Q: {turn.get('q','')}\nA: {turn.get('a','')}")
    blocks.append(f"=== QUESTION ===\n{question}")

    result = client.generate_json(
        _PERSONA,
        "\n\n".join(blocks),
        schema=_SCHEMA,
        temperature=0.3,
        max_output_tokens=config.QA_LLM_MAX_OUTPUT_TOKENS,
        thinking_budget=config.QA_LLM_THINKING_BUDGET,
    )
    data = result.data or {}
    text = (data.get("answer") or "").strip()
    if not text:
        raise llm.LLMError("empty answer")

    # Keep the passages the model said it used; if the citations do not line
    # up with anything retrieved, show the top passages rather than none.
    known = {vid for vid in video_ids}
    cited = [
        (str(c.get("video_id")), float(c.get("sec", -1)))
        for c in data.get("cited") or []
        if str(c.get("video_id")) in known
    ]
    used = [
        c
        for c in retrieved
        if any(vid == c["video_id"] and abs(c["start"] - sec) < 30 for vid, sec in cited)
    ] or retrieved[: config.RAG_TOP_K]

    sources = [
        {
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
            "video_id": c["video_id"],
            "video_title": title_of(c["video_id"]),
        }
        for c in used
    ]

    # A citation can point at the written explanation rather than a retrieved
    # passage — the understanding document carries timestamps too. Dropping
    # those silently loses exactly the cross-video links worth keeping, so
    # they survive as a moment with no quoted text.
    for vid, sec in cited:
        if sec < 0 or any(s["video_id"] == vid and abs(s["start"] - sec) < 30 for s in sources):
            continue
        sources.append(
            {
                "start": sec,
                "end": sec,
                "text": "",
                "video_id": vid,
                "video_title": title_of(vid),
            }
        )
    sources.sort(key=lambda s: (s["video_title"], s["start"]))

    return {
        "answer": text,
        "sources": sources,
        "confidence": "low" if data.get("off_topic") else "high",
        "beyond_video": bool(data.get("beyond_video")),
        "off_topic": bool(data.get("off_topic")),
        "videos_used": sorted({s["video_id"] for s in sources}),
    }
