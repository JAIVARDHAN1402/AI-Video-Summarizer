"""M6 — RAG Q&A: FAISS retrieval over transcript chunks + a generation
backend (local flan-t5, or Gemini when GEMINI_API_KEY is set) that answers
natural-language questions with timestamp-cited sources.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

import config
from core import cache

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = None
_FLAN_PIPELINE = None
_TIKTOKEN_ENCODING = None

_LEADING_FILLER_RE = re.compile(
    r"^(?:so|and|but|then|well|okay|ok|now)\b[\s,]*", re.I
)

_LOW_CONFIDENCE_HEDGE = (
    "This may not be covered in the video, but based on what was said: "
)


def _format_mmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def _select_device_index() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _get_tiktoken_encoding():
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        import tiktoken

        _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODING


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model=%s", config.EMBEDDING_MODEL)
        _EMBEDDING_MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _EMBEDDING_MODEL


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


# --- Indexing ----------------------------------------------------------------------


def _finalize_rag_chunk(segments: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    return {
        "id": idx,
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(s["text"] for s in segments),
    }


def _carry_rag_overlap(segments: list[dict[str, Any]], token_len) -> list[dict[str, Any]]:
    carry: list[dict[str, Any]] = []
    carry_tokens = 0
    for s in reversed(segments):
        t = token_len(s["text"])
        if carry and carry_tokens + t > config.RAG_CHUNK_OVERLAP_TOKENS:
            break
        carry.insert(0, s)
        carry_tokens += t
    return carry


def _chunk_transcript(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack transcript segments into RAG_CHUNK_TOKENS windows with overlap."""
    if not segments:
        return []

    encoding = _get_tiktoken_encoding()

    def token_len(text: str) -> int:
        return len(encoding.encode(text))

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = token_len(seg["text"])
        if current and current_tokens + seg_tokens > config.RAG_CHUNK_TOKENS:
            chunks.append(_finalize_rag_chunk(current, len(chunks)))
            current = _carry_rag_overlap(current, token_len)
            current_tokens = sum(token_len(s["text"]) for s in current)
        current.append(seg)
        current_tokens += seg_tokens

    if current:
        chunks.append(_finalize_rag_chunk(current, len(chunks)))

    return chunks


def build_index(video_id: str, force: bool = False) -> None:
    """Chunk the transcript, embed it, and persist a FAISS index under qa_index/.

    Args:
        video_id: The cache key for this video.
        force: Rebuild even if a cached index already exists.
    """
    import faiss

    cache_dir = cache.get_cache_dir(video_id)
    qa_dir = cache_dir / "qa_index"
    index_path = qa_dir / "index.faiss"
    chunks_path = qa_dir / "chunks.json"

    if not force and index_path.exists() and chunks_path.exists():
        logger.info("using cached qa_index for %s", video_id)
        return

    transcript = cache.read_json(video_id, "transcript")
    if transcript is None:
        raise RuntimeError(f"transcript.json is required to build a Q&A index for {video_id}")

    cache.set_state(video_id, "index", 0.0, "Building Q&A index...")
    qa_dir.mkdir(parents=True, exist_ok=True)

    chunks = _chunk_transcript(transcript["segments"])

    if not chunks:
        dim = _get_embedding_model().get_sentence_embedding_dimension()
        faiss.write_index(faiss.IndexFlatIP(dim), str(index_path))
        chunks_path.write_text(json.dumps([]), encoding="utf-8")
        cache.set_state(video_id, "index", 1.0, "Q&A index complete (no transcript text).")
        return

    cache.set_state(video_id, "index", 0.4, "Embedding transcript chunks...")
    embeddings = _embed_texts([c["text"] for c in chunks])

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    chunks_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    cache.set_state(video_id, "index", 1.0, "Q&A index complete.")


# --- Retrieval -----------------------------------------------------------------------


def _load_index(video_id: str):
    import faiss

    cache_dir = cache.get_cache_dir(video_id)
    qa_dir = cache_dir / "qa_index"
    index_path = qa_dir / "index.faiss"
    chunks_path = qa_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise RuntimeError(f"No Q&A index for {video_id}; call build_index() first.")

    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    return index, chunks


def _retrieve(video_id: str, question: str) -> tuple[list[dict[str, Any]], list[float]]:
    index, chunks = _load_index(video_id)
    if index.ntotal == 0 or not chunks:
        return [], []

    query_embedding = _embed_texts([question])
    k = min(config.RAG_TOP_K, index.ntotal)
    scores, indices = index.search(query_embedding, k)

    retrieved = [chunks[i] for i in indices[0] if i != -1]
    retrieved_scores = [float(s) for s, i in zip(scores[0], indices[0]) if i != -1]
    return retrieved, retrieved_scores


# --- Generation ----------------------------------------------------------------------


def _build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Assemble the generation prompt from the retrieved passages.

    Two choices here are deliberate, and both were wrong before:

    * **Fewer passages than we retrieve.** flan-t5-base has a 512-token
      encoder; feeding it every retrieved chunk meant most of the context was
      silently truncated away and the model answered from whichever fragment
      happened to survive. Generation now sees only the best few passages,
      while citations still list everything retrieved, so no provenance is
      lost.
    * **No timestamp labels in the context.** They are metadata, not
      narration, and a small model reads them as content — the bracketed
      digits were being copied straight into answers.
    """
    encoding = _get_tiktoken_encoding()
    budget = config.QA_CONTEXT_MAX_TOKENS
    parts: list[str] = []
    for chunk in chunks[: config.QA_GENERATION_CHUNKS]:
        text = chunk["text"].strip()
        tokens = encoding.encode(text)
        if len(tokens) > budget:
            text = encoding.decode(tokens[:budget])
        parts.append(text)
        budget -= min(len(tokens), budget)
        if budget <= 0:
            break

    context = chr(10).join(parts)
    return (
        "Read the context and answer the question in one or two complete "
        "sentences. If the context does not contain the answer, reply exactly: "
        "This is not discussed in the video."
        + chr(10) + chr(10) + "Context:" + chr(10) + context
        + chr(10) + chr(10) + f"Question: {question}" + chr(10) + "Answer:"
    )

def _generate_gemini(prompt: str) -> str:
    import requests

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def _get_flan_pipeline():
    global _FLAN_PIPELINE
    if _FLAN_PIPELINE is None:
        from transformers import pipeline

        logger.info("loading local Q&A model=%s", config.QA_LOCAL_MODEL)
        _FLAN_PIPELINE = pipeline(
            "text2text-generation", model=config.QA_LOCAL_MODEL, device=_select_device_index()
        )
    return _FLAN_PIPELINE


def _generate_local(prompt: str) -> str:
    """Generate with the local model.

    ``min_new_tokens`` and beam search are what stop flan-t5-base returning
    four-word sentence fragments; ``no_repeat_ngram_size`` is what stops the
    same settings degenerating into "Borne-Borne-Borne..." when the question
    has no answer in the context and the model is nonetheless obliged to
    produce a minimum number of tokens.
    """
    generator = _get_flan_pipeline()
    output = generator(
        prompt,
        max_new_tokens=config.QA_MAX_NEW_TOKENS,
        min_new_tokens=config.QA_MIN_NEW_TOKENS,
        num_beams=config.QA_NUM_BEAMS,
        no_repeat_ngram_size=config.QA_NO_REPEAT_NGRAM,
        truncation=True,
    )
    return output[0]["generated_text"].strip()


_REAL_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _is_degenerate(text: str) -> bool:
    """True when generation collapsed instead of answering.

    Two failure modes, both of which reached users:

    * **Repetition** — forced past a minimum length with nothing to say, the
      model loops a phrase ("Borne-Borne-Borne...").
    * **Enumeration** — it emits list markers or a run of single letters
      ("(iii). (iv.) a b c d e f g..."). Every token there is unique, so a
      repetition check alone passes it; what gives it away is that almost
      none of the tokens are real words.
    """
    words = text.split()
    if not words:
        return True
    if len(_REAL_WORD_RE.findall(text)) < config.QA_MIN_REAL_WORDS:
        return True
    if len(words) < config.QA_MIN_NEW_TOKENS:
        return False
    return len(set(w.lower() for w in words)) / len(words) < config.QA_MIN_UNIQUE_WORD_RATIO

def _tidy_answer(text: str) -> str:
    """Light cleanup of an extractive answer: trim filler, fix the opening case."""
    text = _LEADING_FILLER_RE.sub("", text.strip()).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]

def _generate(prompt: str) -> str:
    """Single interchangeable entry point for both generation backends."""
    if config.USE_GEMINI:
        try:
            return _generate_gemini(prompt)
        except Exception as exc:
            logger.warning("Gemini generation failed (%s); falling back to local model", exc)
    return _generate_local(prompt)


# --- Public API ------------------------------------------------------------------------


def answer(video_id: str, question: str) -> dict[str, Any]:
    """Answer a natural-language question about the video, with timestamp citations.

    Args:
        video_id: The cache key for this video (build_index() must have run).
        question: The user's natural-language question.

    Returns:
        {"answer": str, "sources": [{"start","end","text"}, ...], "confidence": "high"|"low"}
    """
    retrieved, scores = _retrieve(video_id, question)
    not_discussed = {
        "answer": "This is not discussed in the video.",
        "sources": [],
        "confidence": "low",
    }
    if not retrieved:
        return not_discussed

    # When retrieval finds nothing related, do not generate at all. Asked for
    # an answer that is not in its context, a small instruction model invents
    # a fluent one — "what is the capital of France?" against a coding lecture
    # produced a confident string of invented place names. Refusing to
    # generate is both more honest and the only behaviour consistent with a
    # product whose answers are supposed to be grounded in cited passages.
    top_score = scores[0]
    if top_score < config.RAG_LOW_CONFIDENCE_THRESHOLD:
        return not_discussed

    generated = _generate(_build_prompt(question, retrieved)).strip()
    if not generated or _is_degenerate(generated):
        return not_discussed

    sources = [{"start": c["start"], "end": c["end"], "text": c["text"]} for c in retrieved]
    return {
        "answer": _tidy_answer(generated),
        "sources": sources,
        # Above the floor but still weakly matched: answer, but let the UI
        # show the reader that the grounding is thin.
        "confidence": "high" if top_score >= config.RAG_HIGH_CONFIDENCE_THRESHOLD else "low",
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M6 RAG Q&A in isolation.")
    parser.add_argument("video_id")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ask", help="Question to ask")
    args = parser.parse_args()

    if args.build_index or args.ask:
        build_index(args.video_id, force=args.force)
    if args.ask:
        print(json.dumps(answer(args.video_id, args.ask), indent=2))
