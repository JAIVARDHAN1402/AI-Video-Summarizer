"""M5 — Summarization: map-reduce BART summarization over fused transcript
blocks, plus chapter generation guided by CLIP visual-change signals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
from core import cache

logger = logging.getLogger(__name__)

_SUMMARIZER_PIPELINE = None
_TOKENIZER = None
_TITLE_GENERATOR = None


def _select_device_index() -> int:
    """transformers pipeline device convention: -1 CPU, 0 = first CUDA device."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _select_batch_size() -> int:
    return config.SUMMARIZER_BATCH_SIZE_GPU if _select_device_index() >= 0 else config.SUMMARIZER_BATCH_SIZE_CPU


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
    return _TOKENIZER


def _get_summarizer():
    global _SUMMARIZER_PIPELINE
    if _SUMMARIZER_PIPELINE is None:
        from transformers import pipeline

        logger.info("loading summarizer model=%s", config.SUMMARIZER_MODEL)
        _SUMMARIZER_PIPELINE = pipeline(
            "summarization", model=config.SUMMARIZER_MODEL,
            tokenizer=_get_tokenizer(), device=_select_device_index(),
        )
    return _SUMMARIZER_PIPELINE


def _get_title_generator():
    global _TITLE_GENERATOR
    if _TITLE_GENERATOR is None:
        from transformers import pipeline

        logger.info("loading chapter-title generator model=%s", config.QA_LOCAL_MODEL)
        _TITLE_GENERATOR = pipeline(
            "text2text-generation", model=config.QA_LOCAL_MODEL, device=_select_device_index(),
        )
    return _TITLE_GENERATOR


# --- Map stage: token-aware chunking --------------------------------------------


def _finalize_chunk(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": blocks[0]["start"],
        "end": blocks[-1]["end"],
        "text": " ".join(b["fused_text"] for b in blocks),
    }


def _carry_overlap(blocks: list[dict[str, Any]], token_len) -> list[dict[str, Any]]:
    carry: list[dict[str, Any]] = []
    carry_tokens = 0
    for b in reversed(blocks):
        t = token_len(b["fused_text"])
        if carry and carry_tokens + t > config.SUMMARIZER_CHUNK_OVERLAP_TOKENS:
            break
        carry.insert(0, b)
        carry_tokens += t
    return carry


def _chunk_blocks(blocks: list[dict[str, Any]], tokenizer) -> list[dict[str, Any]]:
    """Greedily pack fused blocks into <= SUMMARIZER_MAX_INPUT_TOKENS chunks,
    carrying SUMMARIZER_CHUNK_OVERLAP_TOKENS of trailing text into the next
    chunk so context isn't lost at chunk boundaries.
    """
    if not blocks:
        return []

    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_len(block["fused_text"])
        if current and current_tokens + block_tokens > config.SUMMARIZER_MAX_INPUT_TOKENS:
            chunks.append(_finalize_chunk(current))
            current = _carry_overlap(current, token_len)
            current_tokens = sum(token_len(b["fused_text"]) for b in current)
        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append(_finalize_chunk(current))

    return chunks


def _safe_summarize(text: str, max_len: int, min_len: int) -> str:
    """Summarize text, gracefully handling inputs too short for min_length."""
    text = text.strip()
    if not text:
        return ""

    tokenizer = _get_tokenizer()
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count < 5:
        return text

    effective_min = min(min_len, max(5, token_count // 2))
    summarizer = _get_summarizer()
    output = summarizer(text, max_length=max_len, min_length=effective_min, truncation=True)
    return output[0]["summary_text"].strip()


def _summarize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []

    summarizer = _get_summarizer()
    batch_size = _select_batch_size()
    texts = [c["text"] for c in chunks]

    outputs = summarizer(
        texts,
        max_length=config.CHUNK_SUMMARY_MAX_LEN,
        min_length=config.CHUNK_SUMMARY_MIN_LEN,
        truncation=True,
        batch_size=batch_size,
    )

    return [
        {"start": chunk["start"], "end": chunk["end"], "text": output["summary_text"].strip()}
        for chunk, output in zip(chunks, outputs)
    ]


# --- Reduce stage ----------------------------------------------------------------


def _reduce_to_overall_summary(
    chunk_summaries: list[dict[str, Any]], tokenizer, depth: int = 0
) -> str:
    combined_text = " ".join(c["text"] for c in chunk_summaries)
    token_count = len(tokenizer.encode(combined_text, add_special_tokens=False))

    if token_count > config.SUMMARIZER_MAX_INPUT_TOKENS and depth < config.REDUCE_MAX_RECURSION_DEPTH:
        pseudo_blocks = [
            {"start": c["start"], "end": c["end"], "fused_text": c["text"]} for c in chunk_summaries
        ]
        regrouped_chunks = _chunk_blocks(pseudo_blocks, tokenizer)
        next_level_summaries = _summarize_chunks(regrouped_chunks)
        return _reduce_to_overall_summary(next_level_summaries, tokenizer, depth + 1)

    return _safe_summarize(combined_text, config.FINAL_SUMMARY_MAX_LEN, config.FINAL_SUMMARY_MIN_LEN)


# --- Chapters ----------------------------------------------------------------------


def _split_points_from_clip_embeddings(
    n_scenes: int, embeddings: np.ndarray, n_splits: int
) -> list[int]:
    """Return scene indices where visual change (CLIP embedding distance) is greatest."""
    if n_scenes < 2 or n_splits <= 0:
        return []

    distances = [
        (1.0 - float(np.dot(embeddings[i - 1], embeddings[i])), i) for i in range(1, n_scenes)
    ]
    distances.sort(key=lambda pair: pair[0], reverse=True)
    return sorted(idx for _, idx in distances[:n_splits])


def _build_chapter_windows(
    scenes: list[dict[str, Any]], embeddings: Optional[np.ndarray], duration_sec: float
) -> list[tuple[float, float]]:
    target = max(1, min(config.TARGET_CHAPTER_COUNT, len(scenes) or 1))

    if scenes and embeddings is not None and len(embeddings) == len(scenes):
        split_indices = _split_points_from_clip_embeddings(len(scenes), embeddings, target - 1)
        boundaries = sorted({0.0, *(scenes[i]["start"] for i in split_indices), duration_sec})
    else:
        step = duration_sec / target
        boundaries = sorted({round(i * step, 3) for i in range(target)} | {duration_sec})

    windows = [
        (boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    return windows or [(0.0, duration_sec)]


def _blocks_in_window(
    blocks: list[dict[str, Any]], start: float, end: float, inclusive_end: bool
) -> list[dict[str, Any]]:
    result = []
    for b in blocks:
        midpoint = (b["start"] + b["end"]) / 2.0
        if start <= midpoint < end or (inclusive_end and midpoint == end):
            result.append(b)
    return result


def _generate_chapter_title(seed_text: str) -> str:
    generator = _get_title_generator()
    prompt = (
        f"Write a short title (at most {config.CHAPTER_TITLE_MAX_WORDS} words) "
        f"for this video segment.\nSegment: {seed_text[:1000]}\nTitle:"
    )
    output = generator(prompt, max_new_tokens=16, truncation=True)[0]["generated_text"]
    words = output.strip().split()[: config.CHAPTER_TITLE_MAX_WORDS]
    title = " ".join(words).strip().rstrip(".,;:!?-")
    return title.title() if title else "Untitled Segment"


def _build_chapters(
    windows: list[tuple[float, float]], fused_blocks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    chapters = []
    last_window = windows[-1] if windows else None

    for start, end in windows:
        inclusive_end = (start, end) == last_window
        blocks_in_window = _blocks_in_window(fused_blocks, start, end, inclusive_end)
        chapter_text = " ".join(b["text"] for b in blocks_in_window if b["text"]).strip()

        if chapter_text:
            summary_text = _safe_summarize(
                chapter_text, config.CHUNK_SUMMARY_MAX_LEN, config.CHUNK_SUMMARY_MIN_LEN
            )
            title = _generate_chapter_title(chapter_text)
        else:
            tags = sorted({tag for b in blocks_in_window for tag in b.get("visual_tags", [])})
            summary_text = f"Visual segment ({', '.join(tags) if tags else 'no speech detected'})."
            title = _generate_chapter_title(", ".join(tags)) if tags else "Visual Segment"

        chapters.append({"start": start, "end": end, "title": title, "summary": summary_text})

    return chapters


# --- Orchestration -----------------------------------------------------------------


def summarize(video_id: str, force: bool = False) -> dict[str, Any]:
    """Map-reduce summarize fused.json into summary.json (overall summary + chapters).

    Args:
        video_id: The cache key for this video.
        force: Re-run even if a cached summary.json already exists.

    Returns:
        The summary.json dict.
    """
    if not force and cache.artifact_exists(video_id, "summary"):
        logger.info("using cached summary.json for %s", video_id)
        return cache.read_json(video_id, "summary")  # type: ignore[return-value]

    fused_data = cache.read_json(video_id, "fused")
    scenes_data = cache.read_json(video_id, "scenes")
    meta = cache.read_json(video_id, "meta")
    if fused_data is None or meta is None:
        raise RuntimeError(f"fused.json and meta.json are required to summarize {video_id}")

    cache.set_state(video_id, "summarize", 0.0, "Summarizing transcript...")
    blocks = fused_data["blocks"]
    scenes = scenes_data["scenes"] if scenes_data else []
    duration_sec = float(meta["duration_sec"])

    has_speech = any(b["text"].strip() for b in blocks)

    embeddings: Optional[np.ndarray] = None
    if scenes_data and scenes_data.get("embeddings_file"):
        embeddings_path = Path(scenes_data["embeddings_file"])
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)

    if not has_speech:
        all_tags = sorted({tag for b in blocks for tag in b.get("visual_tags", [])})
        overall_summary = (
            "No speech was detected in this video. Based on visual analysis, it "
            f"contains scenes featuring: {', '.join(all_tags) if all_tags else 'no identifiable objects'}."
        )
        windows = _build_chapter_windows(scenes, embeddings, duration_sec)
        chapters = _build_chapters(windows, blocks)
        result = {
            "overall_summary": overall_summary,
            "chapters": chapters,
            "chunk_summaries": [],
            "warning": "no_speech_detected",
        }
        cache.write_json(video_id, "summary", result)
        cache.set_state(video_id, "summarize", 1.0, "Summarization complete.")
        return result

    tokenizer = _get_tokenizer()

    cache.set_state(video_id, "summarize", 0.1, "Chunking transcript for summarization...")
    chunks = _chunk_blocks(blocks, tokenizer)

    cache.set_state(video_id, "summarize", 0.3, "Summarizing chunks (map stage)...")
    chunk_summaries = _summarize_chunks(chunks)

    cache.set_state(video_id, "summarize", 0.6, "Reducing to overall summary...")
    overall_summary = _reduce_to_overall_summary(chunk_summaries, tokenizer) if chunk_summaries else ""

    cache.set_state(video_id, "summarize", 0.8, "Generating chapters...")
    windows = _build_chapter_windows(scenes, embeddings, duration_sec)
    chapters = _build_chapters(windows, blocks)

    result = {
        "overall_summary": overall_summary,
        "chapters": chapters,
        "chunk_summaries": chunk_summaries,
    }
    cache.write_json(video_id, "summary", result)
    cache.set_state(video_id, "summarize", 1.0, "Summarization complete.")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M5 summarization in isolation.")
    parser.add_argument("video_id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = summarize(args.video_id, force=args.force)
    print(json.dumps(result, indent=2))
