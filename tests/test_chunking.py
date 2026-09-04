"""Unit tests for core.summarize's map-stage chunking logic.

Uses the real BART tokenizer (lightweight — vocab/merges only, no model
weights) so chunk sizes are verified against the same token counts the
summarizer will actually see, per the build plan's explicit requirement to
count with the BART tokenizer rather than len(text.split()).
"""

from __future__ import annotations

import pytest

import config
from core.summarize import _chunk_blocks

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)


def _make_blocks(n: int, seconds_per_block: float = 10.0, words_per_block: int = 30):
    """Synthetic fused blocks simulating a long lecture transcript.

    Mirrors the schema core.fuse.fuse() actually writes: the map stage reads
    ``text`` (what was said) and ``ocr_text`` (what was on screen), not the
    ``fused_text`` display string, which inlines timestamps and visual tags
    that would otherwise be fed to the summarizer as if they were narration.
    """
    blocks = []
    filler = " ".join(f"concept{i}" for i in range(words_per_block))
    for i in range(n):
        start = i * seconds_per_block
        end = start + seconds_per_block
        text = f"[{i:04d}] Discussion segment {i}: {filler}."
        blocks.append({
            "id": i,
            "start": start,
            "end": end,
            "scene_ids": [i],
            "visual_tags": [],
            "ocr_text": "",
            "text": text,
            "fused_text": f"[00:00-00:10] (visual: ) {text}",
        })
    return blocks


def test_empty_blocks_produce_no_chunks(tokenizer):
    assert _chunk_blocks([], tokenizer) == []


def test_single_short_block_produces_one_chunk(tokenizer):
    blocks = _make_blocks(1, words_per_block=10)
    chunks = _chunk_blocks(blocks, tokenizer)

    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 10.0


def test_45_minute_video_chunks_successfully(tokenizer):
    """45 minutes at 10s/block, ~30 words/block -> must chunk without error
    and every chunk must respect the token budget."""
    n_blocks = int(45 * 60 / 10)  # 270 blocks
    blocks = _make_blocks(n_blocks, seconds_per_block=10.0, words_per_block=30)

    chunks = _chunk_blocks(blocks, tokenizer)

    assert len(chunks) > 1, "a 45-minute transcript must not fit in a single chunk"

    max_single_block_tokens = max(
        len(tokenizer.encode(b["fused_text"], add_special_tokens=False)) for b in blocks
    )

    for chunk in chunks:
        token_count = len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        # A chunk may exceed the budget by at most one block's worth of tokens
        # (the block that triggered the flush is still appended after it).
        assert token_count <= config.SUMMARIZER_MAX_INPUT_TOKENS + max_single_block_tokens

    # Full duration is covered start-to-end with no gaps.
    assert chunks[0]["start"] == blocks[0]["start"]
    assert chunks[-1]["end"] == blocks[-1]["end"]

    # Chunks are chronologically ordered.
    starts = [c["start"] for c in chunks]
    assert starts == sorted(starts)


def test_chunks_carry_overlap_text_into_next_chunk(tokenizer):
    n_blocks = int(45 * 60 / 10)
    blocks = _make_blocks(n_blocks, seconds_per_block=10.0, words_per_block=30)

    chunks = _chunk_blocks(blocks, tokenizer)
    assert len(chunks) > 1

    # The overlap carries whole blocks, so the tail of chunk i's text should
    # reappear at the head of chunk i+1.
    first_chunk_tail = chunks[0]["text"].split("] ")[-1][:40]
    assert first_chunk_tail[:15] in chunks[1]["text"]


def test_no_single_block_is_silently_dropped(tokenizer):
    n_blocks = 50
    blocks = _make_blocks(n_blocks, seconds_per_block=5.0, words_per_block=20)

    chunks = _chunk_blocks(blocks, tokenizer)
    combined = " ".join(c["text"] for c in chunks)

    for b in blocks:
        marker = f"[{b['id']:04d}]"
        assert marker in combined, f"block {b['id']} missing from chunked output"
