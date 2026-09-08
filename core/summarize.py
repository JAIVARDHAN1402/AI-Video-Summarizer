"""M5 — Summarization: map-reduce abstractive summarization over fused
transcript blocks, plus chapter generation guided by CLIP visual-change signals.

Two structural choices keep this stage inside its time budget:

* **Chapters reuse the map stage's output.** They used to re-summarize the raw
  transcript a second time, which meant roughly twice the generation work for
  text the model had already read. Chapters are now assembled from the chunk
  summaries that already exist, and only fall back to generating when a chapter
  window contains no chunk at all.
* **Titles are extracted, not generated.** A whole second model (flan-t5-base,
  ~1 GB) was being loaded and run once per chapter to produce eight words.
  A keyphrase extractor over the chapter's own summary is effectively free and
  reads about as well; the LLM path is still available behind a config flag.

The map stage also batches on CPU, which the previous batch size of 1 disabled
entirely — chunks are independent and near-equal in length, so padding waste is
small and the matmuls make far better use of the cores.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
from core import cache, perf

logger = logging.getLogger(__name__)

_SUMMARIZER_PIPELINE = None
_SUMMARIZER_MODEL_NAME: Optional[str] = None
_TOKENIZER = None
_TOKENIZER_MODEL_NAME: Optional[str] = None
_TITLE_GENERATOR = None


def _select_device_index() -> int:
    """transformers pipeline device convention: -1 CPU, 0 = first CUDA device."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1


def _get_tokenizer(model_name: Optional[str] = None):
    global _TOKENIZER, _TOKENIZER_MODEL_NAME
    model_name = model_name or config.SUMMARIZER_MODEL
    if _TOKENIZER is None or _TOKENIZER_MODEL_NAME != model_name:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_MODEL_NAME = model_name
    return _TOKENIZER


def _get_summarizer(model_name: Optional[str] = None):
    global _SUMMARIZER_PIPELINE, _SUMMARIZER_MODEL_NAME
    model_name = model_name or config.SUMMARIZER_MODEL
    if _SUMMARIZER_PIPELINE is None or _SUMMARIZER_MODEL_NAME != model_name:
        from transformers import pipeline

        logger.info("loading summarizer model=%s", model_name)
        _SUMMARIZER_PIPELINE = pipeline(
            "summarization", model=model_name,
            tokenizer=_get_tokenizer(model_name), device=_select_device_index(),
        )
        _SUMMARIZER_MODEL_NAME = model_name
    return _SUMMARIZER_PIPELINE


def _get_instruct_generator():
    """flan-t5-base pipeline, used for chapter titles only when
    ``config.CHAPTER_TITLES_USE_LLM`` is on."""
    global _TITLE_GENERATOR
    if _TITLE_GENERATOR is None:
        from transformers import pipeline

        logger.info("loading instruction model=%s", config.QA_LOCAL_MODEL)
        _TITLE_GENERATOR = pipeline(
            "text2text-generation", model=config.QA_LOCAL_MODEL, device=_select_device_index(),
        )
    return _TITLE_GENERATOR


_QUOTE_CHARS_RE = re.compile(r'["“”‘’]')
_ADJACENT_QUOTES_RE = re.compile(r'["”]\s+["“]')
_DOUBLE_PERIOD_RE = re.compile(r'\.\s*\.')
_MULTI_SPACE_RE = re.compile(r'\s+')


def _strip_quote_artifacts(text: str) -> str:
    """Remove the news-style quotation marks BART inherits from CNN/DailyMail.

    Two quoted clauses sitting side by side (`"A" "B"`) get a sentence boundary
    where the quotes used to separate them, so the result still reads as
    connected prose rather than a run-on.
    """
    text = _ADJACENT_QUOTES_RE.sub('. ', text)
    text = _QUOTE_CHARS_RE.sub('', text)
    text = _DOUBLE_PERIOD_RE.sub('.', text)
    return _MULTI_SPACE_RE.sub(' ', text).strip()


def _rewrite_as_notes(text: str) -> str:
    """Clean up the summarizer's output into plain, connected notes.

    BART-family checkpoints are trained on news summarization, where quoting
    the source verbatim is normal style. Fed a spoken transcript instead of a
    news article, they keep that habit — wrapping lines in quotes and stringing
    them together rather than paraphrasing.

    An earlier version asked flan-t5-base to rewrite the text via an
    instruction prompt. That was dropped: on short inputs it sometimes
    hallucinated content wholesale (one summary about "having been given a lot"
    came back as "she's been given a lot of money" — nothing about money was
    ever said). A model rewrite isn't worth a factual-accuracy risk, so this
    does deterministic, meaning-preserving cleanup only.
    """
    text = text.strip()
    return _strip_quote_artifacts(text) if text else text


# --- Map stage: token-aware chunking --------------------------------------------


_OCR_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _ocr_for_summary(raw: str) -> str:
    """Filter on-screen text down to what is worth feeding the summarizer.

    Keyframe OCR picks up whatever is on screen, which on a coding site or a
    video player is mostly interface chrome — vote counts, button labels,
    "Add to List Share". BART copies its input closely, so that chrome was
    turning up verbatim inside otherwise-clean summaries.

    Slide and whiteboard content is prose; chrome is short tokens and digits.
    Text that does not clear a minimum of real words is dropped from the
    summarizer input. It is still kept on the scene itself, where the Scenes
    tab and transcript search can use it.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    words = _OCR_WORD_RE.findall(text)
    if len(words) < config.OCR_MIN_WORDS_FOR_SUMMARY:
        return ""
    return " ".join(text.split()[: config.OCR_MAX_WORDS_FOR_SUMMARY])


def _narrative_text(block: dict[str, Any]) -> str:
    """Natural-language text for a block: what was said, plus what was on
    screen, with no timestamp brackets or "(visual: ...)" tag-soup.

    ``fused_text`` (used elsewhere for debug/display) inlines timestamps and
    tag annotations, which reads fine to a human but pollutes the model input
    — it ends up parsing bracketed metadata as narration, which is what
    produced disjointed, quote-fragment output instead of connected prose.
    """
    parts = [block["text"]] if block.get("text") else []
    on_screen = _ocr_for_summary(block.get("ocr_text", ""))
    if on_screen:
        parts.append(f"(On screen: {on_screen}.)")
    return " ".join(parts).strip()

def _finalize_chunk(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": blocks[0]["start"],
        "end": blocks[-1]["end"],
        "text": " ".join(t for t in (_narrative_text(b) for b in blocks) if t),
    }


def _carry_overlap(blocks: list[dict[str, Any]], token_len) -> list[dict[str, Any]]:
    carry: list[dict[str, Any]] = []
    carry_tokens = 0
    for b in reversed(blocks):
        t = token_len(_narrative_text(b))
        if carry and carry_tokens + t > config.SUMMARIZER_CHUNK_OVERLAP_TOKENS:
            break
        carry.insert(0, b)
        carry_tokens += t
    return carry


def _chunk_blocks(
    blocks: list[dict[str, Any]], tokenizer, max_input_tokens: Optional[int] = None
) -> list[dict[str, Any]]:
    """Greedily pack fused blocks into token-bounded chunks, carrying
    ``SUMMARIZER_CHUNK_OVERLAP_TOKENS`` of trailing text into the next chunk so
    context isn't lost at chunk boundaries.

    ``max_input_tokens`` is raised above the default by the caller for long
    videos: fewer, larger chunks is how the map stage's cost is kept bounded
    rather than growing linearly with duration.
    """
    if not blocks:
        return []

    limit = max_input_tokens or config.SUMMARIZER_MAX_INPUT_TOKENS

    def token_len(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunks: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = token_len(_narrative_text(block))
        if current and current_tokens + block_tokens > limit:
            chunks.append(_finalize_chunk(current))
            current = _carry_overlap(current, token_len)
            current_tokens = sum(token_len(_narrative_text(b)) for b in current)
        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append(_finalize_chunk(current))

    return chunks


def _chunk_transcript(
    blocks: list[dict[str, Any]], tokenizer, target_sections: int
) -> list[dict[str, Any]]:
    """Pack the transcript into roughly ``target_sections`` chunks.

    The whole transcript is always included — this only chooses how wide each
    window is. At the default window a long video produces more chunks than
    the target; the window is then widened (up to the model's positional
    ceiling) so the section count lands near the target without dropping a
    single block. There is no longer an extractive selection step: covering
    the entire video is the point of the walkthrough.
    """
    chunks = _chunk_blocks(blocks, tokenizer, config.SUMMARIZER_MAX_INPUT_TOKENS)
    if len(chunks) <= target_sections:
        return chunks

    window = config.SUMMARIZER_MAX_INPUT_TOKENS
    while len(chunks) > target_sections and window < config.SUMMARIZER_MAX_INPUT_TOKENS_CEILING:
        window = min(int(window * 1.4), config.SUMMARIZER_MAX_INPUT_TOKENS_CEILING)
        chunks = _chunk_blocks(blocks, tokenizer, window)

    logger.info(
        "packed transcript into %d sections at a %d-token window (target %d)",
        len(chunks), window, target_sections,
    )
    return chunks

def _truncate_to_token_limit(text: str, tokenizer, max_tokens: int) -> str:
    """Hard-truncate text to at most ``max_tokens`` tokens.

    bart-large-cnn's tokenizer has no ``model_max_length`` configured (it
    reports the "unset" sentinel of ~1e30), so passing ``truncation=True`` to
    the pipeline silently does nothing — the input sails past the model's real
    1024-position limit and crashes deep in BART's positional embedding lookup
    (``IndexError: index out of range in self``). Truncating before the
    pipeline call is the only reliable guard.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _safe_summarize(text: str, max_len: int, min_len: int, model_name: str) -> str:
    """Summarize one text, gracefully handling inputs too short for min_length."""
    text = text.strip()
    if not text:
        return ""

    tokenizer = _get_tokenizer(model_name)
    text = _truncate_to_token_limit(text, tokenizer, config.SUMMARIZER_MAX_INPUT_TOKENS)
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count < 5:
        return text

    effective_min = min(min_len, max(5, token_count // 2))
    output = _get_summarizer(model_name)(
        text, max_length=max_len, min_length=effective_min, truncation=True
    )
    return _rewrite_as_notes(output[0]["summary_text"].strip())


def _summarize_chunks(
    chunks: list[dict[str, Any]], model_name: str, batch_size: int
) -> list[dict[str, Any]]:
    if not chunks:
        return []

    tokenizer = _get_tokenizer(model_name)
    texts = [
        _truncate_to_token_limit(c["text"], tokenizer, config.SUMMARIZER_MAX_INPUT_TOKENS)
        for c in chunks
    ]

    outputs = _get_summarizer(model_name)(
        texts,
        max_length=config.CHUNK_SUMMARY_MAX_LEN,
        min_length=config.CHUNK_SUMMARY_MIN_LEN,
        truncation=True,
        batch_size=batch_size,
    )

    return [
        {
            "start": chunk["start"],
            "end": chunk["end"],
            "text": _rewrite_as_notes(output["summary_text"].strip()),
        }
        for chunk, output in zip(chunks, outputs)
    ]


# --- Reduce stage ----------------------------------------------------------------


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _leading_sentences(text: str, count: int) -> str:
    """The first ``count`` sentences of ``text`` — its topic sentences."""
    parts = _SENTENCE_RE.split(text.strip())
    return " ".join(parts[:count]).strip()


def _reduce_to_overall_summary(
    section_summaries: list[dict[str, Any]], tokenizer, model_name: str,
) -> str:
    """Condense the section summaries into a brief high-level recap.

    Fed the full section texts, this often exceeds the model's window and has
    to recurse, and each recursion pass loses coverage of the later parts of
    the video — the recap ends up restating the opening. So when the combined
    text is too long, it is first reduced to the *topic sentences* of every
    section: that keeps the whole arc of the video represented, and is what a
    spoken recap actually does.
    """
    combined = " ".join(c["text"] for c in section_summaries)
    token_count = len(tokenizer.encode(combined, add_special_tokens=False))

    if token_count > config.SUMMARIZER_MAX_INPUT_TOKENS:
        for sentences in (2, 1):
            trimmed = " ".join(
                _leading_sentences(c["text"], sentences) for c in section_summaries
            )
            if len(tokenizer.encode(trimmed, add_special_tokens=False)) <= config.SUMMARIZER_MAX_INPUT_TOKENS:
                combined = trimmed
                break
        else:
            # Even one sentence each is too much (a very long video): fall back
            # to an even stride so coverage stays spread across the runtime.
            stride = max(1, token_count // config.SUMMARIZER_MAX_INPUT_TOKENS + 1)
            combined = " ".join(
                _leading_sentences(c["text"], 1) for c in section_summaries[::stride]
            )

    return _safe_summarize(
        combined, config.FINAL_SUMMARY_MAX_LEN, config.FINAL_SUMMARY_MIN_LEN, model_name
    )

# --- Chapters ----------------------------------------------------------------------


def _split_points_from_clip_embeddings(
    n_scenes: int, embeddings: np.ndarray, n_splits: int
) -> list[int]:
    """Scene indices where visual change (CLIP embedding distance) is greatest."""
    if n_scenes < 2 or n_splits <= 0:
        return []

    distances = [
        (1.0 - float(np.dot(embeddings[i - 1], embeddings[i])), i) for i in range(1, n_scenes)
    ]
    distances.sort(key=lambda pair: pair[0], reverse=True)
    return sorted(idx for _, idx in distances[:n_splits])


def _build_chapter_windows(
    scenes: list[dict[str, Any]], embeddings: Optional[np.ndarray], duration_sec: float,
    available_summaries: int = 0,
) -> list[tuple[float, float]]:
    """Chapter boundaries, placed where the visual content changes most.

    Capped by ``available_summaries`` as well as by scene count: chapters are
    filled from the map stage's chunk summaries, so asking for more chapters
    than there are summaries just makes consecutive chapters repeat the same
    text back to the reader.
    """
    limits = [config.TARGET_CHAPTER_COUNT, len(scenes) or 1]
    if available_summaries:
        limits.append(available_summaries)
    target = max(1, min(limits))

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


def _items_in_window(
    items: list[dict[str, Any]], start: float, end: float, inclusive_end: bool
) -> list[dict[str, Any]]:
    """Items whose midpoint lands in [start, end) — closed at the end for the
    last window so nothing near the final timestamp is dropped."""
    result = []
    for item in items:
        midpoint = (item["start"] + item["end"]) / 2.0
        if start <= midpoint < end or (inclusive_end and midpoint == end):
            result.append(item)
    return result


# Words that carry no topical signal in a title. Deliberately a small, fixed
# list rather than a full stopword corpus — it only has to survive being the
# first word of an extracted keyphrase.
_TITLE_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its it's he she they we you i him
her them us me my your our their what which who whom when where why how all any
both each few more most other some such no nor not only own same so too very
can will just don should now about into over after before again once here there
also very said says say get got go going one two three like really actually
""".split())

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_TITLE_STEM_RE = re.compile(r"[^a-z]")
# Quotation marks are never wanted in a title; the model adds them constantly.
_TITLE_QUOTES_RE = re.compile(r"[\"'‘’“”]")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# The word boundary is essential here: without it the "so" alternative would
# also match inside "Sometimes", truncating it to "metimes".
_LEADING_FILLER_RE = re.compile(
    r"^(?:so|now|okay|ok|right|and|but|then|well|alright|see|here)\b[\s,]*", re.I
)


def _clause_title(text: str, fallback: str = "Untitled Segment") -> str:
    """Title taken from the opening clause of the segment's own summary.

    Reads far better than a ranked bag of keywords, which is what a pure
    frequency extractor produces — "Copper Ornament Heaviest Costlier Bronze"
    is a word list, not a title. Leading spoken filler ("So, ...", "Okay, ...")
    is stripped because transcripts start with it constantly.
    """
    first = _SENTENCE_SPLIT_RE.split((text or "").strip(), maxsplit=1)[0]
    first = _LEADING_FILLER_RE.sub("", first).strip(" ,;:-—")
    if not first:
        return _keyphrase_title(text, fallback)

    words = first.split()
    if len(words) > config.CHAPTER_TITLE_MAX_WORDS:
        words = words[: config.CHAPTER_TITLE_MAX_WORDS]
    title = " ".join(words).strip(" ,;:-—")
    if len(title) < 12:  # too short to be informative on its own
        return _keyphrase_title(text, fallback)
    return title[0].upper() + title[1:]


def _keyphrase_title(text: str, fallback: str = "Untitled Segment") -> str:
    """Last-resort title built from the most distinctive words in ``text``."""
    words = _WORD_RE.findall(text or "")
    content = [w for w in words if w.lower() not in _TITLE_STOPWORDS and len(w) > 2]
    if not content:
        return fallback

    counts = Counter(w.lower() for w in content)
    top = {w for w, _ in counts.most_common(config.CHAPTER_TITLE_MAX_WORDS)}

    ordered: list[str] = []
    seen: set[str] = set()
    for word in content:
        low = word.lower()
        if low in top and low not in seen:
            ordered.append(word)
            seen.add(low)
        if len(ordered) >= config.CHAPTER_TITLE_MAX_WORDS:
            break

    title = " ".join(ordered).strip(" -,;:")
    return title.title() if title else fallback


def _title_word_stem(word: str) -> str:
    """Normalise a title word for the repetition check.

    Strips quotes, punctuation and possessives so that "'Spy'" and "Spy's"
    count as one word rather than two — which is what let looping output like
    "'Spy': 'Spy's Code" slip past a naive uniqueness ratio.
    """
    return _TITLE_STEM_RE.sub("", word.lower()).rstrip("s") or word.lower()


def _is_usable_title(title: str) -> bool:
    """Reject generated titles that are too short or have collapsed into a loop.

    Given a seed with little to work from, flan-t5 repeats a fragment to fill
    its output — "Still a Still a", "'Spy': 'Spy's Code". A title is only a
    few words long, so no *content* word should appear in it twice; that is a
    sharper test than a uniqueness ratio, which a three-word title with one
    repeat still passes. Short function words are exempt, since "How to add
    and to remove" is a legitimate title.
    """
    words = title.split()
    if len(title) < 8 or not words:
        return False
    content = [stem for stem in (_title_word_stem(w) for w in words) if len(stem) >= 3]
    if not content:
        return False
    return len(set(content)) == len(content)

def _generate_chapter_titles(seeds: list[str]) -> list[str]:
    """Titles for every chapter, in one batched generation call.

    Batching is what makes the model affordable here: the previous version
    loaded flan-t5 and ran it once per chapter, so eight chapters meant eight
    separate forward passes for a few words each. One batch over all chapters
    costs barely more than a single call.

    Falls back to the extracted title per chapter whenever generation fails or
    returns something degenerate, so a bad model output can never leave a
    chapter untitled.
    """
    fallbacks = [
        _clause_title(seed) if seed.strip() else "Visual Segment" for seed in seeds
    ]
    # Seeds with no text are left to the fallback rather than sent to the
    # model, which would otherwise invent a title from nothing.
    generatable = [i for i, seed in enumerate(seeds) if seed.strip()]
    if not config.CHAPTER_TITLES_USE_LLM or not generatable:
        return fallbacks

    prompts = [
        "Write a short, specific title of at most "
        f"{config.CHAPTER_TITLE_MAX_WORDS} words for this video section." + chr(10)
        + f"Section: {seeds[i][:800]}" + chr(10) + "Title:"
        for i in generatable
    ]
    try:
        outputs = _get_instruct_generator()(
            prompts, max_new_tokens=16, truncation=True,
            batch_size=min(len(prompts), config.CHAPTER_TITLE_BATCH_SIZE),
        )
    except Exception:
        logger.warning("chapter title generation failed; using extracted titles", exc_info=True)
        return fallbacks

    titles = list(fallbacks)
    for index, output in zip(generatable, outputs):
        entry = output[0] if isinstance(output, list) else output
        words = entry["generated_text"].strip().split()[: config.CHAPTER_TITLE_MAX_WORDS]
        title = _TITLE_QUOTES_RE.sub("", " ".join(words)).strip().rstrip(".,;:!?- ")
        if _is_usable_title(title):
            titles[index] = title
    return titles


def _build_chapters(
    windows: list[tuple[float, float]],
    fused_blocks: list[dict[str, Any]],
    chunk_summaries: list[dict[str, Any]],
    model_name: str,
) -> list[dict[str, Any]]:
    """Assemble chapters from work the map stage already did.

    A chapter's text is the chunk summaries whose midpoints fall inside its
    window — text the summarizer has already read and condensed. Only a window
    that catches no chunk at all (a purely visual stretch, or a gap left by
    coarse chunking) falls back to summarizing raw blocks, and that path is
    gated behind ``CHAPTERS_RESUMMARIZE_RAW`` for anything else.
    """
    last_window = windows[-1] if windows else None
    drafts: list[dict[str, Any]] = []

    for start, end in windows:
        inclusive_end = (start, end) == last_window
        summaries_here = _items_in_window(chunk_summaries, start, end, inclusive_end)
        blocks_here = _items_in_window(fused_blocks, start, end, inclusive_end)

        if summaries_here and not config.CHAPTERS_RESUMMARIZE_RAW:
            summary_text = " ".join(s["text"] for s in summaries_here if s["text"]).strip()
            # Several chunks in one window can run long for a chapter blurb;
            # condense only in that case, which is rare.
            if len(summary_text.split()) > config.CHAPTER_SUMMARY_MAX_WORDS:
                summary_text = _safe_summarize(
                    summary_text, config.CHUNK_SUMMARY_MAX_LEN,
                    config.CHUNK_SUMMARY_MIN_LEN, model_name,
                )
            seed = summary_text
        else:
            chapter_text = " ".join(
                t for t in (_narrative_text(b) for b in blocks_here) if t
            ).strip()
            if chapter_text:
                summary_text = _safe_summarize(
                    chapter_text, config.CHUNK_SUMMARY_MAX_LEN,
                    config.CHUNK_SUMMARY_MIN_LEN, model_name,
                )
                seed = summary_text
            else:
                tags = sorted({tag for b in blocks_here for tag in b.get("visual_tags", [])})
                summary_text = (
                    f"Visual segment ({', '.join(tags) if tags else 'no speech detected'})."
                )
                # No speech and no summary: the only seed would be a tag list,
                # which is not something a title model can work from.
                seed = ""

        drafts.append({"start": start, "end": end, "summary": summary_text, "seed": seed})

    # Titles are generated for every chapter at once, after all the summaries
    # exist — one batched pass instead of a model call per chapter.
    titles = _generate_chapter_titles([d["seed"] for d in drafts])
    return [
        {"start": d["start"], "end": d["end"], "title": title, "summary": d["summary"]}
        for d, title in zip(drafts, titles)
    ]


# --- Orchestration -----------------------------------------------------------------


def summarize(
    video_id: str, force: bool = False, plan: Optional[perf.ExecutionPlan] = None
) -> dict[str, Any]:
    """Map-reduce fused.json into summary.json (overall summary + chapters).

    Args:
        video_id: The cache key for this video.
        force: Re-run even if a cached summary.json already exists.
        plan: Execution plan for this video; read from meta.json if omitted.

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

    duration_sec = float(meta["duration_sec"])
    if plan is None:
        plan = (
            perf.plan_from_dict(meta["plan"]) if meta.get("plan")
            else perf.plan_for(duration_sec)
        )
    model_name = plan.summarizer_model
    batch_size = plan.summarizer_batch_size

    cache.set_state(video_id, "summarize", 0.0, "Preparing transcript...")
    blocks = fused_data["blocks"]
    scenes = scenes_data["scenes"] if scenes_data else []

    embeddings: Optional[np.ndarray] = None
    if scenes_data and scenes_data.get("embeddings_file"):
        embeddings_path = Path(scenes_data["embeddings_file"])
        if embeddings_path.exists():
            embeddings = np.load(embeddings_path)

    if not any(b.get("text", "").strip() for b in blocks):
        all_tags = sorted({tag for b in blocks for tag in b.get("visual_tags", [])})
        result = {
            "overall_summary": (
                "No speech was detected in this video. Based on visual analysis, it "
                f"contains scenes featuring: "
                f"{', '.join(all_tags) if all_tags else 'no identifiable objects'}."
            ),
            "chapters": _build_chapters(
                _build_chapter_windows(scenes, embeddings, duration_sec), blocks, [], model_name
            ),
            "chunk_summaries": [],
            "warning": "no_speech_detected",
            "model": model_name,
        }
        cache.write_json(video_id, "summary", result)
        cache.set_state(video_id, "summarize", 1.0, "Summarization complete.")
        return result

    started = time.perf_counter()
    tokenizer = _get_tokenizer(model_name)

    cache.set_state(video_id, "summarize", 0.1, "Chunking transcript...")
    chunks = _chunk_transcript(blocks, tokenizer, plan.max_chunks)

    cache.set_state(
        video_id, "summarize", 0.25, f"Summarizing {len(chunks)} sections ({model_name.split('/')[-1]})..."
    )
    chunk_summaries = _summarize_chunks(chunks, model_name, batch_size)

    cache.set_state(video_id, "summarize", 0.7, "Writing the overall summary...")
    overall_summary = (
        _reduce_to_overall_summary(chunk_summaries, tokenizer, model_name)
        if chunk_summaries else ""
    )

    cache.set_state(video_id, "summarize", 0.9, "Building chapters...")
    windows = _build_chapter_windows(scenes, embeddings, duration_sec, len(chunk_summaries))
    chapters = _build_chapters(windows, blocks, chunk_summaries, model_name)

    elapsed = time.perf_counter() - started
    # Recorded per *chunk*, matching the units core.perf plans in.
    # The planner reads this key as SummarizerRung.reference_sec_per_chunk;
    # recording a real-time factor here instead made bart-large-cnn look
    # almost free, so it was chosen for videos it could not finish in time.
    if len(chunk_summaries) >= config.MIN_CHUNKS_FOR_CALIBRATION:
        perf.record_observation(
            f"sum:{model_name.split(chr(47))[-1]}", elapsed / len(chunk_summaries)
        )
    logger.info(
        "summarized %.1f min into %d chunks + %d chapters in %.1f s",
        duration_sec / 60, len(chunk_summaries), len(chapters), elapsed,
    )

    result = {
        "overall_summary": overall_summary,
        "chapters": chapters,
        "chunk_summaries": chunk_summaries,
        "model": model_name,
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

    print(json.dumps(summarize(args.video_id, force=args.force), indent=2))
