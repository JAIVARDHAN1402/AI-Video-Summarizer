"""M2 — Transcription: YouTube captions when they're available and good
enough, otherwise faster-whisper ASR with segment-level timestamps.

Whisper decodes strictly left-to-right, so one call transcribes a single
30-second window at a time no matter how many cores are idle. For anything
longer than a few minutes this module therefore splits the audio at natural
silences, transcribes the pieces concurrently, and shifts each piece's
timestamps back onto the original timeline. On a 16-core machine that is close
to a 4x speedup, and it is what keeps long videos inside the time budget.

The concurrency is **threads over one shared model**, not worker processes.
CTranslate2 (which backs faster-whisper) releases the GIL during compute and
exposes ``num_workers`` for exactly this pattern, so threads parallelise
genuinely — while avoiding two real problems with a process pool: it would load
a full copy of the model per worker, and under Windows' spawn start method the
children re-import ``__main__``, which is actively unsafe inside a Streamlit
process.

Splitting on silence (rather than at fixed offsets) matters: a cut through the
middle of a word both loses that word and tends to make Whisper hallucinate a
plausible-sounding continuation at the boundary.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

import config
from core import cache, perf
from core.models import TranscriptionError

logger = logging.getLogger(__name__)

_MODEL = None  # memoized in-process faster_whisper.WhisperModel
_MODEL_KEY: Optional[tuple[str, int, int]] = None
_MODEL_LOCK = threading.Lock()

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/|v/))([A-Za-z0-9_-]{11})"
)
_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)")


def _extract_youtube_id(url: str) -> Optional[str]:
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def _reindex(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, seg in enumerate(segments):
        seg["id"] = i
    return segments


# --- YouTube captions (the fast path) --------------------------------------------


def _segments_from_caption_entries(raw: Any) -> list[dict[str, Any]]:
    segments = []
    for entry in raw:
        text = str(entry["text"]).strip().replace("\n", " ")
        if not text:
            continue
        start = float(entry["start"])
        segments.append({
            "id": 0,
            "start": start,
            "end": start + float(entry.get("duration", 0.0)),
            "text": text,
        })
    return _reindex(segments)


def _try_youtube_captions(meta: dict[str, Any], allow_generated: bool) -> Optional[dict[str, Any]]:
    """Use existing English YouTube captions instead of running ASR at all.

    Manually-written captions are strictly better than anything we would
    produce locally, so they're always preferred. Auto-generated ones are only
    accepted when ``allow_generated`` is set, because they carry no punctuation
    and unpunctuated text measurably degrades the downstream summaries — that
    tradeoff is only worth taking when the planner has already had to drop to a
    small Whisper model, which would be worse still.
    """
    yt_id = _extract_youtube_id(meta.get("source_ref", ""))
    if yt_id is None:
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(yt_id)
    except Exception as exc:
        logger.info("no caption list for %s: %s", yt_id, exc)
        return None

    for finder, kind in (
        (transcript_list.find_manually_created_transcript, "youtube_captions"),
        (transcript_list.find_generated_transcript, "youtube_auto_captions"),
    ):
        if kind == "youtube_auto_captions" and not allow_generated:
            continue
        try:
            segments = _segments_from_caption_entries(finder(["en"]).fetch())
        except Exception:
            continue
        if segments:
            logger.info("using %s for %s (%d segments)", kind, yt_id, len(segments))
            return {
                "language": "en",
                "source": kind,
                "full_text": " ".join(s["text"] for s in segments),
                "segments": segments,
            }
    return None


# --- Whisper ---------------------------------------------------------------------


def _select_device() -> tuple[str, str]:
    """Pick device/compute_type without importing torch (CTranslate2 backs faster-whisper)."""
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", config.WHISPER_COMPUTE_TYPE


def _get_model(model_size: str, cpu_threads: int, num_workers: int = 1):
    """Load (and memoize) the Whisper model shared by every worker thread.

    ``num_workers`` is CTranslate2's own parallelism knob: it sizes the pool of
    internal execution contexts, which is what allows several threads to call
    ``transcribe()`` on this one instance at the same time.
    """
    global _MODEL, _MODEL_KEY
    key = (model_size, cpu_threads, num_workers)
    with _MODEL_LOCK:
        if _MODEL is None or _MODEL_KEY != key:
            from faster_whisper import WhisperModel

            device, compute_type = _select_device()
            logger.info(
                "loading faster-whisper model=%s device=%s compute=%s threads=%d workers=%d",
                model_size, device, compute_type, cpu_threads, num_workers,
            )
            _MODEL = WhisperModel(
                model_size, device=device, compute_type=compute_type,
                cpu_threads=cpu_threads if device == "cpu" else 0,
                num_workers=num_workers,
            )
            _MODEL_KEY = key
    return _MODEL


def _transcribe_file(
    audio_path: str, model_size: str, beam_size: int, cpu_threads: int,
    time_offset: float = 0.0, num_workers: int = 1,
    on_advance: Optional[Callable[[float], None]] = None,
    chunk_duration: float = 0.0,
    language: Optional[str] = None, task: str = "transcribe",
) -> list[dict[str, Any]]:
    """Transcribe one audio file, returning segments shifted by ``time_offset``.

    Safe to call from several threads at once against the same model, provided
    that model was built with a matching ``num_workers``.

    ``on_advance`` is invoked with this chunk's completed fraction as segments
    arrive. Whisper's output is a generator, so this is the only point at which
    progress is observable at all — without it a multi-minute transcription
    reports nothing between start and finish, which reads as a hang.
    """
    model = _get_model(model_size, cpu_threads, num_workers)
    segment_iter, _info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=True,
        language=language,
        task=task,
        # Feeding each window the previous window's text is what lets Whisper
        # spiral into repetition loops, and it serialises decoding across
        # chunk boundaries that no longer exist once the audio is split.
        condition_on_previous_text=False,
    )
    segments = []
    for seg in segment_iter:
        if on_advance is not None and chunk_duration > 0:
            on_advance(min(seg.end / chunk_duration, 1.0))
        text = seg.text.strip()
        if not text:
            continue
        segments.append({
            "id": 0,
            "start": seg.start + time_offset,
            "end": seg.end + time_offset,
            "text": text,
        })
    return segments


# --- Silence-aware splitting -----------------------------------------------------


def _detect_silences(audio_path: Path) -> tuple[list[float], float]:
    """Return silence midpoints and the total silent seconds.

    One audio-only FFmpeg pass; on a 16 kHz mono WAV this is I/O-bound and
    costs a small fraction of what it saves by enabling parallel ASR.

    The silence total matters beyond splitting: Whisper's VAD skips silence,
    so its cost tracks *speech*, not video length. Reporting cost per
    video-second makes a dense lecture look several times more expensive than
    a pause-heavy one on the same machine, which is what made the planner's
    calibration untransferable between videos.
    """
    cmd = [
        "ffmpeg", "-v", "info", "-i", str(audio_path),
        "-af", f"silencedetect=noise={config.ASR_SILENCE_NOISE_DB}dB:d={config.ASR_SILENCE_MIN_DUR_SEC}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.SubprocessError, OSError):
        logger.warning("silence detection failed; using fixed-offset splits", exc_info=True)
        return [], 0.0

    starts = [float(m) for m in _SILENCE_START_RE.findall(result.stderr)]
    ends = [float(m) for m in _SILENCE_END_RE.findall(result.stderr)]
    pairs = list(zip(starts, ends))
    midpoints = [(a + b) / 2.0 for a, b in pairs]
    silent_sec = sum(max(b - a, 0.0) for a, b in pairs)
    return midpoints, silent_sec

def _plan_split_points(duration: float, n_chunks: int, silences: list[float]) -> list[float]:
    """Ideal evenly-spaced cut points, each snapped to a nearby silence."""
    if n_chunks <= 1:
        return []

    points: list[float] = []
    window = config.ASR_SILENCE_SEARCH_WINDOW_SEC
    for i in range(1, n_chunks):
        ideal = duration * i / n_chunks
        candidates = [s for s in silences if abs(s - ideal) <= window]
        chosen = min(candidates, key=lambda s: abs(s - ideal)) if candidates else ideal
        # Guard against two ideal points snapping onto the same silence, which
        # would otherwise produce a zero-length chunk.
        if points and chosen - points[-1] < 1.0:
            continue
        points.append(chosen)
    return points


def _cut_chunk(audio_path: Path, start: float, end: Optional[float], out_path: Path) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-i", str(audio_path), "-c", "copy", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise TranscriptionError(f"FFmpeg failed to split audio: {result.stderr.strip()[-500:]}")


_PROBE_MODEL = None


def _get_probe_model(cpu_threads: int):
    """Small multilingual model kept aside purely for language detection."""
    global _PROBE_MODEL
    with _MODEL_LOCK:
        if _PROBE_MODEL is None:
            from faster_whisper import WhisperModel

            device, compute_type = _select_device()
            logger.info("loading language-probe model=%s", config.WHISPER_LANGUAGE_DETECT_MODEL)
            _PROBE_MODEL = WhisperModel(
                config.WHISPER_LANGUAGE_DETECT_MODEL, device=device, compute_type=compute_type,
                cpu_threads=cpu_threads if device == "cpu" else 0,
            )
    return _PROBE_MODEL

def _detect_language(audio_path: Path, duration: float, plan: perf.ExecutionPlan) -> tuple[str, float]:
    """Identify the spoken language from probes spread across the audio.

    Whisper's own auto-detection reads only the opening window, which on a
    lecture is often an intro jingle, music, or a sponsor read — none of which
    represent the body of the video. Pinning the language instead (this
    project previously hardcoded English) is worse still: forced to decode
    Hindi-English speech as English, Whisper emits transliterated nonsense
    ("Herki sikale n calls lagi") *and* runs several times slower, because
    every window fights the language model instead of agreeing with it.

    So: sample a few short probes from across the file and take the most
    confident agreement. Cost is a couple of seconds against a stage measured
    in minutes.

    Returns:
        ``(language_code, confidence)``.
    """
    # Detection must use a *multilingual* checkpoint. The ".en" models are
    # English-only and report "en" for any input, so probing with the
    # planned model would defeat the whole exercise. A tiny multilingual
    # model is plenty for identifying a language.
    model = _get_probe_model(plan.asr_threads_per_worker)
    # Probes must not outnumber the audio. On a one-minute clip three
    # 30-second windows overlap almost entirely, so they cost real time to
    # re-confirm what the first one already said.
    probe_count = max(1, min(
        config.WHISPER_LANGUAGE_PROBES,
        int(duration // config.WHISPER_LANGUAGE_PROBE_SEC) or 1,
    ))
    probe_dir = audio_path.parent / "lang_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    votes: dict[str, float] = {}
    try:
        for i in range(probe_count):
            # Spread the probes over the middle of the file, skipping the
            # opening and closing where intros and outros live.
            position = duration * (0.15 + 0.7 * (i / max(probe_count - 1, 1)))
            probe_path = probe_dir / f"probe_{i:02d}.wav"
            try:
                _cut_chunk(audio_path, position, position + config.WHISPER_LANGUAGE_PROBE_SEC, probe_path)
                _segments, info = model.transcribe(str(probe_path), language=None)
            except Exception:
                logger.debug("language probe %d failed", i, exc_info=True)
                continue
            language = getattr(info, "language", None)
            probability = float(getattr(info, "language_probability", 0.0) or 0.0)
            if language:
                votes[language] = votes.get(language, 0.0) + probability
    finally:
        for probe in probe_dir.glob("probe_*.wav"):
            probe.unlink(missing_ok=True)
        try:
            probe_dir.rmdir()
        except OSError:
            pass

    if not votes:
        logger.info("language detection produced no result; assuming English")
        return "en", 0.0

    language = max(votes, key=lambda k: votes[k])
    confidence = votes[language] / max(sum(votes.values()), 1e-6)
    logger.info("detected language=%s (agreement %.2f) from %d probes", language, confidence, probe_count)
    return language, confidence

def _transcribe_parallel(
    video_id: str, audio_path: Path, duration: float, plan: perf.ExecutionPlan,
    language: Optional[str], task: str, asr_model: str,
) -> tuple[list[dict[str, Any]], float]:
    """Split the audio at silences and transcribe the pieces concurrently.

    Returns the segments plus the estimated seconds of actual speech, which
    is what the ASR cost should be calibrated against.
    """
    workers = max(1, plan.asr_workers)
    n_chunks = max(1, min(workers, int(duration // config.ASR_MIN_CHUNK_SEC)))
    if n_chunks <= 1:
        return _transcribe_file(
            str(audio_path), asr_model, plan.asr_beam_size, plan.asr_threads_per_worker,
            language=language, task=task,
        ), duration

    cache.set_state(video_id, "transcribe", 0.02, "Finding natural split points...")
    silences, silent_sec = _detect_silences(audio_path)
    speech_sec = max(duration - silent_sec, 1.0)
    split_points = _plan_split_points(duration, n_chunks, silences)
    boundaries = [0.0, *split_points, duration]

    chunk_dir = audio_path.parent / "audio_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, float]] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < 1.0:
            continue
        chunk_path = chunk_dir / f"chunk_{i:03d}.wav"
        # The final chunk is cut open-ended so float rounding can't clip the
        # last fraction of a second of audio.
        _cut_chunk(audio_path, start, end if i < len(boundaries) - 2 else None, chunk_path)
        jobs.append((chunk_path, start))

    if not jobs:
        return _transcribe_file(
            str(audio_path), asr_model, plan.asr_beam_size, plan.asr_threads_per_worker,
            language=language, task=task,
        ), speech_sec

    logger.info(
        "transcribing %d sections across %d threads (model=%s, beam=%d)",
        len(jobs), workers, asr_model, plan.asr_beam_size,
    )
    cache.set_state(
        video_id, "transcribe", 0.05,
        f"Transcribing {len(jobs)} sections in parallel ({asr_model})...",
    )

    # Build the shared model once, before any thread starts, so the workers
    # don't all race to load it.
    _get_model(asr_model, plan.asr_threads_per_worker, workers)

    results: list[list[dict[str, Any]]] = [[] for _ in jobs]
    # Each thread reports its own fraction; the aggregate is their mean, so the
    # bar advances continuously rather than in whole-section steps.
    fractions = [0.0] * len(jobs)
    progress_lock = threading.Lock()

    def report_progress(index: int, fraction: float) -> None:
        with progress_lock:
            fractions[index] = fraction
            overall = sum(fractions) / len(fractions)
            done = sum(1 for f in fractions if f >= 1.0)
        cache.set_state(
            video_id, "transcribe", 0.05 + 0.9 * overall,
            f"Transcribing {len(jobs)} sections in parallel — "
            f"{overall * 100:.0f}% ({done}/{len(jobs)} finished)...",
        )

    chunk_durations = [
        (jobs[i + 1][1] - offset) if i + 1 < len(jobs) else max(duration - offset, 1.0)
        for i, (_, offset) in enumerate(jobs)
    ]

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="asr") as pool:
            futures = {
                pool.submit(
                    _transcribe_file, str(path), asr_model, plan.asr_beam_size,
                    plan.asr_threads_per_worker, offset, workers,
                    lambda fraction, i=i: report_progress(i, fraction),
                    chunk_durations[i], language, task,
                ): i
                for i, (path, offset) in enumerate(jobs)
            }
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                report_progress(index, 1.0)
                completed += 1
    except Exception:
        # Falling back to a single sequential pass is slower but still correct,
        # which is the right trade for a user-facing pipeline.
        logger.warning("parallel transcription failed; retrying sequentially", exc_info=True)
        return _transcribe_file(
            str(audio_path), asr_model, plan.asr_beam_size, plan.asr_threads_per_worker,
            language=language, task=task,
        ), speech_sec
    finally:
        for path, _ in jobs:
            path.unlink(missing_ok=True)

    return [seg for chunk_segments in results for seg in chunk_segments], speech_sec


def _transcribe_with_whisper(
    video_id: str, meta: dict[str, Any], plan: perf.ExecutionPlan
) -> dict[str, Any]:
    audio_path = cache.get_cache_dir(video_id) / "audio.wav"
    if not audio_path.exists():
        raise TranscriptionError(f"audio.wav not found for {video_id}; run ingest() first.")

    duration = max(float(meta.get("duration_sec", 0.0)), 1.0)

    # Establish the language before any transcription, so every parallel
    # chunk decodes with the same setting. Letting each chunk auto-detect
    # independently would let them disagree mid-video.
    if config.WHISPER_LANGUAGE:
        language, confidence = config.WHISPER_LANGUAGE, 1.0
    else:
        cache.set_state(video_id, "transcribe", 0.01, "Detecting spoken language...")
        language, confidence = _detect_language(audio_path, duration, plan)

    # Every downstream model is English-only, so non-English audio is
    # translated rather than transcribed verbatim.
    translating = config.WHISPER_TRANSLATE_NON_ENGLISH and language != "en"
    task = "translate" if translating else "transcribe"
    asr_model = plan.asr_model
    if language != "en" and asr_model.endswith(".en"):
        # An English-only checkpoint cannot read or translate other
        # languages at all, so fall back to its multilingual sibling.
        asr_model = asr_model[: -len(".en")]
        logger.info("switching to multilingual %s for %s audio", asr_model, language)
    if translating:
        logger.info("audio is %s; translating to English for the English-only downstream models", language)

    # Same reasoning as core/vision.py: model loading is not throughput.
    _get_model(asr_model, plan.asr_threads_per_worker, max(plan.asr_workers, 1))
    started = time.perf_counter()
    if duration >= config.ASR_PARALLEL_MIN_DURATION_SEC and plan.asr_workers > 1:
        segments, speech_sec = _transcribe_parallel(
            video_id, audio_path, duration, plan, language, task, asr_model
        )
    else:
        cache.set_state(video_id, "transcribe", 0.1, f"Transcribing audio ({asr_model})...")
        segments = _transcribe_file(
            str(audio_path), asr_model, plan.asr_beam_size, plan.asr_threads_per_worker,
            language=language, task=task,
        )
        speech_sec = duration
    elapsed = time.perf_counter() - started
    # Calibrated per second of *speech*, not per second of video. VAD skips
    # silence, so a 40-minute lecture with continuous narration costs several
    # times what a pause-heavy one of the same length does — recording against
    # video length produced a number that did not transfer between videos and
    # left the planner badly optimistic on dense content.
    # A short clip measures model loading, not transcription throughput.
    if duration >= config.MIN_DURATION_FOR_CALIBRATION_SEC:
        perf.record_observation(
            perf.asr_key(asr_model, plan.asr_beam_size), elapsed / max(speech_sec, 1.0)
        )
    logger.info(
        "transcribed %.1f min (%.1f min of speech) in %.1f s (%.3f s per speech-sec)",
        duration / 60, speech_sec / 60, elapsed, elapsed / max(speech_sec, 1.0),
    )
    segments.sort(key=lambda s: s["start"])
    segments = _reindex(segments)
    if not segments:
        raise TranscriptionError("No speech was detected in this video's audio.")

    return {
        "language": language,
        "language_confidence": round(confidence, 3),
        "translated": translating,
        "source": "whisper",
        "model": asr_model,
        "full_text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }


# --- Public API --------------------------------------------------------------------


def transcribe(
    video_id: str, force: bool = False, plan: Optional[perf.ExecutionPlan] = None
) -> dict[str, Any]:
    """Produce transcript.json, preferring YouTube captions over Whisper ASR.

    Args:
        video_id: The cache key for this video (see core.cache.get_video_id).
        force: Re-run even if a cached transcript.json already exists.
        plan: Execution plan for this video; read from meta.json if omitted.

    Returns:
        The transcript.json dict.
    """
    if not force and cache.artifact_exists(video_id, "transcript"):
        logger.info("using cached transcript.json for %s", video_id)
        return cache.read_json(video_id, "transcript")  # type: ignore[return-value]

    meta = cache.read_json(video_id, "meta")
    if meta is None:
        raise TranscriptionError(f"No meta.json for {video_id}; run ingest() first.")

    if plan is None:
        plan = (
            perf.plan_from_dict(meta["plan"]) if meta.get("plan")
            else perf.plan_for(float(meta.get("duration_sec", 0.0)))
        )

    cache.set_state(video_id, "transcribe", 0.0, "Checking for existing captions...")

    transcript_data: Optional[dict[str, Any]] = None
    if config.PREFER_YOUTUBE_CAPTIONS and meta.get("source") == "youtube":
        # Auto-generated captions beat a downshifted Whisper model on both
        # speed and accuracy, so they're only unlocked once the planner has
        # already given up the accurate ones.
        allow_generated = plan.asr_model in config.CAPTIONS_PREFERRED_OVER_MODELS
        transcript_data = _try_youtube_captions(meta, allow_generated=allow_generated)

    if transcript_data is None:
        transcript_data = _transcribe_with_whisper(video_id, meta, plan)

    cache.write_json(video_id, "transcript", transcript_data)
    cache.set_state(video_id, "transcribe", 1.0, "Transcription complete.")
    return transcript_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M2 transcription in isolation.")
    parser.add_argument("video_id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = transcribe(args.video_id, force=args.force)
    print(json.dumps({**result, "segments": f"{len(result['segments'])} segments"}, indent=2)[:2000])
