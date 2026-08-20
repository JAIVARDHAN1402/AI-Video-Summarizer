"""M2 — Transcription: YouTube captions when available (fast), else
faster-whisper ASR with word/segment-level timestamps (works for any video).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import config
from core import cache
from core.models import TranscriptionError

logger = logging.getLogger(__name__)

_MODEL = None  # memoized faster_whisper.WhisperModel, loaded lazily

_YOUTUBE_ID_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?v=|shorts/|embed/|v/))([A-Za-z0-9_-]{11})"
)


def _extract_youtube_id(url: str) -> Optional[str]:
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def _reindex(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, seg in enumerate(segments):
        seg["id"] = i
    return segments


def _try_youtube_captions(meta: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Use manually-created (non-auto-generated) English YouTube captions if present.

    This is ~50x faster than running Whisper, so we always try it first for
    YouTube sources before falling back to ASR.
    """
    yt_id = _extract_youtube_id(meta.get("source_ref", ""))
    if yt_id is None:
        return None

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        transcript_list = YouTubeTranscriptApi.list_transcripts(yt_id)
        transcript = transcript_list.find_manually_created_transcript(["en"])
        raw = transcript.fetch()
    except Exception as exc:
        logger.info("no manual YouTube captions for %s: %s", yt_id, exc)
        return None

    segments = []
    for entry in raw:
        text = entry["text"].strip().replace("\n", " ")
        if not text:
            continue
        start = float(entry["start"])
        end = start + float(entry.get("duration", 0.0))
        segments.append({"id": 0, "start": start, "end": end, "text": text})

    if not segments:
        return None

    segments = _reindex(segments)
    full_text = " ".join(s["text"] for s in segments)
    return {
        "language": "en",
        "source": "youtube_captions",
        "full_text": full_text,
        "segments": segments,
    }


def _select_device() -> tuple[str, str]:
    """Pick device/compute_type without importing torch (faster-whisper uses CTranslate2)."""
    try:
        import ctranslate2

        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False

    if has_cuda:
        return "cuda", "float16"
    return "cpu", config.WHISPER_COMPUTE_TYPE


def _get_model():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel

        device, compute_type = _select_device()
        logger.info(
            "loading faster-whisper model=%s device=%s compute_type=%s",
            config.WHISPER_MODEL_SIZE, device, compute_type,
        )
        _MODEL = WhisperModel(config.WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    return _MODEL


def _transcribe_with_whisper(video_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    cache_dir = cache.get_cache_dir(video_id)
    audio_path = cache_dir / "audio.wav"
    if not audio_path.exists():
        raise TranscriptionError(f"audio.wav not found for {video_id}; run ingest() first.")

    model = _get_model()
    duration_sec = max(float(meta.get("duration_sec", 0.0)), 1e-6)

    segment_iter, info = model.transcribe(
        str(audio_path),
        beam_size=config.WHISPER_BEAM_SIZE,
        vad_filter=True,
    )

    segments: list[dict[str, Any]] = []
    for seg in segment_iter:
        text = seg.text.strip()
        if not text:
            continue
        segments.append({"id": len(segments), "start": seg.start, "end": seg.end, "text": text})
        progress = min(seg.end / duration_sec, 1.0)
        cache.set_state(
            video_id, "transcribe", progress,
            f"Transcribing audio (Whisper {config.WHISPER_MODEL_SIZE})...",
        )

    if not segments:
        raise TranscriptionError("Whisper produced no speech segments for this video.")

    full_text = " ".join(s["text"] for s in segments)
    return {
        "language": getattr(info, "language", None) or "en",
        "source": "whisper",
        "full_text": full_text,
        "segments": segments,
    }


def transcribe(video_id: str, force: bool = False) -> dict[str, Any]:
    """Produce transcript.json, preferring YouTube captions over Whisper ASR.

    Args:
        video_id: The cache key for this video (see core.cache.get_video_id).
        force: Re-run even if a cached transcript.json already exists.

    Returns:
        The transcript.json dict.
    """
    if not force and cache.artifact_exists(video_id, "transcript"):
        logger.info("using cached transcript.json for %s", video_id)
        return cache.read_json(video_id, "transcript")  # type: ignore[return-value]

    meta = cache.read_json(video_id, "meta")
    if meta is None:
        raise TranscriptionError(f"No meta.json for {video_id}; run ingest() first.")

    cache.set_state(video_id, "transcribe", 0.0, "Starting transcription...")

    transcript_data: Optional[dict[str, Any]] = None
    if config.PREFER_YOUTUBE_CAPTIONS and meta.get("source") == "youtube":
        transcript_data = _try_youtube_captions(meta)

    if transcript_data is None:
        transcript_data = _transcribe_with_whisper(video_id, meta)

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
    print(json.dumps(result, indent=2)[:2000])
