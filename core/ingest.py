"""M1 — Ingestion: fetch a video (YouTube URL or local upload), probe it,
and extract the raw audio + sampled frames every later stage builds on.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import config
from core import cache
from core.models import DependencyError, IngestError, VideoTooLongError

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Fail fast and clearly if FFmpeg is not on PATH.

    Every downstream module (audio extraction, frame sampling, scene
    detection) silently produces garbage or crashes deep in a subprocess
    if FFmpeg is missing, so we check once, at import time, here.
    """
    missing = [exe for exe in ("ffmpeg", "ffprobe") if shutil.which(exe) is None]
    if missing:
        raise DependencyError(
            f"Required system dependency missing: {', '.join(missing)}. "
            "Install FFmpeg and make sure it is on your PATH "
            "(https://ffmpeg.org/download.html), then restart the app."
        )


_check_dependencies()


def _probe(video_path: Path) -> dict[str, Any]:
    """Return duration_sec/width/height/fps for a video file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-show_entries", "format=duration",
        "-of", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"ffprobe failed on {video_path.name}: {result.stderr.strip()[-1000:]}")

    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration", 0.0))
    streams = data.get("streams") or [{}]
    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    r_frame_rate = stream.get("r_frame_rate", "0/1")
    num_str, _, den_str = r_frame_rate.partition("/")
    try:
        num, den = float(num_str), float(den_str or 1)
        fps = num / den if den else 0.0
    except ValueError:
        fps = 0.0

    return {"duration_sec": duration, "width": width, "height": height, "fps": fps}


def _download_youtube(url: str, dest_dir: Path, max_retries: int) -> tuple[Path, str]:
    """Download a YouTube (or yt-dlp-supported) URL, returning (video_path, title)."""
    import yt_dlp

    out_template = str(dest_dir / "video.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if config.YT_DLP_COOKIES_FROM_BROWSER:
        ydl_opts["cookiesfrombrowser"] = (config.YT_DLP_COOKIES_FROM_BROWSER,)

    last_error: Optional[BaseException] = None
    for attempt in range(1, max_retries + 2):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            video_path = dest_dir / "video.mp4"
            if not video_path.exists():
                candidates = [p for p in dest_dir.glob("video.*") if p.suffix != ".part"]
                if candidates:
                    candidates[0].rename(video_path)
            if not video_path.exists():
                raise IngestError("yt-dlp reported success but no output file was found")
            title = (info or {}).get("title") or url
            return video_path, title
        except Exception as exc:  # yt_dlp.DownloadError and friends
            last_error = exc
            logger.warning("yt-dlp attempt %d/%d failed: %s", attempt, max_retries + 1, exc)

    stderr_tail = str(last_error)[-1500:]
    raise IngestError(
        f"Could not download '{url}' after {max_retries + 1} attempt(s). "
        f"This is often a geo-block, private video, or YouTube bot-check. "
        f"Try uploading the file directly instead. Details: {stderr_tail}"
    )


def _extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",
        "-ac", str(config.AUDIO_CHANNELS),
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"FFmpeg audio extraction failed: {result.stderr.strip()[-1000:]}")


def _extract_frames(video_path: Path, frames_dir: Path, fps: float) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "4",  # mjpeg qscale 4 ~= JPEG quality 85
        pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"FFmpeg frame extraction failed: {result.stderr.strip()[-1000:]}")
    return len(list(frames_dir.glob("frame_*.jpg")))


def ingest(source_ref: str, is_upload: bool = False, force: bool = False) -> dict[str, Any]:
    """Fetch/copy a video, probe it, and extract audio.wav + frames/.

    Args:
        source_ref: A YouTube (or yt-dlp-supported) URL, or a local file path
            when ``is_upload`` is True.
        is_upload: True if ``source_ref`` is a local file path rather than a URL.
        force: Re-run ingestion even if a cached ``meta.json`` already exists.

    Returns:
        The ``meta.json`` dict.

    Raises:
        VideoTooLongError: if the video exceeds ``config.MAX_VIDEO_DURATION_SEC``.
        IngestError: on download/extraction failure.
    """
    video_id = cache.get_video_id(source_ref)

    if not force and cache.artifact_exists(video_id, "meta"):
        logger.info("using cached meta.json for %s", video_id)
        return cache.read_json(video_id, "meta")  # type: ignore[return-value]

    cache.set_state(video_id, "ingest", 0.0, "Fetching video...")
    cache_dir = cache.get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"

    if is_upload:
        source_path = Path(source_ref)
        if not source_path.exists():
            raise IngestError(f"Uploaded file not found: {source_ref}")
        shutil.copyfile(source_path, video_path)
        title = source_path.name
        source = "upload"
    else:
        video_path, title = _download_youtube(source_ref, cache_dir, config.INGEST_MAX_RETRIES)
        source = "youtube"

    cache.set_state(video_id, "ingest", 0.3, "Probing video metadata...")
    probe = _probe(video_path)

    if probe["duration_sec"] > config.MAX_VIDEO_DURATION_SEC:
        raise VideoTooLongError(
            f"Video is {probe['duration_sec'] / 60:.1f} min, which exceeds the "
            f"{config.MAX_VIDEO_DURATION_SEC / 60:.0f} min limit for this prototype."
        )

    cache.set_state(video_id, "ingest", 0.5, "Extracting audio...")
    audio_path = cache_dir / "audio.wav"
    _extract_audio(video_path, audio_path)

    cache.set_state(video_id, "ingest", 0.75, "Sampling frames...")
    frames_dir = cache_dir / "frames"
    n_frames = _extract_frames(video_path, frames_dir, config.FRAME_SAMPLE_FPS)
    logger.info("extracted %d frames for %s", n_frames, video_id)

    meta = {
        "video_id": video_id,
        "source": source,
        "source_ref": source_ref,
        "title": title,
        "duration_sec": probe["duration_sec"],
        "width": probe["width"],
        "height": probe["height"],
        "fps": probe["fps"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    cache.write_json(video_id, "meta", meta)
    cache.set_state(video_id, "ingest", 1.0, "Ingestion complete.")
    return meta


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M1 ingestion in isolation.")
    parser.add_argument("source", help="YouTube URL, or local file path with --upload")
    parser.add_argument("--upload", action="store_true", help="Treat source as a local file")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = ingest(args.source, is_upload=args.upload, force=args.force)
    print(json.dumps(result, indent=2))
