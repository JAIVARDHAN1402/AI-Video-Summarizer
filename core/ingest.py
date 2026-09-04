"""M1 — Ingestion: fetch a video (YouTube URL or local upload), probe it,
and extract the audio track every later stage builds on.

Duration is resolved *before* the download whenever the source allows it, so
the execution plan (core/perf.py) is chosen — and an over-long video is
rejected — without first spending bandwidth on a file we would refuse anyway.

This stage used to also decode the whole video a second time to write one JPEG
per second to ``frames/``. Nothing ever read that directory: the vision stage
seeks its own keyframes straight out of ``video.mp4``. Those thousands of
writes were removed, which is the single largest saving in this module.
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
from core import cache, perf
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


def _ydl_base_opts() -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if config.YT_DLP_COOKIES_FROM_BROWSER:
        opts["cookiesfrombrowser"] = (config.YT_DLP_COOKIES_FROM_BROWSER,)
    return opts


def probe_remote(url: str) -> dict[str, Any]:
    """Fetch duration and title for a URL without downloading the media.

    yt-dlp resolves metadata in a single cheap request. Doing this first lets
    the caller size its execution plan against the real duration, and lets an
    over-long video be rejected before any bytes are transferred.

    Returns:
        ``{"duration_sec": float, "title": str}``. ``duration_sec`` is 0.0 when
        the extractor doesn't report one (a live stream, some embeds), which
        callers treat as "unknown, plan defensively" rather than as an error.
    """
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({**_ydl_base_opts(), "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise IngestError(
            f"Could not read video details for '{url}'. This is usually a "
            f"private/removed video, a geo-block, or a YouTube bot-check. "
            f"Try uploading the file directly instead. Details: {str(exc)[-800:]}"
        ) from exc

    info = info or {}
    return {
        "duration_sec": float(info.get("duration") or 0.0),
        "title": info.get("title") or url,
    }


def _download_youtube(
    url: str, dest_dir: Path, max_retries: int, max_height: int = 720
) -> tuple[Path, str]:
    """Download a yt-dlp-supported URL, returning (video_path, title).

    Only the audio track and a handful of keyframes are ever analysed, so
    ``max_height`` caps resolution at what the vision stage can actually use —
    pixels beyond that cost download time and decode time that no stage spends.
    """
    import yt_dlp

    out_template = str(dest_dir / "video.%(ext)s")
    ydl_opts = {
        **_ydl_base_opts(),
        "format": (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        # Fetching one fragment at a time leaves most of the available
        # bandwidth idle on DASH/HLS sources, and long videos are frequently
        # download-bound.
        "concurrent_fragment_downloads": config.YT_DLP_CONCURRENT_FRAGMENTS,
    }

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
        "ffmpeg", "-y", "-v", "error",
        "-threads", "0",  # let FFmpeg use every core for the decode
        "-i", str(video_path),
        "-vn",
        "-ac", str(config.AUDIO_CHANNELS),
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"FFmpeg audio extraction failed: {result.stderr.strip()[-1000:]}")


def _reject_if_too_long(duration_sec: float) -> None:
    """Raise if a known duration exceeds the configured hard limit.

    A duration of 0 means the extractor didn't report one (live streams, some
    embeds); that is checked again against ffprobe once the file exists.
    """
    if duration_sec and duration_sec > config.MAX_VIDEO_DURATION_SEC:
        raise VideoTooLongError(
            f"This video is {duration_sec / 60:.0f} minutes long, which is over "
            f"the {config.MAX_VIDEO_DURATION_SEC / 60:.0f}-minute limit."
        )


def ingest(
    source_ref: str,
    is_upload: bool = False,
    force: bool = False,
    quality_floor: bool = False,
) -> dict[str, Any]:
    """Fetch/copy a video, probe it, extract audio.wav, and size the run.

    Duration is established before any bytes are downloaded (via
    :func:`probe_remote` for URLs, ffprobe for uploads), so the execution plan
    is chosen against the real length and an over-long video is rejected
    without paying for the transfer first.

    Args:
        source_ref: A YouTube (or yt-dlp-supported) URL, or a local file path
            when ``is_upload`` is True.
        is_upload: True if ``source_ref`` is a local file path rather than a URL.
        force: Re-run ingestion even if a cached ``meta.json`` already exists.
        quality_floor: Pass through to the planner to pin the accurate models.

    Returns:
        The ``meta.json`` dict, including the ``plan`` used for this video.

    Raises:
        VideoTooLongError: if the video exceeds ``config.MAX_VIDEO_DURATION_SEC``.
        IngestError: on download/extraction failure.
    """
    video_id = cache.get_video_id(source_ref)

    if not force and cache.artifact_exists(video_id, "meta"):
        logger.info("using cached meta.json for %s", video_id)
        return cache.read_json(video_id, "meta")  # type: ignore[return-value]

    cache.set_state(video_id, "ingest", 0.0, "Reading video details...")
    cache_dir = cache.get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"

    if is_upload:
        source_path = Path(source_ref)
        if not source_path.exists():
            raise IngestError(f"Uploaded file not found: {source_ref}")
        known_duration = _probe(source_path)["duration_sec"]
        title = source_path.name
        source = "upload"
    else:
        remote = probe_remote(source_ref)
        known_duration = remote["duration_sec"]
        title = remote["title"]
        source = "youtube"

    _reject_if_too_long(known_duration)
    plan = perf.plan_for(known_duration or config.MAX_VIDEO_DURATION_SEC, quality_floor=quality_floor)

    if is_upload:
        cache.set_state(video_id, "ingest", 0.2, "Copying uploaded file...")
        shutil.copyfile(Path(source_ref), video_path)
    else:
        cache.set_state(
            video_id, "ingest", 0.2,
            f"Downloading video ({plan.download_height}p)...",
        )
        video_path, title = _download_youtube(
            source_ref, cache_dir, config.INGEST_MAX_RETRIES, max_height=plan.download_height
        )

    cache.set_state(video_id, "ingest", 0.6, "Probing video metadata...")
    probe = _probe(video_path)

    # The pre-download estimate can be absent (live streams) or slightly off;
    # ffprobe on the actual file is authoritative, so re-check the limit and
    # re-plan if the real duration differs enough to change the configuration.
    _reject_if_too_long(probe["duration_sec"])
    if abs(probe["duration_sec"] - known_duration) > 30.0:
        plan = perf.plan_for(probe["duration_sec"], quality_floor=quality_floor)

    cache.set_state(video_id, "ingest", 0.75, "Extracting audio track...")
    _extract_audio(video_path, cache_dir / "audio.wav")

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
        "plan": perf.plan_to_dict(plan),
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
