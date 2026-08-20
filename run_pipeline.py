"""CLI entrypoint for headless testing of the pipeline (no Streamlit needed).

Usage:
    python run_pipeline.py --url "https://youtube.com/watch?v=..." [--force]
    python run_pipeline.py --file ./sample.mp4
    python run_pipeline.py --video-id abc123 --ask "What is normalization?"
"""

from __future__ import annotations

import argparse
import json
import sys

from core import pipeline, qa
from core.models import VideoSummarizerError


def _print_progress(stage: str, progress: float, message: str) -> None:
    bar_len = 30
    filled = int(bar_len * progress)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r[{bar}] {progress * 100:5.1f}%  {stage:<10} {message}", end="", flush=True)
    if progress >= 1.0 or stage == "error":
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the video summarization pipeline headlessly.")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", help="YouTube (or yt-dlp-supported) URL")
    source_group.add_argument("--file", help="Path to a local video file to process")
    source_group.add_argument("--video-id", help="Existing cached video_id (skip processing)")
    parser.add_argument("--force", action="store_true", help="Re-run all stages, ignoring cache")
    parser.add_argument("--ask", help="Ask a question against an already-processed video")
    args = parser.parse_args()

    if args.video_id:
        if args.ask:
            print(json.dumps(qa.answer(args.video_id, args.ask), indent=2))
            return 0
        print("Nothing to do: pass --ask with --video-id, or provide --url/--file to process a video.")
        return 1

    if not args.url and not args.file:
        parser.error("one of --url, --file, or --video-id is required")

    try:
        if args.url:
            result = pipeline.process(
                args.url, is_upload=False, force=args.force, progress_cb=_print_progress
            )
        else:
            result = pipeline.process(
                args.file, is_upload=True, force=args.force, progress_cb=_print_progress
            )
    except VideoSummarizerError as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Summary ===")
    print(json.dumps(result["summary"], indent=2))

    if args.ask:
        print("\n=== Q&A ===")
        print(json.dumps(qa.answer(result["video_id"], args.ask), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
