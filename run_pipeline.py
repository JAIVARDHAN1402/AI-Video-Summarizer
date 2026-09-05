"""CLI entrypoint for headless runs (no Streamlit needed).

Usage:
    python run_pipeline.py --url "https://youtube.com/watch?v=..." [--force]
    python run_pipeline.py --file ./sample.mp4
    python run_pipeline.py --url "..." --profile accurate
    python run_pipeline.py --video-id abc123 --ask "What is normalization?"
    python run_pipeline.py --plan-only --url "..."
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import config
from core import perf, pipeline, qa
from core.models import VideoSummarizerError

# Mirrors the profiles offered in the Streamlit sidebar.
PROFILES = {
    "fast": {"budget": 150.0, "quality_floor": False},
    "balanced": {"budget": None, "quality_floor": False},
    "accurate": {"budget": None, "quality_floor": True},
}


def _print_progress(stage: str, progress: float, message: str) -> None:
    width = 28
    filled = int(width * min(max(progress, 0.0), 1.0))
    bar = "█" * filled + "░" * (width - filled)
    sys.stdout.write(f"\r  {bar} {progress * 100:5.1f}%  {stage:<10} {message[:52]:<52}")
    sys.stdout.flush()
    if progress >= 1.0 or stage in ("error", "done"):
        sys.stdout.write("\n")


def _print_summary(result: dict) -> None:
    meta, summary = result["meta"], result["summary"]
    duration = meta.get("duration_sec", 0.0)
    elapsed = result["elapsed_sec"]

    print(f"\n{meta['title']}")
    print(f"  {duration / 60:.1f} min video processed in {elapsed / 60:.2f} min", end="")
    if duration:
        print(f"  ({duration / elapsed:.1f}x faster than watching it)")
    else:
        print()

    print("\n  Stage timings:")
    for stage, seconds in result["timings"].items():
        if stage != "total":
            print(f"    {stage:<12} {seconds:>7.1f}s")

    print("\n  Summary:")
    for line in (summary.get("overall_summary") or "(none)").split(". "):
        if line.strip():
            print(f"    {line.strip()}.")

    print("\n  Chapters:")
    for chapter in summary.get("chapters", []):
        start = int(chapter["start"])
        print(f"    [{start // 60:02d}:{start % 60:02d}] {chapter['title']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the video summarization pipeline headlessly."
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--url", help="YouTube (or yt-dlp-supported) URL")
    source_group.add_argument("--file", help="Path to a local video file")
    source_group.add_argument("--video-id", help="Existing cached video_id (skip processing)")
    parser.add_argument("--force", action="store_true", help="Re-run all stages, ignoring cache")
    parser.add_argument(
        "--profile", choices=sorted(PROFILES), default="balanced",
        help="Speed/accuracy trade-off (default: balanced)",
    )
    parser.add_argument(
        "--budget", type=float,
        help=f"Wall-clock target in seconds (default: {config.TARGET_PIPELINE_SEC:.0f})",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Show the execution plan for the source and exit without processing",
    )
    parser.add_argument(
        "--type", default="lecture",
        choices=["lecture", "coding", "meeting", "interview", "talk", "news", "documentary", "other"],
        help="What kind of video it is — steers the explanation (default: lecture)",
    )
    parser.add_argument("--topic", default="", help="The subject, if known — sharpens extraction")
    parser.add_argument("--ask", help="Ask a question against a processed video")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of a report")
    args = parser.parse_args()

    if args.video_id:
        if not args.ask:
            print("Pass --ask with --video-id, or --url/--file to process a video.")
            return 1
        print(json.dumps(qa.answer(args.video_id, args.ask), indent=2))
        return 0

    if not args.url and not args.file:
        parser.error("one of --url, --file, or --video-id is required")

    profile = PROFILES[args.profile]
    if args.budget is not None:
        config.TARGET_PIPELINE_SEC = args.budget
    elif profile["budget"] is not None:
        config.TARGET_PIPELINE_SEC = profile["budget"]

    source_ref = args.url or args.file
    is_upload = args.file is not None

    if args.plan_only:
        preview = pipeline.preview_source(source_ref, is_upload=is_upload)
        plan = perf.plan_from_dict(preview["plan"])
        print(f"{preview['title']}  ({preview['duration_sec'] / 60:.1f} min)")
        print(f"  plan:      {plan.describe()}")
        print(f"  predicted: {json.dumps(plan.predicted_sec)}")
        for note in plan.notes:
            print(f"  note:      {note}")
        print(f"  cached:    {preview['cached']}")
        return 0

    print(f"Processing ({args.profile} profile, "
          f"{config.TARGET_PIPELINE_SEC / 60:.1f} min target)...\n")
    started = time.perf_counter()
    try:
        result = pipeline.process(
            source_ref, is_upload=is_upload, force=args.force,
            progress_cb=_print_progress, quality_floor=profile["quality_floor"],
            video_type=args.type, topic=args.topic,
        )
    except VideoSummarizerError as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result["summary"], indent=2))
    else:
        _print_summary(result)

    if args.ask:
        print("\n  Q&A:")
        answer = qa.answer(result["video_id"], args.ask)
        print(f"    Q: {args.ask}")
        print(f"    A: {answer['answer']}")
        for source in answer["sources"][:3]:
            start = int(source["start"])
            print(f"       [{start // 60:02d}:{start % 60:02d}]")

    print(f"\nDone in {time.perf_counter() - started:.1f}s. "
          f"Cached under cache/{result['video_id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
