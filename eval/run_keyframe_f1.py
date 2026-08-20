"""Evaluate keyframe/scene selection against SumMe/TVSum-style ground truth.

Ground-truth format (produced during dataset prep — see BUILD_PLAN.md eval
notes): a JSON dict mapping video_id -> list of [start_sec, end_sec]
intervals marked "important" by human annotators.

    {"abcdef0123456789": [[12.0, 34.5], [90.0, 110.0]], ...}

A detected scene counts as a hit if its window overlaps any ground-truth
interval; a ground-truth interval counts as covered if any detected scene
overlaps it. This is the standard proxy for keyframe precision/recall used
when full per-frame importance scores aren't available.

Usage:
    python eval/run_keyframe_f1.py --gt eval/ground_truth.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cache


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _score_video(scenes: list[dict], gt_intervals: list[list[float]]) -> dict:
    if not scenes or not gt_intervals:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "n_scenes": len(scenes), "n_gt": len(gt_intervals),
        }

    hits = sum(
        1 for s in scenes
        if any(_overlaps(s["start"], s["end"], g[0], g[1]) for g in gt_intervals)
    )
    covered = sum(
        1 for g in gt_intervals
        if any(_overlaps(s["start"], s["end"], g[0], g[1]) for s in scenes)
    )

    precision = hits / len(scenes)
    recall = covered / len(gt_intervals)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "n_scenes": len(scenes), "n_gt": len(gt_intervals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate keyframe selection against ground truth.")
    parser.add_argument("--gt", required=True, help='JSON: {"video_id": [[start,end], ...]}')
    parser.add_argument("--output", default="eval/results_keyframe_f1.json")
    args = parser.parse_args()

    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))

    rows: list[dict] = []
    for video_id, gt_intervals in gt.items():
        scenes_data = cache.read_json(video_id, "scenes")
        if scenes_data is None:
            print(f"skip {video_id}: no scenes.json cached (run the pipeline first)")
            continue
        result = _score_video(scenes_data["scenes"], gt_intervals)
        result["video_id"] = video_id
        rows.append(result)

    if not rows:
        print("No videos with cached scenes found; nothing to evaluate.")
        return 1

    print(f"{'video_id':<18}{'Precision':>10}{'Recall':>10}{'F1':>10}")
    for row in rows:
        print(f"{row['video_id']:<18}{row['precision']:>10.3f}{row['recall']:>10.3f}{row['f1']:>10.3f}")

    avg_f1 = sum(r["f1"] for r in rows) / len(rows)
    print("-" * 48)
    print(f"Average F1: {avg_f1:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"per_video": rows, "average_f1": avg_f1}, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
