"""Evaluate generated overall_summary text against human reference summaries.

Per BUILD_PLAN.md: SumMe/TVSum are visual-only datasets with little
meaningful speech, so they validate the keyframe module (see
run_keyframe_f1.py) only. Summarization quality must be evaluated
separately on speech-heavy videos (TED/NPTEL lectures) with human
reference summaries written by the team.

Usage:
    python eval/run_rouge.py --input eval/reference_summaries.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute ROUGE/BERTScore for generated summaries.")
    parser.add_argument(
        "--input", required=True,
        help='JSON file: [{"video_id": "...", "reference_summary": "..."}]',
    )
    parser.add_argument("--output", default="eval/results_rouge.json")
    args = parser.parse_args()

    references = json.loads(Path(args.input).read_text(encoding="utf-8"))

    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows: list[dict] = []
    candidates: list[str] = []
    refs: list[str] = []
    for item in references:
        video_id = item["video_id"]
        reference = item["reference_summary"]
        summary_data = cache.read_json(video_id, "summary")
        if summary_data is None:
            print(f"skip {video_id}: no summary.json cached (run the pipeline first)")
            continue
        candidate = summary_data["overall_summary"]
        rouge_scores = scorer.score(reference, candidate)
        rows.append({
            "video_id": video_id,
            "rouge1_f": rouge_scores["rouge1"].fmeasure,
            "rouge2_f": rouge_scores["rouge2"].fmeasure,
            "rougeL_f": rouge_scores["rougeL"].fmeasure,
        })
        candidates.append(candidate)
        refs.append(reference)

    if not rows:
        print("No videos with cached summaries found; nothing to evaluate.")
        return 1

    _, _, bert_f1 = bert_score(candidates, refs, lang="en", verbose=False)
    for row, f1 in zip(rows, bert_f1.tolist()):
        row["bertscore_f1"] = f1

    print(f"{'video_id':<18}{'ROUGE-1':>10}{'ROUGE-2':>10}{'ROUGE-L':>10}{'BERTScore':>12}")
    for row in rows:
        print(
            f"{row['video_id']:<18}{row['rouge1_f']:>10.3f}{row['rouge2_f']:>10.3f}"
            f"{row['rougeL_f']:>10.3f}{row['bertscore_f1']:>12.3f}"
        )

    avg = {
        key: sum(r[key] for r in rows) / len(rows)
        for key in ("rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1")
    }
    print("-" * 60)
    print(
        f"{'AVERAGE':<18}{avg['rouge1_f']:>10.3f}{avg['rouge2_f']:>10.3f}"
        f"{avg['rougeL_f']:>10.3f}{avg['bertscore_f1']:>12.3f}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"per_video": rows, "average": avg}, indent=2), encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
