"""Evaluate the Q&A module against eval/qa_testset.json.

Scores:
  (a) answer similarity — cosine similarity between the generated answer and
      the expected_answer, via sentence-transformers embeddings.
  (b) timestamp hit rate — whether any returned source overlaps
      expected_timestamp_range.

Usage:
    python eval/run_qa_eval.py --testset eval/qa_testset.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from core import qa


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Q&A answer quality and timestamp accuracy.")
    parser.add_argument("--testset", default="eval/qa_testset.json")
    parser.add_argument("--output", default="eval/results_qa_eval.json")
    args = parser.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    testset = [t for t in testset if t.get("video_id") and t.get("question")]

    if not testset:
        print("eval/qa_testset.json has no filled-in entries yet; fill it in before evaluating.")
        return 1

    from sentence_transformers import SentenceTransformer, util

    embedder = SentenceTransformer(config.EMBEDDING_MODEL)

    rows: list[dict] = []
    for item in testset:
        result = qa.answer(item["video_id"], item["question"])

        similarity = None
        if item.get("expected_answer"):
            emb_a = embedder.encode(result["answer"], normalize_embeddings=True)
            emb_b = embedder.encode(item["expected_answer"], normalize_embeddings=True)
            similarity = float(util.cos_sim(emb_a, emb_b).item())

        gt_start, gt_end = item.get("expected_timestamp_range", [0.0, 0.0])
        timestamp_hit = any(
            _overlaps(src["start"], src["end"], gt_start, gt_end) for src in result["sources"]
        )

        rows.append({
            "video_id": item["video_id"],
            "question": item["question"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "answer_similarity": similarity,
            "timestamp_hit": timestamp_hit,
        })

    similarities = [r["answer_similarity"] for r in rows if r["answer_similarity"] is not None]
    avg_similarity = sum(similarities) / len(similarities) if similarities else None
    hit_rate = sum(1 for r in rows if r["timestamp_hit"]) / len(rows)

    print(f"Evaluated {len(rows)} questions")
    if avg_similarity is not None:
        print(f"Average answer similarity: {avg_similarity:.3f}")
    print(f"Timestamp hit rate: {hit_rate:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "per_question": rows,
                "average_answer_similarity": avg_similarity,
                "timestamp_hit_rate": hit_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
