"""SumMe/TVSum metrics: knapsack-summary F1 and rank correlation.

F1 is the historical metric — build a <=15%-length summary by 0/1 knapsack over
KTS shots, compare to the human summaries (max over users for SumMe, mean for
TVSum). Kendall's tau / Spearman's rho (Otani et al., CVPR 2019) compare the
predicted frame ranking directly to the mean human ranking and don't depend on
the segmentation, so they're the more reliable signal on this little data.
"""

from __future__ import annotations

import numpy as np

from ml.kts import segment_bounds

SUMMARY_RATIO = 0.15


def shots_from_change_points(n: int, change_points: np.ndarray) -> list[tuple[int, int]]:
    return segment_bounds(n, np.asarray(change_points, dtype=np.int64))


def knapsack_summary(frame_scores: np.ndarray, shots: list[tuple[int, int]], ratio: float = SUMMARY_RATIO) -> np.ndarray:
    """0/1 knapsack over shots -> binary frame mask, total length <= ratio * n."""
    n = len(frame_scores)
    budget = int(round(ratio * n))
    values = np.array([frame_scores[a:b].sum() for a, b in shots], dtype=np.float64)
    weights = np.array([b - a for a, b in shots], dtype=np.int64)

    m = len(shots)
    dp = np.zeros((m + 1, budget + 1), dtype=np.float64)
    for i in range(1, m + 1):
        w = int(weights[i - 1])
        v = values[i - 1]
        dp[i] = dp[i - 1]
        if w <= budget:
            take = dp[i - 1, : budget + 1 - w] + v
            better = take > dp[i - 1, w:]
            dp[i, w:][better] = take[better]

    chosen = np.zeros(m, dtype=bool)
    c = budget
    for i in range(m, 0, -1):
        if c - weights[i - 1] >= 0 and dp[i, c] != dp[i - 1, c]:
            chosen[i - 1] = True
            c -= int(weights[i - 1])

    mask = np.zeros(n, dtype=bool)
    for k, (a, b) in enumerate(shots):
        if chosen[k]:
            mask[a:b] = True
    return mask


def _f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    tp = np.logical_and(pred_mask, gt_mask).sum()
    if tp == 0:
        return 0.0
    prec = tp / pred_mask.sum()
    rec = tp / gt_mask.sum()
    return float(2 * prec * rec / (prec + rec))


def summary_f1(
    pred_scores: np.ndarray, change_points: np.ndarray, user_scores: np.ndarray, dataset: str,
) -> float:
    """user_scores: (n_frames, n_users). SumMe -> max over users, TVSum -> mean."""
    n = len(pred_scores)
    shots = shots_from_change_points(n, change_points)
    pred_mask = knapsack_summary(pred_scores, shots)

    per_user = []
    for u in range(user_scores.shape[1]):
        us = user_scores[:n, u].astype(np.float64)
        if set(np.unique(us)).issubset({0.0, 1.0}):
            gt_mask = us.astype(bool)
        else:
            gt_mask = knapsack_summary(us, shots)
        if gt_mask.sum() == 0:
            continue
        per_user.append(_f1(pred_mask, gt_mask))
    if not per_user:
        return 0.0
    return max(per_user) if dataset == "summe" else float(np.mean(per_user))


def rank_correlation(pred_scores: np.ndarray, mean_user_scores: np.ndarray) -> tuple[float, float]:
    from scipy.stats import kendalltau, spearmanr

    n = min(len(pred_scores), len(mean_user_scores))
    p, g = pred_scores[:n], mean_user_scores[:n]
    if np.std(p) < 1e-9 or np.std(g) < 1e-9:
        return 0.0, 0.0
    tau = kendalltau(p, g).correlation
    rho = spearmanr(p, g).correlation
    return float(np.nan_to_num(tau)), float(np.nan_to_num(rho))


def evaluate_video(pred_scores: np.ndarray, meta: dict, gt_scores: np.ndarray) -> dict:
    out = {"video_id": meta["video_id"], "dataset": meta["dataset"]}
    tau, rho = rank_correlation(pred_scores, gt_scores)
    out["kendall_tau"] = tau
    out["spearman_rho"] = rho
    us = meta.get("user_scores")
    if us is not None:
        out["f1"] = summary_f1(pred_scores, meta["change_points"], np.asarray(us), meta["dataset"])
    return out


def score_checkpoint(checkpoint: str, prepared_dir: str) -> dict:
    """Run a saved model over every prepared shard and aggregate the metrics."""
    import torch

    from ml import data as mdata
    from ml.model import ImportanceScorer

    model = ImportanceScorer.load(checkpoint)
    ds = mdata.ShardDataset(mdata.load_shards(prepared_dir))
    if len(ds) == 0:
        raise FileNotFoundError(f"no prepared shards under {prepared_dir}")
    rows: list[dict] = []
    for i in range(len(ds)):
        x, y, meta = ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0)).squeeze(0).cpu().numpy()
        rows.append(evaluate_video(pred, meta, y.numpy()))
    return {"summary": aggregate(rows), "per_video": rows}


def _main() -> int:
    import argparse
    import json
    from pathlib import Path

    from ml.data import PREPARED_DIR

    ap = argparse.ArgumentParser(description="Evaluate an importance checkpoint on prepared shards.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prepared", default=str(PREPARED_DIR))
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    result = score_checkpoint(args.checkpoint, args.prepared)
    s = result["summary"]
    print(f"videos: {s['n_videos']}")
    print(f"F1:          {s['f1']}")
    print(f"Kendall tau: {s['kendall_tau']:.4f}")
    print(f"Spearman rho:{s['spearman_rho']:.4f}")
    for ds, m in s["by_dataset"].items():
        print(f"  {ds:8} n={m['n']:<3} f1={m['f1']} tau={m['kendall_tau']:.4f} rho={m['spearman_rho']:.4f}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.output}")
    return 0


def aggregate(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else None

    by_ds: dict[str, dict] = {}
    for ds in {r["dataset"] for r in rows}:
        sub = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {
            "n": len(sub),
            "f1": float(np.mean([r["f1"] for r in sub if "f1" in r])) if any("f1" in r for r in sub) else None,
            "kendall_tau": float(np.mean([r["kendall_tau"] for r in sub])),
            "spearman_rho": float(np.mean([r["spearman_rho"] for r in sub])),
        }
    return {
        "n_videos": len(rows),
        "f1": mean("f1"),
        "kendall_tau": mean("kendall_tau"),
        "spearman_rho": mean("spearman_rho"),
        "by_dataset": by_ds,
    }


if __name__ == "__main__":
    raise SystemExit(_main())
