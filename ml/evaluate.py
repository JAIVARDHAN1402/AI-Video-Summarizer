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
    """Correlation against the *averaged* annotator score.

    This is the model's own training target, so it is a low-variance signal for
    early stopping — but averaging smooths away the disagreement between
    annotators, which makes it far easier than the published metric. Do not
    compare it to numbers in the literature; use
    :func:`rank_correlation_per_user` for that.
    """
    from scipy.stats import kendalltau, spearmanr

    n = min(len(pred_scores), len(mean_user_scores))
    p, g = pred_scores[:n], mean_user_scores[:n]
    if np.std(p) < 1e-9 or np.std(g) < 1e-9:
        return 0.0, 0.0
    tau = kendalltau(p, g).correlation
    rho = spearmanr(p, g).correlation
    return float(np.nan_to_num(tau)), float(np.nan_to_num(rho))


def rank_correlation_per_user(
    pred_scores: np.ndarray, user_scores: np.ndarray,
) -> tuple[float, float]:
    """Mean correlation against each annotator separately (Otani et al., CVPR 2019).

    The comparable metric: correlate the prediction with every annotator's own
    scores and average the coefficients. Much harder than correlating with
    their mean — on TVSum, human-vs-human agreement under this protocol is
    only about 0.18 Kendall tau.
    """
    from scipy.stats import kendalltau, spearmanr

    n = min(len(pred_scores), user_scores.shape[0])
    p = pred_scores[:n]
    if np.std(p) < 1e-9:
        return 0.0, 0.0
    taus, rhos = [], []
    for u in range(user_scores.shape[1]):
        g = user_scores[:n, u].astype(np.float64)
        if np.std(g) < 1e-9:
            continue
        taus.append(np.nan_to_num(kendalltau(p, g).correlation))
        rhos.append(np.nan_to_num(spearmanr(p, g).correlation))
    if not taus:
        return 0.0, 0.0
    return float(np.mean(taus)), float(np.mean(rhos))


def evaluate_video(
    pred_scores: np.ndarray, meta: dict, gt_scores: np.ndarray, with_f1: bool = True,
) -> dict:
    """``with_f1=False`` skips the knapsack summary, which runs once per
    annotator (20 of them on TVSum) and dominates a per-epoch validation pass.
    Early stopping selects on Kendall tau, so F1 is only needed for reporting.
    """
    out = {"video_id": meta["video_id"], "dataset": meta["dataset"]}
    tau, rho = rank_correlation(pred_scores, gt_scores)
    out["kendall_tau_vs_mean"] = tau
    out["spearman_rho_vs_mean"] = rho
    us = meta.get("user_scores")
    if us is not None:
        us = np.asarray(us)
        # The comparable numbers; reported under the plain names.
        ptau, prho = rank_correlation_per_user(pred_scores, us)
        out["kendall_tau"] = ptau
        out["spearman_rho"] = prho
        if with_f1:
            out["f1"] = summary_f1(pred_scores, meta["change_points"], us, meta["dataset"])
    else:
        out["kendall_tau"], out["spearman_rho"] = tau, rho
    return out


def reference_baselines(prepared_dir: str, seed: int = 0) -> dict:
    """Random, human and oracle F1 measured inside *this* pipeline.

    Absolute F1 is not portable between summarisation pipelines — it depends on
    the feature extractor, the frame sampling rate and the shot segmentation,
    all of which differ from the published setups (GoogleNet pool5, their KTS).
    Quoting a paper's random-baseline number against ours is meaningless. These
    are the yardsticks that *are* meaningful, computed on the same shards the
    model is scored on:

    - ``random``: uniform noise as the frame scores, the floor.
    - ``human``: each annotator scored against the *others* (leave-one-out),
      the realistic ceiling for agreement on a subjective task.
    - ``oracle_mean``: the averaged annotator score — the model's own training
      target, so the ceiling this architecture could approach.
    """
    from ml import data as mdata

    rng = np.random.default_rng(seed)
    ds = mdata.ShardDataset(mdata.load_shards(prepared_dir))
    acc: dict[str, dict[str, list]] = {}

    for i in range(len(ds)):
        _, y, meta = ds[i]
        us = meta.get("user_scores")
        if us is None:
            continue
        us = np.asarray(us)
        n, n_users = len(y), us.shape[1]
        dataset, cps = meta["dataset"], meta["change_points"]
        bucket = acc.setdefault(dataset, {"random": [], "human": [], "oracle_mean": []})

        bucket["random"].append(summary_f1(rng.random(n), cps, us, dataset))
        bucket["oracle_mean"].append(summary_f1(y.numpy(), cps, us, dataset))
        if n_users > 1:
            # Leave-one-out: an annotator must not be scored against itself.
            per_user = [
                summary_f1(us[:, u], cps, np.delete(us, u, axis=1), dataset)
                for u in range(n_users)
            ]
            bucket["human"].append(float(np.mean(per_user)))

    return {
        ds_: {k: (float(np.mean(v)) if v else None) for k, v in b.items()}
        for ds_, b in sorted(acc.items())
    }


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
    ap.add_argument("--baselines", action="store_true",
                    help="also report random / human / oracle F1 in this pipeline")
    args = ap.parse_args()

    result = score_checkpoint(args.checkpoint, args.prepared)
    if args.baselines:
        result["baselines"] = reference_baselines(args.prepared)
    s = result["summary"]
    print(f"videos: {s['n_videos']}")
    print(f"F1:          {s['f1']}")
    print(f"Kendall tau: {s['kendall_tau']:.4f}")
    print(f"Spearman rho:{s['spearman_rho']:.4f}")
    for ds, m in s["by_dataset"].items():
        print(f"  {ds:8} n={m['n']:<3} f1={m['f1']} tau={m['kendall_tau']:.4f} rho={m['spearman_rho']:.4f}")
    if "baselines" in result:
        print("\nF1 against baselines measured in this same pipeline:")
        print(f"  {'dataset':<8} {'random':>8} {'model':>8} {'human':>8} {'oracle':>8}")
        for ds, b in result["baselines"].items():
            model = (s["by_dataset"].get(ds) or {}).get("f1")
            fmt = lambda v: f"{v:>8.4f}" if v is not None else f"{'n/a':>8}"
            print(f"  {ds:<8}{fmt(b['random'])}{fmt(model)}{fmt(b['human'])}{fmt(b['oracle_mean'])}")
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.output}")
    return 0


def aggregate(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else None

    keys = ("f1", "kendall_tau", "spearman_rho", "kendall_tau_vs_mean", "spearman_rho_vs_mean")
    by_ds: dict[str, dict] = {}
    for ds in {r["dataset"] for r in rows}:
        sub = [r for r in rows if r["dataset"] == ds]
        by_ds[ds] = {"n": len(sub)}
        for k in keys:
            vals = [r[k] for r in sub if k in r]
            by_ds[ds][k] = float(np.mean(vals)) if vals else None
    return {
        "n_videos": len(rows),
        **{k: mean(k) for k in keys},
        "by_dataset": by_ds,
    }


if __name__ == "__main__":
    raise SystemExit(_main())
