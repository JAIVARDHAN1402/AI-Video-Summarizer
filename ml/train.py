"""5-fold cross-validated training for the importance model.

    python -m ml.train --data-root data/highlight          # full run
    python -m ml.train --smoke                              # 1 epoch, tiny

Splits prepared shards into 5 video-disjoint folds; for each fold trains an
ImportanceScorer on the other four and evaluates on the held-out one. Saves
the best checkpoint per fold to ml/checkpoints/ and writes a metrics summary.
The fold-0 checkpoint is copied to ml/checkpoints/importance.pt as the default
the runtime picks up.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np

from ml import data as mdata
from ml import evaluate as mev
from ml.model import ImportanceScorer, ModelConfig, masked_ranking_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
logger = logging.getLogger("ml.train")

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _folds(paths: list[Path], k: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    return [order[i::k].tolist() for i in range(k)]


def _run_epoch(model, loader, opt, device, train: bool):
    import torch
    import torch.nn.functional as F

    model.train(train)
    total = 0.0
    for xb, yb, mb, lengths, _ in loader:
        xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
        with torch.set_grad_enabled(train):
            pred = model(xb, lengths)
            mse = (F.mse_loss(pred, yb, reduction="none") * mb).sum() / mb.sum().clamp(min=1)
            rank = masked_ranking_loss(pred, yb, mb)
            loss = mse + 0.5 * rank
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
        total += float(loss) * xb.size(0)
    return total / max(len(loader.dataset), 1)


def _eval_fold(model, ds, device, with_f1: bool = True) -> tuple[dict, list[dict]]:
    import torch

    model.eval()
    rows: list[dict] = []
    for i in range(len(ds)):
        x, y, meta = ds[i]
        with torch.no_grad():
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
        rows.append(mev.evaluate_video(pred, meta, y.numpy(), with_f1=with_f1))
    return mev.aggregate(rows), rows


def train_cv(
    shard_paths: list[Path], k: int = 5, epochs: int = 60, lr: float = 5e-4,
    batch_size: int = 4, seed: int = 0, patience: int = 12, device: str = "cpu",
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    input_dim = int(np.load(shard_paths[0])["features"].shape[1])
    folds = _folds(shard_paths, k, seed)
    fold_reports = []

    for fi in range(k):
        val_idx = set(folds[fi])
        train_paths = [p for j, p in enumerate(shard_paths) if j not in val_idx]
        val_paths = [shard_paths[j] for j in folds[fi]]
        train_ds = mdata.ShardDataset(train_paths)
        val_ds = mdata.ShardDataset(val_paths)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=mdata.collate,
        )

        model = ImportanceScorer(ModelConfig(input_dim=input_dim)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_metric, best_state, since_best = -1e9, None, 0
        for ep in range(epochs):
            tr_loss = _run_epoch(model, train_loader, opt, device, train=True)
            sched.step()
            # Selection is on the per-annotator tau — the metric actually
            # reported — so skip the knapsack F1 here; it is the expensive half
            # of a validation pass and nothing reads it until the fold ends.
            agg, _ = _eval_fold(model, val_ds, device, with_f1=False)
            metric = agg["kendall_tau"] or 0.0
            if metric > best_metric:
                best_metric, best_state, since_best = metric, {k_: v.cpu().clone() for k_, v in model.state_dict().items()}, 0
            else:
                since_best += 1
            logger.info("fold %d ep %2d  loss %.4f  val tau %.4f  rho %.4f",
                        fi, ep, tr_loss, agg["kendall_tau"], agg["spearman_rho"])
            if since_best >= patience:
                logger.info("fold %d early stop at ep %d", fi, ep)
                break

        model.load_state_dict(best_state)
        agg, rows = _eval_fold(model, val_ds, device, with_f1=True)
        model.save(CKPT_DIR / f"fold{fi}.pt")
        fold_reports.append({"fold": fi, "metrics": agg, "videos": rows})
        logger.info("fold %d DONE  tau %.4f  rho %.4f  f1 %s",
                    fi, agg["kendall_tau"], agg["spearman_rho"],
                    f"{agg['f1']:.4f}" if agg["f1"] is not None else "n/a")

    summary = _summarize(fold_reports)
    (CKPT_DIR / "cv_report.json").write_text(json.dumps({"summary": summary, "folds": fold_reports}, indent=2))
    shutil.copyfile(CKPT_DIR / "fold0.pt", CKPT_DIR / "importance.pt")
    logger.info("CV done: tau %.4f  rho %.4f  f1 %s  -> %s",
                summary["kendall_tau"], summary["spearman_rho"],
                f"{summary['f1']:.4f}" if summary["f1"] is not None else "n/a",
                CKPT_DIR / "importance.pt")
    return summary


def _summarize(fold_reports: list[dict]) -> dict:
    keys = ["kendall_tau", "spearman_rho", "f1",
            "kendall_tau_vs_mean", "spearman_rho_vs_mean"]
    out = {}
    for key in keys:
        vals = [f["metrics"][key] for f in fold_reports if f["metrics"].get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
        out[f"{key}_std"] = float(np.std(vals)) if vals else None
    return out


def _smoke() -> None:
    """Plumbing check on synthetic shards — no dataset needed."""
    import tempfile

    from ml.kts import cpd_auto

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    for v in range(6):
        n = rng.integers(120, 200)
        feats = rng.standard_normal((n, 32)).astype(np.float32)
        scores = np.clip(feats[:, 0] * 0.3 + 0.5 + rng.normal(0, 0.1, n), 0, 1).astype(np.float32)
        cps = cpd_auto(feats, 10)
        users = np.stack([(scores + rng.normal(0, 0.2, n)) for _ in range(5)], axis=1).astype(np.float32)
        np.savez_compressed(
            tmp / f"synthetic__v{v}.npz", features=feats, scores=scores,
            frame_indices=np.arange(n), change_points=cps, src_fps=np.float32(30),
            n_frames_full=np.int64(n), dataset="synthetic", video_id=f"v{v}",
            user_scores=users,
        )
    summary = train_cv(sorted(tmp.glob("*.npz")), k=3, epochs=3, patience=3)
    print(json.dumps(summary, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, help="datasets root (only needed if shards aren't prepared)")
    ap.add_argument("--prepared", type=Path, default=mdata.PREPARED_DIR)
    ap.add_argument("--prepare", action="store_true", help="(re)build feature shards before training")
    ap.add_argument("--resegment", action="store_true",
                    help="recompute KTS change points in existing shards (keeps the CLIP features)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return 0

    if args.prepare:
        if not args.data_root:
            ap.error("--prepare needs --data-root")
        mdata.prepare_shards(args.data_root, args.prepared)
    elif args.resegment:
        mdata.resegment_shards(args.prepared)

    shards = mdata.load_shards(args.prepared)
    if len(shards) < args.folds:
        ap.error(f"only {len(shards)} shards under {args.prepared}; run with --prepare --data-root ...")
    train_cv(shards, k=args.folds, epochs=args.epochs, lr=args.lr,
             batch_size=args.batch_size, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
