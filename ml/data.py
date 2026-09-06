"""Prepared feature shards and the torch Dataset over them.

``prepare_shards`` does the slow work once — decode each video, run CLIP,
downsample the human scores, segment with KTS — and writes one ``.npz`` per
video under ``ml/prepared/``. Training then just memory-maps those.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ml import features as feat
from ml import kts
from ml.datasets import VideoAnnotation, load_all

logger = logging.getLogger("ml.data")

PREPARED_DIR = Path(__file__).resolve().parent / "prepared"

# KTS candidate change points: one per this many seconds of video, matching the
# SumMe/TVSum evaluation pipelines. A flat cap instead of a duration-scaled one
# makes shots far too long for the 15%-length knapsack to approximate a human
# summary, which shows up as a badly depressed F1 on the longer TVSum clips.
KTS_SECONDS_PER_CHANGE_POINT = 2.0


def kts_change_points(n_sampled: int, sample_fps: float) -> int:
    """How many change points to cut, for ~one shot per
    KTS_SECONDS_PER_CHANGE_POINT seconds. k change points give k+1 shots."""
    duration_sec = n_sampled / max(sample_fps, 1e-6)
    shots = max(int(np.ceil(duration_sec / KTS_SECONDS_PER_CHANGE_POINT)), 1)
    return max(shots - 1, 1)


def _shard_path(out_dir: Path, ann: VideoAnnotation) -> Path:
    return out_dir / f"{ann.dataset}__{ann.video_id}.npz"


def prepare_one(ann: VideoAnnotation, out_dir: Path, target_fps: float, overwrite: bool) -> Optional[Path]:
    path = _shard_path(out_dir, ann)
    if path.exists() and not overwrite:
        return path
    if not ann.video_path or not ann.video_path.exists():
        logger.warning("%s/%s: no video, skipped", ann.dataset, ann.video_id)
        return None

    fps_features, frame_idx, src_fps = feat.extract_clip_features(ann.video_path, target_fps)
    gt = ann.normalized_gt()
    scores = feat.downsample_scores(gt, frame_idx)

    users_sampled = None
    if ann.user_scores is not None and ann.user_scores.shape[0] >= len(gt):
        us = ann.user_scores[np.clip(frame_idx, 0, ann.user_scores.shape[0] - 1)]
        users_sampled = us.astype(np.float32)

    change_points = kts.cpd_fixed(
        fps_features, kts_change_points(len(fps_features), target_fps)
    )

    np.savez_compressed(
        path,
        features=fps_features.astype(np.float32),
        scores=scores.astype(np.float32),
        frame_indices=frame_idx.astype(np.int64),
        change_points=np.asarray(change_points, dtype=np.int64),
        src_fps=np.float32(src_fps),
        n_frames_full=np.int64(ann.n_frames),
        dataset=ann.dataset,
        video_id=ann.video_id,
        **({"user_scores": users_sampled} if users_sampled is not None else {}),
    )
    logger.info("prepared %s (%d sampled frames, %d shots)",
                path.name, len(scores), len(change_points) + 1)
    return path


def prepare_shards(
    data_root: Path, out_dir: Path = PREPARED_DIR, target_fps: float = feat.SAMPLE_FPS,
    which: tuple[str, ...] = ("summe", "tvsum"), overwrite: bool = False,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    anns = load_all(data_root, which)
    shards: list[Path] = []
    for ann in anns:
        try:
            p = prepare_one(ann, out_dir, target_fps, overwrite)
            if p:
                shards.append(p)
        except Exception:
            logger.exception("failed to prepare %s/%s", ann.dataset, ann.video_id)
    logger.info("prepared %d/%d shards under %s", len(shards), len(anns), out_dir)
    return shards


# --- torch Dataset ---------------------------------------------------------


class ShardDataset:
    """List of prepared shards -> (features, scores, meta) samples.

    Shards are ``.npz`` and compressed, so decoding one is not free and the
    training loop touches every shard once per epoch. The whole corpus is
    ~150 MB decompressed, so it is held in memory after first read rather than
    paying the decompression on every epoch.
    """

    def __init__(self, shard_paths: list[Path], cache: bool = True) -> None:
        self.paths = list(shard_paths)
        self._cache: dict[int, tuple] = {} if cache else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        if self._cache is not None and i in self._cache:
            return self._cache[i]

        import torch

        z = np.load(self.paths[i], allow_pickle=True)
        x = torch.from_numpy(z["features"].astype(np.float32))
        y = torch.from_numpy(z["scores"].astype(np.float32))
        meta = {
            "dataset": str(z["dataset"]),
            "video_id": str(z["video_id"]),
            "change_points": z["change_points"],
            "src_fps": float(z["src_fps"]),
            "user_scores": z["user_scores"] if "user_scores" in z.files else None,
        }
        item = (x, y, meta)
        if self._cache is not None:
            self._cache[i] = item
        return item


def collate(batch):
    import torch

    xs, ys, metas = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in xs], dtype=torch.long)
    tmax = int(lengths.max())
    dim = xs[0].size(1)
    xb = torch.zeros(len(xs), tmax, dim)
    yb = torch.zeros(len(xs), tmax)
    mb = torch.zeros(len(xs), tmax)
    for k, (x, y) in enumerate(zip(xs, ys)):
        t = x.size(0)
        xb[k, :t] = x
        yb[k, :t] = y
        mb[k, :t] = 1.0
    return xb, yb, mb, lengths, list(metas)


def load_shards(out_dir: Path = PREPARED_DIR) -> list[Path]:
    return sorted(Path(out_dir).glob("*.npz"))


def resegment_shards(out_dir: Path = PREPARED_DIR, target_fps: float = feat.SAMPLE_FPS) -> int:
    """Recompute KTS change points in place, reusing the stored CLIP features.

    Segmentation settings change more often than the features do, and the
    features are the expensive half of preparation (decode + CLIP over every
    video). This rewrites only ``change_points``.
    """
    paths = load_shards(out_dir)
    for path in paths:
        z = dict(np.load(path, allow_pickle=True))
        features = z["features"]
        z["change_points"] = np.asarray(
            kts.cpd_fixed(features, kts_change_points(len(features), target_fps)),
            dtype=np.int64,
        )
        np.savez_compressed(path, **z)
        logger.info("resegmented %s (%d frames -> %d shots)",
                    path.name, len(features), len(z["change_points"]) + 1)
    logger.info("resegmented %d shards", len(paths))
    return len(paths)
