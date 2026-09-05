"""Video file -> a sampled CLIP feature sequence.

Uses the same CLIP ViT-B/32 the pipeline already loads (``core.vision._get_clip``)
so the training features and the runtime features are identical. Frames are
sampled at a fixed rate (2 fps by default, the SumMe/TVSum convention) and
L2-normalised, matching ``core.vision._run_clip_batch``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ml.features")

SAMPLE_FPS = 2.0
CLIP_BATCH = 64


def sample_frame_indices(n_frames: int, src_fps: float, target_fps: float = SAMPLE_FPS) -> np.ndarray:
    """Evenly spaced frame indices at ``target_fps`` over a video of ``n_frames``."""
    if n_frames <= 0 or src_fps <= 0:
        return np.array([], dtype=np.int64)
    duration = n_frames / src_fps
    count = max(int(round(duration * target_fps)), 1)
    return np.linspace(0, n_frames - 1, count).round().astype(np.int64)


def extract_clip_features(
    video_path: str | Path, target_fps: float = SAMPLE_FPS, batch_size: int = CLIP_BATCH,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (features (N, 512) float32, frame_indices (N,), src_fps).

    Decodes with OpenCV, preprocesses with the CLIP transform, encodes in
    batches. CPU-only friendly: a 3-minute clip at 2 fps is ~360 frames.
    """
    import cv2
    import torch
    from PIL import Image

    from core.vision import _get_clip

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    wanted = set(int(i) for i in sample_frame_indices(n_frames, src_fps, target_fps))
    if not wanted:
        cap.release()
        raise RuntimeError(f"no frames sampled from {video_path} (n_frames={n_frames}, fps={src_fps})")

    model, preprocess = _get_clip()
    kept_idx: list[int] = []
    tensors: list[torch.Tensor] = []
    feats: list[np.ndarray] = []

    def flush() -> None:
        if not tensors:
            return
        with torch.no_grad():
            batch = torch.stack(tensors)
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy().astype(np.float32))
        tensors.clear()

    pos = 0
    ok, frame = cap.read()
    while ok:
        if pos in wanted:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(preprocess(Image.fromarray(rgb)))
            kept_idx.append(pos)
            if len(tensors) >= batch_size:
                flush()
        pos += 1
        ok, frame = cap.read()
    flush()
    cap.release()

    if not feats:
        raise RuntimeError(f"decoded no sampled frames from {video_path}")
    features = np.concatenate(feats, axis=0)
    logger.info("%s: %d frames @ %.1f fps -> %s", Path(video_path).name,
                features.shape[0], target_fps, features.shape)
    return features, np.asarray(kept_idx, dtype=np.int64), float(src_fps)


def downsample_scores(gt_score: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    """Per-frame ground truth -> one score per sampled frame (nearest index)."""
    idx = np.clip(frame_indices, 0, len(gt_score) - 1)
    return gt_score[idx].astype(np.float32)
