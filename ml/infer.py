"""Runtime side of M7: load a checkpoint and score a CLIP feature sequence.

The pipeline calls :func:`score_scene_embeddings` with the per-scene keyframe
embeddings it already computed. That's a shorter, sparser sequence than the
2 fps training data, so treat the output as a *relative* importance ranking
over scenes, not a calibrated per-frame score.

Nothing here imports torch or the checkpoint until first use, and every entry
point degrades to ``None`` / uniform scores if the model file is absent — the
pipeline stays runnable with no checkpoint.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("ml.infer")

DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "importance.pt"

_lock = threading.Lock()
_model = None
_loaded_from: Optional[Path] = None


def available(ckpt_path: str | Path | None = None) -> bool:
    return Path(ckpt_path or DEFAULT_CKPT).exists()


def _get_model(ckpt_path: str | Path | None):
    global _model, _loaded_from
    path = Path(ckpt_path or DEFAULT_CKPT)
    with _lock:
        if _model is not None and _loaded_from == path:
            return _model
        if not path.exists():
            raise FileNotFoundError(f"no importance checkpoint at {path}")
        from ml.model import ImportanceScorer

        _model = ImportanceScorer.load(path)
        _loaded_from = path
        logger.info("loaded importance model from %s", path)
        return _model


def score_sequence(features: np.ndarray, ckpt_path: str | Path | None = None) -> np.ndarray:
    """features: (T, dim) L2-normalised CLIP embeddings -> (T,) importance in [0, 1]."""
    import torch

    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros(max(features.shape[0], 0), dtype=np.float32)
    model = _get_model(ckpt_path)
    with torch.no_grad():
        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
        out = model(x).squeeze(0).cpu().numpy()
    return out.astype(np.float32)


def score_scene_embeddings(
    embeddings: np.ndarray, ckpt_path: str | Path | None = None,
) -> Optional[np.ndarray]:
    """One importance value per scene, min-max normalised. ``None`` if no model."""
    if not available(ckpt_path):
        return None
    try:
        raw = score_sequence(embeddings, ckpt_path)
    except Exception:
        logger.warning("importance scoring failed; falling back to uniform", exc_info=True)
        return None
    lo, hi = float(raw.min()), float(raw.max())
    return np.full_like(raw, 0.5) if hi - lo < 1e-9 else (raw - lo) / (hi - lo)


def attach_importance(scenes_data: dict, cache_dir: str | Path, ckpt_path: str | Path | None = None) -> dict:
    """Add ``importance`` (0-1) to each scene dict and an ``importance_ranked``
    list of scene ids, highest first. No-op when the checkpoint is missing."""
    scenes = scenes_data.get("scenes") or []
    emb_file = Path(scenes_data.get("embeddings_file") or (Path(cache_dir) / "clip_embeddings.npy"))
    if not scenes or not emb_file.exists() or not available(ckpt_path):
        return scenes_data

    embeddings = np.load(emb_file).astype(np.float32)
    idx = [s.get("clip_embedding_idx", i) for i, s in enumerate(scenes)]
    seq = embeddings[np.clip(idx, 0, len(embeddings) - 1)]
    scores = score_scene_embeddings(seq, ckpt_path)
    if scores is None:
        return scenes_data

    for s, v in zip(scenes, scores):
        s["importance"] = round(float(v), 4)
    scenes_data["importance_ranked"] = [
        s["id"] for s in sorted(scenes, key=lambda s: s.get("importance", 0.0), reverse=True)
    ]
    scenes_data["importance_model"] = str(Path(ckpt_path or DEFAULT_CKPT).name)
    return scenes_data
