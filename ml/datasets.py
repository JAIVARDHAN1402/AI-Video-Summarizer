"""Load SumMe and TVSum annotations into one shape the trainer understands.

Both datasets ship frame-level human importance scores; only the packaging
differs. SumMe is one ``.mat`` per video under ``GT/``; TVSum is a single
``ydata-tvsum50.mat`` (with a ``.tsv`` fallback). Point ``--data-root`` at a
directory laid out as::

    <root>/SumMe/GT/*.mat
    <root>/SumMe/videos/*.{mp4,webm,avi,mpg}
    <root>/TVSum/ydata-tvsum50.mat
    <root>/TVSum/video/*.mp4              # or ydata-tvsum50-anno.tsv + info.tsv

``scripts/fetch_datasets.py`` creates that layout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

logger = logging.getLogger("ml.datasets")

_VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mpg", ".mpeg", ".mkv", ".m4v")


@dataclass
class VideoAnnotation:
    dataset: str                       # "summe" | "tvsum"
    video_id: str                      # unique within (dataset)
    n_frames: int
    fps: float
    gt_score: np.ndarray               # (n_frames,) float, mean human importance
    user_scores: Optional[np.ndarray]  # (n_frames, n_users) float, or None
    video_path: Optional[Path]

    @property
    def duration_sec(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def normalized_gt(self) -> np.ndarray:
        """gt_score rescaled to [0, 1] per video (min-max, safe on constants)."""
        g = self.gt_score.astype(np.float64)
        lo, hi = float(g.min()), float(g.max())
        return np.zeros_like(g) if hi - lo < 1e-9 else (g - lo) / (hi - lo)


# --- SumMe -----------------------------------------------------------------


def _find_video(video_dir: Path, stem: str) -> Optional[Path]:
    if not video_dir.is_dir():
        return None
    for ext in _VIDEO_EXTS:
        cand = video_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # names sometimes differ only by spaces/underscores
    norm = stem.lower().replace(" ", "").replace("_", "")
    for p in video_dir.iterdir():
        if p.suffix.lower() in _VIDEO_EXTS and p.stem.lower().replace(" ", "").replace("_", "") == norm:
            return p
    return None


def load_summe(summe_root: Path) -> list[VideoAnnotation]:
    from scipy.io import loadmat

    gt_dir = summe_root / "GT"
    if not gt_dir.is_dir():
        gt_dir = summe_root  # some archives drop the .mat files at the top level
    mats = sorted(gt_dir.glob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"no SumMe .mat files under {summe_root}")

    video_dir = summe_root / "videos"
    out: list[VideoAnnotation] = []
    for mat_path in mats:
        m = loadmat(str(mat_path))
        gt = np.asarray(m["gt_score"], dtype=np.float64).squeeze()
        users = m.get("user_score")
        users_arr = np.asarray(users, dtype=np.float64) if users is not None else None
        if users_arr is not None and users_arr.ndim == 1:
            users_arr = users_arr[:, None]
        n_frames = int(m["nFrames"].squeeze()) if "nFrames" in m else gt.shape[0]
        fps = float(m["FPS"].squeeze()) if "FPS" in m else (
            gt.shape[0] / float(m["video_duration"].squeeze())
            if "video_duration" in m else 30.0
        )
        stem = mat_path.stem
        out.append(VideoAnnotation(
            dataset="summe", video_id=stem, n_frames=n_frames, fps=fps,
            gt_score=gt, user_scores=users_arr,
            video_path=_find_video(video_dir, stem),
        ))
    logger.info("SumMe: %d videos (%d with video files)",
                len(out), sum(a.video_path is not None for a in out))
    return out


# --- TVSum ---------------------------------------------------------------


def _load_tvsum_mat(mat_path: Path, video_dir: Path) -> list[VideoAnnotation]:
    from scipy.io import loadmat

    m = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    entries = m[key]
    entries = [entries] if not isinstance(entries, np.ndarray) else list(entries)

    out: list[VideoAnnotation] = []
    for e in entries:
        vid = str(e.video)
        user = np.asarray(e.user_anno, dtype=np.float64)   # (n_frames, n_users)
        if user.ndim == 1:
            user = user[:, None]
        gt = (np.asarray(e.gt_score, dtype=np.float64).squeeze()
              if hasattr(e, "gt_score") else user.mean(axis=1))
        n_frames = int(getattr(e, "nframes", gt.shape[0]))
        fps = float(getattr(e, "fps", 30.0))
        out.append(VideoAnnotation(
            dataset="tvsum", video_id=vid, n_frames=n_frames, fps=fps,
            gt_score=gt, user_scores=user,
            video_path=_find_video(video_dir, vid),
        ))
    logger.info("TVSum(.mat): %d videos (%d with video files)",
                len(out), sum(a.video_path is not None for a in out))
    return out


def _load_tvsum_tsv(tvsum_root: Path, video_dir: Path) -> list[VideoAnnotation]:
    data_dir = tvsum_root / "data" if (tvsum_root / "data").is_dir() else tvsum_root
    anno = next(data_dir.glob("*anno.tsv"))
    info = next(data_dir.glob("*info.tsv"), None)

    fps_by_id: dict[str, float] = {}
    nframes_by_id: dict[str, int] = {}
    if info and info.exists():
        for line in info.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) >= 7:
                fps_by_id[parts[0]] = float(parts[6])
                nframes_by_id[parts[0]] = int(parts[5])

    per_video: dict[str, list[np.ndarray]] = {}
    for line in anno.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        vid, _cat, scores = parts[0], parts[1], parts[2]
        arr = np.array([float(x) for x in scores.split(",")], dtype=np.float64)
        per_video.setdefault(vid, []).append(arr)

    out: list[VideoAnnotation] = []
    for vid, rows in per_video.items():
        width = min(len(r) for r in rows)
        user = np.stack([r[:width] for r in rows], axis=1)  # (n_frames, n_users)
        out.append(VideoAnnotation(
            dataset="tvsum", video_id=vid,
            n_frames=nframes_by_id.get(vid, width),
            fps=fps_by_id.get(vid, 30.0),
            gt_score=user.mean(axis=1), user_scores=user,
            video_path=_find_video(video_dir, vid),
        ))
    logger.info("TVSum(.tsv): %d videos (%d with video files)",
                len(out), sum(a.video_path is not None for a in out))
    return out


def load_tvsum(tvsum_root: Path) -> list[VideoAnnotation]:
    video_dir = tvsum_root / "video"
    if not video_dir.is_dir():
        video_dir = tvsum_root / "videos"
    mat = next(tvsum_root.glob("**/ydata-tvsum50*.mat"), None) or next(tvsum_root.glob("*.mat"), None)
    if mat is not None:
        try:
            return _load_tvsum_mat(mat, video_dir)
        except Exception:
            logger.warning("TVSum .mat parse failed, trying .tsv", exc_info=True)
    return _load_tvsum_tsv(tvsum_root, video_dir)


# --- combined ----------------------------------------------------------------


def load_all(data_root: Path, which: tuple[str, ...] = ("summe", "tvsum")) -> list[VideoAnnotation]:
    data_root = Path(data_root)
    anns: list[VideoAnnotation] = []
    if "summe" in which and (data_root / "SumMe").is_dir():
        anns += load_summe(data_root / "SumMe")
    if "tvsum" in which and (data_root / "TVSum").is_dir():
        anns += load_tvsum(data_root / "TVSum")
    if not anns:
        raise FileNotFoundError(
            f"no SumMe/TVSum data under {data_root} — run scripts/fetch_datasets.py"
        )
    return anns


def iter_with_video(anns: list[VideoAnnotation]) -> Iterator[VideoAnnotation]:
    for a in anns:
        if a.video_path and a.video_path.exists():
            yield a
        else:
            logger.warning("%s/%s: no video file, skipping feature extraction",
                           a.dataset, a.video_id)
