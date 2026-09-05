"""M7 — the segment-importance (highlight) model.

Trained offline on SumMe + TVSum; the runtime pipeline only touches
``ml.infer`` and only when ``VS_HIGHLIGHT_MODEL`` points at a checkpoint.
Nothing here is imported by ``core`` at import time.

Layout
    datasets.py   SumMe / TVSum annotation loaders -> a common VideoAnnotation
    features.py   video file -> sampled CLIP feature sequence (reuses core CLIP)
    kts.py        Kernel Temporal Segmentation: feature sequence -> shot bounds
    model.py      ImportanceScorer (BiGRU + MLP head), frame scores in [0, 1]
    data.py       torch Dataset over prepared .npz shards, per-video z-norm
    train.py      5-fold CV training loop, checkpoints to ml/checkpoints/
    evaluate.py   F1 (knapsack summary) + Kendall tau / Spearman rho
    infer.py      load a checkpoint, score a CLIP embedding sequence
"""
