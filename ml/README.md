# M7 — scene-importance / highlight model (`ml/`)

A small supervised model that scores how "highlight-worthy" each part of a
video is, trained offline on **SumMe** and **TVSum**. The runtime pipeline
uses it (when a checkpoint exists) to rank scenes — which to feature, which
keyframes to send the LLM, what to put at the top of the Scenes tab.

It is **not** in the pipeline's critical path: with no checkpoint the pipeline
behaves exactly as before.

## What it is

| | |
| --- | --- |
| Features | CLIP ViT-B/32 image embeddings (the same model `core/vision.py` loads), frames sampled at 2 fps, L2-normalised |
| Model | 1-layer bi-GRU (hidden 128) + 2-layer MLP head → per-frame score in [0, 1]. ~0.5M params, CPU inference in ms |
| Loss | masked MSE on min-max-normalised human importance + a pairwise ranking hinge |
| Segmentation | Kernel Temporal Segmentation (`kts.py`) — the standard shot boundaries for the F1 metric |
| Metrics | knapsack-summary **F1** (max-over-users for SumMe, mean for TVSum) and **Kendall τ / Spearman ρ** vs. mean human scores (Otani et al., CVPR 2019) |
| Protocol | 5-fold video-disjoint cross-validation |

## Get the data

```bash
python scripts/fetch_datasets.py --root data/highlight
```

Both download automatically — SumMe ~2.2 GB from ETH Zurich, TVSum ~641 MB
from the author's MIT page (videos included; the old Yahoo! Webscope and
Google Drive routes are gone). Re-running is safe: anything already in place
is skipped. Resulting layout:

```
data/highlight/SumMe/GT/*.mat            data/highlight/SumMe/videos/*
data/highlight/TVSum/ydata-tvsum50.mat   data/highlight/TVSum/video/*.mp4
```

75 videos in total — 25 SumMe, 50 TVSum. Two notes:

- TVSum's v1_1 `.mat` is MATLAB v7.3 (HDF5), which scipy cannot read. The
  `.tsv` files beside it carry the same annotations and are used instead —
  that is the normal path, not a fallback from an error.
- `--tvsum-videos` is only needed if that archive ever ships without its
  `video/` folder; it re-fetches the 50 clips with yt-dlp.

`downloads/` keeps the raw archives (~6.6 GB) and can be deleted once the
layout check passes. If the repo lives in a synced folder (OneDrive, Dropbox),
exclude `data/` from sync — it is gitignored, but the sync client will still
upload it and its file locks break the download's final rename.

## Train

```bash
pip install -r requirements-ml.txt
python -m ml.train --prepare --data-root data/highlight      # prepare shards + 5-fold CV
python -m ml.train --smoke                                   # plumbing check, no data
```

`--prepare` decodes every video once, runs CLIP + KTS, and writes
`ml/prepared/<dataset>__<id>.npz`. Training writes `ml/checkpoints/fold{0..4}.pt`,
a `cv_report.json`, and copies fold 0 to `ml/checkpoints/importance.pt`.

## Evaluate a checkpoint

```bash
python -m ml.evaluate --checkpoint ml/checkpoints/importance.pt --prepared ml/prepared
```

## Wire into the pipeline

The vision stage picks up `ml/checkpoints/importance.pt` automatically. Override
or disable:

```bash
VS_HIGHLIGHT_MODEL=/path/to/other.pt   # use a specific checkpoint
VS_HIGHLIGHT_MODEL=off                 # ignore any checkpoint
```

With a model active, `scenes.json` gains `importance` (0-1) per scene, an
`importance_ranked` scene-id order, and `importance_model`. The per-scene
keyframe embedding sequence is shorter and sparser than the 2 fps training
data, so treat the output as a **relative ranking over scenes**, not a
calibrated per-frame score.

## Files

`datasets.py` loaders · `features.py` video→CLIP sequence · `kts.py` temporal
segmentation · `model.py` the scorer · `data.py` shard prep + Dataset ·
`train.py` CV loop · `evaluate.py` metrics · `infer.py` runtime scoring.
