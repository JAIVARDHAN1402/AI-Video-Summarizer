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

SumMe downloads automatically (~2.2 GB). TVSum's archive is behind Google
Drive — the script prints the steps; once the annotation files are in place,
`--tvsum-videos` pulls the 50 clips with yt-dlp. Target layout:

```
data/highlight/SumMe/GT/*.mat        data/highlight/SumMe/videos/*
data/highlight/TVSum/ydata-tvsum50.mat   data/highlight/TVSum/video/*.mp4
```

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
