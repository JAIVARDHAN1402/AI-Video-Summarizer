"""M3 — Visual analysis: scene detection, keyframe extraction, object tags
(YOLOv8n), and CLIP embeddings for keyframe dedup and future retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import config
from core import cache

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO

        logger.info("loading YOLO model=%s", config.YOLO_MODEL)
        _YOLO_MODEL = YOLO(config.YOLO_MODEL)
    return _YOLO_MODEL


def _get_clip():
    global _CLIP_MODEL, _CLIP_PREPROCESS
    if _CLIP_MODEL is None:
        import open_clip

        logger.info("loading CLIP model=%s pretrained=%s", config.CLIP_MODEL, config.CLIP_PRETRAINED)
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED
        )
        model.eval()
        _CLIP_MODEL = model
        _CLIP_PREPROCESS = preprocess
    return _CLIP_MODEL, _CLIP_PREPROCESS


def _fixed_interval_scenes(duration_sec: float) -> list[tuple[float, float]]:
    step = config.FIXED_INTERVAL_FALLBACK_SEC
    scenes: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec:
        end = min(t + step, duration_sec)
        scenes.append((t, end))
        t = end
    return scenes or [(0.0, duration_sec)]


def _detect_scenes(video_path: Path, duration_sec: float) -> tuple[list[tuple[float, float]], str]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    min_scene_len_frames = max(int(config.MIN_SCENE_LEN_SEC * video.frame_rate), 1)
    scene_manager.add_detector(
        ContentDetector(threshold=config.SCENE_DETECT_THRESHOLD, min_scene_len=min_scene_len_frames)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

    # Static lecture recordings often trigger no content cuts at all.
    if len(scenes) < 2:
        logger.info("content-based scene detection found <2 scenes; falling back to fixed intervals")
        return _fixed_interval_scenes(duration_sec), "fixed_interval"

    return scenes, "content_detector"


def _extract_keyframe(cap, mid_time_sec: float, out_path: Path) -> None:
    import cv2

    cap.set(cv2.CAP_PROP_POS_MSEC, mid_time_sec * 1000)
    ok, frame = cap.read()
    if not ok:
        # fall back to the very first frame if seeking past EOF
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read a keyframe near {mid_time_sec:.1f}s")
    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _run_yolo_batch(keyframe_paths: list[Path]) -> list[list[dict[str, Any]]]:
    if not keyframe_paths:
        return []

    model = _get_yolo()
    results = model.predict(
        source=[str(p) for p in keyframe_paths],
        conf=config.YOLO_CONF_THRESHOLD,
        batch=config.YOLO_BATCH_SIZE,
        verbose=False,
    )

    objects_per_scene: list[list[dict[str, Any]]] = []
    for result in results:
        names = result.names
        best_conf_by_label: dict[str, float] = {}
        if result.boxes is not None:
            for box in result.boxes:
                label = names[int(box.cls[0])]
                conf = float(box.conf[0])
                if label not in best_conf_by_label or conf > best_conf_by_label[label]:
                    best_conf_by_label[label] = conf
        ranked = sorted(best_conf_by_label.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: config.YOLO_MAX_OBJECTS_PER_FRAME]
        objects_per_scene.append([{"label": label, "conf": round(conf, 4)} for label, conf in top])
    return objects_per_scene


def _run_clip_batch(keyframe_paths: list[Path]) -> np.ndarray:
    if not keyframe_paths:
        return np.zeros((0, 512), dtype=np.float32)

    import torch
    from PIL import Image

    model, preprocess = _get_clip()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(keyframe_paths), config.CLIP_BATCH_SIZE):
            batch_paths = keyframe_paths[i : i + config.CLIP_BATCH_SIZE]
            images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths])
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def _dedup_scenes(
    raw_scenes: list[tuple[float, float]],
    embeddings: np.ndarray,
    keyframe_paths: list[Path],
    objects_per_scene: list[list[dict[str, Any]]],
) -> tuple[list[tuple[float, float]], np.ndarray, list[Path], list[list[dict[str, Any]]]]:
    """Merge consecutive near-duplicate scenes (e.g. a static talking-head shot
    that scene detection split at a minor cut)."""
    if len(raw_scenes) <= 1:
        return raw_scenes, embeddings, keyframe_paths, objects_per_scene

    merged_scenes: list[list[float]] = [list(raw_scenes[0])]
    merged_embeddings = [embeddings[0]]
    merged_keyframes = [keyframe_paths[0]]
    merged_objects = [objects_per_scene[0]]

    for i in range(1, len(raw_scenes)):
        similarity = float(np.dot(merged_embeddings[-1], embeddings[i]))
        if similarity > config.KEYFRAME_DEDUP_SIMILARITY:
            merged_scenes[-1][1] = raw_scenes[i][1]  # extend previous scene's end, drop this one
        else:
            merged_scenes.append(list(raw_scenes[i]))
            merged_embeddings.append(embeddings[i])
            merged_keyframes.append(keyframe_paths[i])
            merged_objects.append(objects_per_scene[i])

    scenes = [(s, e) for s, e in merged_scenes]
    return scenes, np.stack(merged_embeddings), merged_keyframes, merged_objects


def analyze_visual(video_id: str, force: bool = False) -> dict[str, Any]:
    """Detect scenes, extract keyframes, tag objects, and embed with CLIP.

    Args:
        video_id: The cache key for this video.
        force: Re-run even if a cached scenes.json already exists.

    Returns:
        The scenes.json dict.
    """
    if not force and cache.artifact_exists(video_id, "scenes"):
        logger.info("using cached scenes.json for %s", video_id)
        return cache.read_json(video_id, "scenes")  # type: ignore[return-value]

    meta = cache.read_json(video_id, "meta")
    if meta is None:
        raise RuntimeError(f"No meta.json for {video_id}; run ingest() first.")

    cache.set_state(video_id, "vision", 0.0, "Detecting scenes...")
    cache_dir = cache.get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"

    raw_scenes, method = _detect_scenes(video_path, float(meta["duration_sec"]))
    logger.info("%s: %d scenes via %s", video_id, len(raw_scenes), method)

    keyframes_dir = cache_dir / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    keyframe_paths: list[Path] = []
    try:
        for i, (start, end) in enumerate(raw_scenes):
            out_path = keyframes_dir / f"scene_{i:03d}.jpg"
            _extract_keyframe(cap, (start + end) / 2.0, out_path)
            keyframe_paths.append(out_path)
            cache.set_state(
                video_id, "vision", 0.1 + 0.3 * (i + 1) / len(raw_scenes),
                f"Extracting keyframe {i + 1}/{len(raw_scenes)}...",
            )
    finally:
        cap.release()

    cache.set_state(video_id, "vision", 0.45, "Running object detection...")
    objects_per_scene = _run_yolo_batch(keyframe_paths)

    cache.set_state(video_id, "vision", 0.65, "Computing visual embeddings...")
    embeddings = _run_clip_batch(keyframe_paths)

    cache.set_state(video_id, "vision", 0.85, "Deduplicating near-identical scenes...")
    scenes, embeddings, keyframe_paths, objects_per_scene = _dedup_scenes(
        raw_scenes, embeddings, keyframe_paths, objects_per_scene
    )

    embeddings_file = cache_dir / "clip_embeddings.npy"
    np.save(embeddings_file, embeddings.astype(np.float32))

    scene_dicts = [
        {
            "id": i,
            "start": start,
            "end": end,
            "keyframe_path": str(keyframe_paths[i]),
            "objects": objects_per_scene[i],
            "clip_embedding_idx": i,
        }
        for i, (start, end) in enumerate(scenes)
    ]

    result = {
        "scenes": scene_dicts,
        "embeddings_file": str(embeddings_file),
        "segmentation_method": method,
    }
    cache.write_json(video_id, "scenes", result)
    cache.set_state(video_id, "vision", 1.0, "Visual analysis complete.")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug M3 visual analysis in isolation.")
    parser.add_argument("video_id")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = analyze_visual(args.video_id, force=args.force)
    print(json.dumps({**result, "scenes": f"{len(result['scenes'])} scenes"}, indent=2))
