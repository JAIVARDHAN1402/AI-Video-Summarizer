"""M3 — Visual analysis: scene detection, keyframe extraction, object tags
(YOLOv8n), on-screen text (EasyOCR, for slides/whiteboards), and CLIP
embeddings for keyframe dedup and chapter boundaries.

Every per-scene cost in the pipeline — object detection, CLIP, OCR, the
chapter grid, the Scenes gallery — scales with the number of keyframes, so
this module caps that number (``plan.max_keyframes``) rather than letting it
grow with video length. Scenes past the cap are merged shortest-first, which
costs timeline granularity but keeps coverage complete, and is what makes a
90-minute video cost roughly what a 10-minute one does here.

Scene detection itself is bounded the other way: it must decode the video, so
the planner supplies a downscale factor and frame-skip that cut the decoded
pixel count near-linearly for longer inputs.
"""

from __future__ import annotations

import heapq
import json
import logging
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
from core import cache, perf

logger = logging.getLogger(__name__)

_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_OCR_READER = None


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


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        import torch

        logger.info("loading EasyOCR reader languages=%s", config.OCR_LANGUAGES)
        _OCR_READER = easyocr.Reader(config.OCR_LANGUAGES, gpu=torch.cuda.is_available())
    return _OCR_READER


# --- Scene segmentation ------------------------------------------------------------


def _fixed_interval_scenes(duration_sec: float) -> list[tuple[float, float]]:
    step = config.FIXED_INTERVAL_FALLBACK_SEC
    scenes: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec:
        end = min(t + step, duration_sec)
        scenes.append((t, end))
        t = end
    return scenes or [(0.0, duration_sec)]


def _detect_scenes(
    video_path: Path, duration_sec: float, plan: perf.ExecutionPlan
) -> tuple[list[tuple[float, float]], str]:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    # Downscaling lives on the SceneManager in PySceneDetect 0.6.x, and its
    # auto mode must be switched off before an explicit factor is honoured.
    if plan.scene_downscale > 1:
        scene_manager.auto_downscale = False
        scene_manager.downscale = plan.scene_downscale

    min_scene_len_frames = max(int(config.MIN_SCENE_LEN_SEC * video.frame_rate), 1)
    scene_manager.add_detector(
        ContentDetector(threshold=config.SCENE_DETECT_THRESHOLD, min_scene_len=min_scene_len_frames)
    )
    scene_manager.detect_scenes(video, frame_skip=plan.scene_frame_skip, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    scenes = [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

    # Static lecture recordings often trigger no content cuts at all.
    if len(scenes) < 2:
        logger.info("content detection found <2 scenes; using fixed intervals")
        return _fixed_interval_scenes(duration_sec), "fixed_interval"

    return scenes, "content_detector"


def _segment_by_sampling(
    video_path: Path, duration_sec: float, plan: perf.ExecutionPlan,
    keyframes_dir: Path, video_id: str,
) -> tuple[list[tuple[float, float]], list[Path], np.ndarray]:
    """Segment a long video by sampling frames instead of decoding all of them.

    Content-based scene detection has to walk every frame, so its cost is
    linear in video length no matter how aggressively the comparison is
    downscaled — decoding, not comparing, is what dominates. Measured on the
    reference machine, a 67-minute video costs ~140 s that way, which alone
    overruns the whole analysis budget.

    Since the pipeline caps keyframes anyway, frame-accurate cut detection buys
    nothing downstream. This instead seeks to a fixed number of evenly spaced
    timestamps (cost proportional to that number, not to the video), embeds
    them with CLIP, and places boundaries at the largest semantic jumps. The
    result is bounded work for any length, and the boundaries track meaning
    rather than hard cuts — which is what the chapter grid actually wants.
    """
    sample_count = min(
        max(plan.max_keyframes * 2, config.SAMPLED_SEGMENTATION_MIN_SAMPLES),
        config.SAMPLED_SEGMENTATION_MAX_SAMPLES,
    )
    # Offset by half a step so samples land inside their slice, not on the
    # boundary (and never at t=0, which is often a black or title frame).
    step = duration_sec / sample_count
    timestamps = [min((i + 0.5) * step, max(duration_sec - 0.1, 0.0)) for i in range(sample_count)]

    keyframes_dir.mkdir(parents=True, exist_ok=True)
    cache.set_state(video_id, "vision", 0.05, f"Sampling {sample_count} frames...")

    paths: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=min(8, sample_count)) as pool:
        futures = {
            pool.submit(
                _extract_one_keyframe, video_path, t, keyframes_dir / f"sample_{i:03d}.jpg"
            ): (i, keyframes_dir / f"sample_{i:03d}.jpg")
            for i, t in enumerate(timestamps)
        }
        done = 0
        for future, (i, path) in futures.items():
            try:
                if future.result():
                    paths[i] = path
            except Exception:
                logger.debug("sample %d failed", i, exc_info=True)
            done += 1
            cache.set_state(
                video_id, "vision", 0.05 + 0.3 * done / sample_count,
                f"Sampling frame {done}/{sample_count}...",
            )

    kept = sorted(paths)
    if not kept:
        raise RuntimeError("Could not read any frames from this video.")

    sample_paths = [paths[i] for i in kept]
    sample_times = [timestamps[i] for i in kept]

    cache.set_state(video_id, "vision", 0.4, "Finding topic boundaries...")
    embeddings = _run_clip_batch(sample_paths, plan.clip_batch_size)

    # Boundaries at the largest consecutive semantic jumps, capped so the
    # output size is fixed regardless of how long the video is.
    n_boundaries = max(0, min(plan.max_keyframes, len(kept)) - 1)
    distances = sorted(
        ((1.0 - float(np.dot(embeddings[i - 1], embeddings[i])), i) for i in range(1, len(kept))),
        key=lambda pair: pair[0], reverse=True,
    )
    cut_indices = sorted(index for _, index in distances[:n_boundaries])

    scenes: list[tuple[float, float]] = []
    representatives: list[int] = []
    group_start_time = 0.0
    group_start_index = 0
    for cut in [*cut_indices, len(kept)]:
        end_time = sample_times[cut] if cut < len(kept) else duration_sec
        scenes.append((group_start_time, end_time))
        # Represent each segment with its middle sample, which is furthest from
        # the transitions on either side.
        representatives.append((group_start_index + cut - 1) // 2)
        group_start_time = end_time
        group_start_index = cut

    chosen_paths = [sample_paths[i] for i in representatives]
    chosen_embeddings = np.stack([embeddings[i] for i in representatives])

    # Unused samples are pure scratch; leaving them would triple the size of
    # the keyframes directory and confuse the Scenes gallery.
    keep = set(representatives)
    for i, path in enumerate(sample_paths):
        if i not in keep:
            path.unlink(missing_ok=True)

    logger.info(
        "sampled %d frames from %.1f min video -> %d segments",
        len(kept), duration_sec / 60, len(scenes),
    )
    return scenes, chosen_paths, chosen_embeddings


def _cap_scene_count(scenes: list[tuple[float, float]], max_scenes: int) -> list[tuple[float, float]]:
    """Merge adjacent scenes until at most ``max_scenes`` remain.

    Repeatedly absorbs the shortest remaining scene into its shorter neighbour,
    so the cuts that survive are the ones separating the most substantial
    stretches of video. Coverage of the timeline is preserved exactly — only
    its granularity drops.
    """
    if len(scenes) <= max_scenes:
        return scenes

    spans = [list(s) for s in scenes]
    # (length, index) min-heap; stale entries are skipped via the alive check.
    heap = [(spans[i][1] - spans[i][0], i) for i in range(len(spans))]
    heapq.heapify(heap)
    alive = [True] * len(spans)
    prev = list(range(-1, len(spans) - 1))
    nxt = list(range(1, len(spans) + 1))
    nxt[-1] = -1
    remaining = len(spans)

    while remaining > max_scenes and heap:
        length, i = heapq.heappop(heap)
        if not alive[i] or abs((spans[i][1] - spans[i][0]) - length) > 1e-9:
            continue  # stale entry from before this scene was extended

        left, right = prev[i], nxt[i]
        if left == -1 and right == -1:
            break
        if left == -1:
            target = right
        elif right == -1:
            target = left
        else:
            len_l = spans[left][1] - spans[left][0]
            len_r = spans[right][1] - spans[right][0]
            target = left if len_l <= len_r else right

        # Absorb i into target, then unlink i.
        spans[target][0] = min(spans[target][0], spans[i][0])
        spans[target][1] = max(spans[target][1], spans[i][1])
        alive[i] = False
        remaining -= 1
        if left != -1:
            nxt[left] = nxt[i]
        if right != -1:
            prev[right] = prev[i]
        heapq.heappush(heap, (spans[target][1] - spans[target][0], target))

    return [(s[0], s[1]) for s, ok in zip(spans, alive) if ok]


# --- Keyframes ---------------------------------------------------------------------


def _extract_one_keyframe(video_path: Path, time_sec: float, out_path: Path) -> bool:
    """Grab a single frame with FFmpeg's fast input-side seek.

    Placing ``-ss`` before ``-i`` seeks to the nearest keyframe without
    decoding everything up to that point, which is what makes extracting N
    thumbnails cost O(N) instead of O(video length). A thumbnail landing on a
    neighbouring keyframe is visually indistinguishable for our purposes.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{max(time_sec, 0.0):.3f}",
        "-i", str(video_path),
        "-frames:v", "1", "-q:v", "3",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and out_path.exists()


def _extract_keyframes(
    video_path: Path, scenes: list[tuple[float, float]], out_dir: Path, video_id: str,
) -> tuple[list[Path], list[int]]:
    """Extract one keyframe per scene, in parallel.

    Each frame is an independent FFmpeg process, so these overlap cleanly on a
    thread pool despite the GIL — the work happens in subprocesses.

    Returns:
        ``(keyframe_paths, kept_scene_indices)`` — scenes whose frame could not
        be read (a seek past a truncated stream, a corrupt tail) are dropped
        from both, so the two stay index-aligned with each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        (i, (start + end) / 2.0, out_dir / f"scene_{i:03d}.jpg")
        for i, (start, end) in enumerate(scenes)
    ]

    done = 0
    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(targets) or 1)) as pool:
        futures = {
            pool.submit(_extract_one_keyframe, video_path, t, path): (i, path)
            for i, t, path in targets
        }
        for future, (i, path) in futures.items():
            try:
                if future.result():
                    results[i] = path
            except Exception:
                logger.debug("keyframe extraction failed for scene %d", i, exc_info=True)
            done += 1
            cache.set_state(
                video_id, "vision", 0.15 + 0.25 * done / len(targets),
                f"Extracting keyframe {done}/{len(targets)}...",
            )

    kept = sorted(results)
    return [results[i] for i in kept], kept


# --- Per-keyframe models -----------------------------------------------------------


def _run_yolo_batch(keyframe_paths: list[Path], batch_size: int) -> list[list[dict[str, Any]]]:
    if not keyframe_paths:
        return []

    model = _get_yolo()
    objects_per_scene: list[list[dict[str, Any]]] = []
    for i in range(0, len(keyframe_paths), batch_size):
        batch = [str(p) for p in keyframe_paths[i : i + batch_size]]
        for result in model.predict(
            source=batch, conf=config.YOLO_CONF_THRESHOLD,
            imgsz=config.YOLO_IMAGE_SIZE, verbose=False,
        ):
            names = result.names
            best_conf_by_label: dict[str, float] = {}
            if result.boxes is not None:
                for box in result.boxes:
                    label = names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if conf > best_conf_by_label.get(label, 0.0):
                        best_conf_by_label[label] = conf
            ranked = sorted(best_conf_by_label.items(), key=lambda kv: kv[1], reverse=True)
            objects_per_scene.append([
                {"label": label, "conf": round(conf, 4)}
                for label, conf in ranked[: config.YOLO_MAX_OBJECTS_PER_FRAME]
            ])
    return objects_per_scene


def _run_clip_batch(keyframe_paths: list[Path], batch_size: int) -> np.ndarray:
    if not keyframe_paths:
        return np.zeros((0, 512), dtype=np.float32)

    import torch
    from PIL import Image

    model, preprocess = _get_clip()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(keyframe_paths), batch_size):
            batch_paths = keyframe_paths[i : i + batch_size]
            images = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths])
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
            batches.append(features.cpu().numpy())
    return np.concatenate(batches, axis=0).astype(np.float32)


def _text_likeness_scores(keyframe_paths: list[Path]) -> list[float]:
    """Cheap per-frame score for "probably contains readable text".

    Slides and whiteboards are dominated by high-frequency, high-contrast
    horizontal edges; talking-head footage is not. A Laplacian variance over a
    small grayscale copy separates the two well enough to choose which frames
    are worth paying EasyOCR for, and costs microseconds per frame against
    OCR's ~1 second.
    """
    import cv2

    scores = []
    for path in keyframe_paths:
        try:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                scores.append(0.0)
                continue
            small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
            scores.append(float(cv2.Laplacian(small, cv2.CV_64F).var()))
        except Exception:
            scores.append(0.0)
    return scores


def _select_ocr_frames(keyframe_paths: list[Path], max_frames: int) -> list[int]:
    """Indices of the frames most worth running OCR on."""
    if max_frames >= len(keyframe_paths):
        return list(range(len(keyframe_paths)))
    scores = _text_likeness_scores(keyframe_paths)
    return sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_frames])


def _ocr_observation_key() -> str:
    """Calibration key for OCR cost, versioned by the settings that drive it.

    A measured cost is only meaningful for the configuration that produced
    it. Folding the input width into the key means changing it starts a fresh
    measurement instead of leaving the planner budgeting against a number
    from the old, slower settings — which is exactly what happened when the
    OCR input was narrowed from 960px to 640px.
    """
    return f"ocr:per_frame:w{config.OCR_MAX_WIDTH}"


def _ocr_one(reader, path: Path) -> str:
    """Read on-screen text from a single keyframe, downscaled first."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return ""
    height, width = image.shape[:2]
    if width > config.OCR_MAX_WIDTH:
        scale = config.OCR_MAX_WIDTH / width
        image = cv2.resize(
            image, (config.OCR_MAX_WIDTH, int(height * scale)), interpolation=cv2.INTER_AREA
        )
    detections = reader.readtext(
        image, canvas_size=config.OCR_MAX_WIDTH, mag_ratio=1.0
    )
    pieces = [
        text.strip()
        for _bbox, text, conf in detections
        if conf >= config.OCR_CONF_THRESHOLD and len(text.strip()) >= config.OCR_MIN_TEXT_LEN
    ]
    return " ".join(pieces)


def _run_ocr(keyframe_paths: list[Path], plan: perf.ExecutionPlan, video_id: str) -> list[str]:
    """Read on-screen text from the keyframes most likely to carry it.

    OCR is the most expensive per-frame operation in the pipeline by a wide
    margin — measured at ~4.7 s/frame uncontended, and more than double that
    while transcription is using the cores — so it is never run over every
    keyframe.

    Two things bound it. Frames are ranked by a cheap text-likeness score and
    only the top ones are considered; then the highest-ranked few are read as a
    **probe**, and if they turn up essentially no text the rest are skipped
    entirely. That second step is what makes talking-head footage free: there
    is no threshold to tune, the video itself answers whether OCR is worth
    paying for.
    """
    if not plan.ocr_enabled or not keyframe_paths:
        return ["" for _ in keyframe_paths]

    selected = _select_ocr_frames(keyframe_paths, plan.ocr_max_frames)
    if not selected:
        return ["" for _ in keyframe_paths]

    reader = _get_ocr_reader()
    texts = ["" for _ in keyframe_paths]
    started = time.perf_counter()
    probe_size = min(config.OCR_PROBE_FRAMES, len(selected))
    read_count = 0

    for n, index in enumerate(selected, start=1):
        try:
            texts[index] = _ocr_one(reader, keyframe_paths[index])
        except Exception:
            logger.debug("OCR failed on %s; treating as no text", keyframe_paths[index], exc_info=True)
        read_count += 1
        cache.set_state(
            video_id, "vision", 0.75 + 0.2 * n / len(selected),
            f"Reading on-screen text {n}/{len(selected)}...",
        )

        if n == probe_size and len(selected) > probe_size:
            found = sum(len(texts[i]) for i in selected[:probe_size])
            if found < config.OCR_PROBE_MIN_CHARS:
                logger.info(
                    "OCR probe found %d chars across %d frames; skipping the remaining %d",
                    found, probe_size, len(selected) - probe_size,
                )
                break

    if read_count >= config.MIN_FRAMES_FOR_CALIBRATION:
        perf.record_observation(
            _ocr_observation_key(), (time.perf_counter() - started) / read_count
        )
    logger.info("read on-screen text from %d/%d keyframes", read_count, len(keyframe_paths))
    return texts


# --- Dedup -------------------------------------------------------------------------


def _dedup_scenes(
    raw_scenes: list[tuple[float, float]],
    embeddings: np.ndarray,
    keyframe_paths: list[Path],
    objects_per_scene: list[list[dict[str, Any]]],
    ocr_texts: list[str],
) -> tuple[list[tuple[float, float]], np.ndarray, list[Path], list[list[dict[str, Any]]], list[str]]:
    """Merge consecutive near-duplicate scenes (e.g. a static talking-head shot
    that scene detection split at a minor cut)."""
    if len(raw_scenes) <= 1:
        return raw_scenes, embeddings, keyframe_paths, objects_per_scene, ocr_texts

    merged_scenes: list[list[float]] = [list(raw_scenes[0])]
    merged_embeddings = [embeddings[0]]
    merged_keyframes = [keyframe_paths[0]]
    merged_objects = [objects_per_scene[0]]
    merged_ocr = [ocr_texts[0]]

    for i in range(1, len(raw_scenes)):
        similarity = float(np.dot(merged_embeddings[-1], embeddings[i]))
        if similarity > config.KEYFRAME_DEDUP_SIMILARITY:
            merged_scenes[-1][1] = raw_scenes[i][1]  # extend previous scene, drop this one
            # A near-duplicate frame can still carry different on-screen text
            # (a slide advancing while the presenter's shot barely changes),
            # so keep both rather than silently dropping the merged one's.
            if ocr_texts[i] and ocr_texts[i] not in merged_ocr[-1]:
                merged_ocr[-1] = f"{merged_ocr[-1]} {ocr_texts[i]}".strip()
        else:
            merged_scenes.append(list(raw_scenes[i]))
            merged_embeddings.append(embeddings[i])
            merged_keyframes.append(keyframe_paths[i])
            merged_objects.append(objects_per_scene[i])
            merged_ocr.append(ocr_texts[i])

    return (
        [(s, e) for s, e in merged_scenes],
        np.stack(merged_embeddings),
        merged_keyframes,
        merged_objects,
        merged_ocr,
    )


# --- Orchestration -----------------------------------------------------------------


def _attach_scene_importance(result: dict[str, Any], cache_dir: Path) -> None:
    """Fold in the learned M7 importance scores, if a checkpoint is configured.

    Best-effort and lazily imported: no torch/ml import cost and no behaviour
    change when ``config.HIGHLIGHT_MODEL_PATH`` is unset or the file is gone.
    """
    ckpt = getattr(config, "HIGHLIGHT_MODEL_PATH", None)
    if not ckpt:
        return
    try:
        from ml import infer as ml_infer

        ml_infer.attach_importance(result, cache_dir, ckpt)
        ranked = result.get("importance_ranked")
        if ranked:
            logger.info("attached M7 importance to %d scenes (top scene id %s)",
                        len(result.get("scenes", [])), ranked[0])
    except Exception:
        logger.warning("scene-importance scoring skipped", exc_info=True)


def analyze_visual(
    video_id: str, force: bool = False, plan: Optional[perf.ExecutionPlan] = None
) -> dict[str, Any]:
    """Detect scenes, extract keyframes, tag objects, read text, and embed.

    Args:
        video_id: The cache key for this video.
        force: Re-run even if a cached scenes.json already exists.
        plan: Execution plan for this video; read from meta.json if omitted.

    Returns:
        The scenes.json dict.
    """
    if not force and cache.artifact_exists(video_id, "scenes"):
        logger.info("using cached scenes.json for %s", video_id)
        return cache.read_json(video_id, "scenes")  # type: ignore[return-value]

    meta = cache.read_json(video_id, "meta")
    if meta is None:
        raise RuntimeError(f"No meta.json for {video_id}; run ingest() first.")

    duration_sec = float(meta["duration_sec"])
    if plan is None:
        plan = (
            perf.plan_from_dict(meta["plan"]) if meta.get("plan")
            else perf.plan_for(duration_sec)
        )

    cache_dir = cache.get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"

    keyframes_dir = cache_dir / "keyframes"

    # Load the models before starting the clock. Cold-loading CLIP alone
    # takes ~25 s on this machine; counting that as segmentation work made
    # the recorded per-frame cost roughly six times the real throughput,
    # which the planner then budgeted against.
    cache.set_state(video_id, "vision", 0.0, "Loading vision models...")
    _get_clip()
    _get_yolo()
    started = time.perf_counter()

    if plan.use_sampled_segmentation:
        method = "clip_sampling"
        scenes_with_frames, keyframe_paths, embeddings = _segment_by_sampling(
            video_path, duration_sec, plan, keyframes_dir, video_id
        )
        raw_scene_count = len(scenes_with_frames)
    else:
        method_started = time.perf_counter()
        cache.set_state(video_id, "vision", 0.0, "Detecting scene changes...")
        raw_scenes, method = _detect_scenes(video_path, duration_sec, plan)
        # Short videos measure model loading and process startup, not
        # throughput: a 66-second clip once recorded an RTF of 1.5, which
        # then predicted 466 s of scene detection for a 5-minute video.
        if duration_sec >= config.MIN_DURATION_FOR_CALIBRATION_SEC:
            perf.record_observation(
                "scene:detect", (time.perf_counter() - method_started) / duration_sec
            )
        raw_scene_count = len(raw_scenes)

        capped_scenes = _cap_scene_count(raw_scenes, plan.max_keyframes)
        logger.info(
            "%s: %d scenes via %s, capped to %d",
            video_id, raw_scene_count, method, len(capped_scenes),
        )
        keyframe_paths, kept_indices = _extract_keyframes(
            video_path, capped_scenes, keyframes_dir, video_id
        )
        if not keyframe_paths:
            raise RuntimeError(f"Could not extract any keyframes from {video_id}'s video.")
        scenes_with_frames = [capped_scenes[i] for i in kept_indices]

        cache.set_state(video_id, "vision", 0.6, "Computing visual embeddings...")
        embeddings = _run_clip_batch(keyframe_paths, plan.clip_batch_size)

    # Only the sampled path is charged per frame; content detection scales
    # with video length and is recorded under its own key above. Sharing one
    # key between the two corrupted it — a 4-keyframe content-detection run
    # recorded 41 s "per frame", which then made the planner refuse to
    # sample anything.
    if plan.use_sampled_segmentation and len(keyframe_paths) >= config.MIN_FRAMES_FOR_CALIBRATION:
        perf.record_observation(
            "segment:sampled:per_frame",
            (time.perf_counter() - started) / len(keyframe_paths),
        )

    cache.set_state(video_id, "vision", 0.65, "Detecting objects...")
    tag_started = time.perf_counter()
    objects_per_scene = _run_yolo_batch(keyframe_paths, plan.yolo_batch_size)
    # A handful of frames measures model loading, not throughput.
    if (len(keyframe_paths) >= config.MIN_FRAMES_FOR_CALIBRATION
            and duration_sec >= config.MIN_DURATION_FOR_CALIBRATION_SEC):
        perf.record_observation(
            "tag:per_frame", (time.perf_counter() - tag_started) / len(keyframe_paths)
        )

    ocr_texts = _run_ocr(keyframe_paths, plan, video_id)

    cache.set_state(video_id, "vision", 0.93, "Merging near-identical scenes...")
    scenes, embeddings, keyframe_paths, objects_per_scene, ocr_texts = _dedup_scenes(
        scenes_with_frames, embeddings, keyframe_paths, objects_per_scene, ocr_texts
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
            "ocr_text": ocr_texts[i],
            "clip_embedding_idx": i,
        }
        for i, (start, end) in enumerate(scenes)
    ]

    result = {
        "scenes": scene_dicts,
        "embeddings_file": str(embeddings_file),
        "segmentation_method": method,
        "raw_scene_count": raw_scene_count,
        "ocr_frames_read": sum(1 for t in ocr_texts if t),
    }
    _attach_scene_importance(result, cache_dir)
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
