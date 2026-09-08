"""Highlight-reel scene picking and importance-aware image selection."""

import pytest


def _scene(i, start, end, importance=None, ocr="text"):
    d = {"id": i, "start": start, "end": end, "ocr_text": ocr, "objects": []}
    if importance is not None:
        d["importance"] = importance
    return d


def test_pick_scenes_greedy_by_importance_within_budget():
    from core.highlight import _pick_scenes

    scenes = [
        _scene(0, 0, 10, 0.2),
        _scene(1, 10, 20, 0.9),
        _scene(2, 20, 30, 0.5),
        _scene(3, 30, 40, 0.8),
        _scene(4, 40, 40.5, 0.95),  # 0.5s -> under MIN_SCENE_SECONDS, excluded
    ]
    picked = _pick_scenes(scenes, max_seconds=25)
    ids = [s["id"] for s in picked]
    assert ids == sorted(ids)                 # chronological order restored
    assert ids == [1, 3]                      # the two highest-importance long scenes
    assert 4 not in ids                       # too short even though most important
    assert sum(s["end"] - s["start"] for s in picked) <= 25


def test_pick_scenes_empty_without_importance():
    from core.highlight import _pick_scenes

    assert _pick_scenes([_scene(0, 0, 10), _scene(1, 10, 20)], 60) == []


def test_select_scene_images_prefers_high_importance(monkeypatch, tmp_path):
    from core import understand

    # fake keyframes on disk + cv2 that returns a tiny gray image
    import numpy as np

    frames = []
    for i in range(6):
        p = tmp_path / f"f{i}.jpg"
        p.write_bytes(b"x")
        frames.append(str(p))

    class FakeCv2:
        IMWRITE_JPEG_QUALITY = 1
        INTER_AREA = 2

        def imread(self, _):
            return np.zeros((10, 20, 3), dtype=np.uint8)

        def resize(self, img, *_a, **_k):
            return img

        def imencode(self, _ext, _img, _params):
            return True, np.frombuffer(b"jpegbytes", dtype=np.uint8)

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2())

    scenes = [
        {"keyframe_path": frames[i], "start": i * 10.0, "ocr_text": "slide",
         "objects": [], "importance": imp}
        for i, imp in enumerate([0.1, 0.9, 0.2, 0.95, 0.3, 0.8])
    ]
    imgs = understand._select_scene_images(scenes, limit=3)
    assert len(imgs) == 3
    # captions embed the timestamp; the three highest-importance scenes are at
    # 10s, 30s, 50s (ids 1, 3, 5) and should come back in time order.
    caps = " ".join(c for c, _ in imgs)
    assert "0:10" in caps and "0:30" in caps and "0:50" in caps
