"""Unit tests for core.fuse alignment logic. No ML dependencies required."""

from __future__ import annotations

from core.fuse import _fuse_segments_and_scenes


def _segment(id_, start, end, text):
    return {"id": id_, "start": start, "end": end, "text": text}


def _scene(id_, start, end, labels=()):
    return {
        "id": id_,
        "start": start,
        "end": end,
        "keyframe_path": f"keyframes/scene_{id_:03d}.jpg",
        "objects": [{"label": label, "conf": 0.9} for label in labels],
    }


def test_every_segment_lands_in_exactly_one_block_and_no_text_lost():
    segments = [
        _segment(0, 0.0, 4.0, "Welcome to the lecture."),
        _segment(1, 4.0, 9.0, "Today we discuss normalization."),
        _segment(2, 9.5, 15.0, "First normal form requires atomic values."),
        _segment(3, 15.0, 22.0, "Second normal form removes partial dependency."),
        _segment(4, 22.0, 30.0, "Let's look at an example."),
    ]
    scenes = [
        _scene(0, 0.0, 10.0, labels=["whiteboard", "person"]),
        _scene(1, 10.0, 20.0, labels=["slide"]),
        _scene(2, 20.0, 30.0, labels=["person"]),
    ]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    # No text lost: every segment's text appears in exactly one block.
    seen_segment_ids: set[int] = set()
    for seg in segments:
        midpoint = (seg["start"] + seg["end"]) / 2.0
        owning_blocks = [b for b in blocks if seg["text"] in b["text"]]
        assert len(owning_blocks) == 1, f"segment {seg['id']} landed in {len(owning_blocks)} blocks"
        seen_segment_ids.add(seg["id"])

    assert seen_segment_ids == {s["id"] for s in segments}

    # concatenated block text should contain every character of every segment's text
    combined_blocks_text = " ".join(b["text"] for b in blocks)
    for seg in segments:
        assert seg["text"] in combined_blocks_text


def test_block_ids_are_sequential_and_windows_match_scenes():
    segments = [_segment(0, 1.0, 2.0, "hello")]
    scenes = [_scene(0, 0.0, 5.0), _scene(1, 5.0, 10.0)]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    assert [b["id"] for b in blocks] == list(range(len(blocks)))
    for b in blocks:
        assert b["start"] < b["end"]


def test_short_silent_scene_is_dropped():
    segments: list[dict] = []
    scenes = [_scene(0, 0.0, 3.0, labels=["person"])]  # 3s < 5s threshold, no text

    blocks = _fuse_segments_and_scenes(segments, scenes)

    assert blocks == []


def test_long_silent_scene_is_kept_with_empty_text():
    segments: list[dict] = []
    scenes = [_scene(0, 0.0, 10.0, labels=["whiteboard"])]  # 10s >= 5s threshold

    blocks = _fuse_segments_and_scenes(segments, scenes)

    assert len(blocks) == 1
    assert blocks[0]["text"] == ""
    assert blocks[0]["visual_tags"] == ["whiteboard"]
    assert "whiteboard" in blocks[0]["fused_text"]


def test_fused_text_contains_timestamp_and_tags():
    segments = [_segment(0, 0.0, 2.0, "hi there")]
    scenes = [_scene(0, 0.0, 12.0, labels=["person", "whiteboard"])]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    assert len(blocks) == 1
    fused_text = blocks[0]["fused_text"]
    assert fused_text.startswith("[00:00-00:12]")
    assert "person" in fused_text and "whiteboard" in fused_text
    assert "hi there" in fused_text


def test_visual_tags_are_deduplicated_and_sorted():
    segments: list[dict] = []
    scenes = [{
        "id": 0, "start": 0.0, "end": 10.0, "keyframe_path": "x.jpg",
        "objects": [
            {"label": "person", "conf": 0.9},
            {"label": "person", "conf": 0.5},
            {"label": "chair", "conf": 0.4},
        ],
    }]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    assert blocks[0]["visual_tags"] == ["chair", "person"]


def test_segment_at_exact_boundary_between_scenes_goes_to_earlier_scene():
    # Midpoint exactly on the shared boundary of two contiguous scenes.
    segments = [_segment(0, 8.0, 12.0, "boundary text")]  # midpoint = 10.0
    scenes = [_scene(0, 0.0, 10.0), _scene(1, 10.0, 20.0)]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    matches = [b for b in blocks if "boundary text" in b["text"]]
    assert len(matches) == 1
    assert matches[0]["start"] == 10.0  # falls in the second (half-open) scene


def test_segment_at_final_video_timestamp_is_not_dropped():
    segments = [_segment(0, 19.0, 21.0, "closing remarks")]  # midpoint = 20.0 == last scene end
    scenes = [_scene(0, 0.0, 10.0), _scene(1, 10.0, 20.0)]

    blocks = _fuse_segments_and_scenes(segments, scenes)

    matches = [b for b in blocks if "closing remarks" in b["text"]]
    assert len(matches) == 1
