"""The progress callback must only ever be invoked from the calling thread.

Streamlit widget calls raise ``NoSessionContext`` when made from anywhere but
its script-runner thread. Because transcription and visual analysis run
concurrently in a thread pool, a sink that published progress directly from
those workers broke every run in the UI while passing every headless test —
which is exactly the kind of regression this guards against.
"""

from __future__ import annotations

import threading

from core import pipeline


def test_sink_records_without_publishing():
    """The sink must not call back out; it only accumulates state."""
    sink = pipeline._AnalyzeProgress(0.15, 0.55)

    sink("vid", "transcribe", 0.5, "Transcribing...", None)
    sink("vid", "vision", 1.0, "Visuals done.", None)

    progress, message = sink.snapshot()
    # 0.15 + 0.55 * (0.5*0.75 + 1.0*0.25)
    assert progress == 0.15 + 0.55 * 0.625
    assert message == "Transcribing..."


def test_snapshot_is_thread_safe():
    """Concurrent writers must not corrupt the merged view."""
    sink = pipeline._AnalyzeProgress(0.0, 1.0)
    errors: list[BaseException] = []

    def hammer(stage: str) -> None:
        try:
            for i in range(500):
                sink("vid", stage, i / 500, f"{stage} {i}", None)
                sink.snapshot()
        except BaseException as exc:  # noqa: BLE001 - recorded for the assert
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(s,)) for s in ("transcribe", "vision")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    progress, _ = sink.snapshot()
    assert 0.0 <= progress <= 1.0


def test_progress_callback_only_runs_on_the_calling_thread(monkeypatch, tmp_path):
    """End-to-end guard over the orchestrator's concurrent phase.

    Stands in fake stages that report progress from their worker threads, and
    asserts every resulting progress_cb invocation still lands on the thread
    that called process().
    """
    import config

    monkeypatch.setattr(config, "CACHE_ROOT", tmp_path)

    main_thread = threading.get_ident()
    callback_threads: set[int] = set()

    from core import cache, fuse, ingest, qa, summarize, transcribe, understand, vision

    meta = {
        "video_id": "vid", "source": "upload", "source_ref": "x.mp4",
        "title": "x", "duration_sec": 60.0, "width": 640, "height": 360,
        "fps": 25.0, "created_at": "now",
    }

    def fake_ingest(*_args, **_kwargs):
        return meta

    def fake_transcribe(video_id, force=False, plan=None):
        for i in range(5):
            cache.set_state(video_id, "transcribe", i / 5, f"seg {i}")
        return {"segments": [], "full_text": ""}

    def fake_vision(video_id, force=False, plan=None):
        for i in range(5):
            cache.set_state(video_id, "vision", i / 5, f"frame {i}")
        return {"scenes": []}

    monkeypatch.setattr(ingest, "ingest", fake_ingest)
    monkeypatch.setattr(transcribe, "transcribe", fake_transcribe)
    monkeypatch.setattr(vision, "analyze_visual", fake_vision)
    monkeypatch.setattr(fuse, "fuse", lambda *a, **k: {"blocks": []})
    # Pin the local path: this test is about which thread the progress
    # callback runs on, and the LLM branch would make it depend on whether
    # an API key happens to be configured.
    monkeypatch.setattr(understand, "is_available", lambda: False)
    monkeypatch.setattr(summarize, "summarize", lambda *a, **k: {"chapters": []})
    monkeypatch.setattr(qa, "build_index", lambda *a, **k: None)

    def record(_stage, _progress, _message):
        callback_threads.add(threading.get_ident())

    pipeline.process("x.mp4", is_upload=True, progress_cb=record)

    assert callback_threads == {main_thread}, (
        "progress_cb was invoked from a worker thread; Streamlit raises "
        "NoSessionContext when that happens"
    )
