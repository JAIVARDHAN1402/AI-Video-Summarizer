"""Background model warm-up.

Loading every model this pipeline uses costs about 70 seconds on the reference
machine — CLIP alone is ~25 s, the summarizer ~19 s, EasyOCR ~12 s. Paid inside
a run, that lands entirely on the first video a user processes, and it is
invisible: the progress bar sits on one stage while nothing appears to happen.

None of it depends on the video, so it can start the moment the app does. By
the time a URL has been pasted and downloaded, the models are resident and the
first run costs what the tenth does.

Warm-up is best-effort. Failures are swallowed deliberately: a model that
cannot be preloaded will simply be loaded on demand later, with its real error
surfacing there, where it can be reported against an actual request.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)

_thread: Optional[threading.Thread] = None
_lock = threading.Lock()
_state: dict[str, object] = {"started": False, "done": False, "loaded": [], "failed": []}


def _targets() -> list[tuple[str, Callable[[], object]]]:
    """What to preload, cheapest and most-used first.

    Ordered so the things every run needs are resident soonest: if the user
    starts a video mid-warm-up, the remaining loads simply happen on demand.
    """
    from core import qa, summarize, vision

    targets: list[tuple[str, Callable[[], object]]] = [
        ("object detection", vision._get_yolo),          # noqa: SLF001 - internal by design
        ("visual embeddings", vision._get_clip),         # noqa: SLF001
        ("retrieval embeddings", qa._get_embedding_model),  # noqa: SLF001
    ]

    # With a provider configured, core.understand replaces the local map-reduce
    # and these never run — but warming them still holds ~700 MB resident for
    # the life of the process, which on a hosted box is a larger instance and a
    # real monthly cost. They stay loadable on demand for the no-key fallback.
    if not config.GEMINI_API_KEY:
        targets.append(
            ("summarizer", lambda: summarize._get_summarizer(config.SUMMARIZER_MODEL))  # noqa: SLF001
        )
        if config.CHAPTER_TITLES_USE_LLM:
            targets.append(("chapter titles", summarize._get_instruct_generator))  # noqa: SLF001
    if config.OCR_ENABLED:
        targets.append(("on-screen text", vision._get_ocr_reader))  # noqa: SLF001
    return targets


def _run() -> None:
    started = time.perf_counter()
    for label, loader in _targets():
        try:
            loader()
        except Exception as exc:  # best-effort: on-demand loading will retry
            logger.warning("could not preload %s: %s", label, exc)
            _state["failed"].append(label)  # type: ignore[union-attr]
        else:
            _state["loaded"].append(label)  # type: ignore[union-attr]
    _state["done"] = True
    logger.info(
        "model warm-up finished in %.1f s (%d loaded, %d failed)",
        time.perf_counter() - started,
        len(_state["loaded"]),  # type: ignore[arg-type]
        len(_state["failed"]),  # type: ignore[arg-type]
    )


def start() -> None:
    """Begin warming models in the background. Safe to call repeatedly."""
    global _thread
    with _lock:
        if _state["started"]:
            return
        _state["started"] = True

        # Import torch synchronously, before any thread exists. Importing it
        # concurrently from two threads races on its C-extension binding and
        # fails with "NameError: name '_C' is not defined" — which is what
        # happened here, because the pipeline reaches torch through
        # ctranslate2 at the same moment this warm-up reaches it through
        # CLIP. Once the module object is complete, later imports are just
        # dictionary lookups and are safe from anywhere.
        try:
            import torch  # noqa: F401  - imported for its side effect
        except Exception:
            logger.warning("torch unavailable; skipping warm-up", exc_info=True)
            _state["done"] = True
            return

        # Daemon so it can never hold the process open; the work is pure
        # caching and is safe to abandon half-done.
        _thread = threading.Thread(target=_run, name="model-warmup", daemon=True)
        _thread.start()
        logger.info("warming models in the background")


def status() -> dict[str, object]:
    """Snapshot of warm-up progress, for display."""
    return {
        "started": _state["started"],
        "done": _state["done"],
        "loaded": list(_state["loaded"]),  # type: ignore[arg-type]
        "failed": list(_state["failed"]),  # type: ignore[arg-type]
        "total": len(_targets()) if _state["started"] else 0,
    }
