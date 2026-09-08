"""Download and cache every model the pipeline can select, ahead of time.

The planner picks models per video, so a first run can otherwise stall for
minutes downloading whichever checkpoint it chose — and on a flaky connection
that surfaces as a confusing failure part-way through processing. Running this
once, up front, turns that into a single explicit step.

    python warmup.py            # everything
    python warmup.py --asr      # just the speech models
    python warmup.py --check    # report what's cached, download nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable

import config
from core import perf


def _asr_targets() -> list[str]:
    return sorted({rung.model for rung in perf.ASR_LADDER})


def _summarizer_targets() -> list[str]:
    return sorted({rung.model for rung in perf.SUMMARIZER_LADDER})


def _load_asr(name: str) -> None:
    from faster_whisper import WhisperModel

    WhisperModel(name, device="cpu", compute_type=config.WHISPER_COMPUTE_TYPE)


def _load_summarizer(name: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    AutoTokenizer.from_pretrained(name)
    AutoModelForSeq2SeqLM.from_pretrained(name)


def _load_embedding(name: str) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(name)


def _load_qa(name: str) -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    AutoTokenizer.from_pretrained(name)
    AutoModelForSeq2SeqLM.from_pretrained(name)


def _load_yolo(name: str) -> None:
    from ultralytics import YOLO

    YOLO(name)


def _load_clip(name: str) -> None:
    import open_clip

    open_clip.create_model_and_transforms(name, pretrained=config.CLIP_PRETRAINED)


def _load_ocr(_name: str) -> None:
    import easyocr

    easyocr.Reader(config.OCR_LANGUAGES, gpu=False)


def _run(label: str, name: str, loader: Callable[[str], None], retries: int = 3) -> bool:
    """Load one model, retrying transient download failures."""
    for attempt in range(1, retries + 1):
        started = time.perf_counter()
        try:
            loader(name)
        except Exception as exc:
            if attempt == retries:
                print(f"  FAILED  {label:<14} {name}\n          {type(exc).__name__}: {exc}")
                return False
            print(f"  retry {attempt}/{retries - 1}  {name} ({type(exc).__name__})")
            time.sleep(2 * attempt)
        else:
            print(f"  ok      {label:<14} {name}  ({time.perf_counter() - started:.1f}s)")
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-download the pipeline's models.")
    parser.add_argument("--asr", action="store_true", help="Speech recognition models only")
    parser.add_argument("--nlp", action="store_true", help="Summarization/Q&A models only")
    parser.add_argument("--vision", action="store_true", help="Vision models only")
    parser.add_argument(
        "--check", action="store_true",
        help="List what would be downloaded, without downloading it",
    )
    args = parser.parse_args()

    everything = not (args.asr or args.nlp or args.vision)
    jobs: list[tuple[str, str, Callable[[str], None]]] = []

    if everything or args.asr:
        jobs += [("speech", name, _load_asr) for name in _asr_targets()]
    if everything or args.nlp:
        jobs += [("summarizer", name, _load_summarizer) for name in _summarizer_targets()]
        jobs += [
            ("q&a", config.QA_LOCAL_MODEL, _load_qa),
            ("embeddings", config.EMBEDDING_MODEL, _load_embedding),
        ]
    if everything or args.vision:
        jobs += [("objects", config.YOLO_MODEL, _load_yolo), ("clip", config.CLIP_MODEL, _load_clip)]
        if config.OCR_ENABLED:
            jobs.append(("on-screen text", "easyocr", _load_ocr))

    if args.check:
        print(f"{len(jobs)} models would be prepared:")
        for label, name, _ in jobs:
            print(f"  {label:<14} {name}")
        return 0

    print(f"Preparing {len(jobs)} models. First run downloads several GB.\n")
    started = time.perf_counter()
    failures = [name for label, name, loader in jobs if not _run(label, name, loader)]

    print(f"\nFinished in {time.perf_counter() - started:.0f}s.")
    if failures:
        print(f"{len(failures)} model(s) could not be downloaded: {', '.join(failures)}")
        print("Re-run this script — downloads resume where they stopped.")
        return 1
    print("All models cached. The pipeline will not need to download anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
