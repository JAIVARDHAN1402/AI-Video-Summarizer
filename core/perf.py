"""Hardware probing and the adaptive execution planner.

The pipeline's contract is a **wall-clock budget**, not a fixed set of models.
:func:`plan_for` is handed a video duration and returns the heaviest
configuration still predicted to finish inside ``config.TARGET_PIPELINE_SEC``
on *this* machine. A 2-minute clip gets the accurate models; a 90-minute
lecture automatically downshifts to distilled ones and drops optional work, so
the 5-minute promise holds at both ends instead of only for short videos.

Predictions are driven by per-stage **real-time factors** (RTF = processing
seconds per second of video) measured on a reference machine and rescaled by a
cheap CPU probe. Every real run reports its actual RTFs back through
:func:`record_observation`, which blends them into a persisted profile — so the
planner calibrates itself to the machine it actually runs on instead of
trusting the shipped constants forever.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

# Reference machine every shipped constant in this module was measured on:
# 16 logical cores, Intel Raptor Lake mobile, 16 GB RAM, CPU-only, int8
# inference. That machine probes to exactly _REFERENCE_CPU_MARK, so its speed
# multiplier is 1.0 and the constants are its raw measurements.
_REFERENCE_CPU_MARK = 100.0

def _learned_ingest_per_min(
    profile: "HardwareProfile", cost_override: "dict[str, float] | None" = None
) -> float:
    """Measured seconds of ingest per video-minute, or 0.0 if never measured."""
    if cost_override and "ingest:per_min" in cost_override:
        return cost_override["ingest:per_min"]
    return profile.observed_rtf.get("ingest:per_min", 0.0)


def _usable_cpus() -> int:
    """Cores this process may actually run on, not the cores the box has."""
    try:
        import os as _os

        if hasattr(_os, "sched_getaffinity"):
            return len(_os.sched_getaffinity(0))
        import psutil

        return len(psutil.Process().cpu_affinity())
    except Exception:
        return os.cpu_count() or 1


def _profile_path() -> Path:
    """Where this process's calibration lives.

    Keyed by the hardware actually available, because the learned costs are
    only valid for it. A worker pinned to 2 of 16 cores measures itself 20x
    slower at object detection — folding that into the machine's own profile
    made the planner predict 19 minutes for a video it can do in 5, long after
    the pinned run had finished. A declared worker class means the same thing
    explicitly, so it separates the store too.
    """
    parts: list[str] = []
    worker_class = os.getenv("VS_WORKER_CLASS", "").strip()
    if worker_class:
        parts.append(re.sub(r"[^A-Za-z0-9._-]+", "-", worker_class))
    usable, total = _usable_cpus(), os.cpu_count() or 0
    if total and usable and usable != total:
        parts.append(f"{usable}of{total}core")
    suffix = ("." + ".".join(parts)) if parts else ""
    return config.CACHE_ROOT / f"hardware_profile{suffix}.json"

# Blend weights for a newly observed RTF against the stored one. These are
# deliberately **asymmetric**: bad news is believed quickly, good news slowly.
#
# The planner exists to hold a time budget, and its cost model is imperfect —
# Whisper's cost tracks *speech*, not video length, because VAD skips silence,
# so a lecture with dense narration is several times more expensive per minute
# than one with long pauses. A symmetric average across both kinds of video
# leaves the planner systematically optimistic, and an optimistic budget
# planner overruns. Converging toward the slower observation keeps the
# guarantee honest at the cost of occasionally under-using the budget.
_OBSERVATION_ALPHA_SLOWER = 0.7
_OBSERVATION_ALPHA_FASTER = 0.15

# Repeats of the CPU probe; the fastest one is kept. See _measure_cpu_mark.
_CPU_PROBE_REPEATS = 5
_REFERENCE_PROBE_SEC = 0.025


# --- Hardware profiling ----------------------------------------------------------


@dataclass
class HardwareProfile:
    """What this machine can do, measured once and cached."""

    cpu_count: int
    has_cuda: bool
    ram_gb: float
    cpu_mark: float  # relative to _REFERENCE_CPU_MARK; higher is faster
    observed_rtf: dict[str, float] = field(default_factory=dict)

    @property
    def speed(self) -> float:
        """Multiplier on the reference machine's throughput (>1 = faster)."""
        return max(self.cpu_mark / _REFERENCE_CPU_MARK, 0.05)

    @property
    def asr_workers(self) -> int:
        """How many audio chunks to transcribe side by side.

        This is the pipeline's main ASR lever: ``_transcribe_parallel`` caps
        the number of parallel chunks at this number, so it decides how much of
        the machine the stage can use. It was cores/4 — four chunks on a
        16-core box — which left most of the CPU idle during the stage that
        dominates every long run.

        Measured on 37 minutes of speech (16 cores, int8 ``tiny``):

            4 workers x 4 threads   277 s
            8 x 2                   234 s
            12 x 1                  222 s   <- best
            16 x 1                  244 s   <- oversubscribed, regresses

        Hence cores*3/4, capped at 12. The workers share one CTranslate2 model
        rather than loading one each, so the RAM guard is about the decode
        pool, not a copy per worker.
        """
        if self.has_cuda:
            return 1  # one GPU worker; batching happens inside the model
        by_cores = self.cpu_count * 3 // 4
        by_ram = int(max(self.ram_gb - 3.0, 1.0) // 0.5)
        return max(1, min(12, by_cores, by_ram))

    @property
    def asr_threads_per_worker(self) -> int:
        """Threads each ASR worker gets, so that workers x threads fits the CPU.

        The floor used to be 2, which quietly broke the budget once the worker
        count went up: 12 workers on a 16-core box asked for 24 threads, and
        vision runs concurrently on the same cores. The measured 12x1 config
        became a 24-thread thrash, and a 98-minute video spent 52 minutes in
        analyze against a predicted 12.
        """
        return max(1, self.cpu_count // max(self.asr_workers, 1))


# Matmul size and repetition for the CPU probe. Large enough that BLAS thread
# startup and cache warm-up don't dominate the measurement — an earlier, much
# smaller probe swung by nearly 2x between consecutive runs, which fed straight
# into every prediction the planner makes.
_PROBE_SIZE = 1024
_PROBE_ITERATIONS = 6
_PROBE_ROUNDS = 3
_PROBE_REFERENCE_SEC = 0.145  # best-of-rounds time on the reference machine


def _measure_cpu_mark() -> float:
    """Place this CPU against the reference with a small dense matmul.

    Takes the **fastest** of several short repeats rather than an average.
    A single timing on a machine that happens to be busy — another run, a
    browser, an indexer — reports a slow CPU, and since every prediction is
    divided by this number, one unlucky probe mis-scales the whole planner
    for as long as the profile is cached. The fastest repeat is the one least
    disturbed by other work, which is what we actually want to know.
    """
    try:
        import numpy as np

        size = 512
        a = np.random.rand(size, size).astype(np.float32)
        b = np.random.rand(size, size).astype(np.float32)
        best = float("inf")
        for _ in range(_CPU_PROBE_REPEATS):
            start = time.perf_counter()
            for _ in range(4):
                a @ b
            best = min(best, time.perf_counter() - start)
        # 4 iterations of 512^3 MACs. The divisor is the reference machine's
        # idle timing, so an unloaded machine of that class scores 1.0x and
        # the RTF constants below are used as measured.
        return max(_REFERENCE_CPU_MARK * (_REFERENCE_PROBE_SEC / max(best, 1e-6)), 1.0)
    except Exception:
        logger.warning("CPU probe failed; assuming reference-class hardware")
        return _REFERENCE_CPU_MARK

def _detect_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    try:  # stdlib fallback (Linux/macOS); Windows lands on the default
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except Exception:
        return 8.0


def _detect_cuda() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return True
    except Exception:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


_PROFILE: Optional[HardwareProfile] = None


def get_profile(refresh: bool = False) -> HardwareProfile:
    """Return this machine's cached profile, probing it on first use."""
    global _PROFILE
    if _PROFILE is not None and not refresh:
        return _PROFILE

    if not refresh and _profile_path().exists():
        try:
            data = json.loads(_profile_path().read_text(encoding="utf-8"))
            _PROFILE = HardwareProfile(**data)
            return _PROFILE
        except Exception:
            logger.warning("hardware profile at %s unreadable; re-probing", _profile_path())

    profile = HardwareProfile(
        cpu_count=os.cpu_count() or 4,
        has_cuda=_detect_cuda(),
        ram_gb=round(_detect_ram_gb(), 1),
        cpu_mark=round(_measure_cpu_mark(), 1),
    )
    _PROFILE = profile
    _save_profile(profile)
    logger.info(
        "hardware: %d cores, cuda=%s, %.1f GB RAM, cpu_mark=%.0f (%.2fx reference), %d ASR workers",
        profile.cpu_count, profile.has_cuda, profile.ram_gb,
        profile.cpu_mark, profile.speed, profile.asr_workers,
    )
    return profile


def _save_profile(profile: HardwareProfile) -> None:
    try:
        _profile_path().parent.mkdir(parents=True, exist_ok=True)
        _profile_path().write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    except OSError:
        logger.debug("could not persist hardware profile (non-fatal)", exc_info=True)


def record_observation(key: str, rtf: float) -> None:
    """Fold a measured real-time factor into the persisted profile.

    ``key`` identifies the exact configuration that produced it (e.g.
    ``"asr:small:beam1"``), so switching models never pollutes the history of a
    different one.
    """
    if not math.isfinite(rtf) or rtf <= 0:
        return
    profile = get_profile()
    previous = profile.observed_rtf.get(key)
    if previous is None:
        blended = rtf
    else:
        alpha = _OBSERVATION_ALPHA_SLOWER if rtf > previous else _OBSERVATION_ALPHA_FASTER
        blended = (1 - alpha) * previous + alpha * rtf
    profile.observed_rtf[key] = round(blended, 5)
    _save_profile(profile)
    logger.debug("recorded rtf %s=%.4f (was %s)", key, blended, previous)


def _rtf(
    key: str, reference_rtf: float, profile: HardwareProfile,
    cost_override: "dict[str, float] | None" = None,
) -> float:
    """Predicted per-unit cost for a config key.

    Resolution order:
      1. ``cost_override[key]`` — a figure calibrated for a *specific* worker
         class, passed by the distributed scheduler when it plans a job "as if
         it ran on class X" instead of on the machine doing the planning.
      2. this machine's own measured figure (``profile.observed_rtf``).
      3. the shipped reference, scaled by the probed CPU speed.
    """
    if cost_override is not None and key in cost_override:
        return cost_override[key]
    observed = profile.observed_rtf.get(key)
    if observed is not None:
        return observed
    return reference_rtf / profile.speed


# --- Configuration ladders --------------------------------------------------------
#
# Ordered best-quality first. The planner walks down each ladder until a rung
# is predicted to fit its slice of the budget, and falls back to the last rung
# if nothing does.


def asr_key(model: str, beam_size: int) -> str:
    """Calibration key for ASR cost.

    The ``:speech`` suffix marks the units: seconds of processing per second
    of *speech*, not of video. Older profiles stored a per-video-second figure
    under the unsuffixed key, and the two are not comparable — the suffix
    retires those observations instead of silently mixing them in.
    """
    return f"asr:{model}:beam{beam_size}:speech"


@dataclass(frozen=True)
class AsrRung:
    key: str
    model: str
    beam_size: int
    reference_sec_per_speech_sec: float
    label: str


# Costs are per second of **speech**, not of video: Whisper's VAD skips
# silence, so a continuous lecture costs several times what a pause-heavy talk
# of the same length does. Measuring per video-second produced a figure that
# did not transfer between videos and left the planner badly optimistic.
#
# Reference machine, four parallel workers, measured against dense narration —
# the case the budget has to survive. Beam search is the single most expensive
# setting and buys little on clear speech, which is why beam 1 is the default.
ASR_LADDER: tuple[AsrRung, ...] = (
    AsrRung(asr_key("small", 5), "small", 5, 0.95, "Whisper small (beam 5)"),
    AsrRung(asr_key("small", 1), "small", 1, 0.33, "Whisper small"),
    AsrRung(asr_key("base.en", 1), "base.en", 1, 0.16, "Whisper base"),
    AsrRung(asr_key("tiny.en", 1), "tiny.en", 1, 0.14, "Whisper tiny"),
)

@dataclass(frozen=True)
class SummarizerRung:
    key: str
    model: str
    reference_sec_per_chunk: float
    label: str


# Summarization cost is driven by the number of generation passes, not by video
# length directly, so these are per-chunk seconds rather than a real-time
# factor. Measured on the reference machine: bart-large-cnn takes 64.9 s for a
# single 900-token chunk but 24.1 s/chunk in batches of four — batching alone is
# a 2.7x win, and it was previously disabled on CPU. The distilled variants
# scale roughly with decoder depth.
SUMMARIZER_LADDER: tuple[SummarizerRung, ...] = (
    SummarizerRung("sum:bart-large-cnn", "facebook/bart-large-cnn", 24.1, "BART-large-CNN"),
    SummarizerRung("sum:distilbart-cnn-12-6", "sshleifer/distilbart-cnn-12-6", 13.3, "DistilBART 12-6"),
    SummarizerRung("sum:distilbart-cnn-6-6", "sshleifer/distilbart-cnn-6-6", 8.0, "DistilBART 6-6"),
)


def estimate_chunk_count(duration_sec: float, max_input_tokens: int) -> int:
    """Predict how many map-stage chunks a video of this length will produce.

    Speech runs about ``WORDS_PER_MINUTE`` words a minute, each word is roughly
    ``TOKENS_PER_WORD`` tokens, and on-screen text read from keyframes inflates
    that further. The chunker carries an overlap into each new window, so the
    effective advance per chunk is the window minus that overlap.
    """
    words = duration_sec / 60.0 * config.SPEECH_WORDS_PER_MINUTE
    tokens = words * config.TOKENS_PER_WORD * config.OCR_TOKEN_INFLATION
    advance = max(max_input_tokens - config.SUMMARIZER_CHUNK_OVERLAP_TOKENS, 100)
    return max(1, math.ceil(tokens / advance))


@dataclass
class ExecutionPlan:
    """Every knob the pipeline reads for one specific video."""

    duration_sec: float
    budget_sec: float

    # ingest
    download_height: int
    # asr
    asr_model: str
    asr_beam_size: int
    asr_workers: int
    asr_threads_per_worker: int
    asr_chunk_sec: float
    # vision
    scene_downscale: int
    scene_frame_skip: int
    use_sampled_segmentation: bool
    max_keyframes: int
    ocr_enabled: bool
    ocr_max_frames: int
    yolo_batch_size: int
    clip_batch_size: int
    # summarize
    summarizer_model: str
    summarizer_batch_size: int
    max_chunks: int
    # bookkeeping
    predicted_sec: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def predicted_total_sec(self) -> float:
        return sum(self.predicted_sec.values())

    def describe(self) -> str:
        return (
            f"{self.asr_model} ASR x{self.asr_workers} · "
            f"{self.summarizer_model.split('/')[-1]} · "
            f"<={self.max_keyframes} keyframes · "
            f"OCR {'on' if self.ocr_enabled else 'off'} · "
            f"~{self.predicted_total_sec / 60:.1f} min predicted"
        )


def ocr_key() -> str:
    """OCR calibration key, versioned by input width. Mirrors core.vision."""
    return f"ocr:per_frame:w{config.OCR_MAX_WIDTH}"


def _vision_seconds(
    duration: float, sampled: bool, max_keyframes: int,
    ocr_frames: int, profile: HardwareProfile,
    cost_override: "dict[str, float] | None" = None,
) -> float:
    """Predicted wall-clock for the whole vision stage under these knobs.

    Content-based detection is charged against video length because it must
    decode every frame; downscaling and frame-skipping reduce the *comparison*
    cost but not the decode, so they barely move this number. Sampled
    segmentation is charged per frame instead, which is the entire point of it.
    """
    if sampled:
        segment_sec = (
            min(max_keyframes * 2, config.SAMPLED_SEGMENTATION_MAX_SAMPLES)
            * _rtf("segment:sampled:per_frame", config.SEGMENT_REFERENCE_SEC_PER_FRAME, profile, cost_override)
        )
    else:
        segment_sec = (
            _rtf("scene:detect", config.SCENE_DETECT_REFERENCE_RTF, profile, cost_override) * duration
            + max_keyframes * config.KEYFRAME_REFERENCE_SEC_PER_FRAME / profile.speed
        )
    tag_sec = max_keyframes * _rtf("tag:per_frame", config.TAG_REFERENCE_SEC_PER_FRAME, profile, cost_override)
    ocr_sec = ocr_frames * _rtf(ocr_key(), config.OCR_REFERENCE_SEC_PER_FRAME, profile, cost_override)
    return segment_sec + tag_sec + ocr_sec


def plan_for(
    duration_sec: float,
    budget_sec: Optional[float] = None,
    profile: Optional[HardwareProfile] = None,
    quality_floor: bool = False,
    cost_override: "dict[str, float] | None" = None,
) -> ExecutionPlan:
    """Choose the heaviest configuration predicted to fit the time budget.

    Args:
        duration_sec: Length of the video being processed.
        budget_sec: Wall-clock target. Defaults to ``config.TARGET_PIPELINE_SEC``.
        profile: Hardware profile; probed if omitted.
        quality_floor: When True, pin both ladders to their top rung and keep
            all optional work. Used by the UI's "Quality" profile, where the
            user has explicitly traded the time budget away for accuracy.
        cost_override: ``{cost_key: seconds_per_unit}`` calibrated for a
            particular worker class. Supplied by the distributed scheduler so
            it can predict a job's cost on each class in its fleet, not just on
            the machine planning. ``None`` = plan for this machine (default).

    Returns:
        A fully-populated :class:`ExecutionPlan`.
    """
    profile = profile or get_profile()
    budget = budget_sec if budget_sec is not None else config.TARGET_PIPELINE_SEC
    duration = max(duration_sec, 1.0)
    # Speech, not video length, is what ASR is charged for. The true ratio
    # is unknown before the audio is analysed, so budget against a
    # deliberately high assumption: overestimating means finishing early,
    # underestimating means breaking the promise.
    speech_estimate = duration * config.ASR_ASSUMED_SPEECH_RATIO
    notes: list[str] = []

    # Budget split across the critical path. transcribe and vision run
    # concurrently, so they share one "analyze" slice rather than each getting
    # their own.
    analyze_budget = budget * config.BUDGET_SPLIT["analyze"]
    summarize_budget = budget * config.BUDGET_SPLIT["summarize"]

    # --- ASR rung ---------------------------------------------------------
    if quality_floor:
        chosen_asr = ASR_LADDER[0]
    else:
        chosen_asr = ASR_LADDER[-1]
        for rung in ASR_LADDER:
            if _rtf(rung.key, rung.reference_sec_per_speech_sec, profile, cost_override) * speech_estimate <= analyze_budget:
                chosen_asr = rung
                break
        else:
            notes.append(
                "This video is long enough that even the fastest transcription "
                "model may overrun the target; accuracy was traded for speed."
            )
    asr_predicted = _rtf(chosen_asr.key, chosen_asr.reference_sec_per_speech_sec, profile, cost_override) * speech_estimate

    # --- Summarizer rung --------------------------------------------------
    # Cost here is (number of chunks) x (seconds per chunk), and the chunk
    # count is itself a lever: widening the window packs the same transcript
    # into fewer generation passes. The planner therefore searches models
    # first, then falls back to widening, so accuracy is given up only after
    # granularity already has been.
    # Natural chunk count at the default window, before any widening.
    est_chunks = estimate_chunk_count(duration, config.SUMMARIZER_MAX_INPUT_TOKENS)

    if quality_floor:
        chosen_sum = SUMMARIZER_LADDER[0]
    else:
        chosen_sum = SUMMARIZER_LADDER[-1]
        for rung in SUMMARIZER_LADDER:
            per_chunk = _rtf(rung.key, rung.reference_sec_per_chunk, profile, cost_override)
            if per_chunk * est_chunks <= summarize_budget:
                chosen_sum = rung
                break

    sum_per_chunk = _rtf(chosen_sum.key, chosen_sum.reference_sec_per_chunk, profile, cost_override)

    # The section-by-section walkthrough is the primary output and always
    # covers the whole video. Its length is chosen from the duration — one
    # section per SUMMARY_SECTION_TARGET_MINUTES — not from the time budget.
    # A long lecture therefore takes longer here rather than losing content;
    # the Report tab shows the predicted cost honestly, and the planner pays
    # for it by trading away optional work (OCR, keyframes) elsewhere.
    target_sections = round(duration / 60.0 / config.SUMMARY_SECTION_TARGET_MINUTES)
    ceiling = config.SUMMARY_MAX_SECTIONS
    if not quality_floor and budget <= config.FAST_PROFILE_BUDGET_SEC:
        # Fast profile: keep the walkthrough short so the run stays quick,
        # roughly what the summarize slice of the budget can afford.
        affordable = int(summarize_budget / max(sum_per_chunk, 0.5))
        ceiling = max(config.SUMMARY_MIN_SECTIONS, min(ceiling, affordable))
    max_chunks = int(max(
        config.SUMMARY_MIN_SECTIONS,
        min(ceiling, target_sections),
    ))

    planned_chunks = min(est_chunks, max_chunks)
    # +1 for the reduce pass that writes the overall summary from the sections.
    sum_predicted = sum_per_chunk * (planned_chunks + config.SUMMARIZE_TAIL_SEC)

    # With a provider configured, core.understand replaces the local map-reduce
    # entirely, and costs something quite different: one remote call (or a few,
    # windowed) instead of a BART pass per chunk. Budgeting the local ladder for
    # a stage that will not run inflated the estimate by minutes and made the
    # planner downshift ASR to buy room it did not need.
    if config.GEMINI_API_KEY:
        learned = profile.observed_rtf.get("understand:per_min")
        if cost_override and "understand:per_min" in cost_override:
            learned = cost_override["understand:per_min"]
        if learned:
            sum_predicted = learned * duration / 60.0
        else:
            sum_predicted = (
                config.UNDERSTAND_REFERENCE_FIXED_SEC
                + config.UNDERSTAND_REFERENCE_SEC_PER_MIN * duration / 60.0
            )
    # --- Vision knobs -----------------------------------------------------
    # Scene-detection cost scales with decoded pixels; downscale and frame-skip
    # each cut it near-linearly, so longer videos get more aggressive values.
    # Frame-accurate detection is worth its linear cost only while that cost is
    # small; past this length the sampled path gives comparable boundaries for
    # the capped number of keyframes we keep, at a cost that no longer grows.
    sampled = duration > config.SCENE_DETECT_MAX_DURATION_SEC and not quality_floor
    downscale, frame_skip = (1, 0) if quality_floor else (2, 1)

    # Keyframe count drives YOLO + CLIP + OCR + the chapter grid. It is capped
    # rather than scaled with duration, so a 90-minute video costs the same
    # here as a 10-minute one.
    # Keyframes scale with length, then hit a ceiling. A cap alone made a
    # one-minute clip pay for 48 keyframes — one every 1.4 seconds — and every
    # per-frame cost downstream (YOLO, CLIP, OCR, the Scenes gallery) with it.
    ocr_enabled = config.OCR_ENABLED

    # One keyframe per KEYFRAME_TARGET_INTERVAL_SEC is what keeps on-screen
    # material readable; the cap is only there to bound cost. Vision runs
    # concurrently with ASR, so frames are effectively free until vision
    # becomes the critical path — spend that slack rather than leaving a long
    # video with the same handful of frames a short one gets.
    ideal_keyframes = max(
        config.MIN_KEYFRAMES, int(duration / config.KEYFRAME_TARGET_INTERVAL_SEC)
    )
    vision_budget = max(asr_predicted, analyze_budget)
    max_keyframes = int(min(ideal_keyframes, config.MAX_KEYFRAMES_ABSOLUTE))

    def _vision_at(frames: int) -> float:
        return _vision_seconds(
            duration, sampled, frames,
            min(config.OCR_MAX_FRAMES, frames) if ocr_enabled else 0,
            profile, cost_override,
        )

    while max_keyframes > config.MAX_KEYFRAMES and _vision_at(max_keyframes) > vision_budget:
        max_keyframes = int(max_keyframes * 0.8)
    max_keyframes = max(max_keyframes, config.MIN_KEYFRAMES)

    ocr_max_frames = min(config.OCR_MAX_FRAMES, max_keyframes)
    vision_predicted = _vision_at(max_keyframes)

    # Vision shares the analyze budget with ASR. If it would become the
    # critical path instead, shed optional work rather than blow the budget.
    if vision_predicted > analyze_budget and not quality_floor:
        ocr_per_frame = _rtf(ocr_key(), config.OCR_REFERENCE_SEC_PER_FRAME, profile, cost_override)
        fixed_cost = _vision_seconds(duration, sampled, max_keyframes, 0, profile, cost_override)
        affordable = int((analyze_budget - fixed_cost) / max(ocr_per_frame, 1e-6))
        if affordable < config.OCR_MIN_USEFUL_FRAMES:
            ocr_enabled = False
            notes.append("On-screen text reading (OCR) was skipped to stay inside the time budget.")
            ocr_max_frames = 0
        else:
            ocr_max_frames = min(ocr_max_frames, affordable)
            notes.append(
                f"On-screen text was read from the {ocr_max_frames} most representative frames only."
            )
        vision_predicted = _vision_seconds(
            duration, sampled, max_keyframes, ocr_max_frames if ocr_enabled else 0, profile,
        )

    # --- Ingest -----------------------------------------------------------
    # Analysis only needs enough resolution for object detection and readable
    # slide text; pixels beyond 720p cost bandwidth and decode time that no
    # stage spends. Long videos drop to 480p, where the download itself is
    # usually the binding constraint.
    download_height = 720 if duration <= 1800 else 480

    plan = ExecutionPlan(
        duration_sec=duration,
        budget_sec=budget,
        download_height=download_height,
        asr_model=chosen_asr.model,
        asr_beam_size=chosen_asr.beam_size,
        asr_workers=profile.asr_workers,
        asr_threads_per_worker=profile.asr_threads_per_worker,
        asr_chunk_sec=max(config.ASR_MIN_CHUNK_SEC, duration / max(profile.asr_workers, 1)),
        scene_downscale=downscale,
        scene_frame_skip=frame_skip,
        use_sampled_segmentation=sampled,
        max_keyframes=max_keyframes,
        ocr_enabled=ocr_enabled,
        ocr_max_frames=ocr_max_frames,
        yolo_batch_size=config.YOLO_BATCH_SIZE,
        clip_batch_size=config.CLIP_BATCH_SIZE,
        summarizer_model=chosen_sum.model,
        summarizer_batch_size=(
            config.SUMMARIZER_BATCH_SIZE_GPU if profile.has_cuda else config.SUMMARIZER_BATCH_SIZE_CPU
        ),
        max_chunks=max_chunks,
        notes=notes,
    )
    plan.predicted_sec = {
        # Download-bound, so it scales with the network rather than the CPU:
        # use the measured per-minute cost when there is one and do not divide
        # it by the CPU probe.
        "ingest": round(
            (
                _learned_ingest_per_min(profile, cost_override) * duration / 60.0
                if _learned_ingest_per_min(profile, cost_override)
                else config.INGEST_REFERENCE_SEC_PER_MIN * duration / 60 / profile.speed
            ),
            1,
        ),
        "analyze": round(max(asr_predicted, vision_predicted), 1),
        "summarize": round(sum_predicted, 1),
        "index": round(config.INDEX_REFERENCE_SEC_PER_MIN * duration / 60 / profile.speed, 1),
    }
    # Tell the user plainly when the full walkthrough pushes past the target,
    # rather than letting the run just take longer than promised silently.
    over_by = plan.predicted_total_sec - budget
    if over_by > 45 and not quality_floor:
        notes.append(
            f"This video is long enough that a full section-by-section "
            f"walkthrough ({max_chunks} sections) takes about "
            f"{plan.predicted_total_sec / 60:.0f} minutes — past the "
            f"{budget / 60:.0f}-minute target. Pick the Fast profile for a "
            f"quicker, lighter pass."
        )
    plan.notes = notes
    logger.info("plan for %.1f min video: %s", duration / 60, plan.describe())
    return plan


def plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    """Serialise a plan for persistence in meta.json and display in the UI."""
    return asdict(plan)


def plan_from_dict(data: dict[str, Any]) -> ExecutionPlan:
    """Rebuild a plan from its serialised form, ignoring unknown keys.

    Tolerating unknown keys keeps a cache written by an older build readable
    after the plan gains a field, instead of turning a schema change into a
    hard failure on every previously-processed video.
    """
    known = {f for f in ExecutionPlan.__dataclass_fields__}
    return ExecutionPlan(**{k: v for k, v in data.items() if k in known})


if __name__ == "__main__":
    profile = get_profile(refresh=True)
    print(json.dumps(asdict(profile), indent=2))
    for minutes in (2, 10, 25, 45, 60, 90):
        p = plan_for(minutes * 60)
        print(f"\n{minutes:>3} min video -> {p.describe()}")
        print(f"          predicted: {p.predicted_sec}")
        for note in p.notes:
            print(f"          note: {note}")
