"""Single source of truth for every tunable constant in the pipeline.

No module outside this file should hardcode a magic number that governs
model choice, thresholds, or sizing. Import from here instead.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# torch (via MKL) and ctranslate2 (via faster-whisper) each bundle their own
# OpenMP runtime; loading both in one process crashes with "OMP Error #15:
# Initializing libiomp5md.dll, but found libiomp5md.dll already initialized"
# on Windows. This must be set before either library is imported anywhere,
# so it lives at the very top of config.py, which every module imports first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv

load_dotenv()

# Offline mode. Hugging Face checks for model updates on every load, so a
# flaky or absent network turns an otherwise fully-cached run into minutes of
# retry loops before it falls back to the local copy — which matters most in
# exactly the situation where it hurts: demoing on conference wifi. Setting
# VS_OFFLINE=1 skips those checks and uses the local cache directly.
if os.getenv("VS_OFFLINE", "").strip().lower() in {"1", "true", "yes"}:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# --- Logging -----------------------------------------------------------
LOG_LEVEL = os.getenv("VS_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# --- Paths ---------------------------------------------------------------
# Anchored to this file's directory rather than the process working
# directory: a relative "./cache" silently points somewhere else when the
# app is launched from a parent directory (e.g.
# `streamlit run Project_Build/app.py`), which makes an already-processed
# video look like it was never processed. An explicit VS_CACHE_DIR is
# honoured exactly as given, absolute or relative.
PROJECT_ROOT = Path(__file__).resolve().parent
_cache_override = os.getenv("VS_CACHE_DIR")
CACHE_ROOT = Path(_cache_override) if _cache_override else PROJECT_ROOT / "cache"

# --- Ingestion -------------------------------------------------------------
# Hard ceiling on input length. The adaptive planner downshifts models to hold
# the time budget, so length is a resource question rather than a capability
# one: a 15-hour source means roughly 10 GB of video, a 1.7 GB wav, and an
# hour-plus of CPU transcription, and its transcript has to be summarised into
# the same UNDERSTANDING_MAX_OUTPUT_TOKENS as a 10-minute one. Tune with
# VS_MAX_VIDEO_MIN.
MAX_VIDEO_DURATION_SEC = float(os.getenv("VS_MAX_VIDEO_MIN", "900")) * 60
AUDIO_SAMPLE_RATE = 16000  # Whisper requirement
AUDIO_CHANNELS = 1
INGEST_MAX_RETRIES = 2
# yt-dlp fetches one fragment at a time by default, which leaves most of the
# available bandwidth unused on DASH/HLS sources. Long videos are frequently
# download-bound, so this is the single biggest ingest win.
YT_DLP_CONCURRENT_FRAGMENTS = 8
# Optional: browser to pull YouTube auth cookies from (e.g. "chrome", "edge",
# "firefox") when a video still hits bot-check / sign-in-required after a
# yt-dlp upgrade (age-restricted or region-locked videos). Unset by default.
YT_DLP_COOKIES_FROM_BROWSER = os.getenv("YT_DLP_COOKIES_FROM_BROWSER") or None

# --- ASR ---------------------------------------------------------------
# The ASR model and beam size are chosen per video by core/perf.py's ladder
# rather than fixed here: a global default would defeat the point of planning
# against a time budget. Only the numeric format stays global.
WHISPER_COMPUTE_TYPE = "int8"  # int8 on CPU, float16 on GPU
PREFER_YOUTUBE_CAPTIONS = True  # try captions before running Whisper

# Long audio is split at natural silences and transcribed by several Whisper
# processes at once (see core/transcribe.py). These bound that split: chunks
# shorter than the minimum aren't worth a process launch, and the search
# window is how far from an ideal cut point we'll look for a silence to land
# on so a split never lands mid-word.
ASR_MIN_CHUNK_SEC = 120.0
ASR_SILENCE_SEARCH_WINDOW_SEC = 20.0
ASR_SILENCE_NOISE_DB = -30  # ffmpeg silencedetect threshold
ASR_SILENCE_MIN_DUR_SEC = 0.35
# Below this duration, parallel chunking costs more in process startup and
# model loading than the overlap saves.
ASR_PARALLEL_MIN_DURATION_SEC = 240.0
# Fraction of a video assumed to be speech when planning, before the audio has
# been analysed. Whisper's VAD skips silence, so its cost tracks speech: a
# continuous lecture costs several times what a pause-heavy talk of the same
# length does. Set high on purpose — overestimating finishes early, and
# underestimating breaks the time budget.
ASR_ASSUMED_SPEECH_RATIO = 0.92
# Whisper sizes that YouTube's auto-generated captions are expected to beat.
# Auto-captions carry no punctuation, which hurts summarization, so they are
# only used once the planner has already downshifted to a model that would be
# worse still — below these sizes, captions win on both speed and accuracy.
CAPTIONS_PREFERRED_OVER_MODELS = ("tiny", "tiny.en", "base", "base.en")
# Language handling. Leaving this unset auto-detects from several probes
# spread across the audio (see core/transcribe.py), rather than trusting
# Whisper's own detection, which reads only the opening window — often an
# intro jingle or sponsor read.
#
# This used to be hardcoded to "en". That silently destroyed any non-English
# video: forced to decode Hindi-English speech as English, Whisper produced
# transliterated nonsense and ran several times slower, which then poisoned
# every downstream stage. Set VS_WHISPER_LANGUAGE to force a specific
# language when you know it.
WHISPER_LANGUAGE = os.getenv("VS_WHISPER_LANGUAGE") or None
# Language detection needs a multilingual checkpoint: the ".en" models are
# English-only and report "en" for any input, so probing with them would
# always agree with itself. "tiny" is multilingual and ~75 MB.
WHISPER_LANGUAGE_DETECT_MODEL = "tiny"
WHISPER_LANGUAGE_PROBES = 3
WHISPER_LANGUAGE_PROBE_SEC = 30.0
# Non-English audio is translated to English rather than transcribed in its
# own language, because every downstream model (BART summarization, flan-t5
# Q&A, MiniLM retrieval) is English-only. A Hindi transcript would be
# faithful and completely unusable by the rest of the pipeline.
WHISPER_TRANSLATE_NON_ENGLISH = True

# --- Vision --------------------------------------------------------------
SCENE_DETECT_THRESHOLD = 27.0  # PySceneDetect ContentDetector
MIN_SCENE_LEN_SEC = 2.0
FIXED_INTERVAL_FALLBACK_SEC = 60.0  # used when scene detection finds < 2 scenes
KEYFRAME_DEDUP_SIMILARITY = 0.95  # CLIP cosine; above this = duplicate
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
# Detection resolution. Only the top few *labels* per keyframe are kept —
# never the boxes — so full 640px inference buys precision the pipeline
# throws away. Measured at 0.089 s/frame here against 0.177 s at 640.
YOLO_IMAGE_SIZE = 416
YOLO_MAX_OBJECTS_PER_FRAME = 5
YOLO_BATCH_SIZE = 32
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_BATCH_SIZE = 32

# --- M7: learned scene-importance / highlight model (ml/) --------------------
# Trained offline on SumMe + TVSum. When a checkpoint is present, the vision
# stage attaches an ``importance`` score (0-1) to every scene and an
# ``importance_ranked`` scene order to scenes.json; the LLM/summarizer then
# prefer high-importance scenes. Absent a checkpoint the pipeline is unchanged.
# VS_HIGHLIGHT_MODEL overrides the path; "off" / "0" disables it outright.
_highlight_env = os.getenv("VS_HIGHLIGHT_MODEL", "").strip()
if _highlight_env.lower() in {"off", "0", "false", "no"}:
    HIGHLIGHT_MODEL_PATH = None
elif _highlight_env:
    HIGHLIGHT_MODEL_PATH = _highlight_env
else:
    _default_ckpt = PROJECT_ROOT / "ml" / "checkpoints" / "importance.pt"
    HIGHLIGHT_MODEL_PATH = str(_default_ckpt) if _default_ckpt.exists() else None

# Highlight reel: greedy-pick top-importance scenes up to this many seconds.
HIGHLIGHT_MAX_SECONDS = float(os.getenv("VS_HIGHLIGHT_MAX_SEC", "90"))
# Build the reel automatically at the end of a pipeline run (needs a checkpoint).
HIGHLIGHT_AUTO_BUILD = os.getenv("VS_HIGHLIGHT_AUTO", "1").strip().lower() in {"1", "true", "yes"}
# Hard cap on keyframes. Every per-scene cost downstream (YOLO, CLIP, OCR,
# the chapter grid, the Scenes gallery) scales with this number, so capping it
# is what makes a 90-minute video cost the same in the vision stage as a
# 10-minute one. Scenes beyond the cap are merged, shortest-first, so the
# timeline stays complete — only its granularity drops.
MAX_KEYFRAMES = 48
# Absolute ceiling, to bound disk and memory. MAX_KEYFRAMES above is the
# *starting* target; the planner raises it toward this when vision has room,
# because vision runs concurrently with ASR and is not the critical path on a
# long video. A flat 48 gave a 12-hour lecture one keyframe every 15 minutes —
# the same budget a 10-minute clip gets — which is why long videos lost most
# of their on-screen material.
MAX_KEYFRAMES_ABSOLUTE = int(os.getenv("VS_MAX_KEYFRAMES", "400"))
# ...but never more than roughly one per this many seconds, and never
# fewer than MIN_KEYFRAMES. Without the interval a short clip paid the
# full 48-keyframe cost for a minute of footage.
KEYFRAME_TARGET_INTERVAL_SEC = 45.0
MIN_KEYFRAMES = 6
# Above this length, scene segmentation switches from frame-accurate content
# detection to CLIP-embedding comparison over sampled frames, whose cost is
# fixed rather than proportional to the video. See core/vision.py.
# Content-based detection measured 153 s on an 8.8-minute video even at
# downscale 2 — it decodes nearly every frame, so it is only affordable on
# genuinely short clips. Sampled segmentation cost ~40 s on a 67-minute video.
SCENE_DETECT_MAX_DURATION_SEC = 300.0   # 5 minutes
SAMPLED_SEGMENTATION_MIN_SAMPLES = 24
SAMPLED_SEGMENTATION_MAX_SAMPLES = 96
# Per-frame costs measured over fewer frames than this are dominated by model
# loading rather than throughput, so they are not recorded as calibration.
MIN_FRAMES_FOR_CALIBRATION = 8
MIN_CHUNKS_FOR_CALIBRATION = 3
# Timings from very short videos are dominated by model loading rather
# than throughput, so they are not recorded as calibration.
MIN_DURATION_FOR_CALIBRATION_SEC = 300.0

OCR_ENABLED = True  # read on-screen text (slides, whiteboards) from keyframes
OCR_LANGUAGES = ["en"]
OCR_CONF_THRESHOLD = 0.4
OCR_MIN_TEXT_LEN = 3  # drop single-character noise detections
# On-screen text is only fed to the summarizer when it looks like slide prose
# rather than interface chrome; it is always kept on the scene for display.
OCR_MIN_WORDS_FOR_SUMMARY = 6
OCR_MAX_WORDS_FOR_SUMMARY = 40
OCR_MAX_FRAMES = 12  # planner lowers this further when the budget is tight
OCR_MIN_USEFUL_FRAMES = 6  # below this, skip OCR entirely rather than sample it
# EasyOCR's cost is dominated by input pixels. Slide text stays legible well
# below source resolution, so keyframes are downscaled to this width first.
OCR_MAX_WIDTH = 640
# The highest-ranked frames are read first as a probe: if they contain almost
# no text, the video has no slides worth reading and the rest are skipped.
OCR_PROBE_FRAMES = 3
OCR_PROBE_MIN_CHARS = 40

# --- Summarization ---------------------------------------------------------
# Default summarizer. The planner picks a distilled variant for long videos
# (see core/perf.py SUMMARIZER_LADDER); this is the value used when no plan is
# available, e.g. the module's own __main__ debug entrypoint.
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
SUMMARIZER_MAX_INPUT_TOKENS = 900  # BART limit is 1024; leave headroom
# Upper bound the map stage may widen its window to when a long transcript
# would otherwise produce more chunks than the time budget allows. Still under
# BART's 1024 positional limit.
SUMMARIZER_MAX_INPUT_TOKENS_CEILING = 1000
# Kept small: a larger overlap makes adjacent walkthrough sections restate
# each other, since both get summarized from the shared text.
SUMMARIZER_CHUNK_OVERLAP_TOKENS = 45
# Per-section summary length. Each walkthrough section should read as a
# real paragraph, not a one-line gist — a one-hour lecture deserves more
# than 100 words of explanation total.
CHUNK_SUMMARY_MAX_LEN = 240
CHUNK_SUMMARY_MIN_LEN = 90
# The overall summary is the short "key points" read at the end, so it
# stays tight even when the walkthrough above it is long.
FINAL_SUMMARY_MAX_LEN = 360
FINAL_SUMMARY_MIN_LEN = 140

# The walkthrough aims for roughly one section per this many minutes of
# video, clamped to [MIN, MAX]. A target, not a budget cap: the whole
# transcript is always summarized, so a long video takes longer rather
# than losing content.
SUMMARY_SECTION_TARGET_MINUTES = 2.5
SUMMARY_MIN_SECTIONS = 3
SUMMARY_MAX_SECTIONS = 40
# Budgets at or below this are treated as the "Fast" profile: the walkthrough
# is capped to what the summarize slice can afford, rather than covering every
# section of a long video.
FAST_PROFILE_BUDGET_SEC = 180.0
TARGET_CHAPTER_COUNT = 8  # approximate
# Chapters are assembled from the map stage's chunk summaries rather than by
# re-summarizing the raw transcript a second time. Set True only to compare
# against the old behaviour; it roughly doubles summarization cost.
CHAPTERS_RESUMMARIZE_RAW = False
# Chapter titles: flan-t5 generates all of them in one batched pass, which
# costs a model load plus a single forward pass for the whole set. Turning this
# off falls back to an extractive title taken from each chapter's own summary —
# free, and still readable, but noticeably blunter.
CHAPTER_TITLES_USE_LLM = True
CHAPTER_TITLE_BATCH_SIZE = 8
# Below this ratio of unique words a generated title has looped, not titled.
TITLE_MIN_UNIQUE_WORD_RATIO = 0.6
SUMMARIZER_BATCH_SIZE_GPU = 8
# Batching seq2seq generation on CPU was previously disabled (batch size 1).
# It is a large win — the map stage is many independent chunks of near-equal
# length, so padding waste is small and the matmuls get much better use of
# the cores.
SUMMARIZER_BATCH_SIZE_CPU = 8
CHAPTER_TITLE_MAX_WORDS = 8
# A chapter blurb longer than this gets condensed; below it, the map stage's
# summaries are shown as-is rather than paying for another generation pass.
CHAPTER_SUMMARY_MAX_WORDS = 90

# --- RAG / QA --------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_TOKENS = 400
RAG_CHUNK_OVERLAP_TOKENS = 80
RAG_TOP_K = 5
# Cosine similarity between a MiniLM query embedding and a transcript passage
# sits far below 1 even for a correct match — a broad question ("what is this
# video about?") against the passage that actually answers it measures ~0.12.
# The old 0.25 threshold therefore marked almost every correct answer as
# low-confidence and prefixed it with a hedge. This is calibrated to separate
# "retrieval found something related" from "retrieval found nothing".
RAG_LOW_CONFIDENCE_THRESHOLD = 0.10   # below this, refuse to answer at all
RAG_HIGH_CONFIDENCE_THRESHOLD = 0.18  # above this, the grounding is solid
QA_LOCAL_MODEL = "google/flan-t5-base"
QA_MAX_NEW_TOKENS = 200
QA_MIN_NEW_TOKENS = 12   # stops flan-t5 emitting four-word fragments
QA_NUM_BEAMS = 4
QA_NO_REPEAT_NGRAM = 3
# Below this ratio of unique words the model has looped rather than answered.
QA_MIN_UNIQUE_WORD_RATIO = 0.45
# An answer with fewer real words than this is a list of markers or letters,
# not a sentence.
QA_MIN_REAL_WORDS = 4
# flan-t5-base has a 512-token encoder. Passages beyond this budget are
# retrieved and cited but not fed to the generator, which would silently
# truncate them anyway.
QA_GENERATION_CHUNKS = 3
QA_CONTEXT_MAX_TOKENS = 380

# --- Q&A with a capable model ------------------------------------------------
# The two settings above exist to fit flan-t5-base's 512-token encoder. A real
# model has no such constraint, and giving it the same cramped context is why
# its answers came back as short as the local model's. When GEMINI_API_KEY is
# set the tutor path uses these instead, and also gets the understanding
# document (explanation, glossary, code) alongside the raw transcript.
QA_LLM_CONTEXT_CHUNKS = int(os.getenv("VS_QA_CHUNKS", "10"))
QA_LLM_CONTEXT_MAX_TOKENS = int(os.getenv("VS_QA_CONTEXT_TOKENS", "12000"))
QA_LLM_MAX_OUTPUT_TOKENS = int(os.getenv("VS_QA_MAX_TOKENS", "2048"))
QA_LLM_THINKING_BUDGET = int(os.getenv("VS_QA_THINKING", "0"))
QA_HISTORY_TURNS = int(os.getenv("VS_QA_HISTORY_TURNS", "6"))
# Sections of the understanding doc to offer as context, most relevant first.
QA_UNDERSTANDING_SECTIONS = int(os.getenv("VS_QA_UNDERSTANDING_SECTIONS", "6"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Any OpenAI-compatible provider -------------------------------------------
# Groq, OpenRouter, Cerebras, Together, Mistral and a local Ollama all speak
# the same chat-completions shape, and several have a free tier with no card.
# Set these three and they take precedence over GEMINI_API_KEY. Examples:
#
#   Groq        VS_LLM_BASE_URL=https://api.groq.com/openai/v1
#               VS_LLM_MODEL=llama-3.3-70b-versatile
#   OpenRouter  VS_LLM_BASE_URL=https://openrouter.ai/api/v1
#               VS_LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
#   Ollama      VS_LLM_BASE_URL=http://localhost:11434/v1
#               VS_LLM_MODEL=llama3.1   (VS_LLM_API_KEY can be anything)
#
# These models are smaller than Gemini 2.5 Flash and are not multimodal, so
# keyframes are not sent and the explanation leans on OCR for what is on
# screen. That is a real step down — and a long way up from the offline
# BART fallback, which is the only other option when a quota runs out.
LLM_BASE_URL = os.getenv("VS_LLM_BASE_URL")
LLM_API_KEY = os.getenv("VS_LLM_API_KEY")
LLM_MODEL = os.getenv("VS_LLM_MODEL")


def _numbered_providers() -> list[dict[str, str]]:
    """Extra endpoints, as VS_LLM2_*, VS_LLM3_* and so on.

    A daily quota belongs to an account, so a second account is a second
    quota — and that is the only thing that raises the ceiling without
    spending money. One eleven-hour video fully explored is about 33 calls
    against a free allowance of 50, so one account is roughly one and a half
    videos a day; three accounts is four or five.
    """
    found: list[dict[str, str]] = []
    for n in range(2, 7):
        base = os.getenv(f"VS_LLM{n}_BASE_URL")
        key = os.getenv(f"VS_LLM{n}_API_KEY")
        model = os.getenv(f"VS_LLM{n}_MODEL")
        if base and key and model:
            found.append({"base_url": base, "api_key": key, "model": model})
    return found


LLM_EXTRA_PROVIDERS = _numbered_providers()
# Free tiers cap the completion far below Gemini's 32k, and asking for
# more is a 400 before any work happens rather than a shorter reply. Clamp
# what the OpenAI-compatible client requests, so one provider's ceiling
# does not mean retuning UNDERSTANDING_MAX_OUTPUT_TOKENS for all of them.
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("VS_LLM_MAX_OUTPUT_TOKENS", "8192"))
# Several free models accept images, so keyframes are sent by default and
# the explanation can describe what is on screen rather than inferring it
# from OCR text. A text-only model rejects them; the client notices that
# and drops them for the rest of the run, so this is only needed to skip
# the one wasted call.
LLM_SEND_IMAGES = os.getenv("VS_LLM_SEND_IMAGES", "1") not in ("0", "false", "False")
# Several free models think before answering, and that hidden reasoning is
# charged against max_tokens: a reply worth 18 tokens cost 163 and came
# back cut off mid-JSON. OpenRouter exposes a switch for it, doing the job
# that thinkingBudget already does on the Gemini client. Left blank the
# switch is sent only to OpenRouter, whose extension it is; "1" or "0"
# forces it on or off for any provider.
LLM_REASONING = os.getenv("VS_LLM_REASONING")
USE_GEMINI = GEMINI_API_KEY is not None
GEMINI_MODEL = os.getenv("VS_GEMINI_QA_MODEL", "gemini-2.5-flash")  # Q&A backend (core/qa.py)

# --- Understanding layer (core/understand.py) -----------------------------
# When a provider key is set, the pipeline uses an LLM to produce a teachable
# explanation instead of the BART/flan-t5 map-reduce. Gemini Flash: ~1M-token
# context (the whole transcript fits, no lossy chunking), fast, cheap.
# gemini-2.5-flash: stable, generally available, 1M context, cheap. A hard-pinned
# preview model (gemini-2.0-flash) got deprecated out from under us; pin a
# specific dated version via VS_UNDERSTANDING_MODEL for reproducibility.
UNDERSTANDING_MODEL = os.getenv("VS_UNDERSTANDING_MODEL", "gemini-2.5-flash")
UNDERSTANDING_TEMPERATURE = float(os.getenv("VS_UNDERSTANDING_TEMPERATURE", "0.35"))
UNDERSTANDING_MAX_OUTPUT_TOKENS = int(os.getenv("VS_UNDERSTANDING_MAX_TOKENS", "32768"))

# One output budget has to cover the whole video, so past a certain length the
# explanation thins out no matter how large that budget is: a 3-hour video gets
# the same tokens as a 10-minute one. Beyond SINGLE_PASS_MAX_MIN the video is
# understood in windows instead — each window gets the full budget, and a final
# pass writes the overview, recap, key points and glossary across them. Cost is
# one call per window plus the reduce.
# What the understanding call costs, for the planner. Fitted to real runs
# (8.8 min -> 36s, 42 min -> 57s, 76 min -> 93s): a fixed round-trip plus a
# per-minute term. Deliberately NOT scaled by the CPU probe — this stage waits
# on a remote model, so a slower laptop does not make it slower.
UNDERSTAND_REFERENCE_FIXED_SEC = 25.0
UNDERSTAND_REFERENCE_SEC_PER_MIN = 0.9

UNDERSTANDING_SINGLE_PASS_MAX_MIN = float(os.getenv("VS_UNDERSTANDING_SINGLE_PASS_MIN", "45"))
UNDERSTANDING_WINDOW_MIN = float(os.getenv("VS_UNDERSTANDING_WINDOW_MIN", "25"))
# Ceiling on how many windows a single video may be split into. Each window
# is one request, so this is really a cap on what one video may cost against
# the provider's quota. An 11-hour video at 25-minute windows is 27 requests
# plus a reduce; on a free tier that exhausts the day's allowance on one
# video, every later window then 429s, and the run falls back to the local
# summariser — which is both far worse and, on a video that long, far slower.
# Past the cap the windows simply widen.
UNDERSTANDING_MAX_WINDOWS = int(os.getenv("VS_UNDERSTANDING_MAX_WINDOWS", "16"))

# --- Problem list (core/problems.py) -------------------------------------
# Every window is a request, and the reply carries whole solutions, so this
# bounds what one video can spend. A three-hour practice video with seventy
# questions fits comfortably; past the cap the windows are spread evenly so
# the list still covers the end of the video rather than stopping halfway.
PROBLEMS_MAX_WINDOWS = int(os.getenv("VS_PROBLEMS_MAX_WINDOWS", "14"))

# Seconds to wait between window calls. A long video is many calls in quick
# succession, which is exactly what trips a per-minute quota — a 12-hour video
# is 29 of them. The pause costs far less than the retries it avoids.
UNDERSTANDING_WINDOW_PAUSE_SEC = float(os.getenv("VS_UNDERSTANDING_WINDOW_PAUSE", "2.0"))
# Reasoning-model thinking budget (output tokens spent planning before the
# visible reply). 0 = answer directly (fastest, cheapest); higher can help
# on coding videos where the model benefits from reasoning through the code.
UNDERSTANDING_THINKING_BUDGET = int(os.getenv("VS_UNDERSTANDING_THINKING", "0"))
UNDERSTANDING_THINKING_BUDGET_CODING = int(os.getenv("VS_UNDERSTANDING_THINKING_CODING", "4096"))

# Send the more informative keyframes to the (multimodal) model alongside
# the text, so it can read diagrams, handwriting and layout that OCR misses.
# Each image is ~260+ tokens; the cap keeps request size and cost bounded.
UNDERSTANDING_SEND_SCENE_IMAGES = os.getenv("VS_UNDERSTANDING_IMAGES", "1") not in ("0", "false", "no")
UNDERSTANDING_MAX_SCENE_IMAGES = int(os.getenv("VS_UNDERSTANDING_MAX_IMAGES", "12"))
UNDERSTANDING_SCENE_IMAGE_WIDTH = 768
# Fall back to the local BART/flan-t5 summarizer if the LLM call fails, rather
# than failing the whole run. Set False to surface LLM errors instead.
UNDERSTANDING_FALLBACK_TO_LOCAL = True
# Longest video for which the local fallback is still worth running when the
# LLM path fails. Beyond this it is a trap: on an 11-hour recording the
# BART map-reduce ran for most of three hours and produced text nobody could
# learn from, and the quiz, flashcards and audio overview stayed off anyway
# because they are built from the explanation it cannot write. Past this
# length the run stops with a reason instead.
LOCAL_FALLBACK_MAX_VIDEO_MIN = float(os.getenv("VS_LOCAL_FALLBACK_MAX_MIN", "90"))

# --- UI / misc -------------------------------------------------------------
TEAM_NAMES = os.getenv(
    "TEAM_NAMES", "Jaivardhan Singh, Sujay Chand, Harshit Raj"
)
GUIDE_NAME = os.getenv("GUIDE_NAME", "Dr. Deepa K")
PROJECT_TITLE = os.getenv("PROJECT_TITLE", "AI-Based Video Summarization System")

# --- Performance budget ------------------------------------------------------
#
# The pipeline targets a wall-clock budget rather than a fixed model set:
# core/perf.py picks the heaviest configuration predicted to fit. See that
# module for how the reference numbers below are used and self-calibrated.

TARGET_PIPELINE_SEC = float(os.getenv("VS_TARGET_SEC", "300"))  # 5 minutes

# The "accurate" profile's budget. It used to pin every ladder to its top rung
# with no budget at all (quality_floor), which on a CPU box turned a two-hour
# video into an all-day run — frame-accurate scene detection alone decodes
# nearly every frame. Giving it a large but real budget keeps the intent ("take
# longer, use better models") while letting the planner refuse the choices that
# cannot finish: on a short video it still lands on the top rung, on a very
# long one it degrades instead of grinding.
ACCURATE_PROFILE_BUDGET_SEC = float(
    os.getenv("VS_ACCURATE_SEC", str(TARGET_PIPELINE_SEC * 4))
)

# How the budget is divided along the critical path. transcribe and vision run
# concurrently, so they share the single "analyze" slice. Must sum to <= 1.0;
# the slack absorbs model loading and process startup.
BUDGET_SPLIT = {
    "ingest": 0.20,
    "analyze": 0.45,
    "summarize": 0.28,
    "index": 0.05,
}

# Reference costs measured on the machine described in core/perf.py. Each is
# rescaled by the probed CPU speed and then replaced outright once a real run
# reports what that stage actually cost on this machine.
# Content-based scene detection must decode every frame, so its cost is linear
# in video length: measured at 140 s for a 67-minute video even with downscale
# and frame-skip applied, because decoding — not comparing — dominates.
SCENE_DETECT_REFERENCE_RTF = 0.035    # sec per sec of video
SEGMENT_REFERENCE_SEC_PER_FRAME = 0.30  # measured warm: 0.118 seek + 0.115 CLIP, +30% for contention  # sampled segmentation: seek + CLIP per frame
KEYFRAME_REFERENCE_SEC_PER_FRAME = 0.06
TAG_REFERENCE_SEC_PER_FRAME = 0.55    # measured warm: 0.376 YOLO + 0.115 CLIP, +contention
# Measured at ~10.6 s/frame while sharing cores with parallel transcription —
# EasyOCR is by far the most expensive per-frame operation in the pipeline,
# which is why it is sampled rather than applied to every keyframe.
OCR_REFERENCE_SEC_PER_FRAME = 5.0
INGEST_REFERENCE_SEC_PER_MIN = 1.6    # download + audio extract, per video-min
INDEX_REFERENCE_SEC_PER_MIN = 0.06
# Inputs to the planner's chunk-count estimate (core.perf.estimate_chunk_count).
SPEECH_WORDS_PER_MINUTE = 150.0   # typical lecture/presentation pace
TOKENS_PER_WORD = 1.35            # BPE expansion for ordinary English
OCR_TOKEN_INFLATION = 1.25        # on-screen text folded into the same chunks
# Reduce + chapter assembly cost roughly this many extra chunk-equivalents on
# top of the map stage.
SUMMARIZE_TAIL_SEC = 3.0
SUMMARIZER_MIN_CHUNKS = 3         # never collapse a video below this granularity

# How often the orchestrator publishes merged progress while the two
# concurrent stages run. Short enough to feel live, long enough that polling
# costs nothing against a multi-minute stage.
PROGRESS_POLL_SEC = 0.4

# --- Pipeline stage progress weights (must sum to 1.0) ---------------------
#
# "analyze" covers transcribe and vision together because they now run
# concurrently and finish as one unit.
STAGE_WEIGHTS = {
    "ingest": 0.15,
    "analyze": 0.55,
    "fuse": 0.03,
    "summarize": 0.22,
    "index": 0.05,
}
