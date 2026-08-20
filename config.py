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

# --- Logging -----------------------------------------------------------
LOG_LEVEL = os.getenv("VS_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# --- Paths ---------------------------------------------------------------
CACHE_ROOT = Path(os.getenv("VS_CACHE_DIR", "./cache"))

# --- Ingestion -------------------------------------------------------------
MAX_VIDEO_DURATION_SEC = 5400  # 90 min hard limit; reject longer
FRAME_SAMPLE_FPS = 1.0  # frames extracted per second
AUDIO_SAMPLE_RATE = 16000  # Whisper requirement
AUDIO_CHANNELS = 1
INGEST_MAX_RETRIES = 2
# Optional: browser to pull YouTube auth cookies from (e.g. "chrome", "edge",
# "firefox") when a video still hits bot-check / sign-in-required after a
# yt-dlp upgrade (age-restricted or region-locked videos). Unset by default.
YT_DLP_COOKIES_FROM_BROWSER = os.getenv("YT_DLP_COOKIES_FROM_BROWSER") or None

# --- ASR ---------------------------------------------------------------
WHISPER_MODEL_SIZE = "small"  # tiny|base|small|medium
WHISPER_COMPUTE_TYPE = "int8"  # int8 on CPU, float16 on GPU
WHISPER_BEAM_SIZE = 5
PREFER_YOUTUBE_CAPTIONS = True  # try captions before running Whisper

# --- Vision --------------------------------------------------------------
SCENE_DETECT_THRESHOLD = 27.0  # PySceneDetect ContentDetector
MIN_SCENE_LEN_SEC = 2.0
FIXED_INTERVAL_FALLBACK_SEC = 60.0  # used when scene detection finds < 2 scenes
KEYFRAME_DEDUP_SIMILARITY = 0.95  # CLIP cosine; above this = duplicate
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF_THRESHOLD = 0.35
YOLO_MAX_OBJECTS_PER_FRAME = 5
YOLO_BATCH_SIZE = 16
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_BATCH_SIZE = 16

# --- Summarization ---------------------------------------------------------
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
SUMMARIZER_MAX_INPUT_TOKENS = 900  # BART limit is 1024; leave headroom
SUMMARIZER_CHUNK_OVERLAP_TOKENS = 100
CHUNK_SUMMARY_MAX_LEN = 130
CHUNK_SUMMARY_MIN_LEN = 40
FINAL_SUMMARY_MAX_LEN = 250
FINAL_SUMMARY_MIN_LEN = 80
TARGET_CHAPTER_COUNT = 8  # approximate
SUMMARIZER_BATCH_SIZE_GPU = 4
SUMMARIZER_BATCH_SIZE_CPU = 1
REDUCE_MAX_RECURSION_DEPTH = 3
CHAPTER_TITLE_MAX_WORDS = 8

# --- RAG / QA --------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAG_CHUNK_TOKENS = 400
RAG_CHUNK_OVERLAP_TOKENS = 80
RAG_TOP_K = 5
RAG_LOW_CONFIDENCE_THRESHOLD = 0.25
QA_LOCAL_MODEL = "google/flan-t5-base"
QA_MAX_NEW_TOKENS = 200
USE_GEMINI = os.getenv("GEMINI_API_KEY") is not None
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# --- UI / misc -------------------------------------------------------------
TEAM_NAMES = os.getenv(
    "TEAM_NAMES", "Jaivardhan Singh, Sujay Chand, Harshit Raj"
)
GUIDE_NAME = os.getenv("GUIDE_NAME", "Dr. Deepa K")
PROJECT_TITLE = os.getenv("PROJECT_TITLE", "AI-Based Video Summarization System")

# --- Pipeline stage progress weights (must sum to 1.0) ---------------------
STAGE_WEIGHTS = {
    "ingest": 0.15,
    "transcribe": 0.35,
    "vision": 0.25,
    "fuse": 0.03,
    "summarize": 0.17,
    "index": 0.05,
}
