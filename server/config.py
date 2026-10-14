"""Server configuration, read from the environment.

The web tier is a thin layer over the same ``core`` pipeline the Streamlit app
uses — this file only adds what a multi-user service needs on top: where the
job database lives, how many workers run, which queue backend to use, and the
upload size limit. Nothing here changes how a video is processed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Anchored to the package, like core.config.CACHE_ROOT, so the service behaves
# the same whatever directory it is launched from.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# The Streamlit/core entry points load this file already, but the FastAPI
# service can start without importing them. Load the project-local variables
# here as well so `python -m server.main` honours the same `.env` file.
load_dotenv(PACKAGE_ROOT / ".env")

# --- Storage -----------------------------------------------------------------
# Job rows, telemetry, and the scheduler's learned cost models. SQLite by
# default so the service runs with no external database; point this at a
# Postgres URL (postgresql+psycopg://...) for a real deployment — the ORM
# layer is unchanged.
DATABASE_URL = os.getenv(
    "VS_DATABASE_URL", f"sqlite:///{PACKAGE_ROOT / 'server_state.db'}"
)

# Where uploaded files and per-video artifacts are kept. Local disk by default;
# server/storage.py is the seam for swapping in S3/R2.
DATA_ROOT = Path(os.getenv("VS_DATA_ROOT", str(PACKAGE_ROOT / "server_data")))
UPLOAD_DIR = DATA_ROOT / "uploads"

MAX_UPLOAD_BYTES = int(os.getenv("VS_MAX_UPLOAD_MB", "512")) * 1024 * 1024
# Extensions accepted by POST /api/uploads (lowercase, with dot).
UPLOAD_ALLOWED_EXTS = {
    e if e.startswith(".") else f".{e}"
    for e in os.getenv("VS_UPLOAD_EXTS", ".mp4,.mov,.mkv,.webm,.avi,.m4v").lower().split(",")
    if e.strip()
}

# --- Object storage -------------------------------------------------------
# "local": uploads + artifacts stay on disk (UPLOAD_DIR, cache/).
# "s3":    uploads and finished cache/<video_id>/ trees are mirrored to an
#          S3-compatible bucket (AWS S3, Cloudflare R2, MinIO). Needs boto3 and
#          VS_S3_BUCKET; VS_S3_ENDPOINT_URL points it at R2/MinIO.
STORAGE_BACKEND = os.getenv("VS_STORAGE_BACKEND", "local").strip().lower()
S3_BUCKET = os.getenv("VS_S3_BUCKET", "").strip()
S3_ENDPOINT_URL = os.getenv("VS_S3_ENDPOINT_URL", "").strip() or None
S3_REGION = os.getenv("VS_S3_REGION", "").strip() or None
S3_PREFIX = os.getenv("VS_S3_PREFIX", "video-summarizer").strip("/")
# Copy each finished cache/<video_id>/ tree to object storage when a job ends.
MIRROR_ARTIFACTS = os.getenv("VS_MIRROR_ARTIFACTS", "0").strip().lower() in {"1", "true", "yes"}

# --- Auth ---------------------------------------------------------------
# Off by default so local dev and the bundled frontend work with no setup.
# Set VS_API_KEYS (comma-separated bearer tokens) and/or VS_JWT_SECRET to
# turn it on; then every /api route except health + keyframe images requires
# `Authorization: Bearer <token>`.
API_KEYS = {k.strip() for k in os.getenv("VS_API_KEYS", "").split(",") if k.strip()}
JWT_SECRET = os.getenv("VS_JWT_SECRET", "").strip() or None
JWT_ALGORITHM = os.getenv("VS_JWT_ALGORITHM", "HS256").strip()
JWT_AUDIENCE = os.getenv("VS_JWT_AUDIENCE", "").strip() or None
AUTH_ENABLED = bool(API_KEYS or JWT_SECRET)

# --- Queue -----------------------------------------------------------------
# "inprocess": a local worker pool inside the API process — zero infra, good
#   for a single box and for demonstrating the scheduler.
# "redis": hand jobs to RQ so workers run as separate processes / machines,
#   which is what the distributed scheduler claim actually needs. Requires a
#   running Redis and `pip install rq redis`.
QUEUE_BACKEND = os.getenv("VS_QUEUE_BACKEND", "inprocess").strip().lower()
REDIS_URL = os.getenv("VS_REDIS_URL", "redis://localhost:6379/0")

# How many jobs one worker process runs at once. One is the safe default: a
# single job already saturates the CPU and holds several GB of models resident.
WORKER_CONCURRENCY = int(os.getenv("VS_WORKER_CONCURRENCY", "1"))

# The wall-clock target the scheduler plans against, in seconds. Mirrors
# core.config.TARGET_PIPELINE_SEC; the "fast" and "accurate" request options
# scale it per job.
DEFAULT_BUDGET_SEC = float(os.getenv("VS_TARGET_SEC", "300"))

# --- Worker identity ------------------------------------------------------
# Each worker advertises a hardware class so the scheduler can route by it and
# keep a separate learned cost model per class. Set WORKER_CLASS to something
# like "gpu-a10" or "cpu-16core" per machine in a real fleet.
WORKER_CLASS = os.getenv("VS_WORKER_CLASS", "").strip() or None

# Emulated relative compute speed for this worker; 1.0 = the real hardware.
# Below 1.0 the worker is deliberately slowed — its wall-clock is padded and
# its learned cost scaled to match — so two workers on one machine produce
# distinct per-class cost models and SLA outcomes without needing two boxes.
# Leave at 1.0 when real throttling is in place (e.g. docker-compose `cpus:`).
WORKER_SPEED = float(os.getenv("VS_WORKER_SPEED", "1.0"))

# Ceiling on the emulated slow-down sleep, so a near-instant cached run or a
# very long real run can't stall a worker indefinitely.
MAX_EMULATED_DELAY_SEC = float(os.getenv("VS_MAX_EMULATED_DELAY_SEC", "900"))

# Queue-name namespace. Per-class queues are "<namespace>.<worker_class>";
# "<namespace>.any" is the fallback every worker also drains for unrouted jobs.
QUEUE_NAMESPACE = os.getenv("VS_QUEUE_NAMESPACE", "video").strip() or "video"


def _parse_local_fleet() -> list[dict]:
    """``VS_LOCAL_FLEET``: JSON list of in-process pseudo-workers, e.g.
    ``[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.4}]``.

    Each entry becomes a worker thread bound to that class's queue, so the
    scheduler routes across classes for real on a single box with zero infra.
    Empty (the default) → one worker at ``WORKER_CLASS`` / ``WORKER_SPEED``
    draining the shared queue.
    """
    raw = os.getenv("VS_LOCAL_FLEET", "").strip()
    if not raw:
        return []
    try:
        members: list[dict] = []
        for entry in json.loads(raw):
            members.append({
                "class": str(entry["class"]),
                "speed": float(entry.get("speed", 1.0)),
                "concurrency": max(int(entry.get("concurrency", 1)), 1),
            })
        return members
    except Exception:
        return []


LOCAL_FLEET = _parse_local_fleet()


def worker_queue_names(worker_class: str | None) -> list[str]:
    """Queues an RQ worker process drains, highest priority first: its own
    class queue, then the shared fallback."""
    names: list[str] = []
    wc = (worker_class or WORKER_CLASS or "").strip()
    if wc:
        names.append(f"{QUEUE_NAMESPACE}.{wc}")
    names.append(f"{QUEUE_NAMESPACE}.any")
    return names

# --- API -----------------------------------------------------------------
CORS_ORIGINS = [
    o.strip() for o in os.getenv("VS_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]
API_TITLE = "AI Video Summarizer API"
API_VERSION = "1.0.0"


def ensure_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --- Accounts and tiers ----------------------------------------------------
# Sign-in routes. Each is optional: with none configured the API stays open,
# which is how it has always run locally.
GOOGLE_CLIENT_ID = os.getenv("VS_GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.getenv("VS_GOOGLE_CLIENT_SECRET", "").strip()
GOOGLE_REDIRECT_URI = os.getenv(
    "VS_GOOGLE_REDIRECT_URI", "http://localhost:3000/auth/google"
).strip()

# Email one-time codes. Without SMTP credentials the code is written to the
# log instead of sent, which is what makes the whole flow demonstrable with
# nothing configured — and is refused outright when DEV_OTP_ECHO is off.
SMTP_HOST = os.getenv("VS_SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("VS_SMTP_PORT", "587"))
SMTP_USER = os.getenv("VS_SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("VS_SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("VS_SMTP_FROM", "").strip() or SMTP_USER

# Phone one-time codes. MSG91 is the cheap Indian option; Twilio works too.
# Both cost money per message, so this stays off unless deliberately set.
SMS_PROVIDER = os.getenv("VS_SMS_PROVIDER", "").strip().lower()   # msg91 | twilio
SMS_API_KEY = os.getenv("VS_SMS_API_KEY", "").strip()
SMS_SENDER = os.getenv("VS_SMS_SENDER", "").strip()
SMS_TWILIO_SID = os.getenv("VS_TWILIO_SID", "").strip()
SMS_TWILIO_FROM = os.getenv("VS_TWILIO_FROM", "").strip()

# With no sender configured, write the code to the log so the flow can be
# demonstrated. Never leave this on anywhere the log is not private.
DEV_OTP_ECHO = os.getenv("VS_DEV_OTP_ECHO", "1") not in ("0", "false", "False")

OTP_LENGTH = int(os.getenv("VS_OTP_LENGTH", "6"))
OTP_TTL_SEC = int(os.getenv("VS_OTP_TTL_SEC", "600"))
OTP_MAX_ATTEMPTS = int(os.getenv("VS_OTP_MAX_ATTEMPTS", "5"))
# One code per destination per this many seconds, so the endpoint cannot be
# used to send somebody a hundred messages.
OTP_RESEND_SEC = int(os.getenv("VS_OTP_RESEND_SEC", "45"))

SESSION_TTL_DAYS = int(os.getenv("VS_SESSION_TTL_DAYS", "30"))

# --- What each tier may do -------------------------------------------------
# Counted in model calls rather than videos: a 4-hour video is about 22 calls
# and a half-hour one about 4, so a per-video limit would charge them alike.
FREE_MAX_VIDEO_HOURS = float(os.getenv("VS_FREE_MAX_VIDEO_HOURS", "4"))
FREE_DAILY_CALLS = int(os.getenv("VS_FREE_DAILY_CALLS", "40"))
PREMIUM_MAX_VIDEO_HOURS = float(os.getenv("VS_PREMIUM_MAX_VIDEO_HOURS", "15"))
PREMIUM_DAILY_CALLS = int(os.getenv("VS_PREMIUM_DAILY_CALLS", "400"))

# --- Payments --------------------------------------------------------------
# Test keys by default. Live keys need a registered business, GST and a
# refund policy, none of which a project needs to demonstrate the flow.
RAZORPAY_KEY_ID = os.getenv("VS_RAZORPAY_KEY_ID", "").strip()
RAZORPAY_KEY_SECRET = os.getenv("VS_RAZORPAY_KEY_SECRET", "").strip()
RAZORPAY_MODE = os.getenv("VS_RAZORPAY_MODE", "test").strip().lower()
PREMIUM_PRICE_PAISE = int(os.getenv("VS_PREMIUM_PRICE_PAISE", "19900"))   # Rs 199
PREMIUM_DAYS = int(os.getenv("VS_PREMIUM_DAYS", "30"))
