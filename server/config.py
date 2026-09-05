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

# Anchored to the package, like core.config.CACHE_ROOT, so the service behaves
# the same whatever directory it is launched from.
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

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
