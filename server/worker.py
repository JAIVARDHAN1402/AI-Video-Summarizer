"""RQ worker entrypoint for the distributed (``VS_QUEUE_BACKEND=redis``) queue.

    VS_WORKER_CLASS=cpu-fast  python -m server.worker
    VS_WORKER_CLASS=gpu-a10   python -m server.worker --burst

Each worker advertises a hardware class (``VS_WORKER_CLASS``) and drains that
class's queue plus the shared ``<namespace>.any`` fallback, so a job the
scheduler routed to ``cpu-fast`` runs on a ``cpu-fast`` worker. Windows and
frozen environments can't ``fork``; there ``SimpleWorker`` (run the job in the
worker process) is selected automatically.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("server.worker")


def main() -> int:
    ap = argparse.ArgumentParser(description="Video-pipeline RQ worker")
    ap.add_argument("--burst", action="store_true",
                    help="process what's queued, then exit (for CI / one-shot runs)")
    ap.add_argument("--queues", default="",
                    help="comma-separated queue names, overriding the class default")
    args = ap.parse_args()

    from server import config

    try:
        from redis import Redis  # type: ignore
        from rq import Queue, SimpleWorker, Worker  # type: ignore
    except ImportError:
        log.error("rq / redis not installed — run `pip install -r requirements-server.txt`")
        return 2

    worker_class = config.WORKER_CLASS
    if not worker_class:
        log.error("set VS_WORKER_CLASS (e.g. cpu-fast, gpu-a10) so the scheduler "
                  "can route jobs to this worker")
        return 2

    queue_names = (
        [q.strip() for q in args.queues.split(",") if q.strip()]
        or config.worker_queue_names(worker_class)
    )
    conn = Redis.from_url(config.REDIS_URL)
    queues = [Queue(n, connection=conn) for n in queue_names]

    # No os.fork on Windows; SimpleWorker executes the job in this process.
    forkless = sys.platform == "win32" or os.getenv("VS_RQ_SIMPLE", "").strip().lower() in {"1", "true", "yes"}
    worker_cls = SimpleWorker if forkless else Worker

    from server.db import init_db
    init_db()

    # Warm the models once so the first job isn't cold (~70s of loading).
    try:
        from core import warm

        warm.start()
    except Exception:
        log.warning("model warm-up could not start", exc_info=True)

    log.info(
        "worker class=%s speed=%.2f queues=%s redis=%s (%s)",
        worker_class, config.WORKER_SPEED, queue_names, config.REDIS_URL,
        worker_cls.__name__,
    )
    worker = worker_cls(queues, connection=conn)
    worker.work(burst=args.burst, with_scheduler=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
