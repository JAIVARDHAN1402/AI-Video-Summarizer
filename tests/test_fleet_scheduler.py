"""The distributed scheduler's routing loop and the worker-speed emulation.

Uses a throwaway SQLite file and a two-class local fleet, set up before any
``server`` import so ``server.config`` reads them.
"""

import os
import tempfile
import time

_TMP_DB = os.path.join(tempfile.mkdtemp(), "fleet_test.db")
os.environ["VS_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["VS_LOCAL_FLEET"] = (
    '[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.1}]'
)

from server import jobs, scheduler  # noqa: E402
from server.db import init_db, session  # noqa: E402

init_db()


def test_routes_explore_then_avoid_the_slow_class():
    # First job: no class has data — the scheduler probes one.
    d1 = scheduler.route(duration_sec=600, video_type="coding", sla_target_sec=8)
    assert d1.worker_class in {"cpu-fast", "cpu-slow"}

    # That job completes and feeds its class's model; the still-unmeasured
    # class is the one the next route probes.
    with session() as db:
        scheduler._upsert_cost(db, d1.worker_class, "total:per_min", 0.05)
        db.commit()
    d2 = scheduler.route(duration_sec=600, video_type="coding", sla_target_sec=8)
    assert {d1.worker_class, d2.worker_class} == {"cpu-fast", "cpu-slow"}

    # Feed back real costs: cpu-slow ~40x cpu-fast, over the SLA at this length.
    with session() as db:
        scheduler._upsert_cost(db, "cpu-fast", "total:per_min", 0.05)
        scheduler._upsert_cost(db, "cpu-slow", "total:per_min", 2.0)
        db.commit()

    # Now both are measured; routing must pick the class that holds the SLA.
    d3 = scheduler.route(duration_sec=600, video_type="coding", sla_target_sec=8)
    assert d3.worker_class == "cpu-fast"
    assert d3.meets_sla


def test_emulated_speed_scales_timings_and_waits():
    # speed >= 1 is a no-op.
    assert jobs._apply_emulated_speed({"total": 10.0}, 5.0, 1.0) == {"total": 10.0}

    # speed 0.5 doubles the recorded timings...
    scaled = jobs._apply_emulated_speed({"total": 10.0, "analyze": 6.0}, 0.0, 0.5)
    assert scaled == {"total": 20.0, "analyze": 12.0}

    # ...and pads the wall-clock by the extra a slower box would have taken.
    start = time.monotonic()
    jobs._apply_emulated_speed({"total": 1.0}, proc_wall_sec=0.2, speed=0.5)
    assert time.monotonic() - start >= 0.15
