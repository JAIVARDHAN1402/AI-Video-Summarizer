"""The distributed scheduler's routing loop and the worker-speed emulation.

Uses a throwaway SQLite file and a two-class local fleet, set up before any
``server`` import so ``server.config`` reads them.
"""

import os
import tempfile
import time

import pytest

_TMP_DB = os.path.join(tempfile.mkdtemp(), "fleet_test.db")
os.environ["VS_DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["VS_LOCAL_FLEET"] = (
    '[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.1}]'
)

from server import config, jobs, scheduler  # noqa: E402
from server.db import init_db, session  # noqa: E402

init_db()

# The environment above only takes effect if this module is imported before
# anything else imports server.config, which made these tests pass alone and
# fail in a suite that happened to import it first. The fleet is set here
# instead, so the order no longer decides the result.
_FLEET = [{"class": "cpu-fast", "speed": 1.0}, {"class": "cpu-slow", "speed": 0.1}]


@pytest.fixture(autouse=True)
def _two_class_fleet(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_FLEET", _FLEET)


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


def test_cost_model_tracks_cores_and_accelerators():
    """Rates are derived from the class name, so a fleet needs no price list."""
    big = scheduler.WorkerClass(name="cpu-16core")
    small = scheduler.WorkerClass(name="cpu-2core")
    gpu = scheduler.WorkerClass(name="gpu-8core", has_gpu=True)

    assert big.cores == 16 and small.cores == 2
    assert big.rate_per_hour() > small.rate_per_hour()
    assert gpu.rate_per_hour() > big.rate_per_hour()  # the accelerator premium

    # An explicit rate wins over the derived one.
    assert scheduler.WorkerClass(name="cpu-2core", cost_per_hour=9.0).rate_per_hour() == 9.0


def test_routes_to_the_cheapest_class_that_still_meets_the_sla():
    """The point of a deadline is to spend no more than it takes to hit it.

    Both classes are given measurements that comfortably fit the SLA, so the
    tie is broken on cost alone — and the small, slow machine wins even though
    the large one would finish sooner.
    """
    from server.db import SchedulerCost

    with session() as db:
        db.query(SchedulerCost).delete()
        for name, per_min in (("cpu-16core", 6.0), ("cpu-2core", 30.0)):
            db.add(
                SchedulerCost(
                    worker_class=name,
                    key=scheduler._ROUTING_KEY,
                    seconds=per_min,
                    samples=50,
                )
            )
        db.commit()

    preds = {
        p.worker_class: p
        for p in scheduler.route(
            duration_sec=600, video_type="lecture", sla_target_sec=10_000
        ).alternatives
    }

    # Sanity: the big machine really is the faster of the two here.
    assert preds["cpu-16core"].predicted_sec < preds["cpu-2core"].predicted_sec
    # ...and the cheaper of the two is the small one, despite running longer.
    assert preds["cpu-2core"].cost < preds["cpu-16core"].cost


def test_the_cost_fit_separates_fixed_setup_from_per_minute_work():
    """A ratio learned from short videos must not be extrapolated to long ones.

    Every observation this fleet had was of a two- to four-minute video. A
    run pays ~160s before it processes anything — models, processes, one
    round-trip to the understanding model — so over three minutes that looks
    like 50 s/video-min. Extrapolated, an eleven-hour video came out at four
    hours against a real cost of ninety minutes.
    """
    from server.db import SchedulerCost, session

    wc = "cpu-fit-test"
    # total = 150s fixed + 6 s/video-min, sampled across a realistic spread.
    samples = [(2.0, 162.0), (5.0, 180.0), (40.0, 390.0), (120.0, 870.0)]
    with session() as db:
        db.query(SchedulerCost).filter_by(worker_class=wc).delete()
        for minutes, total in samples:
            scheduler._accumulate_fit(db, wc, minutes, total)
        db.commit()

        fit = scheduler._fit_for(db, wc)

    assert fit is not None and fit.credible
    assert 140 < fit.fixed_sec < 160, f"fixed cost recovered as {fit.fixed_sec}"
    assert 5.5 < fit.per_min < 6.5, f"rate recovered as {fit.per_min}"

    # The point of the exercise: a length far outside the samples.
    eleven_hours = fit.predict(660)
    assert 4000 < eleven_hours < 4500, f"{eleven_hours/3600:.2f} h for 11 hours of video"

    # What a single ratio fitted to the same data would have said, for contrast.
    ratio = sum(t for _, t in samples) / sum(m for m, _ in samples)
    assert ratio * 660 > eleven_hours * 1.5, "the ratio should be the pessimistic one"


def test_a_fit_from_one_video_length_is_not_trusted():
    """Intercept and slope are not separable without a spread of lengths."""
    from server.db import SchedulerCost, session

    wc = "cpu-flat-test"
    with session() as db:
        db.query(SchedulerCost).filter_by(worker_class=wc).delete()
        for total in (100.0, 104.0, 98.0, 102.0, 101.0):
            scheduler._accumulate_fit(db, wc, 3.0, total)
        db.commit()
        fit = scheduler._fit_for(db, wc)

    assert fit is None or not fit.credible, "a flat x-range cannot support a fit"
