"""Distributed, SLA-constrained scheduler for the video-understanding pipeline.

The single-machine planner in ``core.perf`` chooses model sizes to fit a
wall-clock budget on the box that happens to be planning. This module lifts
that to a fleet: given a job and its SLA, it predicts the cost on **every
registered worker class** using a cost model calibrated separately per class,
routes the job to the cheapest class that still meets the SLA, and — once the
job finishes — folds its real wall-clock back into that class's model.

That feedback loop is the point. A brand-new GPU node starts with a rough seed
estimate; after a handful of jobs its ``total:per_min`` is measured, and
routing decisions for that class become accurate. Every routed job also writes
an SLA observation (predicted vs actual vs a fixed-configuration baseline),
which is the evidence that the adaptive routing holds the budget where a
static assignment would overrun.

Registration
    Worker classes come from ``VS_WORKER_CLASSES`` (JSON list of
    ``{"name": ..., "has_gpu": bool, "seed_per_min": float}``) and are also
    auto-added the first time a job completes on a class not seen before.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from server import config
from server.db import Job, SchedulerCost, SlaObservation, session

logger = logging.getLogger("server.scheduler")

# Asymmetric blend, same rationale as core.perf: believe "slower than expected"
# quickly, "faster" slowly, so the budget planner stays honest.
_ALPHA_SLOWER = 0.6
_ALPHA_FASTER = 0.2

# The coarse routing metric: wall-clock seconds per minute of video, end to end.
_ROUTING_KEY = "total:per_min"

# Rough seed for a class with no measurements yet, relative to a CPU-only box.
_GPU_SEED_FACTOR = 0.18


@dataclass
class WorkerClass:
    name: str
    has_gpu: bool = False
    seed_per_min: Optional[float] = None  # sec of processing per video-minute

    def default_seed(self, cpu_reference_per_min: float) -> float:
        if self.seed_per_min is not None:
            return self.seed_per_min
        return cpu_reference_per_min * (_GPU_SEED_FACTOR if self.has_gpu else 1.0)


@dataclass
class ClassPrediction:
    worker_class: str
    predicted_sec: float
    meets_sla: bool
    from_measurement: bool           # True once the class has real data
    plan: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    worker_class: str
    predicted_sec: float
    sla_target_sec: float
    meets_sla: bool
    plan: dict[str, Any]
    alternatives: list[ClassPrediction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_class": self.worker_class,
            "predicted_sec": round(self.predicted_sec, 1),
            "sla_target_sec": round(self.sla_target_sec, 1),
            "meets_sla": self.meets_sla,
            "plan": self.plan,
            "alternatives": [
                {
                    "worker_class": a.worker_class,
                    "predicted_sec": round(a.predicted_sec, 1),
                    "meets_sla": a.meets_sla,
                    "from_measurement": a.from_measurement,
                }
                for a in self.alternatives
            ],
        }


# --- registry ----------------------------------------------------------------


def _registry() -> dict[str, WorkerClass]:
    classes: dict[str, WorkerClass] = {}
    raw = os.getenv("VS_WORKER_CLASSES", "").strip()
    if raw:
        try:
            for entry in json.loads(raw):
                wc = WorkerClass(
                    name=str(entry["name"]),
                    has_gpu=bool(entry.get("has_gpu", False)),
                    seed_per_min=entry.get("seed_per_min"),
                )
                classes[wc.name] = wc
        except Exception:
            logger.warning("VS_WORKER_CLASSES is not valid JSON; ignoring", exc_info=True)

    # Always include the class this process would run as, so a single-box
    # deployment schedules against itself with zero configuration.
    from core import perf

    p = perf.get_profile()
    local = config.WORKER_CLASS or f"{'gpu' if p.has_cuda else 'cpu'}-{p.cpu_count}core"
    classes.setdefault(local, WorkerClass(name=local, has_gpu=p.has_cuda))

    # Any class that has completed a job but was never declared.
    with session() as db:
        for (wc_name,) in db.query(SchedulerCost.worker_class).distinct():
            classes.setdefault(wc_name, WorkerClass(name=wc_name, has_gpu="gpu" in wc_name))
    return classes


def _cpu_reference_per_min() -> float:
    """A baseline sec/video-min for seeding classes with no data.

    Uses whatever the local planner predicts for a mid-length video, which is
    itself calibrated — good enough to seed a class until it has its own runs.
    """
    from core import perf

    plan = perf.plan_for(20 * 60)
    return max(plan.predicted_total_sec / 20.0, 1.0)


# --- cost model ------------------------------------------------------------


def _class_cost(worker_class: str, key: str = _ROUTING_KEY) -> Optional[tuple[float, int]]:
    with session() as db:
        row = db.get(SchedulerCost, (worker_class, key))
        return (row.seconds, row.samples) if row else None


def _blend(previous: float, observed: float) -> float:
    alpha = _ALPHA_SLOWER if observed > previous else _ALPHA_FASTER
    return (1 - alpha) * previous + alpha * observed


# --- routing --------------------------------------------------------------


def route(
    duration_sec: float, video_type: str, sla_target_sec: float,
) -> RoutingDecision:
    """Pick the worker class for one job.

    For each registered class, predict the end-to-end wall-clock: use the
    class's measured ``total:per_min`` if it has one, else a seed. Return the
    cheapest class that stays within ``sla_target_sec``; if none does, return
    the fastest and flag the miss.
    """
    duration_min = max(duration_sec / 60.0, 0.1)
    classes = _registry()
    cpu_ref = _cpu_reference_per_min()

    predictions: list[ClassPrediction] = []
    for wc in classes.values():
        measured = _class_cost(wc.name)
        if measured is not None:
            per_min, _samples = measured
            from_measurement = True
        else:
            per_min = wc.default_seed(cpu_ref)
            from_measurement = False
        predicted = per_min * duration_min
        predictions.append(ClassPrediction(
            worker_class=wc.name,
            predicted_sec=predicted,
            meets_sla=predicted <= sla_target_sec,
            from_measurement=from_measurement,
        ))

    # Prefer a class that meets the SLA; among those, the cheapest (fastest);
    # if none meets it, take the fastest overall.
    within = [p for p in predictions if p.meets_sla]
    pool = within or predictions
    chosen = min(pool, key=lambda p: p.predicted_sec)

    # Attach the concrete execution plan the worker should use, computed with
    # that class's cost model so the model choices match its real speed.
    from core import perf

    override = _perf_override_for(chosen.worker_class)
    plan = perf.plan_for(
        duration_sec, budget_sec=sla_target_sec, cost_override=override or None,
    )
    chosen.plan = perf.plan_to_dict(plan)

    logger.info(
        "routed %d-min %s job -> %s (predicted %.0fs, SLA %.0fs, %s)",
        duration_min, video_type, chosen.worker_class,
        chosen.predicted_sec, sla_target_sec,
        "OK" if chosen.meets_sla else "OVER",
    )
    return RoutingDecision(
        worker_class=chosen.worker_class,
        predicted_sec=chosen.predicted_sec,
        sla_target_sec=sla_target_sec,
        meets_sla=chosen.meets_sla,
        plan=chosen.plan,
        alternatives=sorted(predictions, key=lambda p: p.predicted_sec),
    )


def _perf_override_for(worker_class: str) -> dict[str, float]:
    """Translate a class's stored fine-grained costs into a core.perf override.

    Only keys the class has actually measured are returned; core.perf falls
    back to its own calibration / references for the rest.
    """
    out: dict[str, float] = {}
    with session() as db:
        rows = db.query(SchedulerCost).filter(
            SchedulerCost.worker_class == worker_class,
            SchedulerCost.key.notlike("total:%"),
        ).all()
        for r in rows:
            out[r.key] = r.seconds
    return out


# --- feedback -----------------------------------------------------------------


def record_outcome(job: Job) -> None:
    """Fold a finished job's real timings into its class's cost model, and
    write the SLA observation row."""
    timings = job.timings or {}
    total = timings.get("total")
    duration = job.duration_sec or 0.0
    worker_class = job.worker_class or job.routed_worker_class or "unknown"
    if not total or duration <= 0:
        return

    duration_min = duration / 60.0
    observed_per_min = total / duration_min

    with session() as db:
        # coarse routing metric
        _upsert_cost(db, worker_class, _ROUTING_KEY, observed_per_min)
        # fine-grained per-stage (per video-minute) for the plan override
        for stage in ("analyze", "summarize", "index"):
            if stage in timings and timings[stage] > 0:
                _upsert_cost(db, worker_class, f"stage:{stage}:per_min", timings[stage] / duration_min)

        predicted = job.predicted_sec or observed_per_min * duration_min
        target = job.sla_target_sec or config.DEFAULT_BUDGET_SEC
        baseline = _baseline_estimate(duration)
        db.add(SlaObservation(
            job_id=job.id,
            worker_class=worker_class,
            duration_sec=duration,
            predicted_sec=predicted,
            actual_sec=total,
            sla_target_sec=target,
            met=total <= target,
            baseline_sec=baseline,
            baseline_met=(baseline <= target) if baseline else None,
        ))
        db.commit()

    logger.info(
        "job %s outcome: %.0fs actual vs %.0fs predicted vs %.0fs SLA (%s)",
        job.id, total, predicted, target, "met" if total <= target else "MISSED",
    )


def _upsert_cost(db, worker_class: str, key: str, observed: float) -> None:
    row = db.get(SchedulerCost, (worker_class, key))
    if row is None:
        db.add(SchedulerCost(worker_class=worker_class, key=key, seconds=observed, samples=1))
    else:
        row.seconds = round(_blend(row.seconds, observed), 5)
        row.samples += 1
        row.updated_at = dt.datetime.now(dt.timezone.utc)


def _baseline_estimate(duration_sec: float) -> Optional[float]:
    """What a fixed 'always the accurate models' configuration would cost.

    The counterfactual the SLA stats compare against — a static assignment
    that never downshifts. Computed with the local planner's quality_floor.
    """
    try:
        from core import perf

        return perf.plan_for(duration_sec, quality_floor=True).predicted_total_sec
    except Exception:
        return None


# --- reporting -----------------------------------------------------------------


def sla_stats() -> dict[str, Any]:
    """Aggregate the SLA observations — the adaptive-vs-static evidence."""
    with session() as db:
        rows = db.query(SlaObservation).all()
        classes = {c.name: c for c in [WorkerClass(**{"name": n}) for (n,) in
                                       db.query(SchedulerCost.worker_class).distinct()]}
        cost_rows = db.query(SchedulerCost).all()

    if not rows:
        return {"observations": 0, "cost_model": _cost_model_view(cost_rows), "classes": list(classes)}

    def rate(pred):
        hit = [r for r in rows if pred(r)]
        return round(len(hit) / len(rows), 3)

    overruns = [r.actual_sec / r.sla_target_sec for r in rows if r.sla_target_sec > 0]
    baseline_overruns = [
        r.baseline_sec / r.sla_target_sec
        for r in rows if r.baseline_sec and r.sla_target_sec > 0
    ]
    per_class: dict[str, Any] = {}
    for r in rows:
        b = per_class.setdefault(r.worker_class, {"n": 0, "met": 0})
        b["n"] += 1
        b["met"] += int(r.met)
    for name, b in per_class.items():
        b["met_rate"] = round(b["met"] / b["n"], 3)

    return {
        "observations": len(rows),
        "sla_met_rate": rate(lambda r: r.met),
        "mean_time_vs_sla": round(sum(overruns) / len(overruns), 3) if overruns else None,
        "prediction_error_ratio": round(
            sum(abs(r.actual_sec - r.predicted_sec) / max(r.predicted_sec, 1) for r in rows) / len(rows),
            3,
        ),
        "baseline_sla_met_rate": (
            round(sum(1 for r in rows if r.baseline_met) / len(rows), 3)
            if any(r.baseline_met is not None for r in rows) else None
        ),
        "baseline_mean_time_vs_sla": (
            round(sum(baseline_overruns) / len(baseline_overruns), 3) if baseline_overruns else None
        ),
        "by_worker_class": per_class,
        "cost_model": _cost_model_view(cost_rows),
    }


def _cost_model_view(cost_rows) -> list[dict[str, Any]]:
    return [
        {
            "worker_class": r.worker_class,
            "key": r.key,
            "seconds": round(r.seconds, 4),
            "samples": r.samples,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in sorted(cost_rows, key=lambda r: (r.worker_class, r.key))
    ]


def classes_view() -> list[dict[str, Any]]:
    reg = _registry()
    cpu_ref = _cpu_reference_per_min()
    out = []
    for wc in reg.values():
        measured = _class_cost(wc.name)
        out.append({
            "name": wc.name,
            "has_gpu": wc.has_gpu,
            "total_per_min_sec": round(measured[0], 3) if measured else round(wc.default_seed(cpu_ref), 3),
            "samples": measured[1] if measured else 0,
            "source": "measured" if measured else "seed",
        })
    return sorted(out, key=lambda c: c["total_per_min_sec"])
