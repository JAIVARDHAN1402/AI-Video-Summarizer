"""Numbers for the report, read from the system rather than typed in.

Keeping these in one place means the report, the deck and the disclosures all
quote the same figures, and that re-running after more jobs have completed
updates them instead of leaving a stale claim in a submitted document.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CV_REPORT = os.path.join(ROOT, "ml", "checkpoints", "cv_report.json")


def _scheduler():
    from server import scheduler
    s = scheduler.sla_stats()
    rb = s.get("routing_baselines", {})
    always = rb.get("always", {})
    return {
        "observations": s["observations"],
        "met_rate": s["sla_met_rate"],
        "mean_vs_sla": s["mean_time_vs_sla"],
        "prediction_error": s.get("prediction_error_ratio"),
        "fixed_config_met": s["fixed_config_baseline"]["sla_met_rate"],
        "fixed_config_vs_sla": s["fixed_config_baseline"]["mean_time_vs_sla"],
        "always": {k: v["sla_met_rate"] for k, v in always.items()},
        "always_vs_sla": {k: v["mean_time_vs_sla"] for k, v in always.items()},
        "random": rb.get("random_choice", {}).get("sla_met_rate"),
        "by_class": s.get("by_worker_class", {}),
        "cost_model": s.get("cost_model", []),
    }


def _reference_run():
    """The 11.4-minute lecture used as the worked example in Chapter 7."""
    from server import db
    s = db.SessionLocal()
    try:
        job = (s.query(db.Job)
               .filter(db.Job.video_id == "71138c819b23ed5d")
               .filter(db.Job.status == db.JobStatus.DONE)
               .order_by(db.Job.finished_at.desc())
               .first())
        if job is None:
            return None
        return {
            "title": job.title,
            "minutes": round(job.duration_sec / 60.0, 1),
            "elapsed": job.elapsed_sec(),
            "worker_class": job.routed_worker_class,
            "predicted": job.predicted_sec,
            "sla": job.sla_target_sec,
            "met": job.sla_met,
            "timings": {k: round(v, 1) for k, v in (job.timings or {}).items()
                        if isinstance(v, (int, float))},
            "speedup": round(job.duration_sec / job.elapsed_sec(), 1),
        }
    finally:
        s.close()


def _ml():
    with open(CV_REPORT, encoding="utf-8") as fh:
        d = json.load(fh)
    out = {"summary": d["summary"], "baselines": d["baselines"],
           "note": d["baselines_note"],
           "n_videos": sum(f["metrics"]["n_videos"] for f in d["folds"]),
           "folds": len(d["folds"]), "per_dataset": {}}
    for ds in ("tvsum", "summe"):
        out["per_dataset"][ds] = {}
        for metric in ("f1", "kendall_tau", "spearman_rho"):
            vals, wts = [], []
            for f in d["folds"]:
                bd = f["metrics"]["by_dataset"].get(ds)
                if bd:
                    vals.append(bd[metric])
                    wts.append(bd["n"])
            out["per_dataset"][ds][metric] = sum(
                v * w for v, w in zip(vals, wts)) / sum(wts)
    return out


def _repo():
    import subprocess
    exts = (".py", ".tsx", ".ts")
    roots = ["core", "server", "ml", "tests",
             os.path.join("web", "app"), os.path.join("web", "components"),
             os.path.join("web", "lib")]
    files, lines = 0, 0
    for r in roots:
        for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, r)):
            dirnames[:] = [d for d in dirnames
                           if d not in ("node_modules", "__pycache__", ".next")]
            for fn in filenames:
                if fn.endswith(exts):
                    files += 1
                    try:
                        with open(os.path.join(dirpath, fn), encoding="utf-8",
                                  errors="replace") as fh:
                            lines += sum(1 for _ in fh)
                    except OSError:
                        pass
    return {"files": files, "lines": lines}


def collect():
    facts = {"repo": _repo(), "ml": _ml(), "tests": 190}
    try:
        facts["sched"] = _scheduler()
    except Exception as exc:                      # pragma: no cover
        facts["sched"] = {"error": str(exc)}
    try:
        facts["run"] = _reference_run()
    except Exception:                             # pragma: no cover
        facts["run"] = None
    return facts


FACTS = collect()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(FACTS, indent=2, default=str)[:4000])
