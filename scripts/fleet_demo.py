"""Exercise the distributed SLA scheduler across a running worker fleet and
print the adaptive-vs-static evidence.

    # start a fleet first, e.g.
    #   docker compose up --build
    # or, zero-infra, a two-class in-process fleet:
    #   VS_LOCAL_FLEET='[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.25}]' \
    #     VS_API_PORT=8000 python -m server.main

    python scripts/fleet_demo.py --api http://localhost:8000 \
        --url https://youtu.be/REOH22Xwdkk --repeat 8 --profile fast

For each job the scheduler picks a worker class against the SLA, the job runs,
and its real wall-clock is folded back into that class's cost model. After a
few jobs the routing has learned which class is fast enough; the summary shows
how often it held the SLA versus a fixed "always the accurate models"
baseline.

Note: when the fleet uses VS_WORKER_SPEED / VS_LOCAL_FLEET speeds below 1.0,
the slower class's wall-clock is emulated (padded + scaled). The scheduler
code path — routing, per-class calibration, SLA feedback — is unchanged and
fully exercised; only the hardware diversity is simulated. A docker-compose
fleet with `cpus:` limits needs no emulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_URLS = ["https://youtu.be/REOH22Xwdkk"]


def _post(api: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        api.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _get(api: str, path: str) -> dict:
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


def _wait(api: str, job_id: str, poll: float, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _get(api, f"/api/jobs/{job_id}")
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(poll)
    return {"status": "timeout", "id": job_id}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.getenv("VS_API", "http://localhost:8000"))
    ap.add_argument("--url", action="append", default=[], help="video URL (repeatable)")
    ap.add_argument("--video-type", default="coding")
    ap.add_argument("--topic", default="")
    ap.add_argument("--profile", default="fast", choices=["fast", "balanced", "accurate"])
    ap.add_argument("--sla", type=float, default=None,
                    help="wall-clock target in seconds, overriding the profile budget")
    ap.add_argument("--calibrate", type=int, default=0,
                    help="jobs to run first at a loose SLA so every class gets measured "
                         "before the real routing phase")
    ap.add_argument("--repeat", type=int, default=6, help="jobs to submit (cycling the URLs)")
    ap.add_argument("--serial", action="store_true", help="wait for each job before submitting the next")
    ap.add_argument("--poll", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    urls = args.url or DEFAULT_URLS

    try:
        health = _get(args.api, "/api/health")
    except urllib.error.URLError as e:
        print(f"cannot reach API at {args.api}: {e}", file=sys.stderr)
        return 2
    print(f"API {args.api}  queue={health.get('queue')}  fleet={json.dumps(health.get('fleet', {}))}\n")

    loose_sla = (args.sla or 300.0) * 20.0
    if args.calibrate:
        print(f"calibration phase — {args.calibrate} jobs at a loose {loose_sla:.0f}s SLA "
              f"so every class gets measured:")
        _run_phase(args, urls, args.calibrate, loose_sla)
        print()

    print(f"routing phase — {args.repeat} jobs at the target "
          f"{args.sla or 'profile-default'} SLA:")
    _run_phase(args, urls, args.repeat, args.sla)

    print("\n" + "=" * 68)
    _print_stats(_get(args.api, "/api/scheduler/sla"))
    print("\nper-class cost model now:")
    for c in _get(args.api, "/api/scheduler/classes"):
        print(f"  {c['name']:14}  {c['total_per_min_sec']:.2f} s/video-min  "
              f"({c['source']}, n={c['samples']})")
    return 0


def _run_phase(args, urls: list[str], count: int, sla: float | None) -> None:
    submitted: list[str] = []
    for i in range(count):
        body = {
            "url": urls[i % len(urls)], "video_type": args.video_type,
            "topic": args.topic, "profile": args.profile, "force": False,
        }
        if sla:
            body["sla_sec"] = sla
        ref = _post(args.api, "/api/jobs", body)
        job = _get(args.api, f"/api/jobs/{ref['id']}")
        print(f"  job {i + 1:2d}/{count}  {ref['id'][:8]}  "
              f"routed -> {str(job.get('routed_worker_class') or '?'):<10}  "
              f"predicted {(job.get('predicted_sec') or 0):6.0f}s  "
              f"SLA {(job.get('sla_target_sec') or 0):.0f}s")
        submitted.append(ref["id"])
        if args.serial:
            _print_job_result(_wait(args.api, ref["id"], args.poll, args.timeout))
    if not args.serial:
        for jid in submitted:
            _print_job_result(_wait(args.api, jid, args.poll, args.timeout))


def _print_job_result(job: dict) -> None:
    jid = str(job.get("id", "?"))[:8]
    total = (job.get("timings") or {}).get("total")
    if total:
        print(f"    {jid}  {job['status']:<7}  "
              f"class {str(job.get('worker_class') or '?'):<10}  actual {total:6.0f}s")
    else:
        print(f"    {jid}  {job.get('status', '?')}")


def _print_stats(s: dict) -> None:
    n = s.get("observations", 0)
    if not n:
        print("no SLA observations yet — let some jobs finish.")
        return
    print(f"SLA observations: {n}\n")
    rows = [
        ("adaptive routing", s.get("sla_met_rate"), s.get("mean_time_vs_sla")),
        ("static baseline", s.get("baseline_sla_met_rate"), s.get("baseline_mean_time_vs_sla")),
    ]
    print(f"  {'':18}  {'SLA met':>9}  {'time / SLA':>11}")
    for label, met, ratio in rows:
        met_s = f"{met * 100:.0f}%" if met is not None else "  n/a"
        ratio_s = f"{ratio:.2f}x" if ratio is not None else "  n/a"
        print(f"  {label:18}  {met_s:>9}  {ratio_s:>11}")
    print(f"\n  prediction error: {s.get('prediction_error_ratio')}")
    by_class = s.get("by_worker_class", {})
    if by_class:
        print("\n  by worker class:")
        for name, b in sorted(by_class.items()):
            print(f"    {name:14}  n={b['n']:<3}  met {b.get('met_rate', 0) * 100:.0f}%")
    print("\n  cost model (sec per video-minute):")
    for c in s.get("cost_model", []):
        if c["key"] == "total:per_min":
            print(f"    {c['worker_class']:14}  {c['seconds']:.2f}  (n={c['samples']})")


if __name__ == "__main__":
    raise SystemExit(main())
