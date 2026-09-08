"""Stand up a genuinely heterogeneous two-class worker fleet and collect the
scheduler's SLA evidence.

The hardware difference is real, not emulated: each worker process is pinned to
a different number of cores, so "cpu-8core" and "cpu-2core" differ in throughput
the way two machines would. VS_WORKER_SPEED pads wall-clock to *simulate* a
slower node — fine for exercising the control plane, weaker as evidence.

    python scripts/run_fleet_local.py --jobs 6
    python scripts/run_fleet_local.py --jobs 8 --fast-cores 12 --slow-cores 2

Needs Redis. On Windows it is started inside WSL (installed there once); on
Linux/macOS a local redis-server is used. An already-listening Redis is reused
and left alone on teardown.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG_DIR = Path(os.getenv("TEMP", "/tmp")) / "vs-fleet"
_children: list[subprocess.Popen] = []


def _venv_python() -> str:
    for rel in ("Scripts/python.exe", "bin/python"):
        p = ROOT / ".venv" / rel
        if p.exists():
            return str(p)
    return sys.executable


def _port_open(port: int, host: str = "localhost") -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def _wait_port(port: int, timeout: float, what: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return
        time.sleep(0.5)
    raise RuntimeError(f"{what} did not come up on :{port} within {timeout:.0f}s")


def _pin_tree(pid: int, cores: list[int], seconds: float = 25.0) -> int:
    """Pin a process and every descendant to ``cores``.

    Pinning just the spawned pid is not enough: a venv's python.exe on Windows
    is a launcher that re-execs the real interpreter as a child, so the pin
    lands on a one-thread stub while all the work runs unpinned on every core —
    silently erasing the difference between the two worker classes. Children
    also appear a moment after the parent, so keep re-applying for a while.
    """
    import psutil

    deadline = time.monotonic() + seconds
    seen: set[int] = set()
    while time.monotonic() < deadline:
        try:
            parent = psutil.Process(pid)
            targets = [parent, *parent.children(recursive=True)]
        except psutil.Error:
            break
        for p in targets:
            try:
                if p.pid not in seen or p.cpu_affinity() != cores:
                    p.cpu_affinity(cores)
                    seen.add(p.pid)
            except psutil.Error:
                pass
        time.sleep(1.0)
    return len(seen)


def _spawn(args: list[str], env_extra: dict[str, str], label: str,
           cores: list[int] | None = None) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = open(LOG_DIR / f"{label}.out.log", "w", encoding="utf-8", buffering=1)
    err = open(LOG_DIR / f"{label}.err.log", "w", encoding="utf-8", buffering=1)
    env = {**os.environ, **env_extra}
    proc = subprocess.Popen(args, cwd=str(ROOT), env=env, stdout=out, stderr=err)
    _children.append(proc)

    if cores:
        import threading

        threading.Thread(target=_pin_tree, args=(proc.pid, cores), daemon=True).start()
        print(f"  {label}: pid {proc.pid}, pinning tree to {len(cores)} cores {cores}")
    else:
        print(f"  {label}: pid {proc.pid}")
    print(f"    logs: {LOG_DIR / (label + '.err.log')}")
    return proc


def _wsl_ip(distro: str) -> str | None:
    try:
        out = subprocess.check_output(["wsl", "-d", distro, "--", "hostname", "-I"],
                                      timeout=30, stderr=subprocess.DEVNULL)
        ip = out.decode("utf-8", "ignore").replace("\x00", "").split()
        return ip[0] if ip else None
    except Exception:
        return None


def _redis_host(distro: str) -> str:
    """Pick a host that actually reaches Redis.

    WSL2's localhost forwarding is not dependable across distro restarts, and
    the distro does restart — so fall back to addressing the VM directly.
    """
    for host in ("localhost", _wsl_ip(distro)):
        if host and _port_open(6379, host):
            return host
    return "localhost"


def _start_redis(distro: str) -> str:
    """Ensure Redis is reachable and return the host to address it by."""
    if sys.platform == "win32":
        # WSL shuts a distro down once nothing is running in it, taking Redis
        # with it mid-run. Hold it open for the lifetime of the fleet.
        _spawn(["wsl", "-d", distro, "--", "sleep", "infinity"], {}, "wsl-keepalive")
        time.sleep(2)
        subprocess.call(["wsl", "-d", distro, "--", "sudo", "-n", "systemctl",
                         "start", "redis-server"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif not _port_open(6379):
        _spawn(["redis-server", "--protected-mode", "no"], {}, "redis")

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        host = _redis_host(distro)
        if _port_open(6379, host):
            print(f"  redis reachable at {host}:6379")
            return host
        time.sleep(1.0)
    raise RuntimeError("Redis did not become reachable on :6379")


def _uncached_videos(video_dir: Path, needed: int) -> list[Path]:
    """Pick videos the pipeline has not already processed.

    A cached run returns in about a second having done no work, so it teaches
    the cost model nothing (the scheduler now refuses to calibrate on one) and
    contributes no usable timing. For an evidence run every job must be real
    work, which means every job needs a video no earlier run has touched.
    """
    from core import cache

    out: list[Path] = []
    for p in sorted(video_dir.glob("*.mp4")):
        if len(out) >= needed:
            break
        try:
            if cache.artifact_exists(cache.get_video_id(str(p)), "summary"):
                continue
        except Exception:
            pass
        out.append(p)
    return out


def _shutdown() -> None:
    for p in reversed(_children):
        if p.poll() is None:
            try:
                p.kill()
            except Exception:
                pass


def _get(api: str, path: str) -> object:
    with urllib.request.urlopen(api.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--calibrate", type=int, default=2,
                    help="jobs run first at a loose SLA so every class gets measured")
    ap.add_argument("--fast-cores", type=int, default=8)
    ap.add_argument("--slow-cores", type=int, default=2)
    ap.add_argument("--profile", default="fast")
    ap.add_argument("--sla", type=float, default=None,
                    help="wall-clock target in seconds. Pick it where the class choice "
                         "actually decides the outcome: too loose and every class passes, "
                         "too tight and none does — neither tells you anything about routing.")
    ap.add_argument("--video-dir", default="data/highlight/SumMe/videos",
                    help="local videos to submit; avoids download variance in the timings")
    ap.add_argument("--url", default="", help="use this URL instead of local files")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--wsl-distro", default="Ubuntu-24.04")
    ap.add_argument("--keep-up", action="store_true")
    args = ap.parse_args()

    py = _venv_python()
    total_cores = os.cpu_count() or 4
    fast = list(range(0, min(args.fast_cores, total_cores)))
    slow = list(range(max(total_cores - args.slow_cores, 0), total_cores))

    if not args.keep_up:
        atexit.register(_shutdown)

    print("Redis...")
    redis_host = _start_redis(args.wsl_distro)

    fast_class = f"cpu-{len(fast)}core"
    slow_class = f"cpu-{len(slow)}core"
    common = {
        "VS_QUEUE_BACKEND": "redis",
        "VS_REDIS_URL": f"redis://{redis_host}:6379/0",
        "VS_OFFLINE": "1",
        # The API must know the fleet's classes, or it routes to its own
        # autodetected class and enqueues onto a queue nothing drains.
        "VS_WORKER_CLASSES": json.dumps([{"name": fast_class}, {"name": slow_class}]),
    }

    print(f"\nAPI on :{args.api_port}...")
    _spawn([py, "-m", "server.main"],
           {**common, "VS_API_PORT": str(args.api_port), "VS_WORKER_CLASS": fast_class},
           "api")
    _wait_port(args.api_port, 180, "API")
    api = f"http://localhost:{args.api_port}"
    print("  api up")

    print("\nWorkers (real core pinning, no speed emulation)...")
    _spawn([py, "-m", "server.worker"], {**common, "VS_WORKER_CLASS": fast_class}, "worker-fast", fast)
    _spawn([py, "-m", "server.worker"], {**common, "VS_WORKER_CLASS": slow_class}, "worker-slow", slow)
    time.sleep(10)

    demo = [py, "scripts/fleet_demo.py", "--api", api, "--repeat", str(args.jobs),
            "--profile", args.profile, "--serial", "--force",
            "--calibrate", str(args.calibrate)]
    if args.sla:
        demo += ["--sla", str(args.sla)]
    if args.url:
        demo += ["--url", args.url]
    else:
        # Jobs run with --force, so a repeat still does full work; distinct
        # videos are preferred only to keep the samples independent.
        needed = args.jobs + args.calibrate
        vids = sorted(Path(args.video_dir).glob("*.mp4"))[:needed]
        if not vids:
            print(f"no videos under {args.video_dir}", file=sys.stderr)
            return 2
        if not vids:
            print(f"no videos under {args.video_dir}", file=sys.stderr)
            return 2
        for v in vids:
            demo += ["--file", str(v)]

    print(f"\nDriving {args.jobs} jobs (+{args.calibrate} calibration)...\n")
    rc = subprocess.call(demo, cwd=str(ROOT))

    print("\n=== scheduler classes ===")
    print(json.dumps(_get(api, "/api/scheduler/classes"), indent=2))
    print("\n=== SLA evidence ===")
    print(json.dumps(_get(api, "/api/scheduler/sla"), indent=2))

    if args.keep_up:
        print(f"\n--keep-up set; fleet still running. Logs in {LOG_DIR}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
