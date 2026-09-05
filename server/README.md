# API tier (`server/`)

The multi-user service over the same `core` pipeline. The Streamlit app stays
as-is; this is additive.

## Run

```bash
pip install -r requirements-server.txt
uvicorn server.app:app --reload --port 8000
# or: python -m server.main
```

Open `http://localhost:8000/docs` for the interactive API.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Submit a URL — `{url, video_type, topic, profile, force}` |
| `GET` | `/api/jobs` | Recent jobs |
| `GET` | `/api/jobs/{id}` | One job's status + live progress |
| `GET` | `/api/jobs/{id}/events` | SSE stream of progress until terminal |
| `GET` | `/api/videos/{video_id}` | Finished result: understanding, scenes, transcript |
| `GET` | `/api/videos/{video_id}/keyframes/{name}` | A keyframe image |
| `POST` | `/api/videos/{video_id}/ask` | Grounded Q&A — `{question}` |
| `GET` | `/api/health` | Status, queue backend, hardware profile |

## Configuration (env)

| Var | Default | Meaning |
| --- | --- | --- |
| `VS_DATABASE_URL` | `sqlite:///server_state.db` | Job store — set a `postgresql+psycopg://` URL for prod |
| `VS_QUEUE_BACKEND` | `inprocess` | `inprocess` (local fleet inside the API) or `redis` (RQ workers) |
| `VS_REDIS_URL` | `redis://localhost:6379/0` | Used when `VS_QUEUE_BACKEND=redis` |
| `VS_WORKER_CONCURRENCY` | `1` | Jobs a worker runs at once (a job saturates the CPU) |
| `VS_WORKER_CLASS` | auto | Hardware label for the scheduler's per-class cost model |
| `VS_WORKER_SPEED` | `1.0` | Emulated relative compute speed; `<1.0` deliberately slows this worker (see below) |
| `VS_LOCAL_FLEET` | — | JSON list of in-process worker classes, e.g. `[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.3}]` |
| `VS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated frontend origins |
| `GEMINI_API_KEY` | — | Enables the LLM understanding layer (see main README) |

## The worker fleet (the scheduler substrate)

The SLA-constrained scheduler (`server/scheduler.py`) routes each job to the
cheapest **worker class** predicted to still hit the wall-clock target, keeps a
learned cost model per class, and writes an SLA observation (predicted vs
actual vs a fixed-configuration baseline) for every job. It needs more than one
worker class to do anything. Three ways to give it one, smallest first:

### 1. Zero infra — in-process two-class fleet

One process, one thread per class, one queue per class. No Redis, no Docker.

```bash
VS_LOCAL_FLEET='[{"class":"cpu-fast","speed":1.0},{"class":"cpu-slow","speed":0.25}]' \
  VS_API_PORT=8000 python -m server.main
```

`speed` below 1.0 emulates a slower box: the job's wall-clock is padded and its
recorded timings scaled by `1/speed`, so `cpu-slow`'s cost model and SLA
outcomes genuinely differ from `cpu-fast`'s. Routing, calibration and the SLA
feedback loop run unchanged — only the hardware diversity is simulated.

### 2. Redis + separate worker processes

```bash
pip install -r requirements-server.txt      # includes rq, redis
# terminal 1 — API
VS_QUEUE_BACKEND=redis python -m server.main
# terminal 2 — a fast worker
VS_QUEUE_BACKEND=redis VS_WORKER_CLASS=cpu-fast python -m server.worker
# terminal 3 — a slow worker (emulated, or a genuinely smaller box)
VS_QUEUE_BACKEND=redis VS_WORKER_CLASS=cpu-slow VS_WORKER_SPEED=0.25 python -m server.worker
```

Each worker drains `video.<its-class>` then the shared `video.any`. On Windows
the worker uses RQ's `SimpleWorker` automatically (no `fork`).

### 3. docker-compose — real CPU throttling, no emulation

```bash
docker compose up --build
```

Brings up Redis, the API, and `worker-fast` (`cpus: 2.0`) + `worker-slow`
(`cpus: 0.5`). The cgroup limits make the slow worker genuinely CPU-starved, so
every number is measured. `GEMINI_API_KEY` is read from a `.env` file beside
`docker-compose.yml`.

## Reading the evidence

```bash
python scripts/fleet_demo.py --api http://localhost:8000 --repeat 8 --profile fast
# or straight from the API
curl -s http://localhost:8000/api/scheduler/sla   | python -m json.tool
curl -s http://localhost:8000/api/scheduler/classes | python -m json.tool
```

`fleet_demo.py` submits N jobs, waits for them, and prints `sla_met_rate` for
the adaptive routing against `baseline_sla_met_rate` for a static
"always the accurate models" assignment, plus the learned per-class
`total:per_min` cost. The first jobs route on seed estimates; once each class
has real data the routing stops sending jobs to the class that would overrun.

## Not yet wired

- Auth — a real deployment puts JWT / Clerk middleware in front of `/api/*`.
- Uploads — `POST /api/jobs` currently takes a URL only.
- Object storage — artifacts live under `cache/<video_id>/`; `server/storage.py`
  is the seam for S3/R2.
