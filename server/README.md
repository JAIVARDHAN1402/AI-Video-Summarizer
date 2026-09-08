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
| `POST` | `/api/uploads` | Upload a video file (multipart `file`) → `{upload_id, filename, bytes}` |
| `POST` | `/api/jobs` | Submit `{url \| upload_id, video_type, topic, profile, force, sla_sec?}` (exactly one of `url`/`upload_id`) |
| `GET` | `/api/jobs` | Recent jobs |
| `GET` | `/api/jobs/{id}` | One job's status + live progress |
| `GET` | `/api/jobs/{id}/events` | SSE stream of progress until terminal |
| `GET` | `/api/videos/{video_id}` | Finished result: understanding, scenes, transcript |
| `GET` | `/api/videos/{video_id}/keyframes/{name}` | A keyframe image (never requires auth) |
| `GET` | `/api/videos/{video_id}/highlight` | M7 highlight reel (mp4), built on demand; 404 if no trained model |
| `POST` | `/api/videos/{video_id}/ask` | Grounded Q&A — `{question}` |
| `GET` | `/api/scheduler/{classes,sla,preview}` | Worker classes, SLA evidence, routing preview |
| `GET` | `/api/health` | Status, queue backend, auth/storage mode, hardware profile |

All `/api/*` routes except `/api/health` and the keyframe images require
`Authorization: Bearer <token>` **when auth is enabled** (see below); otherwise
they're open.

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
| `VS_API_KEYS` | — | Comma-separated bearer tokens. Setting any turns auth on |
| `VS_JWT_SECRET` | — | HS* secret / RS* public key for JWT bearer auth (needs PyJWT). Turns auth on |
| `VS_JWT_ALGORITHM` / `VS_JWT_AUDIENCE` | `HS256` / — | JWT verification params |
| `VS_MAX_UPLOAD_MB` | `512` | Upload size cap |
| `VS_UPLOAD_EXTS` | `.mp4,.mov,.mkv,.webm,.avi,.m4v` | Accepted upload extensions |
| `VS_STORAGE_BACKEND` | `local` | `local` or `s3` (AWS S3 / Cloudflare R2 / MinIO; needs boto3) |
| `VS_S3_BUCKET` / `VS_S3_ENDPOINT_URL` / `VS_S3_REGION` / `VS_S3_PREFIX` | — | S3 target (endpoint URL points at R2/MinIO) |
| `VS_MIRROR_ARTIFACTS` | `0` | With `s3`, copy each finished `cache/<video_id>/` tree to the bucket |
| `VS_HIGHLIGHT_MODEL` | auto | M7 checkpoint path; `off` to disable. Default `ml/checkpoints/importance.pt` if present |
| `VS_HIGHLIGHT_AUTO` / `VS_HIGHLIGHT_MAX_SEC` | `1` / `90` | Build the reel at end of run; reel length cap |

## Auth

Off by default — local dev and the bundled frontend need no setup. Set
`VS_API_KEYS` and/or `VS_JWT_SECRET` to require `Authorization: Bearer <token>`
on every `/api` route except `/api/health` and keyframe images (referenced by
bare `<img>` tags that can't send headers). API keys are static strings; JWTs
are verified with PyJWT against `VS_JWT_SECRET`. A JWT deployment fronts this
with Clerk/Auth0/etc. and passes their token straight through.

## Uploads & storage

`POST /api/uploads` streams the file to the storage backend, capping size and
checking the extension, and returns an `upload_id` (the content SHA-256, so
re-uploading the same file is free). Pass that as `upload_id` to `POST /api/jobs`
instead of `url`. With `VS_STORAGE_BACKEND=s3` the upload is also pushed to the
bucket, `upload_id`s are pulled back down on any worker that needs them, and —
with `VS_MIRROR_ARTIFACTS=1` — finished `cache/<video_id>/` trees are mirrored
up so a different machine can serve `GET /api/videos/{id}`. `server/storage.py`
is the whole seam; `core/` still only ever reads and writes `cache/` locally.

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
