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
| `VS_QUEUE_BACKEND` | `inprocess` | `inprocess` (thread pool in the API) or `redis` (RQ workers) |
| `VS_REDIS_URL` | `redis://localhost:6379/0` | Used when `VS_QUEUE_BACKEND=redis` |
| `VS_WORKER_CONCURRENCY` | `1` | Jobs a worker runs at once (a job saturates the CPU) |
| `VS_WORKER_CLASS` | auto | Hardware label for the scheduler's per-class cost model |
| `VS_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated frontend origins |
| `GEMINI_API_KEY` | — | Enables the LLM understanding layer (see main README) |

## Distributed workers (the scheduler substrate)

With `VS_QUEUE_BACKEND=redis`, jobs go to an RQ queue and workers run as
separate processes — on CPU boxes and GPU boxes independently:

```bash
# each worker machine
pip install rq redis
VS_WORKER_CLASS=gpu-a10 rq worker video --url redis://<host>:6379/0
```

The planner (`core/perf.py`) already keeps a learned cost model; running it
across worker classes is what turns it into a cross-hardware SLA scheduler.

## Not yet wired

- Auth — a real deployment puts JWT / Clerk middleware in front of `/api/*`.
- Uploads — `POST /api/jobs` currently takes a URL only.
- Object storage — artifacts live under `cache/<video_id>/`; `server/storage.py`
  is the seam for S3/R2.
