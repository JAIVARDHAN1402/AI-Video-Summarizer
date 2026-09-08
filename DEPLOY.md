# Deploying

The two halves have very different needs, and only one of them belongs on
Vercel.

| | Frontend (`web/`) | Backend (`server/`) |
| --- | --- | --- |
| What it is | Next.js 15, static + SSR | FastAPI running the whole ML pipeline |
| Size | a few MB | ~1.2 GB of Python deps (torch alone is 1 GB) |
| Per-request work | milliseconds | minutes — a 43-minute video takes ~6 |
| Needs | nothing | ffmpeg, ~4 GB RAM, a persistent disk |
| Host | **Vercel** | **a real container host** |

Vercel's serverless functions cap at 250 MB and 60 s (300 s on Pro), so the
backend cannot run there — it is not a configuration problem.

## 1. Backend first

The frontend needs the backend's URL, so deploy this one first.

**Render** (config is in `render.yaml`):

1. render.com → New → Blueprint → point it at this repo.
2. Set `GEMINI_API_KEY` in the dashboard. Without it the pipeline silently
   falls back to the local summariser, which is much worse.
3. Deploy. The first boot downloads ~2 GB of model weights, so give it time;
   `/api/health` turns green once warm-up finishes.

**Fly.io** or **Railway** work the same way from the `Dockerfile`. Whatever you
pick, it needs:

- **≥ 4 GB RAM.** Whisper, YOLO, CLIP, EasyOCR, FAISS and a sentence-transformer
  are all resident at once. A 512 MB free tier is OOM-killed during warm-up,
  before serving anything.
- **A persistent disk** mounted where `VS_CACHE_DIR` points. Without it every
  deploy throws away every processed video.
- `PORT` respected — `server/main.py` reads it.

Note the price of CPU-only inference: transcription is the long pole and a
cheap shared-CPU instance will be *slower* than the laptop numbers in the
README. A GPU host changes that by 5-10x if it ever matters enough.

## 2. Frontend on Vercel

```bash
cd web
npx vercel login
npx vercel --prod
```

Then set, in the Vercel project's environment variables:

```
API_ORIGIN = https://<your-backend>.onrender.com
```

`web/next.config.ts` reads it and proxies `/api/*` there, so the browser only
ever talks to one origin and you need no CORS for normal use.

## 3. Point the backend back at the frontend

Set on the backend, then redeploy:

```
VS_CORS_ORIGINS = https://<your-app>.vercel.app
```

Only needed for requests the browser makes directly rather than through the
proxy, but set it — it costs nothing and the failure mode (silently blocked
requests) is annoying to diagnose.

## What is not in the image

`.gitignore` excludes model weights, so a fresh deployment has:

- **no `ml/checkpoints/importance.pt`** → scene-importance scores and the
  highlight reel stay off. The pipeline is otherwise unchanged. Train it
  (`ml/README.md`) and ship the checkpoint separately if you want the feature.
- **no `yolov8n.pt`** → ultralytics downloads it on first run.

## Locally

```bash
python -m server.main          # backend on :8000
cd web && npm run dev          # frontend on :3000
```
