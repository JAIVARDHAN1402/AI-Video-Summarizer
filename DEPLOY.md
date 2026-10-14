# Deploying

The two halves have very different needs, and only one of them belongs on
Vercel.

| | Frontend (`web/`) | Backend (`server/`) |
| --- | --- | --- |
| What it is | Next.js 15, static + SSR | FastAPI running the whole ML pipeline |
| Size | a few MB | ~1.2 GB of Python deps (torch alone is 1 GB) |
| Per-request work | milliseconds | minutes — a 43-minute video takes ~6 |
| Needs | nothing | ffmpeg, ~2 GB RAM, ideally a persistent disk |
| Host | **Vercel** | **a real container host** |

Vercel's serverless functions cap at 250 MB and 60 s (300 s on Pro), so the
backend cannot run there — it is not a configuration problem.

## Picking a backend host

The constraint that rules most free tiers out is memory: the API sits at
**1.26 GB resident** with YOLOv8, CLIP and the retrieval embedder warm, and
peaks higher while a job runs twelve parallel Whisper workers. A 512 MB
instance is OOM-killed during warm-up, before serving a single request.

| Host | Free? | Verdict |
| --- | --- | --- |
| **Cloudflare quick tunnel** | Yes, no card, no account | **What works today.** Not hosting: it publishes the app already running on your machine. One command, a public HTTPS URL in seconds. The machine has to stay on and the URL changes each restart. |
| **Hugging Face Spaces** (Docker) | **No — PRO only** | Docker and Gradio Spaces now require a paid plan; only Static Spaces (HTML/JS) are free, and a Python backend cannot run on one. `Dockerfile.hf` is kept for anyone who has PRO. |
| **Render** | No | The free tier is 512 MB and has no persistent disk. `render.yaml` here targets the 2 GB `standard` plan, which is paid. |
| **Fly.io / Railway** | Card required | Workable, but both want payment details up front. |
| **Oracle Cloud Always Free** | Card for identity only | A real VM with generous memory, free indefinitely. Most setup work of the options here. |
| **Cloudflare Tunnel from your own machine** | Yes, no card | Not hosting: it publishes the backend already running on your laptop. Good for a demo; the laptop has to stay on. |

## 1. Public URL with no host at all (free, works today)

One tunnel is enough for the whole app. Next.js proxies `/api/*` to the
backend server-side, so exposing port 3000 exposes both halves — there is no
need to tunnel 8000 as well, and no `API_ORIGIN` to keep in sync.

```bash
# both servers running locally first
python -m server.main          # :8000
cd web && npm run dev          # :3000

cloudflared tunnel --url http://localhost:3000
```

It prints a `https://<words>.trycloudflare.com` URL that anyone can open.
Verified end to end: the home page, `/api/health` and `/api/videos` all
answer through it.

What it costs: your machine has to stay on and connected, and the URL is
random and changes every time you restart the tunnel. For a review demo or
sending a link to someone, that is usually fine. For a URL that outlives your
laptop, use a real host below.

## 2. Backend on Hugging Face Spaces (needs PRO)

1. huggingface.co → **New Space** → SDK **Docker** → **Blank**.
2. Push this repository to the Space, with `SPACE_README.md` renamed to
   `README.md` at the root. Its frontmatter points Spaces at
   `Dockerfile.hf` and port 7860.
3. Space **Settings → Variables and secrets** → add secret `GEMINI_API_KEY`.
   Without it the pipeline falls back to the local BART summariser, which is
   much worse.
4. Build and wait. The first boot downloads ~2 GB of model weights.
   `/api/health` answers once the service is up; its `llm_understanding`
   field tells you whether the key was picked up.

`Dockerfile.hf` differs from `Dockerfile` in two ways that matter and are the
reason it is a separate file: Spaces runs the container as uid 1000, so every
path the process writes to is created and chowned to that user; and Spaces
routes to one declared port rather than injecting `PORT`.

What the free tier costs: storage is ephemeral, so a restart empties the
library — the pipeline simply re-processes. The Space also sleeps when idle
and takes a minute to wake.

> The env-var handling this image depends on is verified: with `PORT`,
> `VS_CACHE_DIR`, `VS_DATA_ROOT` and `VS_DATABASE_URL` set, the service binds
> 7860, writes its cache and database where told, and creates all five tables
> on a clean database. The image itself has **not** been built and run here —
> no Docker daemon was available — so treat the first Space build as the real
> test.

## 3. Backend on Render (paid)

Config is in `render.yaml`: render.com → New → Blueprint → point it at this
repo, set `GEMINI_API_KEY` in the dashboard, deploy. It targets the 2 GB
`standard` plan and a 20 GB disk mounted at `/data`, which is what gives it
persistence the Spaces free tier does not have.

## 4. Frontend on Vercel

```bash
cd web
npx vercel login
npx vercel --prod
```

Then set, in the Vercel project's environment variables:

```
API_ORIGIN = https://<your-space>.hf.space
```

`web/next.config.ts` reads it and proxies `/api/*` there, so the browser only
ever talks to one origin and you need no CORS for normal use.

**Deploy the backend first.** The frontend on its own renders the form and
then fails on every request, because there is nothing behind `/api/*`.

## 5. Point the backend back at the frontend

Set on the backend, then redeploy:

```
VS_CORS_ORIGINS = https://<your-app>.vercel.app
```

Only needed for requests the browser makes directly rather than through the
proxy, but set it — it costs nothing and the failure mode (silently blocked
requests) is annoying to diagnose.

## 6. Optional reader accounts and premium tier

The backend is open by default. To enable the account menu in the bundled
frontend, add these backend secrets before deploy:

```
VS_JWT_SECRET = <long random secret>
VS_DEV_OTP_ECHO = 0
VS_SMTP_HOST = <smtp host>
VS_SMTP_USER = <smtp user>
VS_SMTP_PASSWORD = <smtp password>
VS_SMTP_FROM = <from address>
```

For Google, register `https://<your-frontend>/auth/google` as the exact OAuth
redirect URI and set `VS_GOOGLE_CLIENT_ID`, `VS_GOOGLE_CLIENT_SECRET`, and
`VS_GOOGLE_REDIRECT_URI` on the backend. The callback checks the browser's
one-time state before exchanging Google's code.

Premium checkout remains hidden until both `VS_RAZORPAY_KEY_ID` and
`VS_RAZORPAY_KEY_SECRET` are set. Use Razorpay test keys first; the server,
not the browser, verifies the payment signature before changing a tier.

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
