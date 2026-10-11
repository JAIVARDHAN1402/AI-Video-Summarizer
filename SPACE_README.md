---
title: AI Video Summarizer API
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
dockerfile_path: Dockerfile.hf
app_port: 7860
pinned: false
short_description: Backend for the AI-Based Video Summarization System
---

# AI Video Summarizer — backend

The FastAPI service behind the AI-Based Video Summarization System
(Project-I, BCSE497J, VIT Vellore). The browser interface is deployed
separately and talks to this Space over `/api/*`.

This file belongs at the **root of the Space repository, named `README.md`**.
The frontmatter above is what tells Spaces to build the Docker image rather
than look for a Gradio app.

## Setting it up

1. Create a Space: huggingface.co → New Space → **Docker** → **Blank**.
2. Push this repository to it, with this file renamed to `README.md`.
3. In the Space's **Settings → Variables and secrets**, add a secret:
   `GEMINI_API_KEY`. Without it the pipeline falls back to the local
   BART summariser, which is much weaker.
4. Optionally set `VS_CORS_ORIGINS` to the frontend's URL once it exists.

The first boot downloads roughly 2 GB of model weights, so give it time.
`/api/health` answers once the service is up; model warm-up continues in the
background and the first video is slower than the rest.

## What the free tier costs you

- **Storage is ephemeral.** A restart loses processed videos. Nothing breaks —
  the pipeline re-processes — but the library starts empty each time. Paid
  persistent storage keeps it.
- **The Space sleeps** after a period of inactivity and takes a minute to wake.
- **CPU only.** Transcription is the long pole and a GPU would change it by
  5–10×, but none of this needs one.

## Checking it works

```
GET  /api/health      → {"status":"ok", ...}
GET  /api/videos      → []   (empty until something is processed)
POST /api/jobs        → submit a URL
```

`llm_understanding` in the health response tells you whether the API key was
picked up.

## Pointing the frontend at it

In the Vercel project's environment variables:

```
API_ORIGIN = https://<user>-<space-name>.hf.space
```

`web/next.config.ts` reads it and proxies `/api/*` there, so the browser only
ever talks to one origin.
