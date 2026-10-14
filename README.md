# AI-Based Video Summarization System



Given a video (YouTube URL or local upload), this system produces:

- a **teachable explanation** — the video taught back section by section, not
  just compressed, filling gaps the video leaves implicit,
- for programming videos, a **practice set**: every problem the video solves,
  with its solution in any dialect of the same language family,
- the **on-screen material** — each slide, diagram or whiteboard as a keyframe
  with its text extracted and explained,
- auto-generated chapters, a searchable transcript, a glossary, a quiz,
  flashcards, a mind map, a slide deck and a two-voice audio overview,
- a natural-language Q&A interface that answers with timestamp citations, and
- a highlight reel built from a learned scene-importance model.

Every transcript segment, scene, chapter, problem and Q&A answer carries
`start`/`end` seconds — timestamps are the backbone of the whole pipeline.

Two interfaces ship: a Streamlit app (`app.py`) and a production web tier — a
FastAPI service (`server/`) behind a Next.js frontend (`web/`) with accounts,
tiers and a distributed job scheduler.

---

## Status

| | |
| --- | --- |
| Test suite | **178 passing** (`pytest tests/`) |
| Pipeline stages | M1–M7, all implemented |
| Studio artefacts | 10, all rendering from real data |
| API routes | 36 |
| LLM providers | Gemini + any OpenAI-compatible endpoint, chained |
| Accounts | email OTP, phone OTP, Google OAuth, JWT sessions |
| Tiers | free / premium, metered in model calls, Razorpay (test mode) |
| Highlight model | 5-fold CV on SumMe + TVSum, checkpoint shipped |

---

## Pipeline

```
video (URL/upload)
      │
      ▼
  M1 ingest ──────────────► audio.wav  (+ execution plan)
      │
      ├───────────── run concurrently ─────────────┐
      ▼                                            ▼
M2 transcribe                              M3 vision
(silence-split, parallel Whisper)          (segmentation, YOLO, CLIP, OCR)
      │                                            │
      └────────────────────┬───────────────────────┘
                           ▼
                  M4 fuse (align transcript to scenes)
                           ▼
       M5′ understand (LLM: transcript + scenes + keyframes)
           └─ windowed above 45 min; falls back to M5 summarize
                           ▼
               M6 index (FAISS) ──► RAG Q&A
                           ▼
        M7 highlight (learned scene importance ──► highlight reel)
```

**M5′** ([`core/understand.py`](core/understand.py)) is what makes the output
teachable. It sends the transcript, the scene list and the downscaled
keyframes to a capable model — multimodal, so it reads diagrams and
handwriting that OCR misses. The persona and output schema switch on the video
type you pick. Without a provider the pipeline falls back to the local
BART/flan-t5 map-reduce.

### Long videos are understood in windows

One output budget cannot cover eleven hours: a three-hour video would get the
same number of tokens as a ten-minute one and the explanation would thin out
to nothing. Past `UNDERSTANDING_SINGLE_PASS_MAX_MIN` (45 minutes) the video is
cut into windows, each understood at full depth, and a final reduce pass
writes the overview, recap, key points and glossary across them.

The window count is capped (`VS_UNDERSTANDING_MAX_WINDOWS`, default 16)
because every window is a request. An eleven-hour video at 25-minute windows
is 27 calls plus a reduce, which is enough to exhaust a free tier on a single
video; the cap trades density for being able to finish at all.

Three failure modes on long videos are handled explicitly, each found by
running real eleven- and five-hour videos:

- **A window that fails does not fail the run.** The explanation covers the
  windows that succeeded and records how many did not.
- **A spent quota stops immediately** rather than retrying a limit that will
  not clear today, keeping the sections already written.
- **A failed reduce pass does not discard the windows.** The document is
  assembled from what was written, with an overview that says plainly it is a
  stand-in.

**M7** ([`ml/`](ml/README.md)) is a bi-GRU scene-importance model trained on
SumMe + TVSum. The vision stage attaches an `importance` score to every scene,
which orders the keyframes sent to the LLM and drives the highlight reel.

---

## The language-model layer

[`core/llm.py`](core/llm.py) is a provider seam, not a Gemini client. Two
client types implement the same `generate_json` contract:

- `LLMClient` — Gemini (`gemini-2.5-flash` by default): 1M-token context,
  multimodal, server-side JSON Schema enforcement.
- `OpenAICompatibleClient` — any chat-completions endpoint: OpenRouter, Groq,
  Cerebras, Together, Mistral, or a local Ollama.

### Several accounts, used in order

A daily quota belongs to an *account*, so a second account is a second quota —
and short of paying, that is the only thing that raises the ceiling. The
arithmetic is what matters: one eleven-hour video fully explored (explanation,
problem list, quiz, audio overview) is about **33 calls** against a free
allowance of around **50 a day**.

```
VS_LLM_BASE_URL / VS_LLM_MODEL / VS_LLM_API_KEY      first account
VS_LLM2_BASE_URL / VS_LLM2_MODEL / VS_LLM2_API_KEY   second
VS_LLM3_*  …  up to VS_LLM6_*                        and so on
GEMINI_API_KEY                                        used last
```

Each is used until its quota is spent, then the next. Only a *spent quota*
moves on — a bad reply or a bad minute is that provider's own business and it
retries that itself. A spent account is asked once for the whole run rather
than once per window, so a dead account costs a single request.

`VS_LLM_MODEL` may itself be a comma-separated preference order, for when one
model at a provider is rate-limited upstream but another is not.

### What this layer had to learn the hard way

Each of these was a real failure, reproduced before it was fixed:

| Symptom | Cause | Handling |
| --- | --- | --- |
| 11-hour video took 4.5 h and produced unreadable text | every window got HTTP 429, each retried 3× with 120 s waits — ~2.7 h of sleeping | a *per-day* quota is told apart from a *per-minute* one and fails fast (2.3 s vs ~360 s) |
| Every window of a 3-hour video came back empty | a requested "thinking budget" switched on hidden reasoning, which is charged against the same `max_tokens` as the answer — 4096 asked for, 6385 spent, nothing left | reasoning is off here whatever the caller asks; `VS_LLM_REASONING=1` re-enables it and adds the budget to the cap instead of taking it out |
| A 5.4-hour video lost every window it had paid for | the gateway reports upstream trouble *inside* a 200 — no choices, or `finish_reason: "error"` — both treated as fatal on first sight | both retried as the transient hiccups they are |
| Twelve requests sent against a quota already spent | OpenRouter says it in a shape the check did not read (`free-models-per-day`, `X-RateLimit-Remaining: 0`) | recognised at once, and the reset time reaches the screen: *"it resets at 05:30 on 14 Oct"* |
| A window's reply cut off mid-JSON, losing that window | how much a window needs varies with how dense it is | the call is retried once with a larger cap before the window is given up |
| Keyframes silently dropped on non-Gemini providers | assumed free models were text-only | several free models accept images; they are sent as vision content parts, and a model that refuses them re-sends as text rather than failing |

### Which free model to use

The constraint is rarely the model's intelligence. Measured on this project's
own cache, a 25-minute window of an 11-hour lecture is **7,000–10,000 tokens**,
and free tiers meter tokens per minute as well as per day:

- **OpenRouter** meters *requests per day* (around 50 without purchased
  credits, 1000 after a one-time $10), which suits the few large calls this
  pipeline makes.
- **Groq** is the fastest but allows a few thousand tokens per *minute*, which
  a single window can exceed on its own. Lower `VS_UNDERSTANDING_WINDOW_MIN`
  if you put it in the chain.

Free model availability rotates; `https://openrouter.ai/api/v1/models` is
readable without a key and lists context size, output cap, image support and
`response_format` support, which are the four things that matter here.

---

## What you get back

The video page is a three-pane workspace: the **source** (player, chapters,
scene strip), the **canvas** (Explanation, Scenes, Transcript, Chat, Notes),
and the **Studio** — ten artefacts, each derived from the explanation the
pipeline already produced rather than generated from scratch.

| Artefact | Built from | Notes |
| --- | --- | --- |
| Study Guide | explanation + key points | the walkthrough, one section per ~2.5 min |
| Practice Set | transcript + on-screen code | see below |
| Flashcards | glossary terms | click to reveal |
| Mind Map | chapter structure | SVG, click to seek |
| Data Table | scenes + timings | sortable by time, importance, objects |
| Slide Deck | chapters + keyframes | |
| Quiz | the explanation | 4 options, one correct, distractors from the misconceptions the video warns about |
| Audio Overview | the explanation | two voices, ~5 minutes, edge-tts |
| Infographic | the explanation | one-page visual summary, print/save |
| Run Report | pipeline timings | which levers the planner pulled, and where time went |

Plus **Chat** with saved threads, **Notes** that survive reprocessing, and
**Notebooks** — several processed videos treated as one body of material, with
answers that say which video each came from.

### Practice Set

A three-hour SQL practice video works through seventy questions and then they
are gone. The explanation says what the video *teaches*, not what it *asked*.

[`core/problems.py`](core/problems.py) reads the whole video — the speech and
the queries that were on screen, which is where the answers actually are — and
writes each problem down with its solution, difficulty, topic tags and the
timestamp it was solved at. Measured on that video: **57 problems in 4 parts
in 92 seconds**.

The other languages are deliberately *not* generated with it. Seventy problems
in five dialects is 350 answers nobody asked to read, and on a free tier it is
a month of requests. A problem is translated the first time it is opened in a
language it does not have, and every missing language is filled in together —
five dialects cost one request, not four.

Two families, which never mix (a SQL question "in Python" is a different
exercise, not a translation):

```
sql          MySQL · PostgreSQL · Oracle · SQL Server · SQLite
programming  Python · Java · C++ · JavaScript · C# · Go
```

The dialects genuinely differ where they should:

```sql
MySQL        SELECT * FROM employees ORDER BY hire_date DESC LIMIT 1;
Oracle       SELECT * FROM employees ORDER BY hire_date DESC FETCH FIRST 1 ROWS ONLY;
SQL Server   SELECT TOP 1 * FROM employees ORDER BY hire_date DESC;
```

…with a note saying why: *"Oracle uses FETCH FIRST instead of LIMIT."*

---

## Resume, or start over

Re-submitting a link that has been processed before used to resume from the
cache silently. The run that followed then reported **17 seconds for a
three-hour video** — true of that run, and worthless as a measurement, with
the Run Report going on to call it "624× faster than watching".

Submitting a link now asks first. `GET /api/preflight` reports which of the
five stages already have results, what the last run took, and whether it
failed; the page offers **Resume** (reuse what is done) or **Process fresh**
(redo everything), and says which to pick — resume after a failure, since the
hours of transcription are already paid for; fresh when the number itself is
the point.

So a resumed run can be read correctly afterwards, the pipeline records which
stages it only read back (`reused_stages`). The snapshot is taken before
anything runs, which makes it exact: a stage is cache-first, so an artefact
already on disk is one this run did not produce.

---

## Accounts, tiers and metering

Three ways in, one account. The account is keyed on the email address or phone
number itself rather than on a provider's user id, so signing in with Google
and later with a code sent to the same inbox lands on the same account instead
of quietly creating a second one. **No password is stored anywhere** — every
route in is either a code that expires in minutes or Google's own sign-in.

| Route | Needs | Status in this checkout |
| --- | --- | --- |
| Email + OTP | SMTP credentials | **configured** |
| Phone + OTP | Twilio or MSG91 | SID + key set, `VS_TWILIO_FROM` still needed |
| Google OAuth | `VS_GOOGLE_CLIENT_ID` + secret | not configured |

Sessions are JWTs signed with `VS_JWT_SECRET`; the server stores no session
state. Auth turns itself on when a secret is present, and every `/api` route
except health and the sign-in routes then requires a bearer token.

### The allowance is counted in model calls

A four-hour video is about **23 calls** and a half-hour one about **4**, so a
limit of "three videos a day" would overcharge every short video and
undercharge exactly the long ones the limit exists for. Calls are also what
actually runs out.

| | Free | Premium |
| --- | --- | --- |
| Longest video | 4 hours | 15 hours |
| Calls per day | 40 | 400 |

The estimate is not a guess: `server/quota.py` calls the same windowing the
pipeline will call, so changing the window settings changes the price with
them. A video too long for the plan is refused **before anything is
downloaded**, from the source's own metadata, in about a second.

Premium is sold through **Razorpay in test mode** (₹199 / 30 days). The
signature check is the part that matters — a client that could report its own
success would let anybody upgrade themselves, so the only evidence accepted is
an HMAC the server recomputes over the order and payment ids. Premium that has
expired is free again without a cleanup job, a repeated callback does not
extend the plan twice, and another user's order cannot be claimed.

---

## Designed around a time budget

The pipeline targets a **wall-clock budget rather than a fixed set of models**.
Before any work starts, [`core/perf.py`](core/perf.py) measures the machine,
reads the video's duration, and picks the heaviest configuration predicted to
finish inside `TARGET_PIPELINE_SEC` (5 minutes by default). A short clip gets
the accurate models; a 90-minute lecture automatically downshifts.

Every run reports what each stage actually cost back into a persisted profile,
so the planner **calibrates itself to the machine it runs on** instead of
trusting shipped constants:

```bash
python run_pipeline.py --plan-only --url "https://www.youtube.com/watch?v=..."
```

| Lever | Effect |
| --- | --- |
| Whisper model + beam size | Beam 5 → beam 1 is ~2.9× faster with little accuracy cost on clear speech |
| Parallel transcription | Audio is split at silences and transcribed concurrently — close to 4× on 16 cores |
| Summarizer model | BART-large → DistilBART trades some fluency for ~3× throughput |
| Keyframe cap | Fixes the per-scene cost (YOLO, CLIP, OCR) so a 90-min video costs what a 10-min one does |
| Segmentation method | Above 5 min, frame-accurate detection is replaced by CLIP comparison over sampled frames |
| Walkthrough length | One section per ~2.5 min, so a long lecture takes longer rather than losing content |
| OCR sampling | On-screen text is read from the most text-like frames, and skipped when a probe finds none |

---

## The distributed SLA-constrained scheduler

[`server/scheduler.py`](server/scheduler.py) routes each job to the worker
class that meets its wall-clock SLA most cheaply. It keeps a learned cost
model per hardware class, fitted online by least squares as an **affine
model** — `total = fixed + rate × minutes` — rather than a single ratio.

That shape matters. A single learned ratio, fitted on videos of 1.6–3.8
minutes, quoted **4 h 24 m** for an eleven-hour video. A recorded run of that
video actually took **90.2 minutes**; the affine fit predicts **94.7**.

Routing is cost-aware: cost is `predicted_sec × per-class hourly rate`, so the
cheapest class that still meets the SLA wins rather than the fastest.

Every run records predicted-vs-actual-vs-baseline, which is the evidence
behind the patent claim:

```
GET /api/scheduler/sla      (requires a bearer token when auth is on)
```

Three ways to give it more than one class — in-process, Redis + RQ workers, or
two CPU-limited containers:

```bash
docker compose up --build
python scripts/fleet_demo.py --api http://localhost:8000 --repeat 8
```

---

## Measured results

### Scheduler

Both numbers below are measured over the same recorded jobs, not simulated:
the adaptive scheduler against a fixed-configuration baseline recorded
alongside every run.

| | SLA met |
| --- | --- |
| Adaptive scheduler | **63.3 %** |
| Fixed-configuration baseline | 6.7 % |
| Always route to `cpu-16core` | 60.0 % |
| Always route to `cpu-8core` | 53.3 % |
| Always route to `cpu-2core` | 10.0 % |

### Highlight model (M7)

5-fold cross-validation on SumMe + TVSum, evaluated with the Otani et al.
(CVPR 2019) rank-correlation protocol rather than F1 alone:

| Metric | Value |
| --- | --- |
| Kendall τ (per annotator) | **0.160 ± 0.016** |
| Spearman ρ (per annotator) | 0.208 ± 0.021 |
| Kendall τ (vs mean annotator) | 0.285 ± 0.027 |
| Knapsack-summary F1 | 0.323 ± 0.034 |

Those averages hide the thing that makes them readable: the two datasets
behave completely differently. Measured on the same prepared shards:

| Summary F1 | random | **model** | human | oracle |
| --- | --- | --- | --- | --- |
| TVSum (12 of 15 videos per fold) | 0.166 | **0.265** | 0.260 | 0.406 |
| SumMe (3 of 15) | 0.368 | **0.451** | 0.612 | 0.783 |

On TVSum the model sits at human agreement — 0.265 against 0.260, from a
0.166 floor — and its per-annotator Kendall τ of **0.216** is above the ~0.18
that annotators achieve against each other under this protocol. On SumMe it
clears random but stays well short of human, with τ of 0.061. The headline
0.160 is those two averaged.

"human" is each annotator scored against the others, leave-one-out, which is
the realistic ceiling on a subjective task; "oracle" is the averaged
annotator score, which is the model's own training target. Absolute F1 is
not portable between summarisation pipelines — it depends on the feature
extractor, the sampling rate and the shot segmentation — so a published
paper's random baseline would not be a meaningful comparison. These are.

```bash
python -m ml.evaluate --checkpoint ml/checkpoints/importance.pt --baselines
```

### End-to-end pipeline

On the reference machine (16-core CPU, **no GPU**), against the original
sequential implementation:

| Video | Before | After |
| --- | --- | --- |
| 8.8-min English lecture | — | **3.7 min** |
| 12.6-min English lecture | — | **2.3 min** |
| 32-min Hindi lecture | 15 min | **7.7 min** |
| 42-min Hindi lecture | 26.4 min | **8.4 min** |
| 67-min English lecture | ~50 min | **5.3 min** |
| 3-hour SQL practice video | — | explanation in 7 min (12/13 windows) |
| 5.4-hour SQL video | failed outright | 137 sections, 65 chapters, 58 scene notes |

Component measurements behind those:

| | Before | After |
| --- | --- | --- |
| ASR, 120 s of audio | 93.5 s (beam 5) | 32.3 s (beam 1), ~4× again in parallel |
| ASR, 67 min of audio | ~50 min | 195 s (20× realtime) |
| Summarization, per chunk | 64.9 s | 24.1 s from batching; ~14 s distilled |
| Segmentation, 67-min video | 140 s | ~40 s, overlapped with transcription |
| Object detection, per frame | 0.177 s (640 px) | 0.089 s (416 px) |
| Cold model loading | ~70 s inside the first run | moved to app startup |

**The 5-minute target is not universal.** It holds for videos up to roughly 30
minutes, and for longer videos with sparse speech. It does not hold for a 40+
minute lecture of continuous narration on a CPU-only machine: Whisper is
charged per second of *speech*, and translation costs more than transcription.
The Report tab shows the prediction alongside the actual time, and warns when
even the fastest configuration is expected to overrun.

---

## Language handling

The spoken language is detected from three 30-second probes taken from across
the audio, using a small multilingual model. Two details matter:

- **Not just the opening window.** Whisper's own auto-detection reads only the
  first segment, which on a lecture is often an intro jingle or sponsor read.
- **Not a pinned language.** This project previously hardcoded English. Given
  a Hindi-English lecture, Whisper was forced to decode Hindi as English and
  produced transliterated nonsense — and ran several times slower doing it.
  That garbage transcript then poisoned the summary, the chapters and the Q&A.

When the detected language is not English, the audio is **translated** to
English rather than transcribed verbatim, because every downstream model
(BART, flan-t5, MiniLM retrieval) is English-only. An English-only `.en`
checkpoint is swapped for its multilingual sibling automatically.

Set `VS_WHISPER_LANGUAGE` in `.env` to force a language and skip detection.

---

## Prerequisites

1. **Python 3.10 or 3.11.** Several pinned ML dependencies (`torch==2.3.1`)
   lag behind on wheels for 3.12/3.13.
2. **FFmpeg, on PATH.** Checked at startup.
   - Windows: `winget install --id Gyan.FFmpeg -e`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
3. A CPU-only laptop is enough. A GPU is auto-detected but never required.
4. **Node 18+** for the web frontend.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

On CPU-only machines install the CPU-only torch build first — the default
PyPI wheel bundles CUDA and is several GB:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. **Nothing in it is required** — the pipeline
runs fully offline on local models; everything there buys quality, speed or
the multi-user features.

## First run

Models are downloaded from Hugging Face / Ultralytics / open_clip on first use
and cached under `~/.cache/`. Because the planner chooses models per video, do
it once up front:

```bash
python warmup.py           # everything (~5 GB)
python warmup.py --check   # list what would be downloaded
python warmup.py --asr     # just the speech models
```

Once cached, set `VS_OFFLINE=1` in `.env`: Hugging Face checks for model
updates on every load, which turns a fully-cached run into minutes of retry
loops on a slow network.

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the production web tier

```bash
pip install -r requirements-server.txt
python -m server.main                   # FastAPI on :8000  (/docs for the API)

cd web && npm install && npm run dev     # Next.js on :3000, proxies /api/*
```

Set `VS_JWT_SECRET` to turn on accounts; the web app then shows the account
menu, the remaining allowance, and — when Razorpay is configured — an upgrade
action. With no secret set the API stays open, which is how it runs locally.

### Run headlessly (CLI)

```bash
python run_pipeline.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"
python run_pipeline.py --file ./sample_lecture.mp4
python run_pipeline.py --url "..." --profile accurate
python run_pipeline.py --plan-only --url "..."
python run_pipeline.py --video-id <video_id> --ask "What is normalization?"
```

Profiles are `fast`, `balanced` (default) and `accurate`; `--budget` sets the
target in seconds directly. Re-running the same source is near-instant — every
stage checks its cached artefact under `cache/<video_id>/` first. Pass
`--force` to reprocess.

---

## Configuration

Every constant lives in [`config.py`](config.py) (pipeline) and
[`server/config.py`](server/config.py) (web tier), each overridable by an
environment variable. The ones that change behaviour most:

### Models and providers

| Variable | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Enables the teachable explanation via Gemini |
| `VS_LLM_BASE_URL` / `_MODEL` / `_API_KEY` | — | Any OpenAI-compatible endpoint; takes precedence over Gemini |
| `VS_LLM2_*` … `VS_LLM6_*` | — | Further accounts, used in order as each is spent |
| `VS_LLM_MAX_OUTPUT_TOKENS` | 8192 | Clamp; free models reject a larger ask with a 400 |
| `VS_LLM_SEND_IMAGES` | 1 | Send keyframes to providers that take them |
| `VS_LLM_REASONING` | auto | `0` forces reasoning off, `1` on with its own headroom |
| `VS_WHISPER_LANGUAGE` | auto | Force a language and skip detection |

### Pipeline shape

| Variable | Default | Effect |
| --- | --- | --- |
| `VS_TARGET_SEC` | 300 | The wall-clock budget the planner aims at |
| `VS_UNDERSTANDING_SINGLE_PASS_MIN` | 45 | Above this, understanding is windowed |
| `VS_UNDERSTANDING_WINDOW_MIN` | 25 | Window length in video-minutes |
| `VS_UNDERSTANDING_MAX_WINDOWS` | 16 | Cap, so one video cannot drain a quota |
| `VS_PROBLEMS_MAX_WINDOWS` | 14 | Same cap for the practice set |
| `VS_CACHE_DIR` | `cache/` | Where artefacts live |
| `VS_OFFLINE` | 0 | Skip Hugging Face update checks |

### Accounts, tiers and payments

| Variable | Default | Effect |
| --- | --- | --- |
| `VS_JWT_SECRET` | — | Turns on accounts; signs session tokens |
| `VS_SMTP_HOST` / `_USER` / `_PASSWORD` / `_FROM` | — | Email one-time codes |
| `VS_SMS_PROVIDER` (`twilio`\|`msg91`) + keys | — | Phone one-time codes |
| `VS_TWILIO_SID` / `VS_TWILIO_FROM` | — | Twilio account and sending number |
| `VS_GOOGLE_CLIENT_ID` / `_SECRET` / `_REDIRECT_URI` | — | Google sign-in |
| `VS_DEV_OTP_ECHO` | 1 | Writes the code to the log when no sender is configured — **never leave on in a deployment** |
| `VS_FREE_MAX_VIDEO_HOURS` / `VS_FREE_DAILY_CALLS` | 4 / 40 | The free allowance |
| `VS_PREMIUM_MAX_VIDEO_HOURS` / `VS_PREMIUM_DAILY_CALLS` | 15 / 400 | The premium allowance |
| `VS_RAZORPAY_KEY_ID` / `_KEY_SECRET` / `_MODE` | — / test | Payments |

---

## API

36 routes. `/docs` serves the generated OpenAPI reference.

```
GET    /api/health                                  liveness
GET    /api/auth/methods                            which sign-in routes exist
POST   /api/auth/otp/request | /verify              one-time code sign-in
GET    /api/auth/google/url  | POST /callback       Google sign-in
GET    /api/auth/me                                 the account and its allowance
GET    /api/billing/plan | POST /order | /confirm   upgrading to premium

POST   /api/jobs                                    start a run (quota-gated)
GET    /api/jobs | /{id} | /{id}/events             listing, status, SSE progress
POST   /api/jobs/{id}/cancel                        stop at the next stage boundary
GET    /api/preflight                               what already exists for a link
POST   /api/uploads                                 upload a file to process

GET    /api/videos | /{id}                          library, and one full result
DELETE /api/videos/{id} | ?confirm=DELETE           forget one, or all
GET    /api/videos/{id}/keyframes/{name}            a keyframe image
GET    /api/videos/{id}/highlight                   the highlight reel
POST   /api/videos/{id}/ask                         grounded Q&A with citations
GET/POST /api/videos/{id}/quiz                      the quiz
GET/POST /api/videos/{id}/problems                  the practice set
POST   /api/videos/{id}/problems/{n}/solutions      fill in other languages
GET/POST /api/videos/{id}/audio-overview            two-voice conversation
GET/POST/DELETE /api/videos/{id}/notes[/{id}]       the reader's own notes

GET/POST/DELETE /api/notebooks[/{id}]               multi-video collections
POST   /api/notebooks/{id}/ask                      ask across all of them
GET/POST/DELETE /api/chats[/{id}]                   saved conversation threads

GET    /api/scheduler/classes | /sla | /preview     routing and its evidence
```

---

## Testing

```bash
pytest tests/
```

**178 tests.** `tests/conftest.py` points the whole suite at a throwaway
database — the real `server_state.db` holds the scheduler's measured SLA
observations, which are evidence, and a test that wrote to it would be quietly
editing a result.

| File | Tests | Covers |
| --- | --- | --- |
| `test_openai_compatible.py` | 54 | provider chaining, quota detection, reasoning, vision parts, truncation recovery |
| `test_accounts.py` | 23 | one-time codes, account identity, tiers, allowances, payment signatures |
| `test_problems.py` | 21 | problem extraction, language families, windowing, translation carry-over |
| `test_cache.py` | 17 | artefact read/write, state carry-forward, delete |
| `test_understand_windows.py` | 15 | windowing, quota handling, degraded overview |
| `test_server_auth_uploads.py` | 10 | the auth dependency, uploads, the storage seam |
| `test_fuse.py` | 8 | transcript-to-scene alignment |
| `test_fleet_scheduler.py` | 6 | routing, exploration, speed emulation |
| `test_ml_highlight.py` | 6 | the importance model |
| `test_resume_choice.py` | 5 | which stages are already done |
| `test_chunking.py` | 5 | map-stage chunking on a synthetic 45-min transcript |
| `test_highlight.py` | 3 | reel assembly |
| `test_progress_threading.py` | 3 | merged progress from concurrent stages |
| `test_qa_quota.py` | 2 | what the reader is told when the model cannot answer |

Frontend types: `cd web && npx tsc --noEmit`.

## Evaluation

```bash
python -m ml.train --prepare --data-root data/highlight   # retrain M7
python eval/run_rouge.py --input eval/reference_summaries.json
python eval/run_keyframe_f1.py --gt eval/ground_truth.json
python eval/run_qa_eval.py --testset eval/qa_testset.json
```

SumMe/TVSum are visual-only benchmarks with little meaningful speech, so they
validate the keyframe module only. Transcript summarization is evaluated
separately on speech-heavy videos (TED / NPTEL) against human-written
references.

---

## Repository layout

```
core/
├── perf.py           # hardware profiling + the adaptive execution planner
├── llm.py            # provider seam: Gemini, OpenAI-compatible, chaining
├── ingest.py         # M1 — download/upload, probe, extract audio
├── transcribe.py     # M2 — YouTube captions or parallel faster-whisper ASR
├── vision.py         # M3 — segmentation, keyframes, YOLO, CLIP, OCR
├── fuse.py           # M4 — align transcript to scenes
├── summarize.py      # M5 — local map-reduce fallback
├── understand.py     # M5′ — LLM teachable explanation, windowed for long videos
├── problems.py       # practice set: problems solved + multi-dialect solutions
├── quiz.py           # multiple-choice questions from the explanation
├── notes.py          # the reader's own notes
├── audio_overview.py # two-voice conversation (edge-tts)
├── collection.py     # notebooks: Q&A across several videos
├── qa.py             # M6 — FAISS retrieval + generation
├── highlight.py      # M7 — highlight reel from learned scene importance
├── warm.py           # background model preload
├── pipeline.py       # orchestrator (M1→M7) + UI façade
└── cache.py          # cache dir, job state, stage completion, JSON read/write

server/
├── app.py            # 36 routes: jobs, SSE, results, auth, billing, scheduler
├── identity.py       # one-time codes, Google OAuth, JWT sessions
├── quota.py          # call estimation, tier allowances, usage ledger
├── billing.py        # Razorpay orders and signature verification
├── notify.py         # email/SMS delivery for one-time codes
├── auth.py           # the bearer-token dependency
├── scheduler.py      # SLA-constrained routing + per-class cost calibration
├── jobs.py           # job lifecycle, local fleet / RQ queue, cancellation
├── worker.py         # RQ worker entrypoint (Windows-safe)
├── db.py             # SQLAlchemy models (SQLite → Postgres by URL)
└── storage.py        # local disk / S3 seam

web/                  # Next.js 15 (App Router, Tailwind)
├── app/              # /, /library, /notebooks, /n/[id], /v/[id], /jobs/[id], /auth/google
├── components/       # Workspace, StudioPanel, Problems, Quiz, Chat, AccountMenu, …
└── lib/api.ts        # the typed API client

ui/                   # Streamlit presentation only
ml/                   # M7 — SumMe/TVSum importance model
├── datasets.py features.py kts.py model.py train.py evaluate.py infer.py
└── checkpoints/      # importance.pt + cv_report.json
```

---

## Known limitations

- **Free-tier quotas are the binding constraint, not the code.** One
  eleven-hour video fully explored is about 33 model calls against a free
  allowance of around 50 a day. Chain more accounts (`VS_LLM2_*`), or buy the
  one-time $10 of OpenRouter credit that lifts the daily limit to 1000.
- **Offline Q&A is bounded by `flan-t5-base`.** It answers extractively,
  often lifting a clause from the transcript. Retrieval and citations are
  sound; if retrieval finds nothing related the system declines rather than
  inventing an answer, and when the model is out of quota it says so instead
  of passing an excerpt off as a confident answer.
- **Long videos trade accuracy for time.** Past roughly 45 minutes the planner
  selects smaller ASR and summarization models to hold the budget.
- **YouTube downloads can fail** (geo-block, private video, bot-check). Upload
  the file directly instead — that path is independent of `yt-dlp`. An old
  `yt-dlp` pin is the most common cause of "Please sign in" errors.
- **M7 is a relative ranking, not a calibrated importance score.** It is
  scored on keyframe embeddings, a shorter and sparser sequence than the 2 fps
  training data.
- **The saved `cv_report.json` has no baselines**, though `ml/evaluate.py` can
  compute them. Do not quote τ = 0.160 without them.
- **Cancelling a job lands at the next stage boundary**, not instantly.
- **The in-process worker fleet serialises the pipeline** (one heavy run at a
  time per process, since warmed models are shared). Real parallelism needs
  `VS_QUEUE_BACKEND=redis` or docker-compose.
- **`ml/checkpoints/importance.pt` and `yolov8n.pt` are gitignored**, so a
  fresh clone has no highlight feature until they are shipped separately.
- Real-time/live processing, speaker diarization and emotion/action
  recognition are out of scope.
- Static, single-shot recordings produce few content-based scene cuts;
  `core/vision.py` falls back to fixed 60-second intervals below 2 scenes.
- **Windows only:** `torch` and `ctranslate2` each bundle their own OpenMP
  runtime, which crashes with `OMP Error #15` if both load in one process.
  `config.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import, before either loads.
- If the project lives in a OneDrive/Dropbox folder the sync client can lock
  cache files as they are written; `core/cache.py` retries, but pointing
  `VS_CACHE_DIR` at a non-synced path avoids it entirely.

---

## Security notes

- **No secret is ever committed.** `.env` is gitignored; `.env.example` ships
  empty values. Verified with `git log -S` over the full history.
- **No passwords are stored.** Sign-in is one-time codes or Google only.
- One-time codes are stored **hashed and salted with the destination**, expire
  in ten minutes, are single-use, and are cut off after five wrong attempts —
  six digits is a million guesses, which is nothing for a script.
- The resend endpoint is rate-limited per destination, so it cannot be used to
  send somebody a hundred messages.
- A code that could not be delivered is invalidated, so a failed send never
  leaves a live code nobody received.
- An unverified Google email address is refused: anyone can put someone
  else's address on a profile.
- Payment confirmation trusts **only** an HMAC the server recomputes.
- `VS_DEV_OTP_ECHO` writes codes to the log and must be `0` anywhere the log
  is not private.
