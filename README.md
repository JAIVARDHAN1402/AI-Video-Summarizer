# AI-Based Video Summarization System

**VIT Vellore — Project-I (BCSE497J)**
Jaivardhan Singh (23BCI0036) · Sujay Chand (23BCE0217) · Harshit Raj (23BCE0190)
Guide: Dr. Deepa K, SCOPE

Given a video (YouTube URL or local upload), this system produces:
- a timestamped, abstractive textual summary,
- auto-generated chapters,
- a searchable transcript, and
- a natural-language Q&A interface that answers with timestamp citations.

Every transcript segment, scene, chapter, and Q&A answer carries `start`/`end`
seconds — timestamps are the backbone of the whole pipeline.

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
              M5 summarize (map-reduce, chapters)
                           ▼
               M6 index (FAISS) ──► RAG Q&A
```

M7 (auto-generated highlight video) is a stub — see [`core/highlight.py`](core/highlight.py).

## Designed around a time budget

The pipeline targets a **wall-clock budget rather than a fixed set of models**.
Before any work starts, [`core/perf.py`](core/perf.py) measures the machine,
reads the video's duration, and picks the heaviest configuration predicted to
finish inside `TARGET_PIPELINE_SEC` (5 minutes by default). A short clip gets
the accurate models; a 90-minute lecture automatically downshifts.

Every run reports what each stage actually cost back into a persisted profile,
so the planner **calibrates itself to the machine it runs on** instead of
trusting shipped constants. Inspect what it would choose without processing
anything:

```bash
python run_pipeline.py --plan-only --url "https://www.youtube.com/watch?v=..."
```

The choices it makes, and why each one exists:

| Lever | Effect |
| --- | --- |
| Whisper model + beam size | Beam 5 → beam 1 is ~2.9× faster with little accuracy cost on clear speech |
| Parallel transcription | Audio is split at silences and transcribed concurrently — close to 4× on 16 cores |
| Summarizer model | BART-large → DistilBART trades some fluency for ~3× throughput |
| Keyframe cap | Fixes the per-scene cost (YOLO, CLIP, OCR) so a 90-min video costs what a 10-min one does |
| Segmentation method | Above 5 min, frame-accurate scene detection is replaced by CLIP comparison over sampled frames — it decodes every frame, costing 153 s on a 9-minute video |
| Walkthrough length | The section-by-section walkthrough covers the whole video — one section per ~2.5 min — so a long lecture takes longer rather than losing content (the Fast profile caps it) |
| OCR sampling | On-screen text is read from the most text-like frames, and skipped entirely when a probe finds none |

The **Report** tab in the app shows exactly which of these were applied to the
video you just processed, and where the time went.

## What you get back

The **Summary** tab leads with a *detailed walkthrough* — the whole video
read section by section, one paragraph per ~2.5 minutes, each linking to the
moment it covers — and ends with a short **In summary** recap. Earlier builds
produced only a 100-word summary for an hour of video because most of the
transcript was discarded before summarizing; now nothing is dropped, so the
walkthrough grows with the video (a 67-minute lecture yields ~27 sections).
The Fast profile keeps it short when you want a quick pass.

### Measured effect

End to end on the reference machine (16-core CPU, **no GPU**), against the
original sequential implementation:

| Video | Before | After |
| --- | --- | --- |
| 8.8-min English lecture | — | **3.7 min** |
| 12.6-min English lecture | — | **2.3 min** |
| 32-min Hindi lecture | 15 min | **7.7 min** |
| 42-min Hindi lecture | 26.4 min | **8.4 min** |
| 67-min English lecture | ~50 min | **5.3 min** |

Component measurements behind those:

| | Before | After |
| --- | --- | --- |
| ASR, 120 s of audio | 93.5 s (beam 5) | 32.3 s (beam 1), ~4x again in parallel |
| ASR, 67 min of audio | ~50 min | 195 s (20x realtime) |
| Summarization, per chunk | 64.9 s | 24.1 s from batching; ~14 s distilled |
| Segmentation, 67-min video | 140 s | ~40 s, overlapped with transcription |
| Object detection, per frame | 0.177 s (640px) | 0.089 s (416px) |
| Cold model loading | ~70 s inside the first run | moved to app startup |

**The 5-minute target is not universal.** It holds comfortably for videos up
to roughly 30 minutes, and for longer videos with sparse speech. It does not
hold for a 40+ minute lecture of continuous narration on a CPU-only machine:
Whisper is charged per second of *speech*, and translation (used for
non-English audio) costs more than transcription. The planner is honest about
this — the Report tab shows the prediction alongside the actual time, and it
warns when even the fastest configuration is expected to overrun.

Three changes did most of the work: deleting a per-second frame extraction
whose output nothing ever read, running transcription and visual analysis
concurrently instead of one after the other, and enabling batching in the
summarizer (it was pinned to a batch size of 1 on CPU). A fourth — detecting
the spoken language instead of assuming English — turned out to matter as much
as any of them; see below.
## Language handling

The spoken language is detected from three 30-second probes taken from across
the audio, using a small multilingual model. Two details matter:

- **Not just the opening window.** Whisper's own auto-detection reads only the
  first segment, which on a lecture is often an intro jingle or sponsor read.
- **Not a pinned language.** This project previously hardcoded English. Given
  a Hindi-English lecture, Whisper was forced to decode Hindi as English and
  produced transliterated nonsense — and ran several times slower doing it,
  because every window fought the language model instead of agreeing with it.
  That garbage transcript then poisoned the summary, the chapters and the Q&A.

When the detected language is not English, the audio is **translated** to
English rather than transcribed verbatim, because every downstream model
(BART summarization, flan-t5 Q&A, MiniLM retrieval) is English-only — a
faithful Hindi transcript would be unusable by the rest of the pipeline. An
English-only `.en` checkpoint is swapped for its multilingual sibling
automatically, since `.en` models cannot read other languages at all.

Set `VS_WHISPER_LANGUAGE` in `.env` to force a language and skip detection.

## Prerequisites

1. **Python 3.10 or 3.11.** Several pinned ML dependencies (`torch==2.3.1`)
   lag behind on wheels for 3.12/3.13.
   ```bash
   python --version
   ```
2. **FFmpeg, on PATH.** Checked at startup; the app fails with a clear message
   if it is missing.
   - Windows: `winget install --id Gyan.FFmpeg -e`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

   ```bash
   ffmpeg -version
   ffprobe -version
   ```
3. A CPU-only laptop is enough. A GPU is auto-detected but never required.

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**On CPU-only machines**, install the CPU-only torch build first — the default
PyPI wheel bundles CUDA and is several GB:

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want an optional Gemini API key or a
different cache directory. Nothing in `.env` is required — the pipeline runs
fully offline on local models.

## First run

Models are downloaded from Hugging Face / Ultralytics / open_clip on first use
and cached under `~/.cache/`. Because the planner chooses models per video, a
first run can otherwise stall mid-pipeline downloading whichever checkpoint it
picked. Do it once, up front:

```bash
python warmup.py           # everything (~5 GB)
python warmup.py --check   # list what would be downloaded
python warmup.py --asr     # just the speech models
```

`warmup.py` retries transient download failures and resumes where it stopped.

Once everything is cached, set `VS_OFFLINE=1` in `.env`. Hugging Face checks
for model updates on every load, so a slow or absent network turns a fully-
cached run into minutes of retry loops before it falls back to the local copy.

The app also **preloads models in the background at startup** (see
[`core/warm.py`](core/warm.py)). Cold-loading everything costs about 70
seconds; paid inside a run it lands entirely on the first video and looks
like a hung progress bar. The sidebar shows warm-up progress.

### Run the web app

```bash
streamlit run app.py
```

Paste a URL or upload a file, pick a processing profile, and click
**Summarize video**. Results open across six tabs — Summary, Chapters,
Transcript, Q&A, Scenes, and Report — with Markdown, SRT, and JSON export.

### Run headlessly (CLI)

```bash
python run_pipeline.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"
python run_pipeline.py --file ./sample_lecture.mp4
python run_pipeline.py --url "..." --profile accurate
python run_pipeline.py --plan-only --url "..."
python run_pipeline.py --video-id <video_id> --ask "What is normalization?"
```

Profiles are `fast`, `balanced` (default), and `accurate`; `--budget` sets the
target in seconds directly. Re-running the same source is near-instant — every
stage checks its cached artifact under `cache/<video_id>/` first. Pass
`--force` to reprocess.

## Testing

```bash
pytest tests/
```

`tests/test_cache.py` and `tests/test_fuse.py` are pure unit tests. 
`tests/test_chunking.py` loads the real BART tokenizer (a few MB, no model
weights) to verify map-stage chunking on a synthetic 45-minute transcript.

## Evaluation

```bash
python eval/run_rouge.py --input eval/reference_summaries.json   # you provide this file
python eval/run_keyframe_f1.py --gt eval/ground_truth.json        # you provide this file
python eval/run_qa_eval.py --testset eval/qa_testset.json         # fill this in first
```

**Note:** SumMe/TVSum are visual-only benchmarks with little meaningful speech,
so they validate the keyframe module only. Transcript summarization is
evaluated separately on speech-heavy videos (TED / NPTEL) against
human-written reference summaries.

## Repository layout

See [`config.py`](config.py) for every tunable constant and
[`core/models.py`](core/models.py) for the JSON contracts each module
reads/writes under `cache/<video_id>/`.

```
core/
├── perf.py        # hardware profiling + the adaptive execution planner
├── ingest.py      # M1 — download/upload, probe, extract audio
├── transcribe.py  # M2 — YouTube captions or parallel faster-whisper ASR
├── vision.py      # M3 — segmentation, keyframes, YOLO, CLIP, OCR
├── fuse.py        # M4 — align transcript to scenes
├── summarize.py   # M5 — map-reduce summary + chapters
├── qa.py          # M6 — FAISS retrieval + flan-t5/Gemini generation
├── highlight.py   # M7 — stub, future scope
├── pipeline.py    # orchestrator (M1→M6) + UI façade
└── cache.py       # cache dir, job state, JSON read/write
ui/
├── theme.py       # design tokens and CSS
├── components.py  # reusable render functions
└── format.py      # formatting + Markdown/SRT/JSON export
```

## Known limitations

- **Answer quality is bounded by `flan-t5-base`**, the default offline Q&A
  model. It answers extractively — often lifting a clause from the transcript
  rather than composing a sentence. Retrieval and citations are sound; if
  retrieval finds nothing related, the system declines to answer rather than
  inventing one. Setting `GEMINI_API_KEY` in `.env` swaps in a much stronger
  generator at the cost of an external API call.
- **Long videos trade accuracy for time.** Past roughly 45 minutes the planner
  selects smaller ASR and summarization models to hold the budget, and
  summaries get noticeably blunter. Use the `accurate` profile (or raise
  `--budget`) when quality matters more than turnaround.
- **YouTube downloads can fail** (geo-block, private video, bot-check). Upload
  the file directly instead — that path is fully independent of `yt-dlp`. An
  old `yt-dlp` pin is the most common cause of "Please sign in" errors; run
  `pip install --upgrade yt-dlp` first. For age-restricted or region-locked
  videos, set `YT_DLP_COOKIES_FROM_BROWSER` in `.env`.
- **Highlight video generation (M7)** is out of scope for this prototype —
  see [`core/highlight.py`](core/highlight.py).
- Real-time/live processing, speaker diarization, emotion/action recognition,
  multi-user auth, and cloud deployment are explicitly out of scope (see the
  project abstract).
- Static, single-shot recordings produce very few content-based scene cuts;
  `core/vision.py` falls back to fixed 60-second intervals when fewer than 2
  scenes are found.
- **Windows only:** `torch` and `ctranslate2` each bundle their own OpenMP
  runtime, which crashes with `OMP Error #15` if both load in one process.
  `config.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import time — the standard
  fix, applied before either library is imported anywhere.
- Ultralytics downloads `yolov8n.pt` into the working directory (not
  `~/.cache/`) the first time `core.vision` runs; this is expected and
  gitignored.
- If the project lives in a OneDrive/Dropbox folder, the sync client can
  briefly lock cache files as they are written. `core/cache.py` retries around
  this, but pointing `VS_CACHE_DIR` at a non-synced path avoids it entirely.
