# AI-Based Video Summarization System

**VIT Vellore — Project-I (BCSE497J)**
Jaivardhan Singh (23BCI0036) · Sujay Chand (23BCE0217) · Harshit Raj (23BCE0190)
Guide: Dr. Deepa K, SCOPE

Given a video (YouTube URL or local upload), this system produces:
- a timestamped, abstractive textual summary,
- auto-generated chapters, and
- a natural-language Q&A interface that answers with timestamp citations.

Every transcript segment, scene, chapter, and Q&A answer carries `start`/`end`
seconds — timestamps are the backbone of the whole pipeline.

## Pipeline

```
video (URL/upload)
      │
      ▼
  M1 ingest  ──────────────► audio.wav + sampled frames
      │
      ├──────────────┐
      ▼              ▼
M2 transcribe    M3 vision (scenes, YOLO objects, CLIP embeddings)
      │              │
      └──────┬───────┘
             ▼
        M4 fuse (align transcript to scenes, timestamped blocks)
             │
             ▼
     M5 summarize (map-reduce BART, chapters)
             │
             ▼
      M6 index (FAISS over transcript chunks) ──► RAG Q&A
```

M7 (auto-generated highlight video) is a stub only — see [`core/highlight.py`](core/highlight.py)
and *Known limitations* below.

## Prerequisites

1. **Python 3.10 or 3.11.** Newer versions (3.12/3.13) are not used here because
   several pinned ML dependencies (`torch==2.3.1`, etc.) lag behind on wheel
   availability. Check with:
   ```bash
   python --version
   ```
2. **FFmpeg, installed system-wide and on PATH.** The app checks for it at
   startup and fails with a clear message if missing.
   - Windows: `winget install --id Gyan.FFmpeg -e`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (or your distro's package manager)

   Verify with:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```
3. A CPU-only laptop is sufficient for videos up to ~15 minutes. A GPU is a
   bonus (auto-detected via CUDA) but never required.

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

**Smaller/faster install on CPU-only machines:** the default `torch` wheel on
PyPI bundles CUDA and is several GB. Install the CPU-only build instead
*before* running `pip install -r requirements.txt` (pip will then see it's
already satisfied):

```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you want to set an optional Gemini API key
or override the cache directory — nothing in `.env` is required for the
core pipeline, which runs fully offline with local models.

```bash
cp .env.example .env   # optional
```

## First run

**All models are downloaded from Hugging Face / Ultralytics / open_clip on
first use and cached under `~/.cache/`.** Expect a one-time download of
roughly **3–4 GB** (Whisper `small`, YOLOv8n, CLIP ViT-B/32, BART-large-CNN,
flan-t5-base, all-MiniLM-L6-v2). Make sure you have a stable connection and
enough disk space before the first run.

To pre-warm every model without processing a video:

```bash
python -c "
from faster_whisper import WhisperModel; WhisperModel('small', compute_type='int8')
from ultralytics import YOLO; YOLO('yolov8n.pt')
import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
from transformers import pipeline
pipeline('summarization', model='facebook/bart-large-cnn')
pipeline('text2text-generation', model='google/flan-t5-base')
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('All models cached.')
"
```

### Run headlessly (CLI) — recommended for first verification

```bash
python run_pipeline.py --url "https://www.youtube.com/watch?v=XXXXXXXXXXX"
python run_pipeline.py --file ./sample_lecture.mp4
python run_pipeline.py --video-id <video_id> --ask "What is normalization?"
```

Re-running the same source is fast (cache hit, <5s) because every stage
checks its own cached artifact under `cache/<video_id>/` before doing any
work; pass `--force` to ignore the cache and reprocess from scratch.

### Run the web app

```bash
streamlit run app.py
```

Paste a YouTube URL or upload a file, click **Process**, then explore the
**Summary**, **Chapters**, **Q&A**, and **Scenes** tabs. The sidebar lets you
change the Whisper model size / Q&A top-k and clear the cache for a video.

## Testing

```bash
pytest tests/
```

`tests/test_cache.py` and `tests/test_fuse.py` are pure unit tests (no ML
dependencies). `tests/test_chunking.py` loads the real BART tokenizer (a
few MB, no model weights) to verify map-stage chunking is correct on a
synthetic 45-minute transcript.

## Evaluation

```bash
python eval/run_rouge.py --input eval/reference_summaries.json   # you provide this file
python eval/run_keyframe_f1.py --gt eval/ground_truth.json        # you provide this file
python eval/run_qa_eval.py --testset eval/qa_testset.json         # fill in eval/qa_testset.json first
```

`eval/qa_testset.json` ships as a 50-entry template — fill in `video_id`,
`question`, `expected_answer`, and `expected_timestamp_range` by hand after
processing a few videos.

**Note:** SumMe/TVSum are visual-only benchmarks with little meaningful
speech, so they validate the keyframe module only (`run_keyframe_f1.py`).
Transcript summarization is evaluated separately, on speech-heavy videos
(TED talks / NPTEL lectures) against human-written reference summaries.

## Repository layout

See [`config.py`](config.py) for every tunable constant and
[`core/models.py`](core/models.py) for the JSON data contracts each module
reads/writes under `cache/<video_id>/`.

```
core/
├── ingest.py      # M1 — download/upload, probe, extract audio + frames
├── transcribe.py  # M2 — YouTube captions or faster-whisper ASR
├── vision.py      # M3 — scene detection, YOLO objects, CLIP embeddings
├── fuse.py        # M4 — align transcript to scenes
├── summarize.py   # M5 — map-reduce BART summary + chapters
├── qa.py          # M6 — FAISS retrieval + flan-t5/Gemini generation
├── highlight.py   # M7 — stub, future scope
├── pipeline.py    # orchestrator (M1→M6) + UI façade functions
└── cache.py       # cache dir, job state, JSON read/write
```

## Known limitations

- **YouTube downloads can fail** (geo-block, private video, bot-check).
  If `yt-dlp` fails, upload the file directly instead — the upload path is
  fully independent of `yt-dlp`. YouTube and yt-dlp are in a constant
  cat-and-mouse fight over bot-detection, so an old `yt-dlp` pin is the most
  common cause of a "Please sign in" / "confirm you're not a bot" error —
  run `pip install --upgrade yt-dlp` first. If a specific video still fails
  (age-restricted or region-locked), set `YT_DLP_COOKIES_FROM_BROWSER` in
  `.env` to a browser you're logged into YouTube with (see `.env.example`).
- **flan-t5-base** (the default, fully-offline Q&A backend) is a small model
  and doesn't always perfectly follow the "say if it's not discussed"
  instruction in the RAG prompt. Setting `GEMINI_API_KEY` in `.env` swaps in
  Gemini for noticeably better instruction-following, at the cost of an
  external API call.
- **Highlight video generation (M7)** is out of scope for this prototype —
  see [`core/highlight.py`](core/highlight.py), which stubs the intended
  signature and raises `NotImplementedError`.
- Real-time/live processing, speaker diarization, emotion/action
  recognition, multi-user auth, and cloud deployment are explicitly out of
  scope for this academic prototype (see the project abstract).
- Static, single-shot lecture recordings often produce very few content-based
  scene cuts; `core/vision.py` automatically falls back to fixed 60-second
  intervals when PySceneDetect finds fewer than 2 scenes.
- **Windows only:** `torch` and `ctranslate2` (used by faster-whisper) each
  bundle their own OpenMP runtime, which crashes with `OMP Error #15` if both
  load in one process. `config.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` at import
  time to work around this — it's the standard, widely-used fix for this
  specific conflict, applied before either library is imported anywhere.
- Ultralytics downloads `yolov8n.pt` into the current working directory (not
  `~/.cache/`) the first time `core.vision` runs; this is expected and is
  gitignored (`*.pt`).
