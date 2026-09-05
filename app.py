"""Streamlit UI. No ML logic here — everything goes through core.pipeline.

This file may only import from ``core.pipeline``, ``core.models``, ``config``
and the presentation-only ``ui`` package, keeping the option open to swap
Streamlit for another front end without touching the pipeline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import config
from core import pipeline as pl
from core.models import VideoSummarizerError, describe_exception
from ui import components as C
from ui import format as fmt
from ui.theme import CSS

# Each profile trades wall-clock against accuracy. The planner (core/perf.py)
# turns these into concrete model choices for the specific video.
# Friendly labels for the video-type dropdown, mapped to core.understand.VideoType.
VIDEO_TYPE_LABELS = {
    "Lecture / tutorial": "lecture",
    "Coding / algorithms": "coding",
    "Meeting recording": "meeting",
    "Interview / podcast": "interview",
    "Conference talk": "talk",
    "News segment": "news",
    "Documentary / explainer": "documentary",
    "Other": "other",
}

SPEED_PROFILES = {
    "Fast": {
        "budget": 150.0, "quality_floor": False,
        "help": "Smallest models. Best for a quick look at a long video.",
    },
    "Balanced": {
        "budget": config.TARGET_PIPELINE_SEC, "quality_floor": False,
        "help": f"Targets {config.TARGET_PIPELINE_SEC / 60:.0f} minutes for any video length.",
    },
    "Accurate": {
        "budget": None, "quality_floor": True,
        "help": "Largest models, full analysis. No time limit — can take much longer.",
    },
}


st.set_page_config(
    page_title=config.PROJECT_TITLE,
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _warm_models() -> bool:
    """Start loading models the moment the app does, once per process.

    Cold-loading every model costs about 70 seconds, and paid inside a run
    it lands entirely on the first video the user processes — as a progress
    bar that appears to hang. Started here it overlaps the time the user
    spends pasting a URL and waiting for the download.
    """
    pl.start_warmup()
    return True


_warm_models()


# --- Session state -------------------------------------------------------------

_DEFAULTS = {
    "active_video_id": None,
    "sources": {},        # video_id -> {"source_ref", "is_upload"}
    "qa_history": {},     # video_id -> [{"question", "answer"}]
    "seek_to": 0,         # player start position, in seconds
    "profile": "Balanced",
    "video_type": "Lecture / tutorial",
    "topic": "",
    "pending_error": None,
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)


def _is_streamlit_control_flow(exc: BaseException) -> bool:
    """True for the exceptions Streamlit raises to control the script itself.

    ``st.rerun()`` and Streamlit's file watcher both work by raising. A broad
    ``except Exception`` around a long-running call swallows those, which both
    breaks the rerun and reports it to the user as a pipeline failure — with
    an empty reason, since these carry no message.
    """
    return type(exc).__module__.startswith("streamlit.")


def _save_upload(uploaded_file) -> str:
    destination = Path(tempfile.gettempdir()) / "video_summarizer_uploads"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


def _source_for(video_id: str) -> dict:
    """Where a video came from — from this session, else from its cached meta.

    Falling back to meta keeps deep links and the player working for videos
    processed in an earlier session, which the sidebar library can select.
    """
    known = st.session_state.sources.get(video_id)
    if known:
        return known
    meta = pl.get_meta(video_id) or {}
    return {
        "source_ref": meta.get("source_ref", ""),
        "is_upload": meta.get("source") == "upload",
    }


# --- Sidebar --------------------------------------------------------------------

with st.sidebar:
    st.markdown('<p class="vs-sidebar-label">Library</p>', unsafe_allow_html=True)

    cached_ids = pl.list_cached_videos()
    if not cached_ids:
        st.caption("Nothing processed yet.")
    for video_id in cached_ids[:12]:
        meta = pl.get_meta(video_id)
        if not meta:
            continue
        is_active = video_id == st.session_state.active_video_id
        label = fmt.truncate(meta.get("title", video_id), 34)
        if st.button(
            f"{'●' if is_active else '○'}  {label}",
            key=f"lib_{video_id}", use_container_width=True,
        ):
            st.session_state.active_video_id = video_id
            st.session_state.seek_to = 0
            st.rerun()

    st.divider()
    st.markdown('<p class="vs-sidebar-label">Processing profile</p>', unsafe_allow_html=True)
    st.session_state.profile = st.radio(
        "Profile", list(SPEED_PROFILES), label_visibility="collapsed",
        index=list(SPEED_PROFILES).index(st.session_state.profile),
    )
    st.caption(SPEED_PROFILES[st.session_state.profile]["help"])

    config.RAG_TOP_K = st.slider(
        "Q&A passages retrieved", min_value=1, max_value=10, value=config.RAG_TOP_K,
        help="How much transcript context each answer is grounded in.",
    )

    if st.session_state.active_video_id:
        st.divider()
        if st.button("Delete this video's cache", use_container_width=True):
            victim = st.session_state.active_video_id
            pl.clear_cache(victim)
            st.session_state.qa_history.pop(victim, None)
            st.session_state.sources.pop(victim, None)
            st.session_state.active_video_id = None
            st.rerun()

    st.divider()
    hardware = pl.get_hardware_profile()
    st.markdown('<p class="vs-sidebar-label">System</p>', unsafe_allow_html=True)
    st.caption(
        f"{hardware['cpu_count']} cores · "
        f"{'GPU' if hardware['has_cuda'] else 'CPU only'} · "
        f"{hardware['ram_gb']:.0f} GB RAM"
    )
    warm = pl.warmup_status()
    if warm["started"] and not warm["done"]:
        st.caption(
            f"Warming models… {len(warm['loaded'])}/{warm['total']} ready"
        )
    elif warm["failed"]:
        st.caption(f"{len(warm['failed'])} model(s) will load on first use")

    st.caption(f"{config.TEAM_NAMES} · Guide: {config.GUIDE_NAME}")


# --- Masthead --------------------------------------------------------------------

C.masthead(
    config.PROJECT_TITLE,
    "Multimodal video summarization with timestamped question answering",
    C.pill(f"{st.session_state.profile} profile", "accent"),
)


# --- Input ----------------------------------------------------------------------

if not st.session_state.active_video_id:
    C.hero()

with st.container(border=True):
    tab_url, tab_upload = st.tabs(["  Video link  ", "  Upload a file  "])
    with tab_url:
        url_input = st.text_input(
            "Video URL", placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed", key="url_input",
        )
        url_clicked = st.button("Summarize video", type="primary", key="go_url")
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a video", type=["mp4", "mov", "mkv", "webm", "avi"],
            label_visibility="collapsed",
        )
        upload_clicked = st.button("Summarize video", type="primary", key="go_upload")

    force = st.checkbox(
        "Reprocess from scratch", value=False,
        help="Ignore anything already cached for this video.",
    )


# --- Run the pipeline -------------------------------------------------------------

submitted_url = url_clicked and url_input.strip()
submitted_upload = upload_clicked and uploaded is not None

if url_clicked and not url_input.strip():
    st.warning("Paste a video URL first.")
if upload_clicked and uploaded is None:
    st.warning("Choose a video file first.")

if submitted_url or submitted_upload:
    is_upload = bool(submitted_upload)
    source_ref = _save_upload(uploaded) if is_upload else url_input.strip()
    profile = SPEED_PROFILES[st.session_state.profile]

    # The cache key is derived from the source, so the run's state is
    # addressable before process() returns.
    cache_id = pl.video_id_for(source_ref)

    with st.container(border=True):
        C.section("Processing")
        progress_bar = st.progress(0.0)
        status_slot = st.empty()
        tracker_slot = st.empty()
        seen_timings: dict[str, float] = {}

        def on_progress(stage: str, progress: float, message: str) -> None:
            progress_bar.progress(min(max(progress, 0.0), 1.0))
            status_slot.markdown(f'<p class="vs-dim">{message}</p>', unsafe_allow_html=True)
            seen_timings.update(pl.get_state(cache_id).get("timings") or {})
            with tracker_slot.container():
                C.stage_tracker(stage, progress, seen_timings)

        try:
            if profile["budget"] is not None:
                config.TARGET_PIPELINE_SEC = profile["budget"]

            result = pl.process(
                source_ref, is_upload=is_upload, force=force,
                progress_cb=on_progress, quality_floor=profile["quality_floor"],
            )
            st.session_state.active_video_id = result["video_id"]
            st.session_state.sources[result["video_id"]] = {
                "source_ref": source_ref, "is_upload": is_upload,
            }
            st.session_state.seek_to = 0
            st.session_state.pending_error = None
            st.rerun()
        except VideoSummarizerError as exc:
            st.session_state.pending_error = describe_exception(exc)
        except Exception as exc:  # never let a raw traceback reach the user
            if _is_streamlit_control_flow(exc):
                raise
            st.session_state.pending_error = (
                f"Unexpected error: {describe_exception(exc)}"
            )

if st.session_state.pending_error:
    st.error(st.session_state.pending_error)


# --- Results ---------------------------------------------------------------------

video_id = st.session_state.active_video_id
if not video_id:
    st.stop()

meta = pl.get_meta(video_id)
summary = pl.get_summary(video_id)
if meta is None or summary is None:
    st.warning("This video's cache is incomplete. Reprocess it to rebuild the results.")
    st.stop()

scenes_data = pl.get_scenes(video_id)
transcript = pl.get_transcript(video_id)
understanding = pl.get_understanding(video_id)
state = pl.get_state(video_id)
source = _source_for(video_id)

st.markdown(
    f'<div class="vs-result-head"><h2>{fmt.html.escape(meta["title"])}</h2>'
    f'<div class="vs-result-badges">{C.source_badges(meta, summary, state)}</div></div>',
    unsafe_allow_html=True,
)

player_col, side_col = st.columns([7, 5], gap="large")
with player_col:
    C.video_player(
        source["source_ref"], source["is_upload"],
        str(config.CACHE_ROOT / video_id / "video.mp4"),
        st.session_state.seek_to,
    )
with side_col:
    scene_count = len(scenes_data["scenes"]) if scenes_data else 0
    word_count = len((transcript or {}).get("full_text", "").split())
    C.stat_tiles([
        (str(len(summary.get("chapters", []))), "Chapters"),
        (str(scene_count), "Scenes"),
        (f"{word_count:,}", "Words"),
        (fmt.human_duration(meta.get("duration_sec", 0)), "Length"),
    ])
    st.write("")
    export_a, export_b, export_c = st.columns(3)
    with export_a:
        st.download_button(
            "Markdown", fmt.to_markdown(meta, summary),
            file_name=f"{fmt.slugify(meta['title'])}-summary.md",
            mime="text/markdown", use_container_width=True,
        )
    with export_b:
        st.download_button(
            "Subtitles", fmt.to_srt(transcript or {"segments": []}),
            file_name=f"{fmt.slugify(meta['title'])}.srt",
            mime="text/plain", use_container_width=True,
            disabled=not transcript,
        )
    with export_c:
        st.download_button(
            "JSON", fmt.to_json_export(meta, summary, scenes_data, transcript),
            file_name=f"{fmt.slugify(meta['title'])}.json",
            mime="application/json", use_container_width=True,
        )

st.write("")
tabs = st.tabs(["  Summary  ", "  Chapters  ", "  Transcript  ", "  Q&A  ", "  Scenes  ", "  Report  "])

# --- Summary -----------------------------------------------------------------
with tabs[0]:
    if summary.get("warning"):
        st.info(summary["warning"].replace("_", " ").capitalize())

    if understanding and understanding.get("explanation"):
        C.understanding_view(understanding, source["source_ref"], source["is_upload"])
    else:
        sections = [s for s in (summary.get("chunk_summaries") or []) if s.get("text")]
        overall = summary.get("overall_summary") or ""
        if sections:
            C.section(
                "Detailed walkthrough",
                f"{len(sections)} sections · each links to where it starts in the video",
            )
            for i, section in enumerate(sections, start=1):
                st.markdown(
                    f'<div class="vs-walk">'
                    f'<div class="vs-walk-head">'
                    f'<span class="vs-walk-num">{i}</span>'
                    f'{C.timestamp_link(section["start"], source["source_ref"], source["is_upload"])}'
                    f'</div>'
                    f'<p class="vs-walk-text">{fmt.html.escape(section["text"])}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if overall:
                st.write("")
                C.section("In summary", "The key points, in brief")
                st.markdown(f'<div class="vs-keypoints">{fmt.html.escape(overall)}</div>',
                            unsafe_allow_html=True)
        else:
            C.section("Summary")
            st.write(overall or "_No summary was produced._")
# --- Chapters ------------------------------------------------------------------
with tabs[1]:
    chapters = summary.get("chapters") or []
    if not chapters:
        C.empty_state("⧗", "No chapters", "This video was too short to divide into chapters.")
    else:
        C.chapter_list(chapters, source["source_ref"], source["is_upload"])
        if source["is_upload"]:
            st.write("")
            labels = [f"{fmt.hhmmss(c['start'])} — {c['title']}" for c in chapters]
            choice = st.selectbox("Jump the player to a chapter", labels, index=None)
            if choice is not None:
                target = int(chapters[labels.index(choice)]["start"])
                if target != st.session_state.seek_to:
                    st.session_state.seek_to = target
                    st.rerun()

# --- Transcript ----------------------------------------------------------------
with tabs[2]:
    if not transcript:
        C.empty_state("⧉", "No transcript", "Reprocess this video to generate one.")
    else:
        segments = transcript.get("segments") or []
        query = st.text_input(
            "Search the transcript", placeholder="Search for a word or phrase...",
            label_visibility="collapsed", key="transcript_search",
        )
        matched = C.transcript_view(segments, query, source["source_ref"], source["is_upload"])
        caption = f"{matched:,} of {len(segments):,} lines"
        if matched > 400:
            caption += " — showing the first 400"
        st.caption(caption)

# --- Q&A -----------------------------------------------------------------------
with tabs[3]:
    history = st.session_state.qa_history.setdefault(video_id, [])
    if not history:
        C.empty_state(
            "◇", "Ask about this video",
            "Every answer cites the timestamps it came from.",
        )

    for entry in history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            answer = entry["answer"]
            confidence = answer.get("confidence", "high")
            st.markdown(
                C.pill(
                    f"{confidence} confidence",
                    "success" if confidence == "high" else "warning",
                    dot=True,
                ),
                unsafe_allow_html=True,
            )
            st.write(answer["answer"])
            sources = answer.get("sources") or []
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    rows = "".join(
                        f'<div class="vs-source">'
                        f'{C.timestamp_link(s["start"], source["source_ref"], source["is_upload"])}'
                        f'<div>{fmt.html.escape(fmt.truncate(s["text"], 320))}</div></div>'
                        for s in sources
                    )
                    st.markdown(rows, unsafe_allow_html=True)

    question = st.chat_input("Ask a question about this video...")
    if question:
        with st.spinner("Searching the transcript..."):
            try:
                answer = pl.answer_question(video_id, question)
            except VideoSummarizerError as exc:
                answer = {
                    "answer": f"Couldn't answer that: {describe_exception(exc)}",
                    "sources": [], "confidence": "low",
                }
            except Exception as exc:  # never let a raw traceback reach the user
                if _is_streamlit_control_flow(exc):
                    raise
                answer = {
                    "answer": f"Unexpected error: {describe_exception(exc)}",
                    "sources": [], "confidence": "low",
                }
        history.append({"question": question, "answer": answer})
        st.rerun()

# --- Scenes --------------------------------------------------------------------
with tabs[4]:
    scenes = (scenes_data or {}).get("scenes") or []
    if not scenes:
        C.empty_state("▦", "No scenes detected", "Reprocess this video to run visual analysis.")
    else:
        C.section(
            "Detected scenes",
            f"{len(scenes)} keyframes, each linked to its moment in the video",
        )
        # One CSS grid rather than st.columns: Streamlit columns stack
        # independently, so tiles of differing height leave the rows ragged.
        tiles = []
        for scene in scenes:
            encoded = C.encode_image(Path(scene["keyframe_path"]))
            if not encoded:
                continue
            tags = "".join(
                f'<span class="vs-tag">{fmt.html.escape(o["label"])}</span>'
                for o in scene.get("objects", [])[:3]
            )
            tiles.append(C.scene_card(
                encoded,
                C.timestamp_link(scene["start"], source["source_ref"], source["is_upload"]),
                tags,
                (scene.get("ocr_text") or "").strip(),
            ))
        st.markdown(
            f'<div class="vs-scene-grid">{"".join(tiles)}</div>',
            unsafe_allow_html=True,
        )
# --- Report --------------------------------------------------------------------
with tabs[5]:
    timings = state.get("timings") or {}
    plan = (state.get("plan") or meta.get("plan") or {})
    duration = meta.get("duration_sec", 0)
    total = timings.get("total")

    C.section("How this video was processed")
    if total and duration:
        C.stat_tiles([
            (fmt.human_duration(total), "Total time"),
            (f"{duration / total:.0f}×", "Faster than watching"),
            (f"{total / duration:.2f}", "Sec per video-sec"),
            (fmt.human_duration(duration), "Video length"),
        ])
        st.write("")

    if timings:
        C.section("Time per stage")
        stage_rows = [(label, timings.get(key)) for key, label in C.PIPELINE_STAGES]
        longest = max((v for _, v in stage_rows if v), default=1.0)
        for label, value in stage_rows:
            if value is None:
                continue
            st.markdown(f"**{label}** — {value:.1f}s")
            st.progress(min(value / longest, 1.0))

    if plan:
        st.write("")
        C.section("Configuration chosen for this video")
        st.caption(
            "The pipeline picks models to fit a wall-clock budget, so a long "
            "video automatically runs lighter analysis than a short one."
        )
        st.markdown(
            "".join(C.pill(text) for text in [
                f"ASR: {plan.get('asr_model', '?')}",
                f"beam {plan.get('asr_beam_size', '?')}",
                f"{plan.get('asr_workers', '?')} parallel workers",
                f"Summarizer: {str(plan.get('summarizer_model', '?')).split('/')[-1]}",
                f"≤{plan.get('max_keyframes', '?')} keyframes",
                f"OCR {'on' if plan.get('ocr_enabled') else 'off'}",
                f"{plan.get('download_height', '?')}p download",
            ]),
            unsafe_allow_html=True,
        )
        for note in plan.get("notes") or []:
            st.info(note)
