"""Streamlit UI. No ML logic here — everything goes through core.pipeline.

Per BUILD_PLAN.md section 2, this file may only import from core.pipeline
and core.models, keeping the option open to swap Streamlit for FastAPI
later without touching the pipeline.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

import config
from core import pipeline
from core.models import VideoSummarizerError

WHISPER_SIZES = ["tiny", "base", "small", "medium"]


# --- Helpers -------------------------------------------------------------------------


def _format_hhmmss(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:60] or "video"


def _youtube_deep_link(url: str, start_sec: int) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}t={start_sec}s"


def _build_markdown_export(meta: dict, summary: dict) -> str:
    lines = [f"# {meta['title']}", "", "## Overall Summary", "", summary["overall_summary"], ""]
    if summary.get("warning"):
        lines.insert(2, f"> Note: {summary['warning'].replace('_', ' ')}")
    lines.append("## Chapters")
    lines.append("")
    for ch in summary["chapters"]:
        lines.append(f"- **[{_format_hhmmss(ch['start'])}] {ch['title']}** — {ch['summary']}")
    return "\n".join(lines)


def _save_uploaded_file(uploaded_file) -> str:
    tmp_dir = Path(tempfile.gettempdir()) / "video_summarizer_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getvalue())
    return str(dest)


@st.cache_resource(show_spinner=False)
def _pipeline_module():
    """Import core.pipeline once per process.

    All ML models it transitively uses are memoized as lazy singletons
    inside core/* modules already, so this just pins that warm state
    across Streamlit reruns at the app layer too.
    """
    return pipeline


# --- Page setup ------------------------------------------------------------------------

st.set_page_config(page_title=config.PROJECT_TITLE, page_icon="▶", layout="wide")

st.markdown(
    """
    <style>
    /* Hide Streamlit's default chrome (deploy button, menu, footer) so the
       app reads as a standalone product rather than a hosted demo. */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
    div[data-testid="stDecoration"] { display: none; }

    html, body, [class*="css"] { font-family: "Inter", -apple-system, "Segoe UI", sans-serif; }

    .block-container {
        max-width: 1080px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Hero header */
    .vs-hero { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.15rem; }
    .vs-hero-badge {
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.3rem; flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .vs-hero-title { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; margin: 0; line-height: 1.2; }
    .vs-hero-sub { color: #9aa3b2; font-size: 0.98rem; margin: 0.35rem 0 1.6rem 0; }

    /* Card-ish native containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border-color: rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.015);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        border: none;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #7c3aed);
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.35);
    }
    .stButton > button[kind="primary"]:hover { opacity: 0.92; }

    /* Tabs -> pill segmented control */
    button[data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600;
        padding: 0.4rem 1rem !important;
    }
    div[data-baseweb="tab-highlight"] { display: none; }
    div[data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.04);
        padding: 4px;
        border-radius: 10px;
        width: fit-content;
    }
    button[aria-selected="true"][data-baseweb="tab"] {
        background: rgba(99, 102, 241, 0.18);
        color: #c7c9fb !important;
    }

    /* Confidence badges in Q&A */
    .vs-badge { display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em; }
    .vs-badge-high { background: rgba(34,197,94,0.15); color: #4ade80; }
    .vs-badge-low { background: rgba(245,158,11,0.15); color: #fbbf24; }

    /* Chapter cards */
    .vs-chapter-time { color: #a5b4fc; font-weight: 700; font-size: 0.85rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

if "active_video_id" not in st.session_state:
    st.session_state.active_video_id = None
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = {}  # video_id -> {"source_ref", "is_upload"}
if "qa_history" not in st.session_state:
    st.session_state.qa_history = {}  # video_id -> [{"question", "answer"}]

pl = _pipeline_module()


# --- Header ------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="vs-hero">
        <div class="vs-hero-badge">▶</div>
        <p class="vs-hero-title">{config.PROJECT_TITLE}</p>
    </div>
    <p class="vs-hero-sub">Turn any video into a searchable summary — chapters, keyframes,
    and a Q&amp;A assistant that cites the exact timestamp.</p>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar -------------------------------------------------------------------------

with st.sidebar:
    st.markdown("#### Settings")
    config.WHISPER_MODEL_SIZE = st.selectbox(
        "Whisper model size", WHISPER_SIZES, index=WHISPER_SIZES.index(config.WHISPER_MODEL_SIZE),
        help="Larger = more accurate but slower. Applies to the next video processed.",
    )
    config.RAG_TOP_K = st.slider(
        "Q&A: chunks retrieved (top-k)", min_value=1, max_value=10, value=config.RAG_TOP_K,
    )

    st.divider()
    if st.session_state.active_video_id:
        st.markdown("#### This video")
        if st.button("Clear cache for this video", use_container_width=True):
            pl.clear_cache(st.session_state.active_video_id)
            st.session_state.qa_history.pop(st.session_state.active_video_id, None)
            st.session_state.processed_sources.pop(st.session_state.active_video_id, None)
            st.session_state.active_video_id = None
            st.rerun()
        st.divider()

    st.markdown("#### Recently processed")
    recent_ids = pl.list_cached_videos()[:10]
    if not recent_ids:
        st.caption("Nothing processed yet.")
    for vid in recent_ids:
        vid_meta = pl.get_meta(vid)
        label = vid_meta["title"] if vid_meta else vid
        is_active = vid == st.session_state.active_video_id
        if st.button(
            f"{'●' if is_active else '○'}  {label}",
            key=f"recent_{vid}", use_container_width=True,
        ):
            st.session_state.active_video_id = vid
            st.rerun()


# --- Input row -----------------------------------------------------------------------

with st.container(border=True):
    input_mode = st.radio("Input source", ["YouTube URL", "Upload file"], horizontal=True)

    url_input, uploaded_file = None, None
    if input_mode == "YouTube URL":
        url_input = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a video file", type=["mp4", "mov", "mkv", "webm", "avi"],
            label_visibility="collapsed",
        )

    col_force, col_btn = st.columns([3, 1])
    with col_force:
        force_reprocess = st.checkbox("Force reprocess (ignore cache)", value=False)
    with col_btn:
        process_clicked = st.button("Process", type="primary", use_container_width=True)


# --- Progress area ---------------------------------------------------------------------

if process_clicked:
    is_upload = input_mode == "Upload file"
    if is_upload:
        if uploaded_file is None:
            st.warning("Please upload a video file first.")
            st.stop()
        source_ref = _save_uploaded_file(uploaded_file)
    else:
        if not url_input:
            st.warning("Please enter a YouTube URL first.")
            st.stop()
        source_ref = url_input

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    def _progress_cb(stage: str, progress: float, message: str) -> None:
        progress_bar.progress(min(max(progress, 0.0), 1.0))
        status_placeholder.info(f"**{stage}** — {message}")

    try:
        result = pl.process(
            source_ref, is_upload=is_upload, force=force_reprocess, progress_cb=_progress_cb
        )
        st.session_state.active_video_id = result["video_id"]
        st.session_state.processed_sources[result["video_id"]] = {
            "source_ref": source_ref, "is_upload": is_upload,
        }
        status_placeholder.success("Processing complete.")
    except VideoSummarizerError as exc:
        status_placeholder.error(f"Processing failed: {exc}")
    except Exception as exc:  # never let a raw traceback reach the user
        status_placeholder.error(f"Unexpected error: {exc}")


# --- Results ---------------------------------------------------------------------------

video_id = st.session_state.active_video_id

if video_id:
    meta = pl.get_meta(video_id)
    summary = pl.get_summary(video_id)
    scenes_data = pl.get_scenes(video_id)
    source_info = st.session_state.processed_sources.get(video_id, {})

    if meta is None or summary is None:
        st.warning("This video's cache looks incomplete. Try reprocessing it.")
    else:
        tab_summary, tab_chapters, tab_qa, tab_scenes = st.tabs(
            ["Summary", "Chapters", "Q&A", "Scenes"]
        )

        with tab_summary:
            st.subheader(meta["title"])
            if summary.get("warning"):
                st.warning(summary["warning"].replace("_", " ").capitalize())
            st.write(summary["overall_summary"])

            st.download_button(
                "Download as Markdown",
                data=_build_markdown_export(meta, summary),
                file_name=f"{_slugify(meta['title'])}_summary.md",
                mime="text/markdown",
            )

        with tab_chapters:
            for ch in summary["chapters"]:
                ts = _format_hhmmss(ch["start"])
                with st.container(border=True):
                    col_ts, col_body = st.columns([1, 5])
                    with col_ts:
                        if not source_info.get("is_upload") and source_info.get("source_ref"):
                            link = _youtube_deep_link(source_info["source_ref"], int(ch["start"]))
                            st.markdown(f'<a class="vs-chapter-time" href="{link}">{ts}</a>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<span class="vs-chapter-time">{ts}</span>', unsafe_allow_html=True)
                    with col_body:
                        st.markdown(f"**{ch['title']}**")
                        st.caption(ch["summary"])
                    if source_info.get("is_upload"):
                        with st.expander(f"Play from {ts}"):
                            video_path = config.CACHE_ROOT / video_id / "video.mp4"
                            st.video(str(video_path), start_time=int(ch["start"]))

        with tab_qa:
            qa_history = st.session_state.qa_history.setdefault(video_id, [])

            for entry in qa_history:
                with st.chat_message("user"):
                    st.write(entry["question"])
                with st.chat_message("assistant"):
                    confidence = entry["answer"].get("confidence", "high")
                    st.markdown(
                        f'<span class="vs-badge vs-badge-{confidence}">{confidence} confidence</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(entry["answer"]["answer"])
                    sources = entry["answer"]["sources"]
                    with st.expander(f"Sources ({len(sources)})"):
                        if not sources:
                            st.caption("No sources retrieved.")
                        for src in sources:
                            ts = _format_hhmmss(src["start"])
                            if not source_info.get("is_upload") and source_info.get("source_ref"):
                                link = _youtube_deep_link(source_info["source_ref"], int(src["start"]))
                                st.markdown(f"[`{ts}`]({link}) — {src['text']}")
                            else:
                                st.markdown(f"`{ts}` — {src['text']}")

            question = st.chat_input("Ask a question about this video...")
            if question:
                with st.spinner("Thinking..."):
                    try:
                        answer_dict = pl.answer_question(video_id, question)
                    except VideoSummarizerError as exc:
                        answer_dict = {"answer": f"Couldn't answer that: {exc}", "sources": [], "confidence": "low"}
                    except Exception as exc:  # never let a raw traceback reach the user
                        answer_dict = {"answer": f"Unexpected error: {exc}", "sources": [], "confidence": "low"}
                qa_history.append({"question": question, "answer": answer_dict})
                st.rerun()

        with tab_scenes:
            scenes = scenes_data["scenes"] if scenes_data else []
            if not scenes:
                st.info("No scenes detected.")
            else:
                cols = st.columns(4)
                for i, scene in enumerate(scenes):
                    with cols[i % 4]:
                        with st.container(border=True):
                            st.image(scene["keyframe_path"], use_column_width=True)
                            ts = _format_hhmmss(scene["start"])
                            tags = ", ".join(o["label"] for o in scene.get("objects", []))
                            st.markdown(f'<span class="vs-chapter-time">{ts}</span>', unsafe_allow_html=True)
                            st.caption(tags or "No objects detected")
else:
    st.info("Paste a YouTube URL or upload a video, then click Process to get started.")
