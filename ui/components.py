"""Reusable UI building blocks.

Each function renders one piece of the interface from already-computed data.
Keeping them here — rather than inline in ``app.py`` — is what lets the page
files stay short enough to read as layout rather than as markup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import streamlit as st

from ui import format as fmt
from ui.theme import logo_svg

# Stages in the order they run, with the label shown in the tracker.
PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("ingest", "Fetch & prepare video"),
    ("analyze", "Transcribe audio + analyze visuals"),
    ("fuse", "Align transcript with scenes"),
    ("summarize", "Summarize & build chapters"),
    ("index", "Build Q&A index"),
)


def masthead(title: str, subtitle: str, right_html: str = "") -> None:
    st.markdown(
        f"""
        <div class="vs-masthead">
          <div class="vs-brand">
            <div class="vs-logo">{logo_svg()}</div>
            <div class="vs-brand-text">
              <p class="vs-brand-name">{title}</p>
              <p class="vs-brand-sub">{subtitle}</p>
            </div>
          </div>
          <div class="vs-masthead-meta">{right_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, variant: str = "", dot: bool = False) -> str:
    """A small status chip. Returns HTML so callers can group several inline."""
    classes = f"vs-pill vs-pill-{variant}" if variant else "vs-pill"
    marker = '<span class="vs-dot"></span>' if dot else ""
    return f'<span class="{classes}">{marker}{text}</span>'


def hero() -> None:
    features = "".join(
        pill(text) for text in (
            "Timestamped summary", "Auto chapters", "Scene detection",
            "On-screen text", "Cited Q&A",
        )
    )
    st.markdown(
        f"""
        <div class="vs-hero">
          <h1>Understand any video<br/>without watching it</h1>
          <p>Paste a link or upload a file. You get a timestamped summary,
             navigable chapters, and an assistant that answers questions and
             cites the moment it got the answer from.</p>
          <div class="vs-feature-row">{features}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tiles(stats: list[tuple[str, str]]) -> None:
    """A responsive row of value/label tiles."""
    cells = "".join(
        f'<div class="vs-stat"><div class="vs-stat-value">{value}</div>'
        f'<div class="vs-stat-label">{label}</div></div>'
        for value, label in stats
    )
    st.markdown(f'<div class="vs-stats">{cells}</div>', unsafe_allow_html=True)


def stage_tracker(current_stage: str, progress: float, timings: dict[str, float]) -> None:
    """Per-stage checklist showing what is done, running, and still pending.

    A single bar tells the user how far along a run is but not what it is
    doing; for a multi-minute pipeline that reads as a hang. This shows the
    actual stage list with elapsed times as each one lands.

    ``current_stage`` may be a value outside the pipeline order — "error" in
    particular. Stages are therefore marked complete from the recorded
    timings rather than from position alone, so a failure still shows which
    stages had already succeeded instead of resetting the whole list to
    pending, which is exactly when that information matters most.
    """
    order = [key for key, _ in PIPELINE_STAGES]
    failed = current_stage == "error"
    if current_stage == "done":
        current_index = len(order)
    elif current_stage in order:
        current_index = order.index(current_stage)
    else:
        current_index = -1  # unknown or failed: rely on timings alone

    rows = []
    for index, (key, label) in enumerate(PIPELINE_STAGES):
        elapsed = timings.get(key)
        if elapsed is not None:
            icon_class, icon, row_class = "vs-stage-done", "✓", ""
            time_text = f"{elapsed:.1f}s"
        elif index < current_index:
            icon_class, icon, row_class = "vs-stage-done", "✓", ""
            time_text = ""
        elif index == current_index:
            if failed:
                icon_class, icon, row_class = "vs-stage-failed", "!", ""
                time_text = "failed"
            else:
                icon_class, icon, row_class = "vs-stage-active vs-pulsing", "●", ""
                time_text = "running"
        else:
            icon_class, icon, row_class = "vs-stage-pending", str(index + 1), "pending"
            time_text = ""
        rows.append(
            f'<div class="vs-stage {row_class}">'
            f'<div class="vs-stage-icon {icon_class}">{icon}</div>'
            f'<div class="vs-stage-label">{label}</div>'
            f'<div class="vs-stage-time">{time_text}</div></div>'
        )
    st.markdown(f'<div class="vs-stages">{"".join(rows)}</div>', unsafe_allow_html=True)


def encode_image(path: Path, max_width: int = 420) -> str:
    """Base64-encode a keyframe for inline embedding, downscaled first.

    The gallery renders as one block of markup so the grid stays aligned,
    which means the images travel inline rather than through st.image. Full
    720p frames would be megabytes each; gallery tiles are ~190px wide, so
    they are resized before encoding.

    Returns an empty string when the frame is missing or unreadable, which the
    caller treats as "skip this tile".
    """
    if not path.exists():
        return ""
    try:
        import base64
        import io as _io

        from PIL import Image

        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.width > max_width:
                ratio = max_width / image.width
                image = image.resize((max_width, int(image.height * ratio)), Image.LANCZOS)
            buffer = _io.BytesIO()
            image.save(buffer, format="JPEG", quality=78)
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return ""

def section(title: str, hint: str = "") -> None:
    """A labelled section heading, used to structure long tab bodies."""
    hint_html = f'<span class="vs-section-hint">{hint}</span>' if hint else ""
    st.markdown(
        f'<div class="vs-section"><h4>{title}</h4>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def scene_card(keyframe: str, timestamp_html: str, tags: str, on_screen: str) -> str:
    """One scene tile as a single block of markup.

    Rendering the whole tile at once — rather than stacking an st.image on top
    of separate markdown — is what keeps the gallery aligned. Streamlit columns
    stack independently, so tiles of differing height leave the grid ragged.
    """
    text = (
        f'<p class="vs-scene-text">{fmt.html.escape(fmt.truncate(on_screen, 90))}</p>'
        if on_screen else ""
    )
    return (
        '<div class="vs-scene">'
        f'<img src="data:image/jpeg;base64,{keyframe}" loading="lazy" alt=""/>'
        f'<div class="vs-scene-meta">{timestamp_html}{tags}</div>{text}'
        '</div>'
    )

def timestamp_link(seconds: float, source_ref: str, is_upload: bool) -> str:
    """A timestamp chip — a deep link for URLs, plain text for uploads."""
    label = fmt.hhmmss(seconds)
    if is_upload or not source_ref:
        return f'<span class="vs-ts">{label}</span>'
    return f'<a class="vs-ts" href="{fmt.deep_link(source_ref, seconds)}" target="_blank">{label}</a>'


def chapter_list(chapters: list[dict[str, Any]], source_ref: str, is_upload: bool) -> None:
    rows = []
    for chapter in chapters:
        rows.append(
            f'<div class="vs-chapter">'
            f'{timestamp_link(chapter["start"], source_ref, is_upload)}'
            f'<div class="vs-chapter-body">'
            f'<p class="vs-chapter-title">{fmt.html.escape(chapter["title"])}</p>'
            f'<p class="vs-chapter-summary">{fmt.html.escape(chapter.get("summary", ""))}</p>'
            f'</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def transcript_view(
    segments: list[dict[str, Any]], query: str, source_ref: str, is_upload: bool,
    limit: int = 400,
) -> int:
    """Render transcript lines, filtered by ``query``. Returns the match count."""
    if query:
        matches = [s for s in segments if query.lower() in s["text"].lower()]
    else:
        matches = segments

    rows = []
    for segment in matches[:limit]:
        rows.append(
            f'<div class="vs-line">'
            f'{timestamp_link(segment["start"], source_ref, is_upload)}'
            f'<div class="vs-line-text">{fmt.highlight(segment["text"], query)}</div>'
            f'</div>'
        )
    body = "".join(rows) or '<p class="vs-dim">No lines match that search.</p>'
    st.markdown(f'<div class="vs-transcript">{body}</div>', unsafe_allow_html=True)
    return len(matches)


def video_player(
    source_ref: str, is_upload: bool, video_path: Optional[str], start_sec: int = 0
) -> None:
    """Embed the source video, seeked to ``start_sec`` where possible."""
    yt_id = fmt.youtube_id(source_ref) if not is_upload else None
    if yt_id:
        st.markdown(
            f'<div style="position:relative;padding-bottom:56.25%;height:0;'
            f'border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.07)">'
            f'<iframe src="https://www.youtube.com/embed/{yt_id}?start={start_sec}&rel=0" '
            f'style="position:absolute;top:0;left:0;width:100%;height:100%;border:0" '
            f'allow="accelerometer; clipboard-write; encrypted-media; picture-in-picture" '
            f'allowfullscreen></iframe></div>',
            unsafe_allow_html=True,
        )
    elif video_path:
        st.video(video_path, start_time=start_sec)
    else:
        st.info("The source video is no longer available locally.")


def source_badges(meta: dict[str, Any], summary: dict[str, Any], state: dict[str, Any]) -> str:
    """Status chips for the results header."""
    badges = []
    duration = meta.get("duration_sec")
    if duration:
        badges.append(pill(fmt.human_duration(duration)))

    timings = state.get("timings") or {}
    total = timings.get("total")
    if total:
        badges.append(pill(f"Processed in {fmt.human_duration(total)}", "success", dot=True))
        if duration:
            badges.append(pill(f"{duration / total:.0f}× faster than watching", "accent"))

    if summary.get("warning"):
        badges.append(pill(summary["warning"].replace("_", " ").title(), "warning"))
    return "".join(badges)


def empty_state(icon: str, title: str, body: str) -> None:
    """A centred placeholder for a tab whose data does not exist yet."""
    st.markdown(
        f'<div class="vs-empty">'
        f'<div class="vs-empty-icon">{icon}</div>'
        f'<p class="vs-empty-title">{fmt.html.escape(title)}</p>'
        f'<p class="vs-empty-body">{fmt.html.escape(body)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
