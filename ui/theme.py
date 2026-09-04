"""The application's visual design system.

Everything here is presentation only — no pipeline or model logic — so the UI
can be restyled, or replaced wholesale, without touching ``core/``.

Streamlit ships a component library, not a design language: its defaults are
tuned for internal dashboards, and left alone they read as "a Streamlit app"
rather than as a product. This module replaces that with one coherent system —
a layered surface palette, a single accent ramp, one type scale, one radius
scale, one motion curve — and applies it to every control the app actually uses.

The palette is built on *layers* rather than flat fills. Each surface sits a
measured step above the one behind it and carries a hairline top highlight, so
depth comes from light behaving consistently instead of from drop shadows
scattered per component. That is what stops a dark UI reading as flat.
"""

from __future__ import annotations

# Design tokens, mirrored as CSS custom properties below. Kept here as well so
# Python-side code can reference the same values instead of re-typing hex codes
# that then drift out of sync.
TOKENS = {
    "bg": "#07080B",
    "surface": "#0E1015",
    "surface_raised": "#141821",
    "surface_hover": "#1A1F2A",
    "border": "rgba(255, 255, 255, 0.06)",
    "border_strong": "rgba(255, 255, 255, 0.11)",
    "text": "#EDEFF5",
    "text_muted": "#949CB0",
    "text_dim": "#616A7D",
    "accent": "#6D6AF8",
    "accent_bright": "#8B8AFB",
    "violet": "#A855F7",
    "cyan": "#22D3EE",
    "success": "#34D399",
    "warning": "#FBBF24",
    "danger": "#F87171",
}


def logo_svg() -> str:
    """The mark used in the masthead. Inline so it inherits currentColor."""
    return (
        '<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M9 7.5v9l7-4.5-7-4.5z" fill="#fff"/>'
        "</svg>"
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* Surfaces, as ascending layers of light */
    --bg: #07080B;
    --bg-elevated: #0A0C11;
    --surface: #0E1015;
    --surface-raised: #141821;
    --surface-hover: #1A1F2A;
    --surface-active: #202634;

    --border: rgba(255, 255, 255, 0.06);
    --border-strong: rgba(255, 255, 255, 0.11);
    --highlight: rgba(255, 255, 255, 0.05);

    --text: #EDEFF5;
    --text-muted: #949CB0;
    --text-dim: #616A7D;

    --accent: #6D6AF8;
    --accent-bright: #8B8AFB;
    --accent-soft: rgba(109, 106, 248, 0.14);
    --accent-glow: rgba(109, 106, 248, 0.26);
    --violet: #A855F7;
    --cyan: #22D3EE;

    --success: #34D399;
    --success-soft: rgba(52, 211, 153, 0.13);
    --warning: #FBBF24;
    --warning-soft: rgba(251, 191, 36, 0.13);
    --danger: #F87171;
    --danger-soft: rgba(248, 113, 113, 0.13);

    --radius-xs: 6px;
    --radius-sm: 9px;
    --radius: 13px;
    --radius-lg: 18px;

    /* One shadow ramp and one easing curve, used everywhere */
    --shadow-sm: 0 1px 2px rgba(0,0,0,.4);
    --shadow: 0 2px 4px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.28);
    --shadow-lg: 0 4px 8px rgba(0,0,0,.4), 0 20px 48px rgba(0,0,0,.42);
    --ease: cubic-bezier(.22, .61, .36, 1);
}

/* --- Chrome removal ------------------------------------------------------
   Hide Streamlit's hosted-demo furniture so the app reads as a standalone
   product. The header keeps its height reserved for the sidebar toggle. */
#MainMenu, footer { visibility: hidden; height: 0; }
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

html, body, [class*="css"], button, input, textarea, select {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'cv11', 'ss01';
}
code, pre, .vs-mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }

/* A single soft accent wash behind everything. It gives the page a light
   source, which is what keeps large dark areas from looking like dead space. */
.stApp {
    background:
        radial-gradient(1100px 620px at 50% -10%, rgba(109,106,248,.10), transparent 62%),
        radial-gradient(760px 460px at 92% 4%, rgba(168,85,247,.06), transparent 58%),
        var(--bg);
    background-attachment: fixed;
    color: var(--text);
}

.block-container { max-width: 1260px; padding: 1.5rem 2.25rem 6rem 2.25rem; }

/* --- Type scale ---------------------------------------------------------- */
h1, h2, h3, h4 { color: var(--text); font-weight: 700; letter-spacing: -0.022em; }
h1 { font-size: 1.95rem; }
h2 { font-size: 1.4rem; }
h3 { font-size: 1.12rem; letter-spacing: -0.018em; }
p, li, .stMarkdown { color: var(--text); line-height: 1.7; }
.vs-dim { color: var(--text-muted); font-size: .875rem; margin: 0; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* --- Masthead ------------------------------------------------------------ */
.vs-masthead {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; padding-bottom: 1.05rem; margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.vs-brand { display: flex; align-items: center; gap: .85rem; min-width: 0; }
.vs-logo {
    width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(140deg, var(--accent), var(--violet));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 20px rgba(109,106,248,.38),
                inset 0 1px 0 rgba(255,255,255,.28);
}
.vs-logo svg { width: 21px; height: 21px; display: block; }
.vs-brand-text { min-width: 0; }
.vs-brand-name {
    font-size: 1.04rem; font-weight: 700; letter-spacing: -0.021em;
    margin: 0; line-height: 1.25; color: var(--text);
}
.vs-brand-sub {
    font-size: .78rem; color: var(--text-dim); margin: .12rem 0 0 0; line-height: 1.2;
}
.vs-masthead-meta { display: flex; align-items: center; gap: .45rem; flex-shrink: 0; }

/* --- Pills / badges ------------------------------------------------------ */
.vs-pill {
    display: inline-flex; align-items: center; gap: .35rem;
    padding: .24rem .66rem; border-radius: 999px;
    font-size: .715rem; font-weight: 600; letter-spacing: .012em;
    border: 1px solid var(--border); background: var(--surface-raised);
    color: var(--text-muted); white-space: nowrap; margin: 0 .28rem .3rem 0;
}
.vs-pill-accent  { background: var(--accent-soft);  color: var(--accent-bright); border-color: rgba(109,106,248,.3); }
.vs-pill-success { background: var(--success-soft); color: var(--success);       border-color: rgba(52,211,153,.28); }
.vs-pill-warning { background: var(--warning-soft); color: var(--warning);       border-color: rgba(251,191,36,.28); }
.vs-pill-danger  { background: var(--danger-soft);  color: var(--danger);        border-color: rgba(248,113,113,.28); }
.vs-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; flex-shrink: 0;
    box-shadow: 0 0 8px currentColor;
}

/* --- Hero ---------------------------------------------------------------- */
.vs-hero { text-align: center; padding: 2.9rem 1rem 2rem 1rem; }
.vs-hero h1 {
    font-size: 2.85rem; font-weight: 800; letter-spacing: -0.038em;
    margin: 0 0 .85rem 0; line-height: 1.1;
    background: linear-gradient(168deg, #FFFFFF 12%, #C3C2FD 62%, #9B8AF5 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.vs-hero p {
    color: var(--text-muted); font-size: 1.02rem; max-width: 545px;
    margin: 0 auto; line-height: 1.68;
}
.vs-feature-row {
    display: flex; gap: .4rem; justify-content: center; flex-wrap: wrap;
    margin-top: 1.7rem;
}

/* --- Cards ---------------------------------------------------------------
   The hairline top highlight is what reads as a lit edge; without it a raised
   panel on a dark ground looks like a hole rather than a surface. */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(180deg, var(--surface-raised), var(--surface)) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow), inset 0 1px 0 var(--highlight);
}

/* --- Buttons ------------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    border-radius: var(--radius-sm); font-weight: 600; font-size: .875rem;
    border: 1px solid var(--border-strong);
    background: linear-gradient(180deg, var(--surface-hover), var(--surface-raised));
    color: var(--text); padding: .55rem 1.05rem;
    transition: transform .16s var(--ease), box-shadow .16s var(--ease),
                background .16s var(--ease), border-color .16s var(--ease);
    box-shadow: var(--shadow-sm), inset 0 1px 0 var(--highlight);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(180deg, var(--surface-active), var(--surface-hover));
    border-color: var(--border-strong); color: var(--text);
    transform: translateY(-1px); box-shadow: var(--shadow);
}
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }

.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), var(--violet));
    border: none; color: #fff; font-weight: 650;
    box-shadow: 0 2px 10px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,.22);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 18px var(--accent-glow), inset 0 1px 0 rgba(255,255,255,.28);
    filter: brightness(1.06); transform: translateY(-1px);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
}

/* --- Inputs -------------------------------------------------------------- */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important; font-size: .92rem !important;
    padding: .68rem .85rem !important;
    transition: border-color .16s var(--ease), box-shadow .16s var(--ease);
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
.stTextInput input::placeholder { color: var(--text-dim) !important; }
div[data-baseweb="input"], div[data-baseweb="base-input"] { background: transparent !important; }

[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-elevated); border: 1px dashed var(--border-strong);
    border-radius: var(--radius); transition: all .16s var(--ease);
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent); background: var(--surface);
}

/* Checkbox + radio accents */
.stCheckbox label span, .stRadio label span { color: var(--text-muted) !important; font-size: .875rem; }
[data-testid="stWidgetLabel"] p { color: var(--text-muted) !important; font-size: .82rem !important; font-weight: 500; }

/* --- Tabs as a segmented control ------------------------------------------ */
div[data-baseweb="tab-list"] {
    gap: 3px; background: var(--surface); padding: 4px;
    border-radius: var(--radius); width: fit-content;
    border: 1px solid var(--border); box-shadow: inset 0 1px 2px rgba(0,0,0,.3);
}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
button[data-baseweb="tab"] {
    border-radius: var(--radius-xs) !important; font-weight: 600;
    padding: .42rem .95rem !important; color: var(--text-dim) !important;
    background: transparent !important; transition: all .16s var(--ease);
}
button[data-baseweb="tab"]:hover { color: var(--text) !important; background: var(--surface-hover) !important; }
button[aria-selected="true"][data-baseweb="tab"] {
    background: linear-gradient(180deg, var(--surface-active), var(--surface-hover)) !important;
    color: var(--text) !important;
    box-shadow: var(--shadow-sm), inset 0 1px 0 var(--highlight);
}
button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p { font-size: .85rem; margin: 0; }

/* --- Sidebar -------------------------------------------------------------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface), var(--bg-elevated));
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.vs-sidebar-label {
    font-size: .68rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: var(--text-dim);
    margin: 0 0 .6rem .15rem;
}
/* Library entries read as navigation, not as form buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: 1px solid transparent;
    text-align: left; justify-content: flex-start;
    font-weight: 500; font-size: .845rem; color: var(--text-muted);
    padding: .5rem .6rem; box-shadow: none; border-radius: var(--radius-sm);
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--surface-hover); border-color: var(--border);
    color: var(--text); transform: none; box-shadow: none;
}
section[data-testid="stSidebar"] .stButton > button p { font-size: .845rem !important; }
section[data-testid="stSidebar"] hr { margin: 1.1rem 0 !important; }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--text-dim); font-size: .76rem; line-height: 1.5;
}

/* --- Progress ------------------------------------------------------------- */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--violet)) !important;
    box-shadow: 0 0 14px var(--accent-glow);
}
.stProgress > div > div { background: var(--surface-raised) !important; border-radius: 999px; height: 6px !important; }

.vs-stages { display: flex; flex-direction: column; gap: .1rem; margin-top: .5rem; }
.vs-stage {
    display: flex; align-items: center; gap: .7rem;
    padding: .5rem .3rem; border-bottom: 1px solid var(--border);
    font-size: .875rem;
}
.vs-stage:last-child { border-bottom: none; }
.vs-stage-icon {
    width: 22px; height: 22px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: .7rem; font-weight: 700;
}
.vs-stage-done    { background: var(--success-soft); color: var(--success); }
.vs-stage-active  { background: var(--accent-soft);  color: var(--accent-bright); }
.vs-stage-pending { background: var(--surface-raised); color: var(--text-dim); }
.vs-stage-failed  { background: var(--danger-soft);  color: var(--danger); }
.vs-stage-label { flex: 1; color: var(--text); font-weight: 500; }
.vs-stage.pending .vs-stage-label { color: var(--text-dim); font-weight: 400; }
.vs-stage-time {
    color: var(--text-dim); font-size: .76rem;
    font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums;
}
@keyframes vs-pulse { 0%,100% { opacity: 1; } 50% { opacity: .45; } }
.vs-pulsing { animation: vs-pulse 1.5s ease-in-out infinite; }

/* --- Stat tiles ----------------------------------------------------------- */
.vs-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: .6rem; }
.vs-stat {
    background: linear-gradient(180deg, var(--surface-raised), var(--surface));
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: .8rem .85rem; box-shadow: inset 0 1px 0 var(--highlight);
    transition: border-color .16s var(--ease);
}
.vs-stat:hover { border-color: var(--border-strong); }
.vs-stat-value {
    font-size: 1.35rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.025em; line-height: 1.15;
    font-variant-numeric: tabular-nums;
}
.vs-stat-label {
    font-size: .68rem; font-weight: 600; letter-spacing: .07em;
    text-transform: uppercase; color: var(--text-dim); margin-top: .3rem;
}

/* --- Timestamps ----------------------------------------------------------- */
.vs-ts {
    display: inline-flex; align-items: center; flex-shrink: 0;
    font-family: 'JetBrains Mono', monospace; font-size: .765rem; font-weight: 600;
    color: var(--accent-bright); background: var(--accent-soft);
    padding: .18rem .48rem; border-radius: var(--radius-xs);
    text-decoration: none !important; font-variant-numeric: tabular-nums;
    border: 1px solid transparent; transition: all .14s var(--ease);
}
a.vs-ts:hover { background: rgba(109,106,248,.26); border-color: rgba(109,106,248,.4); }

/* --- Chapters ------------------------------------------------------------- */
.vs-chapter {
    display: flex; gap: .9rem; padding: .95rem .75rem;
    border-radius: var(--radius); border: 1px solid transparent;
    transition: background .16s var(--ease), border-color .16s var(--ease);
}
.vs-chapter:hover { background: var(--surface); border-color: var(--border); }
.vs-chapter-body { min-width: 0; flex: 1; }
.vs-chapter-title {
    font-weight: 650; font-size: .955rem; margin: 0 0 .25rem 0;
    color: var(--text); letter-spacing: -0.012em;
}
.vs-chapter-summary { color: var(--text-muted); font-size: .87rem; line-height: 1.62; margin: 0; }

/* --- Walkthrough sections ----------------------------------------------- */
.vs-walk {
    padding: 1rem 0 1.1rem 0;
    border-bottom: 1px solid var(--border);
}
.vs-walk:last-of-type { border-bottom: none; }
.vs-walk-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }
.vs-walk-num {
    display: flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; flex-shrink: 0; border-radius: 6px;
    font-size: .7rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
    color: var(--text-dim); background: var(--surface-raised);
    border: 1px solid var(--border);
}
.vs-walk-text {
    margin: 0; color: var(--text); font-size: .925rem; line-height: 1.72;
}
.vs-keypoints {
    padding: 1rem 1.15rem; border-radius: var(--radius);
    background: linear-gradient(180deg, var(--accent-soft), transparent);
    border: 1px solid rgba(109,106,248,.22);
    color: var(--text); font-size: .93rem; line-height: 1.7;
}

/* --- Transcript ----------------------------------------------------------- */
.vs-transcript {
    max-height: 580px; overflow-y: auto; padding-right: .55rem;
    border-radius: var(--radius);
}
.vs-line {
    display: flex; gap: .85rem; padding: .45rem .5rem;
    border-radius: var(--radius-xs); transition: background .12s var(--ease);
}
.vs-line:hover { background: var(--surface); }
.vs-line-text { color: var(--text); font-size: .885rem; line-height: 1.64; }
.vs-line mark {
    background: rgba(251,191,36,.26); color: #FDE68A;
    padding: 0 3px; border-radius: 3px;
}

/* --- Sources (Q&A citations) ---------------------------------------------- */
.vs-source {
    display: flex; gap: .7rem; padding: .55rem .3rem;
    border-bottom: 1px solid var(--border); font-size: .845rem;
    color: var(--text-muted); line-height: 1.6;
}
.vs-source:last-child { border-bottom: none; }

/* --- Scene gallery -------------------------------------------------------- */
.vs-scene-meta { display: flex; align-items: center; gap: .38rem; margin-top: .5rem; flex-wrap: wrap; }
.vs-tag {
    font-size: .68rem; font-weight: 600; color: var(--text-muted);
    background: var(--surface-raised); border: 1px solid var(--border);
    padding: .14rem .45rem; border-radius: var(--radius-xs);
}
[data-testid="stImage"] img {
    border-radius: var(--radius-sm); border: 1px solid var(--border);
    transition: transform .18s var(--ease), border-color .18s var(--ease);
}
[data-testid="stImage"] img:hover { transform: scale(1.015); border-color: var(--border-strong); }

/* --- Empty states -------------------------------------------------------- */
.vs-empty {
    text-align: center; padding: 3.2rem 1.5rem;
    border: 1px dashed var(--border-strong); border-radius: var(--radius-lg);
    background: var(--bg-elevated);
}
.vs-empty-icon {
    font-size: 1.75rem; line-height: 1; color: var(--text-dim);
    width: 56px; height: 56px; margin: 0 auto .9rem auto;
    display: flex; align-items: center; justify-content: center;
    background: var(--surface-raised); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: inset 0 1px 0 var(--highlight);
}
.vs-empty-title { font-weight: 650; color: var(--text); margin: 0 0 .35rem 0; font-size: .98rem; }
.vs-empty-body { font-size: .865rem; color: var(--text-muted); margin: 0; }

/* --- Results header -------------------------------------------------------- */
.vs-result-head { margin: 2rem 0 1.25rem 0; padding-top: 1.6rem; border-top: 1px solid var(--border); }
.vs-result-head h2 {
    margin: 0 0 .6rem 0; font-size: 1.45rem; font-weight: 700;
    letter-spacing: -0.026em; line-height: 1.3; color: var(--text);
}
.vs-result-badges { display: flex; flex-wrap: wrap; align-items: center; }

/* --- Section headings ------------------------------------------------------ */
.vs-section {
    display: flex; align-items: baseline; gap: .6rem;
    margin: .3rem 0 .85rem 0; padding-bottom: .5rem;
    border-bottom: 1px solid var(--border);
}
.vs-section h4 {
    margin: 0; font-size: .95rem; font-weight: 650;
    letter-spacing: -0.012em; color: var(--text);
}
.vs-section-hint { font-size: .78rem; color: var(--text-dim); }

/* --- Scene gallery grid ----------------------------------------------------
   A CSS grid rather than Streamlit columns: columns stack independently, so
   tiles of differing height leave the rows ragged. */
.vs-scene-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: .85rem;
}
.vs-scene {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden; padding-bottom: .7rem;
    transition: border-color .16s var(--ease), transform .16s var(--ease);
}
.vs-scene:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.vs-scene img { width: 100%; aspect-ratio: 16/9; object-fit: cover; display: block; }
.vs-scene .vs-scene-meta { padding: 0 .65rem; margin-top: .55rem; }
.vs-scene-text {
    padding: 0 .65rem; margin: .45rem 0 0 0;
    font-size: .755rem; color: var(--text-dim); line-height: 1.5;
}

/* --- Chat (Q&A) ----------------------------------------------------------- */
[data-testid="stChatMessage"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: .95rem 1.1rem; margin-bottom: .75rem;
    box-shadow: inset 0 1px 0 var(--highlight);
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-elevated) !important; border-radius: var(--radius-sm) !important;
}

/* --- Expander ------------------------------------------------------------- */
[data-testid="stExpander"] details {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); overflow: hidden;
}
[data-testid="stExpander"] summary { font-size: .84rem; font-weight: 550; color: var(--text-muted); }
[data-testid="stExpander"] summary:hover { color: var(--text); }

/* --- Alerts --------------------------------------------------------------- */
[data-testid="stAlert"] { border-radius: var(--radius); border: 1px solid var(--border); font-size: .875rem; }

/* --- Slider ------------------------------------------------------------------
   Streamlit paints the filled portion of the track with a hard red gradient
   (rgb(255,75,75)) set from its own stylesheet, so it has to be overridden by
   selector rather than by touching an inline style. */
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft), 0 2px 6px rgba(0,0,0,.4) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-image: linear-gradient(90deg, var(--accent), var(--violet)) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
    background: var(--surface-active) !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"],
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--text-dim) !important; font-size: .72rem;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--accent-bright) !important; font-weight: 600;
    background: var(--accent-soft); border-radius: var(--radius-xs);
    padding: 0 .3rem;
}

/* --- Scrollbars ----------------------------------------------------------- */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface-active); border-radius: 999px; border: 2px solid transparent; background-clip: content-box; }
::-webkit-scrollbar-thumb:hover { background: #2C3444; background-clip: content-box; }

/* --- Motion preference ----------------------------------------------------
   Respecting this is not optional: the pulsing stage indicator and hover
   transforms are exactly the kind of motion that triggers discomfort. */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .001ms !important; animation-iteration-count: 1 !important;
        transition-duration: .001ms !important;
    }
}

/* --- Narrow screens -------------------------------------------------------- */
@media (max-width: 820px) {
    .block-container { padding: 1rem 1.1rem 4rem 1.1rem; }
    .vs-hero h1 { font-size: 2.05rem; }
    .vs-hero { padding: 1.9rem .5rem 1.4rem .5rem; }
    .vs-masthead { flex-wrap: wrap; gap: .6rem; }
    .vs-stats { grid-template-columns: repeat(2, 1fr); }
}
</style>
"""
