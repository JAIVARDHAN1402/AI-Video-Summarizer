"""Design diagrams for Chapter 4 of the Project-I report.

Five figures: a context data-flow diagram, a level-1 data-flow diagram, a
use-case diagram, a class diagram and a sequence diagram. They are drawn from
the code they describe — the modules in core/, the tables in server/db.py and
the call order in server/worker.py — rather than from a sketch of it, so that
the report and the system do not drift apart.

Run from the project root:  python docs/make_diagrams.py
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = "docs/figures"

INK = "#1b2434"
BLUE = "#1d3f72"
BLUE_FILL = "#eef3fb"
GREY_FILL = "#f3f4f6"
AMBER = "#b08900"
AMBER_FILL = "#fff8e6"
GREEN = "#2f7a46"
GREEN_FILL = "#eaf6ee"
MUTED = "#6b7280"


def box(ax, x, y, w, h, text, fc=GREY_FILL, ec=INK, fontsize=8.6, bold=False,
        rounding=0.018, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=zorder + 1,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def rect(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fontsize=8.6, bold=False):
    ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.0, edgecolor=ec,
                               facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def store(ax, x, y, w, h, text, fontsize=8.2):
    """An open-ended data store, in the Gane-Sarson manner."""
    ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.0, edgecolor=INK,
                               facecolor="#ffffff", zorder=2))
    ax.plot([x, x + w], [y + h, y + h], color=INK, lw=1.0, zorder=3)
    ax.plot([x, x + w], [y, y], color=INK, lw=1.0, zorder=3)
    ax.plot([x + w * 0.16, x + w * 0.16], [y, y + h], color=INK, lw=1.0,
            zorder=3)
    ax.text(x + w * 0.58, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=4, linespacing=1.4)


def proc(ax, cx, cy, r, text, fc=BLUE_FILL, fontsize=8.4):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=BLUE,
                        linewidth=1.1, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=3, linespacing=1.4)


def arrow(ax, p, q, color=INK, lw=1.0, dashed=False, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=10, linewidth=lw, color=color,
        zorder=1, shrinkA=2, shrinkB=2, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def label(ax, x, y, text, color=MUTED, fontsize=7.2, ha="center",
          style="normal", rotation=0):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
            zorder=5, style=style, rotation=rotation)


def actor(ax, cx, cy, name, scale=1.0):
    """A stick figure, so the use-case diagram reads as one."""
    s = scale
    ax.add_patch(Circle((cx, cy + 2.6 * s), 0.95 * s, facecolor="#ffffff",
                        edgecolor=INK, linewidth=1.1, zorder=3))
    ax.plot([cx, cx], [cy + 1.6 * s, cy - 1.0 * s], color=INK, lw=1.1, zorder=3)
    ax.plot([cx - 1.4 * s, cx + 1.4 * s], [cy + 0.9 * s, cy + 0.9 * s],
            color=INK, lw=1.1, zorder=3)
    ax.plot([cx, cx - 1.1 * s], [cy - 1.0 * s, cy - 3.0 * s], color=INK,
            lw=1.1, zorder=3)
    ax.plot([cx, cx + 1.1 * s], [cy - 1.0 * s, cy - 3.0 * s], color=INK,
            lw=1.1, zorder=3)
    ax.text(cx, cy - 4.3 * s, name, ha="center", va="center", fontsize=8.2,
            color=INK, zorder=3, linespacing=1.3)


def finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ============================================================ 4.1 DFD level 0
def dfd_level0():
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    proc(ax, 50, 29, 11.5, "0\nVideoSense\nvideo-understanding\nsystem",
         fontsize=9.0)

    rect(ax, 3, 36, 19, 7.5, "Student\n(free or premium)", bold=True)
    rect(ax, 3, 13, 19, 7.5, "Video source\nURL or upload", bold=True)
    rect(ax, 78, 36, 19, 7.5, "Language-model\nprovider", bold=True)
    rect(ax, 78, 13, 19, 7.5, "Payment\ngateway", bold=True)

    arrow(ax, (22, 40), (40.5, 33.5))
    label(ax, 29, 39.5, "link, deadline,\nprofile", fontsize=7.0)
    arrow(ax, (40.5, 26), (22, 18), rad=0.0)
    label(ax, 28.5, 20.0, "fetch request", fontsize=7.0)
    arrow(ax, (22, 15.5), (39.5, 24.0), rad=0.08)
    label(ax, 31.5, 15.4, "media stream", fontsize=7.0)

    arrow(ax, (59.5, 33.5), (78, 40))
    label(ax, 70, 39.4, "windowed prompts", fontsize=7.0)
    arrow(ax, (78, 37.5), (60.5, 31.0), rad=0.08)
    label(ax, 69.5, 32.2, "explanation, quiz", fontsize=7.0)

    arrow(ax, (59.5, 24.5), (78, 18.5))
    label(ax, 70.5, 23.0, "order, signature", fontsize=7.0)
    arrow(ax, (78, 16), (60.5, 23.0), rad=0.08)
    label(ax, 68.5, 15.6, "payment status", fontsize=7.0)

    arrow(ax, (41.5, 35.5), (22, 38.5), rad=-0.18)
    label(ax, 31, 32.0, "estimate before the run starts, then the\n"
                        "explanation, transcript, quiz and practice set",
          fontsize=7.0, color=BLUE)

    ax.text(50, 52.5, "Figure 4.1  Context (level-0) data-flow diagram",
            ha="center", va="center", fontsize=10.2, color=INK)
    finish(fig, "fig-4-1-dfd-level0")


# ============================================================ 4.2 DFD level 1
def dfd_level1():
    """Two rows, read left to right then right to left.

    The first attempt put the processes in a ring, which forced the media
    and calibration flows into long diagonals across the middle of the
    figure and through two of the process circles. Here every flow is either
    along a row or a short step to the store directly above or below it.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.set_xlim(0, 112)
    ax.set_ylim(0, 66)
    ax.axis("off")

    R = 7.2
    top_y, bot_y = 42.0, 17.0
    P1, P2, P3, P4 = 24.0, 46.0, 68.0, 90.0
    P5, P6, P7 = 90.0, 68.0, 46.0

    proc(ax, P1, top_y, R, "1\nAuthenticate\nand check\nallowance", fontsize=7.2)
    proc(ax, P2, top_y, R, "2\nPredict cost\nand route\n(scheduler)", fontsize=7.2)
    proc(ax, P3, top_y, R, "3\nIngest\n(M1)", fontsize=7.4)
    proc(ax, P4, top_y, R, "4\nTranscribe\nand see\n(M2, M3)", fontsize=7.2)
    proc(ax, P5, bot_y, R, "5\nFuse and\nunderstand\n(M4, M5′)", fontsize=7.2)
    proc(ax, P6, bot_y, R, "6\nIndex and\nhighlight\n(M6, M7)", fontsize=7.2)
    proc(ax, P7, bot_y, R, "7\nServe the\nworkspace", fontsize=7.4)

    rect(ax, 2, 25.5, 13, 8.0, "Student", bold=True, fontsize=8.4)
    rect(ax, 60, 55, 16, 6.0, "Video source\nURL or upload", bold=True,
         fontsize=7.8)
    rect(ax, 100, 13.5, 11, 7.0, "LLM\nprovider", bold=True, fontsize=7.8)

    store(ax, 14, 55, 21, 5.0, "D1  Users, quotas,\npayments", fontsize=7.4)
    store(ax, 38, 55, 19, 5.0, "D2  Jobs, SLA\nobservations", fontsize=7.4)
    store(ax, 56, 3.0, 22, 5.0, "D3  Stage artefact cache", fontsize=7.4)
    store(ax, 24, 3.0, 22, 5.0, "D4  Per-class cost models", fontsize=7.4)

    # the request path, left to right along the top
    arrow(ax, (15, 31), (P1 - R + 0.6, 38.5))
    label(ax, 17.5, 35.8, "link,\ndeadline", fontsize=6.8)
    for a, b in ((P1, P2), (P2, P3), (P3, P4)):
        arrow(ax, (a + R, top_y), (b - R, top_y))
    label(ax, (P1 + P2) / 2, top_y + 1.8, "allowed", fontsize=6.8)
    label(ax, (P2 + P3) / 2, top_y + 1.8, "chosen class", fontsize=6.8)
    label(ax, (P3 + P4) / 2, top_y + 1.8, "audio, frames", fontsize=6.8)

    # and back along the bottom
    arrow(ax, (P4, top_y - R), (P5, bot_y + R))
    label(ax, P4 + 6.5, 29.5, "aligned\ndocument", fontsize=6.8)
    for a, b in ((P5, P6), (P6, P7)):
        arrow(ax, (a - R, bot_y), (b + R, bot_y))
    label(ax, (P5 + P6) / 2, bot_y + 1.8, "explanation", fontsize=6.8)
    label(ax, (P6 + P7) / 2, bot_y + 1.8, "index, reel", fontsize=6.8)
    arrow(ax, (P7 - R + 0.6, bot_y + 3.0), (15, 27.5))
    label(ax, 34.0, 27.5, "workspace", fontsize=6.8)

    # stores and external systems, each beside the process that uses it
    arrow(ax, (P1, top_y + R), (24, 55), dashed=True)
    arrow(ax, (P2, top_y + R), (46, 55), dashed=True)
    arrow(ax, (P3, top_y + R), (66, 55))
    arrow(ax, (69.5, 55), (69.5, top_y + R), rad=0.0)
    label(ax, 76.5, 50.5, "fetch /\nmedia", fontsize=6.8)
    arrow(ax, (P6, bot_y - R), (67, 8.0), dashed=True)
    arrow(ax, (P5 + R, bot_y), (100, bot_y))
    arrow(ax, (100, bot_y - 2.6), (P5 + R - 0.5, bot_y - 3.0))

    # the loop that makes the scheduler adaptive
    arrow(ax, (P2 - 2.0, top_y - R), (35, 8.0), color=AMBER, dashed=True,
          rad=0.12)
    arrow(ax, (37.5, 8.0), (P2 + 1.0, top_y - R + 0.4), color=AMBER, rad=0.12)
    label(ax, 2.0, 12.5, "measured cost after every run,\nread back when routing",
          fontsize=6.8, color=AMBER, ha="left")

    ax.text(56, 64.0, "Figure 4.2  Level-1 data-flow diagram",
            ha="center", va="center", fontsize=10.2, color=INK)
    finish(fig, "fig-4-2-dfd-level1")


# ============================================================== 4.3 use cases
def use_cases():
    fig, ax = plt.subplots(figsize=(11.6, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch(
        (26, 4), 48, 62, boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.1, edgecolor=INK, facecolor="#fcfcfd", zorder=0))
    ax.text(50, 64.6, "VideoSense", ha="center", va="center", fontsize=9.4,
            color=INK, fontweight="bold")

    cases = [
        (61.0, "Sign in with a code or Google"),
        (55.5, "Submit a video and see the estimate"),
        (50.0, "Read the chaptered explanation"),
        (44.5, "Ask a question, cited to the moment"),
        (39.0, "Take the quiz and flashcards"),
        (33.5, "Get every problem in another dialect"),
        (28.0, "Resume a part-finished run"),
        (22.5, "Upgrade to premium"),
        (17.0, "Route the job to a worker class"),
        (11.5, "Calibrate the cost model"),
    ]
    for y, text in cases:
        fc = AMBER_FILL if y <= 17.0 else "#ffffff"
        ec = AMBER if y <= 17.0 else BLUE
        ax.add_patch(Ellipse((50, y), 42, 4.4, facecolor=fc, edgecolor=ec,
                             linewidth=1.0, zorder=2))
        ax.text(50, y, text, ha="center", va="center", fontsize=8.0,
                color=INK, zorder=3)

    actor(ax, 11, 44, "Student", scale=1.25)
    actor(ax, 11, 20, "Premium\nstudent", scale=1.25)
    actor(ax, 89, 46, "LLM\nprovider", scale=1.25)
    actor(ax, 89, 26, "Payment\ngateway", scale=1.25)
    actor(ax, 89, 10, "Scheduler\n(system)", scale=1.25)

    for y in (61.0, 55.5, 50.0, 44.5, 39.0, 33.5):
        arrow(ax, (14, 43.5), (29, y), style="-", lw=0.7, color=MUTED)
    for y in (28.0, 22.5):
        arrow(ax, (14, 19.5), (29, y), style="-", lw=0.7, color=MUTED)
    arrow(ax, (86, 45.5), (71, 50.0), style="-", lw=0.7, color=MUTED)
    arrow(ax, (86, 45.5), (71, 39.0), style="-", lw=0.7, color=MUTED)
    arrow(ax, (86, 25.5), (71, 22.5), style="-", lw=0.7, color=MUTED)
    arrow(ax, (86, 9.5), (71, 17.0), style="-", lw=0.7, color=MUTED)
    arrow(ax, (86, 9.5), (71, 11.5), style="-", lw=0.7, color=MUTED)

    arrow(ax, (31.5, 55.5), (31.5, 17.0), style="-|>", lw=0.8, color=MUTED,
          dashed=True, rad=0.28)
    label(ax, 24.0, 36.0, "«include»", fontsize=6.8, rotation=90)

    ax.text(50, 70.4, "Figure 4.3  Use-case diagram",
            ha="center", va="center", fontsize=10.2, color=INK)
    finish(fig, "fig-4-3-use-case")


# ============================================================= 4.4 class view
def class_diagram():
    fig, ax = plt.subplots(figsize=(13.4, 8.0))
    ax.set_xlim(0, 114)
    ax.set_ylim(0, 70)
    ax.axis("off")

    def cls(x, y, w, name, attrs, ops, fc="#ffffff", h_head=4.0):
        ha = 2.6 * len(attrs) + 1.4
        ho = 2.6 * len(ops) + 1.4
        total = h_head + ha + ho
        ax.add_patch(plt.Rectangle((x, y), w, total, linewidth=1.0,
                                   edgecolor=INK, facecolor=fc, zorder=2))
        ax.plot([x, x + w], [y + total - h_head] * 2, color=INK, lw=1.0,
                zorder=3)
        ax.plot([x, x + w], [y + ho] * 2, color=INK, lw=1.0, zorder=3)
        ax.text(x + w / 2, y + total - h_head / 2, name, ha="center",
                va="center", fontsize=8.6, fontweight="bold", color=INK,
                zorder=3)
        for i, a in enumerate(attrs):
            ax.text(x + 1.2, y + total - h_head - 2.0 - i * 2.6, a,
                    ha="left", va="center", fontsize=7.2, color=INK, zorder=3)
        for i, o in enumerate(ops):
            ax.text(x + 1.2, y + ho - 2.0 - i * 2.6, o, ha="left",
                    va="center", fontsize=7.2, color=BLUE, zorder=3)
        return (x, y, w, total)

    cls(2, 44, 20, "User",
        ["id, email, phone", "tier, tier_expires_at", "signup_method"],
        ["+ effective_tier()"], fc=GREEN_FILL)
    cls(2, 23, 20, "Payment",
        ["order_id, amount", "status, signature"],
        ["+ verify()"], fc=GREEN_FILL)
    cls(29, 40, 24, "Job",
        ["id, source_ref, profile", "video_id, duration_sec",
         "routed_worker_class", "predicted_sec, sla_target_sec",
         "status, stage, timings"],
        ["+ elapsed_sec()"], fc=BLUE_FILL)
    cls(29, 14, 24, "Pipeline",
        ["video_id, profile", "reused_stages"],
        ["+ run()", "+ resume()"], fc=BLUE_FILL)
    cls(60, 46, 22, "SlaObservation",
        ["duration_sec, actual_sec", "predicted_sec, baseline_sec",
         "sla_met, baseline_met"],
        ["+ record_outcome()"], fc=AMBER_FILL)
    cls(60, 24, 22, "SchedulerCost",
        ["worker_class, key", "seconds, samples", "updated_at"],
        ["+ update()"], fc=AMBER_FILL)
    cls(86, 32, 24, "CostModel",
        ["n, sx, sy, sxx, sxy", "fixed, rate"],
        ["+ predict(minutes)", "+ observe(x, y)", "+ is_credible()"],
        fc=AMBER_FILL)
    cls(60, 4, 22, "StageArtefactCache",
        ["video_id", "completed_stages"],
        ["+ read_json()", "+ write_json()"], fc=GREY_FILL)

    # User owns jobs and payments
    arrow(ax, (22, 49), (29, 49), style="-", lw=1.0)
    label(ax, 25.5, 50.6, "1..*", fontsize=6.8)
    arrow(ax, (12, 44), (12, 32.2), style="-", lw=1.0)
    label(ax, 14.6, 38.0, "1..*", fontsize=6.8)

    # a job is executed by the pipeline and yields one observation
    arrow(ax, (41, 40), (41, 27.0), style="-|>", lw=1.0)
    label(ax, 46.5, 33.5, "executes", fontsize=6.8)
    arrow(ax, (53, 50), (60, 52), style="-|>", lw=1.0)
    label(ax, 56.5, 53.4, "yields", fontsize=6.8)

    # the observation calibrates the stored cost, which the model reads
    arrow(ax, (71, 46), (71, 39.0), style="-|>", lw=1.0, color=AMBER,
          dashed=True)
    label(ax, 78.0, 42.5, "calibrates", fontsize=6.8, color=AMBER)
    arrow(ax, (82, 33), (86, 38.0), style="-", lw=1.0, color=AMBER)
    label(ax, 83.5, 30.5, "rows of", fontsize=6.8, color=AMBER)

    # the pipeline reads and writes the artefact cache
    arrow(ax, (53, 17), (60, 11.0), style="-|>", lw=1.0, dashed=True)
    label(ax, 55.0, 9.0, "reads / writes", fontsize=6.8)

    ax.text(57, 68.0, "Figure 4.4  Class diagram of the persistent model "
                      "and the scheduling classes",
            ha="center", va="center", fontsize=10.2, color=INK)
    finish(fig, "fig-4-4-class")


# ============================================================== 4.5 sequence
def sequence_diagram():
    fig, ax = plt.subplots(figsize=(12.0, 8.4))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 76)
    ax.axis("off")

    # The last lane needs half its width inside the canvas, or its header
    # box is clipped at the right edge.
    lanes = [
        (8, "Student"),
        (26, "API\n(FastAPI)"),
        (44, "Scheduler"),
        (62, "Worker"),
        (80, "Cache +\nLLM"),
        (98, "Database"),
    ]
    for x, name in lanes:
        box(ax, x - 7.0, 68, 14.0, 5.2, name, fc=BLUE_FILL, ec=BLUE,
            fontsize=8.0, bold=True)
        ax.plot([x, x], [68, 5], color=MUTED, lw=0.8, linestyle=(0, (4, 4)),
                zorder=0)

    steps = [
        (64.0, 8, 26, "POST /api/jobs  (link, profile, deadline)", False),
        (60.0, 26, 98, "check allowance for this tier", False),
        (56.5, 98, 26, "allowance remaining", True),
        (53.0, 26, 44, "route(duration, deadline)", False),
        (49.5, 44, 98, "read per-class cost models", False),
        (46.0, 98, 44, "fixed + rate x minutes, per class", True),
        (42.5, 44, 26, "cheapest class that meets the deadline,\n"
                       "or a declared overrun", True),
        (38.0, 26, 8, "202 Accepted + predicted finishing time", True),
        (35.0, 26, 62, "enqueue the job", False),
        (29.0, 62, 80, "M1 ingest, M2 transcribe, M3 vision", False),
        (25.5, 80, 62, "artefacts, reused where cached", True),
        (22.0, 62, 80, "M4 fuse, M5' understand (windowed)", False),
        (18.5, 80, 62, "explanation, quiz, practice set", True),
        (14.5, 62, 98, "M6 index, M7 highlight, write result", False),
        (11.0, 62, 44, "record_outcome(actual, reused_stages)", False),
        (7.5, 44, 98, "update the affine fit for that class", False),
        (4.0, 26, 8, "GET /api/videos/{id}  ->  the workspace", True),
    ]
    for y, a, b, text, dashed in steps:
        arrow(ax, (a, y), (b, y), lw=1.0, dashed=dashed,
              color=AMBER if "affine" in text or "cost models" in text else INK)
        ax.text((a + b) / 2, y + 1.5, text, ha="center", va="center",
                fontsize=7.0, color=INK, zorder=5, linespacing=1.35)

    # The fragment frame starts above the first message it contains, so its
    # label does not sit on top of that message.
    ax.add_patch(FancyBboxPatch(
        (55, 16.8), 32, 15.2, boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=0.9, edgecolor=GREEN, facecolor="none", zorder=0))
    label(ax, 56.0, 32.8, "loop  per stage", fontsize=7.0, color=GREEN,
          ha="left")

    ax.text(50, 74.4, "Figure 4.5  Sequence of one submitted job",
            ha="center", va="center", fontsize=10.2, color=INK)
    finish(fig, "fig-4-5-sequence")


if __name__ == "__main__":
    dfd_level0()
    dfd_level1()
    use_cases()
    class_diagram()
    sequence_diagram()
