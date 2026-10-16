"""Figures 1 and 3 for the patent disclosure.

Figure 1 is the system architecture, with the feedback path from completed
jobs into the per-class cost models drawn explicitly, because that loop is
the invention.

Figure 3 is the routing decision, transcribed from server/scheduler.py
rather than from a description of it: the four selection branches and the
three sources a prediction can come from are the ones the code actually has.

Run from the project root:  python ip/make_figures.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = "ip"

INK = "#1b2434"
BLUE = "#1d3f72"
AMBER = "#b08900"
AMBER_FILL = "#fff8e6"
BLUE_FILL = "#eef3fb"
GREY_FILL = "#f3f4f6"
RED = "#9c3647"
RED_FILL = "#fdeced"


def box(ax, x, y, w, h, text, fc=GREY_FILL, ec=INK, fontsize=8.6, bold=False,
        rounding=0.018):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={rounding}",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, zorder=3,
            fontweight="bold" if bold else "normal", linespacing=1.45)


def diamond(ax, cx, cy, w, h, text, fontsize=8.0):
    ax.add_patch(plt.Polygon(
        [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
        closed=True, facecolor=AMBER_FILL, edgecolor=AMBER, linewidth=1.0, zorder=2,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=3, linespacing=1.4)


def arrow(ax, p, q, color=INK, style="-|>", lw=1.1, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=11, linewidth=lw,
        color=color, zorder=1, shrinkA=1.5, shrinkB=1.5,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
    ))


def label(ax, x, y, text, color=INK, fontsize=7.4, ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
            zorder=4, style=style)


# ===================================================================== FIG 1
def figure1():
    fig, ax = plt.subplots(figsize=(12.6, 7.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    box(ax, 2, 47, 17, 7.5,
        "Client\nsubmits a link\nor a file", fc="#ffffff", bold=True)

    box(ax, 23, 47, 20, 7.5,
        "Intake\nread duration from the\nsource's own metadata",
        fc=BLUE_FILL, ec=BLUE)
    label(ax, 33, 44.6, "no download yet", color=BLUE, style="italic")

    box(ax, 47, 47, 19, 7.5,
        "Allowance check\nrefuse before any work\nif the plan does not cover it",
        fc=BLUE_FILL, ec=BLUE)

    # --- the routing decision, the subject of Figure 3 --------------------
    box(ax, 70, 45.5, 28, 10.5,
        "ROUTING DECISION   (Figure 3)\n"
        "for each hardware class: predict total time\n"
        "cost = predicted × class hourly rate\n"
        "choose the cheapest that meets the deadline",
        fc=AMBER_FILL, ec=AMBER, bold=False, fontsize=8.4)

    arrow(ax, (19, 50.7), (23, 50.7))
    arrow(ax, (43, 50.7), (47, 50.7))
    arrow(ax, (66, 50.7), (70, 50.7))

    # --- the fleet --------------------------------------------------------
    label(ax, 50, 39.5, "WORKER FLEET — heterogeneous, each class with its own learned cost model",
          fontsize=8.6, color=INK)

    classes = [("cpu-2core", 6), ("cpu-8core", 36), ("cpu-16core", 66)]
    for name, x in classes:
        box(ax, x, 30, 26, 7.0, f"{name}\nworker process", fc="#ffffff", bold=True)

    arrow(ax, (84, 45.5), (79, 37.0), rad=0.12)
    arrow(ax, (78, 45.5), (49, 37.0), rad=0.10)
    arrow(ax, (74, 45.5), (19, 37.0), rad=0.08)

    # --- the pipeline -----------------------------------------------------
    # M2 and M3 run at the same time, so they are drawn on two rails rather
    # than in a line: a serial row would misdescribe the pipeline.
    box(ax, 4, 19, 11, 7.0, "M1\ningest", fc=GREY_FILL, fontsize=8.0)
    box(ax, 19, 22.8, 13, 6.0, "M2  transcribe", fc=GREY_FILL, fontsize=8.0)
    box(ax, 19, 15.6, 13, 6.0, "M3  vision", fc=GREY_FILL, fontsize=8.0)
    box(ax, 36, 19, 10, 7.0, "M4\nfuse", fc=GREY_FILL, fontsize=8.0)
    box(ax, 50, 19, 14, 7.0, "M5′\nunderstand", fc=GREY_FILL, fontsize=8.0)
    box(ax, 68, 19, 10, 7.0, "M6\nindex", fc=GREY_FILL, fontsize=8.0)
    box(ax, 82, 19, 12, 7.0, "M7\nhighlight", fc=GREY_FILL, fontsize=8.0)

    arrow(ax, (15, 22.5), (19, 25.8), lw=0.9)
    arrow(ax, (15, 22.5), (19, 18.6), lw=0.9)
    arrow(ax, (32, 25.8), (36, 22.5), lw=0.9)
    arrow(ax, (32, 18.6), (36, 22.5), lw=0.9)
    for x0, x1 in ((46, 50), (64, 68), (78, 82)):
        arrow(ax, (x0, 22.5), (x1, 22.5), lw=0.9)
    label(ax, 25.5, 29.8, "concurrent", color=BLUE, style="italic")

    arrow(ax, (49, 30), (41, 26.0), rad=-0.1)
    label(ax, 56, 28.2, "the chosen class runs the pipeline\nat the chosen model fidelity",
          fontsize=7.4, color=INK)

    # --- the feedback loop: the invention ---------------------------------
    box(ax, 22, 5.5, 56, 8.5,
        "ON COMPLETION  —  record_outcome()\n"
        "update the class's five running sums (n, Σx, Σy, Σxx, Σxy)   →   "
        "affine fit  total = fixed + rate × minutes\n"
        "store predicted vs actual vs baseline for this job",
        fc=AMBER_FILL, ec=AMBER, fontsize=8.3)

    arrow(ax, (57, 19), (57, 14.0), color=AMBER, lw=1.5)

    # the loop back into routing
    arrow(ax, (78, 9.75), (94, 9.75), color=AMBER, lw=1.5)
    arrow(ax, (94, 9.75), (94, 45.5), color=AMBER, lw=1.5)
    label(ax, 92, 11.9, "the learned cost model feeds the next decision",
          color=AMBER, fontsize=7.8, ha="right")

    box(ax, 2, 5.5, 17, 8.5,
        "A run that reused\ncached stages is\nnot learned from",
        fc=RED_FILL, ec=RED, fontsize=7.8)
    arrow(ax, (22, 9.75), (19, 9.75), color=RED, lw=1.0, dashed=True)

    ax.text(50, 58.2,
            "Figure 1 — System architecture. The highlighted path is the invention: every "
            "completed job updates its\nhardware class's cost model, and that model decides where "
            "the next job runs.",
            ha="center", va="center", fontsize=10.0, color=INK, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure1-architecture.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR}/figure1-architecture.png")


# ===================================================================== FIG 3
def figure3():
    """The routing decision.

    Laid out on a single spine with every outcome on one right-hand column,
    so the path a job takes reads top to bottom and the four ways it can end
    line up for comparison. The earlier version spread small boxes over a
    large canvas, which left the type too small to read at print size.
    """
    fig, ax = plt.subplots(figsize=(11.0, 12.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 118)
    ax.axis("off")

    SPINE = 38          # centre of the decision column
    OUT_X, OUT_W = 66, 33   # the outcome column
    GREEN, GREEN_FILL = "#2f7a46", "#eaf6ee"

    # ---------------- phase 1: predict -----------------------------------
    ax.add_patch(FancyBboxPatch(
        (2, 78.5), 96, 34, boxstyle="round,pad=0.004,rounding_size=0.015",
        linewidth=1.0, edgecolor="#b9c2cf", facecolor="#fafbfc", zorder=0))
    ax.text(4.5, 109.4, "1  PREDICT", fontsize=9.5, color=BLUE,
            fontweight="bold", va="center")

    box(ax, 26, 103, 48, 7.0,
        "A job arrives: input length and deadline are known",
        fc="#ffffff", bold=True, fontsize=9.2)

    box(ax, 18, 93.5, 64, 6.6,
        "For each hardware class, predict the total execution time",
        fc=BLUE_FILL, ec=BLUE, bold=True, fontsize=9.2)
    arrow(ax, (50, 103), (50, 100.1))

    srcs = [
        (5, "Affine fit is credible\n\n\u2265 4 samples, and the\nlengths seen differ by \u2265 2\u00d7\n\n"
            "fixed + rate \u00d7 minutes", BLUE_FILL, BLUE),
        (36, "Only a coarse ratio\n\n\n\n\nper-minute \u00d7 minutes", GREY_FILL, "#6b7280"),
        (67, "Nothing measured yet\n\n\n\n\nseed \u00d7 minutes", RED_FILL, RED),
    ]
    for x, text, fc, ec in srcs:
        box(ax, x, 80.5, 28, 11.5, text, fc=fc, ec=ec, fontsize=8.2)
    for x in (19, 50, 81):
        arrow(ax, (50, 93.5), (x, 92.0), lw=0.9)

    box(ax, 18, 70.5, 64, 6.0,
        "cost  =  predicted seconds  \u00d7  the class\u2019s hourly rate",
        fc=BLUE_FILL, ec=BLUE, fontsize=9.2)
    for x in (19, 50, 81):
        arrow(ax, (x, 80.5), (50, 76.5), lw=0.9, rad=0.0)

    # ---------------- phase 2: choose ------------------------------------
    ax.text(4.5, 64.5, "2  CHOOSE", fontsize=9.5, color=AMBER,
            fontweight="bold", va="center")
    arrow(ax, (50, 70.5), (SPINE, 62.4), lw=1.1, rad=-0.08)

    steps = [
        dict(y=56, q="Is any class\nunmeasured?",
             out="PROBE IT\ncheapest unmeasured class",
             why="a seed is too crude to rule\na class in or out",
             fc=AMBER_FILL, ec=AMBER),
        dict(y=41, q="Of those that meet the deadline,\nis any still thinly measured?\n(fewer than 5 samples)",
             out="PROBE IT\nfewest samples, then cheapest",
             why="exploring only where it cannot\nbreak the deadline",
             fc=AMBER_FILL, ec=AMBER),
        dict(y=26, q="Does any class\nmeet the deadline?",
             out="TAKE THE CHEAPEST OF THEM",
             why="not the fastest \u2014 headroom bought\nby a bigger machine is paid for\nand never used",
             fc=GREEN_FILL, ec=GREEN),
    ]

    for i, s in enumerate(steps):
        y = s["y"]
        diamond(ax, SPINE, y, 44, 12.5, s["q"], fontsize=8.4)

        # outcome, on the shared right-hand column
        h = 9.5
        box(ax, OUT_X, y - h / 2, OUT_W, h,
            s["out"] + "\n\n" + s["why"], fc=s["fc"], ec=s["ec"], fontsize=8.0)
        arrow(ax, (SPINE + 22, y), (OUT_X, y), color=s["ec"], lw=1.2)
        label(ax, (SPINE + 22 + OUT_X) / 2, y + 1.9, "yes", color=s["ec"], fontsize=8.0)

        # fall through to the next decision
        if i < len(steps) - 1:
            arrow(ax, (SPINE, y - 6.25), (SPINE, steps[i + 1]["y"] + 6.25), lw=1.1)
            label(ax, SPINE + 2.6, y - 9.2, "no", fontsize=8.0, ha="left")

    # the last "no"
    arrow(ax, (SPINE, 26 - 6.25), (SPINE, 14.5), color=RED, lw=1.2)
    label(ax, SPINE + 2.6, 17.4, "no", color=RED, fontsize=8.0, ha="left")

    box(ax, 10, 5.0, 52, 9.5,
        "THE DEADLINE IS ALREADY LOST\n\n"
        "take the fastest class, and report the expected\noverrun before the job starts",
        fc=RED_FILL, ec=RED, fontsize=8.2)

    box(ax, OUT_X, 5.0, OUT_W, 9.5,
        "Attach the execution plan computed\nwith that class\u2019s own cost model,\nand enqueue the job",
        fc="#ffffff", fontsize=8.2)
    arrow(ax, (OUT_X + OUT_W / 2, 21.25), (OUT_X + OUT_W / 2, 14.5), color=GREEN, lw=1.2)
    arrow(ax, (62, 9.75), (OUT_X, 9.75), color=RED, lw=1.0, dashed=True)

    ax.text(50, 116,
            "Figure 3 \u2014 The routing decision, as implemented. Exploration never overrides the "
            "deadline;\namong the classes that keep it, the cheapest is taken rather than the fastest.",
            ha="center", va="center", fontsize=10.2, color=INK, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure3-routing.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR}/figure3-routing.png")



if __name__ == "__main__":
    figure1()
    figure3()
