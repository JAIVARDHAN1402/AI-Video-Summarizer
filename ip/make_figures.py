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
    fig, ax = plt.subplots(figsize=(11.0, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 116)
    ax.axis("off")

    box(ax, 30, 97, 40, 6.5,
        "Job arrives: duration known, deadline known", fc="#ffffff", bold=True)

    box(ax, 22, 88, 56, 7.5,
        "FOR EACH HARDWARE CLASS — predict the total execution time",
        fc=BLUE_FILL, ec=BLUE, bold=True)
    arrow(ax, (50, 97), (50, 95.5))

    # three sources of a prediction
    srcs = [
        (2, "Affine fit is credible\n≥ 4 samples and\nlengths spread ≥ 2×\n\n"
            "fixed + rate × minutes", BLUE_FILL, BLUE),
        (35, "Only a coarse ratio\nis available\n\nper-minute × minutes", GREY_FILL, INK),
        (68, "Nothing measured yet\n\nseed × minutes\n(samples = 0)", RED_FILL, RED),
    ]
    for x, text, fc, ec in srcs:
        box(ax, x, 72, 30, 13, text, fc=fc, ec=ec, fontsize=7.8)
    for x in (17, 50, 83):
        arrow(ax, (50, 88), (x, 85), lw=0.9, rad=0.0)

    box(ax, 22, 62, 56, 7.0,
        "cost = predicted seconds × the class's hourly rate",
        fc=BLUE_FILL, ec=BLUE)
    for x in (17, 50, 83):
        arrow(ax, (x, 72), (x if x == 50 else 50, 69), lw=0.9)

    # --- the four branches, in the order the code tries them --------------
    diamond(ax, 50, 53, 46, 11,
            "Any class with no measurements at all?")
    arrow(ax, (50, 62), (50, 58.5))

    box(ax, 72, 48.5, 27, 9,
        "PROBE IT\ncheapest unmeasured class\n\na seed is too crude to\nrule a class in or out",
        fc=AMBER_FILL, ec=AMBER, fontsize=7.6)
    arrow(ax, (73, 53), (72, 53), color=AMBER)
    label(ax, 69.0, 55.2, "yes", color=AMBER, fontsize=7.6)

    diamond(ax, 50, 38, 46, 11,
            "Any class that meets the deadline\nis still thinly measured (< 5 samples)?")
    arrow(ax, (50, 47.5), (50, 43.5))
    label(ax, 52.6, 45.6, "no", fontsize=7.6, ha="left")

    box(ax, 72, 33.5, 27, 9,
        "PROBE IT\nfewest samples, then cheapest\n\nexploring only where it\ncannot break the deadline",
        fc=AMBER_FILL, ec=AMBER, fontsize=7.6)
    arrow(ax, (73, 38), (72, 38), color=AMBER)
    label(ax, 69.0, 40.2, "yes", color=AMBER, fontsize=7.6)

    diamond(ax, 50, 23, 46, 11,
            "Any class meets the deadline?")
    arrow(ax, (50, 32.5), (50, 28.5))
    label(ax, 52.6, 30.6, "no", fontsize=7.6, ha="left")

    box(ax, 66, 18.5, 33, 9,
        "CHEAPEST THAT MEETS IT\n\nnot the fastest: headroom bought\nby a bigger machine is paid for\nand never used",
        fc="#eaf6ee", ec="#2f7a46", fontsize=7.6)
    arrow(ax, (73, 23), (66, 23), color="#2f7a46")
    label(ax, 63.0, 25.2, "yes", color="#2f7a46", fontsize=7.6)

    box(ax, 18, 6, 44, 10,
        "DEADLINE IS ALREADY LOST\n\nchoose the fastest class and report the\nexpected overrun in advance, rather than\n"
        "accepting the job and missing it silently",
        fc=RED_FILL, ec=RED, fontsize=7.8)
    arrow(ax, (50, 17.5), (40, 16), color=RED)
    label(ax, 46, 17.6, "no", color=RED, fontsize=7.6, ha="left")

    box(ax, 66, 6, 33, 8,
        "Attach the execution plan\ncomputed with that class's own\ncost model, and enqueue",
        fc="#ffffff", fontsize=7.8)
    arrow(ax, (82, 18.5), (82, 14))

    ax.text(50, 111,
            "Figure 3 — The routing decision, as implemented. Exploration never overrides the "
            "deadline;\namong the classes that keep it, the cheapest is taken rather than the fastest.",
            ha="center", va="center", fontsize=10.0, color=INK, linespacing=1.5)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/figure3-routing.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT_DIR}/figure3-routing.png")


if __name__ == "__main__":
    figure1()
    figure3()
