"""Figure 2 for the patent disclosure: why the cost model needs an intercept.

Each panel is one hardware class, because the model is per class and pooling
them would hide the very structure being argued for. Every point is a
recorded job; runs that resumed from cached work are excluded and said so,
since they did only part of the work.
"""

import matplotlib
matplotlib.use("Agg")

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from server import db

OUT = r"C:/Users/jaiva/OneDrive/Desktop/Project-1/Project_Build/ip/figure2-cost-model.png"

# A full run on this hardware costs at least ~13 s per video-minute; the
# resumed ones sit between 2.6 and 9.7. The separation is clean and stated.
FULL_RUN_FLOOR = 10.0
SHORT_MIN = 5.0          # what the fleet had seen when the problem appeared

session = db.SessionLocal()
rows = session.query(db.SlaObservation).all()

by_class = defaultdict(list)
excluded = 0
for r in rows:
    minutes = r.duration_sec / 60.0
    if r.actual_sec / minutes < FULL_RUN_FLOOR:
        excluded += 1
        continue
    by_class[r.worker_class].append((minutes, r.actual_sec))

panels = [(wc, sorted(p)) for wc, p in by_class.items() if len(p) >= 4]
panels.sort(key=lambda kv: -len(kv[1]))

fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.9))
if len(panels) == 1:
    axes = [axes]

for ax, (wc, pts) in zip(axes, panels):
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])

    A = np.vstack([np.ones_like(x), x]).T
    fixed, rate = np.linalg.lstsq(A, y, rcond=None)[0]

    short = x <= SHORT_MIN
    const = float(y[short].sum() / x[short].sum()) if short.sum() >= 2 else None

    grid = np.linspace(0, x.max() * 1.08, 400)

    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    if const is not None:
        ax.plot(grid, const * grid, "--", color="#d1495b", linewidth=1.9,
                label=f"Constant rate from videos $\\leq$ {SHORT_MIN:.0f} min\n"
                      f"{const:.1f} s per video-min, no intercept")
    ax.plot(grid, fixed + rate * grid, "-", color="#1d3f72", linewidth=2.1,
            label=f"Affine fit, all points\ntotal = {fixed:.0f} s + {rate:.1f} s "
                  f"$\\times$ minutes")
    ax.scatter(x, y, s=46, color="#f0a500", edgecolor="#6b4a00",
               linewidth=0.7, zorder=5, label=f"Recorded jobs (n = {len(x)})")

    if short.sum() >= 2:
        ax.axvspan(0, SHORT_MIN, color="#d1495b", alpha=0.055, zorder=0)
        ax.text(SHORT_MIN * 0.5, ax.get_ylim()[1] * 0.045,
                "fitted\nhere", ha="center", fontsize=7.6, color="#9c3647")

    ax.set_xlabel("Input video length (minutes)")
    ax.set_ylabel("Measured total execution time (seconds)")
    ax.set_title(f"Hardware class: {wc}   (n = {len(x)})", fontsize=11, pad=8)
    ax.legend(fontsize=8.0, loc="upper left", framealpha=0.95)

    # State the divergence at the longest video this class actually ran, so
    # the claim rests on measurement and not on extrapolation.
    if const is not None:
        longest = float(x.max())
        measured = float(y[np.argmax(x)])
        p_const = const * longest
        p_affine = fixed + rate * longest
        ax.annotate(
            f"At {longest:.1f} min (longest measured):\n"
            f"  measured          {measured:6.0f} s\n"
            f"  affine predicts   {p_affine:6.0f} s   ({abs(p_affine-measured)/measured*100:.0f}% out)\n"
            f"  constant predicts {p_const:6.0f} s   ({abs(p_const-measured)/measured*100:.0f}% out)",
            xy=(longest, measured),
            xytext=(0.42, 0.055), textcoords="axes fraction",
            fontsize=8.0, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff8e6", ec="#b08900", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#b08900", lw=1.1),
        )

fig.suptitle(
    "Figure 2 \u2014 Measured execution cost against input length, per hardware class, "
    "with the fitted affine model\nand a constant-rate model fitted only on short inputs",
    fontsize=10.6, y=0.975,
)
caption = "\n".join([
    "Every point is a recorded job. A run pays a fixed cost before it processes anything, so dividing",
    "total time by length folds that cost into the rate, and the rate then depends on how long the",
    f"observed videos happened to be. {excluded} further jobs that resumed from cached work are",
    "excluded: they did only part of the work, so their wall-clock describes no throughput.",
])
fig.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=8.2,
         color="#555", linespacing=1.6)
fig.tight_layout(rect=[0.012, 0.215, 0.988, 0.925])
fig.savefig(OUT, dpi=200)

print(f"wrote {OUT}")
print(f"  excluded as resumed: {excluded}")
for wc, pts in panels:
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    A = np.vstack([np.ones_like(x), x]).T
    fixed, rate = np.linalg.lstsq(A, y, rcond=None)[0]
    short = x <= SHORT_MIN
    print(f"  {wc}: n={len(x)}  affine = {fixed:.1f}s + {rate:.2f}/min", end="")
    if short.sum() >= 2:
        const = y[short].sum() / x[short].sum()
        longest = x.max(); measured = y[np.argmax(x)]
        print(f"  | const {const:.1f}/min -> at {longest:.1f}min "
              f"const {const*longest:.0f}s vs affine {fixed+rate*longest:.0f}s "
              f"vs measured {measured:.0f}s")
    else:
        print()
