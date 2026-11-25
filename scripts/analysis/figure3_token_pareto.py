"""Figure 3 — token cost distribution & cost-quality Pareto across 4 baselines.

Replaces the published Figure 3 (HyMem-Coord vs ReAct only) with a cross-
baseline comparison against memory-aware peers (ReSum, Thread, A-MEM) on
GAIA L1+L2+L3 (pass@3).

  (a) Token-cost distribution per method. Box (Q1/median/Q3 + 1.5-IQR
      whiskers) + per-task strip (jittered). HyMem's distribution is
      left-shifted: at every quartile it uses fewer tokens than ReSum
      and A-MEM, while Thread's lower mass is "saved by failing early"
      (its Pass@1 in Panel b is far worse). Pass@1 sits above each box.

  (b) Cost-quality Pareto — median(total_tokens) vs Pass@1 (mean over 3
      runs). One point per method; horizontal whiskers = IQR, vertical
      whiskers = 95% bootstrap CI on Pass@1.

Layout matches the original Figure 3 — (a) over (b), single column.

Style is copied from figure1_dilution.py so cross-figure readers see the
same HyMem green throughout the paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# -------------------------------------------------------------------- rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
})


# -------------------------------------------------------------------- palette
METHOD_ORDER  = ("hymem", "resum", "thread", "amem")
METHOD_LABEL  = {"hymem": "HyMem", "resum": "ReSum", "thread": "Thread", "amem": "A-MEM"}
METHOD_COLOR  = {
    "hymem":  "#5fa168",  # teal-green, matches Figure 1
    "resum":  "#5a8fbd",  # muted blue
    "thread": "#d4a04a",  # muted amber
    "amem":   "#a868a4",  # muted plum
}
C_AXIS = "#222222"
C_HELP = "#666666"


def save_pub(fig, filename: str, dpi: int = 600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    # LZW-compressed TIFF — keeps file size manageable for journal upload while
    # remaining lossless. Nature-family upload limits typically reject raw
    # uncompressed 600-DPI TIFFs.
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


# ===================================================================== stats

def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (slope, intercept) of OLS y = slope*x + intercept."""
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, n: int = 1000,
                        seed: int = 0) -> Tuple[float, float, float, float]:
    """Return (slope_mean, slope_lo, slope_hi, slope_se) via bootstrap."""
    rng = np.random.RandomState(seed)
    n_obs = len(x)
    slopes = np.empty(n)
    for i in range(n):
        idx = rng.randint(0, n_obs, size=n_obs)
        slopes[i], _ = _ols(x[idx], y[idx])
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return (float(slopes.mean()), float(lo), float(hi), float(slopes.std()))


def _perm_slope_diff_p(x1: np.ndarray, y1: np.ndarray,
                       x2: np.ndarray, y2: np.ndarray,
                       n: int = 2000, seed: int = 0) -> float:
    """Permutation test on slope_1 - slope_2.

    Null: methods share a common slope. We pool (x, y) and randomly relabel
    membership, recompute slope_1 - slope_2, count how often the absolute
    difference meets or exceeds the observed."""
    s1, _ = _ols(x1, y1)
    s2, _ = _ols(x2, y2)
    obs = abs(s1 - s2)
    x = np.concatenate([x1, x2]); y = np.concatenate([y1, y2])
    n1 = len(x1)
    rng = np.random.RandomState(seed)
    extreme = 0
    for _ in range(n):
        perm = rng.permutation(len(x))
        idx1, idx2 = perm[:n1], perm[n1:]
        a, _ = _ols(x[idx1], y[idx1])
        b, _ = _ols(x[idx2], y[idx2])
        if abs(a - b) >= obs:
            extreme += 1
    return (extreme + 1) / (n + 1)


def _pass1_ci(scores: np.ndarray, n: int = 1000, seed: int = 0
              ) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(scores), size=(n, len(scores)))
    means = scores[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(scores.mean()), float(lo), float(hi)


# ===================================================================== plot

def _plot_panel_a(ax, df: pd.DataFrame, stats: Dict):
    """Panel (a): per-task total-token distribution per method.

    Box (Q1/median/Q3 + 1.5×IQR whiskers) + jittered per-task strip;
    Pass@1 number annotated in a fixed-y aligned summary row above all boxes.
    """
    rng = np.random.RandomState(0)
    x_centers = np.arange(len(METHOD_ORDER))

    # Fixed y for the aligned Pass@1 summary row (above all whiskers).
    PASS_ROW_Y = 4.2e6

    for i, m in enumerate(METHOD_ORDER):
        sub = df[df["method"] == m]
        toks = sub["total_tokens"].values.astype(float)
        color = METHOD_COLOR[m]

        # strip (jittered scatter) — lowered alpha so box stays dominant
        jitter = rng.uniform(-0.18, 0.18, size=len(toks))
        ax.scatter(x_centers[i] + jitter, toks, s=3.5, alpha=0.14,
                   color=color, edgecolor="none", rasterized=True, zorder=1)

        # box (drawn on top of strip)
        bp = ax.boxplot([toks], positions=[x_centers[i]],
                        widths=0.55, patch_artist=True,
                        showfliers=False, whis=1.5, zorder=3)
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
        for med in bp["medians"]:
            med.set_color(color); med.set_linewidth(1.6)
        for whisk in bp["whiskers"]:
            whisk.set_color(color); whisk.set_linewidth(0.9)
        for cap in bp["caps"]:
            cap.set_color(color); cap.set_linewidth(0.9)

        # Pass@1 — aligned at fixed y across methods
        acc = sub["score"].mean() * 100
        ax.text(x_centers[i], PASS_ROW_Y, f"{acc:.1f}%",
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=color)

    # Compact "Pass@1" row label outside the plot area (axes-fraction coords)
    # so it doesn't overlap with HyMem's number.
    ax.text(0.0, 1.04, "Pass@1 →", transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=6.0, color=C_HELP, style="italic")

    # axes
    ax.set_xticks(x_centers)
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
    ax.set_ylim(8e3, 7e6)
    ax.set_yscale("log")
    ax.set_ylabel("Total tokens per task (log)", color=C_AXIS)
    ax.set_xlim(-0.6, len(METHOD_ORDER) - 0.4)

    # color the method tick labels to match dots in panel b
    for tick, m in zip(ax.get_xticklabels(), METHOD_ORDER):
        tick.set_color(METHOD_COLOR[m])
        tick.set_fontweight("bold")

    # panel label — Nature 8 pt bold lowercase
    ax.text(-0.13, 1.04, "a", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=C_AXIS, va="top")


def _plot_panel_b(ax, df: pd.DataFrame, stats: Dict):
    """Panel (b): 4 points, median tokens (IQR) × Pass@1 (95% bootstrap CI)."""

    # Non-overlapping label offsets per method. HyMem and ReSum sit near the
    # same y, so HyMem goes directly above and ReSum gets the right margin.
    # Thread and A-MEM are vertically separated: A-MEM gets the upper-right
    # of its dot; Thread heads to the lower-left empty corner.
    # (x_mult, y_add, ha, va)
    OFFSET = {
        "hymem":  (1.00, +6.0, "center", "bottom"),  # directly above HyMem
        "resum":  (1.35, +0.5, "left",   "center"),  # right of ReSum
        "amem":   (1.30, +0.5, "left",   "center"),  # right of A-MEM
        "thread": (0.78, -4.5, "right",  "top"),     # below-left of Thread
    }

    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        toks = sub["total_tokens"].values.astype(float)
        scores = sub["score"].values.astype(float)
        q1, med, q3 = np.percentile(toks, [25, 50, 75])
        acc_mean, acc_lo, acc_hi = _pass1_ci(scores)

        ax.errorbar(med, acc_mean * 100,
                    xerr=[[med - q1], [q3 - med]],
                    yerr=[[(acc_mean - acc_lo) * 100],
                          [(acc_hi - acc_mean) * 100]],
                    fmt="o", color=METHOD_COLOR[m],
                    markersize=9, markeredgecolor="white", markeredgewidth=0.8,
                    elinewidth=1.0, capsize=2.5, capthick=0.8,
                    ecolor=METHOD_COLOR[m], alpha=0.95, zorder=5)

        dx, dy, ha, va = OFFSET[m]
        # compact 2-line label: "Method\n66.7%, 111k"
        ax.annotate(
            f"{METHOD_LABEL[m]}\n{acc_mean*100:.1f}%, {int(med/1000)}k",
            xy=(med, acc_mean * 100),
            xytext=(med * dx, acc_mean * 100 + dy),
            fontsize=6.5, color=METHOD_COLOR[m], ha=ha, va=va,
            fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(2.5e4, 2.5e6)   # extra right room so ReSum/A-MEM labels fit
    ax.set_ylim(15, 85)
    ax.set_xlabel("Median total tokens per task (log)", color=C_AXIS)
    ax.set_ylabel("Pass@1 (%)", color=C_AXIS)

    # subtle "better →" arrow tucked into the empty upper-left corner
    arrow = FancyArrowPatch(
        (5e5, 22), (3.5e4, 78),
        arrowstyle="->,head_length=3,head_width=2",
        color=C_HELP, linewidth=0.6,
        linestyle=(0, (2, 2)),
        alpha=0.45, zorder=1)
    ax.add_patch(arrow)
    ax.text(3.5e4, 81, "better",
            fontsize=6.5, color=C_HELP, ha="left", va="bottom", style="italic")

    # panel label — Nature 8 pt bold lowercase
    ax.text(-0.13, 1.04, "b", transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=C_AXIS, va="top")


def render(df: pd.DataFrame, stem: Path) -> Dict:
    fig = plt.figure(figsize=(3.3, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    stats: Dict = {
        "n_rows": int(len(df)),
        "stat_legend": {
            "n_definition":  "task-runs per method = 127 GAIA L1+L2+L3 tasks × 3 pass@3 runs = 381",
            "panel_a_center":   "median (box midline)",
            "panel_a_spread":   "IQR (box) + 1.5*IQR whisker; jitter strip = all 381 per-task tokens",
            "panel_a_passrate": "mean over 381 task-runs (run-averaged single-attempt accuracy)",
            "panel_b_x_center": "median (across 381 task-runs) of total_tokens",
            "panel_b_x_spread": "IQR (Q1, Q3) horizontal whiskers",
            "panel_b_y_center": "mean Pass@1 over 381 task-runs",
            "panel_b_y_spread": "95% bootstrap CI (1000 resamples on per-task-run scores)",
            "test":          "none — descriptive comparison only",
            "source_data":   "HyMem/data/figure3/per_task.csv",
            "tasks":         "identical 127 tasks across all 4 methods; per-task overlap=100%",
        },
    }

    # Pass@1, IQR, and box stats per method (used by both panels)
    perf: Dict[str, Dict] = {}
    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        toks = sub["total_tokens"].values.astype(float)
        scores = sub["score"].values.astype(float)
        q1, med, q3 = np.percentile(toks, [25, 50, 75])
        whisker_lo = max(np.min(toks),
                         q1 - 1.5 * (q3 - q1))
        whisker_hi = min(np.max(toks),
                         q3 + 1.5 * (q3 - q1))
        a, lo, hi = _pass1_ci(scores)
        perf[m] = dict(pass1=a, pass1_lo=lo, pass1_hi=hi,
                       tokens_q1=float(q1), tokens_median=float(med),
                       tokens_q3=float(q3),
                       tokens_whisker_lo=float(whisker_lo),
                       tokens_whisker_hi=float(whisker_hi),
                       n=int(len(sub)))
    stats["perf"] = perf

    _plot_panel_a(ax_a, df, stats)
    _plot_panel_b(ax_b, df, stats)

    save_pub(fig, str(stem))
    plt.close(fig)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    HYMEM = Path(__file__).resolve().parents[2]
    ap.add_argument("--data", default=str(HYMEM / "data" / "figure3" / "per_task.csv"))
    ap.add_argument("--out",  default=str(HYMEM / "results" / "figures" / "figure3_token_pareto"))
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print(f"[loaded] {len(df)} rows from {args.data}")

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    stats = render(df, out_stem)

    # print stats summary
    print("\n[Panel a — token distribution + Pass@1 per method]")
    for m, p in stats["perf"].items():
        print(f"  {m:<7} n={p['n']:>4}  "
              f"Q1={int(p['tokens_q1']):>9,}  med={int(p['tokens_median']):>9,}  "
              f"Q3={int(p['tokens_q3']):>9,}  whisker_hi={int(p['tokens_whisker_hi']):>9,}  "
              f"Pass@1={p['pass1']*100:5.1f}%")

    print("\n[Panel b — Pareto: median tokens vs Pass@1]")
    for m, p in stats["perf"].items():
        print(f"  {m:<7} median tokens={int(p['tokens_median']):>9,}  "
              f"Pass@1={p['pass1']*100:5.1f}% [{p['pass1_lo']*100:.1f},{p['pass1_hi']*100:.1f}]")

    stats_path = out_stem.parent / "figure3_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[saved figure] {out_stem}.{{svg,pdf,tiff,png}}")
    print(f"[saved stats ] {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
