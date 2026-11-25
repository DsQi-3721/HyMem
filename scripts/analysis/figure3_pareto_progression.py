"""Figure 3 — three-panel cost/quality + context progression.

  (a) Pass@1 vs median total tokens (k); 5 method dots + Pareto-style connecting line.
  (b) Pass@1 vs median tool_calls_atomic; same five dots, connecting line.
  (c) Per-turn cumulative context tokens (log y) vs normalized task progress;
      4 curves (HyMem-Coord / ReSum / A-MEM / ReAct) with 95% bootstrap CI shading.

Layout: 1x3 horizontal, single double-column-width figure (~7.2 x 2.6 in).
Style: copied from figure1_dilution.py to keep HyMem's green identity
consistent across the manuscript.
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
from matplotlib.lines import Line2D


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
METHOD_ORDER_AB = ("hymem", "resum", "thread", "amem", "react", "summary")
METHOD_ORDER_C  = ("hymem", "hymem_total", "resum", "amem", "react")
METHOD_LABEL = {"hymem":   "HyMem",   "resum":   "ReSum", "thread": "Thread",
                "amem":    "A-MEM",   "react":   "ReAct", "summary": "Summary"}
METHOD_COLOR = {
    "hymem":        "#5fa168",
    "hymem_total":  "#2c5e36",   # darker green — HyMem family
    "resum":        "#5a8fbd",
    "thread":       "#d4a04a",
    "amem":         "#a868a4",
    "react":        "#c45a4f",
    "summary":      "#c97aab",   # rose pink — distinct from existing 5
}
# Distinct marker shape per method so panel (a)/(b) remain readable
# in grayscale print and to color-blind viewers.
METHOD_MARKER = {
    "hymem":   "o",   # circle  — HyMem (the protagonist)
    "resum":   "s",   # square  — ReSum
    "thread":  "^",   # triangle up — Thread
    "amem":    "D",   # diamond — A-MEM
    "react":   "P",   # filled plus — ReAct
    "summary": "X",   # filled X — Summary
}
PANEL_C_LABEL = {"hymem":       "HyMem-Planner",
                 "hymem_total": "HyMem-Total",
                 "resum":       "ReSum",
                 "amem":        "A-MEM",
                 "react":       "ReAct"}
PANEL_C_STYLE = {"hymem":       {"linestyle": "-",  "linewidth": 1.5},
                 "hymem_total": {"linestyle": "--", "linewidth": 1.4},
                 "resum":       {"linestyle": "-",  "linewidth": 1.4},
                 "amem":        {"linestyle": "-",  "linewidth": 1.4},
                 "react":       {"linestyle": "-",  "linewidth": 1.4}}
C_AXIS = "#222222"
C_HELP = "#666666"
C_LINE = "#888888"


def save_pub(fig, filename: str, dpi: int = 600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


# ===================================================================== stats

def _bootstrap_mean_ci(values: np.ndarray, n: int = 1000, seed: int = 0
                       ) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


# ====================================================== panel (a) and (b)

def _per_method_summary(df: pd.DataFrame, x_col: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for m in METHOD_ORDER_AB:
        sub = df[df["method"] == m]
        if sub.empty: continue
        xs = sub[x_col].values.astype(float)
        q1, med, q3 = np.percentile(xs, [25, 50, 75])
        acc_mean, acc_lo, acc_hi = _bootstrap_mean_ci(sub["score"].values.astype(float))
        out[m] = dict(x_q1=float(q1), x_median=float(med), x_q3=float(q3),
                      acc_mean=acc_mean, acc_lo=acc_lo, acc_hi=acc_hi,
                      n=int(len(sub)))
    return out


def _plot_pareto_panel(ax, df: pd.DataFrame, x_col: str, x_scale: float,
                       x_label: str, panel_letter: str) -> Dict[str, Dict]:
    """Clean dot scatter per method — no connecting line, no error bars.
    Marker shape + colour both encode method so the figure stays readable
    in grayscale or for colour-blind readers."""
    summary = _per_method_summary(df, x_col)

    # per-method dot only (no error bars, no Pareto line)
    for m, v in summary.items():
        x = v["x_median"] / x_scale
        y = v["acc_mean"] * 100
        ax.scatter(x, y,
                   marker=METHOD_MARKER[m],
                   color=METHOD_COLOR[m],
                   s=70, edgecolor="white", linewidth=0.7,
                   alpha=0.97, zorder=5)

    # Per-method label placement using offset in POINTS (scale-invariant
    # across the two panels — keeps Thread off the y-axis no matter what
    # the x range is).
    label_offsets = _choose_label_offsets_points()
    for m, v in summary.items():
        x = v["x_median"] / x_scale
        y = v["acc_mean"] * 100
        dx_pt, dy_pt, ha, va = label_offsets[m]
        ax.annotate(METHOD_LABEL[m],
                    xy=(x, y), xytext=(dx_pt, dy_pt),
                    textcoords="offset points",
                    fontsize=6.5, color=METHOD_COLOR[m],
                    ha=ha, va=va, fontweight="bold", zorder=6)

    ax.set_xlabel(x_label, color=C_AXIS)
    ax.set_ylabel("Pass@1 (%)", color=C_AXIS)
    ax.set_ylim(20, 80)

#    ax.text(-0.18, 1.04, panel_letter, transform=ax.transAxes,
#            fontsize=8, fontweight="bold", color=C_AXIS, va="top")
    return summary


def _choose_label_offsets_points(
    ) -> Dict[str, Tuple[float, float, str, str]]:
    """Per-method label offsets in MATPLOTLIB POINTS (1 pt = 1/72 in).
    Scale-invariant across the two panels — Thread stays clear of the
    y-axis regardless of x-axis units.  (dx_pt, dy_pt, ha, va)"""
    return {
        # ABOVE the marker, centred
        "hymem":   ( 0,  10, "center", "bottom"),
        # ABOVE the marker, centred
        "resum":   ( 0,  10, "center", "bottom"),
        # BELOW the marker, centred
        "amem":    ( 0, -10, "center", "top"),
        # ABOVE the marker, centred
        "react":   ( 0,  10, "center", "bottom"),
        # RIGHT of the marker (avoids y-axis collision when x is small)
        "thread":  (10,   0, "left",   "center"),
        # BELOW the marker, centred (Summary lands between ReAct/A-MEM in y;
        # below keeps it clear of ReAct's label above)
        "summary": ( 0, -10, "center", "top"),
    }


# =============================================================== panel (c)

def _plot_progression_panel(ax, cdf: pd.DataFrame, panel_letter: str
                            ) -> Dict[str, Dict]:
    """Per-method context-growth curves with IQR shading.

    Uses **median** + IQR (Q1–Q3) — the same robust statistic Panel (a)
    uses for the X axis, so HyMem-Total's end value here matches the
    HyMem dot in Panel (a). Long-tail HyMem outliers can shift the mean
    by 2-3×, which is why an earlier mean-based version of this panel
    showed an inflated HyMem-Total relative to Panel (a).

    Each curve is normalized to its own starting median so all methods
    begin at 0 — the comparison is about GROWTH, not the fixed system-
    prompt overhead each baseline carries.
    """
    summary: Dict[str, Dict] = {}
    progress_grid = np.sort(cdf["progress"].unique())

    for m in METHOD_ORDER_C:
        sub = cdf[cdf["method"] == m]
        if sub.empty: continue
        # aggregate per progress point — median + IQR
        med, q1, q3 = [], [], []
        for p in progress_grid:
            vals = sub[sub["progress"] == p]["context_tokens"].values.astype(float)
            if len(vals) < 3:
                med.append(np.nan); q1.append(np.nan); q3.append(np.nan); continue
            med.append(float(np.median(vals)))
            q1.append(float(np.percentile(vals, 25)))
            q3.append(float(np.percentile(vals, 75)))
        med = np.array(med); q1 = np.array(q1); q3 = np.array(q3)

        # Normalize: subtract starting median so all curves begin at 0.
        start = med[0] if np.isfinite(med[0]) else 0.0
        med_d = med - start
        q1_d  = q1  - start
        q3_d  = q3  - start

        style = PANEL_C_STYLE.get(m, {})
        ax.plot(progress_grid * 100, np.maximum(med_d, 1),
                color=METHOD_COLOR[m],
                linewidth=style.get("linewidth", 1.4),
                linestyle=style.get("linestyle", "-"),
                alpha=0.95, label=PANEL_C_LABEL[m], zorder=5)
        ax.fill_between(progress_grid * 100,
                        np.maximum(q1_d, 1), np.maximum(q3_d, 1),
                        color=METHOD_COLOR[m], alpha=0.10, linewidth=0,
                        zorder=3)

        summary[m] = dict(start_raw=float(start),
                          end_raw_median=float(med[-1]),
                          end_growth_median=float(med_d[-1]),
                          end_growth_q1=float(q1_d[-1]),
                          end_growth_q3=float(q3_d[-1]))

    ax.set_yscale("log")
    ax.set_ylim(1e1, 3e5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task progress (%)", color=C_AXIS)
    ax.set_ylabel("Context growth from start (tokens, log)", color=C_AXIS)

    # Legend in the bottom-right empty corner — does not overlap any curve
    # because all curves are monotonically increasing (highest at right edge,
    # left-bottom region is empty).
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc="lower right",
                    fontsize=6.0, handlelength=1.7, labelspacing=0.25,
                    borderpad=0.3, borderaxespad=0.4,
                    frameon=True, framealpha=0.92,
                    edgecolor="none", facecolor="white")
    leg.set_zorder(10)

    #ax.text(-0.18, 1.04, panel_letter, transform=ax.transAxes,
    #        fontsize=8, fontweight="bold", color=C_AXIS, va="top")
    return summary


# ================================================================== driver

def render(df: pd.DataFrame, cdf: pd.DataFrame, stem: Path) -> Dict:
    # --- combined 1x3 figure -----------------------------------------------
    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15],
                          wspace=0.40, left=0.07, right=0.985,
                          bottom=0.20, top=0.92)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    sum_a = _plot_pareto_panel(ax_a, df, x_col="total_tokens",
                                x_scale=1000.0,
                                x_label="Token consumption (k)",
                                panel_letter="a")
    sum_b = _plot_pareto_panel(ax_b, df, x_col="tool_calls_atomic",
                                x_scale=1.0,
                                x_label="Number of tool calls",
                                panel_letter="b")
    sum_c = _plot_progression_panel(ax_c, cdf, panel_letter="c")

    save_pub(fig, str(stem))
    plt.close(fig)

    # --- standalone single-panel figures ----------------------------------
    out_dir = stem.parent

    fig_a, ax = plt.subplots(figsize=(2.8, 2.5))
    _plot_pareto_panel(ax, df, x_col="total_tokens", x_scale=1000.0,
                       x_label="Token consumption (k)", panel_letter="a")
    fig_a.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.92)
    save_pub(fig_a, str(out_dir / "figure3a_token_pareto"))
    plt.close(fig_a)

    fig_b, ax = plt.subplots(figsize=(2.8, 2.5))
    _plot_pareto_panel(ax, df, x_col="tool_calls_atomic", x_scale=1.0,
                       x_label="Number of tool calls", panel_letter="b")
    fig_b.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.92)
    save_pub(fig_b, str(out_dir / "figure3b_toolcalls_pareto"))
    plt.close(fig_b)

    fig_c, ax = plt.subplots(figsize=(3.4, 2.5))
    _plot_progression_panel(ax, cdf, panel_letter="c")
    fig_c.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.92)
    save_pub(fig_c, str(out_dir / "figure3c_context_progression"))
    plt.close(fig_c)

    return {
        "panel_a_token_pareto": sum_a,
        "panel_b_toolcalls_pareto": sum_b,
        "panel_c_progression_end": sum_c,
        "stat_legend": {
            "n_definition": "Per-task-runs counts; pass@3 methods: 3 runs × 127 tasks = 381; ReAct: single run × 127 = 127",
            "panel_a_x": "median(total_tokens) per method across all task-runs; horizontal whisker = IQR",
            "panel_a_y": "mean(Pass@1) per method; vertical whisker = 95% bootstrap CI",
            "panel_b_x": "median(tool_calls_atomic); IQR whisker",
            "panel_b_y": "mean(Pass@1); 95% bootstrap CI whisker",
            "panel_c_curves": "per-turn cumulative context tokens, reconstructed by tokenizing message prefixes with Qwen2.5-7B-Instruct; mean ± 95% bootstrap CI across tasks per method, interpolated to 21 grid points on [0, 1] task-progress.",
            "panel_c_methods": "HyMem-Coord uses Coordination-Layer messages (role=planner); others use the entire chat stream (their single-tier baseline). Thread omitted (sub-thread state not in main trace).",
            "react_note": "ReAct = fig1_pilot/react/ run (eval_results.metadata.framework=react); single run only — Pass@1 not run-averaged.",
            "source_data": "HyMem/data/figure3/per_task.csv, HyMem/data/figure3/context_progression.csv",
            "test": "none — descriptive comparison only.",
        },
    }


def main() -> int:
    HYMEM = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default=str(HYMEM / "data" / "figure3" / "per_task.csv"))
    ap.add_argument("--progression",
                    default=str(HYMEM / "data" / "figure3" / "context_progression.csv"))
    ap.add_argument("--out",   default=str(HYMEM / "results" / "figures" / "figure3_pareto_progression"))
    args = ap.parse_args()

    df  = pd.read_csv(args.data)
    cdf = pd.read_csv(args.progression)
    print(f"[loaded] per_task: {len(df)} rows; progression: {len(cdf)} rows")

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    stats = render(df, cdf, out_stem)

    print("\n[Panel a — Token cost (median k) × Pass@1]")
    for m, v in stats["panel_a_token_pareto"].items():
        print(f"  {m:<7} x={v['x_median']/1000:>6.1f}k  Pass@1={v['acc_mean']*100:5.1f}% "
              f"[{v['acc_lo']*100:.1f},{v['acc_hi']*100:.1f}]  n={v['n']}")

    print("\n[Panel b — Tool calls (median) × Pass@1]")
    for m, v in stats["panel_b_toolcalls_pareto"].items():
        print(f"  {m:<7} x={v['x_median']:>5.1f}  Pass@1={v['acc_mean']*100:5.1f}%")

    print("\n[Panel c — Context growth from start (end-of-task, median + IQR)]")
    for m, v in stats["panel_c_progression_end"].items():
        print(f"  {m:<12} start_raw={int(v['start_raw']):>6,}  "
              f"end_raw_median={int(v['end_raw_median']):>8,}  "
              f"growth_median={int(v['end_growth_median']):>8,}  "
              f"IQR=[{int(v['end_growth_q1']):,},{int(v['end_growth_q3']):,}]")

    stats_path = out_stem.parent / "figure3_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n[saved figure] {out_stem}.{{svg,pdf,tiff,png}}")
    print(f"[saved stats ] {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
