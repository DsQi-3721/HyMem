"""Figure 5 — cross-benchmark Pareto: GAIA + BrowseComp-Plus on one figure.

Two panels (1x2):
  (a) Pass@1 vs median total tokens (k)        — 6 methods x 2 benchmarks = 12 dots
  (b) Pass@1 vs median tool_calls_atomic (log) — same 12 dots

Encoding so that benchmark is read at a glance:
  * MARKER SHAPE   = method            (consistent with Figures 3 and 4)
  * SOLID FILL     = GAIA              (the established harder-reasoning benchmark)
  * HOLLOW MARKER  = BrowseComp-Plus   (browsing-heavy benchmark)
  * Dashed line connecting the two same-method dots — the "method trajectory"
    from GAIA to BCP+, in the method's colour.

Benchmark legend:
  * Filled marker = GAIA
  * Hollow marker = BrowseComp+
  * Framed white legend box prevents confusion with in-plot markers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

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
    "legend.frameon": True,
})


# -------------------------------------------------------------------- palette
METHOD_ORDER = ("hymem", "resum", "thread", "amem", "react", "summary")

METHOD_LABEL = {
    "hymem": "HyMem",
    "resum": "ReSum",
    "thread": "Thread",
    "amem": "A-MEM",
    "react": "ReAct",
    "summary": "Summary",
}

METHOD_COLOR = {
    "hymem": "#5fa168",
    "resum": "#5a8fbd",
    "thread": "#d4a04a",
    "amem": "#a868a4",
    "react": "#c45a4f",
    "summary": "#c97aab",
}

METHOD_MARKER = {
    "hymem": "o",
    "resum": "s",
    "thread": "^",
    "amem": "D",
    "react": "P",
    "summary": "X",
}

C_AXIS = "#222222"
C_HELP = "#666666"


def save_pub(fig, filename: str, dpi: int = 600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(
        f"{filename}.tiff",
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


# ===================================================================== stats

def _bootstrap_mean_ci(
    values: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)

    if len(values) == 0:
        return 0.0, 0.0, 0.0

    idx = rng.randint(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])

    return float(values.mean()), float(lo), float(hi)


def _per_method_summary(df: pd.DataFrame, x_col: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}

    for m in METHOD_ORDER:
        sub = df[df["method"] == m]
        if sub.empty:
            continue

        xs = sub[x_col].values.astype(float)
        med = float(np.median(xs))

        acc_mean, acc_lo, acc_hi = _bootstrap_mean_ci(
            sub["score"].values.astype(float)
        )

        out[m] = {
            "x_median": med,
            "acc_mean": acc_mean,
            "acc_lo": acc_lo,
            "acc_hi": acc_hi,
            "n": int(len(sub)),
        }

    return out


# ===================================================================== plot

def _draw_panel(
    ax,
    df_gaia: pd.DataFrame,
    df_bcplus: pd.DataFrame,
    x_col: str,
    x_scale: float,
    x_label: str,
    panel_letter: str,
    log_x: bool,
    panel_key: str = "a",
) -> Dict[str, Dict]:
    """Draw 6 methods × 2 benchmarks with trajectory lines."""

    sum_g = _per_method_summary(df_gaia, x_col)
    sum_b = _per_method_summary(df_bcplus, x_col)

    # Label offsets anchored to the BrowseComp+ hollow marker.
    # Format: dx_pt, dy_pt, horizontal alignment, vertical alignment
    OFFSETS = {
        "a": {
            "hymem": (-50, 10, "left", "bottom"),
            "react": (10, 4, "left", "bottom"),
            "resum": (-2, 20, "right", "top"),
            "summary": (22, -1, "left", "bottom"),
            "amem": (30, 4, "right", "top"),
            "thread": (-12, 15, "left", "top"),
        },
        "b": {
            "hymem": (-10, 8, "right", "bottom"),
            "react": (0, 4, "left", "bottom"),
            "resum": (-18, 15, "left", "center"),
            "summary": (30, 13, "left", "center"),
            "amem": (27, -1, "right", "top"),
            "thread": (-10, 15, "left", "top"),
        },
    }

    panel_offsets = OFFSETS.get(panel_key, OFFSETS["a"])

    out: Dict[str, Dict] = {}

    for m in METHOD_ORDER:
        if m not in sum_g or m not in sum_b:
            continue

        g = sum_g[m]
        b = sum_b[m]

        xg = g["x_median"] / x_scale
        yg = g["acc_mean"] * 100

        xb = b["x_median"] / x_scale
        yb = b["acc_mean"] * 100

        col = METHOD_COLOR[m]
        mk = METHOD_MARKER[m]

        # Trajectory line: GAIA → BrowseComp+
        # Disabled so the two benchmark markers are displayed without a
        # dashed connection.
        # ax.plot(
        #     [xg, xb],
        #     [yg, yb],
        #     color=col,
        #     linewidth=0.9,
        #     linestyle=(0, (3, 2)),
        #     alpha=0.55,
        #     zorder=2,
        # )

        # GAIA marker — filled
        ax.scatter(
            xg,
            yg,
            marker=mk,
            color=col,
            s=70,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.97,
            zorder=5,
        )

        # BrowseComp+ marker — hollow
        ax.scatter(
            xb,
            yb,
            marker=mk,
            facecolor="white",
            edgecolor=col,
            linewidth=1.5,
            s=70,
            alpha=0.97,
            zorder=5,
        )

        # Method label anchored close to BrowseComp+ marker
        dx_pt, dy_pt, ha, va = panel_offsets[m]

        ax.annotate(
            METHOD_LABEL[m],
            xy=(xb, yb),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=6.3,
            color=col,
            ha=ha,
            va=va,
            fontweight="bold",
            zorder=6,
        )

        out[m] = {
            "gaia_x": g["x_median"],
            "gaia_pass1": g["acc_mean"],
            "bcp_x": b["x_median"],
            "bcp_pass1": b["acc_mean"],
        }

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel(x_label, color=C_AXIS)
    ax.set_ylabel("Pass@1 (%)", color=C_AXIS)
    ax.set_ylim(15, 80)



    return out


def _add_benchmark_legend(ax, loc: str = "lower right"):
    """Add benchmark legend with a framed white box.

    The frame makes the legend visually separate from data markers,
    reducing the chance that it is mistaken as an in-plot icon.
    """

    bench_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=C_HELP,
            markeredgecolor=C_HELP,
            markersize=6,
            label="GAIA",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=C_HELP,
            markeredgewidth=1.5,
            markersize=6,
            label="BrowseComp+",
        ),
    ]

    leg = ax.legend(
        handles=bench_handles,
        loc=loc,
        fontsize=6.0,
        labelspacing=0.30,
        handlelength=1.0,
        borderpad=0.45,
        borderaxespad=0.55,
        frameon=True,
        framealpha=0.95,
        fancybox=True,
        edgecolor="#999999",
        facecolor="white",
    )

    leg.get_frame().set_linewidth(0.6)

    return leg


# ================================================================== driver

def render(df_g: pd.DataFrame, df_b: pd.DataFrame, stem: Path) -> Dict:
    # ------------------------------------------------------------------
    # Combined 1x2 figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(5.6, 2.8))

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.30,
        left=0.09,
        right=0.985,
        bottom=0.20,
        top=0.92,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    sum_a = _draw_panel(
        ax_a,
        df_g,
        df_b,
        x_col="total_tokens",
        x_scale=1000.0,
        x_label="Token consumption (k)",
        panel_letter="a",
        log_x=False,
        panel_key="a",
    )

    sum_b = _draw_panel(
        ax_b,
        df_g,
        df_b,
        x_col="tool_calls_atomic",
        x_scale=1.0,
        x_label="Number of tool calls (log)",
        panel_letter="b",
        log_x=True,
        panel_key="b",
    )

    # ------------------------------------------------------------------
    # 修改点：
    # 合并图中，Panel (a) 和 Panel (b) 都添加 benchmark 图释。
    # 图释带白底边框，避免被误认为图中的一个数据点。
    # ------------------------------------------------------------------
    _add_benchmark_legend(ax_a, loc="lower right")
    _add_benchmark_legend(ax_b, loc="lower right")

    save_pub(fig, str(stem))
    plt.close(fig)

    # ------------------------------------------------------------------
    # Standalone Panel (a)
    # ------------------------------------------------------------------
    out_dir = stem.parent

    fig_a, ax = plt.subplots(figsize=(3.2, 2.7))

    _draw_panel(
        ax,
        df_g,
        df_b,
        x_col="total_tokens",
        x_scale=1000.0,
        x_label="Token consumption (k)",
        panel_letter="a",
        log_x=False,
        panel_key="a",
    )

    _add_benchmark_legend(ax, loc="lower right")

    fig_a.subplots_adjust(
        left=0.17,
        right=0.96,
        bottom=0.18,
        top=0.93,
    )

    save_pub(fig_a, str(out_dir / "figure5a_token_pareto_cross"))
    plt.close(fig_a)

    # ------------------------------------------------------------------
    # Standalone Panel (b)
    # ------------------------------------------------------------------
    fig_b, ax = plt.subplots(figsize=(3.2, 2.7))

    _draw_panel(
        ax,
        df_g,
        df_b,
        x_col="tool_calls_atomic",
        x_scale=1.0,
        x_label="Number of tool calls (log)",
        panel_letter="b",
        log_x=True,
        panel_key="b",
    )

    _add_benchmark_legend(ax, loc="lower right")

    fig_b.subplots_adjust(
        left=0.17,
        right=0.96,
        bottom=0.18,
        top=0.93,
    )

    save_pub(fig_b, str(out_dir / "figure5b_toolcalls_pareto_cross"))
    plt.close(fig_b)

    return {
        "panel_a": sum_a,
        "panel_b": sum_b,
        "stat_legend": {
            "benchmarks": "GAIA filled marker and BrowseComp+ hollow marker",
            "encoding": (
                "marker shape = method; fill = benchmark; "
                "dashed line = same-method trajectory across benchmarks"
            ),
            "panel_a_x": "median(total_tokens) per (method, benchmark)",
            "panel_b_x": (
                "median(tool_calls_atomic) per (method, benchmark), "
                "log scale"
            ),
            "y": "mean(Pass@1) per (method, benchmark)",
            "source_data": (
                "HyMem/data/figure3/per_task.csv GAIA "
                "+ HyMem/data/figure4/per_task.csv BrowseComp+"
            ),
        },
    }


def main() -> int:
    HYMEM = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gaia",
        default=str(HYMEM / "data" / "figure3" / "per_task.csv"),
    )

    ap.add_argument(
        "--bcplus",
        default=str(HYMEM / "data" / "figure4" / "per_task.csv"),
    )

    ap.add_argument(
        "--out",
        default=str(
            HYMEM / "results" / "figures" / "figure5_cross_benchmark"
        ),
    )

    args = ap.parse_args()

    df_g = pd.read_csv(args.gaia)
    df_b = pd.read_csv(args.bcplus)

    print(f"[loaded] GAIA: {len(df_g)} rows; BCP+: {len(df_b)} rows")

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    stats = render(df_g, df_b, out_stem)

    print("\n[Panel a — GAIA -> BCP+ trajectory in (tokens, Pass@1)]")

    for m, v in stats["panel_a"].items():
        print(
            f"  {m:<8} "
            f"GAIA=({int(v['gaia_x'] / 1000):>5}k, "
            f"{v['gaia_pass1'] * 100:5.1f}%)  "
            f"->  BCP+=({int(v['bcp_x'] / 1000):>5}k, "
            f"{v['bcp_pass1'] * 100:5.1f}%)"
        )

    print("\n[Panel b — GAIA -> BCP+ trajectory in (tool_calls, Pass@1)]")

    for m, v in stats["panel_b"].items():
        print(
            f"  {m:<8} "
            f"GAIA=({int(v['gaia_x']):>4}, "
            f"{v['gaia_pass1'] * 100:5.1f}%)  "
            f"->  BCP+=({int(v['bcp_x']):>4}, "
            f"{v['bcp_pass1'] * 100:5.1f}%)"
        )

    stats_path = out_stem.parent / "figure5_stats.json"

    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[saved figure] {out_stem}.{{svg,pdf,tiff,png}}")
    print(f"[saved stats ] {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
