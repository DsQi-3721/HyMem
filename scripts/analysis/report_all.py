"""One-shot full comparison report — HyMem vs all baselines on GAIA.

Run this anytime after experiments to get a complete picture:
    python scripts/analysis/report_all.py

Optional outputs:
    --markdown report.md     # save Markdown report
    --latex   table1.tex     # paper Table 1 LaTeX

What it produces:
  1. Per-framework × per-level Pass@1 / tokens / atomic-calls / time
  2. HyMem-vs-baseline head-to-head delta (% points) per level
  3. Cost estimation in USD (DeepSeek pricing)
  4. Anomaly detection (init-crashed tasks, empty predictions)
  5. Win-loss-tie matrix per task
  6. Optional: paper-ready LaTeX Table 1

Reads per-task `*_trace.json` files from logs/ — does NOT depend on
the aggregate `gaia_results_*.json` files (those are partial / stale).
"""
from __future__ import annotations
import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Dict, List


# DeepSeek pricing (USD per 1M tokens, 2026 rates).
PRICING = {
    "deepseek-chat":     (0.27, 1.10),   # input, output
    "deepseek-reasoner": (0.55, 2.19),
}


def load_traces(directory: str) -> List[Dict]:
    out = []
    for p in sorted(glob(f"{directory}/*_trace.json")):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def stat(traces: List[Dict], key: str, default=0):
    return [t.get("stats", {}).get(key, default) for t in traces]


def task_id_set(traces: List[Dict]) -> set:
    return set(t.get("instance_id", "") for t in traces)


def estimate_cost(traces: List[Dict], default_model="deepseek-chat") -> float:
    """USD across all tasks, averaging models_used per task when present."""
    total = 0.0
    for t in traces:
        s = t.get("stats", {})
        pt = s.get("total_prompt_tokens", 0)
        ct = s.get("total_completion_tokens", 0)
        models = s.get("models_used") or {}
        # Use planner model as proxy for ratio of cost
        model = (models.get("planner") or default_model).lower()
        in_p, out_p = PRICING.get(model, PRICING[default_model])
        total += (pt * in_p + ct * out_p) / 1_000_000
    return total


def detect_anomalies(traces: List[Dict]) -> Dict:
    """Identify tasks that look broken: turns=0, empty pred, etc."""
    init_crashed = [t for t in traces
                    if stat([t], "agent_turns")[0] == 0
                    and stat([t], "total_tokens")[0] == 0]
    empty_pred = [t for t in traces
                  if t.get("score", 0) == 0
                  and (not str(t.get("prediction", "")).strip()
                       or str(t.get("prediction", "")).strip() in (".", "...", "None", "null"))]
    return {
        "init_crashed": init_crashed,
        "init_crashed_count": len(init_crashed),
        "empty_pred_count": len(empty_pred),
    }


def summarize_traces(traces: List[Dict]) -> Dict:
    n = len(traces)
    if n == 0:
        return {}
    correct = sum(1 for t in traces if t.get("score", 0) > 0)
    return {
        "n": n,
        "correct": correct,
        "pass@1": correct / n,
        "avg_total_tokens":          sum(stat(traces, "total_tokens")) / n,
        "avg_prompt_tokens":         sum(stat(traces, "total_prompt_tokens")) / n,
        "avg_completion_tokens":     sum(stat(traces, "total_completion_tokens")) / n,
        "avg_tool_calls_atomic":     sum(stat(traces, "tool_calls_atomic")) / n,
        "avg_agent_turns":           sum(stat(traces, "agent_turns")) / n,
        "avg_session_time_s":        sum(stat(traces, "session_time")) / n,
        "total_usd":                 estimate_cost(traces),
        "anomalies":                 detect_anomalies(traces),
    }


def task_level_winloss(by_fw: Dict[str, List[Dict]], pivot="HyMem") -> Dict:
    """Per-task win/loss/tie of HyMem vs each baseline."""
    pivot_by_id = {t["instance_id"]: t.get("score", 0) for t in by_fw[pivot]}
    out = {}
    for fw, traces in by_fw.items():
        if fw == pivot:
            continue
        wins = losses = ties = both_correct = both_wrong = 0
        common = 0
        for t in traces:
            tid = t.get("instance_id")
            if tid not in pivot_by_id:
                continue
            common += 1
            ps = pivot_by_id[tid]
            bs = t.get("score", 0)
            if ps > 0 and bs == 0:
                wins += 1
            elif ps == 0 and bs > 0:
                losses += 1
            elif ps > 0 and bs > 0:
                both_correct += 1
                ties += 1
            else:
                both_wrong += 1
                ties += 1
        out[fw] = {
            "common_tasks": common,
            "pivot_wins": wins,
            "baseline_wins": losses,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        }
    return out


# ---------- Renderers ----------

def render_main_table(summaries: Dict[str, Dict[str, Dict]], frameworks: List[str]) -> str:
    """Per-framework × per-level table (console-friendly)."""
    lines = []
    sep = "=" * 100
    lines.append(sep)
    lines.append(f"{'Framework':<10} {'Level':<6} {'N':>4} {'Pass@1':>8} "
                 f"{'AvgTok':>10} {'AvgAtomic':>10} {'AvgTurns':>9} "
                 f"{'AvgTime':>9} {'USD':>7}")
    lines.append(sep)
    for fw in frameworks:
        levels = sorted([k for k in summaries[fw].keys() if k != "TOTAL"])
        for lvl in levels:
            s = summaries[fw][lvl]
            if not s:
                continue
            lines.append(
                f"{fw:<10} L{lvl:<5} {s['n']:>4} {s['pass@1']:>8.3f} "
                f"{s['avg_total_tokens']:>10,.0f} {s['avg_tool_calls_atomic']:>10.1f} "
                f"{s['avg_agent_turns']:>9.1f} {s['avg_session_time_s']:>8.0f}s "
                f"{s['total_usd']:>6.2f}"
            )
        s = summaries[fw].get("TOTAL")
        if s:
            lines.append(
                f"{fw:<10} {'TOTAL':<6} {s['n']:>4} {s['pass@1']:>8.3f} "
                f"{s['avg_total_tokens']:>10,.0f} {s['avg_tool_calls_atomic']:>10.1f} "
                f"{s['avg_agent_turns']:>9.1f} {s['avg_session_time_s']:>8.0f}s "
                f"{s['total_usd']:>6.2f}"
            )
        lines.append("-" * 100)
    return "\n".join(lines)


def render_oneglance(summaries: Dict[str, Dict[str, Dict]], frameworks: List[str], pivot="HyMem") -> str:
    """One-glance side-by-side with HyMem-as-pivot delta."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"One-glance comparison (TOTAL across all levels) — pivot: {pivot}")
    lines.append("=" * 78)
    lines.append(f"{'Framework':<10} {'N':>4} {'Pass@1':>8} {'Δ vs '+pivot:>10} "
                 f"{'AvgTok':>10} {'Cost/correct':>14}")
    lines.append("-" * 78)
    pivot_p1 = summaries.get(pivot, {}).get("TOTAL", {}).get("pass@1", 0)
    for fw in frameworks:
        s = summaries[fw].get("TOTAL")
        if not s:
            continue
        delta = (s["pass@1"] - pivot_p1) * 100
        delta_str = f"{delta:+.1f}pp" if fw != pivot else "—"
        cost_per_correct = s["total_usd"] / max(s["correct"], 1)
        lines.append(
            f"{fw:<10} {s['n']:>4} {s['pass@1']:>8.3f} {delta_str:>10} "
            f"{s['avg_total_tokens']:>10,.0f} ${cost_per_correct:>12.4f}"
        )
    return "\n".join(lines)


def render_winloss(wl: Dict, pivot="HyMem") -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"Per-task win/loss matrix vs {pivot}")
    lines.append("=" * 78)
    lines.append(f"{'Baseline':<10} {'N common':>10} "
                 f"{pivot+' wins':>12} {'baseline wins':>14} "
                 f"{'both correct':>14} {'both wrong':>12}")
    lines.append("-" * 78)
    for fw, w in wl.items():
        lines.append(
            f"{fw:<10} {w['common_tasks']:>10} "
            f"{w['pivot_wins']:>12} {w['baseline_wins']:>14} "
            f"{w['both_correct']:>14} {w['both_wrong']:>12}"
        )
    return "\n".join(lines)


def render_anomalies(summaries: Dict[str, Dict[str, Dict]], frameworks: List[str]) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("Anomaly detection — broken-run sanity check")
    lines.append("=" * 78)
    lines.append(f"{'Framework':<10} {'init_crashed':>14} {'empty_pred':>12}")
    lines.append("-" * 78)
    any_problem = False
    for fw in frameworks:
        s = summaries[fw].get("TOTAL", {})
        anom = s.get("anomalies", {})
        ic = anom.get("init_crashed_count", 0)
        ep = anom.get("empty_pred_count", 0)
        flag = "  ⚠️" if ic > 0 else ""
        if ic > 0:
            any_problem = True
        lines.append(f"{fw:<10} {ic:>14}{flag} {ep:>12}")
    if any_problem:
        lines.append("")
        lines.append("⚠️  init_crashed > 0 means tasks with turns=0 and tokens=0 — likely runtime")
        lines.append("    errors. Re-run those tasks with current code (they're being counted as fail).")
    return "\n".join(lines)


def render_latex_table1(summaries: Dict[str, Dict[str, Dict]], frameworks: List[str]) -> str:
    """Paper-ready LaTeX Table 1."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Pass@1 on GAIA text-only validation set across difficulty levels. "
        r"All training-free baselines re-implemented in our harness with identical tool "
        r"layer (BMToolManager) and unified atomic-tool-call counter for fair comparison. "
        r"\textbf{Bold} = best per column.}",
        r"\label{tab:gaia_main}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Framework & L1 P@1 & L2 P@1 & L3 P@1 & Avg P@1 & Avg tokens & Tool calls \\",
        r"\midrule",
    ]
    # Find best per column
    best_at = {}
    for col in ["1", "2", "3", "TOTAL"]:
        scores = [(fw, summaries[fw].get(col, {}).get("pass@1", -1)) for fw in frameworks]
        best_at[col] = max(scores, key=lambda x: x[1])[0]

    def fmt(s, key, prec=3):
        v = s.get(key, 0)
        return f"{v:.{prec}f}"

    for fw in frameworks:
        cells = [fw]
        for lvl in ["1", "2", "3", "TOTAL"]:
            s = summaries[fw].get(lvl, {})
            v = fmt(s, "pass@1", 3)
            if best_at.get(lvl) == fw:
                v = r"\textbf{" + v + "}"
            cells.append(v)
        s_total = summaries[fw].get("TOTAL", {})
        cells.append(f"{s_total.get('avg_total_tokens', 0):,.0f}")
        cells.append(f"{s_total.get('avg_tool_calls_atomic', 0):.1f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def render_markdown_report(summaries, wl, frameworks, pivot, latex):
    """Full Markdown report."""
    lines = [
        f"# GAIA text-only Comparison Report",
        f"",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        f"",
        f"## Main Table",
        f"",
        f"```",
        render_main_table(summaries, frameworks),
        f"```",
        f"",
        f"## One-glance ({pivot} as pivot)",
        f"",
        f"```",
        render_oneglance(summaries, frameworks, pivot),
        f"```",
        f"",
        f"## Per-task Win/Loss vs {pivot}",
        f"",
        f"```",
        render_winloss(wl, pivot),
        f"```",
        f"",
        f"## Anomalies",
        f"",
        f"```",
        render_anomalies(summaries, frameworks),
        f"```",
        f"",
        f"## Paper Table 1 (LaTeX)",
        f"",
        f"```latex",
        latex,
        f"```",
    ]
    return "\n".join(lines)


# ---------- Main ----------

DEFAULT_FW_DIRS = {
    "HyMem":  ["logs/gaia_L1_optimized", "logs/gaia_L2_optimized", "logs/gaia_L3_optimized"],
    "ReSum":  ["logs/gaia_resum"],
    "Thread": ["logs/gaia_thread"],
    "A-MEM":  ["logs/gaia_amem"],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repo root (where logs/ lives)")
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Save Markdown report here")
    parser.add_argument("--latex", type=Path, default=None,
                        help="Save paper-ready LaTeX Table 1 here")
    parser.add_argument("--pivot", default="HyMem",
                        help="Framework to use as pivot for delta/winloss")
    args = parser.parse_args(argv)

    root = Path(args.root)
    fw_dirs = DEFAULT_FW_DIRS

    # Load + group by framework × level
    by_fw_level = defaultdict(lambda: defaultdict(list))
    by_fw_all = defaultdict(list)
    for fw, dlist in fw_dirs.items():
        for d in dlist:
            for t in load_traces(str(root / d)):
                lvl = str(t.get("level") or t.get("stats", {}).get("level") or "?")
                by_fw_level[fw][lvl].append(t)
                by_fw_all[fw].append(t)

    frameworks = list(fw_dirs.keys())

    # Summarize
    summaries = defaultdict(dict)
    for fw in frameworks:
        for lvl, traces in by_fw_level[fw].items():
            summaries[fw][lvl] = summarize_traces(traces)
        summaries[fw]["TOTAL"] = summarize_traces(by_fw_all[fw])

    wl = task_level_winloss(by_fw_all, pivot=args.pivot)
    latex = render_latex_table1(summaries, frameworks)

    # Print everything
    print(render_main_table(summaries, frameworks))
    print()
    print(render_oneglance(summaries, frameworks, args.pivot))
    print()
    print(render_winloss(wl, args.pivot))
    print()
    print(render_anomalies(summaries, frameworks))
    print()
    print("=" * 78)
    print("Paper LaTeX Table 1 (also saved if --latex given):")
    print("=" * 78)
    print(latex)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown_report(
            summaries, wl, frameworks, args.pivot, latex))
        print(f"\n📝 Markdown report saved to {args.markdown}")
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex)
        print(f"📊 LaTeX Table 1 saved to {args.latex}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
