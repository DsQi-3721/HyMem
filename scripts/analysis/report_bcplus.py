"""One-shot full comparison report — baselines on BrowseComp-Plus.

Run after experiments to get a complete picture:
    python scripts/analysis/report_bcplus.py

Optional outputs:
    --markdown report.md     # save Markdown report
    --pivot LABEL            # head-to-head delta vs this framework (default: react)

Mirrors `scripts/analysis/report_all.py` for the BCplus benchmark
(150 tasks, split 50 easy / 50 medium / 50 hard).

What it produces:
  1. Per-framework × per-difficulty Pass@1 / tokens / atomic-calls / time / USD
  2. Pivot-vs-rest head-to-head delta (% points)
  3. Cost estimation in USD (DeepSeek pricing)
  4. Anomaly detection (init-crashed tasks, empty predictions)
  5. Win-loss-tie matrix per task vs pivot

Reads per-task `*_trace.json` from logs/bcplus_<fw>/ — independent of
aggregate `results/bcplus/<fw>/bcplus_*_results_*.json` files.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List


PRICING = {  # USD per 1M tokens (input, output)
    "deepseek-chat":     (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}

# Display-friendly names for data_source labels (typo "meduim" kept in raw data)
SOURCE_ORDER = ["bc_test_easy", "bc_test_meduim", "bc_test_hard"]
SOURCE_LABEL = {
    "bc_test_easy":   "Easy",
    "bc_test_meduim": "Medium",
    "bc_test_hard":   "Hard",
}


def load_traces(directory: str) -> List[Dict]:
    out = []
    for p in sorted(glob(f"{directory}/*_trace.json")):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def load_from_aggregate(results_dir: str, framework: str) -> List[Dict]:
    """Fallback: synthesize trace-shaped dicts from `bcplus_<fw>_results_*.json`.

    The aggregate file stores per-task records in a flat schema (fields at
    top level, no `stats` wrapper). We wrap them so downstream code can treat
    them like per-task `*_trace.json` files.
    """
    candidates = sorted(
        glob(f"{results_dir}/{framework}/bcplus_{framework}_results_*.json"),
        key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not candidates:
        return []
    d = json.load(open(candidates[0]))
    out = []
    for r in d.get("results", []):
        out.append({
            "instance_id": r.get("instance_id"),
            "problem": r.get("problem"),
            "prediction": r.get("prediction"),
            "score": r.get("score"),
            "stats": r,  # full record acts as the stats blob
        })
    return out


def stat(traces: List[Dict], key: str, default=0):
    return [t.get("stats", {}).get(key, default) for t in traces]


def estimate_cost(traces: List[Dict], default_model="deepseek-chat") -> float:
    total = 0.0
    for t in traces:
        s = t.get("stats", {})
        pt = s.get("total_prompt_tokens", 0) or 0
        ct = s.get("total_completion_tokens", 0) or 0
        models = s.get("models_used") or {}
        model = (models.get("planner") or default_model).lower()
        in_p, out_p = PRICING.get(model, PRICING[default_model])
        total += (pt * in_p + ct * out_p) / 1_000_000
    return total


def detect_anomalies(traces: List[Dict]) -> Dict:
    init_crashed = [t for t in traces
                    if (stat([t], "agent_turns")[0] or 0) == 0
                    and (stat([t], "total_tokens")[0] or 0) == 0]
    empty_pred = [t for t in traces
                  if (t.get("score") or 0) == 0
                  and (not str(t.get("prediction", "")).strip()
                       or str(t.get("prediction", "")).strip()
                          in (".", "...", "None", "null"))]
    return {
        "init_crashed_count": len(init_crashed),
        "empty_pred_count": len(empty_pred),
    }


def summarize(traces: List[Dict]) -> Dict:
    n = len(traces)
    if n == 0:
        return {"n": 0}
    correct = sum(1 for t in traces if (t.get("score") or 0) > 0)
    return {
        "n": n,
        "correct": correct,
        "pass@1": correct / n,
        "avg_total_tokens":       sum(stat(traces, "total_tokens")) / n,
        "avg_prompt_tokens":      sum(stat(traces, "total_prompt_tokens")) / n,
        "avg_completion_tokens":  sum(stat(traces, "total_completion_tokens")) / n,
        "avg_tool_calls_atomic":  sum(stat(traces, "tool_calls_atomic")) / n,
        "avg_agent_turns":        sum(stat(traces, "agent_turns")) / n,
        "avg_session_time_s":     sum(stat(traces, "session_time")) / n,
        "total_usd":              estimate_cost(traces),
        "anomalies":              detect_anomalies(traces),
    }


def winloss_vs_pivot(by_fw: Dict[str, List[Dict]], pivot: str) -> Dict:
    pivot_by_id = {t.get("instance_id"): (t.get("score") or 0)
                   for t in by_fw.get(pivot, [])}
    out = {}
    for fw, traces in by_fw.items():
        if fw == pivot:
            continue
        wins = losses = both_correct = both_wrong = common = 0
        for t in traces:
            tid = t.get("instance_id")
            if tid not in pivot_by_id:
                continue
            common += 1
            ps, bs = pivot_by_id[tid], (t.get("score") or 0)
            if ps > 0 and bs == 0:
                wins += 1
            elif ps == 0 and bs > 0:
                losses += 1
            elif ps > 0 and bs > 0:
                both_correct += 1
            else:
                both_wrong += 1
        out[fw] = {
            "common_tasks": common,
            "pivot_wins": wins,
            "baseline_wins": losses,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        }
    return out


# ---------- Renderers ----------

def render_main_table(summaries: Dict[str, Dict[str, Dict]],
                      frameworks: List[str]) -> str:
    lines = []
    sep = "=" * 100
    lines.append(sep)
    lines.append(f"{'Framework':<10} {'Diff':<7} {'N':>4} {'Pass@1':>8} "
                 f"{'AvgTok':>10} {'AvgAtomic':>10} {'AvgTurns':>9} "
                 f"{'AvgTime':>9} {'USD':>7}")
    lines.append(sep)
    for fw in frameworks:
        for src in SOURCE_ORDER:
            s = summaries[fw].get(src)
            if not s or s.get("n", 0) == 0:
                continue
            label = SOURCE_LABEL.get(src, src)
            lines.append(
                f"{fw:<10} {label:<7} {s['n']:>4} {s['pass@1']:>8.3f} "
                f"{s['avg_total_tokens']:>10,.0f} {s['avg_tool_calls_atomic']:>10.1f} "
                f"{s['avg_agent_turns']:>9.1f} {s['avg_session_time_s']:>8.0f}s "
                f"{s['total_usd']:>6.2f}"
            )
        s = summaries[fw].get("TOTAL")
        if s and s.get("n", 0) > 0:
            lines.append(
                f"{fw:<10} {'TOTAL':<7} {s['n']:>4} {s['pass@1']:>8.3f} "
                f"{s['avg_total_tokens']:>10,.0f} {s['avg_tool_calls_atomic']:>10.1f} "
                f"{s['avg_agent_turns']:>9.1f} {s['avg_session_time_s']:>8.0f}s "
                f"{s['total_usd']:>6.2f}"
            )
        lines.append("-" * 100)
    return "\n".join(lines)


def render_oneglance(summaries: Dict[str, Dict[str, Dict]],
                     frameworks: List[str], pivot: str) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"One-glance comparison (TOTAL) — pivot: {pivot}")
    lines.append("=" * 78)
    lines.append(f"{'Framework':<10} {'N':>4} {'Pass@1':>8} {'Δ vs '+pivot:>10} "
                 f"{'AvgTok':>10} {'Cost/correct':>14}")
    lines.append("-" * 78)
    pivot_p1 = summaries.get(pivot, {}).get("TOTAL", {}).get("pass@1", 0)
    for fw in frameworks:
        s = summaries[fw].get("TOTAL")
        if not s or s.get("n", 0) == 0:
            continue
        delta = (s["pass@1"] - pivot_p1) * 100
        delta_str = f"{delta:+.1f}pp" if fw != pivot else "—"
        correct = max(s["correct"], 1)
        cpc = s["total_usd"] / correct
        lines.append(
            f"{fw:<10} {s['n']:>4} {s['pass@1']:>8.3f} {delta_str:>10} "
            f"{s['avg_total_tokens']:>10,.0f} ${cpc:>12.4f}"
        )
    return "\n".join(lines)


def render_winloss(wl: Dict[str, Dict], pivot: str) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"Per-task win/loss matrix vs {pivot}")
    lines.append("=" * 78)
    lines.append(f"{'Baseline':<10} {'N common':>9} {pivot+' wins':>12} "
                 f"{'baseline wins':>14} {'both correct':>14} {'both wrong':>12}")
    lines.append("-" * 78)
    for fw, d in wl.items():
        lines.append(
            f"{fw:<10} {d['common_tasks']:>9} {d['pivot_wins']:>12} "
            f"{d['baseline_wins']:>14} {d['both_correct']:>14} "
            f"{d['both_wrong']:>12}"
        )
    return "\n".join(lines)


def render_anomalies(summaries: Dict[str, Dict[str, Dict]],
                     frameworks: List[str]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("Anomaly detection — broken-run sanity check")
    lines.append("=" * 78)
    lines.append(f"{'Framework':<10} {'init_crashed':>14} {'empty_pred':>12}")
    lines.append("-" * 78)
    for fw in frameworks:
        s = summaries[fw].get("TOTAL", {})
        a = s.get("anomalies", {})
        lines.append(
            f"{fw:<10} {a.get('init_crashed_count',0):>14} "
            f"{a.get('empty_pred_count',0):>12}"
        )
    return "\n".join(lines)


def render_by_source_compare(summaries: Dict[str, Dict[str, Dict]],
                             frameworks: List[str]) -> str:
    lines = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("Pass@1 by difficulty — cross-framework")
    lines.append("=" * 78)
    head = f"{'Framework':<10}" + "".join(
        f"{SOURCE_LABEL.get(s,s):>10}" for s in SOURCE_ORDER) + f"{'Total':>10}"
    lines.append(head)
    lines.append("-" * 78)
    for fw in frameworks:
        row = f"{fw:<10}"
        for src in SOURCE_ORDER:
            v = summaries[fw].get(src, {}).get("pass@1")
            row += f"{v:>10.3f}" if v is not None else f"{'—':>10}"
        v = summaries[fw].get("TOTAL", {}).get("pass@1")
        row += f"{v:>10.3f}" if v is not None else f"{'—':>10}"
        lines.append(row)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs",
                        help="Root dir containing per-framework log dirs.")
    parser.add_argument("--results-root", default="results/bcplus",
                        help="Fallback dir for aggregate result JSONs.")
    parser.add_argument("--frameworks", nargs="+",
                        default=["react", "summary", "resum", "thread", "amem", "hymem"])
    parser.add_argument("--pivot", default="hymem",
                        help="Reference framework for Δ and win/loss matrix.")
    parser.add_argument("--markdown", type=Path,
                        help="Optional path to save the rendered report.")
    args = parser.parse_args(argv)

    fw_dir = {fw: f"{args.root}/bcplus_{fw}" for fw in args.frameworks}

    # Load — prefer per-task traces, fall back to aggregate JSON.
    by_fw: Dict[str, List[Dict]] = {}
    for fw, d in fw_dir.items():
        traces = load_traces(d)
        src = d
        if not traces:
            traces = load_from_aggregate(args.results_root, fw)
            src = f"{args.results_root}/{fw}/* (aggregate)"
            if traces:
                print(f"[info] {fw}: using aggregate JSON ({len(traces)} tasks)")
            else:
                print(f"[warn] no data for {fw} in either {d} or {args.results_root}/{fw}")
        by_fw[fw] = traces

    # Group by data_source
    by_fw_src: Dict[str, Dict[str, List[Dict]]] = defaultdict(
        lambda: defaultdict(list))
    for fw, traces in by_fw.items():
        for t in traces:
            src = t.get("stats", {}).get("data_source", "?")
            by_fw_src[fw][src].append(t)

    # Summarize
    summaries: Dict[str, Dict[str, Dict]] = {}
    for fw in args.frameworks:
        summaries[fw] = {}
        for src, ts in by_fw_src[fw].items():
            summaries[fw][src] = summarize(ts)
        summaries[fw]["TOTAL"] = summarize(by_fw[fw])

    # Render
    parts = [
        render_main_table(summaries, args.frameworks),
        "",
        render_by_source_compare(summaries, args.frameworks),
        "",
        render_oneglance(summaries, args.frameworks, args.pivot),
        render_winloss(winloss_vs_pivot(by_fw, args.pivot), args.pivot),
        render_anomalies(summaries, args.frameworks),
    ]
    report = "\n".join(parts)
    print(report)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report)
        print(f"\n[written] {args.markdown}")


if __name__ == "__main__":
    main()
