"""Branch (Isolated Reasoning) trigger-frequency analysis (Step B5).

Reviewer LMtp Q3 explicitly asked: "the Coordination Layer evaluates the
complexity of each subtask to decide whether to trigger the Isolated
Reasoning Module. Further analysis is needed regarding the frequency of
these triggers."

This script reads HyMem result JSONs and aggregates per-difficulty branch
metrics:
  - average branch_calls per task
  - branch success rate
  - branch atomic-call share of total tool_calls
  - routing source breakdown (LLM classifier vs fallback)

Outputs a paper-ready LaTeX table and an optional bar plot. Pure
post-processing — no API calls or new env vars.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

# Difficulty bucket inference. GAIA uses Level 1/2/3; BC-plus uses easy/medium/hard.
GAIA_DIFFICULTY_KEYS = ("level", "Level", "difficulty")
BC_DIFFICULTY_KEYS = ("difficulty", "Difficulty")
DIFFICULTY_NORMALIZE = {
    "1": "L1", "2": "L2", "3": "L3",
    "level 1": "L1", "level 2": "L2", "level 3": "L3",
    "easy": "easy", "medium": "medium", "hard": "hard",
}


def _infer_difficulty(r: Dict) -> str:
    for key in GAIA_DIFFICULTY_KEYS + BC_DIFFICULTY_KEYS:
        v = r.get(key)
        if v is None:
            continue
        v_str = str(v).strip().lower()
        return DIFFICULTY_NORMALIZE.get(v_str, v_str)
    # Try nested under extra_info / problem_meta
    for nested_key in ("extra_info", "problem_meta", "meta"):
        nested = r.get(nested_key) or {}
        if isinstance(nested, dict):
            for key in GAIA_DIFFICULTY_KEYS + BC_DIFFICULTY_KEYS:
                v = nested.get(key)
                if v is not None:
                    v_str = str(v).strip().lower()
                    return DIFFICULTY_NORMALIZE.get(v_str, v_str)
    return "unknown"


def _load_results(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized result file shape: {path}")


def aggregate_by_difficulty(results: List[Dict]) -> Dict[str, Dict]:
    """Group results by difficulty and compute branch-related aggregates."""
    by_diff: Dict[str, List[Dict]] = {}
    for r in results:
        diff = _infer_difficulty(r)
        by_diff.setdefault(diff, []).append(r)

    out: Dict[str, Dict] = {}
    for diff, rows in by_diff.items():
        n = len(rows)
        branch_calls = [r.get("branch_calls", 0) for r in rows]
        branch_success = [r.get("branch_success", 0) for r in rows]
        atomic = [r.get("tool_calls_atomic", r.get("tool_calls_total", 0)) for r in rows]
        # Branch-internal atomic calls (only available post-Step A3)
        by_source = [(r.get("tool_calls_by_source") or {}) for r in rows]
        branch_atomic = [s.get("branch", 0) for s in by_source]

        n_with_branch = sum(1 for b in branch_calls if b > 0)
        total_branch = sum(branch_calls)
        total_branch_success = sum(branch_success)
        total_atomic = sum(atomic)
        total_branch_atomic = sum(branch_atomic)

        # Routing source breakdown
        routing_summary = [r.get("routing_summary") or {} for r in rows]
        from_llm = sum(s.get("from_llm", 0) for s in routing_summary)
        from_fallback = sum(s.get("from_fallback", 0) for s in routing_summary)

        out[diff] = {
            "n_tasks": n,
            "n_tasks_with_branch": n_with_branch,
            "branch_invocation_rate": round(n_with_branch / max(n, 1), 3),
            "avg_branch_calls": round(mean(branch_calls), 3) if branch_calls else 0,
            "avg_branch_success": round(mean(branch_success), 3) if branch_success else 0,
            "branch_success_rate": (
                round(total_branch_success / max(total_branch, 1), 3)
                if total_branch > 0 else None
            ),
            "branch_atomic_share_of_total": (
                round(total_branch_atomic / max(total_atomic, 1), 3)
                if total_atomic > 0 else 0
            ),
            "routing_from_llm": from_llm,
            "routing_from_fallback": from_fallback,
            "routing_llm_share": (
                round(from_llm / max(from_llm + from_fallback, 1), 3)
                if (from_llm + from_fallback) > 0 else None
            ),
        }
    return out


def to_latex(label_to_summary: Dict[str, Dict[str, Dict]]) -> str:
    """Render a multi-difficulty branch trigger table.

    Rows: difficulty buckets; columns: per-framework branch metrics.
    """
    # Collect all difficulty keys across frameworks
    all_diffs: List[str] = []
    for summary in label_to_summary.values():
        for diff in summary.keys():
            if diff not in all_diffs:
                all_diffs.append(diff)
    # Stable order (numeric levels first)
    def _diff_order(d: str) -> str:
        order = {"L1": "0", "L2": "1", "L3": "2", "easy": "3", "medium": "4", "hard": "5"}
        return order.get(d, "9_" + d)
    all_diffs.sort(key=_diff_order)

    labels = list(label_to_summary.keys())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Branch (Isolated Reasoning Module) trigger frequency by "
        r"difficulty. ``Branch rate'' = fraction of tasks that triggered at "
        r"least one branch; ``Avg.\ branches'' = average branch invocations "
        r"per task; ``Atomic share'' = fraction of atomic tool calls made "
        r"inside branches; ``LLM router \%'' = fraction of routing decisions "
        r"made by the dedicated LLM classifier vs the keyword fallback.}",
        r"\label{tab:branch_triggers}",
        r"\small",
        r"\begin{tabular}{l" + "ccccc" * len(labels) + r"}",
        r"\toprule",
    ]

    header = "Difficulty"
    for label in labels:
        header += (
            f" & \\multicolumn{{5}}{{c}}{{{label}}}"
        )
    lines.append(header + r" \\")

    sub_header = ""
    for _ in labels:
        sub_header += (
            r" & N & Br.\ rate & Avg.\ branches & Atomic share & LLM router \%"
        )
    lines.append(sub_header + r" \\")
    lines.append(r"\midrule")

    for diff in all_diffs:
        row = diff
        for label in labels:
            d_summary = label_to_summary[label].get(diff)
            if d_summary is None:
                row += " & - & - & - & - & -"
                continue
            row += (
                f" & {d_summary['n_tasks']}"
                f" & {d_summary['branch_invocation_rate']:.2f}"
                f" & {d_summary['avg_branch_calls']:.2f}"
                f" & {d_summary['branch_atomic_share_of_total']:.2f}"
                f" & {d_summary['routing_llm_share'] if d_summary['routing_llm_share'] is not None else 0:.2f}"
            )
        lines.append(row + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_plot(label_to_summary: Dict[str, Dict[str, Dict]], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return

    diffs: List[str] = []
    for s in label_to_summary.values():
        for d in s:
            if d not in diffs:
                diffs.append(d)
    diffs.sort()

    labels = list(label_to_summary.keys())
    width = 0.8 / max(len(labels), 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = list(range(len(diffs)))
    for i, label in enumerate(labels):
        ys = [label_to_summary[label].get(d, {}).get("avg_branch_calls", 0)
              for d in diffs]
        ax.bar([xi + i * width for xi in x], ys, width=width, label=label)
    ax.set_xticks([xi + width * (len(labels) - 1) / 2 for xi in x])
    ax.set_xticklabels(diffs)
    ax.set_ylabel("Avg. branches per task")
    ax.set_title("Branch trigger frequency by difficulty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Branch trigger frequency analysis")
    parser.add_argument("--results", action="append", required=True, nargs=2,
                        metavar=("LABEL", "PATH"),
                        help="Provide once per result file: --results LABEL path.json")
    parser.add_argument("--output-tex", type=Path, default=Path("branch_triggers_table.tex"))
    parser.add_argument("--output-plot", type=Path, default=Path("branch_triggers_plot.pdf"))
    parser.add_argument("--output-json", type=Path, default=Path("branch_triggers_summary.json"))
    args = parser.parse_args(argv)

    summaries: Dict[str, Dict[str, Dict]] = {}
    for label, path_str in args.results:
        results = _load_results(Path(path_str))
        summary = aggregate_by_difficulty(results)
        summaries[label] = summary
        print(f"\n=== {label} ===")
        for diff, d_summary in summary.items():
            print(
                f"  {diff:>7s}: n={d_summary['n_tasks']:3d} "
                f"br_rate={d_summary['branch_invocation_rate']:.2f} "
                f"avg_branches={d_summary['avg_branch_calls']:.2f} "
                f"atomic_share={d_summary['branch_atomic_share_of_total']:.2f}"
            )

    args.output_tex.write_text(to_latex(summaries))
    print(f"\nLaTeX table written to {args.output_tex}")

    args.output_json.write_text(json.dumps(summaries, indent=2))
    print(f"JSON summary written to {args.output_json}")

    make_plot(summaries, args.output_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
