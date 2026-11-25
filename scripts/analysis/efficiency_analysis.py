"""Efficiency analysis for HyMem result JSONs (Step B4).

Reviewer Bh2U W2 wants efficiency metrics that "quantify the trade-off
between performance gains and resource cost." This script reads one or more
result JSONs (HyMem / ReAct / Fold), computes:

  - tokens-per-correct-answer
  - atomic-tool-calls-per-correct-answer
  - USD-per-correct-answer (using cost_calculator)
  - average per-component latency

and emits both a LaTeX table (paper-ready) and a matplotlib comparison plot.

No new env vars, no API calls, no inference re-runs — pure post-processing
over fields already populated by the upgraded eval pipeline (Steps A3, B1,
B2, B3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

# Reuse the cost calculator for consistent USD numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analysis.cost_calculator import compute_cost  # noqa: E402


def _load_results(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized result file shape: {path}")


def summarize(results: List[Dict], label: str, self_hosted_per_m: float = 0.0) -> Dict:
    """Compute one row of metrics for a single framework's results."""
    n_total = len(results)
    n_correct = sum(1 for r in results if r.get("score", 0))

    tokens = [r.get("total_tokens", 0) for r in results]
    atomic = [r.get("tool_calls_atomic", r.get("tool_calls_total", r.get("tool_calls", 0)))
              for r in results]
    turns = [r.get("agent_turns", 0) for r in results]
    times = [r.get("session_time", 0.0) for r in results]

    # USD costs
    costs_total = []
    costs_by_comp: Dict[str, List[float]] = {}
    for r in results:
        token_by_comp = r.get("token_usage_by_component") or {}
        if not token_by_comp:
            token_by_comp = {
                "planner": {
                    "prompt_tokens": r.get("total_prompt_tokens", 0),
                    "completion_tokens": r.get("total_completion_tokens", 0),
                }
            }
        cost = compute_cost(
            token_by_comp,
            r.get("models_used", {}),
            self_hosted_per_m=self_hosted_per_m,
        )
        costs_total.append(cost["total_usd"])
        for k, v in cost.items():
            if k.endswith("_usd") and k != "total_usd":
                costs_by_comp.setdefault(k, []).append(v)

    # Latency by component
    latency_by_comp: Dict[str, List[float]] = {}
    for r in results:
        lat = r.get("latency_by_component") or {}
        for k in ("planner", "executor", "branch", "memory", "other", "total"):
            v = lat.get(k, None)
            if isinstance(v, (int, float)):
                latency_by_comp.setdefault(k, []).append(float(v))

    sum_tokens = sum(tokens)
    sum_atomic = sum(atomic)
    sum_costs = sum(costs_total)

    return {
        "label": label,
        "n_total": n_total,
        "n_correct": n_correct,
        "pass@1": n_correct / max(n_total, 1),
        "avg_tokens": mean(tokens) if tokens else 0,
        "avg_atomic_calls": mean(atomic) if atomic else 0,
        "avg_agent_turns": mean(turns) if turns else 0,
        "avg_session_time_s": mean(times) if times else 0,
        "avg_cost_usd": mean(costs_total) if costs_total else 0,
        # Efficiency ratios
        "tokens_per_correct": sum_tokens / max(n_correct, 1) if n_correct else None,
        "atomic_calls_per_correct": sum_atomic / max(n_correct, 1) if n_correct else None,
        "usd_per_correct": sum_costs / max(n_correct, 1) if n_correct else None,
        # Per-component breakdowns
        "avg_latency_by_component": {
            k: round(mean(v), 3) for k, v in latency_by_comp.items() if v
        },
        "avg_cost_by_component": {
            k: round(mean(v), 6) for k, v in costs_by_comp.items() if v
        },
    }


def to_latex(rows: List[Dict]) -> str:
    """Render a paper-ready LaTeX table (booktabs)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Efficiency comparison across frameworks (per-task averages "
        r"and cost-normalized ratios). Atomic tool calls counted via the "
        r"unified counter (Sec.~\ref{sec:tool_call_unification}).}",
        r"\label{tab:efficiency}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (r"Framework & Pass@1 & Avg.\ tokens & Atomic calls & "
         r"Cost (USD) & Tokens / correct & Cost / correct (USD) \\"),
        r"\midrule",
    ]
    for r in rows:
        tpc = "n/a" if r["tokens_per_correct"] is None else f"{r['tokens_per_correct']:,.0f}"
        upc = "n/a" if r["usd_per_correct"] is None else f"{r['usd_per_correct']:.4f}"
        lines.append(
            f"{r['label']} & {r['pass@1']:.3f} & {r['avg_tokens']:,.0f} "
            f"& {r['avg_atomic_calls']:.1f} & {r['avg_cost_usd']:.4f} "
            f"& {tpc} & {upc} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def make_plot(rows: List[Dict], out_path: Path) -> None:
    """Optional matplotlib comparison plot. Skipped silently if matplotlib
    is unavailable in the current env."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f"[warn] matplotlib unavailable, skipping plot: {exc}")
        return

    labels = [r["label"] for r in rows]
    pass1 = [r["pass@1"] for r in rows]
    cost_per_correct = [
        r["usd_per_correct"] if r["usd_per_correct"] is not None else 0
        for r in rows
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, pass1)
    axes[0].set_ylabel("Pass@1")
    axes[0].set_title("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, cost_per_correct)
    axes[1].set_ylabel("USD per correct answer")
    axes[1].set_title("Cost efficiency (lower is better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Plot saved to {out_path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Efficiency analysis for result JSONs")
    parser.add_argument("--results", action="append", required=True, nargs=2,
                        metavar=("LABEL", "PATH"),
                        help="Provide once per framework: --results LABEL path.json")
    parser.add_argument("--output-tex", type=Path, default=Path("efficiency_table.tex"))
    parser.add_argument("--output-plot", type=Path, default=Path("efficiency_plot.pdf"))
    parser.add_argument("--output-json", type=Path, default=Path("efficiency_summary.json"))
    parser.add_argument("--self-hosted-cost-per-m", type=float, default=0.0)
    args = parser.parse_args(argv)

    rows = []
    for label, path_str in args.results:
        path = Path(path_str)
        results = _load_results(path)
        row = summarize(results, label=label, self_hosted_per_m=args.self_hosted_cost_per_m)
        rows.append(row)
        print(
            f"{label:>12s}: n={row['n_total']:3d} pass@1={row['pass@1']:.3f} "
            f"avg_tokens={row['avg_tokens']:,.0f} "
            f"avg_atomic={row['avg_atomic_calls']:.1f} "
            f"avg_cost=${row['avg_cost_usd']:.4f}"
        )

    args.output_tex.write_text(to_latex(rows))
    print(f"LaTeX table written to {args.output_tex}")

    args.output_json.write_text(json.dumps(rows, indent=2))
    print(f"JSON summary written to {args.output_json}")

    make_plot(rows, args.output_plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
