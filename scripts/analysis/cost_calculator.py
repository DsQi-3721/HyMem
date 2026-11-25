"""USD cost estimator for HyMem / baseline experiments (Step B3).

Reviewer Bh2U W2 asked for "efficiency metrics to quantify the trade-off
between performance gains and resource cost." This module turns the
``token_usage_by_component`` already recorded in result JSONs into per-task
USD cost figures, with a clearly-cited price table.

Pure post-processing — does not touch any agent code or run any new
inference. No new env vars or endpoints required.

Pricing reference (USD per 1M tokens, public list prices as of 2025-01).
Self-hosted models (e.g. Qwen3-32B on local vLLM) default to 0.0; override
with --self-hosted-cost-per-m if you want to attribute compute amortization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# (input_per_M, output_per_M) USD prices.
# Sources: official model provider pricing pages as of 2025-01.
MODEL_PRICING_USD_PER_M: Dict[str, Tuple[float, float]] = {
    # DeepSeek (https://api-docs.deepseek.com/quick_start/pricing)
    "deepseek-chat":           (0.27, 1.10),
    "deepseek-reasoner":       (0.55, 2.19),
    "deepseek-v3.2":           (0.27, 1.10),
    "deepseek-v3":             (0.27, 1.10),

    # OpenAI (https://openai.com/api/pricing/) — selected GPT-4-class models
    "gpt-4o":                  (2.50, 10.00),
    "gpt-4o-mini":             (0.15, 0.60),
    "gpt-4-turbo":             (10.00, 30.00),
    "o1":                      (15.00, 60.00),
    "o1-mini":                 (3.00, 12.00),

    # Anthropic (https://www.anthropic.com/pricing)
    "claude-opus-4":           (15.00, 75.00),
    "claude-sonnet-4":         (3.00, 15.00),
    "claude-haiku-4":          (1.00, 5.00),

    # Self-hosted open models (default 0; override via CLI flag if needed)
    "qwen3-32b":               (0.0, 0.0),
    "qwen3-7b":                (0.0, 0.0),
    "qwen3-14b":               (0.0, 0.0),
    "qwen2.5-7b":              (0.0, 0.0),
    "qwen2.5-32b":             (0.0, 0.0),
    "llama-3-70b":             (0.0, 0.0),
}

# Map per-component name → which models_used field to bill for that token bucket.
COMPONENT_TO_MODEL_KEY = {
    "planner": "planner",
    "executor": "executor",
    "branch": "planner",   # branch reuses the planner model in current HyMem
    "memory": "aux",
    "other": "aux",
}


def _price_for(model: Optional[str], self_hosted_per_m: float = 0.0) -> Tuple[float, float]:
    """Return (input_per_M, output_per_M) for ``model``. Falls back to
    self_hosted_per_m for unknown models (treat as locally hosted)."""
    if not model:
        return (self_hosted_per_m, self_hosted_per_m)
    key = model.strip().lower()
    if key in MODEL_PRICING_USD_PER_M:
        return MODEL_PRICING_USD_PER_M[key]
    # Fuzzy: strip suffixes like "-instruct", versions
    for known, price in MODEL_PRICING_USD_PER_M.items():
        if known in key or key in known:
            return price
    return (self_hosted_per_m, self_hosted_per_m)


def compute_cost(
    token_usage_by_component: Dict[str, Dict[str, int]],
    models_used: Dict[str, str],
    self_hosted_per_m: float = 0.0,
) -> Dict[str, float]:
    """Convert per-component token usage into per-component USD cost.

    Args:
        token_usage_by_component: e.g. ``{"planner": {"prompt_tokens": ...,
            "completion_tokens": ...}, ...}`` (matches the schema written by
            ``HyMemEventLog.token_usage.to_dict()['by_component']``).
        models_used: ``{"planner": "...", "executor": "...", "aux": "...",
            "judge": "..."}`` from ``env.stats['models_used']``.
        self_hosted_per_m: price (USD per 1M tokens) to charge for unknown
            / self-hosted models. Default 0.

    Returns:
        ``{"planner_usd": ..., "executor_usd": ..., "branch_usd": ...,
           "memory_usd": ..., "total_usd": ..., "models_used": models_used}``
    """
    out: Dict[str, float] = {}
    total = 0.0
    for component, tok in (token_usage_by_component or {}).items():
        p_tok = int(tok.get("prompt_tokens", 0)) if isinstance(tok, dict) else 0
        c_tok = int(tok.get("completion_tokens", 0)) if isinstance(tok, dict) else 0
        model_key = COMPONENT_TO_MODEL_KEY.get(component, "aux")
        model_name = models_used.get(model_key) if models_used else None
        in_per_m, out_per_m = _price_for(model_name, self_hosted_per_m)
        cost = (p_tok * in_per_m + c_tok * out_per_m) / 1_000_000
        out[f"{component}_usd"] = round(cost, 6)
        total += cost
    out["total_usd"] = round(total, 6)
    out["models_used"] = dict(models_used or {})
    return out


def aggregate_results(results: List[Dict], self_hosted_per_m: float = 0.0) -> Dict:
    """Per-task + aggregate cost summary for a list of result dicts loaded
    from a results JSON file."""
    per_task = []
    total_usd = 0.0
    n_correct = 0
    n_total = 0
    for r in results:
        token_by_comp = r.get("token_usage_by_component") or {}
        if not token_by_comp:
            # Some result files only have flat totals — treat as planner-only.
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
        per_task.append({
            "instance_id": r.get("instance_id"),
            "score": r.get("score", 0),
            **cost,
        })
        total_usd += cost["total_usd"]
        n_total += 1
        if r.get("score", 0):
            n_correct += 1

    avg_usd = total_usd / max(n_total, 1)
    avg_usd_per_correct = total_usd / max(n_correct, 1) if n_correct > 0 else None

    return {
        "per_task": per_task,
        "n_total": n_total,
        "n_correct": n_correct,
        "total_usd": round(total_usd, 4),
        "avg_usd_per_task": round(avg_usd, 6),
        "avg_usd_per_correct_answer": (
            round(avg_usd_per_correct, 6) if avg_usd_per_correct is not None else None
        ),
    }


def _load(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unrecognized result file shape: {path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="USD cost estimator for HyMem result JSONs.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Result JSON file (HyMem / ReAct / Fold output)")
    parser.add_argument("--self-hosted-cost-per-m", type=float, default=0.0,
                        help="USD per 1M tokens for unknown / self-hosted models (default 0)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional: write JSON summary here")
    args = parser.parse_args(argv)

    results = _load(args.input)
    summary = aggregate_results(results, self_hosted_per_m=args.self_hosted_cost_per_m)

    print(f"Tasks evaluated: {summary['n_total']} ({summary['n_correct']} correct)")
    print(f"Total USD:               ${summary['total_usd']:.4f}")
    print(f"Average USD per task:    ${summary['avg_usd_per_task']:.6f}")
    if summary["avg_usd_per_correct_answer"] is not None:
        print(f"Average USD per correct: ${summary['avg_usd_per_correct_answer']:.6f}")
    else:
        print("Average USD per correct: n/a (no correct answers)")

    if args.output:
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
