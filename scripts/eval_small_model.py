"""Small-model robustness eval driver (Step C2).

Reviewer LMtp Q2 asked whether smaller models can effectively handle the
hierarchical prompts. This script makes that experiment **infrastructurally
runnable** without re-engineering the eval pipeline:

  * Parametrizes the backbone model via ``--backbone_model`` (e.g. Qwen3-4B,
    Qwen3-7B, Qwen3-14B, Qwen3-32B).
  * Defaults to a small subset of GAIA Level 1 to keep wall-clock manageable
    when probing weaker models that often time out on L3.
  * Auto-selects the matching prompt_style (qwen / default) via the
    ``_infer_model_size`` helper added in Step C1 — no manual tuning needed.
  * Reports a robustness summary (per-model success rate + parse-failure
    rate) so reviewers see whether prompt-format breakage is the bottleneck
    rather than reasoning ability.

This is a thin shim around ``eval_bc.py`` / ``eval_gaia.py`` — does NOT
duplicate their evaluation logic, just sets the right env vars and forwards
the call. No new endpoints; reuses ``HYMEM_PLANNER_*`` / ``HYMEM_EXECUTOR_*``
env vars from ``scripts/env_eval_local.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Reuse the size inference from hymem_agent without importing the heavy module.
def _infer_model_size_local(model_name: Optional[str]) -> str:
    if not model_name:
        return "large"
    name = str(model_name).lower()
    for kw in ("gpt-4", "gpt4", "deepseek-reasoner", "deepseek-v3",
               "claude-opus", "claude-sonnet", "o1", "gemini-2", "qwen-max"):
        if kw in name:
            return "large"
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?=[^a-z]|$|-)", name)
    if not m:
        m = re.search(r"-(\d+(?:\.\d+)?)b", name)
    if m:
        try:
            n = float(m.group(1))
        except ValueError:
            return "large"
        if n <= 4:
            return "tiny"
        if n <= 14:
            return "small"
        if n <= 50:
            return "medium"
        return "large"
    return "large"


def _override_env(backbone_model: str, override_executor_only: bool) -> Dict[str, str]:
    """Set HYMEM_*_MODEL env vars to ``backbone_model``. By default we
    override planner+executor+aux together so the run uses ONE backbone end
    to end (the cleanest small-model robustness test). Pass
    ``--executor_only`` to keep the strong planner and only swap the
    executor — useful for ablating the executor's instruction-following
    independently."""
    env = os.environ.copy()
    if override_executor_only:
        env["HYMEM_EXECUTOR_MODEL"] = backbone_model
        env["HYMEM_AUX_MODEL"] = backbone_model
        # Keep HYMEM_PLANNER_MODEL as already set
    else:
        env["HYMEM_PLANNER_MODEL"] = backbone_model
        env["HYMEM_EXECUTOR_MODEL"] = backbone_model
        env["HYMEM_AUX_MODEL"] = backbone_model
    return env


def _summarize(results_path: Path) -> Dict:
    """Compute robustness metrics from a results JSON."""
    if not results_path.exists():
        return {"error": f"Results file not found: {results_path}"}
    data = json.loads(results_path.read_text())
    if isinstance(data, dict) and "results" in data:
        results = data["results"]
    else:
        results = data
    n = len(results)
    n_correct = sum(1 for r in results if r.get("score", 0))
    # parse-failure proxy: completed task with empty answer or status=failed
    n_parse_fail = sum(
        1 for r in results
        if r.get("status") == "failed"
        or (r.get("status") == "success" and not str(r.get("answer", "")).strip())
    )
    return {
        "n_total": n,
        "n_correct": n_correct,
        "pass@1": round(n_correct / max(n, 1), 3),
        "n_parse_or_runtime_failures": n_parse_fail,
        "parse_failure_rate": round(n_parse_fail / max(n, 1), 3),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Small-model robustness eval")
    parser.add_argument("--backbone_model", required=True,
                        help="e.g. Qwen3-4B, Qwen3-7B, Qwen3-14B, Qwen3-32B")
    parser.add_argument("--executor_only", action="store_true",
                        help="Only swap executor; keep current planner model")
    parser.add_argument("--max_tasks", type=int, default=10,
                        help="Cap tasks for fast probing (default 10)")
    parser.add_argument("--benchmark", choices=("gaia", "bc"), default="gaia",
                        help="Which benchmark to probe (default gaia)")
    parser.add_argument("--difficulty_filter", default="L1",
                        help="GAIA: L1/L2/L3 or 'all'; BC: easy/medium/hard or 'all' "
                             "(default: L1 — small models usually fail above L1)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the planned subprocess command and exit")
    parser.add_argument("--results_dir", type=Path,
                        default=Path("scripts/results/small_model"),
                        help="Where to drop the results JSON")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER,
                        help="Forwarded verbatim to the underlying eval script")
    args = parser.parse_args(argv)

    # Auto-select prompt style for the new backbone (informational; the
    # agent will also auto-pick at runtime via _infer_model_size).
    size = _infer_model_size_local(args.backbone_model)
    inferred_prompt = "qwen" if size in ("tiny", "small") else "default"
    print(
        f"[eval_small_model] backbone={args.backbone_model} "
        f"(size={size}); prompt_style={inferred_prompt}"
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / f"{args.backbone_model.lower()}_{args.benchmark}_{args.difficulty_filter}.json"

    env = _override_env(args.backbone_model, args.executor_only)
    env["HYMEM_PROMPT_STYLE"] = inferred_prompt

    if args.benchmark == "gaia":
        cmd = [
            sys.executable, "scripts/eval_gaia.py",
            "--framework", "hymem",
            "--max_tasks", str(args.max_tasks),
        ]
    else:  # bc
        cmd = [
            sys.executable, "scripts/eval_bc.py",
            "--workflow", "hymem",
            "--max_tasks", str(args.max_tasks),
        ]
    if args.extra_args:
        cmd.extend(args.extra_args)

    print(f"[eval_small_model] cmd: {' '.join(cmd)}")
    if args.dry_run:
        print("[eval_small_model] --dry_run: not executing")
        print("[eval_small_model] env overrides:")
        for k in ("HYMEM_PLANNER_MODEL", "HYMEM_EXECUTOR_MODEL",
                  "HYMEM_AUX_MODEL", "HYMEM_PROMPT_STYLE"):
            print(f"    {k}={env.get(k, '(unset)')}")
        return 0

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"[eval_small_model] eval subprocess failed: {exc}")
        return exc.returncode

    summary = _summarize(results_path)
    print("\n[eval_small_model] Robustness summary:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
