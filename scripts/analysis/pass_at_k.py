"""Pass@K aggregator for GAIA evaluation across K independent runs.

Reads K aggregate `gaia_results_*.json` files (one per run) of the same
baseline and reports Pass@1 / Pass@K / stability statistics, overall and
per GAIA level.

Two CLI modes:

  Mode A — single baseline (auto-glob run dirs)
    python scripts/analysis/pass_at_k.py \\
        --baseline-dir results/gaia/resum --pattern "pass@3_run*" \\
        --label resum --output results/gaia/resum/pass_at_k.json --markdown

  Mode B — multi-baseline cross-comparison
    python scripts/analysis/pass_at_k.py \\
        --group resum  results/gaia/resum/pass@3_run1 ... pass@3_run3 \\
        --group thread results/gaia/thread/pass@3_run1 ... pass@3_run3 \\
        --output results/gaia/pass_at_k_compare.json --markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple


def _newest_results_json(directory: Path) -> Path:
    candidates = sorted(directory.glob("gaia_results_*.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no gaia_results_*.json in {directory}")
    return candidates[0]


def _load_run(path: Path) -> Dict[str, Dict]:
    """Return {instance_id: per-task dict} for a single run file."""
    with path.open() as f:
        data = json.load(f)
    out: Dict[str, Dict] = {}
    for r in data.get("results", []):
        iid = r.get("instance_id")
        if iid is None:
            continue
        out[iid] = r
    return out


def _is_correct(score) -> int:
    try:
        return 1 if float(score) > 0 else 0
    except (TypeError, ValueError):
        return 0


def _safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _aggregate_group(label: str, dirs: List[Path]) -> Dict:
    """Compute Pass@K stats for one baseline (a list of run dirs)."""
    K = len(dirs)
    if K < 2:
        print(f"[warn] group '{label}': only {K} run(s) — Pass@K trivially equals Pass@1",
              file=sys.stderr)

    source_files = [_newest_results_json(d) for d in dirs]
    runs = [_load_run(p) for p in source_files]

    id_sets = [set(r.keys()) for r in runs]
    common = set.intersection(*id_sets) if id_sets else set()
    union = set.union(*id_sets) if id_sets else set()
    missing = union - common
    if missing:
        print(f"[warn] group '{label}': {len(missing)} instance(s) not in all "
              f"{K} runs — dropping. example: {sorted(missing)[:3]}", file=sys.stderr)

    by_level: Dict[str, List[Dict]] = defaultdict(list)
    per_task: List[Dict] = []
    flat_avg_pool: Dict[str, List[float]] = defaultdict(list)

    for iid in sorted(common):
        scores = [_is_correct(r[iid].get("score")) for r in runs]
        # Use level from first run; assume identical across runs.
        level = str(runs[0][iid].get("level", "?"))
        rec = {
            "instance_id": iid,
            "level": level,
            "scores": scores,
            "pass@K": int(any(scores)),
            "all_K": int(all(scores)),
        }
        per_task.append(rec)
        by_level[level].append(rec)

        for r in runs:
            t = r[iid]
            for f in ("total_tokens", "tool_calls_atomic",
                      "agent_turns", "session_time"):
                v = t.get(f)
                if isinstance(v, (int, float)):
                    flat_avg_pool[f].append(float(v))

    def _stats(records: List[Dict]) -> Dict:
        n = len(records)
        if n == 0:
            return {"n": 0, "pass@1": 0.0, "pass@K": 0.0,
                    "all_K": 0.0, "none": 0.0, "flaky": 0.0}
        pass_at_1 = _safe_mean([s for r in records for s in r["scores"]])
        pass_at_K = _safe_mean([r["pass@K"] for r in records])
        all_K = _safe_mean([r["all_K"] for r in records])
        none = _safe_mean([1 - r["pass@K"] for r in records])
        flaky = max(0.0, 1.0 - all_K - none)
        return {"n": n, "pass@1": pass_at_1, "pass@K": pass_at_K,
                "all_K": all_K, "none": none, "flaky": flaky}

    overall = _stats(per_task)
    by_level_stats = {lvl: _stats(recs) for lvl, recs in sorted(by_level.items())}

    avg_per_task = {f: _safe_mean(v) for f, v in flat_avg_pool.items()}

    return {
        "label": label,
        "n_tasks": len(per_task),
        "n_runs": K,
        "source_files": [str(p) for p in source_files],
        "overall": overall,
        "by_level": by_level_stats,
        "avg_per_task": avg_per_task,
        "per_task": per_task,
    }


def _print_group(g: Dict) -> None:
    print(f"\n### Group: {g['label']}  (K={g['n_runs']}, N={g['n_tasks']})\n")
    print("| Level | N   | Pass@1 | Pass@K | All-K | None  | Flaky |")
    print("|-------|-----|--------|--------|-------|-------|-------|")
    for lvl in sorted(g["by_level"].keys()):
        s = g["by_level"][lvl]
        print(f"| {lvl:<5} | {s['n']:<3} | "
              f"{s['pass@1']:.3f}  | {s['pass@K']:.3f}  | "
              f"{s['all_K']:.3f} | {s['none']:.3f} | {s['flaky']:.3f} |")
    s = g["overall"]
    print(f"| All   | {s['n']:<3} | "
          f"{s['pass@1']:.3f}  | {s['pass@K']:.3f}  | "
          f"{s['all_K']:.3f} | {s['none']:.3f} | {s['flaky']:.3f} |")
    a = g["avg_per_task"]
    print()
    print(f"Avg per task across all (K runs × N tasks):")
    if "total_tokens" in a:
        print(f"  total_tokens   = {a['total_tokens']:.0f}")
    if "tool_calls_atomic" in a:
        print(f"  tool_calls     = {a['tool_calls_atomic']:.2f}")
    if "agent_turns" in a:
        print(f"  agent_turns    = {a['agent_turns']:.2f}")
    if "session_time" in a:
        print(f"  session_time_s = {a['session_time']:.1f}")
    print(f"\nSource files:")
    for p in g["source_files"]:
        print(f"  - {p}")


def _print_cross_group_table(groups: List[Dict]) -> None:
    if len(groups) < 2:
        return
    print("\n### Cross-baseline summary\n")
    print("| Baseline | N   | Pass@1 | Pass@K | All-K | None  | Flaky |")
    print("|----------|-----|--------|--------|-------|-------|-------|")
    for g in groups:
        s = g["overall"]
        print(f"| {g['label']:<8} | {s['n']:<3} | "
              f"{s['pass@1']:.3f}  | {s['pass@K']:.3f}  | "
              f"{s['all_K']:.3f} | {s['none']:.3f} | {s['flaky']:.3f} |")


def _resolve_groups(args) -> List[Tuple[str, List[Path]]]:
    if args.group:
        out: List[Tuple[str, List[Path]]] = []
        for spec in args.group:
            if len(spec) < 2:
                raise SystemExit(
                    f"--group needs LABEL + at least 1 dir, got: {spec}")
            label, *dirs = spec
            out.append((label, [Path(d) for d in dirs]))
        return out

    if args.baseline_dir:
        base = Path(args.baseline_dir)
        run_dirs = sorted(base.glob(args.pattern))
        run_dirs = [d for d in run_dirs if d.is_dir()]
        if not run_dirs:
            raise SystemExit(
                f"no dirs matched {base}/{args.pattern}")
        label = args.label or base.name
        return [(label, run_dirs)]

    raise SystemExit(
        "must pass either --baseline-dir (+ --pattern) or one or more --group")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate Pass@K across K independent eval runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--baseline-dir",
                        help="Single-baseline mode: dir containing run subdirs.")
    parser.add_argument("--pattern", default="pass@*_run*",
                        help="Glob (relative to --baseline-dir) for run dirs. "
                             "Default: 'pass@*_run*'")
    parser.add_argument("--label",
                        help="Group label for single-baseline mode "
                             "(default: basename of --baseline-dir).")
    parser.add_argument("--group", action="append", nargs="+", metavar="ARG",
                        help="Multi-baseline mode (repeatable): "
                             "LABEL DIR1 DIR2 ...  Each DIR must contain a "
                             "gaia_results_*.json.")
    parser.add_argument("--output", type=Path,
                        help="Optional path to dump aggregated JSON.")
    parser.add_argument("--markdown", action="store_true",
                        help="Print Markdown tables to stdout.")
    args = parser.parse_args(argv)

    specs = _resolve_groups(args)
    groups = [_aggregate_group(label, dirs) for label, dirs in specs]

    K_set = {g["n_runs"] for g in groups}
    K = max(K_set) if len(K_set) == 1 else None

    if args.markdown:
        title_K = f"K={K}" if K else f"K varies {sorted(K_set)}"
        print(f"## Pass@K Report ({title_K})")
        for g in groups:
            _print_group(g)
        _print_cross_group_table(groups)

    if args.output:
        payload = {
            "K": K,
            "groups": {g["label"]: g for g in groups},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[written] {args.output}")


if __name__ == "__main__":
    main()
