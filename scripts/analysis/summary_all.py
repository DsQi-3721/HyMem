"""Master summary across all 4 frameworks × 3 GAIA levels.

Reads per-task `*_trace.json` files (more reliable than the aggregate
`gaia_results_*.json` which only contains the LAST batch run for each
framework). Outputs:
  - Per-framework × per-level Pass@1 / tokens / tool calls / time table
  - Per-framework total
  - Cross-framework one-glance comparison

Usage:
    python scripts/analysis/summary_all.py                          # default dirs
    python scripts/analysis/summary_all.py --hymem_dirs A B C       # custom
    python scripts/analysis/summary_all.py --output summary.json    # also dump JSON
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List


def load_traces(directory: str) -> List[Dict]:
    out = []
    for p in sorted(glob(f"{directory}/*_trace.json")):
        try:
            out.append(json.load(open(p)))
        except Exception as e:
            print(f"  [warn] skipped {p}: {e}")
    return out


def summarize(traces: List[Dict]) -> Dict:
    n = len(traces)
    if n == 0:
        return {}
    correct = sum(1 for t in traces if t.get("score", 0) > 0)
    fields = ["total_tokens", "total_prompt_tokens", "total_completion_tokens",
              "tool_calls_atomic", "agent_turns", "session_time"]
    avg = {}
    for f in fields:
        avg[f] = sum(t.get("stats", {}).get(f, 0) for t in traces) / n
    return {
        "n": n,
        "correct": correct,
        "pass@1": correct / n,
        "avg": avg,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs",
                        help="Root dir containing per-framework log dirs")
    parser.add_argument("--frameworks", nargs="+",
                        default=["HyMem", "ReSum", "Thread", "A-MEM"])
    parser.add_argument("--hymem_dirs", nargs="+",
                        default=["gaia_L1_optimized", "gaia_L2_optimized", "gaia_L3_optimized"])
    parser.add_argument("--resum_dir", default="gaia_resum")
    parser.add_argument("--thread_dir", default="gaia_thread")
    parser.add_argument("--amem_dir", default="gaia_amem")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional: dump aggregated dict as JSON")
    args = parser.parse_args(argv)

    root = Path(args.root)
    fw_dirs = {
        "HyMem":  [str(root / d) for d in args.hymem_dirs],
        "ReSum":  [str(root / args.resum_dir)],
        "Thread": [str(root / args.thread_dir)],
        "A-MEM":  [str(root / args.amem_dir)],
    }

    # Group traces by framework + level
    by_fw_level: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for fw, dlist in fw_dirs.items():
        for d in dlist:
            for t in load_traces(d):
                lvl = str(t.get("level") or t.get("stats", {}).get("level") or "?")
                by_fw_level[fw][lvl].append(t)

    print("=" * 92)
    print(f"{'Framework':<10} {'Level':<6} {'N':>4} {'Pass@1':>7} {'AvgTok':>10} "
          f"{'AvgAtomic':>10} {'AvgTurns':>9} {'AvgTime(s)':>11}")
    print("=" * 92)

    all_summaries = {}
    for fw in args.frameworks:
        levels = sorted([k for k in by_fw_level[fw].keys() if k != "?"])
        if "?" in by_fw_level[fw]:
            levels.append("?")
        all_traces = []
        for lvl in levels:
            ts = by_fw_level[fw][lvl]
            all_traces.extend(ts)
            s = summarize(ts)
            if not s:
                continue
            print(f"{fw:<10} L{lvl:<5} {s['n']:>4} {s['pass@1']:>7.3f} "
                  f"{s['avg']['total_tokens']:>10,.0f} "
                  f"{s['avg']['tool_calls_atomic']:>10.1f} "
                  f"{s['avg']['agent_turns']:>9.1f} "
                  f"{s['avg']['session_time']:>10.0f}")
            all_summaries[f"{fw}_L{lvl}"] = s
        # Overall row
        s = summarize(all_traces)
        if s:
            print(f"{fw:<10} {'TOTAL':<6} {s['n']:>4} {s['pass@1']:>7.3f} "
                  f"{s['avg']['total_tokens']:>10,.0f} "
                  f"{s['avg']['tool_calls_atomic']:>10.1f} "
                  f"{s['avg']['agent_turns']:>9.1f} "
                  f"{s['avg']['session_time']:>10.0f}")
            all_summaries[f"{fw}_TOTAL"] = s
        print("-" * 92)

    # One-glance side-by-side
    print()
    print("=" * 60)
    print("One-glance Pass@1 comparison (TOTAL across all levels)")
    print("=" * 60)
    print(f"{'Framework':<10} {'N':>4} {'Pass@1':>10} {'Cost-Tok':>14} {'Time/task':>11}")
    for fw in args.frameworks:
        s = all_summaries.get(f"{fw}_TOTAL")
        if s:
            print(f"{fw:<10} {s['n']:>4} {s['pass@1']:>10.3f} "
                  f"{s['avg']['total_tokens']:>13,.0f} "
                  f"{s['avg']['session_time']:>10.0f}s")

    if args.output:
        # Strip individual traces for JSON dump (too big)
        out = {k: {kk: v for kk, v in s.items() if kk != "traces"}
               for k, s in all_summaries.items()}
        args.output.write_text(json.dumps(out, indent=2))
        print(f"\nJSON summary written to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
