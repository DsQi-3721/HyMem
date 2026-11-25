"""Drill into a specific task's full trace.

Usage:
    python scripts/analysis/inspect_task.py <instance_id_prefix>
    python scripts/analysis/inspect_task.py 0383a3ee --framework hymem
    python scripts/analysis/inspect_task.py 0383a3ee --all  # show all 4 frameworks side-by-side
"""
from __future__ import annotations
import argparse
import json
from glob import glob
from pathlib import Path


FW_DIRS = {
    "hymem":  ["logs/gaia_L1_optimized", "logs/gaia_L2_optimized", "logs/gaia_L3_optimized"],
    "resum":  ["logs/gaia_resum"],
    "thread": ["logs/gaia_thread"],
    "amem":   ["logs/gaia_amem"],
}


def find_traces(instance_prefix: str, frameworks):
    """Find one trace file per framework matching the prefix."""
    out = {}
    for fw in frameworks:
        for d in FW_DIRS.get(fw, []):
            for p in glob(f"{d}/*_trace.json"):
                if instance_prefix.lower() in p.lower():
                    out[fw] = p
                    break
            if fw in out:
                break
    return out


def show_trace(label: str, path: str):
    t = json.load(open(path))
    s = t.get("stats", {})
    print(f"\n{'='*70}")
    print(f"{label}: {t['instance_id']}  (level {t.get('level','?')})")
    print(f"{'='*70}")
    print(f"PROBLEM:    {t.get('problem','')[:200]}")
    print(f"GROUND TRUTH: {t.get('ground_truth','')[:120]}")
    print(f"PREDICTION:  {t.get('prediction','')[:120]}")
    print(f"SCORE: {t.get('score',0)}   CONFIDENCE: {t.get('confidence',0):.2f}")
    print()
    print("STATS:")
    keys = ['agent_turns', 'tool_calls_atomic', 'tool_calls_by_source',
            'planner_turns', 'executor_calls', 'branch_calls', 'branch_success',
            'fold_count', 'replan_count',
            'total_tokens', 'total_prompt_tokens', 'total_completion_tokens',
            'session_time',
            # Paired metrics (if present)
            'hymem_fold_events', 'hymem_memory_ops',
            'resum_summaries_fired', 'resum_summary_events',
            'thread_spawn_count', 'thread_spawn_success',
            'amem_notes_created', 'amem_retrievals',
            'context_size_history']
    for k in keys:
        v = s.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}]" + (f"  first={v[0]}" if v else ""))
        elif isinstance(v, dict):
            print(f"  {k}: {dict(list(v.items())[:5])}{'...' if len(v)>5 else ''}")
        else:
            print(f"  {k}: {v}")

    # First few execution steps
    hist = t.get('hymem_history') or t.get('history') or []
    print(f"\nEXECUTION ({len(hist)} entries, showing first 5):")
    for h in hist[:5]:
        role = h.get('role', '?')
        if role == 'planner':
            print(f"  [planner T{h.get('turn','?')}] " + h.get('content', '')[:140])
        elif role == 'executor':
            print(f"  [executor T{h.get('turn','?')}] task={h.get('task','')[:60]} → {h.get('result','')[:80]}")
        elif role == 'branch':
            print(f"  [branch T{h.get('turn','?')}] {h.get('description','')[:50]} success={h.get('success')}")
        elif role == 'fold':
            print(f"  [fold T{h.get('turn','?')}]")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("instance", help="Instance ID prefix (e.g. '0383a3ee')")
    parser.add_argument("--framework", "-fw", default=None,
                        choices=["hymem", "resum", "thread", "amem"])
    parser.add_argument("--all", action="store_true",
                        help="Show all 4 frameworks side-by-side")
    args = parser.parse_args(argv)

    if args.all:
        fws = ["hymem", "resum", "thread", "amem"]
    elif args.framework:
        fws = [args.framework]
    else:
        fws = ["hymem"]

    traces = find_traces(args.instance, fws)
    if not traces:
        print(f"No trace found matching '{args.instance}' in {fws}")
        return 1
    for fw, path in traces.items():
        show_trace(fw.upper(), path)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
