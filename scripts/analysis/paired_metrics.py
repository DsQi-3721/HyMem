"""Paired-metric comparisons (§A/§B/§C) — the head-to-head analyses the
paper subsections need to argue HyMem's specific module-level advantages.

§A — Context management: HyMem fold ↔ ReSum mid-trajectory summary
§B — Memory framework: HyMem typed memory ↔ A-MEM agentic notes
§C — Isolated reasoning: HyMem branch ↔ Thread spawn

Reads the same trace JSONs as summary_all.py and outputs side-by-side
tables that directly answer the reviewer-pointed questions.
"""
from __future__ import annotations
import argparse
import json
from glob import glob
from pathlib import Path
from typing import Dict, List


def load(directory: str) -> List[Dict]:
    out = []
    for p in sorted(glob(f"{directory}/*_trace.json")):
        try:
            out.append(json.load(open(p)))
        except Exception:
            pass
    return out


def stats_of(traces: List[Dict], key: str, default=0):
    """Return list of stats[key] across traces, with safe default."""
    return [t.get("stats", {}).get(key, default) for t in traces]


def section_a_context_mgmt(hymem: List[Dict], resum: List[Dict]):
    """HyMem fold ↔ ReSum summary trigger frequency, reset volume."""
    print("=" * 70)
    print("§A — Context management head-to-head")
    print("=" * 70)
    print(f"{'Metric':<35} {'HyMem':>15} {'ReSum':>15}")
    print("-" * 70)

    # Trigger count per task
    hf = stats_of(hymem, "fold_count")
    rf = stats_of(resum, "resum_summaries_fired")
    print(f"{'Reset count / task (avg)':<35} {sum(hf)/max(len(hf),1):>15.2f} "
          f"{sum(rf)/max(len(rf),1):>15.2f}")
    print(f"{'% tasks with ≥1 reset':<35} "
          f"{sum(1 for x in hf if x>0)/max(len(hf),1)*100:>14.0f}% "
          f"{sum(1 for x in rf if x>0)/max(len(rf),1)*100:>14.0f}%")

    # Avg tokens reset away per fold/summary event
    h_evts = []
    for t in hymem:
        for ev in t.get("stats", {}).get("hymem_fold_events", []) or []:
            tb = ev.get("tokens_before", 0); ta = ev.get("tokens_after_reset", 0)
            if tb > 0 and ta >= 0:
                h_evts.append(tb - ta)
    r_evts = []
    for t in resum:
        for ev in t.get("stats", {}).get("resum_summary_events", []) or []:
            tb = ev.get("tokens_before", 0); ta = ev.get("tokens_after_reset", 0)
            if tb > 0 and ta >= 0:
                r_evts.append(tb - ta)
    if h_evts or r_evts:
        h_avg = sum(h_evts)/max(len(h_evts),1) if h_evts else float('nan')
        r_avg = sum(r_evts)/max(len(r_evts),1) if r_evts else float('nan')
        print(f"{'Avg tokens reset per event':<35} {h_avg:>15,.0f} {r_avg:>15,.0f}")
        print(f"{'(N events recorded)':<35} {len(h_evts):>15d} {len(r_evts):>15d}")
    else:
        print(f"{'(No fold/summary events recorded —':<35}")
        print(f"{' likely G1/ReSum trace fields missing on this run)':<35}")

    # Pass@1 / total cost (already in summary_all but useful here)
    h_p = sum(1 for t in hymem if t.get("score",0)>0)/max(len(hymem),1)
    r_p = sum(1 for t in resum if t.get("score",0)>0)/max(len(resum),1)
    print(f"{'Pass@1':<35} {h_p:>15.3f} {r_p:>15.3f}")
    h_tok = sum(stats_of(hymem, "total_tokens"))/max(len(hymem),1)
    r_tok = sum(stats_of(resum, "total_tokens"))/max(len(resum),1)
    print(f"{'Avg total tokens / task':<35} {h_tok:>15,.0f} {r_tok:>15,.0f}")
    print()


def section_b_memory(hymem: List[Dict], amem: List[Dict]):
    """HyMem typed memory ↔ A-MEM agentic notes."""
    print("=" * 70)
    print("§B — Memory framework head-to-head")
    print("=" * 70)
    print(f"{'Metric':<40} {'HyMem':>13} {'A-MEM':>13}")
    print("-" * 70)

    # Memory writes
    h_writes = []
    for t in hymem:
        op = t.get("stats", {}).get("hymem_memory_ops", {}) or {}
        if op:
            h_writes.append(
                op.get("episode_writes", 0) + op.get("working_writes", 0)
                + op.get("tool_writes", 0)
            )
    a_writes = stats_of(amem, "amem_notes_created")
    print(f"{'Memory writes / task (avg)':<40} "
          f"{sum(h_writes)/max(len(h_writes),1):>13.1f} "
          f"{sum(a_writes)/max(len(a_writes),1):>13.1f}")

    # Memory retrievals
    h_ret = []
    for t in hymem:
        op = t.get("stats", {}).get("hymem_memory_ops", {}) or {}
        if op:
            h_ret.append(op.get("retrievals", 0))
    a_ret = stats_of(amem, "amem_retrievals")
    print(f"{'Memory retrievals / task (avg)':<40} "
          f"{sum(h_ret)/max(len(h_ret),1):>13.1f} "
          f"{sum(a_ret)/max(len(a_ret),1):>13.1f}")

    # Pass@1 / cost
    h_p = sum(1 for t in hymem if t.get("score",0)>0)/max(len(hymem),1)
    a_p = sum(1 for t in amem if t.get("score",0)>0)/max(len(amem),1)
    print(f"{'Pass@1':<40} {h_p:>13.3f} {a_p:>13.3f}")
    h_tok = sum(stats_of(hymem, "total_tokens"))/max(len(hymem),1)
    a_tok = sum(stats_of(amem, "total_tokens"))/max(len(amem),1)
    print(f"{'Avg total tokens / task':<40} {h_tok:>13,.0f} {a_tok:>13,.0f}")

    # Cost ratio: tokens per memory op (lower = more efficient memory)
    if h_writes and a_writes:
        h_eff = h_tok / max(sum(h_writes)/max(len(h_writes),1), 1e-6)
        a_eff = a_tok / max(sum(a_writes)/max(len(a_writes),1), 1e-6)
        print(f"{'Tokens per memory write (lower=better)':<40} "
              f"{h_eff:>13,.0f} {a_eff:>13,.0f}")
    print()


def section_c_isolated_reasoning(hymem: List[Dict], thread: List[Dict]):
    """HyMem branch ↔ Thread spawn."""
    print("=" * 70)
    print("§C — Isolated reasoning head-to-head")
    print("=" * 70)
    print(f"{'Metric':<40} {'HyMem':>13} {'Thread':>13}")
    print("-" * 70)

    h_calls = stats_of(hymem, "branch_calls")
    h_succ = stats_of(hymem, "branch_success")
    t_calls = stats_of(thread, "thread_spawn_count")
    t_succ = stats_of(thread, "thread_spawn_success")

    print(f"{'Sub-problem invocations / task':<40} "
          f"{sum(h_calls)/max(len(h_calls),1):>13.2f} "
          f"{sum(t_calls)/max(len(t_calls),1):>13.2f}")
    print(f"{'Successful invocations / task':<40} "
          f"{sum(h_succ)/max(len(h_succ),1):>13.2f} "
          f"{sum(t_succ)/max(len(t_succ),1):>13.2f}")

    # Success rate (only over tasks that triggered)
    h_total = sum(h_calls); h_total_s = sum(h_succ)
    t_total = sum(t_calls); t_total_s = sum(t_succ)
    h_rate = h_total_s / max(h_total, 1) if h_total else float('nan')
    t_rate = t_total_s / max(t_total, 1) if t_total else float('nan')
    print(f"{'Success rate (succ/total invocations)':<40} "
          f"{h_rate:>13.3f} {t_rate:>13.3f}")
    print(f"{'  (N total invocations)':<40} {h_total:>13d} {t_total:>13d}")

    # Pass@1
    h_p = sum(1 for t in hymem if t.get("score",0)>0)/max(len(hymem),1)
    t_p = sum(1 for t in thread if t.get("score",0)>0)/max(len(thread),1)
    print(f"{'Pass@1':<40} {h_p:>13.3f} {t_p:>13.3f}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs")
    parser.add_argument("--hymem_dirs", nargs="+",
                        default=["gaia_L1_optimized", "gaia_L2_optimized", "gaia_L3_optimized"])
    parser.add_argument("--resum_dir", default="gaia_resum")
    parser.add_argument("--thread_dir", default="gaia_thread")
    parser.add_argument("--amem_dir", default="gaia_amem")
    args = parser.parse_args(argv)

    root = Path(args.root)
    hymem = sum((load(str(root / d)) for d in args.hymem_dirs), [])
    resum = load(str(root / args.resum_dir))
    thread = load(str(root / args.thread_dir))
    amem = load(str(root / args.amem_dir))

    print(f"Loaded: HyMem={len(hymem)} ReSum={len(resum)} "
          f"Thread={len(thread)} A-MEM={len(amem)}\n")

    section_a_context_mgmt(hymem, resum)
    section_b_memory(hymem, amem)
    section_c_isolated_reasoning(hymem, thread)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
