"""Figure 3 — aggregate per-task token / tool-call / accuracy metrics across 5 baselines.

Reads pass@3 traces for HyMem, ReSum, Thread, A-MEM and the single-run ReAct
pilot on GAIA L1+L2+L3 and emits a single tidy CSV that drives Figure 3.

Output schema (one row per task-run):
    task_id              str
    level                int   1, 2, or 3
    method               str   {hymem, resum, thread, amem, react}
    run                  int   1, 2, 3   (ReAct has only run=1)
    agent_turns          int
    total_tokens         int
    tool_calls_atomic    int
    score                float 0.0 or 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Tuple

import pandas as pd


HYMEM = Path(__file__).resolve().parents[2]   # repo/HyMem
LOGS  = HYMEM / "logs"
RESULTS = HYMEM / "results" / "gaia"


# Per-method data sources:
#   * 'pass3'        — trace dir with pass@3_run{1,2,3} subdirs (per-task JSONs)
#   * 'single_dir'   — one flat trace dir (single run)
#   * 'json_levels'  — three eval-output JSONs, one per GAIA level (L1/L2/L3)
METHOD_LAYOUT = {
    "hymem":  ("pass3", LOGS / "gaia_hymem",
               ("pass@3_run1", "pass@3_run2", "pass@3_run3")),
    "resum":  ("json_levels", [
        RESULTS / "resum" / "gaia_results_20260508_145000.json",   # L1 (42)
        RESULTS / "resum" / "gaia_results_20260507_011836.json",   # L2 (66)
        RESULTS / "resum" / "gaia_results_20260506_232436.json",   # L3 (19)
    ]),
    "thread": ("json_levels", [
        RESULTS / "thread" / "gaia_results_20260507_222627.json",  # L1 (42)
        RESULTS / "thread" / "gaia_results_20260508_012151.json",  # L2 (66)
        RESULTS / "thread" / "gaia_results_20260508_122605.json",  # L3 (19)
    ]),
    "amem":   ("json_levels", [
        RESULTS / "amem"   / "gaia_results_20260507_230010.json",  # L1 (42)
        RESULTS / "amem"   / "gaia_results_20260508_031755.json",  # L2 (66)
        RESULTS / "amem"   / "gaia_results_20260508_125007.json",  # L3 (19)
    ]),
    "react":  ("single_dir", LOGS / "fig1_pilot" / "react"),
    "summary": ("json_levels", [
        RESULTS / "summary" / "gaia_results_20260521_185951.json",
    ]),
}


def _iter_traces(run_dir: Path) -> Iterator[Tuple[str, dict]]:
    if not run_dir.is_dir():
        return
    for f in sorted(run_dir.glob("gaia_*_trace.json")):
        tid = f.name[len("gaia_"):-len("_trace.json")]
        try:
            with f.open() as fh:
                yield tid, json.load(fh)
        except Exception as e:
            print(f"  [warn] skip {f}: {e}", file=sys.stderr)


def _record(method: str, run: int, tid: str, data: dict) -> dict:
    """Pull scalar metrics from either:
      * a per-task trace.json   — fields live under data["stats"]
      * a per-task result entry from gaia_results_*.json — fields live at top
    """
    stats = data.get("stats", {}) or {}
    def pick(key, default=0):
        v = data.get(key)
        if v is None:
            v = stats.get(key, default)
        return v if v is not None else default
    return {
        "task_id":           tid,
        "level":             int(pick("level", 0) or 0),
        "method":            method,
        "run":               run,
        "agent_turns":       int(pick("agent_turns", 0) or 0),
        "total_tokens":      int(pick("total_tokens", 0) or 0),
        "tool_calls_atomic": int(pick("tool_calls_atomic",
                                        pick("tool_calls_total", 0)) or 0),
        "score":             float(pick("score", 0.0) or 0.0),
    }


def _iter_json_levels(file_paths: list) -> Iterator[Tuple[str, dict]]:
    """Yield (task_id, per-task entry dict) across one or more
    gaia_results_*.json files (each file = one GAIA level slice)."""
    for fp in file_paths:
        if not Path(fp).is_file():
            print(f"  [warn] missing {fp}", file=sys.stderr); continue
        with open(fp) as fh:
            r = json.load(fh)
        for entry in r.get("results", []) or []:
            tid = entry.get("instance_id") or entry.get("task_id")
            if tid:
                yield str(tid), entry


def collect() -> pd.DataFrame:
    rows: List[dict] = []
    for method, layout in METHOD_LAYOUT.items():
        kind = layout[0]
        if kind == "pass3":
            _, root, runs = layout
            for run in runs:
                run_idx = int(run.split("run")[-1])
                n_seen = 0
                for tid, data in _iter_traces(root / run):
                    rows.append(_record(method, run_idx, tid, data))
                    n_seen += 1
                print(f"  {method:<7} {run}: {n_seen} traces")
        elif kind == "single_dir":
            _, root = layout
            n_seen = 0
            for tid, data in _iter_traces(root):
                rows.append(_record(method, 1, tid, data))
                n_seen += 1
            print(f"  {method:<7} single-run     : {n_seen} traces")
        elif kind == "json_levels":
            _, files = layout
            n_seen = 0
            for tid, entry in _iter_json_levels(files):
                rows.append(_record(method, 1, tid, entry))
                n_seen += 1
            print(f"  {method:<7} 3-level JSONs  : {n_seen} entries")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HYMEM / "data" / "figure3" / "per_task.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = collect()
    print(f"\n[total rows] {len(df)}")

    # HyMem pass@3 (3×127) + resum/thread/amem single (3×127) +
    # ReAct single dir (127) + Summary single JSON (127)
    expected = 1 * 3 * 127 + 3 * 127 + 1 * 127 + 1 * 127
    if len(df) != expected:
        print(f"[warn] expected {expected} rows, got {len(df)}", file=sys.stderr)

    dup = df.duplicated(subset=["task_id", "method", "run"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate (task_id,method,run) rows", file=sys.stderr)

    if out.suffix == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"[saved] {out}")

    print("\n[Pass@1 per method, mean over available runs]")
    agg = df.groupby("method")["score"].mean().sort_values(ascending=False)
    for m, v in agg.items():
        print(f"  {m:<7} {v*100:.1f}%")

    print("\n[median total_tokens per method]")
    med = df.groupby("method")["total_tokens"].median().sort_values()
    for m, v in med.items():
        print(f"  {m:<7} {int(v):>10,}")

    print("\n[median tool_calls_atomic per method]")
    med2 = df.groupby("method")["tool_calls_atomic"].median().sort_values()
    for m, v in med2.items():
        print(f"  {m:<7} {int(v):>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
