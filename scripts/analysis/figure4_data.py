"""Figure 4 — aggregate per-task token / tool-call / accuracy on BrowseComp-Plus.

Reads the 6 BCP+ result JSONs (one per method) and emits a tidy CSV that
drives Figure 4 (the BCP+ analogue of Figure 3).

Output schema (one row per task per method, 6 methods × 150 tasks = 900 rows):
    task_id              str
    method               str   {hymem, resum, thread, amem, react, summary}
    run                  int   1  (single run per method on BCP+)
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


HYMEM = Path(__file__).resolve().parents[2]
RESULTS = HYMEM / "results" / "bcplus"

# One result JSON per method (BCP+ is a single-run setup).
METHOD_FILES = {
    "hymem":   RESULTS / "hymem"   / "bcplus_hymem_results_20260520_061055.json",
    "resum":   RESULTS / "resum"   / "bcplus_resum_results_20260518_164956.json",
    "thread":  RESULTS / "thread"  / "bcplus_thread_results_20260518_190907.json",
    "amem":    RESULTS / "amem"    / "bcplus_amem_results_20260518_164639.json",
    "react":   RESULTS / "react"   / "bcplus_react_results_20260518_164630.json",
    "summary": RESULTS / "summary" / "bcplus_react_results_20260521_162911.json",
}


def _iter_entries(file_path: Path) -> Iterator[Tuple[str, dict]]:
    if not file_path.is_file():
        print(f"  [warn] missing {file_path}", file=sys.stderr); return
    with file_path.open() as fh:
        r = json.load(fh)
    for entry in r.get("results", []) or []:
        tid = entry.get("instance_id") or entry.get("task_id")
        if tid:
            yield str(tid), entry


def _record(method: str, tid: str, entry: dict) -> dict:
    return {
        "task_id":           tid,
        "method":            method,
        "run":               1,
        "total_tokens":      int(entry.get("total_tokens", 0) or 0),
        "tool_calls_atomic": int(entry.get("tool_calls_atomic",
                                            entry.get("tool_calls_total", 0)) or 0),
        "score":             float(entry.get("score", 0.0) or 0.0),
    }


def collect() -> pd.DataFrame:
    rows: List[dict] = []
    for method, fp in METHOD_FILES.items():
        n = 0
        for tid, entry in _iter_entries(fp):
            rows.append(_record(method, tid, entry))
            n += 1
        print(f"  {method:<8} {fp.name:<45} {n} entries")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HYMEM / "data" / "figure4" / "per_task.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = collect()
    print(f"\n[total rows] {len(df)}")

    expected = 6 * 150
    if len(df) != expected:
        print(f"[warn] expected {expected} rows, got {len(df)}", file=sys.stderr)

    dup = df.duplicated(subset=["task_id", "method", "run"]).sum()
    if dup:
        print(f"[warn] {dup} duplicate (task_id,method,run) rows", file=sys.stderr)

    df.to_csv(out, index=False)
    print(f"[saved] {out}")

    print("\n[Pass@1 per method, single run]")
    agg = df.groupby("method")["score"].mean().sort_values(ascending=False)
    for m, v in agg.items():
        print(f"  {m:<8} {v*100:.1f}%")

    print("\n[median total_tokens per method]")
    med = df.groupby("method")["total_tokens"].median().sort_values()
    for m, v in med.items():
        print(f"  {m:<8} {int(v):>12,}")

    print("\n[median tool_calls_atomic per method]")
    med2 = df.groupby("method")["tool_calls_atomic"].median().sort_values()
    for m, v in med2.items():
        print(f"  {m:<8} {int(v):>4}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
