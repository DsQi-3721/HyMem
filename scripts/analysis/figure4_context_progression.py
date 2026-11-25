"""Figure 4 Panel (c) — reconstruct per-turn CUMULATIVE LLM API tokens per
method on BrowseComp-Plus.

Same definition as the Figure 3 counterpart: API tokens = sum over all LLM
calls so far of (prompt_tokens + completion_tokens). Y axis of Panel (c) in
Figure 4 then matches the X axis of Panel (a).

  * HyMem-Coord  -> event_log.token_usage.calls filtered to component=='planner'
  * HyMem-Total  -> event_log.token_usage.calls (all components)
  * ReSum / A-MEM / ReAct
                  -> reconstructed from `messages`: each assistant message k
                     gave one LLM call with prompt = messages[0:k] and
                     completion = messages[k]; cumulative sum.

Thread is omitted (its sub-thread state lives outside the main trace).
Summary is omitted to keep the legend manageable (5 curves, matching Figure 3).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


HYMEM = Path(__file__).resolve().parents[2]
LOGS  = HYMEM / "logs"

PROGRESS_GRID = np.linspace(0.0, 1.0, 21)

# Methods reconstructed from baseline `messages` lists
METHOD_SOURCES = {
    "resum": LOGS / "bcplus_resum",
    "amem":  LOGS / "bcplus_amem",
    "react": LOGS / "bcplus_react",
}
HYMEM_DIR = LOGS / "bcplus_hymem"


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def _cumulative_api_from_messages(messages: List[dict], tokenizer
                                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Iterate the chat history, treat each assistant index as one LLM call,
    and return (progress_raw, cumulative_api)."""
    prefix_tokens = [0]
    for m in messages:
        if not isinstance(m, dict):
            prefix_tokens.append(prefix_tokens[-1]); continue
        content = str(m.get("content", "") or "")
        n = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0
        prefix_tokens.append(prefix_tokens[-1] + n)
    call_indices = [i for i, m in enumerate(messages)
                    if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(call_indices) < 2:
        return None
    cum, prog = [], []
    running = 0
    for ord_idx, k in enumerate(call_indices):
        prompt_tok     = prefix_tokens[k]
        completion_tok = prefix_tokens[k + 1] - prefix_tokens[k]
        running       += prompt_tok + completion_tok
        cum.append(running)
        prog.append((ord_idx + 1) / len(call_indices))
    return np.array(prog), np.array(cum, dtype=float)


def _cumulative_api_from_calls(trace: dict,
                               component_filter: Optional[str] = None
                               ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    el = trace.get("event_log") or {}
    calls = ((el.get("token_usage") or {}).get("calls") or [])
    if component_filter:
        calls = [c for c in calls if c.get("component") == component_filter]
    if len(calls) < 2:
        return None
    ts = np.array([float(c.get("timestamp", 0)) for c in calls], dtype=float)
    tk = np.array([int(c.get("total_tokens", 0)) for c in calls], dtype=float)
    cum = np.cumsum(tk)
    t0, t1 = ts[0], ts[-1]
    if t1 <= t0:
        return None
    prog = (ts - t0) / (t1 - t0)
    return prog, cum


def _resample(progress_raw: np.ndarray, values: np.ndarray
              ) -> Optional[np.ndarray]:
    if len(progress_raw) < 2:
        return None
    return np.interp(PROGRESS_GRID, progress_raw, values).astype(float)


def collect(tokenizer) -> pd.DataFrame:
    rows = []

    # --- baselines from message-prefix reconstruction ---
    for method, run_dir in METHOD_SOURCES.items():
        if not run_dir.is_dir():
            print(f"  [warn] missing dir {run_dir}", file=sys.stderr); continue
        files = sorted(run_dir.glob("bcplus_*_trace.json"))
        n_kept, n_skipped = 0, 0
        for f in files:
            tid = f.name.replace("_trace.json", "")
            if tid.startswith("bcplus_"):
                tid = tid[len("bcplus_"):]
            try:
                with f.open() as fh:
                    data = json.load(fh)
            except Exception:
                n_skipped += 1; continue
            msgs = data.get("messages", []) or []
            rec  = _cumulative_api_from_messages(msgs, tokenizer)
            if rec is None:
                n_skipped += 1; continue
            prog, cum = rec
            curve = _resample(prog, cum)
            if curve is None:
                n_skipped += 1; continue
            for p, v in zip(PROGRESS_GRID, curve):
                rows.append({"task_id": tid, "method": method,
                             "progress": float(p),
                             "context_tokens": float(v)})
            n_kept += 1
        print(f"  {method:<12} {run_dir.name:<22} kept={n_kept}  skipped={n_skipped}")

    # --- HyMem-Coord and HyMem-Total from event_log.calls ---
    files = sorted(HYMEM_DIR.glob("bcplus_*_trace.json"))
    for label, comp in (("hymem", "planner"), ("hymem_total", None)):
        n_kept, n_skipped = 0, 0
        for f in files:
            tid = f.name.replace("_trace.json", "")
            if tid.startswith("bcplus_"):
                tid = tid[len("bcplus_"):]
            try:
                with f.open() as fh:
                    data = json.load(fh)
            except Exception:
                n_skipped += 1; continue
            rec = _cumulative_api_from_calls(data, component_filter=comp)
            if rec is None:
                n_skipped += 1; continue
            prog, cum = rec
            curve = _resample(prog, cum)
            if curve is None:
                n_skipped += 1; continue
            for p, v in zip(PROGRESS_GRID, curve):
                rows.append({"task_id": tid, "method": label,
                             "progress": float(p),
                             "context_tokens": float(v)})
            n_kept += 1
        print(f"  {label:<12} {HYMEM_DIR.name:<22} kept={n_kept}  skipped={n_skipped}")

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HYMEM / "data" / "figure4" / "context_progression.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[loading tokenizer Qwen/Qwen2.5-7B-Instruct]")
    tok = _load_tokenizer()

    print("[reconstructing per-turn cumulative API tokens]")
    df = collect(tok)
    print(f"\n[total rows] {len(df)}")

    df.to_csv(out, index=False)
    print(f"[saved] {out}")

    print("\n[end-of-task (progress=1.0) median cumulative API tokens per method]")
    end = df[df["progress"] >= 0.999].groupby("method")["context_tokens"].median()
    for m, v in end.sort_values().items():
        print(f"  {m:<12} {int(v):>12,}")
    if "hymem" in end.index:
        h = end["hymem"]
        print("\n[ratio vs HyMem-Coord]")
        for m, v in end.sort_values().items():
            print(f"  {m:<12} {v/h:>6.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
