"""Figure 3 Panel (c) — reconstruct per-turn CUMULATIVE LLM API tokens per method.

For each (method, task), produce a curve `progress -> cumulative API tokens`
where API tokens = sum over all LLM calls so far of (prompt_tokens + completion_tokens).

This is the SAME metric Panel (a) uses on its X axis (`stats.total_tokens`),
so HyMem's end-point in Panel (c) matches the HyMem dot in Panel (a).

  * HyMem-Coord  -> event_log.token_usage.calls filtered to component=='planner'
                    (the strategic-layer LLM consumption only).
  * HyMem-Total  -> event_log.token_usage.calls (all components combined).
  * ReSum/A-MEM  -> reconstructed from `messages`: each assistant message k
                    corresponds to one LLM call with prompt = messages[0:k] and
                    completion = messages[k]. Cumulative sum of those.
  * ReAct        -> reconstructed from `planner_context_per_turn[0]['dump']`
                    similarly (each assistant-role entry is one LLM call).

Output: HyMem/data/figure3/context_progression.csv
        columns: task_id, method, progress, context_tokens
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


HYMEM = Path(__file__).resolve().parents[2]
LOGS  = HYMEM / "logs"

PROGRESS_GRID = np.linspace(0.0, 1.0, 21)   # 0%, 5%, ..., 100%

# Methods whose Panel-(c) curve comes from message-prefix tokenization.
METHOD_SOURCES = {
    "hymem":  {"runs": [LOGS / "gaia_hymem" / "pass@3_run1"]},
    "resum":  {"runs": [LOGS / "resum"      / "pass@3_run1"]},
    "amem":   {"runs": [LOGS / "gaia_amem"  / "pass@3_run1"]},
    "react":  {"runs": [LOGS / "fig1_pilot" / "react"]},
}

# HyMem-Total is computed from event_log.token_usage.calls (cumulative
# total_tokens consumed across planner + executor + branch + memory
# calls). Uses the same gaia_hymem/pass@3_run1 traces.
HYMEM_TOTAL_DIR = LOGS / "gaia_hymem" / "pass@3_run1"


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)


def _baseline_messages(method: str, trace: dict) -> List[dict]:
    """Return the chat-history list a baseline-style agent worked with.
    ReAct lives in planner_context_per_turn[0]['dump']; ReSum/A-MEM in
    the top-level `messages` list."""
    if method in ("resum", "amem"):
        return trace.get("messages", []) or []
    if method == "react":
        pct = trace.get("planner_context_per_turn") \
              or trace.get("stats", {}).get("planner_context_per_turn") or []
        if pct and isinstance(pct, list) and isinstance(pct[0], dict):
            return pct[0].get("dump", []) or []
        return []
    return []


def _cumulative_api_from_messages(messages: List[dict], tokenizer
                                  ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Reconstruct cumulative LLM-API tokens from a single-tier chat trace.

    At each assistant message k, the LLM was invoked with prompt =
    messages[0:k] (the chat history so far) and emitted completion =
    messages[k]. The API cost for that call is the sum of the two
    token counts.

    Returns (progress, cumulative_api) where progress is in [0, 1] mapped
    by assistant-call index, and cumulative_api is the running sum.
    """
    # cache prefix token counts so we don't re-tokenize quadratically
    prefix_tokens = [0]
    for m in messages:
        if not isinstance(m, dict):
            prefix_tokens.append(prefix_tokens[-1]); continue
        content = str(m.get("content", "") or "")
        n = len(tokenizer.encode(content, add_special_tokens=False)) if content else 0
        prefix_tokens.append(prefix_tokens[-1] + n)

    # Identify "LLM call" indices: positions where an assistant message appears.
    # Each such call has prompt = messages[0:k] and completion = messages[k].
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
    """Reconstruct cumulative LLM-API tokens from event_log.token_usage.calls.

    `component_filter` of None counts every call (HyMem-Total). Setting it
    to 'planner' returns HyMem-Coord (strategic layer alone).
    Returns (progress, cumulative_api) where progress is timestamp-normalized.
    """
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
    """Interpolate (progress_raw, values) onto the fixed 21-point grid."""
    if len(progress_raw) < 2:
        return None
    return np.interp(PROGRESS_GRID, progress_raw, values).astype(float)


def collect(tokenizer) -> pd.DataFrame:
    rows = []

    # --- (1) ReSum / A-MEM / ReAct: cumulative-API from messages prefix ---
    for method, cfg in METHOD_SOURCES.items():
        if method == "hymem":
            continue  # handled below via event_log.calls
        for run_dir in cfg["runs"]:
            if not run_dir.is_dir():
                print(f"  [warn] missing dir {run_dir}", file=sys.stderr)
                continue
            files = sorted(run_dir.glob("gaia_*_trace.json"))
            n_kept, n_skipped = 0, 0
            for f in files:
                tid = f.name[len("gaia_"):-len("_trace.json")]
                try:
                    with f.open() as fh:
                        data = json.load(fh)
                except Exception:
                    n_skipped += 1; continue
                msgs = _baseline_messages(method, data)
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
            print(f"  {method:<13} {run_dir.name:<20} kept={n_kept}  skipped={n_skipped}")

    # --- (2) HyMem-Coord and HyMem-Total: from event_log.token_usage.calls ---
    files = sorted(HYMEM_TOTAL_DIR.glob("gaia_*_trace.json"))
    for label, comp_filter in (("hymem", "planner"), ("hymem_total", None)):
        n_kept, n_skipped = 0, 0
        for f in files:
            tid = f.name[len("gaia_"):-len("_trace.json")]
            try:
                with f.open() as fh:
                    data = json.load(fh)
            except Exception:
                n_skipped += 1; continue
            rec = _cumulative_api_from_calls(data, component_filter=comp_filter)
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
        print(f"  {label:<13} {HYMEM_TOTAL_DIR.name:<20} kept={n_kept}  skipped={n_skipped}")

    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HYMEM / "data" / "figure3" / "context_progression.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("[loading tokenizer Qwen/Qwen2.5-7B-Instruct]")
    tok = _load_tokenizer()

    print("[reconstructing per-turn cumulative context]")
    df = collect(tok)
    print(f"\n[total rows] {len(df)}  (~21 grid points × kept tasks × methods)")

    df.to_csv(out, index=False)
    print(f"[saved] {out}")

    print("\n[end-of-task (progress=1.0) median context_tokens per method]")
    end = df[df["progress"] >= 0.999].groupby("method")["context_tokens"].median()
    for m, v in end.sort_values().items():
        print(f"  {m:<7} {int(v):>9,}")

    print("\n[end-of-task ratio vs HyMem-Coord]")
    if "hymem" in end.index:
        h = end["hymem"]
        for m, v in end.sort_values().items():
            ratio = v / h if h > 0 else float("nan")
            print(f"  {m:<7} {ratio:>5.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
