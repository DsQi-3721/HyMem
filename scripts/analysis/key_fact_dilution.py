"""Figure-1 dilution experiment — Step 3.

For each trace file (HyMem or ReAct), compute per-planner-turn:
  - KFD_t (Key Fact Density)    = fact-token mass / |C_t|
  - KFSR_t (Survival Rate)      = present_facts / introduced_facts_by_t

Inputs:
  - data/key_facts/gaia.jsonl  (from extract_key_facts.py)
  - trace dir (HyMem: logs/gaia_hymem/..., ReAct: logs/gaia_thread/...
    or logs/fig1_smoke_react/...)

Schema notes:
  HyMem traces:
    stats.planner_context_per_turn = [{turn, context_length, messages:[{role,content}]}]
  Fold-agent (ReAct/Thread/ReSum) traces:
    stats.planner_context_per_turn = [{agent: 'main'|...,
                                       dump: [{turn_idx, role, content,
                                               cumul_token_count}]}]
    A "planner turn" in fold-agent = an `assistant` message in the main
    agent's dump; the input context at that turn = concat of all preceding
    messages.

Output:
  data/dilution/{run_id}_per_turn.parquet — cols:
    [task_id, condition, benchmark, turn, KFD, KFSR,
     context_token_len, n_facts_present, n_facts_introduced,
     n_facts_total, score]

Usage:
  python scripts/analysis/key_fact_dilution.py \
      --trace_dir logs/gaia_hymem/pass@3_run1 \
      --key_facts data/key_facts/gaia.jsonl \
      --condition hymem --benchmark gaia \
      --out data/dilution/hymem_gaia.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from transformers import AutoTokenizer


TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _load_key_facts(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return {task_id: entry} from a JSONL produced by extract_key_facts.py."""
    out = {}
    with path.open() as f:
        for ln in f:
            try:
                d = json.loads(ln)
                out[d["task_id"]] = d
            except Exception as e:
                print(f"  [WARN] skip bad jsonl line: {e}")
    return out


def _extract_contexts_hymem(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of input contexts per planner turn for a HyMem trace.

    Each entry: {turn_idx, messages: [{role, content}]}.
    """
    pcpt = stats.get("planner_context_per_turn") or []
    out = []
    for entry in pcpt:
        if not isinstance(entry, dict):
            continue
        msgs = entry.get("messages") or []
        if not msgs:
            continue
        out.append({
            "turn_idx": entry.get("turn", len(out) + 1),
            "messages": [
                {"role": m.get("role", ""), "content": m.get("content", "")}
                for m in msgs
                if isinstance(m, dict)
            ],
        })
    return out


def _reconstruct_hymem_contexts_from_history(trace: Dict[str, Any]
                                                ) -> List[Dict[str, Any]]:
    """Fallback for old HyMem traces that have NO planner_context_per_turn
    field (pre-instrumentation). Reconstructs an approximate per-planner-
    turn context by accumulating `hymem_history` entries up to (but not
    including) each planner step.

    Limits of the approximation:
      * Branch internals get flattened in linear order — accurate for KFD
        because KFD only counts substring occurrences, order does not
        matter.
      * Fold events trigger a context reset (we drop the accumulated
        history and start from a fold-summary marker). This matches what
        the real planner sees on the next step after fold.
    """
    hist = trace.get("hymem_history") or []
    if not hist:
        return []
    problem = trace.get("problem") or trace.get("stats", {}).get("problem") or ""
    # Real per-planner-turn context length (in tokens) from event_log.
    # We use this as the denominator for KFD so the reconstruction's
    # missing system-prompt / memory-context tokens don't make the
    # density artificially high.
    real_ctx_len_by_turn: Dict[int, int] = {}
    for ev in (trace.get("event_log") or {}).get("events", []) or []:
        if ev.get("event") == "planner_turn":
            t = int(ev.get("turn", 0))
            cl = int(ev.get("context_length", 0))
            if t and cl:
                real_ctx_len_by_turn[t] = cl
    accum: List[Dict[str, str]] = []
    if problem:
        accum.append({"role": "user", "content": problem})
    contexts: List[Dict[str, Any]] = []
    planner_turn = 0
    for h in hist:
        role = (h.get("role") or "").lower()
        if role == "planner":
            planner_turn += 1
            ctx = {
                "turn_idx": planner_turn,
                "messages": [dict(m) for m in accum],
            }
            # Carry the real context-length override if event_log has it
            if planner_turn in real_ctx_len_by_turn:
                ctx["override_token_len"] = real_ctx_len_by_turn[planner_turn]
            contexts.append(ctx)
            content = h.get("content", "") or ""
            if content:
                accum.append({"role": "assistant", "content": content})
        elif role == "executor":
            task = h.get("task", "") or ""
            result = h.get("result", "") or ""
            if task:
                accum.append({"role": "tool_input", "content": task})
            if result:
                accum.append({"role": "tool_output", "content": result})
        elif role == "branch":
            for k in ("description", "prompt", "result"):
                v = h.get(k, "") or ""
                if v:
                    accum.append({"role": f"branch_{k}", "content": v})
        elif role == "fold":
            # post-fold context reset: question + folded summary only
            accum = []
            if problem:
                accum.append({"role": "user", "content": problem})
            summary = h.get("planner_summary", "") or ""
            if summary:
                accum.append({"role": "assistant", "content": summary})
        elif role == "replan":
            content = h.get("content", "") or ""
            if content:
                accum.append({"role": "user", "content": content})
    return contexts


def _extract_contexts_react(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of input contexts per planner turn for a fold-agent trace.

    A "planner turn" is each assistant message in the main agent's dump.
    The input context for that turn = all dump entries BEFORE it.
    """
    pcpt = stats.get("planner_context_per_turn") or []
    # find 'main' agent
    main_dump = None
    for entry in pcpt:
        if isinstance(entry, dict) and entry.get("agent") == "main":
            main_dump = entry.get("dump") or []
            break
    if not main_dump:
        return []
    out = []
    accum = []  # messages added so far
    for d in main_dump:
        role = d.get("role", "")
        content = d.get("content", "")
        if role == "assistant":
            # snapshot input context = accum BEFORE adding this assistant msg
            out.append({
                "turn_idx": len(out) + 1,
                "messages": list(accum),  # shallow copy of refs
            })
        accum.append({"role": role, "content": content})
    return out


def _tokenize_len(tokenizer, text: str) -> int:
    if not text:
        return 0
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return len(ids)
    except Exception:
        return 0


def _fact_token_len_cache(tokenizer, facts: List[Dict[str, Any]]
                          ) -> Dict[str, int]:
    """Cache: fact_text -> token length. Computed once per task."""
    cache = {}
    for f in facts:
        t = f.get("fact", "")
        if t and t not in cache:
            cache[t] = max(1, _tokenize_len(tokenizer, t))
    return cache


def _count_occurrences(text: str, fact: str) -> int:
    if not fact or not text:
        return 0
    # Case-sensitive literal substring count.
    return text.count(fact)


def _compute_per_turn(tokenizer, contexts: List[Dict[str, Any]],
                      facts: List[Dict[str, Any]],
                      window_tokens: int = 2048
                      ) -> List[Dict[str, Any]]:
    """Compute per-turn dilution metrics:

      KFD   — global key-fact density
      KFSR  — survival rate (always 1.0 in non-shrinking ReAct contexts;
              kept for cross-check)
      KFD_w — density in trailing-window (last `window_tokens`). This is
              the "Lost in the Middle" proxy: as distractor pages get
              appended, key facts get pushed OUT of the window even if
              they remain somewhere earlier in context.
      KFR_end — distance (tokens) from end of context to the LAST
              occurrence of any key fact. Higher = key facts further from
              where the LLM is reasoning. Capped at context_token_len.
    """
    fact_tok_len = _fact_token_len_cache(tokenizer, facts)
    introduced: set = set()
    rows = []
    for t_idx, ctx in enumerate(contexts, start=1):
        full_text = "\n".join(m.get("content", "") for m in ctx["messages"])
        # Single tokenization pass with offset mapping so we can locate
        # character positions in token space.
        try:
            enc = tokenizer(
                full_text, add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = enc["input_ids"]
            offsets = enc["offset_mapping"]  # [(char_start, char_end), ...]
        except Exception:
            token_ids, offsets = [], []
        ctx_tok_len = max(1, len(token_ids))
        # If the trace provides a real (instrumented) per-turn context
        # token-count, prefer it as the KFD denominator. This is set by
        # the `_reconstruct_hymem_contexts_from_history` fallback so that
        # the reconstruction's missing scaffolding doesn't bias KFD up.
        override = ctx.get("override_token_len") if isinstance(ctx, dict) else None
        if isinstance(override, int) and override > 0:
            ctx_tok_len = override

        # Global KFD + presence
        fact_mass = 0
        present_now = set()
        last_token_pos: int = -1  # token index of last fact occurrence
        for f in facts:
            ftext = f.get("fact", "")
            if not ftext or not full_text:
                continue
            # Walk every literal substring occurrence.
            n_occ = 0
            char_pos = 0
            while True:
                hit = full_text.find(ftext, char_pos)
                if hit < 0:
                    break
                n_occ += 1
                # Find the token index containing this char position
                # (linear scan over offsets — bounded by ctx_tok_len).
                end_char = hit + len(ftext)
                for tok_i, (cs, ce) in enumerate(offsets):
                    if cs >= end_char:
                        break
                    if ce > hit:
                        # token covers some part of the fact span
                        if tok_i > last_token_pos:
                            last_token_pos = tok_i
                char_pos = hit + max(1, len(ftext))
            if n_occ > 0:
                fact_mass += n_occ * fact_tok_len.get(ftext, 1)
                present_now.add(ftext)
                introduced.add(ftext)

        KFD = fact_mass / ctx_tok_len
        KFSR = (len(present_now & introduced) / len(introduced)
                if introduced else 0.0)

        # Trailing-window KFD: tokenize the trailing slice of full_text
        # and recount fact-mass there. Use last `window_tokens` tokens.
        if ctx_tok_len > window_tokens:
            # decode last W tokens back to text, then count fact occurrences
            tail_ids = token_ids[-window_tokens:]
            try:
                tail_text = tokenizer.decode(tail_ids,
                                              skip_special_tokens=True)
            except Exception:
                tail_text = ""
            window_len = window_tokens
        else:
            tail_text = full_text
            window_len = ctx_tok_len
        tail_mass = 0
        for f in facts:
            ftext = f.get("fact", "")
            if not ftext or not tail_text:
                continue
            n_occ = tail_text.count(ftext)
            if n_occ > 0:
                tail_mass += n_occ * fact_tok_len.get(ftext, 1)
        KFD_w = tail_mass / max(1, window_len)

        # KFR_end: distance from end of context to last fact occurrence.
        # If no fact found, set to ctx_tok_len (worst — fact "infinitely
        # far away").
        if last_token_pos >= 0:
            KFR_end = ctx_tok_len - last_token_pos
        else:
            KFR_end = ctx_tok_len

        rows.append({
            "turn": t_idx,
            "KFD": KFD,
            "KFSR": KFSR,
            "KFD_w": KFD_w,
            "KFR_end": KFR_end,
            "context_token_len": ctx_tok_len,
            "window_tokens": window_tokens,
            "n_facts_present": len(present_now),
            "n_facts_introduced": len(introduced),
            "n_facts_total": len(facts),
        })
    return rows


def process_trace(trace_path: Path, key_facts: Dict[str, Dict[str, Any]],
                  condition: str, benchmark: str, tokenizer
                  ) -> List[Dict[str, Any]]:
    try:
        trace = json.loads(trace_path.read_text())
    except Exception as e:
        print(f"  [WARN] skip unreadable trace {trace_path.name}: {e}")
        return []
    iid = trace.get("instance_id") or trace.get("stats", {}).get("instance_id")
    if not iid or iid not in key_facts:
        return []
    facts = key_facts[iid].get("facts") or []
    if not facts:
        return []

    stats = trace.get("stats") or {}
    framework = (stats.get("framework") or "").lower() or condition.lower()
    if "hymem" in framework:
        contexts = _extract_contexts_hymem(stats)
        if not contexts:
            # Pre-instrumentation old HyMem trace — reconstruct from
            # hymem_history. Approximation: see function docstring.
            contexts = _reconstruct_hymem_contexts_from_history(trace)
    else:
        contexts = _extract_contexts_react(stats)
    if not contexts:
        return []

    score = trace.get("score") or stats.get("score") or 0
    per_turn = _compute_per_turn(tokenizer, contexts, facts)
    for r in per_turn:
        r.update({
            "task_id": iid,
            "condition": condition,
            "benchmark": benchmark,
            "score": int(score),
        })
    return per_turn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace_dir", required=True,
                   help="Dir containing gaia_*_trace.json files")
    p.add_argument("--key_facts", required=True,
                   help="JSONL output of extract_key_facts.py")
    p.add_argument("--condition", required=True,
                   help='Label, e.g. "hymem" or "react"')
    p.add_argument("--benchmark", default="gaia")
    p.add_argument("--out", required=True,
                   help="Output parquet path")
    p.add_argument("--tokenizer", default=TOKENIZER_NAME)
    args = p.parse_args()

    print(f"[dilution] loading tokenizer {args.tokenizer} ...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    key_facts = _load_key_facts(Path(args.key_facts))
    print(f"[dilution] key-facts entries: {len(key_facts)}")

    traces = sorted(Path(args.trace_dir).glob("gaia_*_trace.json"))
    # Skip aggregated logger files like gaia_eval_*.json
    traces = [t for t in traces if "_trace.json" in t.name]
    print(f"[dilution] traces in {args.trace_dir}: {len(traces)}")

    all_rows = []
    t0 = time.time()
    for i, tf in enumerate(traces):
        rows = process_trace(tf, key_facts, args.condition,
                              args.benchmark, tok)
        if rows:
            all_rows.extend(rows)
        if (i + 1) % 20 == 0 or i + 1 == len(traces):
            print(f"  [{i+1}/{len(traces)}] rows so far: {len(all_rows)}")

    if not all_rows:
        print("[dilution] no rows produced — check schema mismatch")
        return 1

    df = pd.DataFrame(all_rows)
    cols = ["task_id", "condition", "benchmark", "turn",
            "KFD", "KFSR", "KFD_w", "KFR_end",
            "context_token_len", "window_tokens",
            "n_facts_present", "n_facts_introduced",
            "n_facts_total", "score"]
    df = df[cols]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[dilution] wrote {len(df)} rows -> {out_path} in "
          f"{time.time()-t0:.1f}s")

    # Quick summary
    print(f"\n[dilution] tasks covered: {df['task_id'].nunique()}")
    print(f"[dilution] turn distribution: "
          f"min={df['turn'].min()} median={df['turn'].median()} "
          f"max={df['turn'].max()}")
    final_per_task = df.sort_values("turn").groupby("task_id").tail(1)
    print(f"[dilution] mean KFSR(final):    {final_per_task['KFSR'].mean():.3f}")
    print(f"[dilution] mean KFD(final):     {final_per_task['KFD'].mean():.4f}")
    print(f"[dilution] mean KFD_w(final):   {final_per_task['KFD_w'].mean():.4f}")
    print(f"[dilution] mean KFR_end(final): {final_per_task['KFR_end'].mean():.0f} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
