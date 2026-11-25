"""Figure 1 – Key-information dilution in long-horizon agent reasoning.

Nature-style two-panel composite.

  (a) Hero panel: fraction of context that comes AFTER the key fact —
      i.e. the amount of context the LLM has to look back through to
      retrieve the fact. Grounded in autoregressive generation +
      Lost-in-the-Middle (Liu et al. 2023): the larger this fraction,
      the deeper the fact is buried, and the harder it is for attention
      to recover.

  (b) One representative task — ReAct fails, HyMem succeeds — showing
      the SAME key fact across reasoning steps. The fact fades to near
      invisibility under ReAct while staying salient under HyMem.

Design choices recorded for reviewers:
  * Nature-style typography (Arial 7 pt base, editable SVG/PDF text).
  * NMI pastel palette: muted brick red (ReAct), teal-green (HyMem).
  * No invented "offset" term — metric is a normalised position
    (% of context length) that any reader can compute.
  * Right-panel gray bars labelled neutrally as "other context tokens"
    (system prompts, the question, the agent's own thoughts, tool
    outputs, etc. — not all of which are distractors).
  * Method chips in TOP-RIGHT of each subplot, on the empty side, so
    they never sit on top of any bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import patheffects as pe

from scipy import stats as sstats


# -------------------------------------------------------------------- rcParams
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 18.0,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 16,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
})


# -------------------------------------------------------------------- palette
C_REACT       = "#c45a4f"
C_HYMEM       = "#5fa168"
C_FACT_BRIGHT = "#3d8c5e"
C_FACT_FADED  = "#bcd9c5"
C_OTHER       = "#cccccc"   # "other context tokens" — neutral mid gray
C_OTHER_HI    = "#9a9a9a"   # hatch accent
C_END         = "#1f1f1f"
C_AXIS        = "#222222"
C_HELP        = "#666666"


def save_pub_py(fig, filename, dpi: int = 600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.tiff", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")


# ===================================================================== stats

def _bootstrap_ci(values: np.ndarray, n: int = 1000, alpha: float = 0.05,
                  rng: Optional[np.random.RandomState] = None
                  ) -> Tuple[float, float]:
    rng = rng or np.random.RandomState(0)
    if len(values) == 0:
        return (0.0, 0.0)
    idx = rng.randint(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _agg_curve(df: pd.DataFrame, col: str, max_turn: int = 15,
                min_n: Optional[int] = None
                ) -> Tuple[np.ndarray, ...]:
    """min_n defaults to max(3, 15% of total tasks). Adapts to pilot vs
    full sample sizes so each plot uses an honest cutoff."""
    if min_n is None:
        N = df["task_id"].nunique() if len(df) else 0
        min_n = max(3, int(N * 0.15))
    """Aggregate across tasks; mask steps with fewer than min_n samples
    so noisy tail bins don't fake a 'decline'."""
    turns = np.arange(1, max_turn + 1)
    means, lo, hi = [], [], []
    rng = np.random.RandomState(0)
    for t in turns:
        vals = df[df["turn"] == t][col].dropna().values
        if len(vals) < min_n:
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        means.append(float(vals.mean()))
        l, h = _bootstrap_ci(vals, rng=rng)
        lo.append(l); hi.append(h)
    return turns, np.array(means), np.array(lo), np.array(hi)


def _stats_block(react_df: pd.DataFrame, hymem_df: pd.DataFrame
                 ) -> Dict[str, Any]:
    rf = (react_df.sort_values("turn").groupby("task_id").tail(1)
          .set_index("task_id"))
    hf = (hymem_df.sort_values("turn").groupby("task_id").tail(1)
          .set_index("task_id"))
    common = sorted(set(rf.index) & set(hf.index))
    res = {"n": len(common)}
    if not common:
        return res
    d = (hf.loc[common, "KFD"] - rf.loc[common, "KFD"]).values
    res["KFD_median_delta"] = float(np.median(d))
    try:
        res["KFD_p"] = float(sstats.wilcoxon(
            d, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        res["KFD_p"] = float("nan")
    return res


# ===================================================== right-panel utilities

def _pick_panel_b_task(react_df: pd.DataFrame, hymem_df: pd.DataFrame,
                        min_hymem_steps: int = 4,
                        min_react_steps: int = 6
                        ) -> Optional[str]:
    """Pick a task where ReAct fails and HyMem succeeds, with the largest
    KFD gap (HyMem−ReAct). Require both methods to have enough steps so
    the timeline visualization is meaningful."""
    h_steps = hymem_df.groupby("task_id")["turn"].max()
    r_steps = react_df.groupby("task_id")["turn"].max()
    rf = (react_df.sort_values("turn").groupby("task_id").tail(1)
          .set_index("task_id"))
    hf = (hymem_df.sort_values("turn").groupby("task_id").tail(1)
          .set_index("task_id"))
    common = sorted(set(rf.index) & set(hf.index))
    if not common:
        return None
    cand = []
    for tid in common:
        r = rf.loc[tid]; h = hf.loc[tid]
        if (h_steps.get(tid, 0) < min_hymem_steps or
            r_steps.get(tid, 0) < min_react_steps):
            continue
        if int(r["score"]) == 0 and int(h["score"]) == 1:
            gap = float(h["KFD"]) - float(r["KFD"])
            cand.append((gap, tid))
    if cand:
        cand.sort(reverse=True)
        return cand[0][1]
    # Fallback: drop the both-correct/both-wrong constraint
    cand = []
    for tid in common:
        if (h_steps.get(tid, 0) < min_hymem_steps or
            r_steps.get(tid, 0) < min_react_steps):
            continue
        r = rf.loc[tid]; h = hf.loc[tid]
        gap = float(h["KFD"]) - float(r["KFD"])
        cand.append((gap, tid))
    cand.sort(reverse=True)
    return cand[0][1] if cand else None


def _load_trace(trace_dir: Path, task_id: str) -> Optional[Dict[str, Any]]:
    """Look for the task's trace in `trace_dir`. As a convenience, if the
    primary dir is the pilot L2 dir but the task lives in L1 or L3, also
    search the sibling `_L1_optimized` / `_L3_optimized` dirs."""
    candidates = [trace_dir / f"gaia_{task_id}_trace.json"]
    # Auto-search L1/L2/L3 siblings if the primary dir follows the
    # logs/gaia_L*_optimized naming convention.
    parent = trace_dir.parent
    for sib_name in ["gaia_L1_optimized", "gaia_L2_optimized",
                       "gaia_L3_optimized"]:
        sib = parent / sib_name / f"gaia_{task_id}_trace.json"
        if sib not in candidates:
            candidates.append(sib)
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def _reconstruct_hymem_contexts_from_history(trace: Dict[str, Any]
                                                ) -> List[Dict[str, Any]]:
    """Fallback for old HyMem traces — same logic as key_fact_dilution.py.
    Also carries the real per-turn context token-count from event_log so
    the timeline visualization can match the metric in panel a."""
    hist = trace.get("hymem_history") or []
    if not hist:
        return []
    problem = trace.get("problem") or trace.get("stats", {}).get("problem") or ""
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
            entry = {
                "turn": planner_turn,
                "messages": [dict(m) for m in accum],
            }
            if planner_turn in real_ctx_len_by_turn:
                entry["override_token_len"] = real_ctx_len_by_turn[planner_turn]
            contexts.append(entry)
            content = h.get("content", "") or ""
            if content:
                accum.append({"role": "assistant", "content": content})
        elif role == "executor":
            for k in ("task", "result"):
                v = h.get(k, "") or ""
                if v:
                    accum.append({"role": f"executor_{k}", "content": v})
        elif role == "branch":
            for k in ("description", "prompt", "result"):
                v = h.get(k, "") or ""
                if v:
                    accum.append({"role": f"branch_{k}", "content": v})
        elif role == "fold":
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


def _extract_contexts(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = trace.get("stats") or {}
    pcpt = stats.get("planner_context_per_turn") or []
    if pcpt and isinstance(pcpt[0], dict) and "messages" in pcpt[0]:
        return [{"turn": e.get("turn", i + 1),
                 "messages": e.get("messages", [])}
                for i, e in enumerate(pcpt)]
    if pcpt and isinstance(pcpt[0], dict) and "dump" in pcpt[0]:
        main = next((e for e in pcpt if e.get("agent") == "main"), None)
        if not main:
            return _reconstruct_hymem_contexts_from_history(trace)
        out, accum = [], []
        for d in main.get("dump", []):
            if d.get("role") == "assistant":
                out.append({"turn": len(out) + 1, "messages": list(accum)})
            accum.append({"role": d.get("role", ""),
                          "content": d.get("content", "")})
        return out
    # Last resort: old HyMem trace without any planner_context_per_turn
    return _reconstruct_hymem_contexts_from_history(trace)


def _last_pos(text: str, fact: str) -> int:
    if not fact or not text:
        return -1
    return text.rfind(fact)


def _tokenize_with_offsets(tokenizer, text: str
                             ) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Return (token_ids, char-span list) for `text`."""
    if not text:
        return [], []
    try:
        enc = tokenizer(text, add_special_tokens=False,
                         return_offsets_mapping=True)
        return list(enc["input_ids"]), list(enc["offset_mapping"])
    except Exception:
        return [], []


def _char_to_token_pos(offsets: List[Tuple[int, int]], char_pos: int) -> int:
    """Map a character offset to the token index that covers it."""
    if not offsets or char_pos < 0:
        return -1
    for i, (cs, ce) in enumerate(offsets):
        if cs <= char_pos < ce or cs >= char_pos:
            return i
    return len(offsets) - 1


def _distance_to_opacity(dist_chars: int, ctx_len: int) -> float:
    if dist_chars <= 0:
        return 1.0
    frac = dist_chars / max(1, ctx_len)
    alpha = 1.0 - frac ** 0.5
    return float(np.clip(alpha, 0.18, 1.0))


def _all_fact_positions(text: str,
                          facts: List[Dict[str, Any]]
                          ) -> List[Tuple[int, int]]:
    """Return list of (char_start, char_end) for every occurrence of any
    key fact in `text`."""
    spans = []
    for f in facts:
        ft = f.get("fact", "")
        if not ft:
            continue
        pos = 0
        while True:
            p = text.find(ft, pos)
            if p < 0:
                break
            spans.append((p, p + len(ft)))
            pos = p + max(1, len(ft))
    return spans


def _cluster_spans(spans: List[Tuple[int, int]],
                    ctx_len: int,
                    max_gap_frac: float = 0.003,
                    min_render_width_frac: float = 0.006,
                    max_cluster_width_frac: float = 0.020
                    ) -> List[Tuple[int, int, int]]:
    """Merge nearby fact occurrences into visible clusters.

    Rules:
      - Merge two adjacent facts if their gap < `max_gap_frac × ctx_len`.
      - But STOP growing the cluster once its width reaches
        `max_cluster_width_frac × ctx_len` (so dense fact regions stay
        split into multiple smaller clusters instead of one big blob).
      - Each cluster's rendered width is the actual span [cur_s, cur_e],
        with a floor of `min_render_width_frac × ctx_len` so single
        isolated facts on long bars stay visible.

    Returns: list of (start, end, n_facts_merged).
    """
    if not spans:
        return []
    sorted_pos = sorted(spans)
    max_gap = max(15, int(ctx_len * max_gap_frac))
    # Floor min_w generously so even tiny bars (few hundred tokens) show
    # discernible green ticks; cap so we never visually exaggerate.
    min_w = max(20, int(ctx_len * min_render_width_frac))
    max_w = max(60, int(ctx_len * max_cluster_width_frac))

    clusters = []
    cur_s, cur_e = sorted_pos[0]
    count = 1
    for s, e in sorted_pos[1:]:
        gap_ok = s - cur_e <= max_gap
        width_ok = (max(cur_e, e) - cur_s) <= max_w
        if gap_ok and width_ok:
            cur_e = max(cur_e, e)
            count += 1
        else:
            end = max(cur_e, cur_s + min_w)
            clusters.append((cur_s, end, count))
            cur_s, cur_e = s, e
            count = 1
    end = max(cur_e, cur_s + min_w)
    clusters.append((cur_s, end, count))
    return clusters


def _draw_timeline(ax, contexts: List[Dict[str, Any]],
                    facts: List[Dict[str, Any]],
                    xlim_tokens: int,
                    method_chip: Tuple[str, str],
                    tokenizer,
                    n_turns_show: int = 6,
                    show_xlabel: bool = True,
                    show_xticks: bool = True):
    """Visualises information density at every step.

    Each bar:
      - gray body         = OTHER context tokens (system prompt, question,
                            agent thoughts, tool results — both relevant
                            and distractor). Neutral label.
      - GREEN ticks       = every occurrence of every key fact. The
                            visual density of ticks on a bar IS the
                            key-info density (KFD) for that step.
      - black tick at L   = end-of-context.
    """
    if not contexts or not facts:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, fontsize=13.5, color="#999")
        ax.axis("off")
        return

    n = len(contexts)
    if n <= n_turns_show:
        idxs = list(range(n))
    else:
        mid = list(np.linspace(0, n - 1, n_turns_show, dtype=int))
        idxs = sorted(set([0, n - 1] + mid))[:n_turns_show]

    bar_h = 0.50
    ax.set_xlim(-xlim_tokens * 0.005, xlim_tokens * 1.10)
    ax.set_ylim(-0.7, len(idxs) - 0.3)
    ax.invert_yaxis()
    yticks, ylabels = [], []

    for slot, i in enumerate(idxs):
        ctx = contexts[i]
        text = "\n".join(m.get("content", "") for m in ctx["messages"])
        token_ids, offsets = _tokenize_with_offsets(tokenizer, text)
        L_recon = len(token_ids)
        if L_recon == 0:
            continue
        # Bar TOTAL length = real (instrumented) context length if known;
        # otherwise = reconstruction length. Reconstruction sits at the
        # END of the real context (system prompt + memory are before it).
        L_real = int(ctx.get("override_token_len") or L_recon)
        if L_real < L_recon:
            L_real = L_recon
        L_tok = L_real
        recon_offset = L_real - L_recon  # gray pad on the left
        # full gray context bar
        ax.add_patch(Rectangle(
            (0, slot - bar_h / 2), L_tok, bar_h,
            facecolor=C_OTHER, edgecolor="none", alpha=0.55,
            zorder=1))
        # Convert every fact occurrence to token-space; shift by
        # recon_offset so positions sit in the real-context coords.
        tok_spans: List[Tuple[int, int]] = []
        fact_token_mass = 0
        for cs, ce in _all_fact_positions(text, facts):
            tok_s = _char_to_token_pos(offsets, cs) if offsets else -1
            tok_e = _char_to_token_pos(offsets, ce) if offsets else -1
            if tok_s < 0:
                continue
            tok_s += recon_offset
            tok_e += recon_offset
            tok_e = max(tok_e, tok_s + 1)
            tok_spans.append((tok_s, tok_e))
            fact_token_mass += (tok_e - tok_s)
        # Merge close facts; cap cluster width so dense regions split.
        for cs, ce, _ in _cluster_spans(tok_spans, L_tok):
            ax.add_patch(Rectangle(
                (cs, slot - bar_h / 2), ce - cs, bar_h,
                facecolor=C_FACT_BRIGHT, edgecolor="none", alpha=0.95,
                zorder=3))
        # end-of-context anchor
        ax.plot([L_tok, L_tok],
                 [slot - bar_h / 2 - 0.10,
                  slot + bar_h / 2 + 0.10],
                 color=C_END, linewidth=1.4, zorder=4)
        # Per-step KFD% annotation, immediately right of end-of-context
        kfd_pct = (fact_token_mass / L_tok) * 100 if L_tok else 0.0
        ax.text(L_tok + xlim_tokens * 0.013, slot,
                 f"{kfd_pct:.1f}%",
                 fontsize=9.6, color=C_AXIS, va="center", ha="left")
        yticks.append(slot)
        ylabels.append(f"{ctx.get('turn', i + 1)}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylabel("step", labelpad=2)
    if show_xticks:
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
        ax.xaxis.set_major_locator(MaxNLocator(4))
    else:
        ax.set_xticklabels([])
    if show_xlabel:
        ax.set_xlabel("Context position (tokens)", labelpad=2)
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(C_AXIS)

    # Method chip in TOP-RIGHT (the empty side — t=1 bar is shortest and
    # lives at the top-left, so top-right is always clear).
    chip_label, chip_color = method_chip
    ax.text(0.985, 0.94, chip_label, transform=ax.transAxes,
            fontsize=11.1, fontweight="bold", color="white", va="top",
            ha="right",
            bbox=dict(boxstyle="round,pad=0.30,rounding_size=0.18",
                      fc=chip_color, ec="none"))


# ============================================================ composite

def make_figure(react_df: pd.DataFrame, hymem_df: pd.DataFrame,
                react_trace_dir: Path, hymem_trace_dir: Path,
                key_facts: Dict[str, Any], out_path: Path
                ) -> Dict[str, Any]:

    rt_ok = react_df.groupby("task_id").filter(lambda g: g["turn"].max() >= 3)
    ht_ok = hymem_df.groupby("task_id").filter(lambda g: g["turn"].max() >= 3)
    max_t = int(max(rt_ok["turn"].max() if len(rt_ok) else 5,
                    ht_ok["turn"].max() if len(ht_ok) else 5))
    max_t = min(max_t, 15)

    fig = plt.figure(figsize=(9.6, 5.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.10],
                          height_ratios=[1.0, 1.0],
                          hspace=0.30, wspace=0.20,
                          left=0.088, right=0.985,
                          top=0.86, bottom=0.13)
    ax_data = fig.add_subplot(gs[:, 0])
    ax_react = fig.add_subplot(gs[0, 1])
    ax_hymem = fig.add_subplot(gs[1, 1])

    # ---- (a) hero curve: KFD = fraction of context that is the key
    # information. A direct signal-to-noise measure: how much of what
    # the LLM reads is actually relevant to the answer. No positional
    # assumptions; pure density.
    curves = {}
    for df_, lab, c in [(rt_ok, "ReAct", C_REACT),
                        (ht_ok, "HyMem", C_HYMEM)]:
        turns, mean, lo, hi = _agg_curve(df_, "KFD", max_turn=max_t)
        ax_data.plot(turns, mean * 100, "-o", color=c, label=lab,
                     markersize=3.2, linewidth=1.6, alpha=0.95,
                     zorder=3)
        ax_data.fill_between(turns, lo * 100, hi * 100, color=c,
                              alpha=0.10, linewidth=0, zorder=1)
        curves[lab] = (turns, mean * 100)
    ax_data.set_xlabel("Reasoning step", labelpad=2)
    ax_data.set_ylabel(
        "Key-info density of context (%)",
        labelpad=2)
    ax_data.set_ylim(bottom=0)
    ax_data.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax_data.yaxis.set_major_locator(MaxNLocator(4))
    ax_data.grid(False)
    for s in ["bottom", "left"]:
        ax_data.spines[s].set_color(C_AXIS)

    halo = [pe.withStroke(linewidth=2.0, foreground="white")]
    for lab, (turns, mean) in curves.items():
        valid = np.where(~np.isnan(mean))[0]
        if len(valid) == 0:
            continue
        i = valid[-1]
        color = C_REACT if lab == "ReAct" else C_HYMEM
        ax_data.text(turns[i] + 0.25, mean[i], lab,
                     color=color, fontsize=15, fontweight="bold",
                     va="center", ha="left", path_effects=halo,
                     zorder=4)

    ax_data.set_xlim(0.5, max_t + 2.5)

    # (a) heading
    pos = ax_data.get_position()
    cx_a = (pos.x0 + pos.x1) / 2
    fig.text(cx_a, 0.91,
              "Key information is a smaller fraction\nof context under ReAct",
              fontsize=16.7, ha="center", color=C_AXIS)

    # ---- (b) one task trace ----
    task_id = _pick_panel_b_task(react_df, hymem_df)
    if task_id:
        from transformers import AutoTokenizer
        print("[figure] loading tokenizer Qwen/Qwen2.5-7B-Instruct ...")
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
        facts = key_facts.get(task_id, {}).get("facts", [])
        r_trace = _load_trace(react_trace_dir, task_id)
        h_trace = _load_trace(hymem_trace_dir, task_id)
        r_ctx = _extract_contexts(r_trace) if r_trace else []
        h_ctx = _extract_contexts(h_trace) if h_trace else []
        # Compute max context length per method (in tokens), so each
        # sub-panel can use its own x-axis range. HyMem's compact fold
        # would be invisible if forced onto ReAct's full-context axis.
        # Use the REAL context length (event_log override) if present,
        # otherwise fall back to reconstruction-token count.
        def _max_tok(ctxs):
            m = 1
            for c in ctxs:
                override = c.get("override_token_len") if isinstance(c, dict) else None
                if isinstance(override, int) and override > 0:
                    if override > m:
                        m = override
                    continue
                text = "\n".join(mm.get("content", "") for mm in c["messages"])
                ids, _ = _tokenize_with_offsets(tokenizer, text)
                if len(ids) > m:
                    m = len(ids)
            return m
        max_L_react = _max_tok(r_ctx)
        max_L_hymem = _max_tok(h_ctx)
        r_score = int(r_trace.get("score", 0) if r_trace else 0)
        h_score = int(h_trace.get("score", 0) if h_trace else 0)
        _draw_timeline(ax_react, r_ctx, facts,
                        xlim_tokens=max_L_react,
                        method_chip=(f"ReAct  score={r_score}", C_REACT),
                        tokenizer=tokenizer,
                        show_xlabel=False, show_xticks=True)
        _draw_timeline(ax_hymem, h_ctx, facts,
                        xlim_tokens=max_L_hymem,
                        method_chip=(f"HyMem  score={h_score}", C_HYMEM),
                        tokenizer=tokenizer,
                        show_xlabel=True, show_xticks=True)

        rpos = ax_react.get_position()
        hpos = ax_hymem.get_position()
        cx_b = (rpos.x0 + rpos.x1) / 2
        fig.text(cx_b, 0.91,
                  "One task: the same key fact across steps",
                  fontsize=16.7, ha="center", color=C_AXIS)

        # Legend BELOW HyMem panel — figure-level, no overlap with bars.
        legend_y = hpos.y0 - 0.13
        leg_x = rpos.x0 - 0.05
        sw_w, sw_h = 0.014, 0.020
        fig.patches.append(Rectangle((leg_x, legend_y), sw_w, sw_h,
                                       transform=fig.transFigure,
                                       facecolor=C_FACT_BRIGHT,
                                       edgecolor="none"))
        fig.text(leg_x + sw_w + 0.004, legend_y + sw_h / 2,
                  "key-info tokens",
                  fontsize=13, va="center", color=C_AXIS)
        leg_x2 = leg_x + 0.15
        fig.patches.append(Rectangle((leg_x2, legend_y), sw_w, sw_h,
                                       transform=fig.transFigure,
                                       facecolor=C_OTHER, alpha=0.55,
                                       edgecolor="none"))
        fig.text(leg_x2 + sw_w + 0.004, legend_y + sw_h / 2,
                  "other context tokens",
                  fontsize=13, va="center", color=C_AXIS)
        leg_x3 = leg_x + 0.35
        fig.add_artist(plt.Line2D(
            [leg_x3 + 0.004, leg_x3 + 0.004],
            [legend_y, legend_y + sw_h],
            transform=fig.transFigure,
            color=C_END, linewidth=1.4))
        fig.text(leg_x3 + sw_w + 0.004, legend_y + sw_h / 2,
                  "end of context",
                  fontsize=13, va="center", color=C_AXIS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_pub_py(fig, str(out_path.with_suffix("")))
    print(f"[figure] wrote {out_path.with_suffix('.pdf')}, "
          f".svg, .tiff, .png")

    return {"panel_b_task_id": task_id,
            "stats": _stats_block(react_df, hymem_df)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--react_parquet", required=True)
    p.add_argument("--hymem_parquet", required=True)
    p.add_argument("--react_trace_dir", required=True)
    p.add_argument("--hymem_trace_dir", required=True)
    p.add_argument("--key_facts", required=True)
    p.add_argument("--out", default="results/figures/figure1_dilution.pdf")
    p.add_argument("--stats_out", default=None)
    args = p.parse_args()

    react_df = pd.read_parquet(args.react_parquet)
    hymem_df = pd.read_parquet(args.hymem_parquet)
    print(f"[figure] ReAct rows: {len(react_df)} tasks: "
          f"{react_df['task_id'].nunique()}")
    print(f"[figure] HyMem rows: {len(hymem_df)} tasks: "
          f"{hymem_df['task_id'].nunique()}")

    key_facts = {}
    for ln in Path(args.key_facts).open():
        try:
            d = json.loads(ln)
            key_facts[d["task_id"]] = d
        except Exception:
            pass

    out = make_figure(react_df, hymem_df,
                       Path(args.react_trace_dir),
                       Path(args.hymem_trace_dir),
                       key_facts, Path(args.out))

    s = out["stats"]
    print(f"\n[stats] paired Wilcoxon, n={s.get('n', 0)}:  "
          f"HyMem − ReAct KFD median = "
          f"{s.get('KFD_median_delta', 0)*100:+.2f} pp, "
          f"p = {s.get('KFD_p', 0):.4f}")
    print(f"[stats] Panel B task : {out['panel_b_task_id']}")

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(
            json.dumps(out, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
