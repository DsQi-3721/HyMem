"""
Unified tool-call metrics across HyMem / BrowseMaster / ReAct / Fold / Summary.

Why
---
Pre-fix, each framework reported a different `tool_calls` semantic:
  - HyMem / BrowseMaster: atomic tool-invocation count (web_search, web_parse,
    parse_file, ...) via ``BMToolManager._record_tool_call``
  - ReAct / FoldAgent: LLM-loop iteration count
    (``env.stats['tool_calls'] = iteration``)
  - Vanilla: nothing

Reviewer LMtp Q4 specifically asked for clarification ("Is an entire isolated
session counted as a single call, or every internal step counted individually?
This needs clarification to ensure a fair comparison."). The pre-fix Table 1
was effectively comparing apples to bananas.

Resolution
----------
A single source of truth for the four metrics every agent should report:

  ``tool_calls_atomic``   - count of atomic tool invocations
                            (e.g. each ``search`` / ``open_page`` / ``web_parse``)
  ``agent_turns``         - count of LLM-loop iterations across all sessions
                            (planner + executor + branch turns combined)
  ``tool_calls_by_source`` - HyMem-only split: {actor, branch, replan} → atomic
                            tool calls attributed to each source. For non-HyMem
                            frameworks this is just {"actor": tool_calls_atomic}.
  ``tool_calls_by_tool``  - per-tool counts (web_search, open_page, ...)

Helpers in this module:

  * ``UnifiedToolCounter`` - lightweight wrapper used by react / fold agents
    that don't have BMToolManager. Reads atomic counts from `env.stats` via
    ``derive_atomic_from_env_stats`` and combines with iteration counts.
  * ``write_unified_stats(env_stats, ...)`` - merges fields into the env.stats
    dict using the standard schema. Idempotent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# Canonical name set of atomic tools tracked across all frameworks.
# Entries here are env.stats keys whose value should be summed into
# ``tool_calls_atomic`` when the framework doesn't already provide it.
ATOMIC_TOOL_KEYS = (
    "search",      # local_search env
    "open_page",   # local_search env
    "web_search",
    "web_parse",
    "batch_search",
    "batch_search_and_filter",
    "check_condition",
    "parse_file",
    "generate_keywords",
)


def derive_atomic_from_env_stats(env_stats: Dict) -> int:
    """Sum atomic tool counts from a (possibly partially-populated) env.stats
    dict, used by ReAct / Fold agents that route tools through the env."""
    total = 0
    for key in ATOMIC_TOOL_KEYS:
        v = env_stats.get(key, 0)
        if isinstance(v, (int, float)):
            total += int(v)
    return total


def derive_tool_calls_by_tool_from_env_stats(env_stats: Dict) -> Dict[str, int]:
    """Build a tool_calls_by_tool dict from per-tool counters the env keeps.
    Lets ReAct / Fold / ReSum / A-MEM populate this field consistently with
    what HyMem's BMToolManager writes — important for fair Table breakdowns."""
    out: Dict[str, int] = {}
    for key in ATOMIC_TOOL_KEYS:
        v = env_stats.get(key, 0)
        if isinstance(v, (int, float)) and v > 0:
            out[key] = int(v)
    return out


def resolve_models_used() -> Dict[str, str]:
    """Resolve which models are actually serving each role from env vars.
    Same fallback chain HyMem uses internally — so all baselines report
    consistent identifiers regardless of how they internally choose models."""
    import os
    planner = os.getenv("HYMEM_PLANNER_MODEL", "") or os.getenv("OPENAI_MODEL", "")
    executor = os.getenv("HYMEM_EXECUTOR_MODEL", "") or planner
    aux = os.getenv("HYMEM_AUX_MODEL", "") or executor
    judge = os.getenv("JUDGE_MODEL", "") or aux
    return {
        "planner": planner or "unknown",
        "executor": executor or "unknown",
        "aux": aux or "unknown",
        "judge": judge or "unknown",
    }


@dataclass
class UnifiedToolCounter:
    """Lightweight counter for ReAct / Fold-style agents that count tool
    calls indirectly through the env. Use ``write_into(env.stats, ...)`` at
    the end of a session.

    HyMem-style agents already get these numbers from ``BMToolManager`` —
    use ``write_unified_stats`` directly without instantiating this class.
    """

    iterations: int = 0  # LLM-loop turn count
    extra_atomic: int = 0  # additional atomic calls outside env (rare)
    by_source: Dict[str, int] = field(default_factory=dict)
    framework: str = "react"  # for logging only

    def record_iteration(self) -> None:
        self.iterations += 1

    def add_atomic(self, n: int = 1, source: str = "actor") -> None:
        if n <= 0:
            return
        self.extra_atomic += n
        self.by_source[source] = self.by_source.get(source, 0) + n

    def write_into(self, env_stats: Dict) -> None:
        derived = derive_atomic_from_env_stats(env_stats)
        atomic = derived + self.extra_atomic
        # Default source split for non-HyMem agents: everything is "actor".
        by_source = dict(self.by_source) if self.by_source else {}
        if atomic > 0 and not by_source:
            by_source = {"actor": atomic}

        write_unified_stats(
            env_stats,
            tool_calls_atomic=atomic,
            agent_turns=self.iterations,
            tool_calls_by_source=by_source,
            framework=self.framework,
        )


def write_unified_stats(
    env_stats: Dict,
    tool_calls_atomic: int,
    agent_turns: int,
    tool_calls_by_source: Optional[Dict[str, int]] = None,
    tool_calls_by_tool: Optional[Dict[str, int]] = None,
    framework: str = "unknown",
) -> None:
    """Write the unified metrics into ``env_stats``. Idempotent.

    Also keeps the legacy ``tool_calls`` and ``tool_calls_total`` aliases
    populated so older analysis scripts continue to work, but the new
    fields (``tool_calls_atomic`` / ``agent_turns``) are the canonical
    ones to compare across frameworks.
    """
    env_stats["tool_calls_atomic"] = int(tool_calls_atomic)
    env_stats["agent_turns"] = int(agent_turns)
    env_stats["tool_calls_by_source"] = dict(tool_calls_by_source or {})
    if tool_calls_by_tool is not None:
        env_stats["tool_calls_by_tool"] = dict(tool_calls_by_tool)
    env_stats["framework"] = framework

    # Back-compat aliases — DO NOT use these for cross-framework comparison.
    env_stats.setdefault("tool_calls_total", int(tool_calls_atomic))
    env_stats.setdefault("tool_calls", int(tool_calls_atomic))


def merge_hymem_tool_stats(
    env_stats: Dict,
    bm_tool_stats: Dict,
    agent_turns: int,
) -> None:
    """Helper for the HyMem agent to populate the unified schema using the
    BMToolManager output. Does NOT overwrite per-tool counts already present.
    """
    by_source = dict(bm_tool_stats.get("tool_calls_by_source", {}))
    by_tool = dict(bm_tool_stats.get("tool_calls_by_tool", {}))
    write_unified_stats(
        env_stats,
        tool_calls_atomic=int(bm_tool_stats.get("tool_calls_total", 0)),
        agent_turns=agent_turns,
        tool_calls_by_source=by_source,
        tool_calls_by_tool=by_tool,
        framework="hymem",
    )


def write_baseline_stats(
    env_stats: Dict,
    *,
    framework: str,
    score: float,
    iterations: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    session_time: float,
    extra_atomic: int = 0,
    extra_by_source: Optional[Dict[str, int]] = None,
    extra_by_tool: Optional[Dict[str, int]] = None,
) -> None:
    """Single-call stats writer for all NON-HyMem baselines.

    Ensures every baseline emits the same set of universally-comparable
    fields (the "Tier 1" union of what eval_gaia.py reads + what the
    paper-Table / cost / efficiency / branch-trigger analysis scripts
    consume), so cross-framework comparisons in the paper are apples to
    apples without per-baseline special cases in the analysis layer.

    What this writes:
      - get_final_score                     (else eval reports 0)
      - tool_calls_atomic / agent_turns /
        tool_calls_by_source / framework /
        tool_calls_total (legacy alias)     (UnifiedToolCounter)
      - tool_calls_by_tool                  (derived from env_stats)
      - total_prompt_tokens / _completion / _tokens
      - token_usage_by_component            (single 'actor' bucket;
                                             HyMem overrides with split)
      - latency_by_component                (single 'actor'/'total' bucket;
                                             HyMem overrides with split)
      - models_used                         (resolved from env vars)
      - session_time

    HyMem-architecture-specific fields (branch_calls / fold_count /
    planner_turns / routing_summary / distillation_stats / etc.) are
    intentionally NOT written here — they default to 0/{} for baselines
    without those concepts, which is the honest outcome.
    """
    # 1. UnifiedToolCounter populates the atomic schema
    counter = UnifiedToolCounter(framework=framework)
    counter.iterations = iterations
    if extra_atomic:
        counter.add_atomic(extra_atomic, source="actor")
    if extra_by_source:
        for src, n in extra_by_source.items():
            counter.add_atomic(n, source=src)
    counter.write_into(env_stats)

    # 2. tool_calls_by_tool (derived). Don't overwrite if the caller
    #    already wrote a richer dict (e.g. Thread keeps its own).
    if not env_stats.get("tool_calls_by_tool"):
        derived = derive_tool_calls_by_tool_from_env_stats(env_stats)
        if extra_by_tool:
            for k, v in extra_by_tool.items():
                derived[k] = derived.get(k, 0) + int(v)
        env_stats["tool_calls_by_tool"] = derived

    # 3. Token totals
    total_tokens = int(total_prompt_tokens) + int(total_completion_tokens)
    env_stats["total_prompt_tokens"] = int(total_prompt_tokens)
    env_stats["total_completion_tokens"] = int(total_completion_tokens)
    env_stats["total_tokens"] = total_tokens

    # 4. token_usage_by_component (single 'actor' bucket — cost_calculator
    #    will route this to the planner model in COMPONENT_TO_MODEL_KEY).
    env_stats["token_usage_by_component"] = {
        "actor": {
            "prompt_tokens": int(total_prompt_tokens),
            "completion_tokens": int(total_completion_tokens),
            "total_tokens": total_tokens,
        }
    }

    # 5. latency_by_component — coarse for non-HyMem (no per-component
    #    timing), but we report it under the same key so analysis scripts
    #    can iterate uniformly. The 'actor' bucket aggregates ALL LLM time.
    env_stats["latency_by_component"] = {
        "actor": float(session_time),
        "total": float(session_time),
    }

    # 6. Models — read from env vars consistently
    env_stats["models_used"] = resolve_models_used()

    # 7. session_time (canonical name; we still let callers also write
    #    their own session_time for backwards compat — both align here)
    env_stats["session_time"] = float(session_time)

    # 8. get_final_score — the LLM judge's verdict. Single source of truth
    #    here so callers don't have to remember to write it (eval_gaia.py
    #    silently defaults to 0 if missing → collapses Pass@1 to 0).
    env_stats["get_final_score"] = score
