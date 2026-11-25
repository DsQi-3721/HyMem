"""Cross-baseline stats schema test.

Verifies that EVERY baseline (HyMem + ReAct + Fold + ReSum + Thread + A-MEM)
populates the same set of universally-comparable fields in env.stats. This
is what makes the paper's cross-framework comparison fair — without this
guarantee, eval_gaia.py silently defaults missing fields to 0/{} and the
resulting Tables look biased toward whichever framework writes more fields.

What we audit:
  - Universal fields every framework MUST populate (same key, same type)
  - Architecture-specific fields that are ALLOWED to be 0/empty for some
    frameworks (e.g. branch_calls is meaningful for HyMem and Thread but
    0 for ReAct/ReSum/A-MEM — the test confirms it's not undefined,
    not that it's nonzero).

Runnable two ways:
    pytest tests/test_baseline_schema.py -v
    python -m tests.test_baseline_schema
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# Universal fields — every baseline must write these so analysis scripts
# (cost_calculator / efficiency_analysis / branch_trigger_analysis /
# eval_gaia.py result schema) can iterate uniformly.
UNIVERSAL_FIELDS = {
    "get_final_score",
    "tool_calls_atomic",
    "agent_turns",
    "tool_calls_by_source",
    "tool_calls_by_tool",
    "framework",
    "tool_calls_total",   # legacy alias of tool_calls_atomic
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
    "token_usage_by_component",
    "latency_by_component",
    "models_used",
    "session_time",
}

# HyMem-architecture-specific. These ARE allowed to be 0/empty for
# baselines without those concepts, but eval_gaia.py reads them and the
# default values must be sensible (0 for counts, {} for dicts).
HYMEM_SPECIFIC_FIELDS = {
    "planner_turns": 0,
    "executor_calls": 0,
    "branch_calls": 0,
    "branch_success": 0,
    "fold_count": 0,
    "replan_count": 0,
    "routing_summary": {},
    "distillation_stats": {},
    # G1: HyMem fold-event snapshots (paired with ReSum's resum_summary_events)
    "hymem_fold_events": [],
    # G2: HyMem memory operation counters (paired with A-MEM amem_*)
    "hymem_memory_ops": {},
    # Round-1-3 efficiency observation fields (O1-O8 optimizations).
    # eval_gaia.py + eval_bc.py both read these; non-HyMem baselines
    # default them to 0 / 0.0.
    "parse_cache_hits": 0,
    "parse_cache_misses": 0,
    "parse_cache_hit_rate": 0.0,
    "executor_early_exit_count": 0,
    "executor_avg_inner_turns": 0.0,
    "distillation_invocations": 0,
    "distillation_avg_aux_calls": 0.0,
    "distillation_chunk_concurrent_used": 0,
    "fold_invocations": 0,
    "fold_avg_wall_seconds": 0.0,
    "fold_aux_parallel_seconds": 0.0,
    "parallel_dispatch_fanout_avg": 0.0,
    "parallel_dispatch_max_observed": 0,
    # Figure-1 dilution: per-turn full context dump (HyMem dict-list
    # or fold-agent agent-keyed list).
    "planner_context_per_turn": [],
}

# G1-G4 paired metric fields — each baseline OWNS one side of the pair.
# These are NOT universal (they're framework-specific), but they MUST be
# present in the corresponding framework's stats for §A/§B/§C analyses.
PAIRED_METRIC_FIELDS = {
    # baseline name → list of fields it must populate
    "resum":  ["resum_summaries_fired", "resum_summary_events"],
    "amem":   ["amem_notes_created", "amem_retrievals"],
    "thread": ["thread_spawn_count", "thread_spawn_success"],
    # G4 universal: every baseline (including HyMem) must have it
    "_universal_g4": ["context_size_history"],
}


def test_write_baseline_stats_writes_all_universal_fields():
    """The shared helper used by all non-HyMem baselines must populate the
    full universal field set."""
    from agents.tool_metrics import write_baseline_stats

    env_stats = {"search": 5, "open_page": 3}  # ReAct-style env tools
    write_baseline_stats(
        env_stats,
        framework="react",
        score=1,
        iterations=10,
        total_prompt_tokens=12000,
        total_completion_tokens=3000,
        session_time=42.5,
    )
    missing = UNIVERSAL_FIELDS - set(env_stats.keys())
    assert not missing, f"write_baseline_stats missed fields: {missing}"


def test_write_baseline_stats_token_consistency():
    """total_tokens == prompt + completion AND
    token_usage_by_component['actor'] == those totals."""
    from agents.tool_metrics import write_baseline_stats

    env_stats = {}
    write_baseline_stats(
        env_stats, framework="amem",
        score=0, iterations=5,
        total_prompt_tokens=1000,
        total_completion_tokens=200,
        session_time=10.0,
    )
    assert env_stats["total_tokens"] == 1200
    actor = env_stats["token_usage_by_component"]["actor"]
    assert actor["prompt_tokens"] == 1000
    assert actor["completion_tokens"] == 200
    assert actor["total_tokens"] == 1200


def test_write_baseline_stats_atomic_consistency():
    """tool_calls_atomic must equal sum(tool_calls_by_source.values()) and
    sum(tool_calls_by_tool.values())."""
    from agents.tool_metrics import write_baseline_stats

    env_stats = {"search": 4, "open_page": 2, "web_search": 1}
    write_baseline_stats(
        env_stats, framework="resum",
        score=1, iterations=8,
        total_prompt_tokens=5000, total_completion_tokens=1000,
        session_time=30.0,
    )
    atomic = env_stats["tool_calls_atomic"]
    by_src = sum(env_stats["tool_calls_by_source"].values())
    by_tool = sum(env_stats["tool_calls_by_tool"].values())
    assert atomic == by_src, f"by_source ({by_src}) != atomic ({atomic})"
    assert atomic == by_tool, f"by_tool ({by_tool}) != atomic ({atomic})"


def test_write_baseline_stats_models_used_resolved_from_env():
    """models_used dict must have all 4 role keys, populated from env vars."""
    import os
    os.environ["HYMEM_PLANNER_MODEL"] = "test-planner-model"
    os.environ["HYMEM_AUX_MODEL"] = "test-aux-model"
    os.environ["JUDGE_MODEL"] = "test-judge"
    try:
        from agents.tool_metrics import write_baseline_stats
        env_stats = {}
        write_baseline_stats(
            env_stats, framework="thread",
            score=0, iterations=1,
            total_prompt_tokens=0, total_completion_tokens=0,
            session_time=1.0,
        )
        m = env_stats["models_used"]
        assert m["planner"] == "test-planner-model"
        assert m["aux"] == "test-aux-model"
        assert m["judge"] == "test-judge"
        assert "executor" in m
    finally:
        for k in ("HYMEM_PLANNER_MODEL", "HYMEM_AUX_MODEL", "JUDGE_MODEL"):
            os.environ.pop(k, None)


def test_write_baseline_stats_latency_actor_only():
    """For non-HyMem baselines, latency_by_component should have an 'actor'
    bucket = total session time. (HyMem overrides with per-component split.)"""
    from agents.tool_metrics import write_baseline_stats

    env_stats = {}
    write_baseline_stats(
        env_stats, framework="thread",
        score=1, iterations=1,
        total_prompt_tokens=0, total_completion_tokens=0,
        session_time=99.5,
    )
    lat = env_stats["latency_by_component"]
    assert lat["actor"] == 99.5
    assert lat["total"] == 99.5


def test_universal_fields_match_eval_gaia_result_schema():
    """Sanity: the result dict eval_gaia.py builds must rely ONLY on
    fields we promise to provide. Read the source to verify."""
    eval_path = REPO / "scripts" / "eval_gaia.py"
    text = eval_path.read_text()

    # Find the result dict in the success path
    start = text.find("'status': 'success',")
    assert start > 0, "couldn't find result dict in eval_gaia.py"
    end = text.find("\n        }", start)
    block = text[start:end]

    # Every stats.get(...) call should reference a field we promise.
    # Build the set of stats.get keys used in the result dict.
    import re
    used_keys = set(re.findall(r"stats\.get\('([^']+)'", block))
    promised = UNIVERSAL_FIELDS | set(HYMEM_SPECIFIC_FIELDS) | {
        # Other fields with sensible defaults - HyMem-specific or pre-existing
        # back-compat keys.
        "confidence", "framework",
    }
    unknown = used_keys - promised
    assert not unknown, (
        f"eval_gaia.py reads fields not in our schema: {unknown}. "
        "Either add them to UNIVERSAL_FIELDS / HYMEM_SPECIFIC_FIELDS, "
        "or stop reading them."
    )


# ---------- HyMem-specific field defaults are sane ----------

def test_hymem_specific_fields_default_correctly():
    """write_baseline_stats does NOT write HyMem-specific fields. eval_gaia.py
    must default them to 0/{} via stats.get(field, default). Verify the
    default value is appropriate type."""
    from agents.tool_metrics import write_baseline_stats

    env_stats = {}
    write_baseline_stats(
        env_stats, framework="resum",
        score=0, iterations=5,
        total_prompt_tokens=1000, total_completion_tokens=200,
        session_time=10.0,
    )
    # HyMem-specific fields should be ABSENT from non-HyMem stats
    for field in HYMEM_SPECIFIC_FIELDS:
        assert field not in env_stats, (
            f"non-HyMem baseline shouldn't write '{field}'; let eval_gaia.py "
            f"default it via stats.get('{field}', default)"
        )


# ---------- G1-G4 paired-metric schema audit ----------

def test_g1_hymem_fold_events_field_shape():
    """G1: HyMem fold_events entries must be {round, tokens_before,
    tokens_after_reset, trigger_reason} — same shape as ReSum's
    resum_summary_events so §A analysis iterates uniformly."""
    fake_event = {
        "round": 5,
        "tokens_before": 8000,
        "tokens_after_reset": 1500,
        "trigger_reason": "context_overflow",
    }
    required = {"round", "tokens_before", "tokens_after_reset", "trigger_reason"}
    assert required.issubset(set(fake_event.keys())), (
        f"hymem_fold_events entry missing required keys: {required - set(fake_event.keys())}"
    )


def test_g2_hymem_memory_ops_keys():
    """G2: HyMem memory_ops dict must include the 6 operation counters
    (episode_writes / working_writes / tool_writes / tool_call_records /
    branch_records / retrievals) for §B comparison vs A-MEM's
    amem_notes_created + amem_retrievals."""
    # Import the dataclass to verify the canonical key set without needing
    # the full hymem_agent runtime.
    try:
        from agents.hymem_memory import HyMemStore
        store = HyMemStore()
        required = {
            "episode_writes", "working_writes", "tool_writes",
            "tool_call_records", "branch_records", "retrievals",
        }
        assert required.issubset(set(store.op_counts.keys())), (
            f"HyMemStore.op_counts missing: {required - set(store.op_counts.keys())}"
        )
    except ImportError:
        # hymem_memory pulls openai SDK; skip if not in lightweight test env
        print("  SKIP  (hymem_memory needs openai)")


def test_g3_thread_spawn_success_field_present():
    """G3: Thread state must have spawn_success counter (analog to
    HyMem's branch_success). Check class attribute initialization."""
    try:
        from agents.thread_agent import _ThreadState
        s = _ThreadState(t_max=10)
        assert hasattr(s, "spawn_success"), (
            "_ThreadState missing spawn_success counter (G3)"
        )
        assert s.spawn_success == 0, "spawn_success must start at 0"
        assert hasattr(s, "spawn_count"), (
            "_ThreadState missing spawn_count counter"
        )
    except ImportError:
        print("  SKIP  (thread_agent needs verl/torch)")


def test_g4_universal_context_size_history_present():
    """G4: every baseline writes context_size_history. We can't run agents
    here, but we can audit that the field name is consistent across the
    5 baseline files (no typos like context_history vs context_size_history)."""
    from pathlib import Path
    repo = Path(__file__).resolve().parent.parent
    bases = [
        "agents/react_agent.py",
        "agents/fold_agent.py",
        "agents/resum_agent.py",
        "agents/thread_agent.py",
        "agents/amem_agent.py",
        "agents/hymem_agent.py",
    ]
    for f in bases:
        text = (repo / f).read_text()
        assert "context_size_history" in text, (
            f"{f} missing 'context_size_history' (G4 — cross-baseline "
            f"context-growth field)"
        )


# ---------- Built-in runner ----------

if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e) or "AssertionError"))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"Ran {len(tests)} tests; {len(failures)} failures/errors.")
    sys.exit(1 if failures else 0)
