"""Unit tests for agents.tool_metrics — verifies all three frameworks
(HyMem / ReAct / Fold) emit the same canonical schema.

Runnable two ways:
    pytest tests/test_tool_metrics.py -v
    python -m tests.test_tool_metrics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tool_metrics import (
    UnifiedToolCounter,
    derive_atomic_from_env_stats,
    write_unified_stats,
    merge_hymem_tool_stats,
    ATOMIC_TOOL_KEYS,
)


# ---------- derive_atomic_from_env_stats ----------

def test_derive_sums_all_atomic_keys():
    stats = {"search": 5, "open_page": 3, "web_search": 2, "irrelevant": 100}
    assert derive_atomic_from_env_stats(stats) == 10  # 5 + 3 + 2


def test_derive_handles_missing_keys():
    assert derive_atomic_from_env_stats({}) == 0
    assert derive_atomic_from_env_stats({"search": 7}) == 7


def test_derive_ignores_non_numeric():
    stats = {"search": "five", "open_page": 3}
    assert derive_atomic_from_env_stats(stats) == 3


# ---------- UnifiedToolCounter ----------

def test_unified_counter_writes_canonical_schema():
    env_stats = {"search": 4, "open_page": 2}  # ReAct-style env populated this
    counter = UnifiedToolCounter(framework="react")
    counter.iterations = 7
    counter.write_into(env_stats)

    assert env_stats["tool_calls_atomic"] == 6  # 4 + 2 derived from env
    assert env_stats["agent_turns"] == 7
    assert env_stats["tool_calls_by_source"] == {"actor": 6}
    assert env_stats["framework"] == "react"
    # legacy aliases preserved
    assert env_stats["tool_calls_total"] == 6
    assert env_stats["tool_calls"] == 6


def test_unified_counter_extra_atomic_adds_to_total():
    env_stats = {"search": 3}
    counter = UnifiedToolCounter(framework="react")
    counter.iterations = 5
    counter.add_atomic(2, source="actor")
    counter.write_into(env_stats)
    assert env_stats["tool_calls_atomic"] == 5  # 3 derived + 2 extra


def test_unified_counter_zero_atomic_no_source():
    env_stats = {}  # no tools called yet
    counter = UnifiedToolCounter(framework="react")
    counter.iterations = 3
    counter.write_into(env_stats)
    assert env_stats["tool_calls_atomic"] == 0
    assert env_stats["agent_turns"] == 3
    assert env_stats["tool_calls_by_source"] == {}


# ---------- merge_hymem_tool_stats ----------

def test_merge_hymem_writes_unified_schema():
    env_stats = {}
    bm_tool_stats = {
        "tool_calls_total": 12,
        "tool_calls_by_tool": {"web_search": 6, "web_parse": 6},
        "tool_calls_by_source": {"actor": 8, "branch": 4},
    }
    merge_hymem_tool_stats(env_stats, bm_tool_stats, agent_turns=20)

    assert env_stats["tool_calls_atomic"] == 12
    assert env_stats["agent_turns"] == 20
    assert env_stats["tool_calls_by_source"] == {"actor": 8, "branch": 4}
    assert env_stats["tool_calls_by_tool"] == {"web_search": 6, "web_parse": 6}
    assert env_stats["framework"] == "hymem"


def test_merge_hymem_branch_actor_sum_equals_total():
    """Invariant: actor + branch atomic counts must equal the total."""
    env_stats = {}
    bm_tool_stats = {
        "tool_calls_total": 15,
        "tool_calls_by_tool": {},
        "tool_calls_by_source": {"actor": 10, "branch": 5},
    }
    merge_hymem_tool_stats(env_stats, bm_tool_stats, agent_turns=18)
    by_source = env_stats["tool_calls_by_source"]
    assert sum(by_source.values()) == env_stats["tool_calls_atomic"]


# ---------- Cross-framework consistency ----------

def test_three_frameworks_emit_same_canonical_keys():
    """tool_calls_atomic / agent_turns / framework / tool_calls_by_source
    must all appear regardless of how the framework derived them."""
    canonical = {"tool_calls_atomic", "agent_turns", "tool_calls_by_source", "framework"}

    # ReAct path
    react_stats = {"search": 5}
    UnifiedToolCounter(framework="react", iterations=7).write_into(react_stats)
    assert canonical.issubset(react_stats.keys())

    # Fold path (no atomic counts in env, just iterations — possible but rare)
    fold_stats = {"search": 0}
    UnifiedToolCounter(framework="fold", iterations=10).write_into(fold_stats)
    assert canonical.issubset(fold_stats.keys())

    # HyMem path
    hymem_stats = {}
    bm = {
        "tool_calls_total": 8,
        "tool_calls_by_tool": {"web_search": 8},
        "tool_calls_by_source": {"actor": 5, "branch": 3},
    }
    merge_hymem_tool_stats(hymem_stats, bm, agent_turns=12)
    assert canonical.issubset(hymem_stats.keys())


def test_atomic_tool_keys_includes_local_search_env_tools():
    """Sanity: the env-emitted tool keys (search, open_page) MUST be in the
    canonical atomic key set, otherwise ReAct would underreport calls."""
    assert "search" in ATOMIC_TOOL_KEYS
    assert "open_page" in ATOMIC_TOOL_KEYS


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
