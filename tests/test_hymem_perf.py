"""Unit tests for the HyMem efficiency optimizations (O1-O6, O8).

Each test pins the BEHAVIOR the optimization promises so a future regression
fails loudly. They run without network or LLM access.

Runnable two ways:
    pytest tests/test_hymem_perf.py -v
    python -m tests.test_hymem_perf
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import bm_tools
from agents.bm_tools import BMToolManager, call_llm
from agents.distillation import (
    chunked_filter,
    task_relevance_gate,
    CandidateFact,
)
from agents.exec_early_exit import detect_executor_diminishing
# O8 prompt module is env-gated; we re-import inside the test under
# different env vars to verify both flavors.
import importlib
import agents.hymem_prompts as _hymem_prompts_mod
import agents.hymem_prompts_qwen as _hymem_prompts_qwen_mod


# ---------- O1: web_parse cache ----------

def test_o1_cache_hit_skips_pipeline():
    """Second web_parse call with same (url, query) must hit cache and
    skip the network+distill pipeline entirely."""
    mgr = BMToolManager.__new__(BMToolManager)  # bypass __init__ (needs API keys)
    # Hand-init only the cache-related fields the test exercises.
    from collections import OrderedDict
    mgr._parse_cache = OrderedDict()
    mgr._raw_page_cache = OrderedDict()
    mgr._parse_cache_max = 256
    mgr._parse_cache_enabled = True
    mgr._parse_cache_hits = 0
    mgr._parse_cache_misses = 0
    mgr._raw_page_hits = 0
    # _record_tool_call needs these
    mgr.tool_calls_total = 0
    mgr.tool_calls_by_tool = {}
    mgr.tool_calls_by_source = {}
    mgr.tool_call_history = []
    mgr._tool_call_source = "actor"
    mgr.distillation_stats_aggregate = {
        "chunks_total": 0, "chunks_kept": 0, "chunks_dropped": 0,
        "candidates_extracted": 0, "candidates_kept_after_gate": 0,
        "total_aux_calls": 0, "invocations": 0,
        "chunk_concurrent_used_max": 0,
    }

    # Pre-warm cache with a fake hit.
    canned = {"tool_result": {"page_content": "from cache"}}
    key = (mgr._canonical_url("http://x.com/a"),
           mgr._normalize_query_for_cache("how many"))
    mgr._parse_cache[key] = canned

    # Patch the module-level web_parse (network path) so a miss would blow up.
    async def _network_should_not_fire(*a, **k):
        raise AssertionError("cache miss path was taken — O1 broken")

    with patch.object(bm_tools, "web_parse", new=_network_should_not_fire):
        result = asyncio.run(mgr.web_parse("http://x.com/a", "how many"))

    assert result["tool_result"]["page_content"] == "from cache"
    assert result["tool_result"]["cache_hit"] is True
    assert mgr._parse_cache_hits == 1
    assert mgr._parse_cache_misses == 0


def test_o1_canonical_url_collapses_trivial_variants():
    """URLs differing only by fragment / trailing slash / scheme case must
    map to the same cache key, otherwise hit rate stays artificially low."""
    a = BMToolManager._canonical_url("HTTPS://Example.COM/path/")
    b = BMToolManager._canonical_url("https://example.com/path#frag")
    assert a == b


# ---------- O2: Stage-3 gate parallel batches preserve order ----------

def test_o2_gate_parallel_correctness():
    """Parallel batched task_relevance_gate must return facts in the same
    order as the serial implementation (gather-over-list preserves order)."""
    n = 200
    candidates = [CandidateFact(f"fact{i}", "number", i) for i in range(n)]

    # Mock LLM caller: always says all indices in the batch are relevant.
    async def caller(prompt: str) -> str:
        # Count "[i] (number)" lines to know batch size.
        import re
        n_in_batch = len(re.findall(r"^\[\d+\]", prompt, re.MULTILINE))
        idxs = ",".join(str(i) for i in range(1, n_in_batch + 1))
        return f'{{"relevant_indices": [{idxs}]}}'

    os.environ["HYMEM_DISTILL_GATE_PARALLEL"] = "1"
    relevant = asyncio.run(
        task_relevance_gate(candidates, "task", caller,
                            max_candidates_per_call=60)
    )
    # Expect all 200 returned, in original order.
    assert len(relevant) == n
    assert [f.text for f in relevant] == [f"fact{i}" for i in range(n)]


# ---------- O3: chunked_filter env-overridable concurrency ----------

def test_o3_chunk_concurrency_env_override():
    """HYMEM_DISTILL_CHUNK_CONCURRENCY env var must override the
    caller-supplied default and be reported in stats."""
    content = "X" * 1500 + "Y" * 1500 + "Z" * 1500

    async def checker(content_, task_):
        return {"tool_result": {"is_relevant": "yes", "reasoning": ""}}

    os.environ["HYMEM_DISTILL_CHUNK_CONCURRENCY"] = "12"
    try:
        _, stats = asyncio.run(
            chunked_filter(content, "task", checker,
                           chunk_size=1500, overlap=0,
                           max_concurrent=4)  # caller default would be 4
        )
        assert stats["max_concurrent_used"] == 12
    finally:
        del os.environ["HYMEM_DISTILL_CHUNK_CONCURRENCY"]


# ---------- O4: exponential backoff + jitter on LLM retries ----------

def test_o4_exponential_backoff():
    """call_llm retry sleeps must follow min(8, 2**attempt) + [0,0.5) jitter,
    not the previous linear 1/2/3 schedule."""
    sleeps = []

    async def fake_sleep(d):
        sleeps.append(d)

    # Always-failing client. AsyncOpenAI is the constructor — return an
    # instance whose .chat.completions.create raises.
    class _Completions:
        @staticmethod
        async def create(**kw):
            raise RuntimeError("boom")

    class _Chat:
        completions = _Completions()

    class _FailingClient:
        chat = _Chat()

    def _failing_factory(*a, **kw):
        return _FailingClient()

    with patch.object(bm_tools, "AsyncOpenAI", new=_failing_factory), \
         patch.object(bm_tools.asyncio, "sleep", new=fake_sleep):
        result = asyncio.run(
            call_llm("p", model="m", api_key="k", base_url="http://x",
                     max_retries=4)
        )
    # On 4 attempts there are 3 sleeps (after attempts 0,1,2; the last attempt
    # has no following sleep — it returns the error).
    assert result.startswith("Error"), f"unexpected: {result!r}"
    assert len(sleeps) == 3, f"expected 3 sleeps, got {sleeps}"
    # attempt 0 → base 1, attempt 1 → base 2, attempt 2 → base 4.
    assert 1.0 <= sleeps[0] < 1.5
    assert 2.0 <= sleeps[1] < 2.5
    assert 4.0 <= sleeps[2] < 4.5


# ---------- O5: executor diminishing-returns detection ----------

def test_o5_early_exit_on_repeat_output():
    """Two consecutive observations with identical first-200-chars trigger
    a 'loop' early-exit signal."""
    obs = "Code output:\n" + "tegla loroupe born 1973 in Kenya. " * 20
    assert detect_executor_diminishing([obs, obs]) == "loop"


def test_o5_early_exit_on_low_info():
    """Two short observations with no digits/entities → 'low_info'."""
    a = "no result"
    b = "nothing here"
    assert detect_executor_diminishing([a, b]) == "low_info"


def test_o5_early_exit_on_consecutive_errors():
    a = "Code output:\nError: timeout"
    b = "Code output:\nError: 429 too many requests"
    assert detect_executor_diminishing([a, b]) == "consecutive_errors"


def test_o5_no_exit_when_progress_is_made():
    """Two informative, distinct observations must NOT trigger early exit."""
    a = "Code output:\nTegla Loroupe born May 9 1973 in Kenya."
    b = "Code output:\nMargaret Okayo set 10000m PB of 30:32.03 in 1999."
    assert detect_executor_diminishing([a, b]) is None


# ---------- O6: fold aux extraction runs in parallel ----------

def test_o6_fold_aux_parallel():
    """generate_memories_from_summary runs episode/working/tool extraction
    via asyncio.gather; total wall clock ≈ max single duration, not sum."""
    from agents.hymem_memory import HyMemManager

    mgr = HyMemManager.__new__(HyMemManager)
    mgr.token_usage = type("T", (), {"calls": []})()
    mgr.fold_aux_durations = []

    sleep_s = 0.3

    async def slow_extract(*a, **k):
        await asyncio.sleep(sleep_s)
        return {"ok": True}

    mgr._extract_episode_memory = slow_extract
    mgr._extract_working_memory = slow_extract
    mgr._extract_tool_memory = slow_extract

    t0 = time.time()
    e, w, t = asyncio.run(mgr.generate_memories_from_summary(
        question="q", planner_summary="s", tool_call_stats="", branch_insights=[]
    ))
    elapsed = time.time() - t0
    # Serial would be ~3*sleep_s = 0.9s; parallel must be < 1.6 * sleep_s.
    assert elapsed < sleep_s * 1.6, f"fold aux not parallel: {elapsed:.2f}s"
    assert e["ok"] and w["ok"] and t["ok"]
    # Timing was recorded for the observation field.
    assert len(mgr.fold_aux_durations) == 1
    assert mgr.fold_aux_durations[0] < sleep_s * 1.6


# ---------- O8: planner prompt parallel-emission nudge is env-gated ----------

def _reimport_prompts(env_value):
    """Re-import prompt modules with HYMEM_PARALLEL_PROMPT set to env_value
    (or unset if None). Returns (EN_prompt, Qwen_prompt) constants."""
    if env_value is None:
        os.environ.pop("HYMEM_PARALLEL_PROMPT", None)
    else:
        os.environ["HYMEM_PARALLEL_PROMPT"] = env_value
    importlib.reload(_hymem_prompts_mod)
    importlib.reload(_hymem_prompts_qwen_mod)
    return (_hymem_prompts_mod.HYMEM_PLANNER_SYSTEM_PROMPT,
            _hymem_prompts_qwen_mod.HYMEM_PLANNER_SYSTEM_PROMPT_QWEN)


def test_o8_default_off_planner_prompt_does_not_push_parallel():
    """Default (HYMEM_PARALLEL_PROMPT unset): planner prompts must NOT
    aggressively push parallel emission — that nudge regressed L3 acc -10pp
    and inflated session_time +42% in the 127-task paired A/B."""
    try:
        en, qwen = _reimport_prompts(None)
        # EN: aggressive section removed, conservative one in its place.
        assert "use AGGRESSIVELY" not in en
        assert "Find the founding year of Stanford University" not in en
        assert "TRULY independent" in en  # conservative wording present
        # Qwen: parallel example block absent by default.
        assert "并行执行" not in qwen
        assert "斯坦福大学" not in qwen
    finally:
        _reimport_prompts(None)


def test_o8_opt_in_enables_aggressive_parallel_hint():
    """When HYMEM_PARALLEL_PROMPT=1 (explicit ablation opt-in), the
    aggressive nudge + examples must come back."""
    try:
        en, qwen = _reimport_prompts("1")
        assert "PARALLEL DELEGATION" in en
        assert "Find the founding year of Stanford University" in en
        assert "并行执行" in qwen
        assert "斯坦福大学" in qwen
    finally:
        _reimport_prompts(None)


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
