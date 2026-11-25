"""Smoke tests for parallel sub-task dispatch (Optimization 1) +
selective branch context inheritance (Optimization 4).

Runnable two ways:
    pytest tests/test_parallel_dispatch.py -v
    python -m tests.test_parallel_dispatch
"""

import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Vendored copies of the new extractors so the test runs without pulling
# torch/verl via agents.hymem_agent.

def _extract_all_tasks(text):
    return [m.strip() for m in re.findall(r'<task>(.*?)</task>', text, re.DOTALL) if m.strip()]


def _extract_all_branches(text):
    blocks = re.findall(r'<branch>(.*?)</branch>', text, re.DOTALL)
    out = []
    for content in blocks:
        d = re.search(r'<description>(.*?)</description>', content, re.DOTALL)
        p = re.search(r'<prompt>(.*?)</prompt>', content, re.DOTALL)
        if d and p:
            out.append({'description': d.group(1).strip(), 'prompt': p.group(1).strip()})
    return out


# ---------- Multi-extract parsers ----------

def test_extract_all_tasks_finds_multiple():
    response = """<think>I need three independent facts.</think>
<task>find X's birthdate</task>
<task>find Y's birthdate</task>
<task>find Z's birthdate</task>"""
    tasks = _extract_all_tasks(response)
    assert len(tasks) == 3
    assert tasks[0] == "find X's birthdate"
    assert tasks[2] == "find Z's birthdate"


def test_extract_all_branches_finds_multiple():
    response = """<branch>
<description>Verify A</description><prompt>Check A satisfies criteria.</prompt>
</branch>
<branch>
<description>Verify B</description><prompt>Check B satisfies criteria.</prompt>
</branch>"""
    branches = _extract_all_branches(response)
    assert len(branches) == 2
    assert branches[0]['description'] == "Verify A"
    assert branches[1]['prompt'].startswith("Check B")


def test_mixed_task_and_branch_in_same_response():
    response = """<task>simple lookup</task>
<branch><description>Hard subproblem</description><prompt>Verify XYZ.</prompt></branch>
<task>another lookup</task>"""
    assert len(_extract_all_tasks(response)) == 2
    assert len(_extract_all_branches(response)) == 1


def test_extract_all_returns_empty_lists_for_no_tags():
    response = "Just thinking, no tags here."
    assert _extract_all_tasks(response) == []
    assert _extract_all_branches(response) == []


# ---------- Concurrent execution semantics ----------

def test_concurrent_dispatch_finishes_in_max_time_not_sum():
    """Three sub-tasks each taking 0.5s should finish in ~0.5s wall-clock
    when dispatched concurrently — not 1.5s as a serial loop would."""

    async def fake_subtask(idx):
        await asyncio.sleep(0.5)
        return idx

    async def serial():
        out = []
        for i in range(3):
            out.append(await fake_subtask(i))
        return out

    async def parallel():
        return await asyncio.gather(*[fake_subtask(i) for i in range(3)])

    t = time.time()
    asyncio.run(serial())
    serial_t = time.time() - t

    t = time.time()
    asyncio.run(parallel())
    parallel_t = time.time() - t

    assert parallel_t < 0.8, f"parallel took {parallel_t:.2f}s — concurrency broken"
    assert serial_t > 1.4, f"serial took {serial_t:.2f}s — fake too fast"
    assert parallel_t < serial_t / 2


def test_dispatch_preserves_emit_order_in_results():
    """Even though sub-tasks complete out of order, the aggregated result list
    must come back in the planner's emit order (so observation reads naturally)."""

    async def fake_subtask(idx):
        await asyncio.sleep(0.05 if idx % 2 == 0 else 0.2)
        return {'idx': idx, 'result': f'r{idx}'}

    async def run():
        results = await asyncio.gather(*[fake_subtask(i) for i in range(4)])
        return sorted(results, key=lambda x: x['idx'])

    out = asyncio.run(run())
    assert [r['idx'] for r in out] == [0, 1, 2, 3]
    assert [r['result'] for r in out] == ['r0', 'r1', 'r2', 'r3']


def test_concurrency_capped_by_semaphore():
    """4-task fan-out under a 2-concurrency cap should serialize into 2
    waves of 0.3s each ≈ 0.6s total (not 0.3s like fully-parallel)."""

    in_flight = {"now": 0, "max": 0}
    sem = asyncio.Semaphore(2)

    async def fake_subtask():
        async with sem:
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            await asyncio.sleep(0.3)
            in_flight["now"] -= 1

    async def run():
        await asyncio.gather(*[fake_subtask() for _ in range(4)])

    t = time.time()
    asyncio.run(run())
    elapsed = time.time() - t
    assert in_flight["max"] == 2, f"semaphore failed: max in-flight = {in_flight['max']}"
    assert 0.55 < elapsed < 0.85, f"serialization wrong: {elapsed:.2f}s (expected ~0.6)"


# ---------- Branch context selective inheritance (Optimization 4) ----------

def _build_inherited(planner_messages, has_memory):
    keep_last_n = 3 if has_memory else 5
    user_budget = 600 if has_memory else 1000
    asst_budget = 500 if has_memory else 800
    tail = planner_messages[1:][-keep_last_n * 2:]
    seen = set()
    parts = []
    for msg in tail:
        role = msg.get('role', '')
        content = (msg.get('content', '') or '').strip()
        if not content:
            continue
        if role == 'user':
            sig = content[:120]
            if sig in seen:
                continue
            seen.add(sig)
            parts.append(content[:user_budget])
        elif role == 'assistant':
            parts.append(content[:asst_budget])
    return parts


def test_branch_context_shrinks_when_memory_is_rich():
    msgs = [{'role': 'system', 'content': 'sys'}]
    for i in range(10):
        msgs.append({'role': 'user', 'content': f'OBS {i} ' + 'x' * 2000})
        msgs.append({'role': 'assistant', 'content': f'PLAN {i} ' + 'y' * 2000})

    no_mem = "\n".join(_build_inherited(msgs, has_memory=False))
    with_mem = "\n".join(_build_inherited(msgs, has_memory=True))
    assert len(with_mem) < len(no_mem)
    assert len(with_mem) < 0.7 * len(no_mem), (
        f"selective inheritance only saved {1 - len(with_mem)/len(no_mem):.0%} (need >=30%)"
    )


def test_branch_context_dedupes_repeated_user_observations():
    dup = "Same observation appearing twice " * 4
    msgs = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': dup},
        {'role': 'assistant', 'content': 'plan A'},
        {'role': 'user', 'content': dup},   # duplicate observation
        {'role': 'assistant', 'content': 'plan B'},
    ]
    parts = _build_inherited(msgs, has_memory=False)
    user_parts = [p for p in parts if dup[:30] in p]
    assert len(user_parts) == 1, f"dedup failed: kept {len(user_parts)} duplicate user msgs"


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
