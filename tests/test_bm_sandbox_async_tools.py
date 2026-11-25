"""Tests for automatic awaiting of async tools in model-generated code."""

import asyncio

from agents.bm_sandbox import SandboxSession


def test_sandbox_awaits_async_tool_call_when_model_omits_await():
    calls = []

    async def web_search(query):
        calls.append(query)
        return {"query": query, "results": [1]}

    session = SandboxSession("auto-await", {"web_search": web_search})
    stdout, error, _ = asyncio.run(
        session.execute_async(
            'result = web_search("qwen")\nprint(result["query"], len(result["results"]))'
        )
    )

    assert error is None
    assert stdout.strip() == "qwen 1"
    assert calls == ["qwen"]


def test_sandbox_preserves_existing_await():
    calls = []

    async def generate_keywords(seed):
        calls.append(seed)
        return [seed]

    session = SandboxSession("existing-await", {"generate_keywords": generate_keywords})
    stdout, error, _ = asyncio.run(
        session.execute_async(
            'result = await generate_keywords("gaia")\nprint(result[0])'
        )
    )

    assert error is None
    assert stdout.strip() == "gaia"
    assert calls == ["gaia"]
