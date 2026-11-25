"""Tests that auxiliary LLM callers do not retry deterministic context errors."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents import bm_tools
from agents.hymem_memory import HyMemManager


_CONTEXT_ERROR = RuntimeError("Error code: 500 - Context size has been exceeded.")


def test_bm_tools_context_error_is_not_retried():
    create = AsyncMock(side_effect=_CONTEXT_ERROR)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    with patch("agents.bm_tools.AsyncOpenAI", return_value=client), \
            patch("agents.bm_tools.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(
            bm_tools.call_llm(
                "prompt", model="test", base_url="http://test/v1",
                api_key="test", max_retries=3,
            )
        )

    assert "Context length exceeded" in result
    assert create.await_count == 1
    assert sleep.await_count == 0


def test_hymem_manager_context_error_is_not_retried():
    create = AsyncMock(side_effect=_CONTEXT_ERROR)
    manager = HyMemManager(aux_model="test", base_url="http://test/v1", api_key="test")
    manager._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    with patch("agents.hymem_memory.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(manager._call_llm("prompt", max_retries=3))

    assert result == ""
    assert create.await_count == 1
    assert sleep.await_count == 0


def test_aux_clients_disable_hidden_sdk_retries():
    with patch("agents.hymem_memory.AsyncOpenAI") as client_cls:
        manager = HyMemManager(
            aux_model="test", base_url="http://test/v1", api_key="test",
            timeout=45,
        )
        manager.client

    client_cls.assert_called_once_with(
        api_key="test",
        base_url="http://test/v1",
        timeout=45.0,
        max_retries=0,
    )
