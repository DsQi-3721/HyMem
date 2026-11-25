"""Network-free tests for CallAPI retry classification."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.hymem_agent import HyMemAgent
from agents.utils import CallAPI, ContextLengthExceededError


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return [1, 2, 3]


def _response(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )


def _call_api(side_effect):
    completions = SimpleNamespace(create=AsyncMock(side_effect=side_effect))
    api = CallAPI.__new__(CallAPI)
    api.tokenizer = _Tokenizer()
    api.config = SimpleNamespace(
        prompt_length=16,
        response_length=16,
        plugin=SimpleNamespace(turn_max_new_tokens=-1),
    )
    api.model = "test-model"
    api.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    api.temperature = 0.6
    api.top_p = 0.95
    api.presence_penalty = 1.1
    api.stop_tokens = CallAPI.DEFAULT_STOP_TOKENS
    return api, completions.create


def test_context_overflow_is_not_retried():
    api, create = _call_api(
        RuntimeError("Error code: 500 - Context size has been exceeded.")
    )

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(api.create_completion([1, 2, 3], messages=[{"role": "user", "content": "x"}]))

    assert create.await_count == 1


def test_exhausted_client_context_budget_never_calls_api():
    api, create = _call_api(_response())

    with pytest.raises(ContextLengthExceededError, match="Client context budget exhausted"):
        asyncio.run(
            api.create_completion(
                list(range(32)),
                messages=[{"role": "user", "content": "x"}],
            )
        )

    assert create.await_count == 0


def test_transient_network_error_is_still_retried():
    api, create = _call_api([RuntimeError("Connection error"), _response()])

    with patch("agents.utils.asyncio.sleep", new=AsyncMock()) as sleep:
        result = asyncio.run(
            api.create_completion([1, 2, 3], messages=[{"role": "user", "content": "x"}])
        )

    assert result["choices"][0]["message"]["content"] == "ok"
    assert create.await_count == 2
    assert sleep.await_count == 1


def test_client_uses_one_configurable_retry_layer():
    config = SimpleNamespace(
        prompt_length=16,
        response_length=16,
        plugin=SimpleNamespace(
            temperature=0.6,
            top_p=0.95,
            presence_penalty=1.1,
            stop_tokens=None,
        ),
    )

    with patch.dict(
        "os.environ",
        {"HYMEM_API_TIMEOUT": "45", "HYMEM_API_MAX_ATTEMPTS": "2"},
    ), patch("openai.AsyncOpenAI") as client_cls:
        api = CallAPI(
            "test-model", None, _Tokenizer(), config, {},
            base_url="http://test/v1", api_key="test",
        )

    assert api.api_timeout == 45.0
    assert api.api_max_attempts == 2
    client_cls.assert_called_once_with(
        api_key="test",
        base_url="http://test/v1",
        timeout=45.0,
        max_retries=0,
    )


def test_hymem_propagates_context_overflow_instead_of_returning_none():
    agent = HyMemAgent.__new__(HyMemAgent)
    agent.tokenizer = _Tokenizer()
    agent.event_log = SimpleNamespace(
        latency=SimpleNamespace(add=lambda *args, **kwargs: None)
    )
    client = SimpleNamespace(
        create_completion=AsyncMock(
            side_effect=ContextLengthExceededError("context length exceeded")
        )
    )

    with pytest.raises(ContextLengthExceededError):
        asyncio.run(
            agent._call_llm(
                client,
                [{"role": "user", "content": "x"}],
                caller="planner_t2",
            )
        )

    assert client.create_completion.await_count == 1
