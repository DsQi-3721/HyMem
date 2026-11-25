"""A-MEM LLM controller — patched fork of upstream.

Source: agiresearch/A-mem @ ceffb860f071, agentic_memory/llm_controller.py
License: MIT

Patches applied (diff vs upstream):
    + OpenAIController.__init__ accepts base_url (forwarded to OpenAI client)
    + OpenAIController.__init__ accepts api_key/base_url fallbacks from
      HYMEM_AUX_* env vars before falling back to OPENAI_API_KEY
    + Lazy import of `litellm` (only required for OllamaController, which
      we do not exercise in our reproduction)

These changes preserve API compatibility — A-MEM's `AgenticMemorySystem`
calls `self.llm_controller.llm.get_completion(...)` with no awareness of
the patch.
"""

from typing import Dict, Optional, Literal, Any
import os
import json
from abc import ABC, abstractmethod


class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM."""
        pass


class OpenAIController(BaseLLMController):
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. pip install openai")

        self.model = model
        # Resolve API key with HyMem env var precedence (no new env vars).
        if api_key is None:
            api_key = (
                os.getenv("HYMEM_AUX_API_KEY")
                or os.getenv("HYMEM_PLANNER_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
        if api_key is None:
            raise ValueError(
                "No API key found. Set HYMEM_AUX_API_KEY / "
                "HYMEM_PLANNER_API_KEY / OPENAI_API_KEY."
            )
        # Same for base_url — lets us route to DeepSeek transparently.
        if base_url is None:
            base_url = (
                os.getenv("HYMEM_AUX_BASE_URL")
                or os.getenv("HYMEM_PLANNER_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            )
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    # Class-level flag so we only print the "endpoint rejected response_format"
    # warning once per process (otherwise A-MEM floods stderr — there are
    # ~6 memory-evolution calls per add_note × N notes per task).
    _warned_no_response_format = False

    def get_completion(
        self,
        prompt: str,
        response_format: dict = None,
        temperature: float = 0.7,
    ) -> str:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

        # Downgrade `json_schema` (OpenAI structured-output) to `json_object`
        # (free-form JSON). DeepSeek API supports json_object on
        # deepseek-chat but rejects json_schema everywhere; OpenAI accepts
        # both. The system prompt above already steers the model to
        # JSON, so this downgrade rarely changes behavior on capable
        # models and unblocks DeepSeek entirely.
        if response_format and isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                response_format = {"type": "json_object"}
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            err_str = str(e)
            # Some endpoints (e.g. deepseek-reasoner) don't support
            # response_format AT ALL. On 400 specifically about
            # response_format, retry without it. We still get JSON-ish
            # output thanks to the system prompt.
            if "response_format" in err_str.lower() and "400" in err_str:
                if not OpenAIController._warned_no_response_format:
                    print(
                        f"[A-MEM/llm_controller] Endpoint rejected response_format "
                        f"({self.model}); retrying without it. Memory "
                        f"evolution may produce slightly less structured output."
                    )
                    OpenAIController._warned_no_response_format = True
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise

        content = response.choices[0].message.content
        # DeepSeek-Reasoner sometimes returns content="" with reasoning_content
        if not content:
            msg = response.choices[0].message
            content = getattr(msg, "reasoning_content", None) or ""
        return content


class OllamaController(BaseLLMController):
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        # Lazy import — litellm is only needed for Ollama backend.
        try:
            from litellm import completion as _litellm_completion
            self._completion = _litellm_completion
        except ImportError:
            raise ImportError(
                "Ollama backend requires litellm. pip install litellm"
            )

    def get_completion(self, prompt: str, response_format: dict = None,
                       temperature: float = 0.7) -> str:
        response = self._completion(
            model=f"ollama_chat/{self.model_name}",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            format="json" if response_format else None,
        )
        return response.choices[0].message.content


class LLMController:
    """Dispatcher matching upstream API."""

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model=model, api_key=api_key, base_url=base_url)
        elif backend == "ollama":
            self.llm = OllamaController(model_name=model)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")

    def get_completion(self, *args, **kwargs):
        return self.llm.get_completion(*args, **kwargs)
