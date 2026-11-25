import os
import time
import copy
import uuid
import json
import random
from unittest.mock import patch
from itertools import groupby
import re, unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Optional
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer
import aiohttp
import torch
import asyncio, httpx
from verl import DataProto
from envs.local_search import LocalSearch

# DEBUG 配置 - 设置环境变量 REACT_DEBUG=true 开启
REACT_DEBUG = os.getenv("REACT_DEBUG", "false").lower() == "true"

def debug_log(msg: str):
    """条件输出 debug 日志"""
    if REACT_DEBUG:
        print(msg)


class ContextLengthExceededError(RuntimeError):
    """The model rejected a request because its context window was exceeded."""


_CONTEXT_LENGTH_ERROR_MARKERS = (
    "context size has been exceeded",
    "context length has been exceeded",
    "context length exceeded",
    "maximum context length",
    "max context length",
    "context_length_exceeded",
    "input is too long",
    "too many tokens",
)


def is_context_length_exceeded_error(error: Exception) -> bool:
    """Return whether an API exception represents a deterministic context overflow."""
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(marker in error_text for marker in _CONTEXT_LENGTH_ERROR_MARKERS)


# ============== 干扰信息注入配置 ==============
# 设置环境变量开启干扰注入实验:
#   DISTRACTOR_ENABLED=true           - 开启干扰注入
#   DISTRACTOR_PATH=/path/to/hotpotqa_dev.json  - HotpotQA 数据路径
#   DISTRACTOR_RATIO=0.5              - 注入概率 (0-1)
#   DISTRACTOR_PARAGRAPHS=2           - 每次注入段落数

class DistractorInjector:
    """干扰信息注入器 - 模拟上下文累积过程中的噪声引入"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.enabled = os.getenv("DISTRACTOR_ENABLED", "false").lower() == "true"
        self.hotpotqa_path = os.getenv("DISTRACTOR_PATH", "")
        self.injection_ratio = float(os.getenv("DISTRACTOR_RATIO", "0.5"))
        self.paragraphs_per_injection = int(os.getenv("DISTRACTOR_PARAGRAPHS", "2"))
        
        self.hotpotqa_data = []
        self.stats = {"total_observations": 0, "injected": 0, "total_paragraphs": 0}
        
        if self.enabled and self.hotpotqa_path:
            try:
                with open(self.hotpotqa_path, 'r', encoding='utf-8') as f:
                    self.hotpotqa_data = json.load(f)
                print(f"[Distractor] Loaded {len(self.hotpotqa_data)} items from {self.hotpotqa_path}")
                print(f"[Distractor] Injection ratio: {self.injection_ratio}, paragraphs: {self.paragraphs_per_injection}")
            except Exception as e:
                print(f"[Distractor] Failed to load hotpotqa data: {e}")
                self.enabled = False
    
    def get_distractor_paragraphs(self) -> str:
        """从 HotpotQA 随机获取干扰段落"""
        if not self.hotpotqa_data:
            return ""
        
        paragraphs = []
        # 随机选择多个不同的条目
        samples = random.sample(self.hotpotqa_data, min(self.paragraphs_per_injection, len(self.hotpotqa_data)))
        
        for item in samples:
            context = item.get('context', [])
            if context:
                # 随机选一个文档
                doc = random.choice(context)
                doc_name, doc_paragraphs = doc[0], doc[1] if len(doc) > 1 else []
                if doc_paragraphs:
                    para = random.choice(doc_paragraphs)
                    if len(para) > 30:  # 确保有实质内容
                        paragraphs.append(f"[{doc_name}] {para[:600]}")
        
        return "\n\n".join(paragraphs)
    
    def inject(self, observation: str) -> str:
        """向观察结果中注入干扰信息"""
        if not self.enabled or not observation:
            return observation
        
        self.stats["total_observations"] += 1
        
        # 按概率决定是否注入
        if random.random() > self.injection_ratio:
            return observation
        
        distractor = self.get_distractor_paragraphs()
        if not distractor:
            return observation
        
        self.stats["injected"] += 1
        self.stats["total_paragraphs"] += distractor.count("[")
        
        # 注入格式：模拟搜索结果中的额外相关信息
        injected_observation = (
            f"{observation}\n\n"
            f"--- Additional Related Information ---\n"
            f"{distractor}\n"
            f"--- End of Additional Information ---"
        )
        
        return injected_observation
    
    def print_stats(self):
        """打印注入统计"""
        if self.enabled:
            print(f"\n[Distractor Stats] {self.stats}")
            if self.stats["total_observations"] > 0:
                ratio = self.stats["injected"] / self.stats["total_observations"]
                print(f"[Distractor Stats] Actual injection rate: {ratio:.2%}")


# 全局注入器实例
_distractor_injector = DistractorInjector()


def get_distractor_stats():
    """获取干扰注入统计信息"""
    return _distractor_injector.stats.copy()


def print_distractor_stats():
    """打印干扰注入统计信息"""
    _distractor_injector.print_stats()


def select_env(ability, config, extra_info=None):
    """
    选择环境类
    
    Args:
        ability: 任务能力类型
        config: 配置对象
        extra_info: 额外信息
    
    Returns:
        对应的环境类
    """
    # 检查是否使用在线搜索
    search_mode = None
    if hasattr(config, 'plugin'):
        search_mode = getattr(config.plugin, "search_mode", None)
    search_mode = search_mode or os.getenv("SEARCH_MODE", "local")
    
    # 判断是否需要使用在线搜索
    use_online = (
        search_mode == "online" or
        os.getenv("USE_ONLINE_SEARCH", "").lower() in ("true", "1", "yes") or
        'browsecomp_zh' in ability.lower() or   # BrowseComp-ZH 专用
        'onlinesearch' in ability.lower() or
        'online' in ability.lower()
    )
    
    if use_online:
        from envs.online_search import OnlineSearch
        return OnlineSearch
    elif ability == 'swe':
        EnvClass = None  # TODO docker env
    elif ability == 'swe_loc':
        EnvClass = None  # TODO read-only swe env
    elif 'LocalSearch' in ability:
        EnvClass = LocalSearch
    else:
        EnvClass = LocalSearch
    return EnvClass


async def call_openai(messages, model='gpt-5-nano', max_retries=3):
    openai_url = os.getenv("OPENAI_URL")
    if isinstance(messages, str):
        messages = [{'role': 'user', 'content': messages}]

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=300.0) as c:
                r = await c.post(openai_url, json={
                    "model": model,
                    "messages": messages
                })
                r.raise_for_status()
                return r.json()["content"]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[CALL OPENAI] Error after {max_retries} attempts: {str(e)}")
                return f"Error after {max_retries} attempts: {str(e)}"
            await asyncio.sleep(1 * (attempt + 1))
    return ""

def decode_conversation(input_ids: list[int], tokenizer) -> tuple[list[dict[str, str]], str]:
    decoded_str = tokenizer.decode(input_ids, skip_special_tokens=False)
    pattern = re.compile(
        re.escape(tokenizer.bos_token)
        + r'(system|user|assistant|tool)\n'
        + r'(.*?)'
        + r'(?=' + re.escape(tokenizer.eos_token) + r')',
        re.DOTALL,
    )
    matches = pattern.findall(decoded_str)
    conversation = [{'role': role, 'content': content} for role, content in matches]
    return conversation, decoded_str

def truncate_text(
        text: str,
        max_lines: int | None = None,
        max_length: int | None = None,
        merge_repeat: bool = False,
        merge_num: int = 128,
        keep_tail_lines: int = 5,
) -> str:
    lines = text.splitlines()

    # 1) Merge repeated lines if requested
    if merge_repeat:
        merged: list[str] = []
        for line, group in groupby(lines):
            grp = list(group)
            cnt = len(grp)
            if cnt > merge_num:
                merged += [line] * 2
                merged.append(f"[This line repeated {cnt - 4} more times]")
                merged += [line] * 2
            else:
                merged += grp
        lines = merged

    # 2) Line-count truncation (keep last keep_tail_lines)
    if max_lines is not None and len(lines) > max_lines:
        total = len(lines)
        if max_lines <= keep_tail_lines + 1:
            lines = lines[:max_lines]
        else:
            head_count = max_lines - keep_tail_lines - 1
            head = lines[:head_count]
            tail = lines[-keep_tail_lines:]
            omitted = total - head_count - keep_tail_lines
            lines = head + [f"… {omitted} lines omitted …"] + tail

    # 3) Per-line character-length truncation
    if max_length is not None:
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > max_length:
                truncated_lines.append(line[:max_length] + "… (truncated)")
            else:
                truncated_lines.append(line)
        lines = truncated_lines
    return "\n".join(lines)


def is_weird(text, repeat_n=128, cjk_limit=128):
    s = unicodedata.normalize('NFKC', text)
    if re.search(rf'(.)\1{{{repeat_n - 1},}}|(.{{2,12}})\2{{{repeat_n - 1},}}', s):
        return True
    CJK = ((0x4E00, 0x9FFF), (0x3040, 0x309F), (0x30A0, 0x30FF), (0xAC00, 0xD7AF))
    cjk_count = sum(any(a <= ord(c) <= b for a, b in CJK) for c in s)
    return cjk_count >= cjk_limit or (len(s) > 0 and cjk_count / len(s) > 0.8)


class CallLLM:  # Call policy LLM in RL env
    # ReAct 工具调用的 stop tokens，参考 DeepResearch 实现
    DEFAULT_STOP_TOKENS = [
        "<tool_response>",
        "\n<tool_response>",
        "</function>\n\nObservation:",
        "</function>\nObservation:",
    ]
    
    def __init__(self, host, port, tokenizer, config, meta_info):
        if ':' in host:
            host = f'[{host}]'
        url = f"http://{host}:{port}/chat/completions"

        self.url = url
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        self.call_openai = getattr(config.plugin, "call_openai", None)
        
        # 从 config.plugin 读取生成参数，参考 DeepResearch 实现
        # temperature: 0.6 更稳定，1.0 更随机
        self.temperature = getattr(config.plugin, 'temperature', 0.6)
        # presence_penalty: 1.1 减少重复生成
        self.presence_penalty = getattr(config.plugin, 'presence_penalty', 1.1)
        # stop tokens: 防止模型在工具调用后继续生成
        self.stop_tokens = getattr(config.plugin, 'stop_tokens', None) or self.DEFAULT_STOP_TOKENS

    async def _create_completion(self, input_ids, **kwargs):
        generation_kwargs = self.meta_info['generation_kwargs']
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_len = min(max_len, self.config.prompt_length + self.config.response_length)
        max_tokens = max_len - len(input_ids)
        # This is used to avoid repetitive generation.
        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs['max_new_tokens'])

        if max_tokens < 10:
            debug_log(f"[DEBUG] max_tokens {max_tokens}, skip rollout")
            return None

        uid = kwargs.pop('uid', self.meta_info.get('uid', None))
        
        # 使用实例配置的参数，如果 generation_kwargs 中有则优先使用
        temperature = generation_kwargs.get('temperature', self.temperature)
        presence_penalty = generation_kwargs.get('presence_penalty', self.presence_penalty)

        request_data = {
            "model": "rollout",
            "messages": {'prompt': input_ids},
            "top_p": generation_kwargs['top_p'],
            "top_k": generation_kwargs['top_k'],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_length": max_len,
            "meta_info": self.meta_info | {'uid': uid},
            # 参考 DeepResearch 添加的参数
            "stop": self.stop_tokens,
            "presence_penalty": presence_penalty,
        }

        import asyncio

        for attempt in range(10):
            try:
                timeout = aiohttp.ClientTimeout(total=9600)
                session = aiohttp.ClientSession(timeout=timeout)
                async with session.post(url=self.url,
                                        headers={"Authorization": "Bearer token-abc123"},
                                        json=request_data,
                                        timeout=timeout) as response:
                    completion = await response.json()
                    completion['choices'][0]['message']['extra_data']['input_ids'] = input_ids
                    assert response.status == 200, f"chat_completions failed msg: {completion}"
                    await session.close()
                    return completion

            except Exception as e:
                print(f"[CallLLM ERROR] {e}")
                await session.close()
                await asyncio.sleep(2 ** attempt)
                if attempt < 2:
                    pass
                else:
                    import traceback
                    traceback.print_exc()
        return completion

    async def create_completion(self, input_ids, **kwargs):
        if self.call_openai:
            # Evaluate API models
            messages = kwargs.get('messages', None) or decode_conversation(input_ids, self.tokenizer)[0]
            kwargs['max_new_tokens'] = 100
            completion, text = await asyncio.gather(
                self._create_completion(input_ids, **kwargs),
                call_openai(model=self.call_openai, messages=messages)
            )
            if completion is not None:
                if text is None:
                    return None
                text_ids = self.tokenizer.encode(text)
                # print('[OPENAI]', len(text_ids))
                completion["choices"][0]["message"]["content"] = text
                completion["choices"][0]["message"]["raw_output_ids"] = text_ids
                completion["choices"][0]["message"]["response_log_probs"] = [0.0] * len(text_ids)
        else:
            completion = await self._create_completion(input_ids, **kwargs)
        return completion

class CallAPI:  # Call external API
    """
    外部 API 调用类，支持独立配置 base_url 和 api_key
    
    Args:
        host: 模型名称
        port: 端口（兼容参数，未使用）
        tokenizer: 分词器
        config: 配置对象
        meta_info: 元信息
        base_url: API 基础 URL（可选，默认使用 OPENAI_BASE_URL 环境变量）
        api_key: API 密钥（可选，默认使用 OPENAI_API_KEY 环境变量）
    """
    # ReAct 工具调用的 stop tokens，防止模型在工具调用后继续生成
    DEFAULT_STOP_TOKENS = [
        "<tool_response>",
        "\n<tool_response>",
        "</function>\n\nObservation:",
        "</function>\nObservation:",
    ]
    
    def __init__(self, host, port, tokenizer, config, meta_info, 
                 base_url: str = None, api_key: str = None):
        self.tokenizer = tokenizer
        self.config = config
        self.meta_info = meta_info
        self.model = host
        from openai import AsyncOpenAI
        import os
        # 支持独立配置，优先使用传入参数，其次使用环境变量
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_timeout = float(os.getenv("HYMEM_API_TIMEOUT", "600"))
        self.api_max_attempts = max(
            1, int(os.getenv("HYMEM_API_MAX_ATTEMPTS", "10"))
        )
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.api_timeout,
            # Retry classification belongs to create_completion(). Keeping
            # SDK retries enabled creates a hidden nested retry loop and also
            # retries deterministic HTTP 500 context-overflow responses.
            max_retries=0,
        )
        
        # 从 config.plugin 读取生成参数，提供合理默认值
        self.temperature = getattr(config.plugin, 'temperature', 0.6)
        self.top_p = getattr(config.plugin, 'top_p', 0.95)
        self.presence_penalty = getattr(config.plugin, 'presence_penalty', 1.1)
        self.stop_tokens = getattr(config.plugin, 'stop_tokens', None) or self.DEFAULT_STOP_TOKENS

    async def create_completion(self, input_ids, **kwargs):
        max_len = kwargs.pop('max_len', None) or self.config.prompt_length + self.config.response_length
        max_tokens = min(max_len, self.config.prompt_length + self.config.response_length) - len(input_ids)

        if getattr(self.config.plugin, 'turn_max_new_tokens', -1) > 0:
            max_tokens = min(max_tokens, self.config.plugin.turn_max_new_tokens)
        if 'max_new_tokens' in kwargs:
            max_tokens = min(max_tokens, kwargs.pop('max_new_tokens'))

        if max_tokens < 10:
            raise ContextLengthExceededError(
                "Client context budget exhausted before the API call: "
                f"input_tokens={len(input_ids)}, max_total_tokens="
                f"{min(max_len, self.config.prompt_length + self.config.response_length)}"
            )
        messages = kwargs.get('messages') or decode_conversation(input_ids, self.tokenizer)[0]

        # Network resilience: distinguish business errors (don't retry) from
        # transient network/proxy errors (retry more aggressively). The
        # symptom we're guarding against on bcplus servers: HTTP proxy
        # randomly drops the connection (httpcore.ReadError /
        # openai.APIConnectionError). The old 5-retry × 2**i (max 16s)
        # schedule could not ride out a 30-60s proxy blip.
        max_attempts = getattr(self, "api_max_attempts", 10)
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    presence_penalty=self.presence_penalty,
                    stop=self.stop_tokens,
                )

                message = response.choices[0].message
                
                # DeepSeek-Reasoner 特殊处理：合并 reasoning_content 和 content
                # reasoning_content 包含推理过程，content 包含函数调用
                # ReAct 框架需要"推理在前，动作在后"的完整格式
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning = message.reasoning_content or ""
                    content = message.content or ""
                    # 如果两者都有，用换行符连接：推理过程 + 函数调用
                    if reasoning and content:
                        text = f"{reasoning}\n\n{content}"
                    else:
                        text = reasoning or content
                else:
                    # 其他模型直接使用 content
                    text = message.content or ""
                
                text_ids = self.tokenizer.encode(text, add_special_tokens=False)

                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": text_ids,
                            "response_log_probs": [0.0] * len(text_ids),
                            "extra_data": {"input_ids": input_ids},
                            "metrics": {"usage": {
                                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                                "completion_tokens": response.usage.completion_tokens if response.usage else len(text_ids),
                                "total_tokens": response.usage.total_tokens if response.usage else 0,
                            }}
                        }
                    }]
                }
            except Exception as e:
                error_str = str(e)
                # A context overflow is deterministic for this exact payload.
                # Retrying it unchanged only wastes time and API capacity.
                if is_context_length_exceeded_error(e):
                    print(
                        f"[CallAPI] Context length exceeded, skipping retries: "
                        f"{error_str[:200]}"
                    )
                    raise ContextLengthExceededError(error_str) from e
                # 内容审核（不重试）
                if "400" in error_str and ("Content Exists Risk" in error_str or "content_filter" in error_str.lower() or "sensitive" in error_str.lower()):
                    print(f"[CallAPI] Content moderation triggered, skipping retries: {error_str[:200]}")
                    return None
                # 透明区分：网络/代理错误 vs 其他
                etype = type(e).__name__
                is_network = (
                    "ConnectionError" in etype
                    or "ReadError" in etype
                    or "ReadTimeout" in etype
                    or "ConnectError" in etype
                    or "APIConnectionError" in etype
                    or "APITimeoutError" in etype
                    or "Connection error" in error_str
                    or "Read timed out" in error_str
                )
                if attempt == max_attempts - 1:
                    print(f"[CallAPI ERROR] Failed after {max_attempts} attempts ({etype}): {error_str[:200]}")
                    return None
                # 网络错误：长 backoff（capped 60s）+ jitter；其他：标准 backoff
                import random as _rand
                if is_network:
                    wait_time = min(60.0, 3.0 * (1.6 ** attempt)) + _rand.uniform(0, 2.0)
                else:
                    wait_time = min(16.0, 2.0 ** attempt)
                print(f"[CallAPI] Attempt {attempt + 1}/{max_attempts} failed ({etype}): "
                      f"{error_str[:120]}. Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        return None


def truncate_prompt(chat, prompt_length, tokenizer, prompt_turn):
    exceed_len = len(tokenizer.apply_chat_template(chat[:prompt_turn])) + 8 - prompt_length
    _cut_idx = 0
    while exceed_len > 0:  # truncate long user prompt
        print('[PROMPT] now exceed', exceed_len, 'work on cut turn', _cut_idx)
        chat[_cut_idx]['content'] = tokenizer.decode(
            tokenizer.encode(chat[_cut_idx]['content'], add_special_tokens=False)[
                exceed_len + 4:], add_special_tokens=False)
        exceed_len = len(tokenizer.apply_chat_template(chat[:prompt_turn])) + 8 - prompt_length
        _cut_idx = _cut_idx + 1
        if _cut_idx >= prompt_turn:
            break
    return chat


class AgentContext:
    # Manage context of an agent
    def __init__(self, chat, tokenizer, config, prompt_turn=2):
        self.tokenizer = tokenizer
        self.config = config
        self.init_len = len(chat)
        self.prompt_turn = prompt_turn
        self.prompt_length = config.prompt_length
        self.response_length = config.response_length
        self.context_uid = str(uuid.uuid4())

        self.chat = copy.deepcopy(chat)
        self.chat = truncate_prompt(self.chat, config.prompt_length, tokenizer, prompt_turn)
        self.chat_completions = [None for _ in range(len(self.chat))]
        self.chat_ids = [self.get_turn_context(i) for i in range(len(self.chat))]
        self.log_probs = [[0.0] * len(turn) for turn in self.chat_ids]
        self.token_mask = [[False] * len(turn) for turn in self.chat_ids]
        self.additional_info = [None for _ in self.chat_ids]
        self.generation_prompt = None
        self.metrics = None
        # Cumulative token usage across ALL step() calls (Agent.metrics only
        # holds the latest call's usage, which is useless for whole-session
        # cost reporting — see fold_agent / Agent.react() which do multiple
        # internal steps). Accumulator is updated inside append() whenever a
        # completion is attached.
        self.cumulative_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "call_count": 0,
        }
        # G4 — Per-turn context size trace. Each entry is (turn_idx,
        # context_token_len). Recorded inside append() whenever a new
        # message is added to the chat. Lets all baselines export the
        # same shape of context-growth curve for Figure 3a-revised
        # (cross-baseline context dynamics plot).
        self.context_size_history: list = []
        # Figure-1 dilution experiment — full per-turn message dump
        # (not truncated). Each entry: {turn_idx, role, content,
        # cumul_token_count}. Lets the offline KFD/KFSR pipeline
        # reconstruct the LLM's active context window at every turn
        # without re-running the agent.
        self.context_dump: list = []
        # Seed with the initial prompt messages (system + initial user
        # query) so the dump contains the full C_0. Without this seed,
        # append()'s dump misses the question text and KFD/KFSR for
        # facts whose evidence_source="question" would be zero.
        try:
            cumul = 0
            for i, msg in enumerate(self.chat):
                cumul += len(self.chat_ids[i]) if i < len(self.chat_ids) else 0
                self.context_dump.append({
                    "turn_idx": i,
                    "role": msg.get("role", "") if isinstance(msg, dict) else "",
                    "content": msg.get("content", "") if isinstance(msg, dict) else "",
                    "cumul_token_count": cumul,
                })
        except Exception:
            pass
        self.prompt_ids_len = len(sum(self.chat_ids[:prompt_turn], []))

    def _apply_chat_template_int_ids(self, msgs, add_generation_prompt: bool):
        """Run apply_chat_template and return a plain List[int].

        Why this exists: in transformers >= 5.x, `apply_chat_template(..., tokenize=True)`
        returns a `BatchEncoding` (tokenizers.Encoding wrapper) rather than the
        old-API `List[int]`. Slicing a BatchEncoding leaks `tokenizers.Encoding`
        objects into chat_ids, which later crashes `tokenizer.pad()` in
        `dataproto()` with "type of Encoding(...) unknown".

        Three paths covered (in order):
          1. return_dict=False → forces List[int] (transformers ≥ 4.42 supports this)
          2. result.input_ids → present on BatchEncoding, gives List[int]
          3. fallback: walk `.encodings[0].ids` for raw tokenizers.Encoding
        """
        try:
            r = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=add_generation_prompt,
                tokenize=True, return_dict=False,
            )
        except TypeError:
            # Older transformers don't accept return_dict — fall back.
            r = self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=add_generation_prompt, tokenize=True,
            )
        # Coerce whatever we got into List[int]
        if isinstance(r, list) and (not r or isinstance(r[0], int)):
            return r
        if hasattr(r, "input_ids"):
            ids = r.input_ids
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                return ids[0]   # batch size 1
            return list(ids)
        if hasattr(r, "ids"):  # tokenizers.Encoding
            return list(r.ids)
        return list(r)

    def get_turn_context(self, i):
        tokens = self._apply_chat_template_int_ids(self.chat[:i + 1], add_generation_prompt=False)
        prev = self._apply_chat_template_int_ids(self.chat[:i], add_generation_prompt=False) if i > 0 else []
        turn_tokens = tokens[len(prev):]
        return list(turn_tokens)

    def get_generation_prompt(self):
        if self.generation_prompt is None:
            tokens = self._apply_chat_template_int_ids(self.chat, add_generation_prompt=False)
            add_tokens = self._apply_chat_template_int_ids(self.chat, add_generation_prompt=True)
            self.generation_prompt = list(add_tokens[len(tokens):])
        return self.generation_prompt

    def messages(self):
        return self.chat

    def context_ids(self, messages=None):
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def context(self, turn_cut: int=None):
        if turn_cut is not None:
            return sum(self.chat_ids[:turn_cut], []) + self.get_generation_prompt()
        return sum(self.chat_ids, []) + self.get_generation_prompt()

    def append(self, turn, completion=None, additional_info=None):
        self.chat.append(turn)
        self.chat_completions.append(completion)
        self.additional_info.append(additional_info)
        if completion is None:
            self.chat_ids.append(self.get_turn_context(len(self.chat) - 1))
            self.log_probs.append([0.0] * len(self.chat_ids[-1]))
            self.token_mask.append([False] * len(self.chat_ids[-1]))
        else:
            completion_tokens = completion["choices"][0]["message"]["raw_output_ids"]
            completion_log_probs = completion["choices"][0]["message"]["response_log_probs"]
            self.chat_ids.append(self.get_generation_prompt() + completion_tokens)
            self.log_probs.append([0.0] * len(self.get_generation_prompt()) + completion_log_probs)
            self.token_mask.append([False] * len(self.get_generation_prompt()) + [True] * len(completion_tokens))
            if len(completion_tokens) == 0 or completion_tokens[-1] != self.tokenizer.eos_token_id:
                self.chat_ids[-1].append(self.tokenizer.eos_token_id)
                self.log_probs[-1].append(0.0)
                self.token_mask[-1].append(False)
            self.metrics = completion["choices"][0]["message"]['metrics']
            # Accumulate (the latest) usage into the cumulative counter so any
            # downstream caller can read agent.cumulative_usage instead of
            # having to intercept every step() return value.
            try:
                _u = self.metrics.get("usage", {}) if isinstance(self.metrics, dict) else {}
                self.cumulative_usage["prompt_tokens"] += int(_u.get("prompt_tokens", 0) or 0)
                self.cumulative_usage["completion_tokens"] += int(_u.get("completion_tokens", 0) or 0)
                self.cumulative_usage["total_tokens"] += int(_u.get("total_tokens", 0) or 0)
                self.cumulative_usage["call_count"] += 1
            except Exception:
                pass

        # G4 — record context size at this turn. Done OUTSIDE the
        # completion-attached branch so it fires for both assistant and
        # user appends. Uses len(self.context()) which is the actual
        # tokenized prompt length the LLM would see on the next step.
        try:
            _ctx_len = len(self.context())
            self.context_size_history.append((len(self.chat), _ctx_len))
            # Figure-1 dilution: full message dump (not truncated)
            self.context_dump.append({
                "turn_idx": len(self.chat),
                "role": turn.get("role", "") if isinstance(turn, dict) else "",
                "content": turn.get("content", "") if isinstance(turn, dict) else "",
                "cumul_token_count": _ctx_len,
            })
        except Exception:
            # context() can fail for edge cases (e.g. empty chat); ignore
            # so logging never blocks the agent.
            pass

    def rollback(self, k=1):
        self.chat = self.chat[:-k]
        self.chat_completions = self.chat_completions[:-k]
        self.chat_ids = self.chat_ids[:-k]
        self.log_probs = self.log_probs[:-k]
        self.token_mask = self.token_mask[:-k]
        self.additional_info = self.additional_info[:-k]

    def get_metrics(self):
        if self.metrics is None:
            return {}
        return self.metrics

    async def dataproto(self):
        # Create and mask prompt
        prompt_turn = self.prompt_turn
        prompt_length = self.prompt_length
        response_length = self.response_length

        prompt_ids = sum(self.chat_ids[:prompt_turn], [])
        if len(prompt_ids) > prompt_length:
            print('[PROMPT] prompt truncated in dataproto')
        with patch.object(self.tokenizer, "padding_side", "left"):
            prompt_output = self.tokenizer.pad(dict(input_ids=[prompt_ids[-prompt_length:]]),
                                               padding="max_length",
                                               max_length=prompt_length,
                                               return_tensors="pt")
            prompt_tensor = prompt_output["input_ids"][:, -prompt_length:].to(torch.int32)
            prompt_mask = prompt_output["attention_mask"][:, -prompt_length:].to(torch.int8)

        # Create and mask response
        response_outputs = sum(self.chat_ids[prompt_turn:], [])
        response_log_probs = sum(self.log_probs[prompt_turn:], [])
        model_output_mask = sum(self.token_mask[prompt_turn:], [])
        off_policy_steps = [-1] * len(response_log_probs)
        process_reward_mask = sum([[info.get('process_reward', 0) if isinstance(info, dict) else 0] * len(turn)
                                   for turn, info in zip(self.chat_ids, self.additional_info)][prompt_turn:], [])

        def _pad(cur_list, target_length, pad_token=-1):
            if len(cur_list) > target_length:
                padded_list = cur_list[:target_length]
            else:
                padded_list = cur_list + [pad_token] * (target_length - len(cur_list))
            return padded_list

        with patch.object(self.tokenizer, "padding_side", "right"):
            response_output = self.tokenizer.pad(dict(input_ids=[response_outputs[:response_length]]),
                                                 padding="max_length",
                                                 max_length=response_length,
                                                 return_tensors="pt")
            response_tensor = response_output["input_ids"][:, :response_length].to(torch.int32)
            response_mask = response_output["attention_mask"][:, :response_length].to(torch.int8)

        input_ids = torch.hstack((prompt_tensor, response_tensor))
        attention_mask = torch.hstack((prompt_mask, response_mask))
        response_log_probs = torch.Tensor([_pad(response_log_probs, response_length, pad_token=0)])
        off_policy_steps = torch.Tensor([_pad(off_policy_steps, response_length, pad_token=-1)])
        model_output_mask = torch.Tensor([_pad(model_output_mask, response_length, pad_token=False)])
        process_reward_mask = torch.Tensor([_pad(process_reward_mask, response_length, pad_token=0)])
        process_reward_mask = process_reward_mask * model_output_mask.int()

        batch = {
            'input_ids': input_ids.to(torch.int32),  # here input_ids become the whole sentences
            'attention_mask': attention_mask.to(torch.int8),
            'rollout_behavior_log_probs': response_log_probs.to(torch.bfloat16),
            'off_policy_steps': off_policy_steps.to(torch.int8),
            'model_output_mask': model_output_mask.to(torch.int8),
            'is_finished': torch.Tensor([True]).to(torch.int8),
            'process_reward_mask': process_reward_mask.to(torch.int8)
        }

        out = DataProto.from_dict(batch)
        return out


class Agent(AgentContext):
    # Agent utils
    def __init__(self, llm_client, conversations, tokenizer, config, prompt_turn=2):
        super().__init__(conversations, tokenizer, config, prompt_turn=prompt_turn)
        self.llm_client = llm_client
        self.retry_cjk = getattr(config.plugin, "retry_cjk", 0)
        self.info_cache = {}

    async def step(self, max_new_tokens=None, retry_cjk=0):
        prompt = self.context()
        max_len = self.prompt_ids_len + self.config.response_length
        if max_new_tokens is not None:
            max_len = min(len(prompt) + max_new_tokens, 131072)
        debug_log(f"[DEBUG Agent.step] Calling LLM with prompt length: {len(prompt)}, max_len: {max_len}")
        completion = await self.llm_client.create_completion(
            prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
        if completion is None:
            debug_log("[DEBUG Agent.step] ⚠️ Completion is None! LLM returned nothing.")
            return None
        debug_log(f"[DEBUG Agent.step] ✓ Got completion, content length: {len(completion['choices'][0]['message']['content'])}")
        if max(self.retry_cjk, retry_cjk):
            if is_weird(completion["choices"][0]["message"]["content"]):
                for _ in range(int(max(self.retry_cjk, retry_cjk))):
                    completion = await self.llm_client.create_completion(
                        prompt, uid=self.context_uid, max_len=max_len, messages=self.chat)
                    if is_weird(completion["choices"][0]["message"]["content"]):
                        continue
                    else:
                        break
        response = completion["choices"][0]["message"]["content"]
        debug_log(f"[DEBUG Agent.step] Response preview: {response[:200]}...")
        self.append({'role': 'assistant', 'content': response}, completion)
        return response

    async def react(self, run_action, max_turn=64, max_tokens=None, session_timeout=60 * 60,
                    should_continue=None, summary_prompt=None, safe_finish=None, observation_prompt=None):
        # Run react for max_turn turn
        if should_continue is None:
            should_continue = lambda st: True
        session_start_time = time.time()
        iteration = 0
        if max_tokens is not None:
            max_tokens = max_tokens - 512
        else:
            max_tokens = self.config.response_length - 512

        last_response = None
        response = None
        init_len = len(self.context(turn_cut=self.prompt_turn))
        while iteration < max_turn:
            if time.time() - session_start_time > session_timeout:  # TODO add session timeout
                print('[SESSION] Session Timeout')
                break
            if len(self.context()) - init_len > max_tokens:  # summary
                break

            iteration += 1
            response = await self.step()
            if response is None:
                break

            if not should_continue(response):
                last_response = response
                break
            if safe_finish is not None and safe_finish(response) is not None:
                observation = safe_finish(response)
            else:
                observation = await run_action(response)
            if observation is None:
                break
            if observation_prompt:
                observation += '\n' + observation_prompt
            self.append({'role': 'user', 'content': observation, })

        if last_response is None and summary_prompt is not None:
            if len(self.context()) - init_len > self.config.response_length - 1024:  # summary
                self.rollback(k=2)
            if self.chat[-1]['role'] == 'user':
                self.append({'role': 'assistant', 'content': "", })
            self.append({'role': 'user', 'content': summary_prompt, })
            last_response = await self.step(max_new_tokens=4096)
        elif last_response is None:
            last_response = str(response)

        return {'last_response': last_response, 'iteration': iteration}

    def set_process_reward(self, turn, reward):
        if isinstance(turn, str) and turn.lower() == 'all':
            turn = [i for i in range(len(self.chat))]
        if not isinstance(turn, list):
            turn = [turn]
        for i in turn:
            if i <= 0:
                continue
            if i > len(self.chat) - 1:
                continue
            if self.chat_completions[i] is None:
                continue
            if self.additional_info[i] is None:
                self.additional_info[i] = {}
            self.additional_info[i]['process_reward'] = reward

    def set_cache(self, key, value):
        self.info_cache[key] = value


@dataclass
class TaskContext:
    config: DictConfig
    global_step: int
    server_host: str
    server_port: int
    is_train: bool
    tokenizer: Optional[PreTrainedTokenizer] = None


async def run_action(env, response):
    debug_log(f"[DEBUG run_action] Input response: {response[:300]}...")
    try:
        try:
            act = time.time()
            debug_log(f"[DEBUG run_action] Calling env.run_action...")
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
            debug_log(f"[DEBUG run_action] env_return keys: {env_return.keys() if isinstance(env_return, dict) else type(env_return)}")
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            debug_log(f"[DEBUG run_action] Action detected: {action}")
            if action == 'finish':
                debug_log("[DEBUG run_action] ⚠️ Finish action detected, returning None")
                return None
        observation = env_return.pop('observation', 'Empty')
        debug_log(f"[DEBUG run_action] Observation: {observation[:200] if observation else 'Empty'}...")
    except Exception as e:
        import traceback
        debug_log(f"[DEBUG run_action] ⚠️ Exception: {e}")
        traceback.print_exc()
        observation = f"Error: {e}"
    
    # ===== 干扰信息注入 =====
    observation = _distractor_injector.inject(observation)
    
    return observation
