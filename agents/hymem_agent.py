"""
HyMem Agent - 混合记忆Agent框架（支持混合模型架构）

融合BrowseMaster的Planner-Executor架构与DeepAgent/FoldAgent的压缩折叠策略。
核心特性：
1. Planner-Executor信息隔离
2. 三种结构化Memory (Episode, Working, Tool)
3. 多触发机制的上下文折叠
4. 无训练的LLM压缩
5. 支持混合模型架构（不同组件使用不同模型）

"""

import os
import re
import time
import copy
import json
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from verl import DataProto

from .utils import (
    CallLLM, CallAPI, Agent, TaskContext, truncate_text,
    ContextLengthExceededError,
)
from .bm_tools import BMToolManager, BMToolsConfig
from .bm_sandbox import BMExecutorSandbox, create_executor_sandbox
from .hymem_memory import (
    HyMemStore, HyMemManager, format_memory_for_context,
    format_tool_memory_for_executor, extract_summary_tag,
    aggregate_tool_call_stats,
)
from .hymem_logger import HyMemEventLog, TokenUsage, LogLimits
from .hymem_prompts import (
    create_planner_prompt, create_executor_prompt, create_replan_prompt,
    format_executor_results_for_planner, format_planner_history_for_memory,
    format_recent_history, HYMEM_FOLD_INSTRUCTION,
    create_branch_prompt, format_branch_result_for_planner,
    extract_branch_info, extract_return, HYMEM_BRANCH_TIMEOUT_PROMPT,
    FOLD_SUMMARY_REQUEST, FOLD_SUMMARY_REQUEST_QWEN,
    create_planner_prompt_after_fold, extract_recent_buffer,
)
from .hymem_prompts_qwen import (
    create_planner_prompt_qwen, create_executor_prompt_qwen, create_replan_prompt_qwen,
    format_executor_results_for_planner_qwen, HYMEM_FOLD_INSTRUCTION_QWEN,
    create_branch_prompt_qwen, format_branch_result_for_planner_qwen,
    HYMEM_BRANCH_TIMEOUT_PROMPT_QWEN,
)
from .hymem_trigger import FoldTrigger, TriggerConfig, estimate_context_length, ComplexityEstimator
from .complexity_router import LLMComplexityRouter
from .exec_early_exit import detect_executor_diminishing


# ============== Model size inference (Step C1) ==============

def _infer_model_size(model_name: Optional[str]) -> str:
    """Infer rough model-size bucket from a model name string.

    Returns one of ``tiny`` (≤4B), ``small`` (5-14B), ``medium`` (15-50B),
    or ``large`` (>50B / unknown frontier). Used by hymem_agent to auto-pick
    the right prompt_style — smaller models need the simpler ``qwen`` prompts.
    """
    if not model_name:
        return "large"
    name = str(model_name).lower()
    # Frontier-class identifiers shortcut
    for kw in ("gpt-4", "gpt4", "deepseek-reasoner", "deepseek-v3",
               "claude-opus", "claude-sonnet", "o1", "gemini-2", "qwen-max"):
        if kw in name:
            return "large"
    # Parse trailing size token like "qwen3-7b", "llama-3-70b", "qwen2.5-14b-instruct"
    import re as _re
    m = _re.search(r"(\d+(?:\.\d+)?)\s*[bB](?=[^a-z]|$|-)", name)
    if not m:
        # Try "32b", "0.5b" without preceding non-digit
        m = _re.search(r"-(\d+(?:\.\d+)?)b", name)
    if m:
        try:
            n = float(m.group(1))
        except ValueError:
            return "large"
        if n <= 4:
            return "tiny"
        if n <= 14:
            return "small"
        if n <= 50:
            return "medium"
        return "large"
    return "large"


# ============== DEBUG 配置 ==============
# 设置为 True 开启详细调试日志
HYMEM_DEBUG = os.getenv("HYMEM_DEBUG", "false").lower() == "true"
# 设置调试日志保存路径（可选）
HYMEM_DEBUG_LOG_DIR = os.getenv("HYMEM_DEBUG_LOG_DIR", "/data1/test/xq/hymem/FoldAgent/scripts/debug")
# 是否打印完整响应（而不是截断）
HYMEM_DEBUG_FULL_RESPONSE = os.getenv("HYMEM_DEBUG_FULL_RESPONSE", "false").lower() == "true"

# 创建调试日志目录
if HYMEM_DEBUG and HYMEM_DEBUG_LOG_DIR:
    os.makedirs(HYMEM_DEBUG_LOG_DIR, exist_ok=True)

# 全局调试计数器
_debug_session_id = None
_debug_turn_count = 0

def init_debug_session():
    """初始化调试会话"""
    global _debug_session_id, _debug_turn_count
    from datetime import datetime
    _debug_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _debug_turn_count = 0
    if HYMEM_DEBUG:
        print(f"\n{'#'*70}")
        print(f"[HYMEM_DEBUG] Session ID: {_debug_session_id}")
        print(f"[HYMEM_DEBUG] Debug log dir: {HYMEM_DEBUG_LOG_DIR}")
        print(f"{'#'*70}\n")

def debug_log(msg: str, level: str = "INFO"):
    """输出调试日志"""
    if HYMEM_DEBUG:
        print(f"[HYMEM_DEBUG][{level}] {msg}")

def debug_save_to_file(filename: str, content: str):
    """保存调试信息到文件"""
    if HYMEM_DEBUG and HYMEM_DEBUG_LOG_DIR:
        try:
            filepath = os.path.join(HYMEM_DEBUG_LOG_DIR, f"{_debug_session_id}_{filename}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            debug_log(f"Saved debug log to: {filepath}")
        except Exception as e:
            debug_log(f"Failed to save debug log: {e}", "ERROR")

def debug_print_messages(messages: List[Dict], prefix: str = ""):
    """打印完整的 messages 列表"""
    if not HYMEM_DEBUG:
        return
    
    print(f"\n{'-'*60}")
    print(f"[HYMEM_DEBUG] {prefix} Messages ({len(messages)} total):")
    print(f"{'-'*60}")
    
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # 对于长内容，决定是否截断
        if HYMEM_DEBUG_FULL_RESPONSE:
            display_content = content
        else:
            display_content = content[:1000] + "..." if len(content) > 1000 else content
        
        print(f"\n[{i}] Role: {role}")
        print(f"    Length: {len(content)} chars")
        print(f"    Content:")
        print(f"    {'-'*40}")
        # 缩进打印
        for line in display_content.split('\n')[:50]:  # 最多50行
            print(f"    {line}")
        if len(display_content.split('\n')) > 50:
            print(f"    ... ({len(display_content.split(chr(10)))} lines total)")
        print(f"    {'-'*40}")
    
    print(f"{'-'*60}\n")

def debug_response(response: str, turn: int):
    """详细分析模型响应"""
    global _debug_turn_count
    _debug_turn_count = turn
    
    if not HYMEM_DEBUG:
        return
    
    print("\n" + "=" * 70)
    print(f"[HYMEM_DEBUG] ========== TURN {turn} MODEL RESPONSE ANALYSIS ==========")
    print("=" * 70)
    
    if response is None:
        print("[HYMEM_DEBUG] ❌ Response is None!")
        debug_save_to_file(f"turn_{turn}_response_NONE.txt", "Response was None")
        return
    
    print(f"[HYMEM_DEBUG] Response length: {len(response)} chars")
    
    # 根据设置决定是否打印完整响应
    if HYMEM_DEBUG_FULL_RESPONSE:
        print(f"[HYMEM_DEBUG] FULL Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
    else:
        print(f"[HYMEM_DEBUG] First 2000 chars:")
        print("-" * 50)
        print(response[:2000])
        if len(response) > 2000:
            print(f"\n... [truncated, {len(response) - 2000} more chars]")
        print("-" * 50)
    
    # 保存完整响应到文件
    debug_save_to_file(f"turn_{turn}_response.txt", response)
    
    # 检查各种标签
    tags_check = {
        '<task>': '<task>' in response,
        '</task>': '</task>' in response,
        '<answer>': '<answer>' in response,
        '</answer>': '</answer>' in response,
        '<confidence>': '<confidence>' in response,
        '</confidence>': '</confidence>' in response,
        '<results>': '<results>' in response,
        '</results>': '</results>' in response,
        '<code>': '<code>' in response,
        '</code>': '</code>' in response,
        '<think>': '<think>' in response,
        '</think>': '</think>' in response,
        '<branch>': '<branch>' in response,
        '</branch>': '</branch>' in response,
        '<return>': '<return>' in response,
        '</return>': '</return>' in response,
    }
    
    print("\n[HYMEM_DEBUG] Tag Detection:")
    found_any = False
    for tag, found in tags_check.items():
        if found:
            found_any = True
        status = "✓" if found else "✗"
        print(f"  {status} {tag}")
    
    # 尝试提取内容
    task = extract_task(response)
    answer = extract_answer(response)
    confidence = extract_confidence(response)
    branch_info = extract_branch_info(response)
    
    print("\n[HYMEM_DEBUG] Extracted Content:")
    print(f"  Task: {task[:500] + '...' if task and len(task) > 500 else task}")
    print(f"  Answer: {answer[:500] + '...' if answer and len(answer) > 500 else answer}")
    print(f"  Confidence: {confidence}")
    print(f"  Branch: {branch_info}")
    
    # 诊断问题
    print("\n[HYMEM_DEBUG] ========== DIAGNOSIS ==========")
    issues = []
    
    if not found_any:
        issues.append("❌ CRITICAL: No HyMem XML tags found at all!")
        issues.append("   The model is NOT following the expected output format.")
    
    if not task and not answer and not branch_info:
        issues.append("⚠️ WARNING: No <task>, <answer>, or <branch> tags found!")
        issues.append("   The agent will not know what action to take.")
        
        # 检查是否有其他格式
        if "```" in response:
            issues.append("   → Found markdown code blocks (```) - model may be using wrong format")
        if "search" in response.lower() or "查" in response or "搜索" in response:
            issues.append("   → Response mentions search but no <task> tag")
        if "answer" in response.lower() or "答案" in response or "回答" in response:
            issues.append("   → Response mentions answer but no <answer> tag")
        
        # 检查模型是否输出了其他格式
        if "<function=" in response:
            issues.append("   → Model used <function=...> format (ReAct style, not HyMem)")
        if "Action:" in response or "Observation:" in response:
            issues.append("   → Model used Action/Observation format (ReAct style)")
    else:
        if task:
            print("  ✓ Task extraction successful")
        if answer:
            print("  ✓ Answer extraction successful")
        if branch_info:
            print("  ✓ Branch extraction successful")
    
    if issues:
        print("\n  " + "\n  ".join(issues))
        # 保存诊断信息
        debug_save_to_file(f"turn_{turn}_diagnosis.txt", "\n".join(issues) + f"\n\nFull response:\n{response}")
    else:
        print("  ✓ Response format looks correct")
    
    print("=" * 70 + "\n")


def extract_task(text: str) -> Optional[str]:
    matches = re.findall(r'<task>(.*?)</task>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


# ========== Parallel-aware extractors (Optimization 1) ==========

def extract_all_tasks(text: str) -> List[str]:
    """Return EVERY <task>...</task> block in order so a single planner turn
    can fan out multiple independent searches concurrently. Backwards
    compatible — when the planner emits one task, this returns a 1-element
    list, which then runs through the normal serial path semantically."""
    return [m.strip() for m in re.findall(r'<task>(.*?)</task>', text, re.DOTALL) if m.strip()]


def extract_all_branches(text: str) -> List[Dict[str, str]]:
    """Return every <branch>...</branch> block parsed into {description, prompt}.
    Multiple branches in one turn are fanned out in parallel."""
    blocks = re.findall(r'<branch>(.*?)</branch>', text, re.DOTALL)
    out: List[Dict[str, str]] = []
    for content in blocks:
        desc_m = re.search(r'<description>(.*?)</description>', content, re.DOTALL)
        prompt_m = re.search(r'<prompt>(.*?)</prompt>', content, re.DOTALL)
        if desc_m and prompt_m:
            out.append({
                'description': desc_m.group(1).strip(),
                'prompt': prompt_m.group(1).strip(),
            })
    return out


def extract_answer(text: str) -> Optional[str]:
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_confidence(text: str) -> float:
    matches = re.findall(r'<confidence>(.*?)</confidence>', text, re.DOTALL)
    if matches:
        try:
            conf = float(matches[-1].strip())
            return min(max(conf, 0.0), 1.0)
        except:
            pass
    return 0.5


def extract_results(text: str) -> Optional[str]:
    matches = re.findall(r'<results>(.*?)</results>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_code_blocks(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r'<code>(.*?)</code>', text, re.DOTALL)]


class HyMemAgent:
    """
    HyMem Agent主类 - 支持混合模型架构
    
    支持为不同组件配置不同的模型和 API 端点：
    - Planner: 通过 planner_client 配置（推荐使用强推理模型）
    - Executor: 通过 executor_client 配置（可使用本地模型）
    - Memory Manager: 通过 aux_model/aux_base_url/aux_api_key 配置（可使用本地模型）
    """
    
    def __init__(
        self,
        planner_client,
        executor_client,
        tool_manager: BMToolManager,
        tokenizer,
        config,
        aux_model: str = None,
        aux_base_url: str = None,
        aux_api_key: str = None,
        max_planner_turns: int = 20,
        max_executor_turns: int = 10,
        max_branch_turns: int = 8,
        prompt_style: str = "default",  # "default" 或 "qwen"
        # ---- Ablation switches (paper ablation table) ----
        # disable_isolated_reasoning: turn off the Branch mechanism. Explicit
        #   <branch> blocks are demoted to plain Executor searches and the
        #   complexity router never promotes a <task> to a branch.
        # disable_memory_management: turn off the fold + structured
        #   episode/working/tool memory. should_fold never fires, so the memory
        #   store stays empty and planner context accumulates unmanaged.
        disable_isolated_reasoning: bool = False,
        disable_memory_management: bool = False,
    ):
        self.planner_client = planner_client
        self.executor_client = executor_client
        self.tool_manager = tool_manager
        self.executor_sandbox = create_executor_sandbox(tool_manager, timeout=180)
        self.tokenizer = tokenizer
        self.config = config
        self.max_planner_turns = max_planner_turns
        self.max_executor_turns = max_executor_turns
        self.max_branch_turns = max_branch_turns
        self.prompt_style = prompt_style  # "default" 或 "qwen"
        self.disable_isolated_reasoning = disable_isolated_reasoning
        self.disable_memory_management = disable_memory_management
        if disable_isolated_reasoning:
            print("[HyMem][ABLATION] Isolated Reasoning (Branch) DISABLED")
        if disable_memory_management:
            print("[HyMem][ABLATION] Memory Management (fold + structured memory) DISABLED")
        
        # 根据 prompt_style 选择对应的函数
        if prompt_style == "qwen":
            self._create_planner_prompt = create_planner_prompt_qwen
            self._create_executor_prompt = create_executor_prompt_qwen
            self._create_replan_prompt = create_replan_prompt_qwen
            self._create_branch_prompt = create_branch_prompt_qwen
            self._format_executor_results = format_executor_results_for_planner_qwen
            self._format_branch_result = format_branch_result_for_planner_qwen
            self._fold_instruction = HYMEM_FOLD_INSTRUCTION_QWEN
            self._branch_timeout_prompt = HYMEM_BRANCH_TIMEOUT_PROMPT_QWEN
            print(f"[HyMem] Using Qwen3 optimized prompts")
        else:
            self._create_planner_prompt = create_planner_prompt
            self._create_executor_prompt = create_executor_prompt
            self._create_replan_prompt = create_replan_prompt
            self._create_branch_prompt = create_branch_prompt
            self._format_executor_results = format_executor_results_for_planner
            self._format_branch_result = format_branch_result_for_planner
            self._fold_instruction = HYMEM_FOLD_INSTRUCTION
            self._branch_timeout_prompt = HYMEM_BRANCH_TIMEOUT_PROMPT
        
        self.memory_store = HyMemStore()
        # Memory Manager 支持独立配置
        self.memory_manager = HyMemManager(
            aux_model=aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key
        )
        
        trigger_config = TriggerConfig(
            planner_context_threshold=0.80,
            max_folds=getattr(config.plugin, "hymem_max_folds", 3) if hasattr(config, 'plugin') else 3,
            min_turns_between_folds=5,
            low_confidence_threshold=0.5,
            consecutive_low_confidence_trigger=2,
            max_branches=getattr(config.plugin, "hymem_max_branches", 5) if hasattr(config, 'plugin') else 5,
            max_branch_turns=max_branch_turns
        )
        self.fold_trigger = FoldTrigger(trigger_config)

        # Step A1: LLM-driven complexity router (paper Sec 3.2.1).
        # Replaces the previous keyword-only heuristic. Uses the auxiliary
        # model and falls back to ComplexityEstimator on LLM failure /
        # low-confidence verdict. No new env vars introduced.
        self.complexity_router = LLMComplexityRouter(
            aux_model=aux_model,
            base_url=aux_base_url,
            api_key=aux_api_key,
            fallback_estimator=ComplexityEstimator.estimate,
            confidence_threshold=0.6,
        )
        
        self.confidence_threshold = 0.7
        self.max_replan_attempts = 2
        self.current_replan_count = 0
        self.max_context_length = getattr(config, 'response_length', 32000) if config else 32000
        
        self.planner_messages = []
        self.executor_results_history = []
        self.branch_results_history = []
        self.full_history = []

        # G1: per-fold-event token snapshots, structurally analogous to
        # ReSum's resum_summary_events. Each entry =
        #   {round, tokens_before, tokens_after_reset, trigger_reason}
        # populated by _fold_planner_context. Used by §A context-management
        # comparison to quantify "how much context HyMem fold reset vs ReSum
        # summary reset" head-to-head.
        self.fold_events: List[Dict[str, Any]] = []

        # O5/O6/O8 measurement counters. Surfaced into env.stats at session
        # end so paper efficiency tables can quantify each optimization.
        self._executor_early_exit_count: int = 0
        self._executor_early_exit_turns_saved: int = 0
        self._executor_inner_turns_used: List[int] = []
        self._fold_wall_seconds: List[float] = []
        self._parallel_dispatch_fanouts: List[int] = []
        # Figure-1 dilution: per-planner-turn full context snapshot.
        # Each entry: {turn, context_length, messages: [{role, content}, ...]}
        # Captured BEFORE _call_llm in the planner loop. Survives fold (fold
        # rebuilds self.planner_messages but does not touch this list).
        self._planner_context_snapshots: List[Dict[str, Any]] = []

        self.event_log = HyMemEventLog()
    
    async def _call_llm(self, client, messages: List[Dict], max_tokens: int = 4096, caller: str = "unknown") -> Optional[str]:
        # Step B2 — wrap LLM call with wall-clock timing for per-component
        # latency reporting (Bh2U W2 efficiency metrics).
        _t0 = time.time()
        try:
            if HYMEM_DEBUG:
                print(f"\n{'='*60}")
                debug_log(f"[{caller}] Calling LLM, {len(messages)} messages, max_tokens={max_tokens}")
                input_dump = json.dumps(messages, ensure_ascii=False, indent=2)
                debug_save_to_file(f"turn_{_debug_turn_count}_{caller}_input.json", input_dump)

            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )

            completion = await client.create_completion(
                input_ids, messages=messages, max_new_tokens=max_tokens
            )
            if completion is None:
                self.event_log.latency.add(caller, time.time() - _t0)
                debug_log(f"[{caller}] LLM returned None completion!", "WARNING")
                return None
            
            msg_data = completion["choices"][0]["message"]
            content = msg_data["content"]

            usage = msg_data.get("metrics", {}).get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(input_ids))
            completion_tokens = usage.get("completion_tokens", len(self.tokenizer.encode(content)) if content else 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            self.event_log.log_llm_call(
                caller=caller,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                input_length=sum(len(m.get("content", "")) for m in messages),
                output_length=len(content) if content else 0,
            )
            
            if HYMEM_DEBUG:
                debug_log(f"[{caller}] tokens: prompt={prompt_tokens}, completion={completion_tokens}")
                if content:
                    debug_save_to_file(f"turn_{_debug_turn_count}_{caller}_output.txt", content)

            self.event_log.latency.add(caller, time.time() - _t0)
            return content
        except ContextLengthExceededError as e:
            self.event_log.latency.add(caller, time.time() - _t0)
            print(f"[HyMem] Non-retryable context error ({caller}): {e}")
            raise
        except Exception as e:
            self.event_log.latency.add(caller, time.time() - _t0)
            print(f"[HyMem] LLM call error ({caller}): {e}")
            if HYMEM_DEBUG:
                debug_log(f"[{caller}] Exception: {str(e)}", "ERROR")
                import traceback
                traceback.print_exc()
            return None
    
    async def _fold_planner_context(self, problem: str, trigger_reason: str = ""):
        """两阶段 fold: Planner 自压缩 → 辅助模型结构化提取

        Stage 1: 向当前 Planner（拥有完整上下文）追加 summary 请求，
                 让 Planner 自己生成高质量状态总结。
        Stage 2: 将 Planner summary 交给辅助模型提取结构化 Memory JSON。

        Args:
            trigger_reason: Why fold fired (e.g. "context_overflow",
                "executor_calls_threshold"). Recorded into self.fold_events
                for §A paired analysis with ReSum.
        """
        fold_num = self.memory_store.fold_count + 1
        print(f"[HyMem] Folding planner context (fold #{fold_num})")

        # O6 measurement: total fold wall-clock (Stage1 self-summarize +
        # Stage2 aux extract). Pair with memory_manager.fold_aux_durations
        # to see how much of fold time is the parallel-able portion.
        _fold_t0 = time.time()

        ctx_before = estimate_context_length(self.planner_messages, self.tokenizer)
        msg_before = len(self.planner_messages)
        self.event_log.log_fold_start(
            turn=len(self.full_history) + 1,
            context_tokens_before=ctx_before,
            message_count_before=msg_before,
        )

        # --- Stage 1: Planner 在完整上下文中自压缩 ---
        summary_request = (FOLD_SUMMARY_REQUEST_QWEN
                           if self.prompt_style == "qwen"
                           else FOLD_SUMMARY_REQUEST)
        self.planner_messages.append({
            "role": "user", "content": summary_request
        })
        summary_response = await self._call_llm(
            self.planner_client, self.planner_messages,
            max_tokens=4096, caller="planner_fold_summary",
        )

        planner_summary = ""
        if summary_response:
            planner_summary = extract_summary_tag(summary_response) or summary_response
        if not planner_summary:
            print("[HyMem] Warning: Planner summary generation failed, using legacy fallback")
            planner_summary = format_planner_history_for_memory(self.planner_messages)

        recent_buffer = extract_recent_buffer(self.planner_messages, n_turns=2)

        # --- Stage 2: 辅助模型从 summary 中提取结构化 Memory ---
        tool_stats = aggregate_tool_call_stats(self.memory_store.tool_call_history)
        branch_insights = self.memory_store.get_branch_insights(
            max_items=5, max_result_len=500
        )

        episode_memory, working_memory, tool_memory = \
            await self.memory_manager.generate_memories_from_summary(
                question=problem,
                planner_summary=planner_summary,
                tool_call_stats=tool_stats,
                branch_insights=branch_insights,
            )

        for call in self.memory_manager.token_usage.calls:
            if call.get("timestamp", 0) > self.event_log.session_start_time:
                self.event_log.log_llm_call(
                    caller=call["caller"],
                    prompt_tokens=call["prompt_tokens"],
                    completion_tokens=call["completion_tokens"],
                    total_tokens=call["total_tokens"],
                )

        if episode_memory is not None:
            # G2: route through setter so op_counts['episode_writes'] increments
            self.memory_store.set_episode_memory(episode_memory)
        else:
            print("[HyMem] Warning: Episode memory extraction failed, keeping previous")

        if working_memory is not None:
            self.memory_store.set_working_memory(working_memory)
        else:
            print("[HyMem] Warning: Working memory extraction failed, keeping previous")

        if tool_memory:
            merged = self.memory_manager.merge_tool_memories(
                self.memory_store.tool_memory, tool_memory
            )
            self.memory_store.set_tool_memory(merged)

        self.memory_store.planner_summary = planner_summary
        self.memory_store.record_fold(len(self.full_history))

        # --- 重建 context: summary + structured memory + 近期缓冲 ---
        system_prompt = self.planner_messages[0].get("content", "") if self.planner_messages else ""
        self.planner_messages = create_planner_prompt_after_fold(
            problem=problem,
            planner_summary=planner_summary,
            system_prompt=system_prompt,
            memory_store=self.memory_store,
            recent_buffer=recent_buffer,
            prompt_style=self.prompt_style,
        )
        self.fold_trigger.reset()
        self.fold_trigger.reset_branches()

        ctx_after = estimate_context_length(self.planner_messages, self.tokenizer)
        self.event_log.log_fold_end(
            turn=len(self.full_history) + 1,
            context_tokens_after=ctx_after,
            message_count_after=len(self.planner_messages),
            has_episode=bool(episode_memory),
            has_working=bool(working_memory),
            has_tool=bool(tool_memory),
        )

        # G1: record fold event for paired comparison with ReSum's
        # resum_summary_events. Same field shape on purpose so analysis
        # scripts iterate uniformly.
        # O6: record total fold wall-clock for efficiency reporting.
        self._fold_wall_seconds.append(time.time() - _fold_t0)

        self.fold_events.append({
            "round": len(self.full_history) + 1,
            "tokens_before": ctx_before,
            "tokens_after_reset": ctx_after,
            "trigger_reason": trigger_reason or "unknown",
        })

        self.full_history.append({
            "role": "fold",
            "planner_summary": planner_summary[:2000],
            "episode_memory": episode_memory,
            "working_memory": working_memory,
            "tool_memory": tool_memory,
            "turn": len(self.full_history) + 1,
        })

        print(
            f"[HyMem] Fold complete. Summary: {len(planner_summary)} chars, "
            f"Episode: {bool(episode_memory)}, "
            f"Working: {bool(working_memory)}, "
            f"Tool: {bool(tool_memory)}"
        )
    
    async def run_executor(self, search_target: str, parent_turn: int = 0, parent: str = "planner") -> str:
        """运行Executor执行搜索任务"""
        messages = self._create_executor_prompt(search_target, self.memory_store.tool_memory)
        executor_tool_calls = []

        self.event_log.log_executor_start(turn=parent_turn, task=search_target, parent=parent)

        if HYMEM_DEBUG:
            debug_log(f"[EXECUTOR] Starting: {search_target[:200]}")
            debug_log(f"[EXECUTOR] Max turns: {self.max_executor_turns}")

        # === O5: diminishing-returns early exit ===
        # Fires from turn ≥ 2 if EITHER:
        #   (a) two consecutive observation outputs hash-match on first 200 chars (loop)
        #   (b) two consecutive observations are < 200 chars AND lack digits/named entities (low-info)
        #   (c) two consecutive observations both contain "Error" (repeated failure)
        # On trigger: append synthesis prompt and break to the existing summary path
        # at end of loop. Preserves quality on fast-converging tasks (no-op for them).
        early_exit_enabled = os.getenv("HYMEM_EXEC_EARLY_EXIT", "1") != "0"
        recent_obs: List[str] = []  # rolling window of last 2 observations
        _detect_diminishing = detect_executor_diminishing

        for turn in range(self.max_executor_turns):
            if HYMEM_DEBUG:
                debug_log(f"[EXECUTOR] Turn {turn + 1}/{self.max_executor_turns}")

            response = await self._call_llm(self.executor_client, messages, caller=f"executor_t{turn+1}")

            if HYMEM_DEBUG:
                debug_log(f"[EXECUTOR T{turn+1}] len={len(response) if response else 0}")

            if response is None:
                self.event_log.log_executor_end(turn=parent_turn, result="API call failed", total_executor_turns=turn)
                return "[Executor error: API call failed - possibly due to content moderation or network issues]"

            # 提取results - 但验证内容不是代码
            results = extract_results(response)
            if results:
                # 检查results是否包含未执行的代码（错误格式）
                if '<code>' in results or 'await ' in results[:100]:
                    # results中包含代码，需要先执行
                    code_output = await self.executor_sandbox.execute_from_response(f"<code>{results}</code>")
                    if code_output and "[No code blocks found" not in code_output:
                        results = code_output
                
                self.executor_results_history.append(results[:500])
                executor_tool_calls.append({'tool': 'results', 'output': results[:300], 'turn': turn + 1})
                self.event_log.log_executor_step(turn=parent_turn, executor_turn=turn + 1,
                                                 step_type="results", content=search_target, output=results)
                final_result = f"<results>\n{results}\n</results>"
                self.event_log.log_executor_end(turn=parent_turn, result=final_result, total_executor_turns=turn + 1)
                return final_result
            
            # 执行代码
            code_blocks = extract_code_blocks(response)
            if code_blocks:
                code_output = await self.executor_sandbox.execute_from_response(response)
                
                for i, code in enumerate(code_blocks):
                    executor_tool_calls.append({
                        'tool': 'code_execution', 'code': code[:200],
                        'output': code_output[:300] if code_output else 'No output', 'turn': turn + 1
                    })
                self.event_log.log_executor_step(turn=parent_turn, executor_turn=turn + 1,
                                                 step_type="code_execution",
                                                 content=code_blocks[0][:300],
                                                 output=code_output[:500] if code_output else "")
                
                if "[No code blocks found" in code_output:
                    observation = "No code executed. Write code within <code></code> tags, or provide results in <results></results>."
                else:
                    if len(code_output) > 3000:
                        code_output = await self.memory_manager.compress_tool_output(
                            code_output, search_target, max_length=2000
                        )
                    observation = f"Code output:\n{code_output}"
                    
                    if "Error" in code_output:
                        self.fold_trigger.consecutive_failures += 1
                    else:
                        self.fold_trigger.consecutive_failures = 0
            else:
                observation = "No code blocks found. Write Python code within <code></code> tags to search, or provide final results in <results></results>."

            messages.append({'role': 'assistant', 'content': response})
            messages.append({'role': 'user', 'content': observation})

            # === O5 early-exit check ===
            if early_exit_enabled and turn >= 1:  # need 2 turns to compare
                recent_obs.append(observation)
                if len(recent_obs) > 2:
                    recent_obs.pop(0)
                exit_reason = _detect_diminishing(recent_obs)
                if exit_reason is not None:
                    if HYMEM_DEBUG:
                        debug_log(f"[EXECUTOR] Early-exit triggered ({exit_reason}) at turn {turn+1}")
                    # Track for stats
                    self._executor_early_exit_count = getattr(
                        self, '_executor_early_exit_count', 0) + 1
                    self._executor_early_exit_turns_saved = (
                        getattr(self, '_executor_early_exit_turns_saved', 0)
                        + (self.max_executor_turns - turn - 1)
                    )
                    self._executor_inner_turns_used = (
                        getattr(self, '_executor_inner_turns_used', []) + [turn + 1]
                    )
                    # Break to summary path below by emitting a special prompt
                    # before exiting the for-loop
                    summary_prompt_early = (
                        f"[Auto-detected diminishing returns: {exit_reason}.] "
                        f"Synthesize all findings so far into <results></results>."
                    )
                    messages.append({'role': 'user', 'content': summary_prompt_early})
                    response = await self._call_llm(self.executor_client, messages,
                                                    caller=f"executor_early_exit_t{turn+1}")
                    if response:
                        results = extract_results(response)
                        if results:
                            self.executor_results_history.append(results[:500])
                            final_result = f"<results>\n{results}\n</results>"
                            self.event_log.log_executor_end(
                                turn=parent_turn, result=final_result,
                                total_executor_turns=turn + 1,
                            )
                            return final_result
                        # Fallback: wrap the response
                        clean = response.replace('<code>', '').replace('</code>', '')
                        final_result = f"<results>\n{clean[:2000]}\n</results>"
                        self.event_log.log_executor_end(
                            turn=parent_turn, result=final_result,
                            total_executor_turns=turn + 1,
                        )
                        return final_result
                    # If synthesis failed, fall through to the standard
                    # max-turns summary path below.
                    break

        # Normal completion → record turns used
        self._executor_inner_turns_used = (
            getattr(self, '_executor_inner_turns_used', [])
            + [self.max_executor_turns]
        )

        # 超时处理
        summary_prompt = "Maximum turns reached. Summarize all findings in <results></results> tags."
        messages.append({'role': 'user', 'content': summary_prompt})
        
        if HYMEM_DEBUG:
            debug_log("[EXECUTOR] Timeout reached, requesting summary")
        
        response = await self._call_llm(self.executor_client, messages, caller="executor_timeout")
        if response:
            results = extract_results(response)
            if results:
                self.executor_results_history.append(results[:500])
                final_result = f"<results>\n{results}\n</results>"
                self.event_log.log_executor_end(turn=parent_turn, result=final_result, total_executor_turns=self.max_executor_turns + 1)
                return final_result
            clean_response = response.replace('<code>', '').replace('</code>', '')
            final_result = f"<results>\n{clean_response[:2000]}\n</results>"
            self.event_log.log_executor_end(turn=parent_turn, result=final_result, total_executor_turns=self.max_executor_turns + 1)
            return final_result
        
        if executor_tool_calls:
            summary = "Executor findings:\n"
            for tc in executor_tool_calls[-3:]:
                summary += f"- {tc.get('tool', 'unknown')}: {tc.get('output', '')[:200]}\n"
            final_result = f"<results>\n{summary}\n</results>"
            self.event_log.log_executor_end(turn=parent_turn, result=final_result, total_executor_turns=self.max_executor_turns)
            return final_result
        
        self.event_log.log_executor_end(turn=parent_turn, result="Failed", total_executor_turns=self.max_executor_turns)
        return "[Executor: Failed to produce results]"
    
    async def run_branch(self, description: str, prompt: str, parent_turn: int = 0) -> Tuple[str, bool]:
        """
        运行Branch推理子任务

        Branch继承Planner上下文和Memory，独立执行推理任务。
        Branch可以调用Executor执行搜索，但不能创建新的Branch。
        """
        if not self.fold_trigger.can_branch():
            return f"[Branch limit reached ({self.fold_trigger.branch_count}/{self.fold_trigger.config.max_branches})]", False

        print(f"[HyMem] Starting branch: {description}")
        self.event_log.log_branch_start(turn=parent_turn, description=description, prompt=prompt)
        # Step A3 — attribute all tool calls made during this branch to the
        # 'branch' source so reviewers can see how many atomic calls happen
        # inside isolated reasoning vs in the outer Actor layer.
        _prev_source = self.tool_manager.get_source()
        self.tool_manager.set_source("branch")
        try:
            return await self._run_branch_inner(description, prompt, parent_turn)
        finally:
            self.tool_manager.set_source(_prev_source)

    async def _run_branch_inner(self, description: str, prompt: str, parent_turn: int = 0) -> Tuple[str, bool]:
        
        branch_messages = self._create_branch_prompt(
            planner_messages=self.planner_messages,
            description=description,
            prompt=prompt,
            memory_store=self.memory_store
        )
        
        branch_history = []
        
        for turn in range(self.max_branch_turns):
            context_length = estimate_context_length(branch_messages, self.tokenizer)
            
            should_end, end_reason = self.fold_trigger.should_end_branch(
                context_length=context_length,
                max_length=self.max_context_length,
                branch_turn=turn
            )
            
            if should_end and turn > 0:
                print(f"[HyMem] Branch force end: {end_reason}")
                branch_messages.append({'role': 'user', 'content': self._branch_timeout_prompt})
                response = await self._call_llm(self.planner_client, branch_messages, caller="branch_timeout")
                if response:
                    return_content = extract_return(response)
                    # Bug fix: only mark success when an explicit <return>...</return>
                    # block was emitted. If we had to fall back to the first 1000
                    # chars of a still-unfinished reply, that's an incomplete branch
                    # and must NOT inflate branch_success.
                    if return_content:
                        result = return_content
                        success = True
                    else:
                        result = f"[Branch timeout — partial reply] {response[:1000]}"
                        success = False
                    self.fold_trigger.record_branch(description, success, result)
                    self.memory_store.add_branch_result(description, prompt, result, success)
                    self.branch_results_history.append(result[:500])
                    self.event_log.log_branch_end(turn=parent_turn, description=description, result=result, success=success)
                    return result, success
                break
            
            response = await self._call_llm(self.planner_client, branch_messages, caller=f"branch_t{turn+1}")
            
            if HYMEM_DEBUG:
                debug_log(f"[BRANCH T{turn+1}] Response length: {len(response) if response else 0}")
                if response:
                    debug_log(f"[BRANCH T{turn+1}] Has <return>: {'<return>' in response}")
                    debug_log(f"[BRANCH T{turn+1}] Has <task>: {'<task>' in response}")
            
            if response is None:
                print(f"[HyMem] Branch error: No response at turn {turn + 1}")
                break
            
            branch_history.append({
                'role': 'branch_assistant',
                'content': response,
                'turn': turn + 1
            })
            self.event_log.log_branch_step(turn=parent_turn, branch_turn=turn + 1,
                                           step_type="reasoning", content=response)
            
            return_content = extract_return(response)
            if return_content:
                print(f"[HyMem] Branch completed at turn {turn + 1}")
                self.fold_trigger.record_branch(description, True, return_content)
                self.memory_store.add_branch_result(description, prompt, return_content, True)
                self.branch_results_history.append(return_content[:500])
                self.event_log.log_branch_end(turn=parent_turn, description=description, result=return_content, success=True)
                return return_content, True
            
            task = extract_task(response)
            if task:
                print(f"[HyMem] Branch executing search: {task[:50]}...")               
                executor_result = await self.run_executor(task, parent_turn=parent_turn, parent="branch")
                
                self.memory_store.add_tool_call(
                    tool_name="branch_executor",
                    arguments={"branch": description, "task": task},
                    response=executor_result[:2000],
                    success="Error" not in executor_result
                )
                
                branch_history.append({
                    'role': 'branch_executor',
                    'task': task,
                    'result': executor_result[:500],
                    'turn': turn + 1
                })
                self.event_log.log_branch_step(turn=parent_turn, branch_turn=turn + 1,
                                               step_type="executor", content=task, output=executor_result)
                
                observation = self._format_executor_results(executor_result, task)
                branch_messages.append({'role': 'assistant', 'content': response})
                branch_messages.append({'role': 'user', 'content': observation})
            else:
                branch_messages.append({'role': 'assistant', 'content': response})
                branch_messages.append({
                    'role': 'user', 
                    'content': "Continue your analysis. Use <task>...</task> to search for information, or <return>...</return> to report your findings."
                })
        
        if branch_history:
            last_entry = branch_history[-1]
            if last_entry.get('role') == 'branch_executor':
                last_response = f"Task: {last_entry.get('task', '')}\nResult: {last_entry.get('result', '')}"
            else:
                last_response = last_entry.get('content', '')
        else:
            last_response = ""
        partial_result = f"[Branch incomplete] Last response:\n{last_response[:500]}"
        self.fold_trigger.record_branch(description, False, partial_result)
        self.event_log.log_branch_end(turn=parent_turn, description=description, result=partial_result, success=False)
        return partial_result, False
    
    async def run(self, problem: str) -> Tuple[str, float, List[Dict]]:
        """运行完整的HyMem流程"""
        init_debug_session()
        
        self.event_log = HyMemEventLog()
        self.event_log.log_session_start(problem, config={
            "max_planner_turns": self.max_planner_turns,
            "max_executor_turns": self.max_executor_turns,
            "max_branch_turns": self.max_branch_turns,
            "max_context_length": self.max_context_length,
            "prompt_style": self.prompt_style,
        })
        
        self.planner_messages = self._create_planner_prompt(problem, self.memory_store)
        self.full_history = []
        self.current_replan_count = 0
        
        if HYMEM_DEBUG:
            debug_log(f"Session start: problem={len(problem)} chars, max_turns={self.max_planner_turns}")
            prompt_dump = json.dumps(self.planner_messages, ensure_ascii=False, indent=2)
            debug_save_to_file("initial_prompt.json", prompt_dump)
        
        final_answer = ""
        confidence = 0.0
        empty_response_count = 0
        max_empty_retries = 2
        
        for turn in range(self.max_planner_turns):
            print(f"[HyMem] Planner turn {turn + 1}/{self.max_planner_turns}")
            
            context_length = estimate_context_length(self.planner_messages, self.tokenizer)

            # Figure-1 dilution: snapshot the FULL planner context BEFORE the
            # LLM call so the offline KFD/KFSR pipeline can reconstruct what
            # the planner actually saw at this turn. Shallow-copy the message
            # list (entries are never mutated, only appended).
            self._planner_context_snapshots.append({
                "turn": turn + 1,
                "context_length": context_length,
                "messages": [{"role": m.get("role", ""),
                              "content": m.get("content", "")}
                             for m in self.planner_messages],
            })

            if HYMEM_DEBUG:
                debug_log(f"Turn {turn + 1}: ctx={context_length}, msgs={len(self.planner_messages)}")

            response = await self._call_llm(self.planner_client, self.planner_messages, caller=f"planner_t{turn+1}")
            
            if HYMEM_DEBUG:
                debug_response(response, turn + 1)
            
            # 处理空响应：区分API失败和短响应
            if response is None:
                # API调用失败（网络问题等），直接重试，不计入空响应
                empty_response_count += 1
                print(f"[HyMem] API failure, retrying ({empty_response_count}/{max_empty_retries})")
                if empty_response_count >= max_empty_retries:
                    print("[HyMem] Too many API failures, adding prompt to continue")
                    self.planner_messages.append({
                        'role': 'user', 
                        'content': "Please continue your analysis."
                    })
                    empty_response_count = 0
                continue
            
            if len(response.strip()) < 3:
                # 短响应：可能是模型特性，记录但不触发压缩
                print(f"[HyMem] Short response detected: '{response.strip()[:50]}'")
                self.fold_trigger.record_planner_turn(response)
                self.planner_messages.append({
                    'role': 'user', 
                    'content': "Please provide more detail. Use <task>...</task> for search, or <answer>...</answer> for final answer."
                })
                continue
            
            empty_response_count = 0
            self.fold_trigger.record_planner_turn(response)
            
            self.full_history.append({
                'role': 'planner',
                'content': response,
                'turn': turn + 1
            })
            self.event_log.log_planner_turn(turn=turn + 1, content=response, context_length=context_length)
            
            # 使用新的多重触发机制
            # Ablation: w/o Memory Management → never fold, keep memory store empty.
            if self.disable_memory_management:
                should_fold, fold_reason = False, ""
            else:
                should_fold, fold_reason = self.fold_trigger.should_fold(
                    context_length=context_length,
                    max_length=self.max_context_length,
                    response=response,
                    memory_store=self.memory_store,
                    current_turn=turn
                )

            if should_fold:
                print(f"[HyMem] Fold triggered: {fold_reason}")
                await self._fold_planner_context(problem, trigger_reason=fold_reason)
                continue
            
            answer = extract_answer(response)
            if answer:
                final_answer = answer
                confidence = extract_confidence(response)
                print(f"[HyMem] Found answer with confidence {confidence:.2f}")
                
                should_replan, replan_reason = self.fold_trigger.should_replan(confidence)
                
                if should_replan and self.current_replan_count < self.max_replan_attempts:
                    self.current_replan_count += 1
                    print(f"[HyMem] Replan triggered ({replan_reason}), attempt {self.current_replan_count}")
                    
                    replan_prompt = self._create_replan_prompt(answer, confidence, problem)
                    self.planner_messages.append({'role': 'assistant', 'content': response})
                    self.planner_messages.append({'role': 'user', 'content': replan_prompt})
                    
                    self.full_history.append({
                        'role': 'replan',
                        'reason': replan_reason,
                        'previous_answer': answer,
                        'confidence': confidence,
                        'turn': turn + 1
                    })
                    self.event_log.log_replan(turn=turn + 1, reason=replan_reason,
                                              previous_answer=answer, confidence=confidence)
                    continue
                
                break
            
            # ============== Optimization 1: parallel sub-task dispatch ==============
            # Pull every <task> and <branch> from this single planner turn so
            # genuinely independent sub-problems run concurrently. Falls back
            # to serial semantics when only one is emitted (backwards-compat).
            all_tasks = extract_all_tasks(response)
            all_branches = extract_all_branches(response)
            n_dispatch = len(all_tasks) + len(all_branches)
            if HYMEM_DEBUG:
                debug_log(f"Extracted {len(all_tasks)} task(s) + {len(all_branches)} branch(es) for this turn")

            if n_dispatch > 0:
                self.planner_messages.append({'role': 'assistant', 'content': response})

                # Build the concurrent dispatch list. Keep the planner's emit order
                # so observation aggregation reads naturally.
                dispatch_plan: List[Dict] = []  # ordered list
                for t in all_tasks:
                    # Ask the router whether this <task> is actually a search
                    # or should be promoted to a branch (the LLM may underuse
                    # <branch> for complex sub-problems).
                    # Ablation: w/o Isolated Reasoning → never route to branch.
                    hint_branch = bool(all_branches) and not self.disable_isolated_reasoning
                    dispatch_plan.append({'kind': 'task_pending_routing', 'task': t, 'hint_branch': hint_branch})
                for b in all_branches:
                    if self.disable_isolated_reasoning:
                        # Demote the branch to a plain executor search so the
                        # sub-problem is still attempted, just without an
                        # isolated reasoning sub-thread.
                        demoted = b['prompt'] or b['description']
                        dispatch_plan.append({'kind': 'executor', 'task': demoted})
                    else:
                        dispatch_plan.append({'kind': 'branch', 'description': b['description'], 'prompt': b['prompt']})

                # Resolve routing for the task entries (these classify calls
                # themselves can run in parallel — they're tiny LLM calls).
                routing_coros = [
                    self.complexity_router.classify(d['task'], hint_branch=d['hint_branch'])
                    for d in dispatch_plan if d['kind'] == 'task_pending_routing'
                ]
                routing_results = await asyncio.gather(*routing_coros) if routing_coros else []
                routing_iter = iter(routing_results)
                for d in dispatch_plan:
                    if d['kind'] == 'task_pending_routing':
                        r = next(routing_iter)
                        if (r.requires_branch and self.fold_trigger.can_branch()
                                and not self.disable_isolated_reasoning):
                            d['kind'] = 'branch'
                            d['description'] = f"Research: {d['task'][:50]}"
                            d['prompt'] = d.pop('task')
                            d['auto_routed'] = True
                            d['routing_source'] = r.source
                            d['routing_confidence'] = r.confidence
                        else:
                            d['kind'] = 'executor'

                # Concurrent execution. Cap concurrency so we don't overwhelm
                # remote APIs (DeepSeek/Serper) on big fan-outs.
                # O8: widened from 4 → 6 (or env override). Combined with the
                # planner-prompt nudge (HyMem PLANNER_SYSTEM_PROMPT) this lets
                # multi-fact tasks fan out farther per turn.
                # O8 default OFF: widening MAX_PARALLEL + the planner
                # prompt nudge to emit independent <task> blocks made the
                # planner over-fan on tasks with HIDDEN dependencies — the
                # paired 127-task A/B showed L3 acc -10pp and session +42%.
                # Set HYMEM_PARALLEL_PROMPT=1 (and optionally raise
                # HYMEM_MAX_PARALLEL) to re-enable for ablation.
                _parallel_cap = 6 if os.getenv("HYMEM_PARALLEL_PROMPT") == "1" else 4
                try:
                    _env_cap = int(os.getenv("HYMEM_MAX_PARALLEL", "0"))
                    if _env_cap > 0:
                        _parallel_cap = _env_cap
                except ValueError:
                    pass
                MAX_PARALLEL = max(1, min(n_dispatch, _parallel_cap))
                sem = asyncio.Semaphore(MAX_PARALLEL)
                # Track for stats
                if not hasattr(self, '_parallel_dispatch_fanouts'):
                    self._parallel_dispatch_fanouts = []
                self._parallel_dispatch_fanouts.append(n_dispatch)

                async def _run_one(idx: int, d: Dict):
                    async with sem:
                        if d['kind'] == 'executor':
                            print(f"[HyMem] [{idx+1}/{n_dispatch}] Executor: {d['task'][:60]}...")
                            res = await self.run_executor(d['task'], parent_turn=turn + 1)
                            ok = ("Error" not in res) and len(res) > 50
                            return {'kind': 'executor', 'task': d['task'],
                                    'result': res, 'success': ok, 'idx': idx}
                        else:  # branch
                            label = "Branch (auto-routed)" if d.get('auto_routed') else "Branch (explicit)"
                            print(f"[HyMem] [{idx+1}/{n_dispatch}] {label}: {d['description']}")
                            res, ok = await self.run_branch(
                                description=d['description'], prompt=d['prompt'],
                                parent_turn=turn + 1,
                            )
                            return {'kind': 'branch', 'description': d['description'],
                                    'prompt': d['prompt'], 'result': res, 'success': ok,
                                    'auto_routed': d.get('auto_routed', False), 'idx': idx}

                t_dispatch = time.time()
                results_list = await asyncio.gather(*[_run_one(i, d) for i, d in enumerate(dispatch_plan)])
                if n_dispatch > 1:
                    print(f"[HyMem] Parallel dispatch ({n_dispatch} sub-tasks) completed in {time.time()-t_dispatch:.1f}s")

                # Aggregate observation in emit-order, plus history bookkeeping.
                observation_parts: List[str] = []
                for r in sorted(results_list, key=lambda x: x['idx']):
                    if r['kind'] == 'executor':
                        self.fold_trigger.record_executor_call(r['success'])
                        self.memory_store.add_tool_call(
                            tool_name="executor",
                            arguments={"task": r['task']},
                            response=r['result'][:2000],
                            success=r['success'],
                        )
                        self.full_history.append({
                            'role': 'executor', 'task': r['task'],
                            'result': r['result'][:1000], 'turn': turn + 1,
                        })
                        observation_parts.append(
                            self._format_executor_results(r['result'], r['task'])
                        )
                    else:  # branch
                        self.fold_trigger.record_executor_call(r['success'])
                        self.full_history.append({
                            'role': 'branch',
                            'description': r['description'],
                            'prompt': r['prompt'],
                            'result': r['result'][:1000],
                            'success': r['success'],
                            'turn': turn + 1,
                            'auto_routed': r['auto_routed'],
                        })
                        observation_parts.append(
                            self._format_branch_result(r['result'], r['description'])
                        )

                # Single combined observation back to planner — keeps the
                # planner context clean (one user message instead of N).
                if len(observation_parts) == 1:
                    observation = observation_parts[0]
                else:
                    observation = "\n\n".join(
                        f"### Result {i+1} of {len(observation_parts)}\n{p}"
                        for i, p in enumerate(observation_parts)
                    )
                self.planner_messages.append({'role': 'user', 'content': observation})
                self.memory_store.executor_calls_this_session += sum(
                    1 for r in results_list if r['kind'] == 'executor'
                )

            else:
                # ============== Legacy single-branch fallback path ==============
                # extract_all_branches already covered the multi-branch case;
                # this path now only fires when nothing was extracted at all.
                branch_info = extract_branch_info(response)
                if branch_info and self.disable_isolated_reasoning:
                    # Ablation: w/o Isolated Reasoning → demote the lone branch
                    # to a plain executor search instead of a reasoning sub-thread.
                    demoted_task = branch_info['prompt'] or branch_info['description']
                    print(f"[HyMem][ABLATION] Branch demoted to executor: {demoted_task[:60]}...")
                    executor_result = await self.run_executor(demoted_task, parent_turn=turn + 1)
                    self.fold_trigger.record_executor_call("Error" not in executor_result)
                    self.memory_store.add_tool_call(
                        tool_name="executor",
                        arguments={"task": demoted_task},
                        response=executor_result[:2000],
                        success="Error" not in executor_result,
                    )
                    self.full_history.append({
                        'role': 'executor', 'task': demoted_task,
                        'result': executor_result[:1000], 'turn': turn + 1,
                    })
                    observation = self._format_executor_results(executor_result, demoted_task)
                    self.planner_messages.append({'role': 'assistant', 'content': response})
                    self.planner_messages.append({'role': 'user', 'content': observation})
                elif branch_info:
                    print(f"[HyMem] Explicit branch request: {branch_info['description']}")

                    branch_result, branch_success = await self.run_branch(
                        description=branch_info['description'],
                        prompt=branch_info['prompt'],
                        parent_turn=turn + 1
                    )
                    
                    self.full_history.append({
                        'role': 'branch',
                        'description': branch_info['description'],
                        'prompt': branch_info['prompt'],
                        'result': branch_result[:1000],
                        'success': branch_success,
                        'turn': turn + 1
                    })
                    
                    observation = self._format_branch_result(branch_result, branch_info['description'])
                    self.planner_messages.append({'role': 'assistant', 'content': response})
                    self.planner_messages.append({'role': 'user', 'content': observation})
                else:
                    observation = "No task or answer found. Please either:\n1. Assign a search task using <task>...</task>\n2. Provide your final answer using <answer>...</answer>"
                    self.planner_messages.append({'role': 'assistant', 'content': response})
                    self.planner_messages.append({'role': 'user', 'content': observation})
        
        if not final_answer and self.full_history:
            for entry in reversed(self.full_history):
                if entry.get('role') == 'planner':
                    content = entry.get('content', '')
                    final_answer = extract_answer(content) or "Unable to determine answer"
                    confidence = extract_confidence(content)
                    break
        
        # 合并 memory_manager token usage 到 event_log（非 fold 路径产生的，如 compress_tool_output）
        for call in self.memory_manager.token_usage.calls:
            already_logged = any(
                e.get("event") == "llm_call" and e.get("caller") == call["caller"]
                and abs(e.get("timestamp", 0) - call.get("timestamp", 0)) < 0.01
                for e in self.event_log.events
            )
            if not already_logged:
                self.event_log.log_llm_call(
                    caller=call["caller"],
                    prompt_tokens=call["prompt_tokens"],
                    completion_tokens=call["completion_tokens"],
                    total_tokens=call["total_tokens"],
                )

        session_stats = {
            "planner_turns": len([h for h in self.full_history if h.get('role') == 'planner']),
            "executor_calls": len([h for h in self.full_history if h.get('role') == 'executor']),
            "branch_calls": len([h for h in self.full_history if h.get('role') == 'branch']),
            "fold_count": len([h for h in self.full_history if h.get('role') == 'fold']),
            "replan_count": len([h for h in self.full_history if h.get('role') == 'replan']),
        }
        self.event_log.log_session_end(answer=final_answer or "", confidence=confidence, stats=session_stats)

        print(f"[HyMem] Session complete. {self.event_log.token_usage.summary_str()}")
        
        if HYMEM_DEBUG:
            debug_log(f"Answer: {final_answer[:500] if final_answer else 'None'}, confidence={confidence}")
            debug_save_to_file("full_history.json", json.dumps(self.full_history, ensure_ascii=False, indent=2, default=str))
            debug_save_to_file("event_log.json", json.dumps(self.event_log.to_dict(), ensure_ascii=False, indent=2, default=str))
        
        return final_answer, confidence, self.full_history
    
    async def close(self):
        await self.tool_manager.close()
        await self.memory_manager.close()


async def process_item(
    item: DataProto,
    context: TaskContext,
    LLMClass=CallAPI,
) -> DataProto:
    """HyMem评测入口函数"""
    # Preserve caller-configured proxy exclusions. This is required when the
    # model endpoint is on a private network while judge/search use a proxy.
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train
    
    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    
    ability = item.non_tensor_batch['ability'][0]
    
    from .utils import select_env
    EnvClass = select_env(ability, config)
    print(f"[HyMem] is_train={is_train}, EnvClass={EnvClass}")
    env = EnvClass(config, tokenizer, ability)
    
    try:
        await env.init_env(item)
        await env.get_data(item, context)
    except Exception as e:
        print(f"[HyMem] Environment init error: {e}")
    
    extra_info = item.non_tensor_batch.get('extra_info', [{}])[0]
    problem = extra_info.get('query', '') or extra_info.get('problem_statement', '')
    if not problem and hasattr(env, 'instance_info'):
        problem = env.instance_info.get('problem_statement', '') or env.instance_info.get('query', '')
    if not problem and hasattr(env, 'question'):
        problem = env.question or ''
    
    print(f"[HyMem] Problem: {problem[:200]}...")
    
    max_planner_turns = getattr(config.plugin, "hymem_max_planner_turns", 20)
    max_executor_turns = getattr(config.plugin, "hymem_max_executor_turns", 10)
    max_branch_turns = getattr(config.plugin, "hymem_max_branch_turns", 3)
    search_base_url = getattr(config.plugin, "search_base_url", None) or os.getenv("LOCAL_SEARCH_URL")
    
    # ============== 混合模型配置 ==============
    # Planner 配置（推荐使用强推理模型，如 DeepSeek Reasoner）
    planner_model = getattr(config.plugin, "hymem_planner_model", None) or os.getenv("HYMEM_PLANNER_MODEL") or \
                    getattr(config.plugin, "hymem_llm_model", None) or os.getenv("HYMEM_LLM_MODEL", "deepseek-reasoner")
    planner_base_url = getattr(config.plugin, "hymem_planner_base_url", None) or os.getenv("HYMEM_PLANNER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    planner_api_key = getattr(config.plugin, "hymem_planner_api_key", None) or os.getenv("HYMEM_PLANNER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Executor 配置（可使用本地模型，如 Qwen-32B）
    executor_model = getattr(config.plugin, "hymem_executor_model", None) or os.getenv("HYMEM_EXECUTOR_MODEL") or planner_model
    executor_base_url = getattr(config.plugin, "hymem_executor_base_url", None) or os.getenv("HYMEM_EXECUTOR_BASE_URL") or planner_base_url
    executor_api_key = getattr(config.plugin, "hymem_executor_api_key", None) or os.getenv("HYMEM_EXECUTOR_API_KEY") or planner_api_key
    
    # Aux (Memory Manager) 配置（可使用本地模型）
    aux_model = getattr(config.plugin, "hymem_aux_model", None) or os.getenv("HYMEM_AUX_MODEL") or executor_model
    aux_base_url = getattr(config.plugin, "hymem_aux_base_url", None) or os.getenv("HYMEM_AUX_BASE_URL") or executor_base_url
    aux_api_key = getattr(config.plugin, "hymem_aux_api_key", None) or os.getenv("HYMEM_AUX_API_KEY") or executor_api_key
    
    # 兼容旧配置：如果只设置了 hymem_llm_model，所有组件使用相同配置
    llm_model = planner_model  # 用于 tool_manager 等其他组件
    
    # 搜索模式配置：local, online, auto
    search_mode = getattr(config.plugin, "search_mode", None) or os.getenv("SEARCH_MODE", "auto")
    google_api_key = getattr(config.plugin, "google_api_key", None) or os.getenv("GOOGLE_API_KEY")
    
    host = context.server_host
    port = context.server_port
    
    # Step B1 — explicit per-component model disclosure (Reviewer LMtp asked
    # whether the auxiliary LLM is the same size as the main planner; this
    # banner + the env.stats['models_used'] field below answer that directly).
    judge_model_for_log = os.getenv("JUDGE_MODEL", "(env JUDGE_MODEL unset)")
    print(f"[HyMem] Models: planner={planner_model} | executor={executor_model} | aux={aux_model} | judge={judge_model_for_log}")
    print(f"[HyMem] Model Config:")
    print(f"  - Planner: {planner_model} @ {planner_base_url or 'default'}")
    print(f"  - Executor: {executor_model} @ {executor_base_url or 'default'}")
    print(f"  - Aux/Memory: {aux_model} @ {aux_base_url or 'default'}")
    
    # 创建各组件的 LLM 客户端（支持独立配置）
    planner_client = LLMClass(planner_model, port, tokenizer, config, meta_info={},
                              base_url=planner_base_url, api_key=planner_api_key)
    executor_client = LLMClass(executor_model, port, tokenizer, config, meta_info={},
                               base_url=executor_base_url, api_key=executor_api_key)
    
    tool_config = BMToolsConfig(
        llm_model=executor_model,  # 工具使用 Executor 模型（本地模型）
        search_topk=getattr(config.plugin, "hymem_search_topk", 10),
        max_keywords=getattr(config.plugin, "hymem_max_keywords", 8)
    )
    # BMToolManager 使用 Executor 配置（本地模型），而不是 Planner 配置
    tool_manager = BMToolManager(
        search_base_url=search_base_url,
        llm_model=executor_model,
        llm_base_url=executor_base_url,
        llm_api_key=executor_api_key,
        config=tool_config,
        search_mode=search_mode,
        google_api_key=google_api_key
    )
    
    # 打印 ToolManager 配置
    print(f"  - ToolManager: {executor_model} @ {executor_base_url or 'default'}")
    
    # 获取 prompt_style 配置
    # Step C1 — auto-select prompt_style based on model size when the user
    # didn't explicitly set HYMEM_PROMPT_STYLE. Smaller models get the
    # simpler "qwen"-style prompts (200-char thinking cap, no nested tags);
    # 32B+ models get the "default" full prompts.
    explicit_style = (
        getattr(config.plugin, "hymem_prompt_style", None)
        or os.getenv("HYMEM_PROMPT_STYLE")
    )
    if explicit_style:
        prompt_style = explicit_style
    else:
        size = _infer_model_size(planner_model)
        prompt_style = "qwen" if size in ("tiny", "small") else "default"
        print(f"[HyMem] Auto-selected prompt_style={prompt_style} (planner size={size})")
    
    # ============== Ablation configuration ==============
    # Selects which HyMem component to disable for the paper ablation table.
    # Sources (in priority order): config.plugin.hymem_ablation, then env
    # HYMEM_ABLATION. Accepted values:
    #   full | (none)              → HyMem (Full)
    #   no_isolated_reasoning|no_ir → w/o Isolated Reasoning (disable Branch)
    #   no_memory_management|no_mm  → w/o Memory Management (disable fold+memory)
    ablation = (getattr(config.plugin, "hymem_ablation", None)
                or os.getenv("HYMEM_ABLATION", "full") or "full").strip().lower()
    disable_isolated_reasoning = ablation in ("no_isolated_reasoning", "no_ir", "wo_ir")
    disable_memory_management = ablation in ("no_memory_management", "no_mm", "wo_mm")
    if ablation not in ("full", "none", "", "no_isolated_reasoning", "no_ir",
                        "wo_ir", "no_memory_management", "no_mm", "wo_mm"):
        print(f"[HyMem][ABLATION] WARNING: unknown HYMEM_ABLATION='{ablation}', "
              f"running HyMem (Full)")
    print(f"[HyMem] Ablation mode: {ablation}")

    agent = HyMemAgent(
        planner_client=planner_client,
        executor_client=executor_client,
        tool_manager=tool_manager,
        tokenizer=tokenizer,
        config=config,
        aux_model=aux_model,
        aux_base_url=aux_base_url,
        aux_api_key=aux_api_key,
        max_planner_turns=max_planner_turns,
        max_executor_turns=max_executor_turns,
        max_branch_turns=max_branch_turns,
        prompt_style=prompt_style,
        disable_isolated_reasoning=disable_isolated_reasoning,
        disable_memory_management=disable_memory_management,
    )
    
    session_start_time = time.time()
    
    try:
        answer, confidence, history = await agent.run(problem)
        print(f"[HyMem] Answer: {answer[:200]}...")
        print(f"[HyMem] Confidence: {confidence:.2f}")
    except Exception as e:
        print(f"[HyMem] Execution error: {e}")
        traceback.print_exc()
        answer = ""
        confidence = 0.0
        history = []
    finally:
        await agent.close()
    
    env.stats['session_time'] = time.time() - session_start_time
    
    from .hymem_prompts import HYMEM_PLANNER_SYSTEM_PROMPT, HYMEM_PLANNER_USER_PROMPT
    messages = [
        {'role': 'system', 'content': HYMEM_PLANNER_SYSTEM_PROMPT},
        {'role': 'user', 'content': HYMEM_PLANNER_USER_PROMPT.format(problem=problem)}
    ]
    
    for entry in history:
        if entry['role'] == 'planner':
            messages.append({'role': 'assistant', 'content': entry['content']})
        elif entry['role'] == 'executor':
            messages.append({'role': 'user', 'content': f"Task: {entry['task']}\n\nResults: {entry['result']}"})
        elif entry['role'] == 'branch':
            messages.append({'role': 'user', 'content': f"Branch '{entry['description']}' result:\n{entry['result']}"})
    
    if answer:
        final_response = f"<answer>{answer}</answer>\n<confidence>{confidence}</confidence>"
        messages.append({'role': 'assistant', 'content': final_response})
    
    env.finish = bool(answer)
    env.is_finish = bool(answer)
    env.predicted_answer = (answer, None, confidence)
    
    print('[HyMem] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, messages, context), timeout=60 * 10)
        score = (score_msg, reward)
        print(f"[HyMem] Score: {score}")
    except Exception as e:
        print(f"[HyMem] Reward error: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}
    
    prompt_turn = 2
    from .utils import Agent as AgentContext
    agent_context = AgentContext(planner_client, messages[:prompt_turn], tokenizer, config, prompt_turn=prompt_turn)
    
    for msg in messages[prompt_turn:]:
        agent_context.append(msg)
    
    tool_stats = tool_manager.get_tool_stats()
    token_data = agent.event_log.token_usage.to_dict()
    env.stats['get_final_score'] = score[1]
    env.stats['confidence'] = confidence
    env.stats['planner_turns'] = len([h for h in history if h['role'] == 'planner'])
    env.stats['executor_calls'] = len([h for h in history if h['role'] == 'executor'])
    env.stats['branch_calls'] = len([h for h in history if h['role'] == 'branch'])
    env.stats['branch_success'] = len([h for h in history if h['role'] == 'branch' and h.get('success', False)])
    env.stats['fold_count'] = len([h for h in history if h['role'] == 'fold'])
    env.stats['replan_count'] = len([h for h in history if h['role'] == 'replan'])
    # G1 — per-fold-event token snapshots for §A paired comparison with ReSum
    env.stats['hymem_fold_events'] = list(agent.fold_events)
    # G2 — memory operation counters for §B paired comparison with A-MEM
    env.stats['hymem_memory_ops'] = dict(agent.memory_store.op_counts)
    # G4 — per-turn context size trace for cross-baseline Figure 3a-revised.
    # We have richer per-event token snapshots in event_log, but expose a
    # uniform-shape series here so analysis scripts iterate over all 6
    # frameworks identically. Each entry = (turn_idx, context_token_len)
    # for the Coordination Layer (planner_messages) — this is the line
    # whose "≈ 1/2 ReAct slope" claim Figure 3a is built on.
    _coord_history = []
    try:
        for ev in agent.event_log.events:
            if ev.get("event") == "planner_turn":
                _coord_history.append((ev.get("turn", 0), ev.get("context_length", 0)))
    except Exception:
        pass
    env.stats['context_size_history'] = _coord_history
    # Figure-1 dilution: per-planner-turn full message snapshot. Used by
    # scripts/analysis/key_fact_dilution.py to reconstruct C_t and
    # substring-match key facts.
    env.stats['planner_context_per_turn'] = list(
        getattr(agent, '_planner_context_snapshots', []) or []
    )
    # Step A3 — unified tool-call accounting (paper-friendly metric).
    # Writes tool_calls_atomic / agent_turns / tool_calls_by_source / framework
    # plus legacy tool_calls_total alias. Branch-internal calls are tagged
    # 'branch' (see run_branch wrapper).
    from .tool_metrics import merge_hymem_tool_stats
    _agent_turns = (
        env.stats['planner_turns']
        + env.stats['executor_calls']
        + env.stats['branch_calls']
    )
    merge_hymem_tool_stats(env.stats, tool_stats, agent_turns=_agent_turns)
    env.stats['tool_call_history_full'] = tool_stats.get('tool_call_history', [])
    env.stats['distillation_stats'] = tool_stats.get('distillation_stats_aggregate', {})
    # Step B1 — model disclosure (LMtp comment about aux model size)
    env.stats['models_used'] = {
        'planner': planner_model,
        'executor': executor_model,
        'aux': aux_model,
        'judge': os.getenv('JUDGE_MODEL', 'unknown'),
    }
    # Ablation provenance — which variant produced this result.
    env.stats['ablation'] = ablation
    env.stats['ablation_flags'] = {
        'disable_isolated_reasoning': disable_isolated_reasoning,
        'disable_memory_management': disable_memory_management,
    }
    # Step B2 — per-component wall-clock latency (Bh2U W2)
    env.stats['latency_by_component'] = agent.event_log.latency.to_dict()
    # Step A1: complexity-router visibility for paper-ready analysis
    env.stats['routing_decisions'] = agent.complexity_router.routing_decisions()
    env.stats['routing_summary'] = agent.complexity_router.stats()
    env.stats['hymem_history'] = history
    env.stats['episode_memory'] = agent.memory_store.episode_memory
    env.stats['working_memory'] = agent.memory_store.working_memory
    env.stats['tool_memory'] = agent.memory_store.tool_memory
    env.stats['planner_summary'] = agent.memory_store.planner_summary
    env.stats['tool_call_history'] = list(agent.memory_store.tool_call_history)
    env.stats['branch_history'] = agent.memory_store.branch_history
    # Token usage from unified event log
    env.stats['total_prompt_tokens'] = token_data['total']['prompt_tokens']
    env.stats['total_completion_tokens'] = token_data['total']['completion_tokens']
    env.stats['total_tokens'] = token_data['total']['total_tokens']
    env.stats['token_usage_by_component'] = token_data['by_component']
    env.stats['event_log'] = agent.event_log.to_dict()

    # ------------------------------------------------------------------
    # Round 1-3 efficiency observation fields (paper Table: Bh2U W2)
    # All names mirror the perf-plan whitelist so analysis scripts can
    # uniformly slice them across HyMem runs.
    # ------------------------------------------------------------------
    _tool_stats_d = tool_stats or {}
    _distill_agg = _tool_stats_d.get('distillation_stats_aggregate', {}) or {}

    # O1 — URL-level web_parse cache
    env.stats['parse_cache_hits'] = _tool_stats_d.get('parse_cache_hits', 0)
    env.stats['parse_cache_misses'] = _tool_stats_d.get('parse_cache_misses', 0)
    env.stats['parse_cache_hit_rate'] = _tool_stats_d.get('parse_cache_hit_rate', 0.0)

    # O5 — executor early-exit
    _inner = list(getattr(agent, '_executor_inner_turns_used', []) or [])
    env.stats['executor_early_exit_count'] = getattr(
        agent, '_executor_early_exit_count', 0)
    env.stats['executor_avg_inner_turns'] = (
        sum(_inner) / len(_inner) if _inner else 0.0
    )

    # O2-O3 — distillation density / concurrency
    _invs = int(_distill_agg.get('invocations', 0))
    env.stats['distillation_invocations'] = _invs
    env.stats['distillation_avg_aux_calls'] = (
        _distill_agg.get('total_aux_calls', 0) / _invs if _invs else 0.0
    )
    env.stats['distillation_chunk_concurrent_used'] = _distill_agg.get(
        'chunk_concurrent_used_max', 0)

    # O6 — fold wall + parallel aux
    _folds = list(getattr(agent, '_fold_wall_seconds', []) or [])
    _aux_durs = list(getattr(agent.memory_manager, 'fold_aux_durations', []) or [])
    env.stats['fold_invocations'] = len(_folds)
    env.stats['fold_avg_wall_seconds'] = (
        sum(_folds) / len(_folds) if _folds else 0.0
    )
    env.stats['fold_aux_parallel_seconds'] = (
        sum(_aux_durs) / len(_aux_durs) if _aux_durs else 0.0
    )

    # O8 — planner parallel-dispatch fanout
    _fanouts = list(getattr(agent, '_parallel_dispatch_fanouts', []) or [])
    env.stats['parallel_dispatch_fanout_avg'] = (
        sum(_fanouts) / len(_fanouts) if _fanouts else 0.0
    )
    env.stats['parallel_dispatch_max_observed'] = max(_fanouts) if _fanouts else 0

    out = await agent_context.dataproto()
    out = await env.update_dataproto(out, item, messages, score, reward_dict, tag='main', metrics={})
    
    res = DataProto.concat([out])
    return res


async def run_hymem_simple(
    problem: str,
    search_base_url: str = None,
    # Planner 配置（推荐使用强推理模型）
    planner_model: str = None,
    planner_base_url: str = None,
    planner_api_key: str = None,
    # Executor 配置（可使用本地模型）
    executor_model: str = None,
    executor_base_url: str = None,
    executor_api_key: str = None,
    # Memory 配置（可使用本地模型）
    aux_model: str = None,
    aux_base_url: str = None,
    aux_api_key: str = None,
    # 其他配置
    search_mode: str = "auto",
    google_api_key: str = None,
    prompt_style: str = "default",  # "default" 或 "qwen"
    # 兼容旧参数
    llm_model: str = None
) -> Tuple[str, float, Dict]:
    """
    简化版HyMem运行函数，支持混合模型架构
    
    Args:
        problem: 要解决的问题
        search_base_url: LocalSearch 服务地址（search_mode="local" 时使用）
        
        # Planner 配置（用于规划和 Branch 推理，推荐使用强模型）
        planner_model: Planner 模型名称（默认: deepseek-reasoner）
        planner_base_url: Planner API 端点
        planner_api_key: Planner API 密钥
        
        # Executor 配置（执行搜索，可使用本地模型）
        executor_model: Executor 模型名称（默认: 跟随 planner_model）
        executor_base_url: Executor API 端点（默认: 跟随 planner）
        executor_api_key: Executor API 密钥（默认: 跟随 planner）
        
        # Memory 配置（压缩上下文，可使用本地模型）
        aux_model: Memory 模型名称（默认: 跟随 executor）
        aux_base_url: Memory API 端点（默认: 跟随 executor）
        aux_api_key: Memory API 密钥（默认: 跟随 executor）
        
        search_mode: 搜索模式 ("local", "online", "auto")
        google_api_key: Google Custom Search API Key（search_mode="online" 时可选）
        llm_model: 兼容旧参数，等同于 planner_model
        
    Returns:
        (answer, confidence, stats)
        
    Usage Examples:
        # 示例1：全部使用 DeepSeek Reasoner
        await run_hymem_simple(problem, planner_model="deepseek-reasoner")
        
        # 示例2：Planner 用 DeepSeek，Executor 用本地 Qwen
        await run_hymem_simple(
            problem,
            planner_model="deepseek-reasoner",
            planner_base_url="https://api.deepseek.com/v1",
            planner_api_key="sk-xxx",
            executor_model="Qwen3-32B",
            executor_base_url="http://localhost:8000/v1",
            executor_api_key="token-local"
        )
    """
    from openai import AsyncOpenAI
    
    # 解析配置，支持继承和默认值
    # Planner 配置
    _planner_model = planner_model or llm_model or os.getenv("HYMEM_PLANNER_MODEL") or os.getenv("HYMEM_LLM_MODEL", "deepseek-reasoner")
    _planner_base_url = planner_base_url or os.getenv("HYMEM_PLANNER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _planner_api_key = planner_api_key or os.getenv("HYMEM_PLANNER_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    # Executor 配置（默认继承 Planner）
    _executor_model = executor_model or os.getenv("HYMEM_EXECUTOR_MODEL") or _planner_model
    _executor_base_url = executor_base_url or os.getenv("HYMEM_EXECUTOR_BASE_URL") or _planner_base_url
    _executor_api_key = executor_api_key or os.getenv("HYMEM_EXECUTOR_API_KEY") or _planner_api_key
    
    # Memory 配置（默认继承 Executor）
    _aux_model = aux_model or os.getenv("HYMEM_AUX_MODEL") or _executor_model
    _aux_base_url = aux_base_url or os.getenv("HYMEM_AUX_BASE_URL") or _executor_base_url
    _aux_api_key = aux_api_key or os.getenv("HYMEM_AUX_API_KEY") or _executor_api_key
    
    # 打印配置信息
    print(f"[HyMem Simple] Model Config:")
    print(f"  - Planner: {_planner_model} @ {_planner_base_url or 'default'}")
    print(f"  - Executor: {_executor_model} @ {_executor_base_url or 'default'}")
    print(f"  - Memory: {_aux_model} @ {_aux_base_url or 'default'}")
    
    class SimpleLLMClient:
        """支持独立配置的 LLM 客户端"""
        def __init__(self, model: str, base_url: str = None, api_key: str = None):
            self.model = model
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        
        async def create_completion(self, input_ids, messages=None, max_new_tokens=4096):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_new_tokens
                )
                message = response.choices[0].message
                text = message.content or ""
                # DeepSeek-Reasoner: 如果content为空，尝试使用reasoning_content
                if not text and hasattr(message, 'reasoning_content') and message.reasoning_content:
                    text = message.reasoning_content
                usage_data = {}
                if response.usage:
                    usage_data = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    }
                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": [],
                            "response_log_probs": [],
                            "metrics": {"usage": usage_data},
                        }
                    }]
                }
            except Exception as e:
                error_str = str(e)
                if "400" in error_str and ("Content Exists Risk" in error_str or "content_filter" in error_str.lower()):
                    print(f"[SimpleLLM] Content moderation triggered: {error_str[:200]}")
                else:
                    print(f"[SimpleLLM] Error: {e}")
                return None
    
    class SimpleTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
            return list(range(100))
        def encode(self, text):
            return list(range(len(text) // 4))
    
    class SimpleConfig:
        response_length = 32000
        prompt_length = 8000
        class plugin:
            pass
    
    tool_manager = BMToolManager(
        search_base_url=search_base_url,
        llm_model=_planner_model,
        search_mode=search_mode,
        google_api_key=google_api_key
    )
    
    # 为不同组件创建独立的客户端
    planner_client = SimpleLLMClient(_planner_model, _planner_base_url, _planner_api_key)
    executor_client = SimpleLLMClient(_executor_model, _executor_base_url, _executor_api_key)
    
    agent = HyMemAgent(
        planner_client=planner_client,
        executor_client=executor_client,
        tool_manager=tool_manager,
        tokenizer=SimpleTokenizer(),
        config=SimpleConfig(),
        aux_model=_aux_model,
        aux_base_url=_aux_base_url,
        aux_api_key=_aux_api_key,
        max_planner_turns=15,
        max_executor_turns=8,
        prompt_style=prompt_style
    )
    
    try:
        answer, confidence, history = await agent.run(problem)
        t_stats = tool_manager.get_tool_stats()
        token_data = agent.event_log.token_usage.to_dict()
        stats = {
            "planner_turns": len([h for h in history if h['role'] == 'planner']),
            "executor_calls": len([h for h in history if h['role'] == 'executor']),
            "branch_calls": len([h for h in history if h['role'] == 'branch']),
            "branch_success": len([h for h in history if h['role'] == 'branch' and h.get('success', False)]),
            "fold_count": len([h for h in history if h['role'] == 'fold']),
            "replan_count": len([h for h in history if h['role'] == 'replan']),
            "tool_calls_total": t_stats.get('tool_calls_total', 0),
            "tool_calls_by_tool": t_stats.get('tool_calls_by_tool', {}),
            "tool_call_history": list(agent.memory_store.tool_call_history),
            "search_mode": tool_manager.search_mode,
            "is_online_mode": tool_manager.is_online_mode(),
            "total_prompt_tokens": token_data['total']['prompt_tokens'],
            "total_completion_tokens": token_data['total']['completion_tokens'],
            "total_tokens": token_data['total']['total_tokens'],
            "token_usage_by_component": token_data['by_component'],
            "event_log": agent.event_log.to_dict(),
        }
        return answer, confidence, stats
    finally:
        await agent.close()


async def _test_hymem():
    """测试HyMem Agent - 支持混合模型架构"""
    print("Testing HyMem Agent (Hybrid Model Architecture)...")
    print("=" * 60)
    
    # 测试问题
    problem = "What is the capital of France?"
    
    try:
        print(f"\n[Test] Problem: {problem}")
        
        # 示例1：全部使用 DeepSeek Reasoner（默认模式）
        print("\n[Test Mode 1] All components use DeepSeek Reasoner...")
        answer, confidence, stats = await run_hymem_simple(
            problem,
            search_mode="online",
            planner_model="deepseek-reasoner"
        )
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Stats: {stats}")
        

    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()


async def _test_hymem_hybrid():
    """测试混合模型架构 - Planner 用强模型，Executor 用本地模型"""
    print("Testing HyMem Agent (Hybrid Model Architecture)...")
    print("=" * 60)
    
    problem = "2024年诺贝尔物理学奖获得者是谁？他们的主要贡献是什么？"
    
    # 检查必要的环境变量
    if not os.getenv("HYMEM_PLANNER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set HYMEM_PLANNER_API_KEY or OPENAI_API_KEY")
        return
    
    try:
        print(f"\n[Test] Problem: {problem}")
        print("\n[Config] Hybrid Model Architecture:")
        print("  - Planner/Branch: DeepSeek Reasoner (deep reasoning)")
        print("  - Executor/Memory: Local model (execution)")
        
        answer, confidence, stats = await run_hymem_simple(
            problem,
            search_mode="online",
            # 可以通过环境变量或参数配置不同的模型
            # Planner 使用 DeepSeek Reasoner
            planner_model=os.getenv("HYMEM_PLANNER_MODEL", "deepseek-reasoner"),
            # Executor 使用本地模型（如果配置了）
            executor_model=os.getenv("HYMEM_EXECUTOR_MODEL"),
            executor_base_url=os.getenv("HYMEM_EXECUTOR_BASE_URL"),
        )
        
        print(f"\n[Result]")
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "hybrid":
        asyncio.run(_test_hymem_hybrid())
    else:
        asyncio.run(_test_hymem())
