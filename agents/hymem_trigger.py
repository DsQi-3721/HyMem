"""
HyMem Trigger Module

多重触发机制 + 任务复杂度判断 + 执行路由
融合FoldAgent、DeepAgent、BrowseMaster的核心思想

NOTE on ComplexityEstimator (Step A1):
  This estimator is now the FALLBACK path for the LLM-based router
  (``agents.complexity_router.LLMComplexityRouter``). It is NOT the primary
  routing mechanism — the LLM router is. The estimator runs only when the
  LLM call fails or returns a low-confidence verdict.

  It has been upgraded from English-only keyword scoring to a multi-feature
  multilingual scorer (Chinese + English + multi-hop indicators + length
  + entity count) so it remains useful as a safety net.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from .hymem_memory import HyMemStore


@dataclass
class TriggerConfig:
    """触发器配置 — Memory Manager activation rules (HyMem paper Sec 3.2.4 / Sec A.2).

    Six concrete trigger conditions evaluated in priority order each turn
    (see ``FoldTrigger.should_fold``):

      1. **active_fold**          — model emits ``<fold_thought>`` marker
      2. **executor_calls**       — Executor invocations since last fold
                                    >= ``executor_calls_threshold``
      3. **context_overflow**     — context_length / max_length
                                    > ``planner_context_threshold``
      4. **consecutive_failures** — consecutive Executor failures
                                    >= ``consecutive_failures_threshold``
      5. **tool_history**         — total tool-call history length
                                    >= ``tool_history_threshold``
      6. **planner_turns**        — Planner turns since last fold
                                    >= ``planner_turns_threshold``
      (also: consecutive low-confidence verdicts; see ``should_replan``)

    The paper Sec A.2 currently mentions only conditions (3) and "executor
    fails multiple times" (≈ condition 4). Conditions (1), (2), (5), (6),
    (7) are implementation refinements added on top — the paper's Sec A.2
    table needs to be updated to list all six. None of these conditions
    change the agent's correctness; they only affect *when* fold/replan
    fires, so adjusting the paper's description (rather than removing
    triggers) preserves both behavior and reviewer transparency.
    """
    # 上下文阈值（降低以更早触发）
    planner_context_threshold: float = 0.99
    executor_context_threshold: float = 0.99
    branch_context_threshold: float = 0.99
    # 折叠限制
    max_folds: int = 5
    min_turns_between_folds: int = 3
    # 轮次触发（核心改进）
    executor_calls_threshold: int = 5
    planner_turns_threshold: int = 8
    tool_history_threshold: int = 10
    # 失败触发
    consecutive_failures_threshold: int = 3
    consecutive_empty_responses_threshold: int = 2
    # 置信度触发
    low_confidence_threshold: float = 0.5
    very_low_confidence_threshold: float = 0.3
    consecutive_low_confidence_trigger: int = 2
    # 主动触发标记
    fold_thought_marker: str = "<fold_thought>"
    # Branch 配置 — paper Sec A.2: max_iso_reasoning=5, max_iso_reasoning_turn=3
    max_branches: int = 5
    max_branch_turns: int = 3
    # Branch 快速失败配置（实现细节，未在论文中报告）
    branch_enable_early_stop: bool = True
    branch_max_consecutive_search_failures: int = 3
    branch_max_total_search_attempts: int = 4
    branch_min_result_length: int = 50


BRANCH_REASONING_INDICATORS = [
    # English
    "verify", "check if", "check whether", "confirm", "validate",
    "compare", "analyze", "evaluate", "determine if", "determine whether",
    "investigate", "explore", "assess", "examine", "review",
    "which one", "which of", "is it true", "does it satisfy",
    "find out if", "figure out", "reason about", "think through",
    # Chinese
    "验证", "核实", "确认", "判断", "对比", "比较", "分析", "评估",
    "推理", "证明", "确定", "鉴别", "辨别", "审查", "考察",
]

SEARCH_ONLY_INDICATORS = [
    # English
    "search for", "find information", "look up", "get data",
    "retrieve", "fetch", "list all", "show me",
    # Chinese
    "搜索", "查找", "查询", "检索", "获取", "拿到", "找出", "列出", "找到",
]

# Multi-hop / chained-inference indicators across languages
MULTI_HOP_INDICATORS = [
    " and then ", " then ", " after that ", " followed by ",
    "之后", "然后", "接着", "最后", " vs ", " versus ", " compared to ",
]


def should_use_branch(task_description: str) -> bool:
    """
    判断任务是否适合用Branch而非Executor
    
    Branch适合：需要推理、验证、比较、分析的子问题
    Executor适合：简单的信息检索任务
    """
    task_lower = task_description.lower()
    
    for indicator in SEARCH_ONLY_INDICATORS:
        if indicator in task_lower:
            has_reasoning = any(ind in task_lower for ind in BRANCH_REASONING_INDICATORS)
            if not has_reasoning:
                return False
    
    reasoning_score = sum(1 for ind in BRANCH_REASONING_INDICATORS if ind in task_lower)
    
    if reasoning_score >= 2:
        return True
    
    if reasoning_score >= 1:
        complex_markers = ["multiple", "several", "various", "different", "both"]
        if any(marker in task_lower for marker in complex_markers):
            return True
    
    return False


def classify_task_type(task_description: str) -> str:
    """
    分类任务类型
    
    Returns:
        'branch': 需要独立推理的子问题
        'executor': 简单的搜索任务
        'hybrid': 可能两者皆可
    """
    task_lower = task_description.lower()
    
    reasoning_score = sum(1 for ind in BRANCH_REASONING_INDICATORS if ind in task_lower)
    search_score = sum(1 for ind in SEARCH_ONLY_INDICATORS if ind in task_lower)
    
    if reasoning_score >= 2 and search_score == 0:
        return 'branch'
    elif search_score >= 1 and reasoning_score == 0:
        return 'executor'
    elif reasoning_score > search_score:
        return 'branch'
    elif search_score > reasoning_score:
        return 'executor'
    else:
        return 'hybrid'


class ComplexityEstimator:
    """任务复杂度估计器 — Step A1 fallback only.

    Multi-feature scoring (language-agnostic):
      - keyword indicators (Chinese + English) for branch vs. search intent
      - multi-hop chain indicators ("then", "之后", "vs", ...)
      - task length
      - question density (how many ? / ？)
      - entity-density proxy (light count of capitalized sequences and
        common entity markers)

    Each feature contributes to a complex_score / simple_score; the higher
    score wins. Confidence reflects margin of victory.
    """

    @staticmethod
    def _feature_scores(task: str) -> Tuple[float, float, dict]:
        task_lower = task.lower()
        complex_score = 0.0
        simple_score = 0.0
        breakdown: dict = {}

        # 1) keyword indicators
        kw_complex = sum(1 for ind in BRANCH_REASONING_INDICATORS if ind in task_lower)
        kw_simple = sum(1 for ind in SEARCH_ONLY_INDICATORS if ind in task_lower)
        complex_score += kw_complex
        simple_score += kw_simple
        breakdown["kw_complex"] = kw_complex
        breakdown["kw_simple"] = kw_simple

        # 2) multi-hop indicators
        multihop = sum(1 for ind in MULTI_HOP_INDICATORS if ind in task_lower)
        complex_score += multihop * 1.5
        breakdown["multihop"] = multihop

        # 3) length
        if len(task) > 400:
            complex_score += 1.5
        elif len(task) > 200:
            complex_score += 0.7
        breakdown["length"] = len(task)

        # 4) question density
        n_q = task.count("?") + task.count("？")
        if n_q >= 2:
            complex_score += 1.0
        breakdown["questions"] = n_q

        # 5) entity proxy (multi-word capitalized sequences are reasonably
        # cheap to count without a real NER model)
        import re as _re
        n_ents = len(_re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", task))
        if n_ents >= 3:
            complex_score += 1.0
        breakdown["entities"] = n_ents

        return complex_score, simple_score, breakdown

    @staticmethod
    def estimate(task: str) -> Tuple[str, float]:
        """估计任务复杂度，返回 (complexity_type, confidence)。"""
        if not task:
            return "simple", 0.5
        complex_score, simple_score, _ = ComplexityEstimator._feature_scores(task)
        margin = abs(complex_score - simple_score)
        # Map margin to a confidence in [0.5, 0.9]
        conf = min(0.9, 0.5 + 0.1 * margin)
        if complex_score > simple_score:
            return "complex", conf
        elif simple_score > complex_score:
            return "simple", conf
        else:
            # Tie — slight bias toward complex (safer to spawn a branch when
            # truly ambiguous, since branches early-exit on simple findings)
            return "complex", 0.5

    @staticmethod
    def should_use_branch(task: str) -> bool:
        complexity, _ = ComplexityEstimator.estimate(task)
        return complexity == "complex"


class FoldTrigger:
    """多重触发机制的折叠触发器"""
    
    def __init__(self, config: TriggerConfig = None):
        self.config = config or TriggerConfig()
        self.confidence_history: List[float] = []
        self.consecutive_empty_responses: int = 0
        self.consecutive_failures: int = 0
        self.executor_calls_since_fold: int = 0
        self.planner_turns_since_fold: int = 0
        self.last_trigger_reason: str = ""
        self.branch_count: int = 0
        self.branch_results: List[Dict] = []
    
    def reset(self):
        """折叠后重置计数器"""
        self.confidence_history.clear()
        self.consecutive_empty_responses = 0
        self.consecutive_failures = 0
        self.executor_calls_since_fold = 0
        self.planner_turns_since_fold = 0
        self.last_trigger_reason = ""
    
    def reset_branches(self):
        self.branch_count = 0
        self.branch_results.clear()
    
    def can_branch(self) -> bool:
        return self.branch_count < self.config.max_branches
    
    def record_branch(self, description: str, success: bool, result_summary: str = ""):
        self.branch_count += 1
        self.branch_results.append({
            'description': description,
            'success': success,
            'result_summary': result_summary[:200]
        })
    
    def record_planner_turn(self, response: str):
        """记录Planner轮次"""
        self.planner_turns_since_fold += 1
        if not response or len(response.strip()) < 10:
            self.consecutive_empty_responses += 1
        else:
            self.consecutive_empty_responses = 0
    
    def record_executor_call(self, success: bool = True):
        """记录Executor调用"""
        self.executor_calls_since_fold += 1
        if not success:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
    
    def record_confidence(self, confidence: float):
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 10:
            self.confidence_history.pop(0)
    
    def can_fold(self, memory_store: HyMemStore, current_turn: int) -> bool:
        if memory_store.fold_count >= self.config.max_folds:
            return False
        if memory_store.fold_count > 0:
            turns_since = current_turn - memory_store.last_fold_turn
            if turns_since < self.config.min_turns_between_folds:
                return False
        return True
    
    def should_fold(
        self,
        context_length: int,
        max_length: int,
        response: str,
        memory_store: HyMemStore,
        current_turn: int
    ) -> Tuple[bool, str]:
        """
        多重触发机制判断是否需要折叠
        
        触发条件优先级：
        1. 主动触发：模型输出 <fold_thought>
        2. 轮次触发：Executor调用次数超过阈值
        3. 上下文触发：上下文使用率超过阈值
        4. 失败触发：连续空输出或错误
        5. 信息量触发：工具调用历史超过阈值
        6. Planner轮次触发
        """
        if not self.can_fold(memory_store, current_turn):
            return False, ""
        
        if self.config.fold_thought_marker in response:
            self.last_trigger_reason = "active_fold"
            return True, "active_fold"
        
        if self.executor_calls_since_fold >= self.config.executor_calls_threshold:
            self.last_trigger_reason = "executor_calls_threshold"
            return True, "executor_calls_threshold"
        
        ratio = context_length / max_length if max_length > 0 else 0
        if ratio > self.config.planner_context_threshold:
            self.last_trigger_reason = "context_overflow"
            return True, "context_overflow"
        
        # 注意：空响应不再作为压缩触发条件
        # 因为空响应可能由多种原因导致（API问题、模型特性等），与上下文长度无直接关系
        # 仅记录用于日志，不触发压缩
        if self.consecutive_empty_responses >= self.config.consecutive_empty_responses_threshold:
            print(f"[HyMem] Warning: {self.consecutive_empty_responses} consecutive empty responses (not triggering fold)")
        
        if self.consecutive_failures >= self.config.consecutive_failures_threshold:
            self.last_trigger_reason = "consecutive_failures"
            return True, "consecutive_failures"
        
        if len(memory_store.tool_call_history) >= self.config.tool_history_threshold:
            self.last_trigger_reason = "tool_history_threshold"
            return True, "tool_history_threshold"
        
        if self.planner_turns_since_fold >= self.config.planner_turns_threshold:
            self.last_trigger_reason = "planner_turns_threshold"
            return True, "planner_turns_threshold"
        
        if len(self.confidence_history) >= self.config.consecutive_low_confidence_trigger:
            recent = self.confidence_history[-self.config.consecutive_low_confidence_trigger:]
            if all(c < self.config.very_low_confidence_threshold for c in recent):
                self.last_trigger_reason = "consecutive_low_confidence"
                return True, "consecutive_low_confidence"
        
        return False, ""
    
    def should_end_branch(
        self,
        context_length: int,
        max_length: int,
        branch_turn: int
    ) -> Tuple[bool, str]:
        """判断是否需要强制结束Branch"""
        if branch_turn >= self.config.max_branch_turns:
            return True, "max_turns_reached"
        ratio = context_length / max_length if max_length > 0 else 0
        if ratio > self.config.branch_context_threshold:
            return True, "context_overflow"
        return False, ""
    
    def should_replan(self, confidence: float) -> Tuple[bool, str]:
        """判断是否需要重规划"""
        self.record_confidence(confidence)
        
        if confidence < self.config.very_low_confidence_threshold:
            return True, "very_low_confidence"
        elif confidence < self.config.low_confidence_threshold:
            return True, "low_confidence"
        
        return False, ""
    
    def get_status(self, memory_store: HyMemStore, current_turn: int) -> Dict:
        """获取完整状态"""
        return {
            "fold_count": memory_store.fold_count,
            "max_folds": self.config.max_folds,
            "can_fold": self.can_fold(memory_store, current_turn),
            "executor_calls_since_fold": self.executor_calls_since_fold,
            "planner_turns_since_fold": self.planner_turns_since_fold,
            "consecutive_empty_responses": self.consecutive_empty_responses,
            "consecutive_failures": self.consecutive_failures,
            "recent_confidences": self.confidence_history[-5:],
            "branch_count": self.branch_count,
            "last_trigger_reason": self.last_trigger_reason
        }


def estimate_context_length(messages: list, tokenizer=None) -> int:
    """估算上下文长度"""
    if tokenizer:
        try:
            text = "".join(m.get('content', '') for m in messages)
            return len(tokenizer.encode(text))
        except:
            pass
    
    total_chars = sum(len(m.get('content', '')) for m in messages)
    return int(total_chars / 3.5)


def should_compress_response(response_length: int, max_length: int = 2000) -> bool:
    """判断是否需要压缩响应"""
    return response_length > max_length


class AdaptiveTrigger(FoldTrigger):
    """自适应触发器，根据任务复杂度调整阈值"""
    
    def __init__(self, config: TriggerConfig = None):
        super().__init__(config)
        self.task_complexity: str = "medium"
        self.successful_folds: int = 0
        self.failed_after_fold: int = 0
    
    def set_task_complexity(self, complexity: str):
        """设置任务复杂度: simple, medium, complex"""
        self.task_complexity = complexity
        if complexity == "simple":
            self.config.planner_context_threshold = 0.90
            self.config.max_folds = 1
        elif complexity == "complex":
            self.config.planner_context_threshold = 0.70
            self.config.max_folds = 5
    
    def record_fold_outcome(self, success: bool):
        """记录折叠后的结果"""
        if success:
            self.successful_folds += 1
        else:
            self.failed_after_fold += 1
        
        if self.failed_after_fold > 2:
            self.config.planner_context_threshold = min(
                self.config.planner_context_threshold + 0.05, 
                0.95
            )
    
    def estimate_task_complexity(self, problem: str) -> str:
        """估算任务复杂度"""
        indicators = {
            "complex": ["multiple", "all", "every", "both", "and", "criteria", "conditions"],
            "simple": ["what is", "who is", "when did", "where is"]
        }
        
        problem_lower = problem.lower()
        
        complex_count = sum(1 for word in indicators["complex"] if word in problem_lower)
        simple_count = sum(1 for word in indicators["simple"] if word in problem_lower)
        
        if complex_count >= 3 or len(problem) > 500:
            return "complex"
        elif simple_count >= 1 and complex_count == 0:
            return "simple"
        return "medium"
