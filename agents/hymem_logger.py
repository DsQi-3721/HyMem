"""
HyMem Unified Logging System

统一的事件日志和 Token 追踪系统，取代分散的 print/debug/Logger 三套机制。

核心组件：
- TokenUsage: 按组件累计 token 消耗
- HyMemEventLog: 记录完整的 Planner/Executor/Branch/Fold/Memory 事件链
- HyMemLogFormatter: 将结构化日志导出为可读文本或 JSON
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# ==================== 截断常量 ====================

class LogLimits:
    PLANNER_CONTENT = 5000
    EXECUTOR_RESULT = 3000
    BRANCH_RESULT = 2000
    TOOL_RESPONSE = 1000
    THINKING = 3000
    TASK_DESC = 500
    BRANCH_PROMPT = 500


# ==================== Token 追踪 ====================

@dataclass
class TokenUsage:
    """按组件累计 token 消耗"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    planner_prompt_tokens: int = 0
    planner_completion_tokens: int = 0
    executor_prompt_tokens: int = 0
    executor_completion_tokens: int = 0
    branch_prompt_tokens: int = 0
    branch_completion_tokens: int = 0
    memory_prompt_tokens: int = 0
    memory_completion_tokens: int = 0

    call_count: int = 0
    calls: List[Dict] = field(default_factory=list)

    def record(self, caller: str, prompt_tokens: int, completion_tokens: int, total_tokens: int = 0):
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.call_count += 1

        component = self._classify_component(caller)
        if component == "planner":
            self.planner_prompt_tokens += prompt_tokens
            self.planner_completion_tokens += completion_tokens
        elif component == "executor":
            self.executor_prompt_tokens += prompt_tokens
            self.executor_completion_tokens += completion_tokens
        elif component == "branch":
            self.branch_prompt_tokens += prompt_tokens
            self.branch_completion_tokens += completion_tokens
        elif component == "memory":
            self.memory_prompt_tokens += prompt_tokens
            self.memory_completion_tokens += completion_tokens

        self.calls.append({
            "caller": caller,
            "component": component,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "timestamp": time.time(),
        })

    @staticmethod
    def _classify_component(caller: str) -> str:
        c = caller.lower()
        if "planner" in c:
            return "planner"
        if "executor" in c:
            return "executor"
        if "branch" in c:
            return "branch"
        if "memory" in c or "episode" in c or "working" in c or "tool_mem" in c or "compress" in c:
            return "memory"
        return "other"

    def to_dict(self) -> Dict:
        return {
            "total": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "call_count": self.call_count,
            },
            "by_component": {
                "planner": {
                    "prompt_tokens": self.planner_prompt_tokens,
                    "completion_tokens": self.planner_completion_tokens,
                    "total_tokens": self.planner_prompt_tokens + self.planner_completion_tokens,
                },
                "executor": {
                    "prompt_tokens": self.executor_prompt_tokens,
                    "completion_tokens": self.executor_completion_tokens,
                    "total_tokens": self.executor_prompt_tokens + self.executor_completion_tokens,
                },
                "branch": {
                    "prompt_tokens": self.branch_prompt_tokens,
                    "completion_tokens": self.branch_completion_tokens,
                    "total_tokens": self.branch_prompt_tokens + self.branch_completion_tokens,
                },
                "memory": {
                    "prompt_tokens": self.memory_prompt_tokens,
                    "completion_tokens": self.memory_completion_tokens,
                    "total_tokens": self.memory_prompt_tokens + self.memory_completion_tokens,
                },
            },
            "calls": self.calls,
        }

    def summary_str(self) -> str:
        lines = [
            f"Token Usage Summary (total {self.call_count} LLM calls):",
            f"  Total:    prompt={self.prompt_tokens:,}  completion={self.completion_tokens:,}  sum={self.total_tokens:,}",
            f"  Planner:  prompt={self.planner_prompt_tokens:,}  completion={self.planner_completion_tokens:,}",
            f"  Executor: prompt={self.executor_prompt_tokens:,}  completion={self.executor_completion_tokens:,}",
            f"  Branch:   prompt={self.branch_prompt_tokens:,}  completion={self.branch_completion_tokens:,}",
            f"  Memory:   prompt={self.memory_prompt_tokens:,}  completion={self.memory_completion_tokens:,}",
        ]
        return "\n".join(lines)


# ==================== Per-Component Latency 追踪 (Step B2) ====================

@dataclass
class LatencyTracker:
    """Wall-clock latency per agent component.

    Reviewer Bh2U W2 asked for "efficiency metrics to quantify the trade-off
    between performance gains and resource cost." session_time alone hides
    *where* the time is spent. This tracker accumulates seconds spent inside
    LLM calls grouped by component (planner / executor / branch / memory /
    other), so reports can attribute latency to the four-module architecture.

    Pure observation — does not change agent behavior or LLM costs.
    """

    planner_seconds: float = 0.0
    executor_seconds: float = 0.0
    branch_seconds: float = 0.0
    memory_seconds: float = 0.0
    other_seconds: float = 0.0
    by_caller: Dict[str, float] = field(default_factory=dict)
    call_count: int = 0

    def add(self, caller: str, elapsed_seconds: float) -> None:
        """Record a single LLM (or aux) call latency, attributed by caller."""
        if elapsed_seconds < 0:
            return
        component = TokenUsage._classify_component(caller)
        if component == "planner":
            self.planner_seconds += elapsed_seconds
        elif component == "executor":
            self.executor_seconds += elapsed_seconds
        elif component == "branch":
            self.branch_seconds += elapsed_seconds
        elif component == "memory":
            self.memory_seconds += elapsed_seconds
        else:
            self.other_seconds += elapsed_seconds
        self.by_caller[caller] = self.by_caller.get(caller, 0.0) + elapsed_seconds
        self.call_count += 1

    @property
    def total_seconds(self) -> float:
        return (
            self.planner_seconds + self.executor_seconds + self.branch_seconds
            + self.memory_seconds + self.other_seconds
        )

    def to_dict(self) -> Dict:
        return {
            "planner": round(self.planner_seconds, 3),
            "executor": round(self.executor_seconds, 3),
            "branch": round(self.branch_seconds, 3),
            "memory": round(self.memory_seconds, 3),
            "other": round(self.other_seconds, 3),
            "total": round(self.total_seconds, 3),
            "call_count": self.call_count,
            "by_caller": {k: round(v, 3) for k, v in self.by_caller.items()},
        }


# ==================== 事件日志 ====================

class HyMemEventLog:
    """
    HyMem 统一事件日志。

    记录完整的 Planner / Executor / Branch / Fold / Memory / Replan 事件链，
    每条事件携带时间戳和当前 token 消耗快照。
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.token_usage = TokenUsage()
        self.latency = LatencyTracker()  # Step B2 — per-component wall-clock
        self.session_start_time: float = time.time()

    def _base_entry(self, event_type: str, **kwargs) -> Dict[str, Any]:
        entry = {
            "event": event_type,
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self.session_start_time, 2),
            "token_snapshot": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
            },
        }
        entry.update(kwargs)
        return entry

    # ---------- Session ----------

    def log_session_start(self, problem: str, config: Dict = None):
        self.session_start_time = time.time()
        self.events.append(self._base_entry(
            "session_start",
            problem=problem[:LogLimits.PLANNER_CONTENT],
            config=config or {},
        ))

    def log_session_end(self, answer: str, confidence: float, stats: Dict = None):
        self.events.append(self._base_entry(
            "session_end",
            answer=answer,
            confidence=confidence,
            session_time=round(time.time() - self.session_start_time, 2),
            stats=stats or {},
        ))

    # ---------- LLM Call ----------

    def log_llm_call(self, caller: str, prompt_tokens: int, completion_tokens: int,
                     total_tokens: int = 0, input_length: int = 0, output_length: int = 0):
        """记录单次 LLM 调用并累加 token。"""
        self.token_usage.record(caller, prompt_tokens, completion_tokens, total_tokens)
        self.events.append(self._base_entry(
            "llm_call",
            caller=caller,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens or (prompt_tokens + completion_tokens),
            input_chars=input_length,
            output_chars=output_length,
        ))

    # ---------- Planner ----------

    def log_planner_turn(self, turn: int, content: str, thinking: str = None,
                         context_length: int = 0):
        self.events.append(self._base_entry(
            "planner_turn",
            turn=turn,
            content=content[:LogLimits.PLANNER_CONTENT],
            thinking=(thinking[:LogLimits.THINKING] if thinking else None),
            context_length=context_length,
        ))

    # ---------- Executor ----------

    def log_executor_start(self, turn: int, task: str, parent: str = "planner"):
        self.events.append(self._base_entry(
            "executor_start",
            turn=turn,
            task=task[:LogLimits.TASK_DESC],
            parent=parent,
        ))

    def log_executor_step(self, turn: int, executor_turn: int,
                          step_type: str, content: str, output: str = ""):
        self.events.append(self._base_entry(
            "executor_step",
            turn=turn,
            executor_turn=executor_turn,
            step_type=step_type,
            content=content[:LogLimits.EXECUTOR_RESULT],
            output=output[:LogLimits.EXECUTOR_RESULT],
        ))

    def log_executor_end(self, turn: int, result: str, total_executor_turns: int):
        self.events.append(self._base_entry(
            "executor_end",
            turn=turn,
            result=result[:LogLimits.EXECUTOR_RESULT],
            total_executor_turns=total_executor_turns,
        ))

    # ---------- Branch ----------

    def log_branch_start(self, turn: int, description: str, prompt: str, auto_routed: bool = False):
        self.events.append(self._base_entry(
            "branch_start",
            turn=turn,
            description=description[:LogLimits.TASK_DESC],
            prompt=prompt[:LogLimits.BRANCH_PROMPT],
            auto_routed=auto_routed,
        ))

    def log_branch_step(self, turn: int, branch_turn: int,
                        step_type: str, content: str, output: str = ""):
        self.events.append(self._base_entry(
            "branch_step",
            turn=turn,
            branch_turn=branch_turn,
            step_type=step_type,
            content=content[:LogLimits.BRANCH_RESULT],
            output=output[:LogLimits.BRANCH_RESULT],
        ))

    def log_branch_end(self, turn: int, description: str, result: str, success: bool):
        self.events.append(self._base_entry(
            "branch_end",
            turn=turn,
            description=description[:LogLimits.TASK_DESC],
            result=result[:LogLimits.BRANCH_RESULT],
            success=success,
        ))

    # ---------- Fold ----------

    def log_fold_start(self, turn: int, context_tokens_before: int, message_count_before: int):
        self.events.append(self._base_entry(
            "fold_start",
            turn=turn,
            context_tokens_before=context_tokens_before,
            message_count_before=message_count_before,
        ))

    def log_fold_end(self, turn: int, context_tokens_after: int, message_count_after: int,
                     has_episode: bool, has_working: bool, has_tool: bool):
        self.events.append(self._base_entry(
            "fold_end",
            turn=turn,
            context_tokens_after=context_tokens_after,
            message_count_after=message_count_after,
            has_episode=has_episode,
            has_working=has_working,
            has_tool=has_tool,
        ))

    # ---------- Replan ----------

    def log_replan(self, turn: int, reason: str, previous_answer: str, confidence: float):
        self.events.append(self._base_entry(
            "replan",
            turn=turn,
            reason=reason,
            previous_answer=previous_answer[:LogLimits.TASK_DESC],
            confidence=confidence,
        ))

    # ---------- Query ----------

    def get_events(self, event_type: str = None) -> List[Dict]:
        if event_type is None:
            return self.events
        return [e for e in self.events if e["event"] == event_type]

    def to_dict(self) -> Dict:
        return {
            "session_time": round(time.time() - self.session_start_time, 2),
            "token_usage": self.token_usage.to_dict(),
            "event_count": len(self.events),
            "events": self.events,
        }


# ==================== 格式化输出 ====================

class HyMemLogFormatter:
    """将 HyMemEventLog 格式化为可读文本或 JSON"""

    SEPARATOR_HEAVY = "=" * 80
    SEPARATOR_MEDIUM = "-" * 60
    SEPARATOR_LIGHT = "·" * 40

    @classmethod
    def format_task_log(cls, instance_id: str, problem: str,
                        event_log: HyMemEventLog,
                        answer: str, confidence: float, score: float,
                        stats: Dict = None, extra: Dict = None) -> str:
        """生成单个任务的完整可读日志"""
        stats = stats or {}
        extra = extra or {}
        parts = []

        # Header
        parts.append(cls.SEPARATOR_HEAVY)
        parts.append(f"# HyMem Task: {instance_id}")
        parts.append(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        parts.append(cls.SEPARATOR_HEAVY)
        parts.append("")

        # Problem
        parts.append("== PROBLEM ==")
        parts.append(problem[:LogLimits.PLANNER_CONTENT])
        parts.append("")

        # Extra (e.g. ground truth, file info for GAIA)
        if extra.get("ground_truth"):
            parts.append(f"Ground Truth: {extra['ground_truth']}")
        if extra.get("file_name"):
            parts.append(f"File: {extra['file_name']}")
        if extra:
            parts.append("")

        # Execution Trace
        parts.append(cls.SEPARATOR_HEAVY)
        parts.append("# EXECUTION TRACE")
        parts.append(cls.SEPARATOR_HEAVY)

        for ev in event_log.events:
            formatted = cls._format_event(ev)
            if formatted:
                parts.append(formatted)

        # Result
        parts.append("")
        parts.append(cls.SEPARATOR_HEAVY)
        parts.append("# FINAL RESULT")
        parts.append(cls.SEPARATOR_HEAVY)
        parts.append(f"Answer: {answer if answer else '(none)'}")
        parts.append(f"Confidence: {confidence:.2f}")
        parts.append(f"Score: {score}")
        parts.append("")

        # Statistics
        parts.append(cls.SEPARATOR_MEDIUM)
        parts.append("STATISTICS")
        parts.append(cls.SEPARATOR_MEDIUM)
        parts.append(f"  Planner Turns:  {stats.get('planner_turns', 0)}")
        parts.append(f"  Executor Calls: {stats.get('executor_calls', 0)}")
        parts.append(f"  Branch Calls:   {stats.get('branch_calls', 0)}")
        parts.append(f"  Branch Success: {stats.get('branch_success', 0)}")
        parts.append(f"  Fold Count:     {stats.get('fold_count', 0)}")
        parts.append(f"  Replan Count:   {stats.get('replan_count', 0)}")
        parts.append(f"  Tool Calls:     {stats.get('tool_calls_total', 0)}")
        parts.append(f"  Session Time:   {stats.get('session_time', 0):.2f}s")
        parts.append("")

        # Token Usage
        parts.append(cls.SEPARATOR_MEDIUM)
        parts.append("TOKEN USAGE")
        parts.append(cls.SEPARATOR_MEDIUM)
        parts.append(event_log.token_usage.summary_str())
        parts.append("")

        parts.append(cls.SEPARATOR_HEAVY)
        return "\n".join(parts)

    @classmethod
    def _format_event(cls, ev: Dict) -> Optional[str]:
        etype = ev.get("event", "")
        ts = f"[+{ev.get('elapsed_s', 0):.1f}s]"
        tok = ev.get("token_snapshot", {})
        tok_str = f"[tokens: {tok.get('total_tokens', 0):,}]"

        if etype == "planner_turn":
            turn = ev.get("turn", "?")
            ctx = ev.get("context_length", 0)
            content = ev.get("content", "")
            thinking = ev.get("thinking")
            lines = [
                f"\n{cls.SEPARATOR_MEDIUM}",
                f"PLANNER TURN {turn}  {ts} {tok_str}  context={ctx:,}",
                cls.SEPARATOR_MEDIUM,
            ]
            if thinking:
                lines.append(f"[Thinking]: {thinking[:2000]}")
                lines.append("")
            lines.append(f"[Output]: {content[:3000]}")
            return "\n".join(lines)

        if etype == "executor_start":
            return f"\n{cls.SEPARATOR_LIGHT}\n  EXECUTOR START (turn {ev.get('turn')}) {ts}: {ev.get('task', '')}"

        if etype == "executor_step":
            et = ev.get("executor_turn", "?")
            st = ev.get("step_type", "")
            return f"  [Executor T{et}] {st}: {ev.get('content', '')[:500]}"

        if etype == "executor_end":
            et = ev.get("total_executor_turns", 0)
            return f"  EXECUTOR END (turns={et}) {ts} {tok_str}\n  Result: {ev.get('result', '')[:1000]}"

        if etype == "branch_start":
            auto = " [AUTO-ROUTED]" if ev.get("auto_routed") else ""
            return f"\n{cls.SEPARATOR_LIGHT}\n  BRANCH START{auto} (turn {ev.get('turn')}) {ts}: {ev.get('description', '')}"

        if etype == "branch_step":
            bt = ev.get("branch_turn", "?")
            st = ev.get("step_type", "")
            return f"  [Branch T{bt}] {st}: {ev.get('content', '')[:500]}"

        if etype == "branch_end":
            status = "SUCCESS" if ev.get("success") else "INCOMPLETE"
            return f"  BRANCH END [{status}] {ts} {tok_str}\n  Result: {ev.get('result', '')[:1000]}"

        if etype == "fold_start":
            return f"\n{cls.SEPARATOR_LIGHT}\n  FOLD START (turn {ev.get('turn')}) {ts}  tokens_before={ev.get('context_tokens_before', 0):,}"

        if etype == "fold_end":
            return (f"  FOLD END {ts} {tok_str}  tokens_after={ev.get('context_tokens_after', 0):,}"
                    f"  episode={ev.get('has_episode')} working={ev.get('has_working')} tool={ev.get('has_tool')}")

        if etype == "replan":
            return (f"\n  REPLAN (turn {ev.get('turn')}) {ts}  reason={ev.get('reason')}"
                    f"  confidence={ev.get('confidence', 0):.2f}")

        if etype == "llm_call":
            caller = ev.get("caller", "?")
            pt = ev.get("prompt_tokens", 0)
            ct = ev.get("completion_tokens", 0)
            return f"  [LLM] {caller}: prompt={pt:,} completion={ct:,}"

        return None


# ==================== 评测日志聚合 ====================

class HyMemEvalLogger:
    """
    评测级别的日志聚合器（替代分散在各 eval 脚本中的 HyMemLogger / GAIALogger）。

    每个任务通过 add_result() 添加，最终通过 save_all / save_summary 保存。
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []

    def add_result(self, result: Dict):
        self.results.append(result)

    # backward compatibility alias
    add_log = add_result

    def save_task_log(self, instance_id: str, content: str):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        log_file = self.log_dir / f"hymem_task_{safe_id}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(content)

    def save_task_json(self, instance_id: str, data: Dict):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        log_file = self.log_dir / f"hymem_task_{safe_id}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def save_task_trace(self, instance_id: str, trace: Dict, prefix: str = "task"):
        """Save the full machine-readable trace for one task.

        Mirrors GAIALogger.save_task_trace (scripts/eval_gaia.py) so the
        same --resume scan logic works for any benchmark. Filename is
        ``{prefix}_{safe_id}_trace.json`` — caller picks the prefix
        ('gaia' / 'bcplus' / etc.) so resume can glob deterministically.
        """
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        path = self.log_dir / f"{prefix}_{safe_id}_trace.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

    def save_all(self, filename: str = None) -> Path:
        if filename is None:
            filename = f"hymem_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_file = self.log_dir / filename
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        return log_file

    def save_summary(self, filename: str = None) -> Path:
        if filename is None:
            filename = f"hymem_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        summary_file = self.log_dir / filename
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("HyMem Evaluation Summary\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            total = len(self.results)
            success = sum(1 for r in self.results if r.get("status") == "success")
            scores = [r.get("score", 0) for r in self.results]
            f.write(f"Total Tasks:  {total}\n")
            f.write(f"Success:      {success}\n")
            if scores:
                f.write(f"Avg Score:    {np.mean(scores):.4f}\n")
            f.write("\n")

            # Aggregate stats
            stat_keys = [
                ("planner_turns", "Avg Planner Turns"),
                ("executor_calls", "Avg Executor Calls"),
                ("branch_calls", "Avg Branch Calls"),
                ("fold_count", "Avg Fold Count"),
                ("tool_calls_total", "Avg Tool Calls"),
                ("session_time", "Avg Session Time (s)"),
            ]
            for key, label in stat_keys:
                vals = [r.get(key, 0) for r in self.results]
                if vals:
                    f.write(f"  {label}: {np.mean(vals):.2f}\n")

            # Token usage aggregate
            token_keys = [
                ("total_prompt_tokens", "Avg Prompt Tokens"),
                ("total_completion_tokens", "Avg Completion Tokens"),
                ("total_tokens", "Avg Total Tokens"),
            ]
            f.write("\nToken Usage:\n")
            for key, label in token_keys:
                vals = [r.get(key, 0) for r in self.results]
                if any(v > 0 for v in vals):
                    f.write(f"  {label}: {np.mean(vals):,.0f}\n")

            f.write(f"\n{'=' * 80}\n")

            for r in self.results:
                f.write(f"\n{'-' * 60}\n")
                f.write(f"Instance:   {r.get('instance_id', 'unknown')}\n")
                f.write(f"Score:      {r.get('score', 0)}\n")
                f.write(f"Status:     {r.get('status', 'unknown')}\n")
                f.write(f"Confidence: {r.get('confidence', 0):.2f}\n")
                f.write(f"Tokens:     {r.get('total_tokens', 0):,}\n")

        return summary_file
