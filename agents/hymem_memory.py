"""
HyMem Memory Module

两阶段 Memory 生成流水线：
  Stage 1: Planner 在完整上下文中自压缩（Self-Compression）
  Stage 2: 辅助模型从高质量 summary 中提取结构化 Memory

三种 Memory 结构：
- Episode Memory: 关键事件、已验证事实、里程碑
- Working Memory: 当前目标、假设、下一步行动
- Tool Memory: 工具使用模式、成功/失败经验
"""

import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from openai import AsyncOpenAI

from .hymem_logger import TokenUsage
from .utils import is_context_length_exceeded_error


# ---------------------------------------------------------------------------
# HyMemStore — 混合记忆存储
# ---------------------------------------------------------------------------

@dataclass
class HyMemStore:
    """混合记忆存储"""
    episode_memory: Optional[Dict] = None
    working_memory: Optional[Dict] = None
    tool_memory: Optional[Dict] = None
    planner_summary: Optional[str] = None
    result_cache: Dict[str, Any] = field(default_factory=dict)

    fold_count: int = 0
    last_fold_turn: int = 0
    planner_turns_since_fold: int = 0
    executor_calls_this_session: int = 0

    planner_history_summary: List[Dict] = field(default_factory=list)
    tool_call_history: List[Dict] = field(default_factory=list)
    branch_history: List[Dict] = field(default_factory=list)

    # G2 — Per-task memory operation counters for §B paired comparison with
    # A-MEM (whose `amem_notes_created` / `amem_retrievals` we already
    # record). These count the structural memory operations HyMem performs
    # so reviewers can compare "memory ops × cost vs accuracy" head-to-head.
    op_counts: Dict[str, int] = field(default_factory=lambda: {
        "episode_writes": 0,    # set-events on episode_memory
        "working_writes": 0,    # set-events on working_memory
        "tool_writes": 0,       # set/merge-events on tool_memory
        "tool_call_records": 0, # add_tool_call() invocations
        "branch_records": 0,    # add_branch_result() invocations
        "retrievals": 0,        # format_*_for_context / get_branch_insights
    })

    def reset_session_counters(self):
        self.planner_turns_since_fold = 0
        self.executor_calls_this_session = 0

    def record_fold(self, turn: int):
        self.fold_count += 1
        self.last_fold_turn = turn
        self.reset_session_counters()

    def set_episode_memory(self, ep: Optional[Dict]):
        """Setter that bumps op_counts['episode_writes']. Use this from the
        agent / memory_manager instead of `store.episode_memory = ep`."""
        self.episode_memory = ep
        if ep:
            self.op_counts["episode_writes"] += 1

    def set_working_memory(self, wm: Optional[Dict]):
        self.working_memory = wm
        if wm:
            self.op_counts["working_writes"] += 1

    def set_tool_memory(self, tm: Optional[Dict]):
        self.tool_memory = tm
        if tm:
            self.op_counts["tool_writes"] += 1

    def record_retrieval(self, kind: str = "context"):
        """Bump retrieval counter — call when memory is read into a prompt."""
        self.op_counts["retrievals"] += 1

    def add_tool_call(self, tool_name: str, arguments: Dict,
                      response: str, success: bool):
        self.tool_call_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "response": response,
            "success": success,
        })
        self.op_counts["tool_call_records"] += 1

    def add_branch_result(self, description: str, prompt: str,
                          result: str, success: bool):
        self.branch_history.append({
            "description": description,
            "prompt": prompt,
            "result": result,
            "success": success,
        })
        self.op_counts["branch_records"] += 1

    def get_branch_insights(self, max_items: int = 5,
                            max_result_len: int = 500) -> List[str]:
        insights = []
        for branch in self.branch_history[-max_items:]:
            if branch.get("success"):
                r = branch["result"]
                if len(r) > max_result_len:
                    r = r[:max_result_len] + "..."
                insights.append(f"[{branch['description']}]: {r}")
        if insights:
            self.op_counts["retrievals"] += 1
        return insights

    def to_dict(self) -> Dict:
        return {
            "episode_memory": self.episode_memory,
            "working_memory": self.working_memory,
            "tool_memory": self.tool_memory,
            "planner_summary": self.planner_summary,
            "fold_count": self.fold_count,
            "last_fold_turn": self.last_fold_turn,
            "branch_count": len(self.branch_history),
            "op_counts": dict(self.op_counts),
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def parse_json_from_response(response: str) -> Optional[Dict]:
    if not response:
        return None
    try:
        return json.loads(response)
    except Exception:
        pass
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass
    brace_match = re.search(r'\{[\s\S]*\}', response)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except Exception:
            pass
    return None


def extract_summary_tag(text: str) -> Optional[str]:
    """从 <summary>...</summary> 标签中提取内容"""
    if not text:
        return None
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def aggregate_tool_call_stats(tool_call_history: List[Dict]) -> str:
    """将原始 tool_call_history 预聚合为紧凑的统计文本"""
    if not tool_call_history:
        return "No tool calls recorded."

    tool_stats: Dict[str, Dict] = {}
    for call in tool_call_history:
        name = call.get("tool_name", "unknown")
        if name not in tool_stats:
            tool_stats[name] = {
                "total": 0, "success": 0, "failed": 0,
                "effective_args": [], "failed_args": [],
            }
        stats = tool_stats[name]
        stats["total"] += 1
        args_str = json.dumps(call.get("arguments", {}), ensure_ascii=False)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        if call.get("success"):
            stats["success"] += 1
            stats["effective_args"].append(args_str)
        else:
            stats["failed"] += 1
            stats["failed_args"].append(args_str)

    lines = []
    for name, s in tool_stats.items():
        lines.append(
            f"Tool: {name} | calls: {s['total']}, "
            f"success: {s['success']}, failed: {s['failed']}"
        )
        if s["effective_args"]:
            recent = s["effective_args"][-5:]
            lines.append(f"  Effective args: {recent}")
        if s["failed_args"]:
            recent = s["failed_args"][-3:]
            lines.append(f"  Failed args: {recent}")
    return "\n".join(lines)


def _dedupe_ordered(items: list, max_count: int) -> list:
    """去重并保持顺序（最新在后），保留最近 max_count 条"""
    seen = set()
    result = []
    for item in reversed(items):
        key = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(item)
        if len(result) >= max_count:
            break
    return list(reversed(result))


# ---------------------------------------------------------------------------
# HyMemManager — Memory 管理器
# ---------------------------------------------------------------------------

class HyMemManager:
    """
    Memory 管理器 — 两阶段流水线

    Stage 1 由 HyMemAgent 调用 Planner LLM 完成（在完整上下文中自压缩）。
    Stage 2 由本类的 generate_memories_from_summary() 完成（辅助模型结构化提取）。
    """

    def __init__(self, aux_model: str = None, base_url: str = None,
                 api_key: str = None, timeout: float = 120.0):
        self.aux_model = aux_model or os.getenv(
            "HYMEM_AUX_MODEL", "deepseek-reasoner"
        )
        self.base_url = (
            base_url
            or os.getenv("HYMEM_AUX_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", None)
        )
        self.api_key = (
            api_key
            or os.getenv("HYMEM_AUX_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.timeout = float(os.getenv("HYMEM_AUX_TIMEOUT", str(timeout)))
        self._client = None
        self.token_usage = TokenUsage()
        # O6 measurement: per-fold aux extraction wall time (gather over
        # episode/working/tool). Aggregated by HyMemAgent into env.stats.
        self.fold_aux_durations: List[float] = []

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _call_llm(self, prompt: str, max_retries: int = None,
                        caller: str = "memory") -> str:
        if max_retries is None:
            max_retries = max(
                1, int(os.getenv("HYMEM_AUX_MAX_ATTEMPTS", "3"))
            )
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.aux_model,
                    messages=messages,
                    temperature=0.7,
                )
                message = resp.choices[0].message
                text = message.content or ""
                if (not text and hasattr(message, "reasoning_content")
                        and message.reasoning_content):
                    text = message.reasoning_content

                if resp.usage:
                    self.token_usage.record(
                        caller=caller,
                        prompt_tokens=resp.usage.prompt_tokens or 0,
                        completion_tokens=resp.usage.completion_tokens or 0,
                        total_tokens=resp.usage.total_tokens or 0,
                    )
                return text
            except Exception as e:
                if is_context_length_exceeded_error(e):
                    print(f"[HyMemManager] Context length exceeded, skipping retries: {str(e)[:200]}")
                    return ""
                if attempt == max_retries - 1:
                    print(f"[HyMemManager] LLM call failed: {e}")
                    return ""
                await asyncio.sleep(1 * (attempt + 1))
        return ""

    # ------------------------------------------------------------------
    # Stage 2: 从 Planner summary 中提取结构化 Memory（并行）
    # ------------------------------------------------------------------

    async def generate_memories_from_summary(
        self,
        question: str,
        planner_summary: str,
        tool_call_stats: str = "",
        branch_insights: List[str] = None,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """
        从 Planner 自压缩的 summary 中并行提取三类结构化 Memory。
        输入是 Planner 在完整上下文中生成的高质量总结，而非截断碎片。
        """
        branch_section = ""
        if branch_insights:
            branch_section = "\n\nBranch Research Findings:\n" + "\n".join(
                f"- {ins}" for ins in branch_insights
            )

        import time as _time
        _t0 = _time.time()
        # O6: gather episode/working/tool aux extraction in parallel.
        # Env override `HYMEM_FOLD_AUX_PARALLEL=0` falls back to serial for
        # A/B isolation (see efficiency rebuttal protocol).
        if os.getenv("HYMEM_FOLD_AUX_PARALLEL", "1") != "0":
            episode_memory, working_memory, tool_memory = await asyncio.gather(
                self._extract_episode_memory(question, planner_summary, branch_section),
                self._extract_working_memory(question, planner_summary, branch_section),
                self._extract_tool_memory(question, planner_summary, tool_call_stats),
            )
        else:
            episode_memory = await self._extract_episode_memory(
                question, planner_summary, branch_section)
            working_memory = await self._extract_working_memory(
                question, planner_summary, branch_section)
            tool_memory = await self._extract_tool_memory(
                question, planner_summary, tool_call_stats)
        self.fold_aux_durations.append(_time.time() - _t0)
        return episode_memory, working_memory, tool_memory

    async def _extract_episode_memory(
        self, question: str, summary: str, branch_section: str
    ) -> Optional[Dict]:
        prompt = f"""You are a memory extraction assistant. Extract episode memory (key milestones and verified facts) from the planner's comprehensive state summary.

Task: {question}

Planner's State Summary:
{summary}{branch_section}

Extract into JSON format:
```json
{{
  "task_description": "Summary of the task and overall goals",
  "key_events": [
    {{"description": "What was done or discovered", "outcome": "What was learned/achieved", "evidence": "Supporting facts or sources"}}
  ],
  "verified_facts": [
    {{"fact": "A confirmed piece of information", "source": "Where it came from", "confidence": "high/medium/low"}}
  ],
  "current_progress": "Summary of progress and remaining work"
}}
```

IMPORTANT: Preserve ALL verified facts with their sources. These are critical for continuity.
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_episode")
        return parse_json_from_response(response)

    async def _extract_working_memory(
        self, question: str, summary: str, branch_section: str
    ) -> Optional[Dict]:
        prompt = f"""You are a working memory extractor. Extract the current research state — goals, challenges, hypotheses, and next actions — from the planner's summary.

Task: {question}

Planner's State Summary:
{summary}{branch_section}

Extract into JSON format:
```json
{{
  "immediate_goal": "What the planner is currently trying to determine",
  "current_challenges": "Main obstacles or gaps in information",
  "active_hypotheses": [
    {{"hypothesis": "A candidate answer or theory", "supporting_evidence": "Evidence for it", "gaps": "What's missing"}}
  ],
  "failed_approaches": ["Strategies that didn't work — to avoid repetition"],
  "next_actions": [
    {{"type": "search/verify/analyze", "description": "Specific next action", "rationale": "Why this action"}}
  ]
}}
```

Focus on ACTIONABLE information. Prioritize what needs to be done next.
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_working")
        return parse_json_from_response(response)

    async def _extract_tool_memory(
        self, question: str, summary: str, tool_call_stats: str
    ) -> Optional[Dict]:
        prompt = f"""You are a tool experience extractor. Synthesize tool usage patterns from statistics and the planner's observations.

Task: {question}

Tool Usage Statistics:
{tool_call_stats}

Planner's Observations on Search Strategy:
{summary}

Extract into JSON format:
```json
{{
  "effective_strategies": ["Search strategies that worked well"],
  "dead_ends": ["Queries or approaches that returned nothing useful"],
  "derived_rules": ["Practical tips for future searches"],
  "useful_sources": ["URLs, docids, or data sources that contained valuable info"]
}}
```

Be concise. Focus on actionable insights, not raw statistics.
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_tool")
        return parse_json_from_response(response)

    # ------------------------------------------------------------------
    # Tool output compression
    # ------------------------------------------------------------------

    async def compress_tool_output(self, tool_output: str, task: str,
                                   max_length: int = 2000) -> str:
        if len(tool_output) <= max_length:
            return tool_output

        input_budget = min(len(tool_output), 12000)

        prompt = f"""Compress the following tool output to under {max_length} characters.
Keep ONLY information directly relevant to the task. Preserve:
- Specific names, dates, numbers, URLs
- Direct quotes that serve as evidence
- Source attributions (docid, URL)

Task: {task}

Tool Output:
{tool_output[:input_budget]}

Output the compressed information directly, no preamble."""

        response = await self._call_llm(prompt, caller="memory_compress")
        if response and len(response) <= max_length * 1.2:
            return response[:max_length]

        truncated = tool_output[:max_length]
        for sep in ["\n\n", "\n", ". ", "。"]:
            idx = truncated.rfind(sep)
            if idx > max_length * 0.7:
                return truncated[: idx + len(sep)] + "\n[Output truncated]"
        return truncated + "\n[Output truncated]"

    # ------------------------------------------------------------------
    # Tool memory merge (preserves insertion order)
    # ------------------------------------------------------------------

    def merge_tool_memories(self, old_memory: Optional[Dict],
                            new_memory: Optional[Dict]) -> Dict:
        if not old_memory:
            return new_memory or {}
        if not new_memory:
            return old_memory

        merged: Dict[str, list] = {
            "effective_strategies": [],
            "dead_ends": [],
            "derived_rules": [],
            "useful_sources": [],
        }

        for key in merged:
            combined = (
                old_memory.get(key, []) + new_memory.get(key, [])
            )
            limit = 20 if key == "useful_sources" else 10
            merged[key] = _dedupe_ordered(combined, limit)

        return merged

    # ------------------------------------------------------------------
    # Legacy: generate_all_memories (保留用于 fallback / 无 Planner 场景)
    # ------------------------------------------------------------------

    async def generate_all_memories(
        self,
        question: str,
        planner_history: str,
        recent_history: str,
        tool_call_history: List[Dict],
        executor_results: List[str] = None,
    ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
        """Legacy fallback — 直接从截断历史生成 (不推荐)"""
        episode_memory, working_memory, tool_memory = await asyncio.gather(
            self._legacy_generate_episode_memory(
                question, planner_history, executor_results
            ),
            self._legacy_generate_working_memory(question, recent_history),
            self._legacy_generate_tool_memory(question, tool_call_history),
        )
        return episode_memory, working_memory, tool_memory

    async def _legacy_generate_episode_memory(
        self, question: str, planner_history: str,
        executor_results: List[str] = None,
    ) -> Optional[Dict]:
        executor_summary = ""
        if executor_results:
            executor_summary = (
                "\n\nExecutor Results Summary:\n"
                + "\n---\n".join(executor_results[-5:])
            )
        prompt = f"""You are a memory compression assistant. Summarize the key events.

Task: {question}

Planner Reasoning History:
{planner_history}{executor_summary}

Output JSON:
```json
{{
  "task_description": "...",
  "key_events": [{{"description": "...", "outcome": "..."}}],
  "current_progress": "..."
}}
```
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_episode")
        return parse_json_from_response(response)

    async def _legacy_generate_working_memory(
        self, question: str, recent_history: str,
    ) -> Optional[Dict]:
        prompt = f"""Create a working memory snapshot.

Task: {question}

Recent Reasoning:
{recent_history}

Output JSON:
```json
{{
  "immediate_goal": "...",
  "current_challenges": "...",
  "active_hypotheses": ["..."],
  "next_actions": [{{"type": "...", "description": "..."}}]
}}
```
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_working")
        return parse_json_from_response(response)

    async def _legacy_generate_tool_memory(
        self, question: str, tool_call_history: List[Dict],
    ) -> Optional[Dict]:
        if not tool_call_history:
            return {"derived_rules": [], "useful_sources": []}
        stats_text = aggregate_tool_call_stats(tool_call_history)
        prompt = f"""Synthesize tool usage patterns.

Task: {question}

Tool Call Statistics:
{stats_text}

Output JSON:
```json
{{
  "derived_rules": ["..."],
  "useful_sources": ["..."]
}}
```
Output only the JSON."""

        response = await self._call_llm(prompt, caller="memory_tool")
        return parse_json_from_response(response)


# ---------------------------------------------------------------------------
# Memory → Context 格式化函数
# ---------------------------------------------------------------------------

def format_memory_for_context(
    memory_store: HyMemStore,
    include_summary: bool = True,
) -> str:
    """格式化 Memory 为可注入 Planner 上下文的文本

    优先使用 planner_summary（高保真自然语言总结）作为主体，
    结构化 Memory 作为补充。
    """
    # G2: count this as one memory retrieval event so we can compare with
    # A-MEM's amem_retrievals counter in §B.
    memory_store.record_retrieval("context_inject")

    sections = []

    if include_summary and memory_store.planner_summary:
        sections.append(
            "## Research State Summary\n" + memory_store.planner_summary
        )

    if memory_store.episode_memory:
        ep = memory_store.episode_memory
        ep_lines = ["## Key Findings"]
        if ep.get("verified_facts"):
            for fact in ep["verified_facts"]:
                src = fact.get("source", "unknown")
                conf = fact.get("confidence", "")
                ep_lines.append(
                    f"- {fact.get('fact', '')} (Source: {src}, Confidence: {conf})"
                )
        if ep.get("key_events"):
            ep_lines.append("\nMilestones:")
            for event in ep["key_events"]:
                ep_lines.append(
                    f"- {event.get('description', '')} → "
                    f"{event.get('outcome', '')}"
                )
        if ep.get("current_progress"):
            ep_lines.append(f"\nProgress: {ep['current_progress']}")
        sections.append("\n".join(ep_lines))

    if memory_store.working_memory:
        wm = memory_store.working_memory
        wm_lines = ["## Current Focus"]
        if wm.get("immediate_goal"):
            wm_lines.append(f"Goal: {wm['immediate_goal']}")
        if wm.get("current_challenges"):
            wm_lines.append(f"Challenges: {wm['current_challenges']}")
        if wm.get("active_hypotheses"):
            wm_lines.append("Hypotheses:")
            for h in wm["active_hypotheses"]:
                if isinstance(h, dict):
                    wm_lines.append(
                        f"- {h.get('hypothesis', '')} "
                        f"(Evidence: {h.get('supporting_evidence', 'N/A')}, "
                        f"Gaps: {h.get('gaps', 'N/A')})"
                    )
                else:
                    wm_lines.append(f"- {h}")
        if wm.get("failed_approaches"):
            wm_lines.append(
                "Failed approaches (DO NOT repeat): "
                + "; ".join(str(a) for a in wm["failed_approaches"][:5])
            )
        if wm.get("next_actions"):
            wm_lines.append("Planned Actions:")
            for action in wm["next_actions"][:5]:
                if isinstance(action, dict):
                    wm_lines.append(
                        f"- [{action.get('type', 'action')}] "
                        f"{action.get('description', '')} "
                        f"({action.get('rationale', '')})"
                    )
                else:
                    wm_lines.append(f"- {action}")
        sections.append("\n".join(wm_lines))

    if memory_store.branch_history:
        insights = memory_store.get_branch_insights(max_items=5, max_result_len=500)
        if insights:
            branch_lines = ["## Branch Research Results"]
            for insight in insights:
                branch_lines.append(f"- {insight}")
            sections.append("\n".join(branch_lines))

    if not sections:
        return ""

    return (
        "=== Memory from Previous Session ===\n\n"
        + "\n\n".join(sections)
        + "\n\n=== End Memory ===\n\n"
    )


def format_tool_memory_for_executor(tool_memory: Optional[Dict]) -> str:
    """格式化 Tool Memory 为 Executor 可用的提示"""
    if not tool_memory:
        return ""

    sections = []

    if tool_memory.get("derived_rules"):
        rules_text = "Search Tips from Previous Experience:\n"
        for rule in tool_memory["derived_rules"][:5]:
            rules_text += f"- {rule}\n"
        sections.append(rules_text)

    if tool_memory.get("effective_strategies"):
        strat_text = "Effective Strategies:\n"
        for strat in tool_memory["effective_strategies"][:5]:
            strat_text += f"- {strat}\n"
        sections.append(strat_text)

    if tool_memory.get("dead_ends"):
        dead_text = "Known Dead Ends (avoid):\n"
        for dead in tool_memory["dead_ends"][:3]:
            dead_text += f"- {dead}\n"
        sections.append(dead_text)

    if tool_memory.get("useful_sources"):
        sources_text = "Previously Useful Sources:\n"
        for source in tool_memory["useful_sources"][:5]:
            sources_text += f"- {source}\n"
        sections.append(sources_text)

    if not sections:
        return ""

    return (
        "=== Tool Experience ===\n"
        + "\n".join(sections)
        + "\n=== End Tool Experience ===\n\n"
    )
