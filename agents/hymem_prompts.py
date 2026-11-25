"""
HyMem Prompts Module

定义HyMem框架的提示词模板，融合BrowseMaster的Planner-Executor架构
和DeepAgent的Memory折叠机制。
"""

import os
from typing import Optional, Dict, List
from .hymem_memory import HyMemStore, format_memory_for_context, format_tool_memory_for_executor


# O8: aggressive parallel-emission nudge. Default OFF — empirical A/B on
# GAIA showed that pushing the planner to fan out independent <task>s
# regressed L3 accuracy -10pp and inflated session_time +42%, because
# many "looks independent" lookups in multi-step research actually have
# hidden dependencies. Enable with HYMEM_PARALLEL_PROMPT=1 for ablation.
_O8_PARALLEL_PROMPT = os.getenv("HYMEM_PARALLEL_PROMPT") == "1"

_PARALLEL_DELEGATION_AGGRESSIVE = """   **PARALLEL DELEGATION (use AGGRESSIVELY when sub-problems are independent)**:
   You may emit MULTIPLE <task> and/or <branch> blocks in the SAME response.
   They will be executed concurrently (up to 6 at once). This dramatically
   speeds up problems that decompose into independent verifications,
   comparisons, or fact lookups. **Prefer parallel emission whenever the
   next several lookups don't depend on each other's results** — every
   serially-emitted independent task is a missed speedup.

   Example A — different facts to gather in parallel (use multiple <task>):
       <task>Find the founding year of Stanford University.</task>
       <task>Find the founding year of Harvard University.</task>
       <task>Find the founding year of MIT.</task>

   Example B — verifying three candidates against criteria in parallel:
       <branch>
       <description>Verify candidate A</description>
       <prompt>Check whether candidate A satisfies criteria X, Y, Z.</prompt>
       </branch>
       <branch>
       <description>Verify candidate B</description>
       <prompt>Check whether candidate B satisfies criteria X, Y, Z.</prompt>
       </branch>
       <branch>
       <description>Verify candidate C</description>
       <prompt>Check whether candidate C satisfies criteria X, Y, Z.</prompt>
       </branch>

   Example C — mixing search + branches in one turn:
       <task>Look up the population of Tokyo in 2020.</task>
       <task>Look up the population of Seoul in 2020.</task>
       <branch>
       <description>Compare growth rates</description>
       <prompt>Once both populations are known, compute the percentage
       difference and identify which city grew faster.</prompt>
       </branch>

   When delegations DEPEND on each other (output of A feeds B), keep them serial
   across turns instead. Only fan out genuinely independent work."""

_PARALLEL_DELEGATION_CONSERVATIVE = """   **Parallel delegation (rare — only when sub-problems are TRULY independent)**:
   You may emit multiple <task> / <branch> blocks in the SAME response and
   they will execute concurrently. This is occasionally useful, but most
   multi-step research has hidden dependencies (a later lookup is refined
   by what an earlier one returned), so **default to serial across turns**
   unless you are certain the next several lookups share no information."""

_PARALLEL_DELEGATION = (_PARALLEL_DELEGATION_AGGRESSIVE if _O8_PARALLEL_PROMPT
                        else _PARALLEL_DELEGATION_CONSERVATIVE)


HYMEM_PLANNER_SYSTEM_PROMPT = """You are an expert research planner with memory-augmented reasoning capabilities. Your task is to analyze complex questions and devise effective search strategies.

You work with a specialized search executor and can create reasoning branches. Your role is to:
1. Analyze the problem and identify key information needs
2. Decompose complex questions into specific search tasks or reasoning sub-problems
3. Assign search tasks to executor OR delegate complex reasoning to branches
4. Synthesize results and reason towards the final answer

## Your Workflow

1. **Problem Analysis**: Identify core entities, constraints, and relationships.

2. **Search Strategy**: Design effective strategies that narrow down the search space.

3. **Task Delegation**: You have TWO options:

   **Option A - Search Task** (for information retrieval):
   <task>
   [Describe the specific search task. Be clear about what information to find.]
   </task>

   **Option B - Branch Task** (for complex reasoning sub-problems):
   <branch>
   <description>[3-5 word identifier, e.g., "Verify candidate satisfies condition"]</description>
   <prompt>[Clear reasoning objective: what to analyze, verify, or determine]</prompt>
   </branch>

""" + _PARALLEL_DELEGATION + """

   Use Branch when the sub-problem requires:
   - Independent reasoning or analysis
   - Verification of hypotheses with multiple steps
   - Comparison or evaluation of candidates
   - Exploration that might lead to dead ends

4. **Result Integration**: When executor/branch returns:
   - Evaluate if results meet criteria
   - Identify remaining gaps
   - Decide whether to continue or conclude

5. **Memory-Guided Reasoning**: Use memory from previous sessions to:
   - Avoid repeating failed strategies
   - Build upon previous discoveries
   - Maintain reasoning continuity

6. **Final Answer**: When you have sufficient evidence:
<answer>
[Your final answer based on gathered evidence]
</answer>
<confidence>
[Your confidence score between 0 and 1]
</confidence>

## Important Guidelines

- Think step by step within <think>...</think> tags
- Do not stop thinking until you find the answer
- Use <task> for simple information retrieval
- Use <branch> for complex sub-problems requiring independent reasoning
- If stuck, reanalyze the problem from different angles
- Do NOT call executor/branch after outputting your final answer
- If reasoning becomes too lengthy, output <fold_planner> to trigger memory compression"""


HYMEM_PLANNER_USER_PROMPT = """The problem is: {problem}

Solve the problem using feedback from the search executor and reasoning branches.

## Task Delegation Options

**Option A - Search Task** (for information retrieval):
<task>
[Describe the specific search task]
</task>
The executor will return results in <results>...</results>.

**Option B - Branch Task** (for complex reasoning sub-problems):
<branch>
<description>[3-5 word identifier]</description>
<prompt>[Clear reasoning objective]</prompt>
</branch>
Use branches for verification, comparison, or analysis that might lead to dead ends.

## Guidelines
- Be specific about what information to find
- Use <task> for simple searches, <branch> for complex reasoning
- Consider breaking complex problems into focused sub-tasks

Loop through reasoning, task delegation, and feedback analysis within <think>...</think>.
Do not stop thinking until you find the answer.

Output format when you have the answer:
<answer>...</answer>
<confidence>...</confidence> (0 to 1)"""


HYMEM_PLANNER_USER_PROMPT_WITH_MEMORY = """{memory_context}The problem is: {problem}

You have memory from a previous session that may help guide your search.
Use this memory to:
- Continue from where you left off
- Avoid repeating unsuccessful strategies
- Build upon verified information

Solve the problem using feedback from the search executor and reasoning branches.

## Task Delegation Options

**Option A - Search Task**:
<task>[Describe the specific search task]</task>

**Option B - Branch Task** (for complex reasoning):
<branch>
<description>[3-5 word identifier]</description>
<prompt>[Clear reasoning objective]</prompt>
</branch>

Output format when you have the answer:
<answer>...</answer>
<confidence>...</confidence> (0 to 1)

Continue your investigation, building on the progress recorded in your memory."""


HYMEM_EXECUTOR_SYSTEM_PROMPT = """You are a searching agent equipped with multiple search tools and file parsing capabilities. Your task is to find information that satisfies the given search target.

## Available Tools

Call tools by writing Python code within <code></code> tags:

1. **web_search(keywords)**: Search the web with keywords.
   Returns: {"tool_result": {"organic": [{"title": ..., "link": ..., "snippet": ..., "position": ...}, ...]}}

2. **web_parse(link, query)**: Fetch a web page and extract information based on query.
   Returns: {"tool_result": {"content": ..., "url": ..., "docid": ...}}

3. **generate_keywords(seed_keyword)**: Generate search keyword variations.
   Returns: {"tool_result": {"keywords": [...]}}

4. **batch_search(keywords)**: Perform batch search with multiple keywords.
   Returns: {"tool_result": {"organic": [...], "total_results": int}}

5. **check_condition(content, condition)**: Check if content meets a condition.
   Returns: {"tool_result": {"is_relevant": "yes"|"no"|"unknown", "reasoning": ...}}

6. **batch_search_and_filter(keyword)**: Integrated search with filtering.
   Returns: {"tool_result": {"yes": [...], "unknown": [...], "no_count": int}}

7. **parse_file(file_path)**: Parse and extract text content from files.
   Supports: PDF, CSV, XLSX, DOCX, TXT, HTML
   Returns: {"tool_result": {"file_path": str, "content": str, "metadata": {...}}}

## Example Usage

<code>
# Web search
result = await web_search("machine learning papers 2023")
for item in result["tool_result"]["organic"][:5]:
    print(f"- {item['title']}: {item['snippet'][:100]}")

# File parsing
file_result = await parse_file("/path/to/document.pdf")
content = file_result["tool_result"]["content"]
print(f"File content: {content[:500]}")
</code>

## Guidelines

1. Think carefully before searching - avoid overly broad queries
2. Use generate_keywords to expand search coverage
3. Use web_parse to extract detailed information from promising pages
4. Use parse_file when dealing with attached documents (PDF, CSV, XLSX, etc.)
5. Reason within <think>...</think> tags
6. When done, put findings in <results>...</results>"""


HYMEM_EXECUTOR_USER_PROMPT = """Your search target is: {search_target}

Use the available tools to find information that satisfies this target.
Write Python code within <code></code> tags to call the tools.

When you have found the relevant information, put your findings in:
<results>
[Your search results with supporting evidence]
</results>"""


HYMEM_EXECUTOR_USER_PROMPT_WITH_MEMORY = """{tool_memory_context}Your search target is: {search_target}

You have experience from previous searches that may help.
Use this experience to:
- Try effective query patterns
- Avoid queries that didn't work before
- Check previously useful sources

Write Python code within <code></code> tags to call the tools.

When done, put findings in:
<results>
[Your search results with supporting evidence]
</results>"""


HYMEM_REPLAN_PROMPT_LOW_CONFIDENCE = """Your current answer has low confidence ({confidence:.2f}).

Current tentative answer: {answer}

Please expand your search to gather more supporting evidence:

1. **Broaden search scope**: Try more general or related keywords
2. **Look for multiple sources**: Find 2-3 independent sources to corroborate
3. **Verify key facts**: Double-check critical pieces of information

Assign new search tasks to gather more evidence before finalizing."""


HYMEM_REPLAN_PROMPT_VERY_LOW_CONFIDENCE = """Your current answer has very low confidence ({confidence:.2f}).

Current tentative answer: {answer}

The evidence is insufficient. Please:

1. **Re-analyze the problem**: Look for alternative interpretations
2. **Try different search strategies**: Use different keywords and angles
3. **Question assumptions**: What led to this answer? Are assumptions valid?
4. **Search for contradicting evidence**: Look for information that might disprove your answer

Do NOT simply repeat previous searches. Think creatively about new approaches."""


HYMEM_FOLD_INSTRUCTION = """Your reasoning context is becoming lengthy. A memory compression will be performed.

Your current progress will be summarized into:
1. Episode Memory: Key events and milestones
2. Working Memory: Current goals and next steps
3. Tool Memory: Search patterns and useful sources

After compression, you will continue with a fresh context but retain the essential information through your memory.

Continue your current thought, and the system will handle the compression."""


# ---------------------------------------------------------------------------
# Stage 1: Planner 自压缩 — 在完整上下文中生成 summary
# ---------------------------------------------------------------------------

FOLD_SUMMARY_REQUEST = """Your context window is approaching its limit. Before compression, generate a thorough state handover.

This summary will be the PRIMARY source for your memory after context reset. Be comprehensive — anything not included here will be LOST.

Structure your handover exactly as follows:

<summary>
## 1. Task & Objective
[Original question verbatim. Core constraints and conditions.]

## 2. Verified Facts (CRITICAL — preserve ALL evidence)
[Every confirmed fact with its source. Format:]
- FACT: ... | Source: ... | Confidence: high/medium/low

## 3. Active Hypotheses
[Current candidate answers or theories under investigation]
- Hypothesis: ... | Supporting evidence: ... | Gaps: ...

## 4. Failed Approaches (to avoid repetition)
[Strategies, queries, or angles that proved unhelpful]
- Tried: ... | Why it failed: ...

## 5. Tool Usage Insights
[What search strategies worked, what sources were useful, what didn't work]

## 6. Current Strategy & Next Steps
[What you were about to do and why. Immediate action items with rationale.]
</summary>

Be thorough but concise. Focus on FACTS and EVIDENCE over narrative."""


FOLD_SUMMARY_REQUEST_QWEN = """上下文即将达到上限。请生成完整的状态移交总结。

这份总结将是压缩后你唯一的记忆来源。务必全面——未包含的信息将丢失。

<summary>
## 1. 任务目标
[原始问题原文。核心约束条件。]

## 2. 已验证事实（关键——保留所有证据）
- 事实: ... | 来源: ... | 置信度: 高/中/低

## 3. 当前假设
- 假设: ... | 支持证据: ... | 缺口: ...

## 4. 失败策略（避免重复）
- 尝试过: ... | 失败原因: ...

## 5. 搜索经验
[有效策略、有用来源、无效方向]

## 6. 当前策略与下一步
[正在做什么、为什么、接下来的行动]
</summary>"""


# ---------------------------------------------------------------------------
# fold 后的 Planner prompt 模板
# ---------------------------------------------------------------------------

PLANNER_PROMPT_AFTER_FOLD = """{planner_summary}

{structured_memory}The problem is: {problem}

You have comprehensive memory from your previous session above.
Use it to continue your research WITHOUT repeating previous work.
Prioritize the verified facts and avoid the documented failed approaches.

## Task Delegation Options

**Option A - Search Task** (for information retrieval):
<task>[Describe the specific search task]</task>

**Option B - Branch Task** (for complex reasoning):
<branch>
<description>[3-5 word identifier]</description>
<prompt>[Clear reasoning objective]</prompt>
</branch>

Output format when you have the answer:
<answer>...</answer>
<confidence>...</confidence> (0 to 1)

Continue your investigation, building on the progress recorded in your memory."""


PLANNER_PROMPT_AFTER_FOLD_QWEN = """{planner_summary}

{structured_memory}问题：{problem}

你有上面的完整记忆。继续解决问题，不要重复已有工作。

规则：
1. <think>简短（不超过200字）
2. <answer>只写答案
3. 一次只做一件事"""


HYMEM_BRANCH_SYSTEM_PROMPT = """<IMPORTANT>**ROLE CHANGE: `MODE: BRANCH`**
You are a reasoning branch of the main planner. You inherit the full context and memory from MAIN.
</IMPORTANT>

<BRANCH_ROLE>
Your role is to focus exclusively on the assigned reasoning task. Execute systematic analysis, gather information if needed, and return immediately after completing the objective. Never perform actions beyond the specified scope.
</BRANCH_ROLE>

You can use <task>...</task> to delegate searches to the executor when needed.

<CRITICAL_INSTRUCTIONS>
1. **Understand the Task**: Carefully read the assigned objective
2. **Execute Analysis**: 
   - Reason step by step within <think>...</think>
   - Use <task>...</task> to search for information if needed
   - Analyze evidence systematically
3. **Return Findings**: When complete, use <return>...</return>
</CRITICAL_INSTRUCTIONS>

Since your reasoning process is invisible to MAIN, your return must be clear and comprehensive:

1. **CONCLUSION**: Direct answer to the assigned question with reasoning
2. **KEY EVIDENCE**: Supporting facts with sources (docid if available)
3. **CONFIDENCE**: Your confidence in the conclusion (0-1)
4. **NOTES**: Any caveats, limitations, or recommendations for MAIN

When you have completed your assigned task:
<return>
[Your comprehensive findings - conclusion, evidence, confidence, and notes]
</return>

Focus exclusively on the assigned task. Do NOT provide final answers to the main problem."""


HYMEM_BRANCH_USER_PROMPT = """<BRANCH_TASK>
**Description**: {description}

**Objective**: {prompt}
</BRANCH_TASK>

You inherit the full context from MAIN planner shown above.

Instructions:
1. Focus ONLY on the assigned objective
2. Use <task>...</task> to search if you need information
3. Reason thoroughly within <think>...</think>
4. When done, return your findings using <return>...</return>

Do NOT:
- Make final answers to the main problem
- Perform actions beyond the assigned scope
- Create new branches (you cannot branch further)

Start by analyzing the objective and planning your approach."""


HYMEM_BRANCH_USER_PROMPT_WITH_MEMORY = """{memory_context}<BRANCH_TASK>
**Description**: {description}

**Objective**: {prompt}
</BRANCH_TASK>

You inherit the full context and memory from MAIN planner.

Instructions:
1. Use the memory to guide your reasoning
2. Focus ONLY on the assigned objective
3. Use <task>...</task> to search if needed
4. When done, return findings using <return>...</return>

Start by analyzing the objective with the help of your memory."""


HYMEM_BRANCH_TIMEOUT_PROMPT = """You have reached the turn limit for this branch.
Please summarize your current findings and return immediately.

Use <return>...</return> to provide:
1. Your best conclusion so far
2. Key evidence found
3. What remains unverified

Return now with your partial findings."""


def create_planner_prompt(problem: str, memory_store: Optional[HyMemStore] = None) -> list:
    """创建Planner的提示词"""
    messages = [
        {'role': 'system', 'content': HYMEM_PLANNER_SYSTEM_PROMPT}
    ]
    
    if memory_store and (memory_store.episode_memory or memory_store.working_memory):
        memory_context = format_memory_for_context(memory_store)
        user_content = HYMEM_PLANNER_USER_PROMPT_WITH_MEMORY.format(
            memory_context=memory_context,
            problem=problem
        )
    else:
        user_content = HYMEM_PLANNER_USER_PROMPT.format(problem=problem)
    
    messages.append({'role': 'user', 'content': user_content})
    return messages


def create_executor_prompt(search_target: str, tool_memory: Optional[Dict] = None) -> list:
    """创建Executor的提示词"""
    messages = [
        {'role': 'system', 'content': HYMEM_EXECUTOR_SYSTEM_PROMPT}
    ]
    
    if tool_memory:
        tool_memory_context = format_tool_memory_for_executor(tool_memory)
        user_content = HYMEM_EXECUTOR_USER_PROMPT_WITH_MEMORY.format(
            tool_memory_context=tool_memory_context,
            search_target=search_target
        )
    else:
        user_content = HYMEM_EXECUTOR_USER_PROMPT.format(search_target=search_target)
    
    messages.append({'role': 'user', 'content': user_content})
    return messages


def create_replan_prompt(answer: str, confidence: float, problem: str) -> str:
    """创建重规划提示词"""
    if confidence < 0.3:
        return HYMEM_REPLAN_PROMPT_VERY_LOW_CONFIDENCE.format(
            confidence=confidence,
            answer=answer
        )
    else:
        return HYMEM_REPLAN_PROMPT_LOW_CONFIDENCE.format(
            confidence=confidence,
            answer=answer
        )


def format_executor_results_for_planner(results: str, task: str) -> str:
    """格式化Executor结果返回给Planner"""
    return f"""Executor has completed the search task.

Search Task: {task}

{results}

Analyze these results and decide:
1. Is the information sufficient to answer the question?
2. Do you need additional searches from different angles?
3. Are there any contradictions that need verification?"""


def create_planner_prompt_after_fold(
    problem: str,
    planner_summary: str,
    system_prompt: str,
    memory_store: Optional[HyMemStore] = None,
    recent_buffer: list = None,
    prompt_style: str = "default",
) -> list:
    """fold 后创建新的 Planner prompt

    包含三部分信息：
    1. Planner 自总结（高保真，来自完整上下文的自压缩）
    2. 结构化 Memory（辅助模型提取的补充信息）
    3. 近期对话缓冲（最近 N 轮完整对话，防止断崖式失忆）

    Args:
        system_prompt: Planner 的 system prompt（由调用方根据 prompt_style 提供）
    """
    tpl = (PLANNER_PROMPT_AFTER_FOLD_QWEN
           if prompt_style == "qwen"
           else PLANNER_PROMPT_AFTER_FOLD)

    structured_memory = ""
    if memory_store:
        structured_memory = format_memory_for_context(
            memory_store, include_summary=False
        )

    user_content = tpl.format(
        planner_summary=planner_summary,
        structured_memory=structured_memory,
        problem=problem,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if recent_buffer:
        messages.extend(recent_buffer)

    return messages


def extract_recent_buffer(messages: list, n_turns: int = 2) -> list:
    """提取最近 n 轮完整的 assistant+user 对作为短期缓冲"""
    pairs = []
    count = 0
    i = len(messages) - 1
    while i >= 0 and count < n_turns:
        if messages[i].get("role") == "user" and i > 0:
            if messages[i - 1].get("role") == "assistant":
                pairs.insert(0, messages[i - 1])
                pairs.insert(1 + 2 * count, messages[i])
                count += 1
                i -= 2
                continue
        i -= 1
    return pairs


def format_planner_history_for_memory(messages: list) -> str:
    """Legacy: 格式化 Planner 历史用于 Memory 生成 (fallback)"""
    history_parts = []
    turn = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            turn += 1
            content_preview = (
                content[:500] + "..." if len(content) > 500 else content
            )
            history_parts.append(f"Turn {turn} [Planner]:\n{content_preview}")
        elif role == "user" and "<results>" in content:
            history_parts.append(
                f"Turn {turn} [Executor Results]:\n{content[:300]}..."
            )

    return "\n\n".join(history_parts[-10:])


def format_recent_history(messages: list, n_turns: int = 5) -> str:
    """Legacy: 提取最近 n 轮的历史 (fallback)"""
    recent = []
    count = 0

    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            count += 1
            content = msg.get("content", "")[:300]
            recent.append(f"Recent thought: {content}")
            if count >= n_turns:
                break

    return "\n".join(reversed(recent))


def create_branch_prompt(
    planner_messages: list,
    description: str,
    prompt: str,
    memory_store: Optional[HyMemStore] = None
) -> list:
    """
    创建Branch的提示词 — uses *selective* context inheritance.

    Optimization 4: branches receive only the most recent planner exchanges
    (cap N), with per-message char budgets that shrink when structured memory
    is rich. Rationale: working/episode memory already carries the verified
    long-horizon facts; over-inheriting raw planner text just bloats input
    tokens without adding signal. This aligns directly with the paper's
    "selective context inheritance" framing.
    """
    messages = []

    messages.append({'role': 'system', 'content': HYMEM_BRANCH_SYSTEM_PROMPT})

    has_memory = bool(memory_store and (memory_store.episode_memory or memory_store.working_memory))
    # When structured memory is available, lean on it rather than raw context.
    keep_last_n = 3 if has_memory else 5
    user_budget = 600 if has_memory else 1000
    asst_budget = 500 if has_memory else 800

    # Walk only the last keep_last_n planner exchanges. Skip the system msg
    # at index 0, then take the tail.
    tail = planner_messages[1:][-keep_last_n * 2:]  # *2: one user + one asst per exchange
    context_parts = []
    seen_user_hashes = set()
    for msg in tail:
        role = msg.get('role', '')
        content = (msg.get('content', '') or '').strip()
        if not content:
            continue
        # Skip near-duplicate user messages (same observation appearing
        # twice — common when the planner re-issues a similar task).
        if role == 'user':
            sig = content[:120]
            if sig in seen_user_hashes:
                continue
            seen_user_hashes.add(sig)
            context_parts.append(f"[Context - User/Results]:\n{content[:user_budget]}")
        elif role == 'assistant':
            context_parts.append(f"[Context - Planner]:\n{content[:asst_budget]}")

    inherited_context = "\n\n---\n\n".join(context_parts)

    if has_memory:
        memory_context = format_memory_for_context(memory_store)
        user_content = HYMEM_BRANCH_USER_PROMPT_WITH_MEMORY.format(
            memory_context=memory_context,
            description=description,
            prompt=prompt
        )
    else:
        user_content = HYMEM_BRANCH_USER_PROMPT.format(
            description=description,
            prompt=prompt
        )

    if inherited_context:
        full_user_content = f"<INHERITED_CONTEXT>\n{inherited_context}\n</INHERITED_CONTEXT>\n\n{user_content}"
    else:
        full_user_content = user_content

    messages.append({'role': 'user', 'content': full_user_content})
    
    return messages


def format_branch_result_for_planner(branch_result: str, description: str) -> str:
    """格式化Branch结果返回给Planner"""
    return f"""Branch "{description}" has completed its reasoning task.

<branch_result>
{branch_result}
</branch_result>

Analyze this branch result and decide:
1. Does it sufficiently address the sub-problem?
2. Do you need additional branches or searches?
3. Can you now make progress on the main problem?"""


def extract_branch_info(text: str) -> Optional[Dict[str, str]]:
    """从Planner响应中提取Branch信息"""
    import re
    
    branch_match = re.search(r'<branch>(.*?)</branch>', text, re.DOTALL)
    if not branch_match:
        return None
    
    branch_content = branch_match.group(1)
    
    desc_match = re.search(r'<description>(.*?)</description>', branch_content, re.DOTALL)
    prompt_match = re.search(r'<prompt>(.*?)</prompt>', branch_content, re.DOTALL)
    
    if desc_match and prompt_match:
        return {
            'description': desc_match.group(1).strip(),
            'prompt': prompt_match.group(1).strip()
        }
    
    return None


def extract_return(text: str) -> Optional[str]:
    """从Branch响应中提取return内容"""
    import re
    matches = re.findall(r'<return>(.*?)</return>', text, re.DOTALL)
    return matches[-1].strip() if matches else None
