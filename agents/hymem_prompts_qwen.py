"""
HyMem Prompts Module - Qwen3 优化版本

针对 Qwen3-32B 模型特点进行优化的提示词模板：
1. 更简洁直接的指令
2. 强调简短输出，避免冗长推理
3. 明确的格式要求和示例
4. 减少复杂的嵌套结构
"""

import os
from typing import Optional, Dict
from .hymem_memory import HyMemStore, format_memory_for_context, format_tool_memory_for_executor


# O8: aggressive parallel-emission nudge (Qwen). Default OFF — see comment
# in hymem_prompts.py. Enable with HYMEM_PARALLEL_PROMPT=1 for ablation.
_O8_PARALLEL_PROMPT = os.getenv("HYMEM_PARALLEL_PROMPT") == "1"

_QWEN_PARALLEL_AGGRESSIVE = """
### 多个独立子问题时（强烈推荐并行）：
当下一步需要查询多个**互相独立**的事实（不同实体、不同时间段、对比的不同方面），
请在同一 turn 内输出多个 <task>，它们会**并行执行**（最多 6 个）。这能大幅加速。
仅当后续 task 依赖前一个的结果时才串行。

示例：
<think>需要分别查三所大学的成立年份，互不依赖，可并行</think>
<task>斯坦福大学的成立年份</task>
<task>哈佛大学的成立年份</task>
<task>麻省理工学院的成立年份</task>
"""

_QWEN_PARALLEL_CONSERVATIVE = ""  # default: no parallel nudge in Qwen prompt

_QWEN_PARALLEL_SECTION = (_QWEN_PARALLEL_AGGRESSIVE if _O8_PARALLEL_PROMPT
                          else _QWEN_PARALLEL_CONSERVATIVE)


# ============== Qwen3 优化版 Planner Prompt ==============

HYMEM_PLANNER_SYSTEM_PROMPT_QWEN = """你是一个高效的研究助手。你的任务是分析问题并找到答案。

## 核心规则（必须严格遵守）

1. **思考要简短**：<think>标签内只写关键推理，不超过200字
2. **答案要直接**：<answer>标签内只写最终答案，不要解释
3. **一次只做一件事**：要么搜索，要么回答，不要同时进行

## 输出格式

### 需要搜索时：
<think>简短说明为什么需要搜索</think>
<task>具体的搜索关键词或问题</task>
""" + _QWEN_PARALLEL_SECTION + """
### 有答案时：
<answer>直接的答案（不要解释）</answer>
<confidence>0.0到1.0之间的数字</confidence>

## 禁止事项
- 禁止长篇大论的思考
- 禁止在答案中重复问题
- 禁止同时输出 <task> 和 <answer>
- 禁止不使用规定标签"""


HYMEM_PLANNER_USER_PROMPT_QWEN = """问题：{problem}

请开始解决。记住：
1. <think>内容不超过200字
2. <answer>只写答案本身
3. 一次只输出一个操作

现在开始："""


HYMEM_PLANNER_USER_PROMPT_WITH_MEMORY_QWEN = """{memory_context}

问题：{problem}

你有之前的记忆可以参考。继续解决问题。

规则：
1. <think>简短（不超过200字）
2. <answer>只写答案
3. 一次只做一件事"""


# ============== Qwen3 优化版 Executor Prompt ==============

HYMEM_EXECUTOR_SYSTEM_PROMPT_QWEN = """你是搜索执行器。使用工具找到信息，然后报告结果。

## 可用工具

用<code></code>包裹Python代码调用工具：

```
result = await web_search("搜索关键词")  # 网络搜索
result = await web_parse(url, "问题")     # 解析网页
result = await parse_file(path)           # 解析文件
```

## 输出格式

1. 执行搜索时：
<code>
result = await web_search("关键词")
print(result)
</code>

2. 报告结果时：
<results>
找到的关键信息（简洁）
</results>

## 规则
- 每次只执行一个搜索
- 结果要简洁，只保留相关信息
- 找到答案后立即用<results>报告"""


HYMEM_EXECUTOR_USER_PROMPT_QWEN = """搜索目标：{search_target}

用<code>执行搜索，找到后用<results>报告。"""


HYMEM_EXECUTOR_USER_PROMPT_WITH_MEMORY_QWEN = """{tool_memory_context}

搜索目标：{search_target}

参考之前的搜索经验。用<code>搜索，用<results>报告。"""


# ============== Qwen3 优化版 Branch Prompt ==============

HYMEM_BRANCH_SYSTEM_PROMPT_QWEN = """你是推理分支。专注于分配给你的子任务。

## 规则
1. 只处理指定的子任务
2. 需要搜索时用<task>
3. 完成后用<return>报告结论

## 输出格式

需要搜索：
<task>搜索内容</task>

完成时：
<return>
结论：xxx
证据：xxx
置信度：0.x
</return>"""


HYMEM_BRANCH_USER_PROMPT_QWEN = """子任务：{description}
目标：{prompt}

专注于这个目标。完成后用<return>报告。"""


HYMEM_BRANCH_USER_PROMPT_WITH_MEMORY_QWEN = """{memory_context}

子任务：{description}
目标：{prompt}

利用记忆完成目标，用<return>报告结论。"""


HYMEM_BRANCH_TIMEOUT_PROMPT_QWEN = """时间到。立即用<return>报告当前结论：

<return>
结论：xxx
已找到的证据：xxx
未完成的部分：xxx
</return>"""


# ============== Qwen3 优化版 Replan Prompt ==============

HYMEM_REPLAN_PROMPT_LOW_CONFIDENCE_QWEN = """置信度{confidence:.2f}太低。当前答案：{answer}

继续搜索更多证据。用<task>指定搜索。"""


HYMEM_REPLAN_PROMPT_VERY_LOW_CONFIDENCE_QWEN = """置信度{confidence:.2f}非常低。当前答案：{answer}

尝试不同的搜索策略。用<task>指定新的搜索方向。"""


# ============== Qwen3 优化版 Fold 指令 ==============

HYMEM_FOLD_INSTRUCTION_QWEN = """上下文过长，将进行压缩。你的进度已保存到记忆中。继续当前任务。"""


# ============== 辅助函数 ==============

def create_planner_prompt_qwen(problem: str, memory_store: Optional[HyMemStore] = None) -> list:
    """创建 Qwen3 优化版 Planner 的提示词"""
    messages = [
        {'role': 'system', 'content': HYMEM_PLANNER_SYSTEM_PROMPT_QWEN}
    ]
    
    if memory_store and (memory_store.episode_memory or memory_store.working_memory):
        memory_context = format_memory_for_context(memory_store)
        user_content = HYMEM_PLANNER_USER_PROMPT_WITH_MEMORY_QWEN.format(
            memory_context=memory_context,
            problem=problem
        )
    else:
        user_content = HYMEM_PLANNER_USER_PROMPT_QWEN.format(problem=problem)
    
    messages.append({'role': 'user', 'content': user_content})
    return messages


def create_executor_prompt_qwen(search_target: str, tool_memory: Optional[Dict] = None) -> list:
    """创建 Qwen3 优化版 Executor 的提示词"""
    messages = [
        {'role': 'system', 'content': HYMEM_EXECUTOR_SYSTEM_PROMPT_QWEN}
    ]
    
    if tool_memory:
        tool_memory_context = format_tool_memory_for_executor(tool_memory)
        user_content = HYMEM_EXECUTOR_USER_PROMPT_WITH_MEMORY_QWEN.format(
            tool_memory_context=tool_memory_context,
            search_target=search_target
        )
    else:
        user_content = HYMEM_EXECUTOR_USER_PROMPT_QWEN.format(search_target=search_target)
    
    messages.append({'role': 'user', 'content': user_content})
    return messages


def create_replan_prompt_qwen(answer: str, confidence: float, problem: str) -> str:
    """创建 Qwen3 优化版重规划提示词"""
    if confidence < 0.3:
        return HYMEM_REPLAN_PROMPT_VERY_LOW_CONFIDENCE_QWEN.format(
            confidence=confidence,
            answer=answer
        )
    else:
        return HYMEM_REPLAN_PROMPT_LOW_CONFIDENCE_QWEN.format(
            confidence=confidence,
            answer=answer
        )


def create_branch_prompt_qwen(
    planner_messages: list,
    description: str,
    prompt: str,
    memory_store: Optional[HyMemStore] = None
) -> list:
    """创建 Qwen3 优化版 Branch 的提示词"""
    messages = []
    
    messages.append({'role': 'system', 'content': HYMEM_BRANCH_SYSTEM_PROMPT_QWEN})
    
    # 简化继承的上下文
    context_parts = []
    for msg in planner_messages[1:][-4:]:  # 只取最近4条
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role == 'user':
            context_parts.append(f"[背景]: {content[:500]}")
        elif role == 'assistant':
            context_parts.append(f"[已知]: {content[:400]}")
    
    inherited_context = "\n".join(context_parts)
    
    if memory_store and (memory_store.episode_memory or memory_store.working_memory):
        memory_context = format_memory_for_context(memory_store)
        user_content = HYMEM_BRANCH_USER_PROMPT_WITH_MEMORY_QWEN.format(
            memory_context=memory_context,
            description=description,
            prompt=prompt
        )
    else:
        user_content = HYMEM_BRANCH_USER_PROMPT_QWEN.format(
            description=description,
            prompt=prompt
        )
    
    if inherited_context:
        full_user_content = f"{inherited_context}\n\n{user_content}"
    else:
        full_user_content = user_content
    
    messages.append({'role': 'user', 'content': full_user_content})
    
    return messages


def format_executor_results_for_planner_qwen(results: str, task: str) -> str:
    """Qwen3 优化版：格式化 Executor 结果"""
    # 简化结果，避免过长
    truncated_results = results[:2000] if len(results) > 2000 else results
    return f"""搜索完成。

任务：{task}
{truncated_results}

分析结果，决定：
1. 信息足够 → 用<answer>回答
2. 需要更多 → 用<task>继续搜索"""


def format_branch_result_for_planner_qwen(branch_result: str, description: str) -> str:
    """Qwen3 优化版：格式化 Branch 结果"""
    truncated = branch_result[:1500] if len(branch_result) > 1500 else branch_result
    return f"""分支"{description}"完成。

结果：
{truncated}

继续主任务。有答案用<answer>，需搜索用<task>。"""

