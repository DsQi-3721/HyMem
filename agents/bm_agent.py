"""
BrowseMaster Agent实现

本模块实现BrowseMaster框架的Planner-Executor双Agent架构。

核心思想：
- Planner: 高层推理Agent，负责问题分析、任务分解和结果综合
- Executor: 执行Agent，通过代码调用搜索工具执行具体搜索任务

工作流程：
1. Planner接收问题，分析并输出搜索任务（<task>标签）
2. Executor接收任务，生成并执行代码调用bm_tools中的工具
3. Executor返回搜索结果给Planner
4. Planner根据结果继续推理或输出最终答案

本实现适配FoldAgent评测框架，复用其基础设施。
"""

import os
import re
import ast
import sys
import time
import copy
import json
import asyncio
import traceback
from typing import Dict, List, Any, Optional, Tuple, Callable
from functools import partial
from dataclasses import dataclass, field

import numpy as np
import torch

from verl import DataProto
from .utils import CallLLM, CallAPI, Agent, TaskContext, truncate_text, is_weird
from .bm_tools import BMToolManager, BMToolsConfig
from .bm_sandbox import BMExecutorSandbox, create_executor_sandbox


# ==================== 提示词定义 ====================

BM_PLANNER_SYSTEM_PROMPT = """You are an expert research planner. Your task is to analyze complex questions and devise effective search strategies to find accurate answers.

You work with a specialized search executor that can perform various search operations. Your role is to:
1. Analyze the problem and identify key information needs
2. Decompose complex questions into specific search tasks
3. Assign search tasks to the executor
4. Synthesize search results and reason towards the final answer

## Your Workflow

1. **Problem Analysis**: Carefully analyze the question to identify:
   - Core entities and concepts
   - Constraints and conditions
   - Relationships between different pieces of information needed

2. **Search Strategy**: Design effective search strategies:
   - Identify which features can most effectively narrow down the search space
   - Consider multiple search angles for complex problems
   - Plan verification searches for important claims

3. **Task Assignment**: Delegate search tasks to the executor using:
<task>
[Describe the specific search task here. Be clear and specific about what information to find.]
</task>

4. **Result Integration**: When the executor returns results:
   - Evaluate if the results meet the search criteria
   - Identify gaps in information
   - Decide whether to continue searching or conclude

5. **Final Answer**: When you have sufficient evidence:
<answer>
[Your final answer based on the gathered evidence]
</answer>
<confidence>
[Your confidence score between 0 and 1]
</confidence>

## Important Guidelines

- Think step by step within <think>...</think> tags
- Do not stop thinking until you find the answer
- If stuck, reanalyze the problem from different angles
- The answer exists - keep searching with different strategies
- Do NOT call the executor after outputting your final answer
"""

BM_PLANNER_USER_PROMPT = """The problem is: {problem}

Solve the problem using the help of feedback from a search executor. The executor is equipped with powerful search and analysis capabilities.

To delegate a task to the executor, use the following format:
<task>
[Describe the specific search task here]
</task>

The executor will return search results in <results>...</results>.

IMPORTANT: When assigning a search task:
- Be specific about what information to find
- Avoid overly broad search targets
- Consider breaking complex searches into focused sub-tasks

Based on your reasoning and executor feedback, you can call the executor multiple times. Loop through reasoning, task delegation, and feedback analysis within <think>...</think>. Do not stop thinking until you find the answer.

Output format:
When you believe you have found the answer, stop thinking and put it in <answer>...</answer>.
Put your confidence in <confidence>...</confidence>, ranging from 0 to 1.

Start by analyzing the problem deeply. Identify relationships between features and look for key insights that can narrow down the search space."""


BM_EXECUTOR_SYSTEM_PROMPT = """You are a searching agent equipped with multiple search tools. Your task is to reason and search exhaustively to find all information that satisfies the given search target.

## Available Tools

You have access to these search tools:

1. **generate_keywords(seed_keyword)**: Generates multiple search keyword variations from a seed keyword.
   - Returns: {"tool_result": {"keywords": [...]}}

2. **batch_search(keywords)**: Performs batch search with multiple keywords.
   - Input: List of keywords
   - Returns: {"tool_result": {"organic": [{"title": ..., "link": ..., "snippet": ..., "position": ...}, ...]}}

3. **check_condition(content, condition)**: Evaluates if content meets a condition using LLM.
   - Returns: {"tool_result": {"is_relevant": "yes"|"no"|"unknown", "reasoning": ...}}

4. **batch_search_and_filter(keyword)**: Integrated search solution that generates keywords, searches, and filters results.
   - Returns: {"tool_result": {"yes": [...], "unknown": [...], "no_count": int}}

5. **web_search(keywords)**: Searches the web and returns top results.
   - Returns: {"tool_result": {"searchParameters": {...}, "organic": [...]}}

6. **web_parse(link, query)**: Fetches a web page and uses LLM to answer a query based on its content.
   - Returns: {"tool_result": {"content": ..., "url": ..., "docid": ...}}

## How to Use Tools

Call tools by writing Python code enclosed within <code></code> tags. The code will be executed and you can see the output. You can write code multiple times, like cells in a Jupyter notebook.

Example usage:
<code>
# Generate keywords and search
result = await generate_keywords("machine learning papers 2023")
keywords = result["tool_result"]["keywords"]
print(f"Generated keywords: {keywords}")

# Batch search
search_result = await batch_search(keywords[:5])
for item in search_result["tool_result"]["organic"][:3]:
    print(f"- {item['title']}: {item['snippet'][:100]}")
</code>

## Guidelines

1. Think carefully before searching - avoid overly broad searches
2. Use generate_keywords to expand search coverage
3. Use batch_search_and_filter for comprehensive searches with filtering
4. Use web_parse to extract detailed information from promising pages
5. Loop through reasoning, tool usage, and feedback analysis within <think>...</think>
6. Use asyncio for parallel operations when appropriate

Output format:
When you have found all relevant results, stop thinking and put them in:
<results>
[Your search results with supporting evidence]
</results>
"""

# ==================== 结果提取器 ====================

def extract_task(text: str) -> Optional[str]:
    """从Planner响应中提取<task>标签内容"""
    matches = re.findall(r'<task>(.*?)</task>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_answer(text: str) -> Optional[str]:
    """从Planner响应中提取<answer>标签内容"""
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_confidence(text: str) -> float:
    """从Planner响应中提取<confidence>标签内容"""
    matches = re.findall(r'<confidence>(.*?)</confidence>', text, re.DOTALL)
    if matches:
        try:
            conf = float(matches[-1].strip())
            return min(max(conf, 0.0), 1.0)
        except:
            pass
    return 0.5


def extract_results(text: str) -> Optional[str]:
    """从Executor响应中提取<results>标签内容"""
    matches = re.findall(r'<results>(.*?)</results>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_think(text: str) -> Optional[str]:
    """从响应中提取<think>标签内容"""
    matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL)
    return matches[-1].strip() if matches else None


# ==================== BrowseMaster Agent主类 ====================

class BMAgent:
    """
    BrowseMaster Agent主类
    
    实现Planner-Executor双Agent架构。
    """
    
    def __init__(
        self,
        planner_client,
        executor_client,
        tool_manager: BMToolManager,
        tokenizer,
        config,
        max_planner_turns: int = 20,
        max_executor_turns: int = 10
    ):
        """
        初始化BMAgent
        
        Args:
            planner_client: Planner的LLM客户端
            executor_client: Executor的LLM客户端
            tool_manager: 工具管理器
            tokenizer: 分词器
            config: 配置
            max_planner_turns: Planner最大轮次
            max_executor_turns: Executor最大轮次
        """
        self.planner_client = planner_client
        self.executor_client = executor_client
        self.tool_manager = tool_manager
        # 使用隔离沙箱执行Executor代码（BrowseMaster论文核心设计）
        self.executor_sandbox = create_executor_sandbox(tool_manager, timeout=180)
        self.tokenizer = tokenizer
        self.config = config
        self.max_planner_turns = max_planner_turns
        self.max_executor_turns = max_executor_turns
        
        # 置信度引导重规划参数
        self.confidence_threshold_high = 0.8  # 高置信度阈值，超过此值可以结束
        self.confidence_threshold_low = 0.3   # 低置信度阈值，低于此值需要扩大搜索
        self.max_replan_attempts = 3          # 最大重规划次数
        self.current_replan_count = 0         # 当前重规划计数
        
        # 会话历史
        self.planner_history = []
        self.executor_history = []
        self.full_history = []  # 完整交互历史
    
        # HyMem Memory支持（可选）
        self.memory_store = None
    
    def inject_memory(self, memory_store):
        """
        注入HyMem Memory存储
        
        Args:
            memory_store: HyMemStore实例，包含Episode/Working/Tool Memory
        """
        self.memory_store = memory_store
    
    def _create_planner_prompt(self, problem: str, memory_store=None) -> List[Dict]:
        """
        创建Planner的初始提示
        
        Args:
            problem: 问题描述
            memory_store: 可选的HyMemStore，用于注入Memory上下文
        """
        store = memory_store or self.memory_store
        
        if store is not None and (store.episode_memory or store.working_memory):
            from .hymem_memory import format_memory_for_context
            memory_context = format_memory_for_context(store)
            user_prompt = BM_PLANNER_USER_PROMPT.format(problem=problem) + "\n\n" + memory_context
        else:
            user_prompt = BM_PLANNER_USER_PROMPT.format(problem=problem)
        
        return [
            {'role': 'system', 'content': BM_PLANNER_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt}
        ]
    
    def _create_executor_prompt(self, search_target: str) -> List[Dict]:
        """创建Executor的提示"""
        user_content = f"""Your search target is: {search_target}

Use the available tools to find all information that satisfies this target. 
Write Python code within <code></code> tags to call the tools.
When done, put your findings in <results>...</results>."""
        
        return [
            {'role': 'system', 'content': BM_EXECUTOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_content}
        ]
    
    async def _call_llm(self, client, messages: List[Dict], max_tokens: int = 4096) -> Optional[str]:
        """调用LLM"""
        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True
            )
            
            completion = await client.create_completion(
                input_ids,
                messages=messages,
                max_new_tokens=max_tokens
            )
            
            if completion is None:
                return None
            
            return completion["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[BM_AGENT] LLM call error: {e}")
            return None
    
    def _generate_replan_prompt(self, current_answer: str, confidence: float, problem: str) -> str:
        """
        生成置信度引导的重规划提示
        
        根据当前置信度水平，生成不同强度的重规划指导：
        - 极低置信度(< 0.3): 建议完全重新分析问题，尝试不同搜索策略
        - 低置信度(0.3-0.5): 建议扩大搜索范围，寻找更多证据
        - 中等置信度(0.5-0.8): 建议验证当前答案，寻找确认性证据
        
        Args:
            current_answer: 当前得到的答案
            confidence: 当前置信度
            problem: 原始问题
            
        Returns:
            重规划提示文本
        """
        if confidence < 0.3:
            # 极低置信度：完全重新分析
            return f"""Your current answer has very low confidence ({confidence:.2f}).

Current tentative answer: {current_answer}

The confidence level suggests that the answer may be incorrect or the evidence is insufficient. Please:

1. **Re-analyze the problem from scratch**: Look for alternative interpretations or overlooked constraints
2. **Try completely different search strategies**: Use different keywords, explore different angles
3. **Question your assumptions**: What assumptions led to this answer? Are they valid?
4. **Search for contradicting evidence**: Actively look for information that might disprove your current answer

Do NOT simply repeat previous searches. Think creatively about new approaches to find the correct answer.

If after additional searching you still cannot find sufficient evidence, you may provide your best answer with the appropriate confidence level."""

        elif confidence < 0.5:
            # 低置信度：扩大搜索范围
            return f"""Your current answer has low confidence ({confidence:.2f}).

Current tentative answer: {current_answer}

Please expand your search to gather more supporting evidence:

1. **Broaden search scope**: Try more general or related keywords
2. **Look for multiple sources**: Find at least 2-3 independent sources to corroborate
3. **Explore edge cases**: Check if there are exceptions or special cases
4. **Verify key facts**: Double-check the most critical pieces of information

Assign new search tasks to gather more evidence before finalizing your answer."""

        else:
            # 中等置信度：验证确认
            return f"""Your current answer has moderate confidence ({confidence:.2f}).

Current tentative answer: {current_answer}

To increase confidence, please verify your answer:

1. **Cross-reference**: Find additional sources that confirm the key facts
2. **Check for updates**: Ensure the information is current and accurate
3. **Validate details**: Confirm specific names, dates, numbers, or other details

If you find confirming evidence, provide your final answer with increased confidence.
If you find contradicting evidence, revise your answer accordingly."""
    
    async def run_executor(self, search_target: str) -> str:
        """
        运行Executor执行搜索任务
        
        Args:
            search_target: 搜索目标
            
        Returns:
            搜索结果
        """
        messages = self._create_executor_prompt(search_target)
        self.executor_history = messages.copy()
        
        for turn in range(self.max_executor_turns):
            response = await self._call_llm(self.executor_client, messages)
            
            if response is None:
                return "[Executor error: No response from LLM]"
            
            self.executor_history.append({'role': 'assistant', 'content': response})
            
            # 检查是否有最终结果
            results = extract_results(response)
            if results:
                return f"<results>\n{results}\n</results>"
            
            # 在隔离沙箱中执行代码块（BrowseMaster论文核心设计：隔离执行环境）
            code_output = await self.executor_sandbox.execute_from_response(response)
            
            if "[No code blocks found" in code_output:
                # 没有代码块，可能需要提示Executor
                observation = "No code was executed. Please write code within <code></code> tags to use the search tools, or provide your final results in <results></results>."
            else:
                observation = f"Code execution output:\n{code_output}"
            
            messages.append({'role': 'assistant', 'content': response})
            messages.append({'role': 'user', 'content': observation})
            self.executor_history.append({'role': 'user', 'content': observation})
        
        # 达到最大轮次，请求总结
        summary_prompt = "You have reached the maximum number of turns. Please summarize your findings so far in <results></results>."
        messages.append({'role': 'user', 'content': summary_prompt})
        
        response = await self._call_llm(self.executor_client, messages)
        if response:
            results = extract_results(response)
            if results:
                return f"<results>\n{results}\n</results>"
            return f"<results>\n{response}\n</results>"
        
        return "[Executor error: Failed to produce results]"
    
    async def run(self, problem: str) -> Tuple[str, float, List[Dict]]:
        """
        运行完整的BrowseMaster流程
        
        Args:
            problem: 要解决的问题
            
        Returns:
            (答案, 置信度, 完整交互历史)
        """
        # 初始化Planner
        messages = self._create_planner_prompt(problem)
        self.planner_history = messages.copy()
        self.full_history = []
        
        final_answer = ""
        confidence = 0.0
        
        for turn in range(self.max_planner_turns):
            print(f"[BM_AGENT] Planner turn {turn + 1}/{self.max_planner_turns}")
            
            # 调用Planner
            response = await self._call_llm(self.planner_client, messages)
            
            if response is None:
                print("[BM_AGENT] Planner error: No response")
                break
            
            self.planner_history.append({'role': 'assistant', 'content': response})
            self.full_history.append({
                'role': 'planner',
                'content': response,
                'turn': turn + 1
            })
            
            # 检查是否有最终答案
            answer = extract_answer(response)
            if answer:
                final_answer = answer
                confidence = extract_confidence(response)
                print(f"[BM_AGENT] Found answer with confidence {confidence}")
                
                # ========== 置信度引导重规划（BrowseMaster论文核心设计）==========
                # 如果置信度低于阈值且未达到最大重规划次数，触发重规划
                if confidence < self.confidence_threshold_high and self.current_replan_count < self.max_replan_attempts:
                    self.current_replan_count += 1
                    print(f"[BM_AGENT] Low confidence ({confidence:.2f}), triggering replan {self.current_replan_count}/{self.max_replan_attempts}")
                    
                    # 生成重规划提示
                    replan_prompt = self._generate_replan_prompt(answer, confidence, problem)
                    messages.append({'role': 'assistant', 'content': response})
                    messages.append({'role': 'user', 'content': replan_prompt})
                    self.planner_history.append({'role': 'user', 'content': replan_prompt})
                    
                    self.full_history.append({
                        'role': 'replan',
                        'previous_answer': answer,
                        'confidence': confidence,
                        'turn': turn + 1
                    })
                    
                    # 不break，继续循环让Planner重新搜索
                    continue
                
                # 高置信度或已达到最大重规划次数，结束
                break
            
            # 检查是否有搜索任务
            task = extract_task(response)
            if task:
                print(f"[BM_AGENT] Executing search task: {task[:100]}...")
                
                # 运行Executor
                executor_result = await self.run_executor(task)
                
                self.full_history.append({
                    'role': 'executor',
                    'task': task,
                    'result': executor_result,
                    'turn': turn + 1
                })
                
                # 将结果反馈给Planner
                observation = f"Executor has completed the search task. Here are the results:\n\n{executor_result}"
                messages.append({'role': 'assistant', 'content': response})
                messages.append({'role': 'user', 'content': observation})
                self.planner_history.append({'role': 'user', 'content': observation})
            else:
                # 没有任务也没有答案，提示Planner
                observation = "No search task was found in your response. Please either:\n1. Assign a search task using <task>...</task>\n2. Provide your final answer using <answer>...</answer>"
                messages.append({'role': 'assistant', 'content': response})
                messages.append({'role': 'user', 'content': observation})
                self.planner_history.append({'role': 'user', 'content': observation})
        
        # 如果没有找到答案，尝试从最后的响应中提取
        if not final_answer and self.planner_history:
            last_response = self.planner_history[-1].get('content', '')
            if isinstance(last_response, str):
                # 尝试从think标签中提取可能的答案
                final_answer = extract_answer(last_response) or "Unable to determine answer"
                confidence = extract_confidence(last_response)
        
        return final_answer, confidence, self.full_history


# ==================== 评测入口函数 ====================

async def process_item(
    item: DataProto,
    context: TaskContext,
    LLMClass=CallAPI,
) -> DataProto:
    """
    BrowseMaster评测入口函数
    
    适配FoldAgent评测框架，实现BrowseMaster工作流。
    
    Args:
        item: 数据项
        context: 任务上下文
        LLMClass: LLM客户端类
        
    Returns:
        评测结果DataProto
    """
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train
    
    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    
    ability = item.non_tensor_batch['ability'][0]
    
    # 选择环境
    from .utils import select_env
    EnvClass = select_env(ability, config)
    print(f"[BM_AGENT] is_train={is_train}, EnvClass={EnvClass}")
    env = EnvClass(config, tokenizer, ability)
    
    try:
        await env.init_env(item)
        # 调用get_data来初始化instance_info（LocalSearch在此方法中设置instance_info）
        await env.get_data(item, context)
    except Exception as e:
        print(f"[BM_AGENT] Error during environment init: {str(e)}")
    
    # 获取问题 - 优先从extra_info获取，兼容不同环境
    extra_info = item.non_tensor_batch.get('extra_info', [{}])[0]
    problem = extra_info.get('query', '') or extra_info.get('problem_statement', '')
    if not problem and hasattr(env, 'instance_info'):
        problem = env.instance_info.get('problem_statement', '') or env.instance_info.get('query', '')
    if not problem and hasattr(env, 'question'):
        problem = env.question or ''
    
    print(f"[BM_AGENT] Problem: {problem[:200]}...")
    
    # 配置参数
    max_planner_turns = getattr(config.plugin, "bm_max_planner_turns", 20)
    max_executor_turns = getattr(config.plugin, "bm_max_executor_turns", 10)
    search_base_url = getattr(config.plugin, "search_base_url", None) or os.getenv("LOCAL_SEARCH_URL")
    llm_model = getattr(config.plugin, "bm_llm_model", None) or os.getenv("BM_LLM_MODEL", "deepseek-chat")
    
    # 搜索模式配置：local, online, auto
    search_mode = getattr(config.plugin, "search_mode", None) or os.getenv("SEARCH_MODE", "auto")
    google_api_key = getattr(config.plugin, "google_api_key", None) or os.getenv("GOOGLE_API_KEY")
    
    # 创建LLM客户端
    host = context.server_host
    port = context.server_port
    
    # Planner和Executor可以使用不同的模型
    planner_model = getattr(config.plugin, "bm_planner_model", None) or llm_model
    executor_model = getattr(config.plugin, "bm_executor_model", None) or llm_model
    
    planner_client = LLMClass(planner_model, port, tokenizer, config, meta_info={})
    executor_client = LLMClass(executor_model, port, tokenizer, config, meta_info={})
    
    # 创建工具管理器
    tool_config = BMToolsConfig(
        llm_model=llm_model,
        search_topk=getattr(config.plugin, "bm_search_topk", 10),
        max_keywords=getattr(config.plugin, "bm_max_keywords", 8),
        batch_search_topk=getattr(config.plugin, "bm_batch_search_topk", 5)
    )
    tool_manager = BMToolManager(
        search_base_url=search_base_url,
        llm_model=llm_model,
        config=tool_config,
        search_mode=search_mode,
        google_api_key=google_api_key
    )
    
    # 创建BMAgent
    agent = BMAgent(
        planner_client=planner_client,
        executor_client=executor_client,
        tool_manager=tool_manager,
        tokenizer=tokenizer,
        config=config,
        max_planner_turns=max_planner_turns,
        max_executor_turns=max_executor_turns
    )
    
    session_start_time = time.time()
    
    try:
        # 运行BMAgent
        answer, confidence, history = await agent.run(problem)
        
        print(f"[BM_AGENT] Answer: {answer[:200]}...")
        print(f"[BM_AGENT] Confidence: {confidence}")
        
    except Exception as e:
        print(f"[BM_AGENT] Error during execution: {e}")
        traceback.print_exc()
        answer = ""
        confidence = 0.0
        history = []
    finally:
        # 关闭工具管理器
        await tool_manager.close()
    
    env.stats['session_time'] = time.time() - session_start_time
    
    # 构建消息历史用于评分
    messages = [
        {'role': 'system', 'content': BM_PLANNER_SYSTEM_PROMPT},
        {'role': 'user', 'content': BM_PLANNER_USER_PROMPT.format(problem=problem)}
    ]
    
    # 添加交互历史
    for entry in history:
        if entry['role'] == 'planner':
            messages.append({'role': 'assistant', 'content': entry['content']})
        elif entry['role'] == 'executor':
            messages.append({'role': 'user', 'content': f"Search task: {entry['task']}\n\nResults: {entry['result']}"})
    
    # 添加最终答案
    if answer:
        final_response = f"<answer>{answer}</answer>\n<confidence>{confidence}</confidence>"
        messages.append({'role': 'assistant', 'content': final_response})
    
    # 设置finish标志
    env.finish = bool(answer)
    env.is_finish = bool(answer)
    env.predicted_answer = (answer, None, confidence)
    
    # 获取奖励
    print('[BM_AGENT] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, messages, context), timeout=60 * 10)
        score = (score_msg, reward)
        print(f"[BM_AGENT] Score: {score}")
    except Exception as e:
        print(f"[BM_AGENT] Error getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}
    
    # 构建Agent对象用于创建DataProto
    prompt_turn = 2  # system + user
    agent_context = Agent(planner_client, messages[:prompt_turn], tokenizer, config, prompt_turn=prompt_turn)
    
    # 添加后续消息
    for msg in messages[prompt_turn:]:
        agent_context.append(msg)
    
    # 更新统计信息
    tool_stats = tool_manager.get_tool_stats()
    env.stats['get_final_score'] = score[1]
    env.stats['confidence'] = confidence
    env.stats['planner_turns'] = len([h for h in history if h['role'] == 'planner'])
    env.stats['executor_calls'] = len([h for h in history if h['role'] == 'executor'])
    env.stats['replan_count'] = len([h for h in history if h['role'] == 'replan'])
    env.stats['tool_calls_total'] = tool_stats.get('tool_calls_total', 0)
    env.stats['tool_calls_by_tool'] = tool_stats.get('tool_calls_by_tool', {})
    env.stats['tool_call_history_full'] = tool_stats.get('tool_call_history', [])
    
    # 保存完整的执行历史用于日志记录
    env.stats['bm_history'] = history
    env.stats['bm_problem'] = problem
    env.stats['bm_answer'] = answer
    env.stats['bm_messages'] = messages
    
    # 创建输出
    out = await agent_context.dataproto()
    out = await env.update_dataproto(out, item, messages, score, reward_dict, tag='main', metrics={})
    
    res = DataProto.concat([out])
    return res


# ==================== 简化版入口（用于测试） ====================

async def run_browsemaster_simple(
    problem: str,
    search_base_url: str = None,
    llm_model: str = "deepseek-chat",
    search_mode: str = "auto",
    google_api_key: str = None
) -> Tuple[str, float]:
    """
    简化版BrowseMaster运行函数，用于测试
    
    Args:
        problem: 要解决的问题
        search_base_url: 搜索服务器URL（search_mode="local" 时使用）
        llm_model: LLM模型名称
        search_mode: 搜索模式
            - "local": 使用 LocalSearch
            - "online": 使用 OnlineSearch (Google/DuckDuckGo)
            - "auto": 自动选择
        google_api_key: Google Custom Search API Key（search_mode="online" 时可选）
        
    Returns:
        (答案, 置信度)
    """
    from openai import AsyncOpenAI
    
    # 创建简单的LLM客户端
    class SimpleLLMClient:
        def __init__(self, model: str):
            self.model = model
            self.client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", None)
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
                return {
                    "choices": [{
                        "message": {
                            "content": text,
                            "raw_output_ids": [],
                            "response_log_probs": []
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
    
    # 创建简单的tokenizer
    class SimpleTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
            return list(range(100))  # 返回假的token ids
    
    # 创建组件
    tool_manager = BMToolManager(
        search_base_url=search_base_url,
        llm_model=llm_model,
        search_mode=search_mode,
        google_api_key=google_api_key
    )
    planner_client = SimpleLLMClient(llm_model)
    executor_client = SimpleLLMClient(llm_model)
    
    # 创建简单的config
    class SimpleConfig:
        response_length = 32000
        prompt_length = 8000
        
        class plugin:
            pass
    
    agent = BMAgent(
        planner_client=planner_client,
        executor_client=executor_client,
        tool_manager=tool_manager,
        tokenizer=SimpleTokenizer(),
        config=SimpleConfig(),
        max_planner_turns=15,
        max_executor_turns=8
    )
    
    try:
        answer, confidence, _ = await agent.run(problem)
        return answer, confidence
    finally:
        await tool_manager.close()


# ==================== 测试函数 ====================

async def _test_bm_agent():
    """测试BrowseMaster Agent"""
    print("Testing BrowseMaster Agent...")
    print("=" * 60)
    
    problem = "What is the capital of France?"
    
    try:
        print(f"\n[Test] Problem: {problem}")
        print("[Test] Using search_mode='online' (DuckDuckGo)...")
        
        answer, confidence = await run_browsemaster_simple(
            problem,
            search_mode="online",  # 使用在线搜索
            llm_model="deepseek-reasoner"
        )
        
        print(f"\nProblem: {problem}")
        print(f"Answer: {answer}")
        print(f"Confidence: {confidence}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(_test_bm_agent())
