import os
import re
import time
import copy
import asyncio
from functools import partial
import random

import numpy as np
import torch

from verl import DataProto
from .utils import CallLLM, Agent, select_env, truncate_text, is_weird, TaskContext, run_action
from .prompts import create_chat
from .tool_metrics import UnifiedToolCounter, write_baseline_stats


# 默认配置常量
DEFAULT_SESSION_TIMEOUT = 150 * 60  # 150 分钟，参考 DeepResearch
DEFAULT_MIN_RESPONSE_TOKENS = 4096  # 预留给 response 的最小 token 数


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallLLM,
) -> DataProto:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    ability = item.non_tensor_batch['ability'][0]
    # Select env
    EnvClass = select_env(ability, config, )
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                       "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = agent_config.get("max_turn", 64)
    host = context.server_host
    port = context.server_port
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))
    prompt_turn = len(user_prompt)

    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    
    # 从配置读取超时参数
    session_timeout = getattr(config.plugin, 'session_timeout', DEFAULT_SESSION_TIMEOUT)
    
    # 动态计算上下文限制：基于 config 中的 prompt_length + response_length，预留 response 空间
    max_model_len = config.prompt_length + config.response_length
    min_response_tokens = getattr(config.plugin, 'min_response_tokens', DEFAULT_MIN_RESPONSE_TOKENS)
    # 如果配置了 max_context_tokens 就使用，否则动态计算
    max_context_tokens = getattr(config.plugin, 'max_context_tokens', None)
    if max_context_tokens is None:
        max_context_tokens = max_model_len - min_response_tokens
    
    iteration = 0
    start_time = time.time()

    while iteration < max_turn:
        iteration += 1

        # 超时检测
        elapsed_time = time.time() - start_time
        if elapsed_time > session_timeout:
            print(f'[TIMEOUT] Session exceeded {session_timeout/60:.1f} minutes, terminating...')
            break

        # 上下文长度检测：检查剩余可用 token 数
        current_context = agent.context()
        current_len = len(current_context)
        remaining_tokens = max_model_len - current_len

        if remaining_tokens < min_response_tokens:
            print(f'[TOKEN LIMIT] Context ({current_len} tokens) too long, only {remaining_tokens} tokens remaining (need {min_response_tokens}), terminating...')
            break

        response = await agent.step()
        if response is None:
            break
        observation = await run_action(env, response)
        if observation is None:
            break
        agent.append({'role': 'user', 'content': observation})

    # Cumulative tokens auto-accumulated by Agent.append() across all
    # step() calls — see agents/utils.py:Agent.cumulative_usage.
    total_prompt_tokens = agent.cumulative_usage["prompt_tokens"]
    total_completion_tokens = agent.cumulative_usage["completion_tokens"]

    session_time = time.time() - start_time

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    # === Unified stats schema (paper-consistent across all baselines) ===
    write_baseline_stats(
        env.stats,
        framework="react",
        score=score[1],
        iterations=iteration,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        session_time=session_time,
    )
    env.stats['get_final_score'] = score[1]
    env.stats['total_turns'] = len(agent.messages())
    env.stats['messages'] = agent.messages()
    # G4 — per-turn context size trace for cross-baseline Figure 3a-revised
    env.stats['context_size_history'] = list(agent.context_size_history)

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                         tag='main', metrics=agent.get_metrics())
    res = DataProto.concat([out])
    return res


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
