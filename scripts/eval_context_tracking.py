#!/usr/bin/env python3
"""
上下文追踪对比实验脚本
用于对比 ReAct 和 HyMem 框架在解决问题时的 token 使用趋势
"""
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
import time
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.react_agent import process_item as react_process_item
from agents.hymem_agent import process_item as hymem_process_item
from agents.hymem_trigger import estimate_context_length
from agents.utils import CallAPI, TaskContext, Agent
from verl import DataProto
import os


# ==================== 数据集加载函数 ====================

def load_gaia_dataset(parquet_path: str, gaia_root: str = None) -> pd.DataFrame:
    """加载GAIA数据集（参考 eval_gaia.py）"""
    df = pd.read_parquet(parquet_path)
    
    if gaia_root is None:
        gaia_root = str(Path(parquet_path).parent.parent.parent)
    
    records = []
    for idx, row in df.iterrows():
        task_id = row.get('task_id', f'gaia_{idx:04d}')
        question = row.get('Question', '')
        level = row.get('Level', 1)
        answer = row.get('Final answer', '')
        file_name = row.get('file_name', None)
        file_path = row.get('file_path', None)
        
        if pd.isna(file_name):
            file_name = None
        if pd.isna(file_path):
            file_path = None
        
        abs_file_path = None
        if file_path:
            abs_file_path = os.path.join(gaia_root, file_path)
            if not os.path.exists(abs_file_path):
                abs_file_path = None
        
        record = {
            'extra_info': {
                'instance_id': task_id,
                'query': question,
                'problem_statement': question,
                'answer': answer,
                'level': level,
                'file_name': file_name,
                'file_path': file_path,
                'abs_file_path': abs_file_path,
                'data_source': 'gaia'
            },
            'ability': 'OnlineSearch_GAIA',
            'data_source': 'gaia',
            'level': level,
            'reward_model': {'type': 'exact_match'}
        }
        records.append(record)
    
    print(f"[GAIA] Loaded {len(records)} items from {parquet_path}")
    print(f"[GAIA] With files: {sum(1 for r in records if r['extra_info']['abs_file_path'])}")
    print(f"[GAIA] Text-only: {sum(1 for r in records if not r['extra_info']['abs_file_path'])}")
    
    return pd.DataFrame(records)


def load_bc_dataset(parquet_path: str) -> pd.DataFrame:
    """加载 BrowseComp 数据集"""
    df = pd.read_parquet(parquet_path)
    print(f"[BC] Loaded {len(df)} items from {parquet_path}")
    return df


def load_dataset(args) -> pd.DataFrame:
    """根据数据源类型加载数据集"""
    # 自动检测数据源类型
    data_source = args.data_source
    if data_source == 'auto' or ('GAIA' in args.data_path or 'gaia' in args.data_path):
        # 检测是否为 GAIA 数据
        try:
            temp_df = pd.read_parquet(args.data_path)
            if 'Question' in temp_df.columns and 'Final answer' in temp_df.columns:
                data_source = 'gaia'
                print(f"[Auto] 检测到 GAIA 数据格式")
            elif 'extra_info' in temp_df.columns:
                data_source = 'bc'
            else:
                data_source = args.data_source
        except:
            data_source = args.data_source
    
    if data_source == 'gaia':
        df = load_gaia_dataset(args.data_path, args.gaia_root)
        if args.text_only:
            df = df[df['extra_info'].apply(lambda x: x.get('abs_file_path') is None)]
            print(f"[GAIA] Filtered to {len(df)} text-only items")
            df = df.sample(n=20, random_state=42)

    else:  # bc (BrowseComp)
        df = load_bc_dataset(args.data_path)
    
    return df


# ==================== Token 追踪的全局变量 ====================
GLOBAL_TOKEN_TRACKER = []
GLOBAL_FRAMEWORK = None


# ==================== Monkey Patch: 注入 Token 追踪 ====================

original_agent_step = Agent.step
original_agent_append = Agent.append

async def tracked_agent_step(self, max_new_tokens=None, retry_cjk=0):
    """为 Agent.step 添加 token 追踪"""
    if GLOBAL_FRAMEWORK == 'react':
        current_len = len(self.context())
        max_model_len = self.config.prompt_length + self.config.response_length
        remaining_tokens = max_model_len - current_len
        
        GLOBAL_TOKEN_TRACKER.append({
            'turn': len(self.messages()),
            'context_tokens': current_len,
            'remaining_tokens': remaining_tokens,
            'event': 'before_step',
            'timestamp': time.time()
        })
    
    return await original_agent_step(self, max_new_tokens, retry_cjk)


def tracked_agent_append(self, turn, completion=None, additional_info=None):
    """为 Agent.append 添加追踪"""
    result = original_agent_append(self, turn, completion, additional_info)
    
    if GLOBAL_FRAMEWORK == 'react':
        current_len = len(self.context())
        GLOBAL_TOKEN_TRACKER.append({
            'turn': len(self.messages()),
            'context_tokens': current_len,
            'event': 'after_append',
            'role': turn.get('role', 'unknown'),
            'timestamp': time.time()
        })
    
    return result


# 应用 monkey patch
Agent.step = tracked_agent_step
Agent.append = tracked_agent_append


# ==================== HyMem 全面 Token 追踪注入 ====================

def inject_hymem_comprehensive_tracking():
    """
    HyMem 全面 Token 追踪 - 追踪所有组件的 Token 变化
    
    追踪的组件：
    1. Planner - 主规划器上下文
    2. Executor - 独立的执行器上下文
    3. Branch - 分支推理上下文
    4. Fold - 上下文折叠前后变化
    5. Memory - 三种记忆的变化
    """
    from agents import hymem_agent
    from agents.hymem_trigger import estimate_context_length
    
    # 保存原始方法的引用
    _original_run = hymem_agent.HyMemAgent.run
    _original_run_executor = hymem_agent.HyMemAgent.run_executor
    _original_run_branch = hymem_agent.HyMemAgent.run_branch
    _original_fold = hymem_agent.HyMemAgent._fold_planner_context
    
    async def tracked_run(self, problem: str):
        """追踪完整的 HyMem 执行过程"""
        
        # ========== 1. 初始化追踪 ==========
        problem_tokens = len(self.tokenizer.encode(problem)) if self.tokenizer else len(problem) // 4
        GLOBAL_TOKEN_TRACKER.append({
            'event': 'session_init',
            'problem_tokens': problem_tokens,
            'problem_length_chars': len(problem),
            'max_context_length': self.max_context_length,
            'max_planner_turns': self.max_planner_turns,
            'max_executor_turns': self.max_executor_turns,
            'max_branch_turns': self.max_branch_turns,
            'timestamp': time.time()
        })
        
        # 保存原始方法
        original_call_llm = self._call_llm
        original_run_executor = self.run_executor
        original_run_branch = self.run_branch
        original_fold = self._fold_planner_context
        
        # 内部计数器
        executor_call_count = [0]
        branch_call_count = [0]
        planner_turn_count = [0]
        
        # ========== 2. 追踪每次 LLM 调用（区分组件）==========
        async def tracked_call_llm(client, messages, max_tokens=4096, caller="unknown"):
            context_tokens = estimate_context_length(messages, self.tokenizer)
            
            # 根据 caller 确定事件类型和附加信息
            if 'planner' in caller:
                event_type = 'planner_llm_call'
                planner_turn_count[0] += 1
                extra_info = {
                    'planner_turn': planner_turn_count[0],
                    'planner_message_count': len(self.planner_messages),
                    'planner_turns_since_fold': self.memory_store.planner_turns_since_fold,
                    'fold_count': self.memory_store.fold_count,
                    # Memory 状态快照
                    'memory_episode_chars': len(str(self.memory_store.episode_memory or '')),
                    'memory_working_chars': len(str(self.memory_store.working_memory or '')),
                    'memory_tool_chars': len(str(self.memory_store.tool_memory or '')),
                    'tool_call_history_count': len(self.memory_store.tool_call_history),
                    'branch_history_count': len(self.memory_store.branch_history),
                }
            elif 'executor' in caller:
                event_type = 'executor_llm_call'
                extra_info = {
                    'executor_call_index': executor_call_count[0],
                    'executor_calls_this_session': self.memory_store.executor_calls_this_session,
                }
            elif 'branch' in caller:
                event_type = 'branch_llm_call'
                extra_info = {
                    'branch_call_index': branch_call_count[0],
                    'branch_count': self.fold_trigger.branch_count,
                }
            else:
                event_type = 'other_llm_call'
                extra_info = {}
            
            GLOBAL_TOKEN_TRACKER.append({
                'event': event_type,
                'caller': caller,
                'context_tokens': context_tokens,
                'remaining_tokens': self.max_context_length - context_tokens,
                'timestamp': time.time(),
                **extra_info
            })
            
            result = await original_call_llm(client, messages, max_tokens, caller)
            
            # 记录响应 token
            if result:
                response_tokens = len(self.tokenizer.encode(result)) if self.tokenizer else len(result) // 4
                GLOBAL_TOKEN_TRACKER.append({
                    'event': f'{event_type}_response',
                    'caller': caller,
                    'response_tokens': response_tokens,
                    'response_length_chars': len(result),
                    'timestamp': time.time()
                })
            
            return result
        
        # ========== 3. 追踪 Executor ==========
        async def tracked_run_executor(search_target: str) -> str:
            executor_call_count[0] += 1
            
            # 获取 Executor 开始时的上下文
            executor_messages = self._create_executor_prompt(search_target, self.memory_store.tool_memory)
            start_tokens = estimate_context_length(executor_messages, self.tokenizer)
            planner_tokens = estimate_context_length(self.planner_messages, self.tokenizer)
            
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'executor_start',
                'executor_call_index': executor_call_count[0],
                'executor_context_tokens': start_tokens,
                'executor_message_count': len(executor_messages),
                'search_target_length': len(search_target),
                'planner_context_tokens': planner_tokens,
                'tool_memory_chars': len(str(self.memory_store.tool_memory or '')),
                'timestamp': time.time()
            })
            
            result = await _original_run_executor(self, search_target)
            
            result_tokens = len(self.tokenizer.encode(result)) if self.tokenizer and result else 0
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'executor_end',
                'executor_call_index': executor_call_count[0],
                'result_tokens': result_tokens,
                'result_length_chars': len(result) if result else 0,
                'success': result and 'Error' not in result and len(result) > 50,
                'timestamp': time.time()
            })
            
            return result
        
        # ========== 4. 追踪 Branch ==========
        async def tracked_run_branch(description: str, prompt: str):
            branch_call_count[0] += 1
            
            # Branch 继承 Planner 上下文
            branch_messages = self._create_branch_prompt(
                self.planner_messages, description, prompt, self.memory_store
            )
            start_tokens = estimate_context_length(branch_messages, self.tokenizer)
            planner_tokens = estimate_context_length(self.planner_messages, self.tokenizer)
            
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'branch_start',
                'branch_call_index': branch_call_count[0],
                'branch_context_tokens': start_tokens,
                'inherited_planner_tokens': planner_tokens,
                'branch_overhead_tokens': start_tokens - planner_tokens,
                'description': description[:100],
                'branch_count': self.fold_trigger.branch_count,
                'timestamp': time.time()
            })
            
            result, success = await _original_run_branch(self, description, prompt)
            
            result_tokens = len(self.tokenizer.encode(str(result))) if self.tokenizer and result else 0
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'branch_end',
                'branch_call_index': branch_call_count[0],
                'result_tokens': result_tokens,
                'result_length_chars': len(str(result)) if result else 0,
                'success': success,
                'timestamp': time.time()
            })
            
            return result, success
        
        # ========== 5. 追踪 Fold（关键！）==========
        async def tracked_fold(problem: str):
            before_tokens = estimate_context_length(self.planner_messages, self.tokenizer)
            before_msg_count = len(self.planner_messages)
            before_memory = {
                'episode_chars': len(str(self.memory_store.episode_memory or '')),
                'working_chars': len(str(self.memory_store.working_memory or '')),
                'tool_chars': len(str(self.memory_store.tool_memory or '')),
                'tool_call_count': len(self.memory_store.tool_call_history),
                'branch_count': len(self.memory_store.branch_history),
            }
            
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'fold_start',
                'planner_tokens_before': before_tokens,
                'planner_message_count_before': before_msg_count,
                'memory_before': before_memory,
                'fold_index': self.memory_store.fold_count,
                'planner_turns_since_fold': self.memory_store.planner_turns_since_fold,
                'executor_calls_this_session': self.memory_store.executor_calls_this_session,
                'timestamp': time.time()
            })
            
            await _original_fold(self, problem)
            
            after_tokens = estimate_context_length(self.planner_messages, self.tokenizer)
            after_msg_count = len(self.planner_messages)
            after_memory = {
                'episode_chars': len(str(self.memory_store.episode_memory or '')),
                'working_chars': len(str(self.memory_store.working_memory or '')),
                'tool_chars': len(str(self.memory_store.tool_memory or '')),
            }
            
            tokens_reduced = before_tokens - after_tokens
            compression_ratio = after_tokens / before_tokens if before_tokens > 0 else 1.0
            
            GLOBAL_TOKEN_TRACKER.append({
                'event': 'fold_end',
                'planner_tokens_after': after_tokens,
                'planner_message_count_after': after_msg_count,
                'memory_after': after_memory,
                'tokens_reduced': tokens_reduced,
                'compression_ratio': compression_ratio,
                'messages_reduced': before_msg_count - after_msg_count,
                'fold_index': self.memory_store.fold_count,
                'timestamp': time.time()
            })
        
        # 替换方法
        self._call_llm = tracked_call_llm
        self.run_executor = tracked_run_executor
        self.run_branch = tracked_run_branch
        self._fold_planner_context = tracked_fold
        
        try:
            result = await _original_run(self, problem)
        finally:
            # 恢复原始方法
            self._call_llm = original_call_llm
            self.run_executor = original_run_executor
            self.run_branch = original_run_branch
            self._fold_planner_context = original_fold
        
        # ========== 6. 记录最终状态 ==========
        final_planner_tokens = estimate_context_length(self.planner_messages, self.tokenizer)
        GLOBAL_TOKEN_TRACKER.append({
            'event': 'session_end',
            'final_planner_tokens': final_planner_tokens,
            'total_planner_turns': planner_turn_count[0],
            'total_executor_calls': executor_call_count[0],
            'total_branch_calls': branch_call_count[0],
            'total_fold_count': self.memory_store.fold_count,
            'total_tool_calls': self.tool_manager.tool_calls_total if hasattr(self, 'tool_manager') else len(self.memory_store.tool_call_history),
            'final_memory': {
                'episode_chars': len(str(self.memory_store.episode_memory or '')),
                'working_chars': len(str(self.memory_store.working_memory or '')),
                'tool_chars': len(str(self.memory_store.tool_memory or '')),
            },
            'timestamp': time.time()
        })
        
        return result
    
    hymem_agent.HyMemAgent.run = tracked_run


# 保留旧函数名作为别名，向后兼容
def inject_hymem_tracking():
    """旧版追踪函数，现在调用全面追踪"""
    inject_hymem_comprehensive_tracking()


def inject_hymem_fold_tracking():
    """旧版 Fold 追踪函数，现在调用全面追踪"""
    inject_hymem_comprehensive_tracking()


def get_final_context_tokens(tracker: List[Dict], framework: str) -> int:
    """从追踪记录中提取最终上下文 token 数
    
    Args:
        tracker: Token 追踪记录列表
        framework: 框架类型 ('react' 或 'hymem')
    
    Returns:
        最终上下文 token 数
    """
    if not tracker:
        return 0
    
    last_event = tracker[-1]
    
    # HyMem 框架使用 session_end 事件
    if framework == 'hymem':
        if last_event.get('event') == 'session_end':
            return last_event.get('final_planner_tokens', 0)
        # 回退：查找最后一个包含 context_tokens 的事件
        for event in reversed(tracker):
            if 'context_tokens' in event:
                return event.get('context_tokens', 0)
            if 'final_planner_tokens' in event:
                return event.get('final_planner_tokens', 0)
            if 'planner_tokens_after' in event:
                return event.get('planner_tokens_after', 0)
        return 0
    
    # ReAct 框架
    return last_event.get('context_tokens', 0)


# ==================== 核心评测逻辑 ====================

async def eval_one_problem(row, config, tokenizer, model_name, framework, problem_order=0, is_continuous=False):
    """评测单个问题并追踪 token 使用"""
    global GLOBAL_TOKEN_TRACKER, GLOBAL_FRAMEWORK
    
    # 重置追踪器（单问题模式）或继续追踪（连续模式）
    if not is_continuous:
        GLOBAL_TOKEN_TRACKER = []
    
    GLOBAL_FRAMEWORK = framework
    
    context = TaskContext(
        config=config, 
        global_step=0, 
        server_host=model_name,
        server_port=0, 
        is_train=False, 
        tokenizer=tokenizer
    )
    
    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id', 'unknown')], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}
    
    # 连续模式：在问题前添加分隔符
    if is_continuous and problem_order > 0:
        row['extra_info']['query'] = f"--- Problem {problem_order + 1} ---\n{row['extra_info']['query']}"
    
    start_time = time.time()
    
    # 根据框架选择处理函数
    if framework == 'react':
        output = await react_process_item(item, context, CallAPI)
    else:  # hymem
        output = await hymem_process_item(item, context, CallAPI)
    
    session_time = time.time() - start_time
    
    # 提取统计信息
    stats = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}) if output else {}
    
    # 构建结果
    result = {
        'instance_id': row['extra_info'].get('instance_id', 'unknown'),
        'problem_order': problem_order,
        'data_source': row.get('data_source', 'unknown'),
        'score': stats.get('score', 0),
        'session_time': session_time,
        'status': 'success' if output else 'failed',
        'token_trajectory': copy.deepcopy(GLOBAL_TOKEN_TRACKER),
    }
    
    # 添加框架特定的统计
    if framework == 'react':
        result.update({
            'tool_calls': stats.get('tool_calls', 0),
            'total_turns': stats.get('total_turns', 0),
            'final_context_tokens': get_final_context_tokens(GLOBAL_TOKEN_TRACKER, framework)
        })
    else:  # hymem
        result.update({
            'tool_calls': stats.get('tool_calls_total', 0),
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'branch_success': stats.get('branch_success', 0),
            'fold_count': stats.get('fold_count', 0),
            'replan_count': stats.get('replan_count', 0),
            'final_context_tokens': get_final_context_tokens(GLOBAL_TOKEN_TRACKER, framework)
        })
    
    return result


async def run_single_mode(problems, args):
    """单问题模式：每次解决一个问题，清空上下文"""
    print(f"\n{'='*60}")
    print(f"运行模式: 单问题模式 (Single Mode)")
    print(f"框架: {args.framework.upper()}")
    print(f"问题数量: {len(problems)}")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    # 基础配置
    plugin_config = {
        'workflow': args.workflow,
        'max_turn': args.max_turn,
        'val_max_turn': args.val_max_turn,
        'max_session': args.max_session,
        'val_max_session': args.val_max_session,
        'session_timeout': 5400,
        'process_reward': None,
        'max_traj': None,
        'must_finish': False,
        'double_check': False,
        'must_search': False,
    }
    
    # 添加 HyMem 特定配置（与单问题/连续模式保持一致）
    if args.framework == 'hymem':
        plugin_config.update({
            'hymem_max_planner_turns': args.hymem_max_planner_turns,
            'hymem_max_executor_turns': args.hymem_max_executor_turns,
            'hymem_max_branch_turns': args.hymem_max_branch_turns,
            'hymem_max_branches': args.hymem_max_branches,
            'hymem_max_folds': args.hymem_max_folds,
            'hymem_search_topk': args.hymem_search_topk,
            'hymem_llm_model': args.model_name,
            'hymem_aux_model': args.hymem_aux_model or args.model_name,
            'search_base_url': args.local_search_url,
        })
    
    # 根据框架添加特定配置
    if args.framework == 'hymem':
        plugin_config.update({
            'hymem_max_planner_turns': args.hymem_max_planner_turns,
            'hymem_max_executor_turns': args.hymem_max_executor_turns,
            'hymem_max_branch_turns': args.hymem_max_branch_turns,
            'hymem_max_branches': args.hymem_max_branches,
            'hymem_max_folds': args.hymem_max_folds,
            'hymem_search_topk': args.hymem_search_topk,
            'hymem_llm_model': args.model_name,
            'hymem_aux_model': args.hymem_aux_model or args.model_name,
            'search_base_url': args.local_search_url,
        })
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': plugin_config
        }}
    })
    
    results = []
    for i, row in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] 处理问题: {row['extra_info'].get('instance_id', 'unknown')}")
        
        result = await eval_one_problem(
            row, config, tokenizer, args.model_name, 
            args.framework, problem_order=i, is_continuous=False
        )
        
        results.append(result)
        print(f"  得分: {result['score']:.2f}, Token轨迹点: {len(result['token_trajectory'])}, "
              f"最终Token数: {result['final_context_tokens']}")
    
    return results


async def run_continuous_mode(problems, args):
    """连续问题模式：连续解决多个问题，不清空上下文"""
    print(f"\n{'='*60}")
    print(f"运行模式: 连续问题模式 (Continuous Mode)")
    print(f"框架: {args.framework.upper()}")
    print(f"问题数量: {len(problems)}")
    print(f"警告: 此模式下上下文会持续累积，可能触发token限制")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    # 基础配置
    plugin_config = {
        'workflow': args.workflow,
        'max_turn': args.max_turn,
        'val_max_turn': args.val_max_turn,
        'max_session': args.max_session,
        'val_max_session': args.val_max_session,
        'session_timeout': 5400,
        'process_reward': None,
        'max_traj': None,
        'must_finish': False,
        'double_check': False,
        'must_search': False,
    }
    
    # 根据框架添加特定配置
    if args.framework == 'hymem':
        plugin_config.update({
            'hymem_max_planner_turns': args.hymem_max_planner_turns,
            'hymem_max_executor_turns': args.hymem_max_executor_turns,
            'hymem_max_branch_turns': args.hymem_max_branch_turns,
            'hymem_max_branches': args.hymem_max_branches,
            'hymem_max_folds': args.hymem_max_folds,
            'hymem_search_topk': args.hymem_search_topk,
            'hymem_llm_model': args.model_name,
            'hymem_aux_model': args.hymem_aux_model or args.model_name,
            'search_base_url': args.local_search_url,
        })
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': plugin_config
        }}
    })
    
    results = []
    global GLOBAL_TOKEN_TRACKER
    GLOBAL_TOKEN_TRACKER = []  # 初始化一次
    
    for i, row in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] 处理问题: {row['extra_info'].get('instance_id', 'unknown')}")
        
        prev_token_count = len(GLOBAL_TOKEN_TRACKER)
        
        result = await eval_one_problem(
            row, config, tokenizer, args.model_name, 
            args.framework, problem_order=i, is_continuous=True
        )
        
        new_token_points = len(GLOBAL_TOKEN_TRACKER) - prev_token_count
        
        results.append(result)
        print(f"  得分: {result['score']:.2f}, 新增Token轨迹点: {new_token_points}, "
              f"累积Token数: {result['final_context_tokens']}")
        
        # 如果token数过高，提前终止
        if result['final_context_tokens'] > (args.prompt_length + args.response_length) * 0.9:
            print(f"\n警告: Token数接近上限，提前终止连续模式")
            break
    
    return results


# ==================== 历史轨迹累积模式 ====================

def format_accumulated_history(histories: List[Dict], max_total_tokens: int = 60000) -> str:
    """将历史轨迹格式化为可注入的上下文
    
    Args:
        histories: 历史任务列表
        max_total_tokens: 总注入token阈值（所有任务加起来不超过这个值）
    """
    if not histories:
        return ""
    
    formatted = []
    total_char_count = 0
    max_total_chars = max_total_tokens * 4  # 粗略估计 1 token ≈ 4 chars
    
    for h in histories:
        if total_char_count >= max_total_chars:
            formatted.append("[... earlier tasks omitted due to token limit ...]\n")
            break
            
        task_summary = f"""=== Completed Task #{h['task_index']}: {h['task_id']} ===
Question: {h['original_query']}

Execution Trace:
"""
        history = h.get('history', [])
        for entry in history:
            if total_char_count >= max_total_chars:
                task_summary += "[... truncated ...]\n"
                break
            role = entry.get('role', '')
            content = entry.get('content', '')
            
            # 跳过系统提示词
            if role == 'system':
                continue
            
            if role == 'assistant':
                line = f"[Agent]: {content}\n"
            elif role == 'user':
                line = f"[Observation]: {content}\n"
            elif role == 'trace':
                line = f"[Trace]: {content}\n"
            else:
                continue
            
            task_summary += line
            total_char_count += len(line)
        
        prediction = h.get('prediction', '')
        if isinstance(prediction, tuple):
            prediction = prediction[0] if prediction else ''
        task_summary += f"\nResult: Score={h.get('score', 0)}, Answer={str(prediction)[:500]}"
        task_summary += "\n" + "="*50 + "\n"
        
        formatted.append(task_summary)
    
    return "\n".join(formatted)


def analyze_performance_decay(results: List[Dict]):
    """分析随着历史累积，性能衰减趋势"""
    print(f"\n{'='*60}")
    print("性能衰减分析 (Performance Decay Analysis)")
    print(f"{'='*60}")
    
    if len(results) < 2:
        print("样本数不足，无法分析趋势")
        return {}
    
    # 按问题顺序分析
    print("\n问题序号 | 得分 | 最终Token | 注入历史数 | 注入Token")
    print("-" * 60)
    for r in results:
        print(f"  {r['problem_order']+1:3d}    | {r['score']:.2f} | {r['final_context_tokens']:8d} | "
              f"{r['num_injected_histories']:7d}    | {r['injected_history_tokens']:8d}")
    
    # 计算趋势
    scores = [r['score'] for r in results]
    injected_tokens = [r['injected_history_tokens'] for r in results]
    
    analysis = {
        'scores_by_order': scores,
        'injected_tokens_by_order': injected_tokens,
    }
    
    # 前半 vs 后半对比
    mid = len(results) // 2
    if mid > 0:
        first_half_score = np.mean(scores[:mid])
        second_half_score = np.mean(scores[mid:])
        
        print(f"\n前半部分平均得分: {first_half_score:.4f}")
        print(f"后半部分平均得分: {second_half_score:.4f}")
        decay_rate = ((second_half_score - first_half_score) / first_half_score * 100) if first_half_score > 0 else 0
        print(f"性能变化: {decay_rate:.2f}%")
        
        analysis['first_half_avg_score'] = first_half_score
        analysis['second_half_avg_score'] = second_half_score
        analysis['decay_rate'] = decay_rate
    
    # 相关性分析
    if len(results) >= 3:
        correlation = np.corrcoef(injected_tokens, scores)[0, 1]
        if not np.isnan(correlation):
            print(f"\n注入Token数与得分的相关系数: {correlation:.4f}")
            if correlation < -0.3:
                print("⚠️ 负相关: 随着注入历史增多，性能呈下降趋势")
            elif correlation > 0.3:
                print("✓ 正相关: 历史上下文可能对后续任务有帮助")
            else:
                print("○ 弱相关: 历史累积对性能影响不明显")
            analysis['correlation'] = correlation
    
    return analysis


async def run_history_accumulate_mode(problems, args):
    """历史轨迹累积模式：将前面问题的完整历史注入后续问题的上下文"""
    print(f"\n{'='*60}")
    print(f"运行模式: 历史轨迹累积模式 (History Accumulate Mode)")
    print(f"框架: {args.framework.upper()}")
    print(f"问题数量: {len(problems)}")
    print(f"历史保留数: 前 {args.history_window} 个问题的历史")
    print(f"{'='*60}\n")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    # 基础配置
    plugin_config = {
        'workflow': args.workflow,
        'max_turn': args.max_turn,
        'val_max_turn': args.val_max_turn,
        'max_session': args.max_session,
        'val_max_session': args.val_max_session,
        'session_timeout': 5400,
        'process_reward': None,
        'max_traj': None,
        'must_finish': False,
        'double_check': False,
        'must_search': False,
    }
    
    # 添加 HyMem 特定配置
    if args.framework == 'hymem':
        plugin_config.update({
            'hymem_max_planner_turns': args.hymem_max_planner_turns,
            'hymem_max_executor_turns': args.hymem_max_executor_turns,
            'hymem_max_branch_turns': args.hymem_max_branch_turns,
            'hymem_max_branches': args.hymem_max_branches,
            'hymem_max_folds': args.hymem_max_folds,
            'hymem_search_topk': args.hymem_search_topk,
            'hymem_llm_model': args.model_name,
            'hymem_aux_model': args.hymem_aux_model or args.model_name,
            'search_base_url': args.local_search_url,
        })
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': plugin_config
        }}
    })
    
    results = []
    accumulated_histories = []  # 存储前面问题的完整历史
    
    global GLOBAL_TOKEN_TRACKER, GLOBAL_FRAMEWORK
    GLOBAL_FRAMEWORK = args.framework
    
    for i, row in enumerate(problems):
        instance_id = row['extra_info'].get('instance_id', f'problem_{i}')
        print(f"\n[{i+1}/{len(problems)}] 处理问题: {instance_id}")
        
        # 重置 token 追踪器
        GLOBAL_TOKEN_TRACKER = []
        
        # 保存原始问题
        original_query = row['extra_info']['query']
        original_problem = row['extra_info'].get('problem_statement', original_query)
        
        # 计算注入历史的 token 数
        injected_history_tokens = 0
        
        if accumulated_histories:
            # 选取最近 history_window 个问题的历史
            recent_histories = accumulated_histories[-args.history_window:]
            
            # 格式化历史轨迹
            history_context = format_accumulated_history(recent_histories)
            injected_history_tokens = len(tokenizer.encode(history_context))
            
            # 注入历史到当前问题
            modified_query = f"""[Previous Task Context - {len(recent_histories)} completed tasks]
{history_context}

[Current Task - Task #{i+1}]
{original_query}

Note: The above context contains information from previously completed tasks. Focus on the current task."""
            
            # 创建新的 row 副本，避免修改原数据
            row = copy.deepcopy(row)
            row['extra_info']['query'] = modified_query
            row['extra_info']['problem_statement'] = modified_query
            
            print(f"  注入历史: {len(recent_histories)} 个任务, {injected_history_tokens} tokens")
        else:
            print(f"  首个任务，无历史注入")
        
        # 执行评测
        start_time = time.time()
        
        context = TaskContext(
            config=config, 
            global_step=0, 
            server_host=args.model_name,
            server_port=0, 
            is_train=False, 
            tokenizer=tokenizer
        )
        
        item = DataProto()
        item.non_tensor_batch = {
            'ability': np.array([row['ability']], dtype=object),
            'extra_info': np.array([row['extra_info']], dtype=object),
            'uid': np.array([instance_id], dtype=object),
            'reward_model': np.array([row['reward_model']], dtype=object),
        }
        item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}
        
        try:
            if args.framework == 'hymem':
                output = await hymem_process_item(item, context, CallAPI)
            else:
                output = await react_process_item(item, context, CallAPI)
        except Exception as e:
            print(f"  ⚠️ 执行失败: {e}")
            output = None
        
        session_time = time.time() - start_time
        
        # 提取统计和历史
        stats = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}) if output else {}
        
        # 获取完整对话历史
        history = stats.get('messages', [])  # 从 stats 中获取完整 messages
        if not history and output:
            # 降级：使用 message_str
            message_str = output.non_tensor_batch.get('message_str', [''])[0]
            if message_str:
                history = [{'role': 'trace', 'content': message_str}]
        
        # 保存当前问题的历史轨迹（用于注入到后续问题）
        prediction = stats.get('prediction', '')
        if hasattr(output, 'non_tensor_batch'):
            # 尝试从 extra_data 获取预测
            extra_data = output.non_tensor_batch.get('extra_data', [{}])[0]
            if 'prediction' in extra_data:
                prediction = extra_data['prediction']
        
        current_history = {
            'task_id': instance_id,
            'task_index': i + 1,
            'original_query': original_query,
            'history': history,
            'score': stats.get('get_final_score', stats.get('score', 0)),
            'prediction': str(prediction)[:500],
            'tool_calls': stats.get('tool_calls_total', stats.get('tool_calls', 0)),
        }
        accumulated_histories.append(current_history)
        
        # 构建结果
        final_tokens = get_final_context_tokens(GLOBAL_TOKEN_TRACKER, args.framework)
        
        result = {
            'instance_id': instance_id,
            'problem_order': i,
            'original_query': original_query[:500],
            'data_source': row.get('data_source', 'unknown'),
            'score': stats.get('get_final_score', stats.get('score', 0)),
            'session_time': session_time,
            'status': 'success' if output else 'failed',
            'token_trajectory': copy.deepcopy(GLOBAL_TOKEN_TRACKER),
            'final_context_tokens': final_tokens,
            # 历史累积相关统计
            'num_injected_histories': len(accumulated_histories) - 1,
            'injected_history_tokens': injected_history_tokens,
        }
        
        # 添加框架特有字段
        if args.framework == 'react':
            result.update({
                'tool_calls': stats.get('tool_calls', 0),
                'total_turns': stats.get('total_turns', len(GLOBAL_TOKEN_TRACKER)),
            })
        else:  # hymem
            result.update({
                'tool_calls': stats.get('tool_calls_total', 0),
                'planner_turns': stats.get('planner_turns', 0),
                'executor_calls': stats.get('executor_calls', 0),
                'branch_calls': stats.get('branch_calls', 0),
                'fold_count': stats.get('fold_count', 0),
            })
        
        results.append(result)
        print(f"  得分: {result['score']:.2f}, 最终Token: {result['final_context_tokens']}, "
              f"工具调用: {result['tool_calls']}")
    
    # 分析性能衰减趋势
    decay_analysis = analyze_performance_decay(results)
    
    return results, accumulated_histories, decay_analysis


def parse_args():
    parser = argparse.ArgumentParser(description='上下文追踪对比实验：ReAct vs HyMem')
    
    # 数据源选择
    parser.add_argument('--data_source', default='auto', choices=['auto', 'bc', 'gaia'],
                        help='数据源: auto (自动检测) / bc (BrowseComp) / gaia (GAIA)')
    parser.add_argument('--data_path', default='/data1/test/xq/dataset/bc_test.parquet',
                        help='测试数据路径')
    parser.add_argument('--output_dir', default='results/context_tracking',
                        help='结果输出目录')
    parser.add_argument('--num_problems', type=int, default=5,
                        help='测试问题数量')
    
    # GAIA 特定参数
    parser.add_argument('--gaia_root', default='/data1/test/xq/dataset/GAIA',
                        help='GAIA 数据集根目录（用于文件附件）')
    parser.add_argument('--text_only', action='store_true',
                        help='[GAIA] 仅评测无附件的纯文本问题')
    
    # 选择特定任务ID
    parser.add_argument('--instance_ids', type=str, default=None,
                        help='指定任务ID列表，用逗号分隔，例如: "89,718,1161,203"')
    parser.add_argument('--enable_fold_tracking', action='store_true',
                        help='[已废弃] 现在默认启用全面追踪，此参数保留仅为兼容性')
    
    # 框架选择
    parser.add_argument('--framework', default='react', choices=['react', 'hymem'],
                        help='测试框架: react 或 hymem')
    
    # 测试模式
    parser.add_argument('--test_mode', default='single', 
                        choices=['single', 'continuous', 'history_accumulate'],
                        help='测试模式: single (每次清空) / continuous (累积) / history_accumulate (历史注入)')
    
    # 历史累积模式参数
    parser.add_argument('--history_window', type=int, default=5,
                        help='历史轨迹累积窗口大小（保留最近N个任务的历史）')
    
    # 模型配置
    parser.add_argument('--model_name', default='deepseek-reasoner',
                        help='模型名称')
    parser.add_argument('--prompt_length', type=int, default=98304,
                        help='最大提示长度')
    parser.add_argument('--response_length', type=int, default=65536,
                        help='最大响应长度')
    
    # Agent 通用配置
    parser.add_argument('--workflow', default='search',
                        help='工作流: search (ReAct) 或 hymem (HyMem)')
    parser.add_argument('--max_turn', type=int, default=40,
                        help='最大轮数')
    parser.add_argument('--val_max_turn', type=int, default=60,
                        help='验证最大轮数')
    parser.add_argument('--max_session', type=int, default=10,
                        help='最大会话数')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='验证最大会话数')
    
    # HyMem 框架特定配置
    parser.add_argument('--hymem_max_planner_turns', type=int, default=12,
                        help='HyMem最大Planner轮数')
    parser.add_argument('--hymem_max_executor_turns', type=int, default=10,
                        help='HyMem最大Executor轮数')
    parser.add_argument('--hymem_max_branch_turns', type=int, default=3,
                        help='HyMem每个Branch最大轮数')
    parser.add_argument('--hymem_max_branches', type=int, default=3,
                        help='HyMem最大Branch数量')
    parser.add_argument('--hymem_max_folds', type=int, default=7,
                        help='HyMem最大折叠次数')
    parser.add_argument('--hymem_search_topk', type=int, default=5,
                        help='HyMem搜索TopK结果数')
    parser.add_argument('--hymem_aux_model', default=None,
                        help='HyMem辅助模型（用于记忆压缩）')
    
    # 搜索配置
    parser.add_argument('--search_mode', default='online', choices=['local', 'online', 'auto'],
                        help='搜索模式')
    parser.add_argument('--local_search_url', default='http://localhost:8021',
                        help='本地搜索服务器URL')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置环境变量
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url
    if args.search_mode == 'online':
        os.environ["USE_ONLINE_SEARCH"] = "true"
        os.environ["SEARCH_MODE"] = "online"
    elif args.search_mode == 'local':
        os.environ["SEARCH_MODE"] = "local"
    
    # 打印配置信息
    print(f"\n{'='*60}")
    print(f"上下文追踪实验 - {args.framework.upper()} 框架")
    print(f"{'='*60}")
    print(f"测试模式: {args.test_mode}")
    print(f"数据源: {args.data_source.upper()}")
    print(f"数据路径: {args.data_path}")
    if args.data_source == 'gaia':
        print(f"GAIA根目录: {args.gaia_root}")
        print(f"仅文本问题: {args.text_only}")
    print(f"模型: {args.model_name}")
    print(f"问题数量: {args.num_problems}")
    print(f"搜索模式: {args.search_mode}")
    print(f"搜索URL: {args.local_search_url}")
    print(f"Token限制: prompt={args.prompt_length}, response={args.response_length}")
    
    if args.framework == 'hymem':
        print(f"\nHyMem 配置:")
        print(f"  - Max Planner Turns: {args.hymem_max_planner_turns}")
        print(f"  - Max Executor Turns: {args.hymem_max_executor_turns}")
        print(f"  - Max Branch Turns: {args.hymem_max_branch_turns}")
        print(f"  - Max Branches: {args.hymem_max_branches}")
        print(f"  - Max Folds: {args.hymem_max_folds}")
        print(f"  - Search TopK: {args.hymem_search_topk}")
        if args.hymem_aux_model:
            print(f"  - Aux Model: {args.hymem_aux_model}")
    else:
        print(f"\nReAct 配置:")
        print(f"  - Max Turn: {args.val_max_turn}")
        print(f"  - Workflow: {args.workflow}")
    
    print(f"{'='*60}\n")
    
    # 为 HyMem 注入全面追踪
    if args.framework == 'hymem':
        print("✓ 启用 HyMem 全面 Token 追踪模式")
        print("  追踪组件: Planner / Executor / Branch / Fold / Memory")
        inject_hymem_comprehensive_tracking()
    
    # 加载数据
    print(f"\n加载数据: {args.data_path}")
    df = load_dataset(args)
    
    # 按 instance_ids 筛选
    if args.instance_ids:
        instance_id_list = [id.strip() for id in args.instance_ids.split(',')]
        print(f"\n筛选指定的任务ID: {instance_id_list}")
        
        # 筛选匹配的行
        df = df[df.apply(lambda row: str(row['extra_info'].get('instance_id', '')) in instance_id_list, axis=1)]
        
        print(f"找到 {len(df)} 个匹配的任务")
        
        if len(df) == 0:
            print("错误: 未找到匹配的任务ID")
            return
        
        # 按指定顺序排序
        df['_sort_order'] = df.apply(
            lambda row: instance_id_list.index(str(row['extra_info'].get('instance_id', ''))) 
            if str(row['extra_info'].get('instance_id', '')) in instance_id_list else 999,
            axis=1
        )
        df = df.sort_values('_sort_order').drop(columns=['_sort_order'])
    elif len(df) > args.num_problems:
        df = df.head(args.num_problems)
        print(f"限制为前 {args.num_problems} 个问题")
    
    problems = [df.iloc[i] for i in range(len(df))]
    
    # 运行评测
    decay_analysis = None
    accumulated_histories = None
    
    if args.test_mode == 'single':
        results = asyncio.run(run_single_mode(problems, args))
    elif args.test_mode == 'continuous':
        results = asyncio.run(run_continuous_mode(problems, args))
    elif args.test_mode == 'history_accumulate':
        print(f"\n历史累积模式配置:")
        print(f"  - 历史窗口大小: {args.history_window}")
        results, accumulated_histories, decay_analysis = asyncio.run(
            run_history_accumulate_mode(problems, args)
        )
    
    # 计算统计
    avg_score = np.mean([r['score'] for r in results])
    avg_session_time = np.mean([r['session_time'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"\n{'='*60}")
    print(f"评测完成 - {args.framework.upper()} 框架")
    print(f"{'='*60}")
    print(f"完成任务: {len(results)}")
    print(f"成功任务: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"平均得分: {avg_score:.4f}")
    print(f"平均时间: {avg_session_time:.1f}秒")
    
    if args.framework == 'react':
        avg_tool_calls = np.mean([r['tool_calls'] for r in results])
        avg_turns = np.mean([r['total_turns'] for r in results])
        avg_final_tokens = np.mean([r['final_context_tokens'] for r in results])
        print(f"\nReAct 统计:")
        print(f"  - 平均工具调用: {avg_tool_calls:.1f}")
        print(f"  - 平均轮数: {avg_turns:.1f}")
        print(f"  - 平均最终Token数: {avg_final_tokens:.0f}")
    else:
        avg_planner = np.mean([r['planner_turns'] for r in results])
        avg_executor = np.mean([r['executor_calls'] for r in results])
        avg_branch = np.mean([r['branch_calls'] for r in results])
        avg_fold = np.mean([r['fold_count'] for r in results])
        avg_final_tokens = np.mean([r['final_context_tokens'] for r in results])
        print(f"\nHyMem 统计:")
        print(f"  - 平均Planner轮数: {avg_planner:.1f}")
        print(f"  - 平均Executor调用: {avg_executor:.1f}")
        print(f"  - 平均Branch调用: {avg_branch:.1f}")
        print(f"  - 平均折叠次数: {avg_fold:.1f}")
        print(f"  - 平均最终Token数: {avg_final_tokens:.0f}")
    
    # 历史累积模式额外统计
    if args.test_mode == 'history_accumulate' and decay_analysis:
        print(f"\n历史累积模式统计:")
        avg_injected = np.mean([r.get('injected_history_tokens', 0) for r in results])
        print(f"  - 平均注入Token: {avg_injected:.0f}")
        if 'first_half_avg_score' in decay_analysis:
            print(f"  - 前半部分平均得分: {decay_analysis['first_half_avg_score']:.4f}")
            print(f"  - 后半部分平均得分: {decay_analysis['second_half_avg_score']:.4f}")
            print(f"  - 性能变化: {decay_analysis['decay_rate']:.2f}%")
        if 'correlation' in decay_analysis:
            print(f"  - 注入Token与得分相关系数: {decay_analysis['correlation']:.4f}")
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"{args.framework}_{args.test_mode}_{timestamp}.json"
    
    # 构建元数据
    metadata = {
        'framework': args.framework,
        'test_mode': args.test_mode,
        'timestamp': timestamp,
        'model_name': args.model_name,
        'prompt_length': args.prompt_length,
        'response_length': args.response_length,
        'num_problems': len(results),
        'success_count': success_count,
        'avg_score': float(avg_score),
        'avg_session_time': float(avg_session_time),
        'search_mode': args.search_mode,
    }
    
    # 添加框架特定配置
    if args.framework == 'hymem':
        metadata.update({
            'hymem_max_planner_turns': args.hymem_max_planner_turns,
            'hymem_max_executor_turns': args.hymem_max_executor_turns,
            'hymem_max_branch_turns': args.hymem_max_branch_turns,
            'hymem_max_branches': args.hymem_max_branches,
            'hymem_max_folds': args.hymem_max_folds,
            'hymem_search_topk': args.hymem_search_topk,
        })
    else:
        metadata.update({
            'max_turn': args.max_turn,
            'val_max_turn': args.val_max_turn,
            'workflow': args.workflow,
        })
    
    # 历史累积模式配置
    if args.test_mode == 'history_accumulate':
        metadata['history_window'] = args.history_window
    
    output_data = {
        'metadata': metadata,
        'problems': results
    }
    
    # 历史累积模式额外数据
    if args.test_mode == 'history_accumulate':
        if decay_analysis:
            output_data['decay_analysis'] = decay_analysis
        if accumulated_histories:
            # 简化历史保存（避免文件过大）
            output_data['accumulated_histories_summary'] = [
                {
                    'task_id': h['task_id'],
                    'task_index': h['task_index'],
                    'score': h['score'],
                    'original_query': h['original_query'][:300],
                }
                for h in accumulated_histories
            ]
    
    # HyMem 全面 Token 分析
    if args.framework == 'hymem':
        hymem_analysis = analyze_hymem_token_flow(results)
        output_data['hymem_token_analysis'] = hymem_analysis
        
        # 打印 HyMem Token 流分析
        print(f"\n{'='*60}")
        print(f"HyMem Token 流分析")
        print(f"{'='*60}")
        
        # Planner 统计
        print(f"\n[Planner 组件]")
        print(f"  总 LLM 调用次数: {hymem_analysis['planner']['total_calls']}")
        print(f"  平均上下文 Token: {hymem_analysis['planner']['avg_context_tokens']:.0f}")
        print(f"  最大上下文 Token: {hymem_analysis['planner']['max_context_tokens']}")
        print(f"  平均响应 Token: {hymem_analysis['planner']['avg_response_tokens']:.0f}")
        
        # Executor 统计
        print(f"\n[Executor 组件]")
        print(f"  总调用次数: {hymem_analysis['executor']['total_calls']}")
        print(f"  平均上下文 Token: {hymem_analysis['executor']['avg_context_tokens']:.0f}")
        print(f"  成功率: {hymem_analysis['executor']['success_rate']:.1%}")
        
        # Branch 统计
        print(f"\n[Branch 组件]")
        print(f"  总调用次数: {hymem_analysis['branch']['total_calls']}")
        print(f"  平均上下文 Token: {hymem_analysis['branch']['avg_context_tokens']:.0f}")
        print(f"  平均继承 Planner Token: {hymem_analysis['branch']['avg_inherited_tokens']:.0f}")
        print(f"  成功率: {hymem_analysis['branch']['success_rate']:.1%}")
        
        # Fold 统计
        print(f"\n[Fold 压缩]")
        print(f"  总 Fold 次数: {hymem_analysis['fold']['total_folds']}")
        if hymem_analysis['fold']['total_folds'] > 0:
            print(f"  平均压缩比: {hymem_analysis['fold']['avg_compression_ratio']:.2%}")
            print(f"  平均节省 Token: {hymem_analysis['fold']['avg_tokens_saved']:.0f}")
            print(f"  总节省 Token: {hymem_analysis['fold']['total_tokens_saved']}")
        
        # Memory 统计
        print(f"\n[Memory 状态]")
        print(f"  平均 Episode Memory: {hymem_analysis['memory']['avg_episode_chars']:.0f} chars")
        print(f"  平均 Working Memory: {hymem_analysis['memory']['avg_working_chars']:.0f} chars")
        print(f"  平均 Tool Memory: {hymem_analysis['memory']['avg_tool_chars']:.0f} chars")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    print(f"Token轨迹数据可用于后续分析")


def analyze_hymem_token_flow(results):
    """
    分析 HyMem 框架的全面 Token 流动
    
    分析维度：
    1. Planner 组件：调用次数、上下文增长、响应token
    2. Executor 组件：调用次数、独立上下文、成功率
    3. Branch 组件：调用次数、继承token、开销token
    4. Fold 压缩：压缩次数、压缩比、节省token
    5. Memory 变化：三种memory的增长趋势
    """
    # 收集所有事件
    all_planner_calls = []
    all_planner_responses = []
    all_executor_starts = []
    all_executor_ends = []
    all_branch_starts = []
    all_branch_ends = []
    all_fold_starts = []
    all_fold_ends = []
    all_session_ends = []
    
    for result in results:
        trajectory = result.get('token_trajectory', [])
        
        for event in trajectory:
            event_type = event.get('event', '')
            
            if event_type == 'planner_llm_call':
                all_planner_calls.append(event)
            elif event_type == 'planner_llm_call_response':
                all_planner_responses.append(event)
            elif event_type == 'executor_start':
                all_executor_starts.append(event)
            elif event_type == 'executor_end':
                all_executor_ends.append(event)
            elif event_type == 'branch_start':
                all_branch_starts.append(event)
            elif event_type == 'branch_end':
                all_branch_ends.append(event)
            elif event_type == 'fold_start':
                all_fold_starts.append(event)
            elif event_type == 'fold_end':
                all_fold_ends.append(event)
            elif event_type == 'session_end':
                all_session_ends.append(event)
    
    # Planner 分析
    planner_context_tokens = [e.get('context_tokens', 0) for e in all_planner_calls]
    planner_response_tokens = [e.get('response_tokens', 0) for e in all_planner_responses]
    
    planner_analysis = {
        'total_calls': len(all_planner_calls),
        'avg_context_tokens': np.mean(planner_context_tokens) if planner_context_tokens else 0,
        'max_context_tokens': max(planner_context_tokens) if planner_context_tokens else 0,
        'min_context_tokens': min(planner_context_tokens) if planner_context_tokens else 0,
        'avg_response_tokens': np.mean(planner_response_tokens) if planner_response_tokens else 0,
        'total_response_tokens': sum(planner_response_tokens),
    }
    
    # Executor 分析
    executor_context_tokens = [e.get('executor_context_tokens', 0) for e in all_executor_starts]
    executor_success = [e.get('success', False) for e in all_executor_ends]
    
    executor_analysis = {
        'total_calls': len(all_executor_starts),
        'avg_context_tokens': np.mean(executor_context_tokens) if executor_context_tokens else 0,
        'max_context_tokens': max(executor_context_tokens) if executor_context_tokens else 0,
        'success_rate': sum(executor_success) / len(executor_success) if executor_success else 0,
        'success_count': sum(executor_success),
        'fail_count': len(executor_success) - sum(executor_success),
    }
    
    # Branch 分析
    branch_context_tokens = [e.get('branch_context_tokens', 0) for e in all_branch_starts]
    branch_inherited_tokens = [e.get('inherited_planner_tokens', 0) for e in all_branch_starts]
    branch_overhead_tokens = [e.get('branch_overhead_tokens', 0) for e in all_branch_starts]
    branch_success = [e.get('success', False) for e in all_branch_ends]
    
    branch_analysis = {
        'total_calls': len(all_branch_starts),
        'avg_context_tokens': np.mean(branch_context_tokens) if branch_context_tokens else 0,
        'avg_inherited_tokens': np.mean(branch_inherited_tokens) if branch_inherited_tokens else 0,
        'avg_overhead_tokens': np.mean(branch_overhead_tokens) if branch_overhead_tokens else 0,
        'success_rate': sum(branch_success) / len(branch_success) if branch_success else 0,
        'success_count': sum(branch_success),
    }
    
    # Fold 分析
    fold_tokens_before = [e.get('planner_tokens_before', 0) for e in all_fold_starts]
    fold_tokens_after = [e.get('planner_tokens_after', 0) for e in all_fold_ends]
    fold_tokens_saved = [e.get('tokens_reduced', 0) for e in all_fold_ends]
    fold_compression_ratios = [e.get('compression_ratio', 1.0) for e in all_fold_ends]
    
    fold_analysis = {
        'total_folds': len(all_fold_starts),
        'avg_tokens_before': np.mean(fold_tokens_before) if fold_tokens_before else 0,
        'avg_tokens_after': np.mean(fold_tokens_after) if fold_tokens_after else 0,
        'avg_tokens_saved': np.mean(fold_tokens_saved) if fold_tokens_saved else 0,
        'total_tokens_saved': sum(fold_tokens_saved),
        'avg_compression_ratio': np.mean(fold_compression_ratios) if fold_compression_ratios else 1.0,
        'min_compression_ratio': min(fold_compression_ratios) if fold_compression_ratios else 1.0,
        'max_compression_ratio': max(fold_compression_ratios) if fold_compression_ratios else 1.0,
    }
    
    # Memory 分析（从 planner_llm_call 事件中提取）
    episode_chars = [e.get('memory_episode_chars', 0) for e in all_planner_calls]
    working_chars = [e.get('memory_working_chars', 0) for e in all_planner_calls]
    tool_chars = [e.get('memory_tool_chars', 0) for e in all_planner_calls]
    
    memory_analysis = {
        'avg_episode_chars': np.mean(episode_chars) if episode_chars else 0,
        'max_episode_chars': max(episode_chars) if episode_chars else 0,
        'avg_working_chars': np.mean(working_chars) if working_chars else 0,
        'max_working_chars': max(working_chars) if working_chars else 0,
        'avg_tool_chars': np.mean(tool_chars) if tool_chars else 0,
        'max_tool_chars': max(tool_chars) if tool_chars else 0,
    }
    
    # Session 整体统计
    final_tokens = [e.get('final_planner_tokens', 0) for e in all_session_ends]
    
    session_analysis = {
        'total_problems': len(results),
        'avg_final_planner_tokens': np.mean(final_tokens) if final_tokens else 0,
        'max_final_planner_tokens': max(final_tokens) if final_tokens else 0,
    }
    
    return {
        'planner': planner_analysis,
        'executor': executor_analysis,
        'branch': branch_analysis,
        'fold': fold_analysis,
        'memory': memory_analysis,
        'session': session_analysis,
    }


def analyze_fold_events(results):
    """分析所有任务中的 Fold 事件（兼容旧版格式）"""
    all_folds = []
    
    for result in results:
        trajectory = result.get('token_trajectory', [])
        
        fold_events = []
        i = 0
        while i < len(trajectory):
            event = trajectory[i]
            
            if event.get('event') == 'before_fold':
                # 查找对应的 after_fold
                after_fold = None
                for j in range(i + 1, min(i + 10, len(trajectory))):
                    if trajectory[j].get('event') == 'after_fold':
                        after_fold = trajectory[j]
                        break
                
                if after_fold:
                    fold_event = {
                        'instance_id': result['instance_id'],
                        'turn': event.get('turn', 0),
                        'before_tokens': event.get('context_tokens', 0),
                        'after_tokens': after_fold.get('context_tokens', 0),
                        'compression_ratio': after_fold.get('compression_ratio', 1.0),
                        'tokens_saved': after_fold.get('tokens_saved', 0),
                        'before_episode_mem': event.get('episode_memory_size', 0),
                        'after_episode_mem': after_fold.get('episode_memory_size', 0),
                        'before_working_mem': event.get('working_memory_size', 0),
                        'after_working_mem': after_fold.get('working_memory_size', 0),
                    }
                    fold_events.append(fold_event)
                    all_folds.append(fold_event)
            
            i += 1
        
        result['fold_events'] = fold_events
    
    if not all_folds:
        return {
            'total_folds': 0,
            'avg_compression_ratio': 0,
            'avg_tokens_saved': 0,
            'max_compression_ratio': 0,
            'min_compression_ratio': 0,
        }
    
    compression_ratios = [f['compression_ratio'] for f in all_folds]
    tokens_saved = [f['tokens_saved'] for f in all_folds]
    
    return {
        'total_folds': len(all_folds),
        'avg_compression_ratio': np.mean(compression_ratios),
        'avg_tokens_saved': np.mean(tokens_saved),
        'max_compression_ratio': max(compression_ratios),
        'min_compression_ratio': min(compression_ratios),
        'all_fold_events': all_folds
    }


if __name__ == "__main__":
    main()

