#!/usr/bin/env python3
"""
GAIA Benchmark 评测脚本（支持 HyMem 和 ReAct 框架）
支持 text-only 和带附件的问题
"""
import asyncio
import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import sys
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.hymem_agent import process_item as hymem_process_item
from agents.fold_agent import process_item as react_process_item
from agents.resum_agent import process_item as resum_process_item
from agents.thread_agent import process_item as thread_process_item
# A-MEM: imported lazily inside the dispatcher to avoid loading chromadb /
# sentence-transformers when not needed (those deps are optional, only
# required when --framework amem).
from agents.bm_tools import BMToolManager
from agents.file_tools import HymemFileParser
from agents.utils import CallAPI, TaskContext, print_distractor_stats
from agents.hymem_logger import HyMemEventLog, HyMemLogFormatter, TokenUsage
from verl import DataProto




def load_gaia_dataset(parquet_path: str, gaia_root: str = None) -> pd.DataFrame:
    """加载GAIA数据集"""
    df = pd.read_parquet(parquet_path)
    #df = df.sample(n=3, random_state=42)
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


async def parse_file_content(file_path: str, parser: HymemFileParser, max_length: int = 4000) -> str:
    """解析文件内容"""
    if not file_path or not os.path.exists(file_path):
        return ""
    
    try:
        result = await parser.parse_file(file_path)
        content = result.get('content', '')
        if len(content) > max_length:
            content = content[:max_length] + "\n[Content truncated...]"
        return content
    except Exception as e:
        return f"[File parse error: {str(e)}]"


class GAIALogger:
    """GAIA评测日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
    
    def format_task_log(self, instance_id: str, problem: str, history: List[Dict],
                        answer: str, confidence: float, score: float,
                        ground_truth: str, stats: Dict, file_info: Dict = None) -> str:
        lines = []
        lines.append(f"{'#'*80}")
        lines.append(f"# GAIA Task: {instance_id}")
        lines.append(f"# Level: {stats.get('level', 'N/A')}")
        lines.append(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"{'#'*80}\n")
        
        lines.append("== PROBLEM ==")
        lines.append(problem[:2000])
        lines.append("")
        
        if file_info and file_info.get('file_name'):
            lines.append("== FILE INFO ==")
            lines.append(f"File: {file_info.get('file_name')}")
            lines.append(f"Path: {file_info.get('file_path')}")
            lines.append("")
        
        lines.append("== EXECUTION ==")
        for entry in history[:20]:
            role = entry.get('role', '')
            turn = entry.get('turn', 0)
            if role == 'planner':
                lines.append(f"\n[Planner T{turn}]")
                lines.append(entry.get('content', '')[:1500])
            elif role == 'executor':
                lines.append(f"\n[Executor T{turn}] Task: {entry.get('task', '')[:200]}")
                lines.append(f"Result: {entry.get('result', '')[:800]}")
            elif role == 'fold':
                lines.append(f"\n[Fold T{turn}] Memory compressed")
        lines.append("")
        
        lines.append("== RESULT ==")
        lines.append(f"Prediction: {answer}")
        lines.append(f"Ground Truth: {ground_truth}")
        lines.append(f"Score: {score}")
        lines.append(f"Confidence: {confidence:.2f}")
        lines.append("")
        
        lines.append("== STATS ==")
        lines.append(f"Planner Turns: {stats.get('planner_turns', 0)}")
        lines.append(f"Executor Calls: {stats.get('executor_calls', 0)}")
        lines.append(f"Tool Calls (atomic): {stats.get('tool_calls_atomic', stats.get('tool_calls_total', 0))}")
        lines.append(f"Agent Turns: {stats.get('agent_turns', 0)}")
        lines.append(f"Branch Calls: {stats.get('branch_calls', 0)}")
        lines.append(f"Fold Count: {stats.get('fold_count', 0)}")
        lines.append(f"Session Time: {stats.get('session_time', 0):.2f}s")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_task_log(self, instance_id: str, content: str):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        with open(self.log_dir / f"gaia_{safe_id}.log", 'w', encoding='utf-8') as f:
            f.write(content)

    def save_task_trace(self, instance_id: str, trace: Dict):
        """Save the full machine-readable trace for one task (Step:
        complete-process recording). Includes event_log, tool_call_history,
        routing_decisions, distillation_stats, models_used, latency, and
        per-component tokens — everything needed for offline forensics or
        re-running cost / efficiency analysis without re-executing the task.
        """
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        path = self.log_dir / f"gaia_{safe_id}_trace.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

    def add_log(self, entry: Dict):
        self.logs.append(entry)
    
    def save_all(self, filename: str = None):
        if filename is None:
            filename = f"gaia_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(self.log_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        return self.log_dir / filename


async def eval_one_gaia(row, config, tokenizer, model_name, logger: GAIALogger, 
                        file_parser: HymemFileParser, save_logs: bool, framework: str = 'hymem'):
    """评测单个GAIA实例"""
    extra_info = row['extra_info']
    instance_id = extra_info.get('instance_id', 'unknown')
    question = extra_info.get('query', '')
    ground_truth = extra_info.get('answer', '')
    abs_file_path = extra_info.get('abs_file_path')
    level = extra_info.get('level', 1)
    
    problem = question
    if abs_file_path:
        file_content = await parse_file_content(abs_file_path, file_parser)
        if file_content and not file_content.startswith("["):
            problem = f"{question}\n\n[Attached File Content]:\n{file_content}"
    
    context = TaskContext(
        config=config, 
        global_step=0, 
        server_host=model_name,
        server_port=0, 
        is_train=False, 
        tokenizer=tokenizer
    )

    modified_extra_info = extra_info.copy()
    modified_extra_info['query'] = problem
    modified_extra_info['problem_statement'] = problem

    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([modified_extra_info], dtype=object),
        'uid': np.array([instance_id], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {
        'generation_kwargs': {}, 
        'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn
    }

    try:
        # 根据框架选择不同的 process_item 函数
        if framework == 'hymem':
            output = await hymem_process_item(item, context, CallAPI)
        elif framework == 'resum':
            output = await resum_process_item(item, context, CallAPI)
        elif framework == 'thread':
            output = await thread_process_item(item, context, CallAPI)
        elif framework == 'amem':
            from agents.amem_agent import process_item as amem_process_item
            output = await amem_process_item(item, context, CallAPI)
        else:  # react / fold
            output = await react_process_item(item, context, CallAPI)
        
        extra_data = output.non_tensor_batch.get('extra_data', [{}])[0] if output else {}
        stats = extra_data.get('stats', {})
        
        # 根据框架提取历史和答案
        # Fix: every framework writes a different key (history shape) and uses a
        # different answer-emission format. Generic extractor below tries each
        # in order so prediction!='' for non-HyMem baselines too.
        if framework == 'hymem':
            history = stats.get('hymem_history', [])
        elif framework in ('thread', 'amem', 'resum', 'react', 'fold'):
            # These agents write env.stats['messages'] (chat list) — fall back
            # to 'history' if some legacy agent uses that key.
            history = stats.get('messages', []) or stats.get('history', [])
        else:
            history = stats.get('messages', []) or stats.get('history', []) \
                      or stats.get('hymem_history', [])
        confidence = stats.get('confidence', 0.0)

        # Multi-format answer extraction:
        #   1. HyMem-style <answer>...</answer>           (HyMem)
        #   2. Thread-style ACTION: finish(VALUE)         (Thread)
        #   3. ReAct-style <function=finish><parameter=answer>...  (env XML)
        #   4. Last assistant message (fallback)
        answer = ''
        for entry in reversed(history):
            content = entry.get('content', '') or ''
            # Pattern 1: <answer>...</answer>
            m = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                break
            # Pattern 2: ACTION: finish(VALUE)
            m = re.search(r'finish\s*\(\s*(.+?)\s*\)\s*$', content, re.MULTILINE | re.DOTALL)
            if m:
                answer = m.group(1).strip().strip('"\'')
                break
            # Pattern 3: <parameter=answer>VALUE</parameter>
            m = re.search(r'<parameter=answer>(.*?)</parameter>', content, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                break
        # Pattern 4: fallback to LAST assistant message (works when agent
        # ended naturally without explicit answer tag)
        if not answer:
            for entry in reversed(history):
                if entry.get('role') == 'assistant':
                    c = (entry.get('content', '') or '').strip()
                    if c and len(c) < 800:  # reasonable answer length
                        answer = c
                        break
        
        # 使用 HyMem 框架内部的 LLM Judge 评分
        score = stats.get('get_final_score', 0)
        stats['level'] = level
        
        if save_logs and history:
            event_log_data = stats.get('event_log')
            if event_log_data and framework == 'hymem':
                event_log_obj = HyMemEventLog()
                event_log_obj.events = event_log_data.get('events', [])
                event_log_obj.token_usage = TokenUsage()
                tu = event_log_data.get('token_usage', {}).get('total', {})
                event_log_obj.token_usage.prompt_tokens = tu.get('prompt_tokens', 0)
                event_log_obj.token_usage.completion_tokens = tu.get('completion_tokens', 0)
                event_log_obj.token_usage.total_tokens = tu.get('total_tokens', 0)
                log_content = HyMemLogFormatter.format_task_log(
                    instance_id=instance_id,
                    problem=question,
                    event_log=event_log_obj,
                    answer=answer,
                    confidence=confidence,
                    score=score,
                    stats=stats,
                    extra={
                        'ground_truth': ground_truth,
                        'file_name': extra_info.get('file_name'),
                        'file_path': extra_info.get('file_path'),
                    },
                )
            else:
                log_content = logger.format_task_log(
                    instance_id=instance_id,
                    problem=question,
                    history=history,
                    answer=answer,
                    confidence=confidence,
                    score=score,
                    ground_truth=ground_truth,
                    stats=stats,
                    file_info={'file_name': extra_info.get('file_name'), 'file_path': extra_info.get('file_path')}
                )
            logger.save_task_log(instance_id, log_content)
        
        result = {
            'instance_id': instance_id,
            'level': level,
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'prediction': answer[:500],
            'ground_truth': ground_truth[:500],
            'has_file': bool(abs_file_path),
            # Ablation provenance (hymem only; other frameworks leave it absent)
            'ablation': stats.get('ablation'),
            'ablation_flags': stats.get('ablation_flags', {}),
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'branch_success': stats.get('branch_success', 0),
            'fold_count': stats.get('fold_count', 0),
            'replan_count': stats.get('replan_count', 0),
            # Step A3 — unified tool-call schema (paper-friendly comparison).
            'tool_calls_atomic': stats.get('tool_calls_atomic',
                                           stats.get('tool_calls_total', 0)),
            'agent_turns': stats.get('agent_turns', 0),
            'tool_calls_by_source': stats.get('tool_calls_by_source', {}),
            'framework': stats.get('framework', framework),
            'tool_calls_total': stats.get('tool_calls_total', 0),  # legacy alias
            'tool_calls_by_tool': stats.get('tool_calls_by_tool', {}),
            # Step A1 — routing decisions
            'routing_summary': stats.get('routing_summary', {}),
            # Step A2 — distillation pipeline aggregate
            'distillation_stats': stats.get('distillation_stats', {}),
            # Step B1 — model identities (LMtp aux-model size question)
            'models_used': stats.get('models_used', {}),
            # Step B2 — per-component wall-clock latency (Bh2U W2)
            'latency_by_component': stats.get('latency_by_component', {}),
            # Tokens — total + per-component (cost_calculator needs the latter)
            'total_prompt_tokens': stats.get('total_prompt_tokens', 0),
            'total_completion_tokens': stats.get('total_completion_tokens', 0),
            'total_tokens': stats.get('total_tokens', 0),
            'token_usage_by_component': stats.get('token_usage_by_component', {}),
            'session_time': stats.get('session_time', 0),
            # Round 1-3 efficiency observation fields (Bh2U W2 rebuttal)
            'parse_cache_hits': stats.get('parse_cache_hits', 0),
            'parse_cache_misses': stats.get('parse_cache_misses', 0),
            'parse_cache_hit_rate': stats.get('parse_cache_hit_rate', 0.0),
            'executor_early_exit_count': stats.get('executor_early_exit_count', 0),
            'executor_avg_inner_turns': stats.get('executor_avg_inner_turns', 0.0),
            'distillation_invocations': stats.get('distillation_invocations', 0),
            'distillation_avg_aux_calls': stats.get('distillation_avg_aux_calls', 0.0),
            'distillation_chunk_concurrent_used': stats.get('distillation_chunk_concurrent_used', 0),
            'fold_invocations': stats.get('fold_invocations', 0),
            'fold_avg_wall_seconds': stats.get('fold_avg_wall_seconds', 0.0),
            'fold_aux_parallel_seconds': stats.get('fold_aux_parallel_seconds', 0.0),
            'parallel_dispatch_fanout_avg': stats.get('parallel_dispatch_fanout_avg', 0.0),
            'parallel_dispatch_max_observed': stats.get('parallel_dispatch_max_observed', 0),
            # Figure-1 dilution: per-turn full context dump (HyMem dict-list
            # or fold-agent agent-keyed list — schema differs by framework).
            'planner_context_per_turn': stats.get('planner_context_per_turn', []),
        }

        # Save the FULL machine-readable trace per task (event_log timeline,
        # tool call history, routing decisions). Lets us re-run cost /
        # efficiency / branch-trigger analysis offline without re-executing.
        if save_logs:
            full_trace = {
                'instance_id': instance_id,
                'level': level,
                'problem': question,
                'ground_truth': ground_truth,
                'prediction': answer,
                'score': score,
                'confidence': confidence,
                'stats': result,
                'event_log': stats.get('event_log', {}),
                'tool_call_history': stats.get('tool_call_history_full', []),
                'routing_decisions': stats.get('routing_decisions', []),
                'memory': {
                    'episode': stats.get('episode_memory', {}),
                    'working': stats.get('working_memory', {}),
                    'tool': stats.get('tool_memory', {}),
                    'planner_summary': stats.get('planner_summary', ''),
                },
                'hymem_history': stats.get('hymem_history', []) if framework == 'hymem' else stats.get('history', []),
                # Forensic: full chat messages (post-hoc inspection of what the
                # agent actually said). Critical for non-HyMem baselines whose
                # answer-extraction patterns may differ — without this you can't
                # reconstruct what the agent answered if extraction returned ''.
                'messages': stats.get('messages', []),
            }
            logger.save_task_trace(instance_id, full_trace)

    except Exception as e:
        print(f"[ERROR] GAIA Task {instance_id} failed: {e}")
        traceback.print_exc()
        result = {
            'instance_id': instance_id,
            'level': level,
            'score': 0,
            'confidence': 0.0,
            'status': 'failed',
            'error': str(e),
            'prediction': '',
            'ground_truth': ground_truth[:500],
            'has_file': bool(abs_file_path),
            'planner_turns': 0,
            'executor_calls': 0,
            'branch_calls': 0,
            'branch_success': 0,
            'fold_count': 0,
            'replan_count': 0,
            'tool_calls_atomic': 0,
            'agent_turns': 0,
            'tool_calls_by_source': {},
            'framework': framework,
            'tool_calls_total': 0,
            'tool_calls_by_tool': {},
            'routing_summary': {},
            'distillation_stats': {},
            'models_used': {},
            'latency_by_component': {},
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'token_usage_by_component': {},
            'session_time': 0,
            'parse_cache_hits': 0,
            'parse_cache_misses': 0,
            'parse_cache_hit_rate': 0.0,
            'executor_early_exit_count': 0,
            'executor_avg_inner_turns': 0.0,
            'distillation_invocations': 0,
            'distillation_avg_aux_calls': 0.0,
            'distillation_chunk_concurrent_used': 0,
            'fold_invocations': 0,
            'fold_avg_wall_seconds': 0.0,
            'fold_aux_parallel_seconds': 0.0,
            'parallel_dispatch_fanout_avg': 0.0,
            'parallel_dispatch_max_observed': 0,
            'planner_context_per_turn': [],
        }

    logger.add_log(result)
    return result


async def worker(worker_id, rows, args, pbar, shared_scores, logger, file_parser):
    """评测工作线程"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 根据框架构建不同的配置
    if args.framework == 'hymem':
        plugin_config = {
            'workflow': 'hymem',
            'max_turn': args.max_turn,
            'val_max_turn': args.val_max_turn,
            'max_session': args.max_session,
            'val_max_session': args.val_max_session,
            'session_timeout': 1200,
            'hymem_max_planner_turns': args.max_planner_turns,
            'hymem_max_executor_turns': args.max_executor_turns,
            'hymem_max_branch_turns': args.max_branch_turns,
            'hymem_max_branches': args.max_branches,
            'hymem_max_folds': args.max_folds,
            'hymem_search_topk': args.search_topk,
            'hymem_ablation': args.ablation,
            'hymem_llm_model': args.model_name,
            'hymem_aux_model': args.aux_model,
            'search_base_url': args.search_url,
            'search_mode': args.search_mode,
        }
    else:  # react
        plugin_config = {
            'workflow': args.workflow,
            'max_turn': args.max_turn,
            'val_max_turn': args.val_max_turn,
            'max_session': args.max_session,
            'val_max_session': args.val_max_session,
            'session_timeout': 1200,
            'process_reward': None,
            'max_traj': None,
            'must_finish': False,
            'double_check': False,
            'must_search': False,
            'enable_summary': getattr(args, 'enable_summary', False),
        }
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': plugin_config
        }}
    })

    results = []
    for row in rows:
        result = await eval_one_gaia(row, config, tokenizer, args.model_name, 
                                     logger, file_parser, args.save_logs, args.framework)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        
        pbar.set_postfix({
            'avg': f"{avg_score:.3f}",
            'L': result['level'],
            's': f"{result['score']:.1f}",
        })
        pbar.update(1)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HyMem/ReAct on GAIA Benchmark')
    # Default paths resolve from the GAIA_ROOT env var when set (lets the
    # same script run on the server and on the user's local machine without
    # editing defaults). Fall back to the in-repo demo path otherwise.
    _default_root = os.getenv("GAIA_ROOT", "/Users/hoi/project/HyMem2026/GAIA")
    _default_parquet = os.path.join(_default_root, "2023/validation/metadata.level1.parquet")
    parser.add_argument('--data_path',
                        default=_default_parquet,
                        help='Path to GAIA parquet file (default from $GAIA_ROOT)')
    parser.add_argument('--gaia_root',
                        default=_default_root,
                        help='Root directory of GAIA dataset (default from $GAIA_ROOT)')
    parser.add_argument('--output_dir', default='results/gaia', help='Output directory')
    parser.add_argument('--log_dir', default='logs/gaia_eval', help='Log directory')
    parser.add_argument('--prompt_length', type=int, default=65536)
    parser.add_argument('--response_length', type=int, default=65536)
    parser.add_argument('--max_turn', type=int, default=40)
    parser.add_argument('--val_max_turn', type=int, default=60)
    parser.add_argument('--max_session', type=int, default=10)
    parser.add_argument('--val_max_session', type=int, default=10)
    # model_name is just an identifier passed through to context.server_host;
    # the actual model used is picked from HYMEM_PLANNER_MODEL / HYMEM_EXECUTOR_MODEL
    # / HYMEM_AUX_MODEL env vars (see scripts/env_eval_local.sh).
    parser.add_argument('--model_name',
                        default=os.getenv('HYMEM_PLANNER_MODEL', 'Qwen3-32B'),
                        help='Identifier label (actual model from HYMEM_*_MODEL env vars)')
    parser.add_argument('--aux_model', default=None)
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Concurrency. Keep small (1-2) when calling remote APIs to '
                             'avoid rate limits; raise to 5+ when pointing at local vLLM.')
    parser.add_argument('--search_url', default='http://localhost:8021',
                        help='Local search service URL (only used when --search_mode=local)')
    parser.add_argument('--search_mode', default='online', choices=['local', 'online', 'auto'],
                        help="'online' uses Google/Serper/Jina via env vars (no local server "
                             "needed); 'local' requires --search_url; 'auto' picks based on env.")
    parser.add_argument('--tokenizer_path',
                        default=os.getenv('HYMEM_TOKENIZER', 'Qwen/Qwen3-8B'),
                        help='HF tokenizer ID or local path (default Qwen/Qwen2.5-7B-Instruct, '
                             'auto-downloads on first use)')
    # Convenience shortcut: --api_mode flips the defaults above to the
    # local-machine + remote-API workflow (online search, low concurrency).
    parser.add_argument('--api_mode', action='store_true',
                        help='Shorthand for local + remote-API testing: forces --search_mode=online '
                             'and caps --num_workers at 2. Combine with --sample N for quick probes.')
    # Resume support — skip tasks whose trace JSON already exists. Essential
    # for full GAIA runs (165 tasks × ~5 min each = many hours).
    parser.add_argument('--resume', action='store_true',
                        help='Skip tasks that already have a {id}_trace.json in --log_dir. '
                             'Lets you restart an interrupted full GAIA run without redoing work.')
    parser.add_argument('--no_resume', action='store_true',
                        help='Force re-running every task (overrides --resume).')
    
    # 框架选择参数
    parser.add_argument('--framework', default='hymem',
                        choices=['hymem', 'react', 'resum', 'thread', 'amem'],
                        help='Evaluation framework: hymem | react | resum | thread | amem '
                             '(default: hymem). amem requires `pip install -r requirements_amem.txt`.')
    parser.add_argument('--workflow', default='search',
                        help='Workflow for react framework: search (ReAct) or search_branch (Context-Folding)')
    
    # HyMem 特定参数
    parser.add_argument('--max_planner_turns', type=int, default=12)
    parser.add_argument('--max_executor_turns', type=int, default=10)
    parser.add_argument('--max_branch_turns', type=int, default=3)
    parser.add_argument('--max_branches', type=int, default=5)
    parser.add_argument('--max_folds', type=int, default=7)
    parser.add_argument('--search_topk', type=int, default=5)
    # Ablation selector (hymem framework only). Maps to HYMEM_ABLATION.
    parser.add_argument('--ablation', default='full',
                        choices=['full', 'no_isolated_reasoning', 'no_memory_management'],
                        help='HyMem ablation variant: full (default) | '
                             'no_isolated_reasoning (disable Branch) | '
                             'no_memory_management (disable fold + structured memory)')
    
    # ReAct 特定参数
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode for react framework')
    
    parser.add_argument('--save_logs', action='store_true', default=True)
    parser.add_argument('--text_only', action='store_true', 
                        help='Only evaluate text-only questions (no file attachments)')
    parser.add_argument('--sample', type=int, default=None, help='Sample N items for testing')
    return parser.parse_args()


def main():
    args = parse_args()

    # --api_mode shorthand: force online search + small worker count, suitable
    # for testing on a local machine against remote LLM APIs.
    if args.api_mode:
        args.search_mode = 'online'
        args.num_workers = min(args.num_workers, 2)

    # 根据框架设置不同的环境变量
    if args.framework == 'react':
        os.environ["LOCAL_SEARCH_URL"] = args.search_url
    else:
        os.environ["SEARCH_MODE"] = args.search_mode
    if args.search_mode == 'online':
        os.environ["USE_ONLINE_SEARCH"] = "true"

    # Validate prerequisites for API mode so the user fails fast (rather than
    # 30 seconds into a doomed run).
    missing = []
    if not os.getenv("HYMEM_PLANNER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        missing.append("HYMEM_PLANNER_API_KEY (or OPENAI_API_KEY)")
    if args.search_mode == 'online':
        if not (os.getenv("SERPER_KEY") or os.getenv("GOOGLE_API_KEY")):
            missing.append("SERPER_KEY or GOOGLE_API_KEY (for online search)")
    if not os.path.exists(args.data_path):
        missing.append(f"--data_path file: {args.data_path}")
    if missing:
        print("[GAIA] ERROR: missing prerequisites:")
        for m in missing:
            print(f"  - {m}")
        print("\nTip: source scripts/env_eval_local.sh first, or pass --data_path explicitly.")
        return 1

    print(f"\n{'='*60}")
    print(f"GAIA Benchmark Evaluation ({args.framework.upper()})")
    print(f"{'='*60}")
    print(f"Framework: {args.framework}")
    if args.framework == 'hymem':
        print(f"Ablation: {args.ablation}")
    if args.framework == 'react':
        print(f"Workflow: {args.workflow}")
    print(f"Data: {args.data_path}")
    print(f"GAIA Root: {args.gaia_root}")
    print(f"Tokenizer: {args.tokenizer_path}")
    # Show actual model identities resolved from env vars (Step B1 visibility).
    print(f"Models:")
    print(f"  - Planner:  {os.getenv('HYMEM_PLANNER_MODEL', '(unset → defaults to ' + args.model_name + ')')}")
    print(f"  - Executor: {os.getenv('HYMEM_EXECUTOR_MODEL', '(unset → falls back to planner)')}")
    print(f"  - Aux:      {os.getenv('HYMEM_AUX_MODEL', '(unset → falls back to executor)')}")
    print(f"  - Judge:    {os.getenv('JUDGE_MODEL', '(unset)')}")
    if args.framework == 'hymem':
        print(f"Search Mode: {args.search_mode}")
    print(f"Text Only: {args.text_only}")
    print(f"Workers: {args.num_workers}")
    print(f"API mode: {args.api_mode}")
    print(f"{'='*60}\n")

    df = load_gaia_dataset(args.data_path, args.gaia_root)

    if args.text_only:
        df = df[df['extra_info'].apply(lambda x: x.get('abs_file_path') is None)]
        print(f"[GAIA] Filtered to {len(df)} text-only items")
        # Old code did df.sample(n=50) then df.iloc[15:] — would crash if
        # the level-N parquet has <50 text-only items. Use --sample for any
        # subsampling instead.
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"[GAIA] Sampled {len(df)} items for testing")
    
    if len(df) == 0:
        print("[GAIA] ERROR: dataset filtered to 0 rows. Check --text_only / --sample.")
        return 1

    logger = GAIALogger(args.log_dir)
    file_parser = HymemFileParser()

    # --- Resume support: read previously completed traces and skip them ---
    completed_results: List[Dict] = []
    if args.resume and not args.no_resume:
        existing_traces = list(Path(args.log_dir).glob("gaia_*_trace.json"))
        completed_ids = set()
        for trace_path in existing_traces:
            try:
                trace = json.load(open(trace_path))
                tid = trace.get('instance_id') or trace.get('stats', {}).get('instance_id')
                if tid:
                    completed_ids.add(tid)
                    completed_results.append(trace.get('stats', trace))
            except Exception:
                continue
        if completed_ids:
            before = len(df)
            df = df[df['extra_info'].apply(lambda x: x.get('instance_id') not in completed_ids)]
            print(f"[GAIA] --resume: skipping {before - len(df)} already-completed tasks "
                  f"(loaded from {len(existing_traces)} traces in {args.log_dir})")
            if len(df) == 0:
                print("[GAIA] All requested tasks already completed; jumping to summary.")

    if len(df) > 0:
        num_workers = max(1, min(args.num_workers, len(df)))
        chunk_size = max(1, len(df) // num_workers)
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < num_workers-1 else len(df)]
                  for i in range(num_workers)]

        async def run_all():
            shared_scores = []
            with tqdm(total=len(df), desc="Evaluating GAIA", unit="task") as pbar:
                tasks = [
                    worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))],
                           args, pbar, shared_scores, logger, file_parser)
                    for i in range(num_workers)
                ]
                return await asyncio.gather(*tasks)

        all_results = asyncio.run(run_all())
        new_results = [r for worker_results in all_results for r in worker_results]
    else:
        new_results = []

    # Combine resumed + newly-completed results so the aggregate summary
    # reflects the full evaluation, not just this incremental run.
    results = completed_results + new_results
    if completed_results:
        print(f"[GAIA] Combined {len(completed_results)} resumed + "
              f"{len(new_results)} new = {len(results)} total tasks for summary.")

    print(f"\n{'='*60}")
    print(f"GAIA EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    avg_score = np.mean([r['score'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"Overall Score: {avg_score:.4f}")
    print(f"Success Rate: {success_count}/{len(results)}")
    
    by_level = defaultdict(list)
    for r in results:
        by_level[r['level']].append(r)
    
    print(f"\nBy Level:")
    for level in sorted(by_level.keys()):
        lvl_results = by_level[level]
        lvl_score = np.mean([r['score'] for r in lvl_results])
        print(f"  Level {level}: {lvl_score:.4f} ({len(lvl_results)} items)")
    
    by_file = defaultdict(list)
    for r in results:
        key = 'with_file' if r['has_file'] else 'text_only'
        by_file[key].append(r)
    
    print(f"\nBy File Type:")
    for ftype, f_results in by_file.items():
        f_score = np.mean([r['score'] for r in f_results])
        print(f"  {ftype}: {f_score:.4f} ({len(f_results)} items)")
    
    def _safe_mean(seq):
        seq = [x for x in seq if isinstance(x, (int, float))]
        return float(np.mean(seq)) if seq else 0.0

    # Aggregate the new (Step A1-B5) metrics so the operator sees them at a
    # glance after each run, not just buried in the per-task JSON.
    print(f"\nAvg Stats:")
    print(f"  Planner Turns:        {_safe_mean([r['planner_turns'] for r in results]):.2f}")
    print(f"  Executor Calls:       {_safe_mean([r['executor_calls'] for r in results]):.2f}")
    print(f"  Branch Calls:         {_safe_mean([r.get('branch_calls', 0) for r in results]):.2f}")
    print(f"  Branch Success:       {_safe_mean([r.get('branch_success', 0) for r in results]):.2f}")
    print(f"  Fold Count:           {_safe_mean([r['fold_count'] for r in results]):.2f}")
    print(f"  Replan Count:         {_safe_mean([r.get('replan_count', 0) for r in results]):.2f}")
    print(f"  Tool Calls (atomic):  {_safe_mean([r.get('tool_calls_atomic', r.get('tool_calls_total', 0)) for r in results]):.2f}")
    print(f"  Agent Turns:          {_safe_mean([r.get('agent_turns', 0) for r in results]):.2f}")
    print(f"  Total Tokens:         {_safe_mean([r['total_tokens'] for r in results]):,.0f}")
    print(f"  Session Time:         {_safe_mean([r['session_time'] for r in results]):.2f}s")

    # Sub-thread atomic share (HyMem 'branch' or Thread 'child_thread' —
    # both represent atomic calls made inside a delegated sub-thread).
    branch_atomic = sum(
        (r.get('tool_calls_by_source', {}) or {}).get('branch', 0)
        + (r.get('tool_calls_by_source', {}) or {}).get('child_thread', 0)
        for r in results
    )
    actor_atomic = sum((r.get('tool_calls_by_source', {}) or {}).get('actor', 0) for r in results)
    total_atomic = branch_atomic + actor_atomic
    if total_atomic > 0:
        print(f"  Sub-thread atomic share: {branch_atomic / total_atomic:.1%} "
              f"({branch_atomic}/{total_atomic})  "
              f"[hymem.branch + thread.child_thread]")

    # Routing decisions: how often did the LLM router fire vs the keyword fallback?
    routing_llm = sum((r.get('routing_summary', {}) or {}).get('from_llm', 0) for r in results)
    routing_fb  = sum((r.get('routing_summary', {}) or {}).get('from_fallback', 0) for r in results)
    if (routing_llm + routing_fb) > 0:
        print(f"  Router decisions:     LLM={routing_llm} / fallback={routing_fb} "
              f"({routing_llm/(routing_llm+routing_fb):.0%} LLM)")

    # Per-component latency (Step B2). HyMem reports per-component split
    # (planner/executor/branch/memory); other baselines bucket everything
    # into 'actor' or 'total'. Print whichever buckets are populated.
    lat_components = ('planner', 'executor', 'branch', 'memory', 'actor', 'total')
    lat_lines = []
    for comp in lat_components:
        v = _safe_mean([(r.get('latency_by_component', {}) or {}).get(comp, 0) for r in results])
        if v > 0:
            lat_lines.append(f"    - {comp}: {v:.2f}")
    if lat_lines:
        print(f"  Latency by component (s):")
        for line in lat_lines:
            print(line)

    # Distillation pipeline (Step A2)
    dist_invocations = sum((r.get('distillation_stats', {}) or {}).get('invocations', 0) for r in results)
    if dist_invocations > 0:
        chunks_kept = sum((r.get('distillation_stats', {}) or {}).get('chunks_kept', 0) for r in results)
        chunks_dropped = sum((r.get('distillation_stats', {}) or {}).get('chunks_dropped', 0) for r in results)
        print(f"  Distillation:         {dist_invocations} long pages, "
              f"{chunks_kept} chunks kept / {chunks_dropped} dropped")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_dir) / f"gaia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Pull models_used from any successful task (they're identical across tasks
    # in one run, since model selection is global).
    sample_models = next((r.get('models_used', {}) for r in results
                          if r.get('models_used')), {})

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_label': args.model_name,
            'models_used': sample_models,        # actual planner/executor/aux/judge IDs
            'framework': args.framework,
            'ablation': args.ablation if args.framework == 'hymem' else 'n/a',
            'data_path': args.data_path,
            'text_only': args.text_only,
            'total_items': len(results),
            'success_count': success_count,
            # Reproducibility — exact CLI args & key env state
            'cli_args': vars(args),
            'env_snapshot': {
                'HYMEM_PLANNER_MODEL': os.getenv('HYMEM_PLANNER_MODEL', ''),
                'HYMEM_EXECUTOR_MODEL': os.getenv('HYMEM_EXECUTOR_MODEL', ''),
                'HYMEM_AUX_MODEL': os.getenv('HYMEM_AUX_MODEL', ''),
                'JUDGE_MODEL': os.getenv('JUDGE_MODEL', ''),
                'SEARCH_MODE': os.getenv('SEARCH_MODE', ''),
            },
        },
        'summary': {
            'overall_score': float(avg_score),
            'by_level': {str(k): float(_safe_mean([r['score'] for r in v])) for k, v in by_level.items()},
            'by_file_type': {k: float(_safe_mean([r['score'] for r in v])) for k, v in by_file.items()},
            # Aggregate efficiency numbers (handy for paper tables)
            'avg_tool_calls_atomic': _safe_mean([r.get('tool_calls_atomic', r.get('tool_calls_total', 0)) for r in results]),
            'avg_agent_turns': _safe_mean([r.get('agent_turns', 0) for r in results]),
            'avg_total_tokens': _safe_mean([r['total_tokens'] for r in results]),
            'avg_session_time_s': _safe_mean([r['session_time'] for r in results]),
            'avg_branch_calls': _safe_mean([r.get('branch_calls', 0) for r in results]),
            'avg_branch_success': _safe_mean([r.get('branch_success', 0) for r in results]),
            'branch_atomic_share': (branch_atomic / total_atomic) if total_atomic > 0 else 0.0,
            'router_llm_share': (routing_llm / (routing_llm + routing_fb))
                                if (routing_llm + routing_fb) > 0 else None,
        },
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to:        {output_file}")

    log_file = logger.save_all()
    print(f"Per-task logs saved to:  {logger.log_dir}")
    print(f"  - {{instance_id}}.log         (human-readable summary)")
    print(f"  - {{instance_id}}_trace.json  (full event_log + tool history + routing)")
    print(f"Aggregated log saved to: {log_file}")

    # 打印干扰注入统计（如果启用）
    print_distractor_stats()

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"  Quick analysis next steps:")
    print(f"    python scripts/analysis/cost_calculator.py --input {output_file}")
    print(f"    python scripts/analysis/efficiency_analysis.py --results HyMem {output_file}")
    print(f"    python scripts/analysis/branch_trigger_analysis.py --results HyMem {output_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    raise SystemExit(main() or 0)

