#!/usr/bin/env python3
"""
HyMem on BrowseComp Evaluation Script

专门用于使用 HyMem 框架评测 OpenAI BrowseComp 数据集的脚本。
"""
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
import traceback
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.hymem_agent import process_item as hymem_process_item
from agents.utils import CallAPI, TaskContext
from agents.hymem_logger import HyMemEventLog, HyMemLogFormatter, TokenUsage
from verl import DataProto
from scripts.eval_logger import HyMemLogger
import os


# ==================== OpenAI BrowseComp 数据解密与加载 ====================

def derive_key(password: str, length: int) -> bytes:
    """从密码派生固定长度的密钥（使用 SHA256）"""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """使用 XOR 解密 base64 编码的密文"""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def load_browsecomp_openai(cache_path: str = None) -> pd.DataFrame:
    """
    加载 OpenAI BrowseComp 官方数据集并转换为兼容格式
    
    Args:
        cache_path: 可选的本地缓存路径，避免重复下载
        
    Returns:
        转换后的 DataFrame
    """
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    # 处理缓存路径：如果是目录，则在目录下创建默认文件名
    if cache_path:
        if os.path.isdir(cache_path):
            cache_path = os.path.join(cache_path, 'browsecomp_openai_cache.csv')
            print(f"[BrowseComp] Cache path is directory, using: {cache_path}")
    
    # 尝试从缓存加载或下载
    if cache_path and os.path.exists(cache_path):
        print(f"[BrowseComp] Loading from cache: {cache_path}")
        df_raw = pd.read_csv(cache_path)
    else:
        print(f"[BrowseComp] Downloading from {url}")
        df_raw = pd.read_csv(url)
        if cache_path:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            df_raw.to_csv(cache_path, index=False)
            print(f"[BrowseComp] Cached to {cache_path}")
    
    # 打印类别分布
    if 'problem_topic' in df_raw.columns:
        print(f"\n[BrowseComp] 类别分布 (共 {len(df_raw)} 题):")
        topic_counts = df_raw['problem_topic'].value_counts()
        for topic, count in topic_counts.items():
            print(f"  - {topic}: {count} ({100*count/len(df_raw):.1f}%)")
        print()
    
    records = []
    for idx, row in df_raw.iterrows():
        # 解密问题和答案
        try:
            canary = row.get('canary', '')
            problem_encrypted = row.get('problem', '')
            answer_encrypted = row.get('answer', '')
            
            problem = decrypt(problem_encrypted, canary)
            answer = decrypt(answer_encrypted, canary)
            
            record = {
                'extra_info': {
                    'instance_id': f"browsecomp_{idx:04d}",
                    'query': problem,
                    'answer': answer,
                    'problem_statement': problem,
                    'data_source': 'browsecomp',
                    'topic': row.get('problem_topic', 'Unknown'),
                    'canary': canary,
                },
                'ability': 'OnlineSearch_BrowseComp',
                'data_source': 'browsecomp',
                'topic': row.get('problem_topic', 'Unknown'),
                'reward_model': {'type': 'llm_judge', 'use_chinese': False}
            }
            records.append(record)
        except Exception as e:
            print(f"[BrowseComp] Failed to decrypt row {idx}: {e}")
            continue
    
    print(f"[BrowseComp] Successfully loaded {len(records)} items")
    return pd.DataFrame(records)


def stratified_sample_by_topic(df: pd.DataFrame, n_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    按类别进行分层采样
    
    Args:
        df: 数据框
        n_samples: 总采样数量
        random_state: 随机种子
        
    Returns:
        采样后的数据框
    """
    if 'topic' not in df.columns:
        print("[Sampling] No 'topic' column found, using random sampling")
        return df.sample(n=min(n_samples, len(df)), random_state=random_state)
    
    # 获取类别分布
    topic_counts = df['topic'].value_counts()
    total_items = len(df)
    
    print(f"\n[Sampling] 按类别等比例采样 {n_samples} 条:")
    
    sampled_dfs = []
    for topic, count in topic_counts.items():
        # 按原始比例计算每个类别应该采样的数量
        topic_ratio = count / total_items
        topic_sample_size = int(n_samples * topic_ratio)
        
        # 确保至少采样1个（如果该类别有数据）
        if topic_sample_size == 0 and count > 0:
            topic_sample_size = 1
        
        # 从该类别中采样
        topic_df = df[df['topic'] == topic]
        actual_sample_size = min(topic_sample_size, len(topic_df))
        
        if actual_sample_size > 0:
            sampled = topic_df.sample(n=actual_sample_size, random_state=random_state)
            sampled_dfs.append(sampled)
            print(f"  - {topic}: {actual_sample_size}/{count} ({100*actual_sample_size/n_samples:.1f}%)")
    
    result = pd.concat(sampled_dfs, ignore_index=True)
    # 随机打乱顺序
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"[Sampling] 总共采样: {len(result)} 条\n")
    return result


# ==================== 评测函数 ====================

async def eval_one_hymem(row, config, tokenizer, model_name, logger: HyMemLogger, save_task_logs: bool):
    """评测单个实例（HyMem框架）"""
    instance_id = row['extra_info'].get('instance_id', 'unknown')
    
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
        'uid': np.array([instance_id], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {
        'generation_kwargs': {}, 
        'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn
    }

    try:
        output = await hymem_process_item(item, context, CallAPI)
        
        extra_data = output.non_tensor_batch.get('extra_data', [{}])[0] if output else {}
        stats = extra_data.get('stats', {})
        
        score = stats.get('get_final_score', 0)
        confidence = stats.get('confidence', 0.0)
        history = stats.get('hymem_history', [])
        
        extra_info = row.get('extra_info', {})
        problem = extra_info.get('query', '') or extra_info.get('problem_statement', '')
        
        answer = ''
        for entry in reversed(history):
            if entry.get('role') == 'planner':
                content = entry.get('content', '')
                import re
                ans_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if ans_match:
                    answer = ans_match.group(1).strip()
                    break
        
        memory_store_data = {
            'episode_memory': stats.get('episode_memory', {}),
            'working_memory': stats.get('working_memory', {}),
            'tool_memory': stats.get('tool_memory', {}),
            'tool_call_history': stats.get('tool_call_history', []),
            'branch_history': stats.get('branch_history', []),
            'fold_count': stats.get('fold_count', 0)
        }
        
        if save_task_logs and history:
            event_log_data = stats.get('event_log')
            if event_log_data:
                event_log_obj = HyMemEventLog()
                event_log_obj.events = event_log_data.get('events', [])
                event_log_obj.token_usage = TokenUsage()
                tu = event_log_data.get('token_usage', {}).get('total', {})
                event_log_obj.token_usage.prompt_tokens = tu.get('prompt_tokens', 0)
                event_log_obj.token_usage.completion_tokens = tu.get('completion_tokens', 0)
                event_log_obj.token_usage.total_tokens = tu.get('total_tokens', 0)
                log_content = HyMemLogFormatter.format_task_log(
                    instance_id=instance_id,
                    problem=problem,
                    event_log=event_log_obj,
                    answer=answer,
                    confidence=confidence,
                    score=score,
                    stats=stats,
                )
            else:
                log_content = f"# HyMem Task: {instance_id}\nAnswer: {answer}\nScore: {score}\n"
            logger.save_task_log(instance_id, log_content)
        
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'topic': row.get('topic', 'Unknown'),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'branch_success': stats.get('branch_success', 0),
            'fold_count': stats.get('fold_count', 0),
            'replan_count': stats.get('replan_count', 0),
            'tool_calls_total': stats.get('tool_calls_total', 0),
            'tool_calls_by_tool': stats.get('tool_calls_by_tool', {}),
            'total_prompt_tokens': stats.get('total_prompt_tokens', 0),
            'total_completion_tokens': stats.get('total_completion_tokens', 0),
            'total_tokens': stats.get('total_tokens', 0),
            'session_time': stats.get('session_time', 0),
            'answer': answer[:500] if answer else '',
            'problem': problem[:200] if problem else ''
        }
        
    except Exception as e:
        print(f"[ERROR] HyMem Task {instance_id} failed: {e}")
        traceback.print_exc()
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'topic': row.get('topic', 'Unknown'),
            'score': 0,
            'confidence': 0.0,
            'status': 'failed',
            'error': str(e),
            'planner_turns': 0,
            'executor_calls': 0,
            'branch_calls': 0,
            'branch_success': 0,
            'fold_count': 0,
            'replan_count': 0,
            'tool_calls_total': 0,
            'tool_calls_by_tool': {},
            'session_time': 0,
            'answer': '',
            'problem': ''
        }
    
    logger.add_log(result)
    return result


async def worker(worker_id, rows, args, pbar, shared_scores, logger):
    """工作线程"""
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    plugin_config = {
        'workflow': 'hymem',
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
        'enable_summary': args.enable_summary,
        'search_mode': 'online',
        'use_chinese_judge': False,
        'hymem_max_planner_turns': args.hymem_max_planner_turns,
        'hymem_max_executor_turns': args.hymem_max_executor_turns,
        'hymem_max_branch_turns': args.hymem_max_branch_turns,
        'hymem_max_branches': args.hymem_max_branches,
        'hymem_max_folds': args.hymem_max_folds,
        'hymem_search_topk': args.hymem_search_topk,
        'hymem_llm_model': args.model_name,
        'hymem_aux_model': args.hymem_aux_model or args.model_name,
        'search_base_url': args.local_search_url,
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
        result = await eval_one_hymem(row, config, tokenizer, args.model_name, logger, args.save_task_logs)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        
        pbar.set_postfix({
            'avg': f"{avg_score:.3f}",
            'id': result['instance_id'][-15:],
            'score': f"{result['score']:.2f}",
            'conf': f"{result['confidence']:.2f}"
        })
        pbar.update(1)

    return results


# ==================== 命令行参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate HyMem on BrowseComp benchmark')
    
    # 数据相关
    parser.add_argument('--browsecomp_cache', default='/data1/test/xq/dataset/browsecomp',
                        help='Cache directory/file for BrowseComp dataset')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Sample size for testing (use stratified sampling by topic)')
    parser.add_argument('--sample_random_state', type=int, default=42,
                        help='Random seed for sampling')
    
    # 输出相关
    parser.add_argument('--output_dir', default='results/browsecomp',
                        help='Directory to save evaluation results')
    parser.add_argument('--log_dir', default='logs/browsecomp_hymem',
                        help='Directory to save detailed execution logs')
    parser.add_argument('--save_task_logs', action='store_true', default=True,
                        help='Save individual task logs')
    
    # 模型相关
    parser.add_argument('--model_name', default='deepseek-reasoner',
                        help='Model name for API')
    parser.add_argument('--tokenizer_path', default='/data0/models/Qwen2.5-7B-Instruct',
                        help='Path to tokenizer model')
    parser.add_argument('--prompt_length', type=int, default=65536,
                        help='Maximum prompt length in tokens')
    parser.add_argument('--response_length', type=int, default=65536,
                        help='Maximum response length in tokens')
    
    # HyMem 特定参数
    parser.add_argument('--hymem_max_planner_turns', type=int, default=12,
                        help='Max planner turns for HyMem')
    parser.add_argument('--hymem_max_executor_turns', type=int, default=10,
                        help='Max executor turns for HyMem')
    parser.add_argument('--hymem_max_branch_turns', type=int, default=4,
                        help='Max turns per branch for HyMem')
    parser.add_argument('--hymem_max_branches', type=int, default=5,
                        help='Max branches per session for HyMem')
    parser.add_argument('--hymem_max_folds', type=int, default=0,
                        help='Max fold operations for HyMem')
    parser.add_argument('--hymem_search_topk', type=int, default=5,
                        help='Top-k results per search for HyMem')
    parser.add_argument('--hymem_aux_model', default=None,
                        help='Auxiliary model for memory compression')
    
    # 执行相关
    parser.add_argument('--max_turn', type=int, default=40,
                        help='Maximum turns')
    parser.add_argument('--val_max_turn', type=int, default=60,
                        help='Maximum turns during validation')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum sessions')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum sessions during validation')
    parser.add_argument('--num_workers', type=int, default=15,
                        help='Number of parallel evaluation workers')
    parser.add_argument('--local_search_url', default='http://localhost:8021',
                        help='URL of the local search server')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置环境变量（BrowseComp 使用在线搜索）
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url
    os.environ["USE_ONLINE_SEARCH"] = "true"
    os.environ["SEARCH_MODE"] = "online"
    
    # 创建日志目录
    logger = HyMemLogger(args.log_dir)
    
    print(f"\n{'='*70}")
    print(f"HyMem on BrowseComp Evaluation")
    print(f"{'='*70}")
    print(f"Model: {args.model_name}")
    print(f"Search URL: {args.local_search_url}")
    print(f"Log Dir: {args.log_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"\nHyMem Config:")
    print(f"  - Max Planner Turns: {args.hymem_max_planner_turns}")
    print(f"  - Max Executor Turns: {args.hymem_max_executor_turns}")
    print(f"  - Max Branch Turns: {args.hymem_max_branch_turns}")
    print(f"  - Max Branches: {args.hymem_max_branches}")
    print(f"  - Max Folds: {args.hymem_max_folds}")
    print(f"  - Search Top-K: {args.hymem_search_topk}")
    print(f"{'='*70}\n")
    
    # 加载 BrowseComp 数据集
    df = load_browsecomp_openai(cache_path=args.browsecomp_cache)
    df = df.sample(n=50, random_state=42)
    
    # 采样（如果指定）
    if args.sample_size and args.sample_size < len(df):
        print(f"原始数据: {len(df)} 条")
        df = stratified_sample_by_topic(df, args.sample_size, args.sample_random_state)
    
    print(f"最终评测: {len(df)} 条\n")

    # 分配任务给workers
    num_workers = min(args.num_workers, len(df))
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < num_workers-1 else len(df)]
              for i in range(num_workers)]

    # 运行评测
    async def run_all():
        shared_scores = []
        desc = "Evaluating HyMem on BrowseComp"
        with tqdm(total=len(df), desc=desc, unit="task") as pbar:
            tasks = [
                worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], 
                       args, pbar, shared_scores, logger)
                for i in range(num_workers)
            ]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # 汇总统计
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    
    avg_score = np.mean([r['score'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"Overall Score: {avg_score:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"\nHyMem Statistics:")
    print(f"  - Planner Turns: {np.mean([r['planner_turns'] for r in results]):.2f} avg")
    print(f"  - Executor Calls: {np.mean([r['executor_calls'] for r in results]):.2f} avg")
    print(f"  - Branch Calls: {np.mean([r.get('branch_calls', 0) for r in results]):.2f} avg")
    print(f"  - Branch Success: {np.mean([r.get('branch_success', 0) for r in results]):.2f} avg")
    print(f"  - Tool Calls (Unified): {np.mean([r.get('tool_calls_total', 0) for r in results]):.2f} avg")
    print(f"  - Fold Count: {np.mean([r.get('fold_count', 0) for r in results]):.2f} avg")
    print(f"  - Replans: {np.mean([r.get('replan_count', 0) for r in results]):.2f} avg")
    print(f"  - Session Time: {np.mean([r['session_time'] for r in results]):.1f}s avg")

    # 按类别统计
    from collections import defaultdict
    by_topic = defaultdict(list)
    for r in results:
        by_topic[r.get('topic', 'Unknown')].append(r)

    print(f"\nBy Topic:")
    for topic in sorted(by_topic.keys()):
        topic_results = by_topic[topic]
        topic_scores = [r['score'] for r in topic_results]
        print(f"  {topic}: {np.mean(topic_scores):.4f} ({len(topic_scores)} items)")

    # 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = Path(args.output_dir) / f"hymem_browsecomp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary_by_topic = {
        topic: {
            'avg_score': float(np.mean([r['score'] for r in rs])),
            'avg_confidence': float(np.mean([r['confidence'] for r in rs])),
            'count': len(rs)
        }
        for topic, rs in by_topic.items()
    }
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model_name,
            'framework': 'HyMem',
            'dataset': 'BrowseComp (OpenAI)',
            'total_items': len(results),
            'success_count': success_count,
            'hymem_config': {
                'max_planner_turns': args.hymem_max_planner_turns,
                'max_executor_turns': args.hymem_max_executor_turns,
                'max_branch_turns': args.hymem_max_branch_turns,
                'max_branches': args.hymem_max_branches,
                'max_folds': args.hymem_max_folds,
            }
        },
        'summary': {
            'avg_score': float(avg_score),
            'avg_confidence': float(avg_confidence),
            'avg_planner_turns': float(np.mean([r['planner_turns'] for r in results])),
            'avg_executor_calls': float(np.mean([r['executor_calls'] for r in results])),
            'avg_branch_calls': float(np.mean([r.get('branch_calls', 0) for r in results])),
            'avg_branch_success': float(np.mean([r.get('branch_success', 0) for r in results])),
            'avg_fold_count': float(np.mean([r.get('fold_count', 0) for r in results])),
            'avg_replan_count': float(np.mean([r.get('replan_count', 0) for r in results])),
            'avg_session_time': float(np.mean([r['session_time'] for r in results])),
        },
        'by_topic': summary_by_topic,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    # 保存日志汇总
    if hasattr(logger, 'save_all'):
        log_json_file = logger.save_all()
        log_summary_file = logger.save_summary()
    else:
        log_json_file = logger.save_all_logs()
        log_summary_file = logger.save_readable_summary()
    print(f"Log JSON saved to: {log_json_file}")
    print(f"Log Summary saved to: {log_summary_file}")
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

