#!/usr/bin/env python3
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.fold_agent import process_item
from agents.utils import CallAPI, TaskContext, print_distractor_stats
from verl import DataProto
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
        转换后的 DataFrame，格式与现有评测兼容
    """
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    # 处理缓存路径：如果是目录，则在目录下创建默认文件名
    if cache_path:
        if os.path.isdir(cache_path):
            cache_path = os.path.join(cache_path, 'browsecomp_openai_cache.csv')
            print(f"[BrowseComp-OpenAI] Cache path is directory, using: {cache_path}")
    
    # 尝试从缓存加载或下载
    if cache_path and os.path.exists(cache_path):
        print(f"[BrowseComp-OpenAI] Loading from cache: {cache_path}")
        df_raw = pd.read_csv(cache_path)
    else:
        print(f"[BrowseComp-OpenAI] Downloading from {url}")
        df_raw = pd.read_csv(url)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True) if os.path.dirname(cache_path) else None
            df_raw.to_csv(cache_path, index=False)
            print(f"[BrowseComp-OpenAI] Cached to {cache_path}")
    
    # 打印类别分布
    if 'problem_topic' in df_raw.columns:
        print(f"\n[BrowseComp-OpenAI] 类别分布:")
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
                    'instance_id': f"browsecomp_openai_{idx:04d}",
                    'query': problem,
                    'answer': answer,
                    'problem_statement': problem,
                    'data_source': 'browsecomp_openai',
                    'topic': row.get('problem_topic', 'Unknown'),
                    'canary': canary,
                },
                'ability': 'OnlineSearch_BrowseComp',
                'data_source': 'browsecomp_openai',
                'topic': row.get('problem_topic', 'Unknown'),
                'reward_model': {'type': 'llm_judge', 'use_chinese': False}
            }
            records.append(record)
        except Exception as e:
            print(f"[BrowseComp-OpenAI] Failed to decrypt row {idx}: {e}")
            continue
    
    print(f"[BrowseComp-OpenAI] Loaded {len(records)} items")
    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate agents on BrowseComp-Plus benchmark')
    parser.add_argument('--data_path', default='/data1/test/xq/dataset/bc_test.parquet',
                        help='Path to test data parquet file (default: data/bc_test.parquet)')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save evaluation results (default: results)')
    # BrowseComp 数据格式参数
    parser.add_argument('--data_format', default='parquet',
                        choices=['parquet', 'browsecomp_openai'],
                        help='Data format: parquet or browsecomp_openai (OpenAI BrowseComp)')
    parser.add_argument('--browsecomp_cache', default=None,
                        help='Cache path for BrowseComp OpenAI dataset (optional)')
    parser.add_argument('--search_mode', default='local',
                        choices=['local', 'online', 'auto'],
                        help='Search mode: local (LocalSearch), online (OnlineSearch), or auto')
    parser.add_argument('--prompt_length', type=int, default=16384,
                        help='Maximum prompt length in tokens (default: 16384)')
    parser.add_argument('--response_length', type=int, default=16384 ,
                        help='Maximum response length in tokens (default: 32768)')
    parser.add_argument('--workflow', default='search',
                        help='Agent workflow: "search" for ReAct, "search_branch" for Context-Folding (default: search_branch)')
    parser.add_argument('--max_turn', type=int, default=100,
                        help='Maximum turns during training (default: 100, matches paper Sec A.2 ReAct cap)')
    parser.add_argument('--val_max_turn', type=int, default=100,
                        help='Maximum turns during validation/evaluation (default: 100, matches paper Sec A.2 ReAct cap)')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during training (default: 10)')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during validation (default: 10)')
    parser.add_argument('--model_name', default='Qwen3-32B',
                        help='Model name for API (e.g., gpt-5-nano, gpt-4o, or vLLM model path) (default: gpt-5-nano)')
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of parallel evaluation workers (default: 150)')
    parser.add_argument('--local_search_url', default='http://localhost:8021',
                        help='URL of the local search server (default: http://localhost:8021)')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode (use with workflow=search for Summary agent)')
    # 测试模型 API 配置
    parser.add_argument('--base_url', default=None,
                        help='Base URL for test model API (default: use OPENAI_BASE_URL env)')
    parser.add_argument('--api_key', default=None,
                        help='API key for test model (default: use OPENAI_API_KEY env)')
    # Judge 模型 API 配置
    parser.add_argument('--judge_model', default=None,
                        help='Model name for judge (default: use JUDGE_MODEL env or same as test model)')
    parser.add_argument('--judge_base_url', default=None,
                        help='Base URL for judge model API (default: use JUDGE_BASE_URL env or same as test model)')
    parser.add_argument('--judge_api_key', default=None,
                        help='API key for judge model (default: use JUDGE_API_KEY env or same as test model)')
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name):
    context = TaskContext(config=config, global_step=0, server_host=model_name,
                          server_port=0, is_train=False, tokenizer=tokenizer)

    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id', 'unknown')], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}

    output = await process_item(item, context, CallAPI)

    # 提取统计信息
    stats = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}) if output else {}
    
    result = {
        'instance_id': row['extra_info'].get('instance_id', 'unknown'),
        'data_source': row.get('data_source', 'unknown'),
        'score': stats.get('score', 0),
        # Step A3 — unified tool-call schema. Compare 'tool_calls_atomic'
        # (atomic counts) across frameworks; old 'tool_calls' was the LLM
        # iteration counter and is kept only as a legacy alias.
        'tool_calls_atomic': stats.get('tool_calls_atomic', 0),
        'agent_turns': stats.get('agent_turns', 0),
        'tool_calls_by_source': stats.get('tool_calls_by_source', {}),
        'framework': stats.get('framework', 'react'),
        'tool_calls': stats.get('tool_calls', 0),  # legacy alias
        'total_turns': stats.get('total_turns', 0),
        'session_time': stats.get('session_time', 0),
        'status': 'success' if output else 'failed'
    }
    return result


async def worker(worker_id, rows, args, pbar, shared_scores):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': {
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
                'enable_summary': args.enable_summary
            }
        }}
    })

    results = []
    for row in rows:
        result = await eval_one(row, config, tokenizer, args.model_name)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        pbar.set_postfix({'avg_score': f"{avg_score:.3f}", 'id': result['instance_id']})
        pbar.update(1)

    return results


def main():
    args = parse_args()
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url
    
    # 设置测试模型 API 配置
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    
    # 设置 Judge 模型 API 配置
    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model
    if args.judge_base_url:
        os.environ["JUDGE_BASE_URL"] = args.judge_base_url
    if args.judge_api_key:
        os.environ["JUDGE_API_KEY"] = args.judge_api_key
    
    # 设置搜索模式环境变量
    if args.search_mode == 'online':
        os.environ["USE_ONLINE_SEARCH"] = "true"
        os.environ["SEARCH_MODE"] = "online"
    elif args.search_mode == 'local':
        os.environ["SEARCH_MODE"] = "local"

    # 加载数据 - 支持多种格式
    print(f"\n{'='*60}")
    print(f"Data Format: {args.data_format}")
    if args.data_format == 'browsecomp_openai':
        print(f"BrowseComp Cache: {args.browsecomp_cache or 'None (will download)'}")
    print(f"Search Mode: {args.search_mode}")
    print(f"\n[Test Model Config]")
    print(f"  Model: {args.model_name}")
    print(f"  Base URL: {os.environ.get('OPENAI_BASE_URL', '(not set)')}")
    print(f"  API Key: {'***' + os.environ.get('OPENAI_API_KEY', '')[-4:] if os.environ.get('OPENAI_API_KEY') else '(not set)'}")
    print(f"\n[Judge Model Config]")
    print(f"  Model: {os.environ.get('JUDGE_MODEL', '(use default)')}")
    print(f"  Base URL: {os.environ.get('JUDGE_BASE_URL', '(use test model config)')}")
    print(f"  API Key: {'***' + os.environ.get('JUDGE_API_KEY', '')[-4:] if os.environ.get('JUDGE_API_KEY') else '(use test model config)'}")
    print(f"{'='*60}\n")
    
    if args.data_format == 'browsecomp_openai':
        # 加载 OpenAI BrowseComp 官方数据集
        df = load_browsecomp_openai(cache_path=args.browsecomp_cache)
    else:
        # 原有 Parquet 加载
        df = pd.read_parquet(args.data_path)

    #df = df.sample(n=50, random_state=21)
    print(f"Loaded {len(df)} items")

    # Split for workers
    chunk_size = len(df) // args.num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < args.num_workers-1 else len(df)]
              for i in range(args.num_workers)]

    # Run workers with progress bar
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating", unit="item") as pbar:
            tasks = [worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], args, pbar, shared_scores)
                     for i in range(args.num_workers)]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # Summary overall
    avg_score = np.mean([r['score'] for r in results])
    avg_tool_calls = np.mean([r['tool_calls'] for r in results])
    avg_turns = np.mean([r['total_turns'] for r in results])
    avg_time = np.mean([r['session_time'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"Overall - Avg Score: {avg_score:.4f}, Success: {sum(r['status']=='success' for r in results)}/{len(results)}")
    print(f"Tool Calls: {avg_tool_calls:.1f} avg, Turns: {avg_turns:.1f} avg, Time: {avg_time:.1f}s avg")

    # Summary by data_source
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        by_source[r['data_source']].append(r['score'])

    print(f"\nBy Data Source:")
    for source in sorted(by_source.keys()):
        scores = by_source[source]
        print(f"  {source}: {np.mean(scores):.4f} ({len(scores)} items)")

    # Save
    Path(args.output_dir).mkdir(exist_ok=True)
    output_file = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_by_source = {src: {'avg_score': float(np.mean(scores)), 'count': len(scores)}
                         for src, scores in by_source.items()}
    json.dump({
        'avg_score': avg_score,
        'avg_tool_calls': avg_tool_calls,
        'avg_turns': avg_turns,
        'avg_session_time': avg_time,
        'by_source': summary_by_source,
        'results': results
    }, open(output_file, 'w'), indent=2)
    print(f"\nSaved to {output_file}")
    
    # 打印干扰注入统计（如果启用）
    print_distractor_stats()


if __name__ == "__main__":
    main()