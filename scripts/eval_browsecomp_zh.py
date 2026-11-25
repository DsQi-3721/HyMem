#!/usr/bin/env python3
"""
BrowseComp-ZH 评测脚本

专门用于在 BrowseComp-ZH 数据集上评测 HyMem Agent。
强制使用在线搜索，使用中文评判模板。

使用方法:
    python scripts/eval_browsecomp_zh.py \
        --data_path ../BrowseComp-ZH/raw_data/browsecomp-zh-decrypted.json \
        --model_name deepseek-reasoner \
        --num_workers 5

环境变量:
    OPENAI_API_KEY: OpenAI API Key（用于 LLM 调用和评判）
    OPENAI_BASE_URL: OpenAI API 基础 URL（可选）
    GOOGLE_API_KEY: Google Custom Search API Key（可选）
    GOOGLE_SEARCH_ENGINE_ID: Google Custom Search Engine ID（可选）

注意:
    - 如果没有配置 Google API Key，将自动使用 DuckDuckGo 搜索
    - 评测使用中文评判模板进行 LLM-as-Judge 评分
"""

import os
import sys

# 强制使用在线搜索模式
os.environ["USE_ONLINE_SEARCH"] = "true"
os.environ["SEARCH_MODE"] = "online"

# 添加父目录到路径
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agents.hymem_agent import process_item as hymem_process_item
from agents.utils import CallAPI, TaskContext
from verl import DataProto


def load_browsecomp_zh(json_path: str) -> pd.DataFrame:
    """加载 BrowseComp-ZH JSON 数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for idx, item in enumerate(data):
        record = {
            'extra_info': {
                'instance_id': f"browsecomp_zh_{idx:04d}",
                'query': item['Question'],
                'answer': item['Answer'],
                'topic': item.get('Topic', ''),
                'problem_statement': item['Question'],
                'data_source': 'browsecomp_zh'
            },
            'ability': 'OnlineSearch_BrowseComp_ZH',
            'data_source': 'browsecomp_zh',
            'reward_model': {'type': 'llm_judge', 'use_chinese': True}
        }
        records.append(record)
    
    return pd.DataFrame(records)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate HyMem on BrowseComp-ZH benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本使用
    python scripts/eval_browsecomp_zh.py \\
        --data_path ../BrowseComp-ZH/raw_data/browsecomp-zh-decrypted.json
    
    # 指定模型和并发数
    python scripts/eval_browsecomp_zh.py \\
        --data_path ../BrowseComp-ZH/raw_data/browsecomp-zh-decrypted.json \\
        --model_name deepseek-reasoner \\
        --num_workers 3
    
    # 评测前 N 条数据
    python scripts/eval_browsecomp_zh.py \\
        --data_path ../BrowseComp-ZH/raw_data/browsecomp-zh-decrypted.json \\
        --max_samples 10
"""
    )
    
    parser.add_argument('--data_path', required=True,
                        help='Path to BrowseComp-ZH JSON file')
    parser.add_argument('--output_dir', default='results/browsecomp_zh',
                        help='Directory to save evaluation results')
    parser.add_argument('--log_dir', default='logs/browsecomp_zh',
                        help='Directory to save detailed execution logs')
    parser.add_argument('--model_name', default='deepseek-reasoner',
                        help='Model name for API')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of parallel evaluation workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (for testing)')
    parser.add_argument('--tokenizer_path', default='/data0/models/Qwen2.5-7B-Instruct',
                        help='Path to tokenizer model')
    
    # HyMem 配置
    parser.add_argument('--max_planner_turns', type=int, default=12,
                        help='Max planner turns')
    parser.add_argument('--max_executor_turns', type=int, default=10,
                        help='Max executor turns')
    parser.add_argument('--max_branch_turns', type=int, default=4,
                        help='Max turns per branch')
    parser.add_argument('--max_branches', type=int, default=5,
                        help='Max branches per session')
    parser.add_argument('--max_folds', type=int, default=3,
                        help='Max fold operations')
    
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name):
    """评测单个实例"""
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
        'data_source': np.array([row['data_source']], dtype=object),
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
        
        result = {
            'instance_id': instance_id,
            'topic': row['extra_info'].get('topic', ''),
            'question': row['extra_info'].get('query', '')[:100],
            'label': row['extra_info'].get('answer', ''),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'fold_count': stats.get('fold_count', 0),
            'session_time': stats.get('session_time', 0),
        }
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Task {instance_id} failed: {e}")
        traceback.print_exc()
        result = {
            'instance_id': instance_id,
            'topic': row['extra_info'].get('topic', ''),
            'question': row['extra_info'].get('query', '')[:100],
            'label': row['extra_info'].get('answer', ''),
            'score': 0,
            'confidence': 0.0,
            'status': 'failed',
            'error': str(e),
            'planner_turns': 0,
            'executor_calls': 0,
            'branch_calls': 0,
            'fold_count': 0,
            'session_time': 0,
        }
    
    return result


async def worker(worker_id, rows, args, pbar, shared_scores, config, tokenizer):
    """工作线程"""
    results = []
    for row in rows:
        result = await eval_one(row, config, tokenizer, args.model_name)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        
        pbar.set_postfix({
            'avg': f"{avg_score:.3f}",
            'id': result['instance_id'][-8:],
            'score': f"{result['score']:.0f}"
        })
        pbar.update(1)

    return results


def main():
    args = parse_args()
    
    print(f"\n{'='*60}")
    print("BrowseComp-ZH Evaluation (HyMem + OnlineSearch)")
    print(f"{'='*60}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Workers: {args.num_workers}")
    print(f"Search Mode: ONLINE (forced)")
    print(f"Judge Template: CHINESE (forced)")
    print(f"{'='*60}\n")
    
    # 检查环境变量
    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] OPENAI_API_KEY not set. LLM calls may fail.")
    
    google_key = os.getenv("GOOGLE_API_KEY")
    google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    if not (google_key and google_engine_id):
        print("[INFO] No Google API key found. Using DuckDuckGo for search.")
    
    # 加载数据
    df = load_browsecomp_zh(args.data_path)
    df = df.sample(n=10, random_state=21)
    print(f"Loaded {len(df)} items from BrowseComp-ZH")
    
    if args.max_samples:
        df = df.head(args.max_samples)
        print(f"Limited to first {args.max_samples} samples")
    
     
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # 创建配置
    plugin_config = {
        'workflow': 'hymem',
        'max_turn': 200,
        'val_max_turn': 200,
        'max_session': 10,
        'val_max_session': 10,
        'session_timeout': 5400,
        'must_finish': False,
        'double_check': False,
        'must_search': False,
        'hymem_max_planner_turns': args.max_planner_turns,
        'hymem_max_executor_turns': args.max_executor_turns,
        'hymem_max_branch_turns': args.max_branch_turns,
        'hymem_max_branches': args.max_branches,
        'hymem_max_folds': args.max_folds,
        'hymem_search_topk': 10,
        'hymem_llm_model': args.model_name,
        'hymem_aux_model': args.model_name,
        # BrowseComp-ZH 专用配置
        'search_mode': 'online',
        'use_chinese_judge': True,
    }
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': 16384,
            'response_length': 32768,
            'plugin': plugin_config
        }}
    })
    
    # 分配任务
    num_workers = min(args.num_workers, len(df))
    chunk_size = len(df) // num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < num_workers-1 else len(df)]
              for i in range(num_workers)]
    
    # 运行评测
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating BrowseComp-ZH", unit="task") as pbar:
            tasks = [
                worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))],
                       args, pbar, shared_scores, config, tokenizer)
                for i in range(num_workers)
            ]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # 统计结果
    print(f"\n{'='*60}")
    print("BROWSECOMP-ZH EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    avg_score = np.mean([r['score'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    correct_count = sum(1 for r in results if r['score'] == 1)
    
    print(f"Overall Accuracy: {avg_score:.4f} ({correct_count}/{len(results)})")
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"Average Planner Turns: {np.mean([r['planner_turns'] for r in results]):.2f}")
    print(f"Average Executor Calls: {np.mean([r['executor_calls'] for r in results]):.2f}")
    
    # 按类别统计
    from collections import defaultdict
    by_topic = defaultdict(list)
    for r in results:
        by_topic[r['topic']].append(r)
    
    print(f"\nBy Topic:")
    for topic in sorted(by_topic.keys()):
        topic_results = by_topic[topic]
        topic_scores = [r['score'] for r in topic_results]
        correct = sum(1 for s in topic_scores if s == 1)
        print(f"  {topic}: {np.mean(topic_scores):.4f} ({correct}/{len(topic_scores)})")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(args.output_dir) / f"browsecomp_zh_results_{timestamp}.json"
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model_name,
            'total_items': len(results),
            'correct_count': correct_count,
            'accuracy': float(avg_score),
        },
        'by_topic': {
            topic: {
                'accuracy': float(np.mean([r['score'] for r in rs])),
                'count': len(rs),
                'correct': sum(1 for r in rs if r['score'] == 1)
            }
            for topic, rs in by_topic.items()
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
