#!/usr/bin/env python3
"""
HyMem Qwen3 测试脚本

专门用于测试 Qwen3-32B 与 HyMem 框架的兼容性。
使用优化版 prompt (prompt_style="qwen") 解决 Qwen3 的输出格式问题。

主要改进：
1. 简化的 prompt 格式，更适合 Qwen3 理解
2. 强制限制思考长度，避免过长的 <think> 输出
3. 更明确的输出格式要求

使用方法：
    # 使用默认配置
    python eval_bc_qwen.py

    # 指定数据和模型
    python eval_bc_qwen.py --data_path /path/to/data.parquet --model_name Qwen3-32B
    
    # 使用在线搜索
    python eval_bc_qwen.py --search_mode online
"""
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import os
import warnings
import traceback
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
from verl import DataProto


def load_browsecomp_zh_json(json_path: str) -> pd.DataFrame:
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
    
    print(f"[Qwen3-Test] Loaded {len(records)} items from {json_path}")
    return pd.DataFrame(records)


class QwenTestLogger:
    """Qwen3 测试专用日志记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
    
    def add_log(self, log_entry: Dict[str, Any]):
        self.logs.append(log_entry)
    
    def save_task_log(self, instance_id: str, log_content: str):
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        log_file = self.log_dir / f"qwen_task_{safe_id}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
    
    def save_all_logs(self, filename: str = None):
        if filename is None:
            filename = f"qwen_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_file = self.log_dir / filename
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        
        return log_file
    
    def save_summary(self, filename: str = None):
        if filename is None:
            filename = f"qwen_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        summary_file = self.log_dir / filename
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Qwen3-32B HyMem Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总任务数: {len(self.logs)}\n")
            f.write(f"成功任务: {sum(1 for l in self.logs if l.get('status') == 'success')}\n")
            if self.logs:
                f.write(f"平均分数: {np.mean([l.get('score', 0) for l in self.logs]):.4f}\n")
                f.write(f"平均置信度: {np.mean([l.get('confidence', 0) for l in self.logs]):.4f}\n")
                f.write(f"平均Planner轮次: {np.mean([l.get('planner_turns', 0) for l in self.logs]):.2f}\n")
            f.write("\n")
            
            for log in self.logs:
                f.write(f"\n{'-'*50}\n")
                f.write(f"Instance: {log.get('instance_id', 'unknown')}\n")
                f.write(f"Score: {log.get('score', 0)}\n")
                f.write(f"Confidence: {log.get('confidence', 0):.2f}\n")
                f.write(f"Status: {log.get('status', 'unknown')}\n")
                if log.get('error'):
                    f.write(f"Error: {log.get('error')}\n")
        
        return summary_file


def parse_args():
    parser = argparse.ArgumentParser(description='Test HyMem with Qwen3-32B using optimized prompts')
    
    # 数据配置
    parser.add_argument('--data_path', default='/data1/test/xq/dataset/bc_test.parquet',
                        help='Path to test data')
    parser.add_argument('--data_format', default='parquet', choices=['parquet', 'json'],
                        help='Data format')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to test (0 for all)')
    
    # 模型配置
    parser.add_argument('--model_name', default='Qwen3-32B',
                        help='Model name')
    parser.add_argument('--tokenizer_path', default='/data0/models/Qwen2.5-7B-Instruct',
                        help='Path to tokenizer')
    
    # 搜索配置
    parser.add_argument('--search_mode', default='local', choices=['local', 'online', 'auto'],
                        help='Search mode')
    parser.add_argument('--local_search_url', default='http://localhost:8021',
                        help='Local search server URL')
    
    # HyMem 配置
    parser.add_argument('--max_planner_turns', type=int, default=12,
                        help='Max planner turns')
    parser.add_argument('--max_executor_turns', type=int, default=8,
                        help='Max executor turns')
    parser.add_argument('--max_branch_turns', type=int, default=4,
                        help='Max branch turns')
    parser.add_argument('--max_branches', type=int, default=3,
                        help='Max branches per session')
    parser.add_argument('--max_folds', type=int, default=5,
                        help='Max fold operations')
    
    # 输出配置
    parser.add_argument('--output_dir', default='results',
                        help='Output directory')
    parser.add_argument('--log_dir', default='logs/qwen_test',
                        help='Log directory')
    
    # 调试配置
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--save_task_logs', action='store_true', default=True,
                        help='Save individual task logs')
    
    return parser.parse_args()


async def eval_one_qwen(row, config, tokenizer, model_name, logger, save_task_logs: bool):
    """评测单个实例（Qwen3 + HyMem）"""
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
        
        # 提取答案
        import re
        answer = ''
        for entry in reversed(history):
            if entry.get('role') == 'planner':
                content = entry.get('content', '')
                ans_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                if ans_match:
                    answer = ans_match.group(1).strip()
                    break
        
        # 保存详细日志
        if save_task_logs:
            log_content = f"""
{'='*60}
Qwen3-32B HyMem Task: {instance_id}
{'='*60}

Problem: {problem}

History ({len(history)} entries):
"""
            for entry in history:
                role = entry.get('role', '')
                turn = entry.get('turn', 0)
                if role == 'planner':
                    content = entry.get('content', '')[:2000]
                    log_content += f"\n[Turn {turn}] Planner:\n{content}\n"
                elif role == 'executor':
                    task = entry.get('task', '')
                    result = entry.get('result', '')[:1000]
                    log_content += f"\n[Turn {turn}] Executor:\nTask: {task}\nResult: {result}\n"
                elif role == 'branch':
                    desc = entry.get('description', '')
                    result = entry.get('result', '')[:500]
                    log_content += f"\n[Turn {turn}] Branch ({desc}):\n{result}\n"
            
            log_content += f"""
{'='*60}
Final Answer: {answer}
Confidence: {confidence:.2f}
Score: {score}
{'='*60}
"""
            logger.save_task_log(instance_id, log_content)
        
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'fold_count': stats.get('fold_count', 0),
            'session_time': stats.get('session_time', 0),
            'answer': answer[:500] if answer else '',
            'problem': problem[:200] if problem else ''
        }
        
    except Exception as e:
        print(f"[ERROR] Qwen Task {instance_id} failed: {e}")
        traceback.print_exc()
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': 0,
            'confidence': 0.0,
            'status': 'failed',
            'error': str(e),
            'planner_turns': 0,
            'executor_calls': 0,
            'branch_calls': 0,
            'fold_count': 0,
            'session_time': 0,
            'answer': '',
            'problem': ''
        }
    
    logger.add_log(result)
    return result


async def main_async(args):
    """异步主函数"""
    # 设置环境变量
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url
    os.environ["SEARCH_MODE"] = args.search_mode
    os.environ["HYMEM_PROMPT_STYLE"] = "qwen"  # 关键：使用 Qwen 优化版 prompt
    
    if args.debug:
        os.environ["HYMEM_DEBUG"] = "true"
    
    # 初始化日志记录器
    logger = QwenTestLogger(args.log_dir)
    
    print(f"\n{'='*60}")
    print("Qwen3-32B HyMem Evaluation (Optimized Prompts)")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Prompt Style: qwen (optimized)")
    print(f"Data: {args.data_path}")
    print(f"Search Mode: {args.search_mode}")
    print(f"Max Planner Turns: {args.max_planner_turns}")
    print(f"Max Executor Turns: {args.max_executor_turns}")
    print(f"Debug: {args.debug}")
    print(f"{'='*60}\n")
    
    # 加载数据
    if args.data_format == 'json':
        df = load_browsecomp_zh_json(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)
    
    # 采样
    if args.num_samples > 0 and args.num_samples < len(df):
        df = df.sample(n=args.num_samples, random_state=42)
    
    print(f"Testing {len(df)} samples")
    
    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    # 创建配置
    plugin_config = {
        'workflow': 'hymem',
        'val_max_turn': 60,
        'search_mode': args.search_mode,
        'hymem_max_planner_turns': args.max_planner_turns,
        'hymem_max_executor_turns': args.max_executor_turns,
        'hymem_max_branch_turns': args.max_branch_turns,
        'hymem_max_branches': args.max_branches,
        'hymem_max_folds': args.max_folds,
        'hymem_llm_model': args.model_name,
        'hymem_aux_model': args.model_name,
        'hymem_prompt_style': 'qwen',  # 使用 Qwen 优化版 prompt
        'search_base_url': args.local_search_url,
    }
    
    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': 16384,
            'response_length': 16384,
            'plugin': plugin_config
        }}
    })
    
    # 运行评测
    results = []
    with tqdm(total=len(df), desc="Evaluating Qwen3", unit="task") as pbar:
        for idx in range(len(df)):
            row = df.iloc[idx]
            result = await eval_one_qwen(row, config, tokenizer, args.model_name, logger, args.save_task_logs)
            results.append(result)
            
            # 更新进度条
            avg_score = np.mean([r['score'] for r in results])
            pbar.set_postfix({
                'avg': f"{avg_score:.3f}",
                'id': result['instance_id'][:15],
                'score': f"{result['score']:.2f}"
            })
            pbar.update(1)
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("QWEN3-32B HYMEM EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    avg_score = np.mean([r['score'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"Overall Score: {avg_score:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"Average Planner Turns: {np.mean([r['planner_turns'] for r in results]):.2f}")
    print(f"Average Executor Calls: {np.mean([r['executor_calls'] for r in results]):.2f}")
    print(f"Average Branch Calls: {np.mean([r.get('branch_calls', 0) for r in results]):.2f}")
    print(f"Average Fold Count: {np.mean([r.get('fold_count', 0) for r in results]):.2f}")
    
    # 保存结果
    Path(args.output_dir).mkdir(exist_ok=True)
    
    output_file = Path(args.output_dir) / f"qwen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model_name,
            'prompt_style': 'qwen',
            'total_items': len(results),
            'success_count': success_count,
        },
        'summary': {
            'avg_score': float(avg_score),
            'avg_confidence': float(avg_confidence),
            'avg_planner_turns': float(np.mean([r['planner_turns'] for r in results])),
            'avg_executor_calls': float(np.mean([r['executor_calls'] for r in results])),
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")
    
    # 保存日志
    log_file = logger.save_all_logs()
    summary_file = logger.save_summary()
    print(f"Log saved to: {log_file}")
    print(f"Summary saved to: {summary_file}")
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

