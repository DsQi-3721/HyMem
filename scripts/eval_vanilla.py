#!/usr/bin/env python3
"""
Vanilla LLM 评测脚本（无框架对比基准）

直接使用 LLM 回答问题，不使用任何 Agent 框架（HyMem/ReAct），
仅用于对比实验，使用 LLM as a Judge 评分。
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
import base64
import hashlib
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from openai import AsyncOpenAI


# ==================== 数据解密（BrowseComp OpenAI） ====================

def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


# ==================== 数据加载 ====================

def load_browsecomp_openai(cache_path: str = None) -> pd.DataFrame:
    """加载 OpenAI BrowseComp 官方数据集"""
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
    if cache_path and os.path.isdir(cache_path):
        cache_path = os.path.join(cache_path, 'browsecomp_openai_cache.csv')
    
    if cache_path and os.path.exists(cache_path):
        print(f"[BrowseComp-OpenAI] Loading from cache: {cache_path}")
        df_raw = pd.read_csv(cache_path)
    else:
        print(f"[BrowseComp-OpenAI] Downloading from {url}")
        df_raw = pd.read_csv(url)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            df_raw.to_csv(cache_path, index=False)
    
    records = []
    for idx, row in df_raw.iterrows():
        try:
            canary = row.get('canary', '')
            problem = decrypt(row.get('problem', ''), canary)
            answer = decrypt(row.get('answer', ''), canary)
            records.append({
                'instance_id': f"browsecomp_openai_{idx:04d}",
                'question': problem,
                'answer': answer,
                'topic': row.get('problem_topic', 'Unknown'),
                'data_source': 'browsecomp_openai',
            })
        except Exception as e:
            print(f"[BrowseComp-OpenAI] Failed to decrypt row {idx}: {e}")
    
    print(f"[BrowseComp-OpenAI] Loaded {len(records)} items")
    return pd.DataFrame(records)


def load_browsecomp_parquet(parquet_path: str) -> pd.DataFrame:
    """加载 bc_test.parquet 数据"""
    df = pd.read_parquet(parquet_path)
    records = []
    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        records.append({
            'instance_id': extra_info.get('instance_id', f'bc_{idx:04d}'),
            'question': extra_info.get('query', '') or extra_info.get('problem_statement', ''),
            'answer': extra_info.get('answer', ''),
            'topic': extra_info.get('topic', row.get('data_source', 'unknown')),
            'data_source': row.get('data_source', 'browsecomp'),
        })
    print(f"[BrowseComp-Parquet] Loaded {len(records)} items from {parquet_path}")
    return pd.DataFrame(records)


def load_gaia_dataset(parquet_path: str, gaia_root: str = None) -> pd.DataFrame:
    """加载 GAIA 数据集"""
    df = pd.read_parquet(parquet_path)
    if gaia_root is None:
        gaia_root = str(Path(parquet_path).parent.parent.parent)
    
    records = []
    for idx, row in df.iterrows():
        task_id = row.get('task_id', f'gaia_{idx:04d}')
        question = row.get('Question', '')
        answer = row.get('Final answer', '')  # 注意：字段名是 "Final answer"
        level = row.get('Level', '1')
        file_name = row.get('file_name', None)
        file_path = row.get('file_path', None)
        
        # 处理 NaN 值
        if pd.isna(file_name):
            file_name = None
        if pd.isna(file_path):
            file_path = None
        if pd.isna(answer):
            answer = ""
        if pd.isna(question):
            question = ""
        
        # 确保 answer 是字符串
        answer = str(answer) if answer else ""
        
        # 确保 level 是整数（parquet 中可能是字符串 "1"）
        try:
            level = int(level) if not pd.isna(level) else 1
        except (ValueError, TypeError):
            level = 1
        
        # 处理文件附件
        file_content = ""
        has_file = False
        if file_path:
            abs_path = os.path.join(gaia_root, file_path)
            has_file = True
            if os.path.exists(abs_path):
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_content = f.read()[:4000]
                except:
                    pass
        
        full_question = question
        if file_content:
            full_question = f"{question}\n\n[Attached File Content]:\n{file_content}"
        
        records.append({
            'instance_id': task_id,
            'question': full_question,
            'answer': answer,
            'level': level,
            'has_file': has_file,
            'data_source': 'gaia',
        })
    
    print(f"[GAIA] Loaded {len(records)} items from {parquet_path}")
    print(f"[GAIA] With files: {sum(1 for r in records if r['has_file'])}")
    print(f"[GAIA] Text-only: {sum(1 for r in records if not r['has_file'])}")
    print(f"[GAIA] Levels: {dict(pd.DataFrame(records)['level'].value_counts().sort_index())}")
    return pd.DataFrame(records)


# ==================== 默认 System Prompt ====================

VANILLA_SYSTEM_PROMPT = """You are a research assistant. Your task is to answer the question accurately and concisely.

## Output Format (MUST follow strictly)

Your response MUST contain the following tags:

<answer>Your exact, succinct final answer here</answer>
<confidence>Your confidence score between 0.0 and 1.0</confidence>

## Rules
1. The <answer> tag should contain ONLY the direct answer (name, number, date, etc.)
2. Do NOT include explanations or reasoning inside the <answer> tag
3. You may include your reasoning BEFORE the <answer> tag
4. Always provide both <answer> and <confidence> tags

## Example

Question: Who wrote "Pride and Prejudice"?

Jane Austen is the author of "Pride and Prejudice", published in 1813.

<answer>Jane Austen</answer>
<confidence>0.99</confidence>
""".strip()


# ==================== 答案提取 ====================

def extract_answer(response: str) -> str:
    """从模型回答中提取 <answer> 标签内的内容"""
    if not response:
        return ""
    
    # 尝试提取 <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 如果没有 <answer> 标签，返回最后一行非空内容作为答案
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1][:200]
    
    return response[:200]


# ==================== LLM 调用 ====================

async def call_llm(
    client: AsyncOpenAI,
    model: str,
    question: str,
    system_prompt: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """直接调用 LLM 回答问题"""
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.append({'role': 'user', 'content': question})
    
    for attempt in range(3):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            
            # DeepSeek-Reasoner 特殊处理
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content or ""
                if reasoning:
                    content = f"{reasoning}\n\n{content}"
            
            return content
        except Exception as e:
            if attempt == 2:
                print(f"[LLM ERROR] {e}")
                return f"Error: {e}"
            await asyncio.sleep(2 ** attempt)
    return ""


# ==================== LLM as a Judge ====================

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, contains all the essential information from [correct_answer], is equivalent despite minor wording/order differences (such as name order, inclusion or omission of middle names/initials, common honorifics, standard shortenings of first names, inclusion/omission of non-contradictory date parts like year, minor articles like "a"/"the", extra descriptive context, non-essential descriptive prefixes/suffixes such as "Restaurant", "Inc.", "Ltd.", or sports suffixes like "FC", "CF", "SC", inclusion/omission of subtitles in titles, minor spacing/punctuation differences — including presence/absence of quotation marks, interchangeable punctuation such as ":" / "-" / "–", case-only differences, or presence/absence of diacritics), or is within a small margin of error for numerical problems. Answer 'no' only if the extracted answer is factually incorrect, missing essential identifying information, or contradicts the [correct_answer].

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_judge_response(judge_response: str) -> dict:
    """解析评判响应"""
    result = {"correct": None, "parse_error": False}
    if not judge_response:
        result["parse_error"] = True
        return result
    
    # 提取 correct 字段
    correct_match = re.search(r"correct:\s*(yes|no)", judge_response, re.IGNORECASE)
    if not correct_match:
        correct_match = re.search(r"\*\*correct\*\*:\s*(yes|no)", judge_response, re.IGNORECASE)
    if correct_match:
        result["correct"] = correct_match.group(1).lower() == "yes"
    else:
        result["parse_error"] = True
    
    return result


async def judge(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    correct_answer: str,
    full_response: str,
    instance_id: str = "",
) -> tuple:
    """
    使用 LLM 评判答案是否正确
    
    Returns:
        (score, extracted_answer): 评分和提取的答案
    """
    # 从完整回答中提取答案
    extracted_answer = extract_answer(full_response)
    
    # 简单精确匹配
    if correct_answer.strip().lower() == extracted_answer.strip().lower():
        score = 1
        print(f"\n[Judged] score={score} (exact match)")
        print(f"  ID: {instance_id}")
        print(f"  Label: {correct_answer}")
        print(f"  Model: {extracted_answer}")
        return score, extracted_answer
    
    if not extracted_answer.strip():
        score = 0
        print(f"\n[Judged] score={score} (empty response)")
        print(f"  ID: {instance_id}")
        print(f"  Label: {correct_answer}")
        print(f"  Model: (empty)")
        return score, extracted_answer
    
    prompt = GRADER_TEMPLATE.format(
        question=question,
        response=full_response,  # 使用完整回答供 Judge 分析
        correct_answer=correct_answer
    )
    
    for _ in range(3):
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1024,
                temperature=0.1,
            )
            judge_text = response.choices[0].message.content or ""
            result = parse_judge_response(judge_text)
            if not result['parse_error']:
                score = int(result['correct'])
                print(f"\n[Judged] score={score}")
                print(f"  ID: {instance_id}")
                print(f"  Label: {correct_answer}")
                print(f"  Model: {extracted_answer}")
                return score, extracted_answer
        except Exception as e:
            await asyncio.sleep(1)
    
    # 重试失败
    print(f"\n[Judged] score=0 (judge failed)")
    print(f"  ID: {instance_id}")
    print(f"  Label: {correct_answer}")
    print(f"  Model: {extracted_answer}")
    return 0, extracted_answer


# ==================== 评测单个样本 ====================

async def eval_one(
    row: Dict,
    client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    model: str,
    judge_model: str,
    system_prompt: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Dict:
    """评测单个样本"""
    instance_id = row['instance_id']
    question = row['question']
    correct_answer = row['answer']
    
    # 使用默认 system prompt（如果未指定）
    prompt_to_use = system_prompt if system_prompt else VANILLA_SYSTEM_PROMPT
    
    start_time = time.time()
    
    try:
        # 1. 调用 LLM 回答问题
        full_response = await call_llm(client, model, question, prompt_to_use, max_tokens, temperature)
        
        # 2. 使用 LLM Judge 评分（使用独立的 judge_client）
        score, extracted_answer = await judge(judge_client, judge_model, question, correct_answer, full_response, instance_id)
        
        elapsed = time.time() - start_time
        
        return {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'topic': row.get('topic', ''),
            'level': row.get('level', 0),
            'score': score,
            'status': 'success',
            'prediction': extracted_answer,  # 提取的答案
            'full_response': full_response[:1000],  # 完整回答（截断）
            'ground_truth': correct_answer,
            'elapsed': elapsed,
        }
    except Exception as e:
        return {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'topic': row.get('topic', ''),
            'level': row.get('level', 0),
            'score': 0,
            'status': 'failed',
            'error': str(e),
            'prediction': '',
            'full_response': '',
            'ground_truth': correct_answer,
            'elapsed': 0,
        }


async def worker(
    worker_id: int,
    rows: List[Dict],
    client: AsyncOpenAI,
    judge_client: AsyncOpenAI,
    model: str,
    judge_model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    pbar,
    shared_scores: List,
) -> List[Dict]:
    """工作线程"""
    results = []
    for row in rows:
        result = await eval_one(row, client, judge_client, model, judge_model, system_prompt, max_tokens, temperature)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        pbar.set_postfix({
            'avg': f"{avg_score:.3f}",
            'id': result['instance_id'][:15],
            's': result['score'],
        })
        pbar.update(1)
    return results


# ==================== 主函数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Vanilla LLM Evaluation (No Agent Framework)')
    
    # 数据相关
    parser.add_argument('--dataset', default='browsecomp', 
                        choices=['browsecomp', 'browsecomp_openai', 'gaia'],
                        help='Dataset to evaluate')
    parser.add_argument('--data_path', default='/data1/test/xq/dataset/bc_test.parquet',
                        help='Path to data file (parquet)')
    parser.add_argument('--gaia_root', default='/data1/test/xq/dataset/GAIA',
                        help='GAIA dataset root directory')
    parser.add_argument('--cache_path', default=None,
                        help='Cache path for BrowseComp OpenAI dataset')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N items for testing')
    
    # 模型相关
    parser.add_argument('--model', default='Qwen3-32B',
                        help='Model name for answering questions')
    parser.add_argument('--judge_model', default='deepseek-reasoner',
                        help='Model for judging (default: same as --model)')
    parser.add_argument('--base_url', default=None,
                        help='API base URL (or set OPENAI_BASE_URL env)')
    parser.add_argument('--api_key', default=None,
                        help='API key (or set OPENAI_API_KEY env)')
    
    # Judge 模型独立配置
    parser.add_argument('--judge_base_url', default=None,
                        help='API base URL for judge model (or set JUDGE_BASE_URL env)')
    parser.add_argument('--judge_api_key', default=None,
                        help='API key for judge model (or set JUDGE_API_KEY env)')
    
    # 生成参数
    parser.add_argument('--max_tokens', type=int, default=8192,
                        help='Max tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Temperature for generation')
    parser.add_argument('--system_prompt', default=None,
                        help='System prompt for the model')
    
    # 运行参数
    parser.add_argument('--num_workers', type=int, default=10,
                        help='Number of parallel workers')
    parser.add_argument('--output_dir', default='results/vanilla',
                        help='Output directory')
    
    # GAIA 特定参数
    parser.add_argument('--text_only', action='store_true',
                        help='Only evaluate text-only questions (no file attachments, GAIA only)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 回答模型的 API 配置
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    # 评分模型的 API 配置（可独立设置）
    judge_base_url = args.judge_base_url or os.getenv("JUDGE_BASE_URL") or base_url
    judge_api_key = args.judge_api_key or os.getenv("JUDGE_API_KEY") or api_key
    judge_model = args.judge_model or args.model
    
    # 创建两个独立的 client
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    judge_client = AsyncOpenAI(api_key=judge_api_key, base_url=judge_base_url)
    
    print(f"\n{'='*60}")
    print("Vanilla LLM Evaluation (No Agent Framework)")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Base URL: {base_url}")
    print(f"Judge Model: {judge_model}")
    print(f"Judge Base URL: {judge_base_url}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Workers: {args.num_workers}")
    if args.dataset == 'gaia':
        print(f"Text Only: {args.text_only}")
    print(f"{'='*60}\n")
    
    # 加载数据
    if args.dataset == 'browsecomp_openai':
        df = load_browsecomp_openai(args.cache_path)
    elif args.dataset == 'gaia':
        df = load_gaia_dataset(args.data_path, args.gaia_root)
        # 过滤 text-only 数据
        if args.text_only:
            df = df[df['has_file'] == False]
            print(f"[GAIA] Filtered to {len(df)} text-only items")
    else:  # browsecomp (parquet)
        df = load_browsecomp_parquet(args.data_path)
    
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"Sampled {len(df)} items")
    
    # 转为字典列表
    rows = df.to_dict('records')
    
    # 分配任务
    num_workers = min(args.num_workers, len(rows))
    chunk_size = len(rows) // num_workers
    chunks = [rows[i*chunk_size:(i+1)*chunk_size if i < num_workers-1 else len(rows)]
              for i in range(num_workers)]
    
    # 运行评测
    async def run_all():
        shared_scores = []
        with tqdm(total=len(rows), desc="Evaluating (Vanilla)", unit="item") as pbar:
            tasks = [
                worker(i, chunks[i], client, judge_client, args.model, judge_model, 
                       args.system_prompt, args.max_tokens, args.temperature, pbar, shared_scores)
                for i in range(num_workers)
            ]
            return await asyncio.gather(*tasks)
    
    all_results = asyncio.run(run_all())
    results = [r for chunk_results in all_results for r in chunk_results]
    
    # 汇总统计
    print(f"\n{'='*60}")
    print("VANILLA LLM EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    avg_score = np.mean([r['score'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    avg_time = np.mean([r['elapsed'] for r in results])
    
    print(f"Overall Score: {avg_score:.4f}")
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"Avg Response Time: {avg_time:.2f}s")
    
    # 按数据源/主题/级别分组统计
    by_source = defaultdict(list)
    by_topic = defaultdict(list)
    by_level = defaultdict(list)
    
    for r in results:
        by_source[r['data_source']].append(r)
        if r.get('topic'):
            by_topic[r['topic']].append(r)
        if r.get('level'):
            by_level[r['level']].append(r)
    
    if len(by_source) > 1:
        print(f"\nBy Data Source:")
        for src in sorted(by_source.keys()):
            src_results = by_source[src]
            print(f"  {src}: {np.mean([r['score'] for r in src_results]):.4f} ({len(src_results)} items)")
    
    if by_level:
        print(f"\nBy Level:")
        for lvl in sorted(by_level.keys()):
            lvl_results = by_level[lvl]
            print(f"  Level {lvl}: {np.mean([r['score'] for r in lvl_results]):.4f} ({len(lvl_results)} items)")
    
    if by_topic and len(by_topic) <= 20:
        print(f"\nBy Topic:")
        for topic in sorted(by_topic.keys()):
            topic_results = by_topic[topic]
            print(f"  {topic}: {np.mean([r['score'] for r in topic_results]):.4f} ({len(topic_results)} items)")
    
    # 保存结果
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_dir) / f"vanilla_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model,
            'judge_model': judge_model,
            'dataset': args.dataset,
            'framework': 'vanilla (no agent)',
            'total_items': len(results),
            'success_count': success_count,
        },
        'summary': {
            'overall_score': float(avg_score),
            'avg_response_time': float(avg_time),
            'by_source': {k: float(np.mean([r['score'] for r in v])) for k, v in by_source.items()},
            'by_level': {str(k): float(np.mean([r['score'] for r in v])) for k, v in by_level.items()} if by_level else {},
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

