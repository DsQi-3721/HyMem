#!/usr/bin/env python3
"""
BrowseMaster评测脚本

用于在BrowseComp数据集上评测BrowseMaster的实现。
包含完整的日志记录功能，记录Planner-Executor交互过程。
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
# 使用BrowseMaster Agent (legacy --workflow bm_search)
from agents.bm_agent import process_item as bm_process_item
# Framework agent entrypoints (--framework hymem|react|resum|thread|amem)
from agents.hymem_agent import process_item as hymem_process_item
from agents.fold_agent import process_item as react_process_item
from agents.resum_agent import process_item as resum_process_item
from agents.thread_agent import process_item as thread_process_item
# A-MEM is lazy-imported inside eval_one_framework() to avoid mandatory
# chromadb / sentence-transformers install for users who don't run --framework amem.
from agents.utils import CallAPI, TaskContext, print_distractor_stats
from verl import DataProto
from scripts.eval_logger import HyMemLogger
from agents.hymem_logger import HyMemEventLog, HyMemLogFormatter, HyMemEvalLogger
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
        转换后的 DataFrame，格式与 BrowseComp-ZH 兼容
    """
    url = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
    
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
                    # 保留原始加密数据用于验证
                    'canary': canary,
                },
                'ability': 'OnlineSearch_BrowseComp',
                'data_source': 'browsecomp_openai',
                'reward_model': {'type': 'llm_judge', 'use_chinese': False}
            }
            records.append(record)
        except Exception as e:
            print(f"[BrowseComp-OpenAI] Failed to decrypt row {idx}: {e}")
            continue
    
    print(f"[BrowseComp-OpenAI] Loaded {len(records)} items")
    return pd.DataFrame(records)


# ==================== BrowseComp-ZH 数据加载 ====================

def load_browsecomp_zh_json(json_path: str) -> pd.DataFrame:
    """
    加载 BrowseComp-ZH JSON 数据并转换为兼容格式
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        转换后的 DataFrame，包含 extra_info, ability, data_source, reward_model 字段
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for idx, item in enumerate(data):
        record = {
            'extra_info': {
                'instance_id': f"browsecomp_zh_{idx:04d}",
                'query': item['Question'],           # 问题
                'answer': item['Answer'],            # 标准答案
                'topic': item.get('Topic', ''),      # 类别
                'problem_statement': item['Question'],
                'data_source': 'browsecomp_zh'
            },
            'ability': 'OnlineSearch_BrowseComp_ZH',  # 标记使用在线搜索
            'data_source': 'browsecomp_zh',
            'reward_model': {'type': 'llm_judge', 'use_chinese': True}
        }
        records.append(record)
    
    print(f"[BrowseComp-ZH] Loaded {len(records)} items from {json_path}")
    return pd.DataFrame(records)


# ==================== BrowseComp-Plus search-agent prompt ====================
#
# Why this exists:
#   BrowseComp-Plus parquet ships with `extra_info['prompt']` set to a
#   SWE-Bench OpenHands system prompt (execute_bash / str_replace_editor
#   tools). LocalSearch.get_data() in envs/local_search.py:440-451 reads
#   that prompt verbatim and gives it to the agent. HyMem ignores it
#   (builds its own prompt from hymem_prompts.py), so HyMem works. The
#   other 4 baselines (react / resum / thread / amem) USE it and end up
#   trying to call execute_bash — which envs/local_search.py:run_action()
#   doesn't recognize, so they loop until max_turn.
#
# The fix: rewrite extra_info['prompt'] before passing to the agent.
# The new prompt lists ONLY the three functions LocalSearch.run_action()
# actually understands: search, open_page, finish.

_BCPLUS_SEARCH_SYSTEM_PROMPT = """You are a research agent that answers a difficult question by searching a corpus of web documents. You must rely entirely on the tools below — do NOT use prior knowledge or guess.

You have access to exactly THREE tools. You MUST call them via the following XML format. Do NOT call any other function (e.g. `execute_bash`, `str_replace_editor`, `python`, `code_interpreter` — none of these exist here).

---- FUNCTION #1: search ----
Description: Search the corpus and return up to `topk` results (default 10). Each result has a `docid`, `url`, and a truncated text snippet.
Parameters:
  - query (string, required): the search query
  - topk (integer, optional, default 10): number of results to return

---- FUNCTION #2: open_page ----
Description: Fetch the full content of one document by its `docid` (preferred) or `url`. `docid` and `url` MUST come from a prior `search` result.
Parameters:
  - docid (string, optional): document id from a search result
  - url (string, optional): absolute url from a search result
  (Provide at least one of the two; docid is preferred when both are available.)

---- FUNCTION #3: finish ----
Description: End the task and return your final answer. Call this only when you have a definitive answer or you cannot make further progress.
Parameters:
  - answer (string, required): a concise, final answer
  - explanation (string, required): a brief evidence-grounded explanation. Cite supporting docids inline in square brackets, e.g. "...as shown in [serper_3_1202]."
  - confidence (string, optional): your confidence between 0% and 100%

---- CALL FORMAT ----
You call a function by writing exactly this XML, with no suffix:

<function=NAME>
<parameter=PARAM_1>value 1</parameter>
<parameter=PARAM_2>value 2 (can span multiple lines)</parameter>
</function>

Hard rules:
  1. Call ONLY one function per response. Reasoning is allowed BEFORE the call but NOT after.
  2. Use only `search`, `open_page`, `finish`. Calling any other function name returns an error.
  3. End the session with `finish` once you have an answer.

Strategy guidance:
  - Decompose the question into atomic factual claims to verify.
  - Use `search` to gather candidate documents; use `open_page` to confirm exact wording / numbers / dates.
  - Cross-check at least 2 sources for non-trivial claims before `finish`.
"""


def _build_search_prompt_list(question: str) -> list:
    """Construct the chat list LocalSearch.get_data expects:
    `[{role:system,...}, {role:user,...}]`."""
    return [
        {'role': 'system', 'content': _BCPLUS_SEARCH_SYSTEM_PROMPT},
        {'role': 'user', 'content': question or ''},
    ]


def _rewrite_df_prompts(df):
    """Replace each row's ``extra_info['prompt']`` with the search-agent
    chat list. Uses ``df.at`` to ensure the mutation persists back to
    the DataFrame (`iterrows()` returns a copy, so plain in-place dict
    mutation on the yielded row would not reach worker()).

    Also handles the case where parquet deserializes ``prompt`` to a
    numpy ndarray instead of a list (real bcplus parquet does this).
    """
    n_rewritten = 0
    seen_old_prompt = 0
    failed = 0
    for idx in df.index:
        ei_raw = df.at[idx, 'extra_info']
        # Accept dict OR dict-like (numpy struct, frozen dict).
        if isinstance(ei_raw, dict):
            ei = dict(ei_raw)
        elif hasattr(ei_raw, 'keys') and hasattr(ei_raw, '__getitem__'):
            try:
                ei = {k: ei_raw[k] for k in ei_raw.keys()}
            except Exception:
                failed += 1
                continue
        else:
            failed += 1
            continue
        # Detect SWE-Bench style prompt (ndarray OR list, both supported).
        old_prompt = ei.get('prompt')
        if old_prompt is not None:
            try:
                p0 = old_prompt[0]
                sys0 = p0.get('content', '') if hasattr(p0, 'get') else ''
                if 'OpenHands' in sys0 or 'execute_bash' in sys0:
                    seen_old_prompt += 1
            except (IndexError, TypeError, KeyError):
                pass
        question = (
            ei.get('query') or ei.get('problem_statement')
            or ei.get('question') or ''
        )
        ei['prompt'] = _build_search_prompt_list(question)
        df.at[idx, 'extra_info'] = ei
        n_rewritten += 1
    print(f"[BC+] rewrote extra_info['prompt'] for {n_rewritten} rows "
          f"(detected {seen_old_prompt} SWE-Bench prompts pre-patch, "
          f"{failed} rows skipped due to unparseable extra_info)")

    # Sanity check — verify the patch actually persisted on row 0.
    if n_rewritten > 0:
        first_ei = df.at[df.index[0], 'extra_info']
        first_prompt = first_ei.get('prompt') if isinstance(first_ei, dict) else None
        if first_prompt and len(first_prompt) > 0:
            sys_content = first_prompt[0].get('content', '') if hasattr(first_prompt[0], 'get') else ''
            if 'research agent' in sys_content and 'OpenHands' not in sys_content[:500]:
                print(f"[BC+]   ✓ verified: row 0 sys-prompt starts with "
                      f"'{sys_content[:60]}...' (len={len(sys_content)})")
            else:
                print(f"[BC+]   ✗ WARNING: row 0 sys-prompt did NOT change. "
                      f"First 120 chars: {sys_content[:120]!r}")
                print(f"[BC+]   This suggests the running code is stale "
                      f"(iterrows-version, not df.at-version). Pull latest.")
    return df


# ==================== 日志格式化工具 ====================

class BMLogger:
    """
    BrowseMaster执行日志记录器
    
    记录完整的Planner-Executor交互过程，包括：
    - 问题描述
    - Planner的推理和决策
    - Executor的搜索任务和执行结果
    - 工具调用详情
    - 置信度和重规划过程
    - 最终答案和评分
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
    
    def format_separator(self, char: str = "=", length: int = 80) -> str:
        return char * length
    
    def format_header(self, title: str, level: int = 1) -> str:
        """格式化标题"""
        if level == 1:
            return f"\n{'#' * 80}\n# {title}\n{'#' * 80}\n"
        elif level == 2:
            return f"\n{'=' * 60}\n== {title}\n{'=' * 60}\n"
        else:
            return f"\n{'-' * 40}\n-- {title}\n{'-' * 40}\n"
    
    def format_planner_turn(self, turn: int, content: str) -> str:
        """格式化Planner回合"""
        lines = [
            self.format_header(f"PLANNER TURN {turn}", level=2),
            "【Planner 输出】:",
            content[:5000] + "..." if len(content) > 5000 else content,
            ""
        ]
        return "\n".join(lines)
    
    def format_executor_turn(self, turn: int, task: str, result: str) -> str:
        """格式化Executor回合"""
        lines = [
            self.format_header(f"EXECUTOR TURN {turn}", level=2),
            "【搜索任务】:",
            task,
            "",
            "【执行结果】:",
            result[:8000] + "..." if len(result) > 8000 else result,
            ""
        ]
        return "\n".join(lines)
    
    def format_replan_event(self, turn: int, previous_answer: str, confidence: float) -> str:
        """格式化重规划事件"""
        lines = [
            self.format_header(f"REPLAN EVENT (Turn {turn})", level=3),
            f"【置信度不足触发重规划】",
            f"  - 当前置信度: {confidence:.2f}",
            f"  - 临时答案: {previous_answer[:500]}..." if len(previous_answer) > 500 else f"  - 临时答案: {previous_answer}",
            ""
        ]
        return "\n".join(lines)
    
    def format_task_log(self, instance_id: str, problem: str, history: List[Dict], 
                        answer: str, confidence: float, score: float,
                        stats: Dict[str, Any]) -> str:
        """
        格式化完整的任务执行日志
        
        Args:
            instance_id: 实例ID
            problem: 问题描述
            history: 执行历史
            answer: 最终答案
            confidence: 置信度
            score: 评分
            stats: 统计信息
        """
        log_parts = []
        
        # 任务头部
        log_parts.append(self.format_header(f"TASK: {instance_id}", level=1))
        log_parts.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_parts.append("")
        
        # 问题描述
        log_parts.append(self.format_header("PROBLEM", level=2))
        log_parts.append(problem)
        log_parts.append("")
        
        # 执行过程
        log_parts.append(self.format_header("EXECUTION TRACE", level=1))
        
        if history:
            for entry in history:
                role = entry.get('role', '')
                turn = entry.get('turn', 0)
                
                if role == 'planner':
                    log_parts.append(self.format_planner_turn(turn, entry.get('content', '')))
                elif role == 'executor':
                    log_parts.append(self.format_executor_turn(
                        turn, 
                        entry.get('task', ''), 
                        entry.get('result', '')
                    ))
                elif role == 'replan':
                    log_parts.append(self.format_replan_event(
                        turn,
                        entry.get('previous_answer', ''),
                        entry.get('confidence', 0.0)
                    ))
        else:
            log_parts.append("【无执行历史记录】")
        
        # 最终结果
        log_parts.append(self.format_header("FINAL RESULT", level=1))
        log_parts.append(f"【最终答案】:")
        log_parts.append(answer if answer else "(无答案)")
        log_parts.append("")
        log_parts.append(f"【置信度】: {confidence:.2f}")
        log_parts.append(f"【评分】: {score}")
        log_parts.append("")
        
        # 统计信息
        log_parts.append(self.format_header("STATISTICS", level=2))
        log_parts.append(f"  - Planner轮次: {stats.get('planner_turns', 0)}")
        log_parts.append(f"  - Executor调用: {stats.get('executor_calls', 0)}")
        log_parts.append(f"  - 重规划次数: {stats.get('replan_count', 0)}")
        log_parts.append(f"  - 会话时间: {stats.get('session_time', 0):.2f}s")
        log_parts.append("")
        
        log_parts.append(self.format_separator("="))
        
        return "\n".join(log_parts)
    
    def add_log(self, log_entry: Dict[str, Any]):
        """添加日志条目"""
        self.logs.append(log_entry)
    
    def save_task_log(self, instance_id: str, log_content: str):
        """保存单个任务的详细日志"""
        # 清理instance_id中的非法字符
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in instance_id)
        log_file = self.log_dir / f"task_{safe_id}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(log_content)
    
    def save_all_logs(self, filename: str = None):
        """保存所有日志的汇总"""
        if filename is None:
            filename = f"bm_eval_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        log_file = self.log_dir / filename
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
        
        return log_file
    
    def save_readable_summary(self, filename: str = None):
        """保存可读的日志摘要"""
        if filename is None:
            filename = f"bm_eval_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        summary_file = self.log_dir / filename
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self.format_header("BrowseMaster Evaluation Summary", level=1))
            f.write(f"总任务数: {len(self.logs)}\n")
            f.write(f"成功任务: {sum(1 for l in self.logs if l.get('status') == 'success')}\n")
            f.write(f"平均分数: {np.mean([l.get('score', 0) for l in self.logs]):.4f}\n")
            f.write("\n")
            
            for log in self.logs:
                f.write(f"\n{'-'*60}\n")
                f.write(f"Instance: {log.get('instance_id', 'unknown')}\n")
                f.write(f"Score: {log.get('score', 0)}\n")
                f.write(f"Status: {log.get('status', 'unknown')}\n")
                f.write(f"Confidence: {log.get('confidence', 0):.2f}\n")
                f.write(f"Planner Turns: {log.get('planner_turns', 0)}\n")
                f.write(f"Executor Calls: {log.get('executor_calls', 0)}\n")
        
        return summary_file



# HyMemLogger 已移至 agents.hymem_logger (通过 eval_logger.py 向后兼容导入)


# ==================== 命令行参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate BrowseMaster/HyMem on BrowseComp benchmark')
    parser.add_argument('--data_path', default='/home/wangxinqi/datasets/bc_test.parquet',
                        help='Path to test data parquet file')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save evaluation results')
    parser.add_argument('--log_dir', default='logs/bm_eval',
                        help='Directory to save detailed execution logs')
    parser.add_argument('--prompt_length', type=int, default=16384,
                        help='Maximum prompt length in tokens')
    parser.add_argument('--response_length', type=int, default=32768,
                        help='Maximum response length in tokens')
    # NOTE: --workflow is the legacy flag (bm_search / hymem only).
    # --framework supersedes it and supports all 5 baselines aligned with
    # eval_gaia.py. If both are given, --framework takes precedence.
    parser.add_argument('--workflow', default=None,
                        choices=['bm_search', 'hymem'],
                        help='[DEPRECATED] Use --framework instead. '
                             'Kept for backward compat with old scripts.')
    parser.add_argument('--framework', default='hymem',
                        choices=['hymem', 'react', 'resum', 'thread', 'amem'],
                        help='Evaluation framework, aligned with eval_gaia.py. '
                             'Default: hymem. For ReAct + summary mode: '
                             '--framework react --enable_summary.')
    parser.add_argument('--ablation', default='full',
                        choices=['full', 'no_isolated_reasoning',
                                 'no_memory_management'],
                        help='HyMem ablation variant: full (default) | '
                             'no_isolated_reasoning (disable Branch) | '
                             'no_memory_management (disable fold + structured memory)')
    # Resume support — skip tasks whose trace JSON already exists.
    # Mirrors eval_gaia.py:598-602.
    parser.add_argument('--resume', action='store_true',
                        help='Skip tasks that already have a {id}_trace.json in '
                             '--log_dir. Lets you restart an interrupted full '
                             'browsecomp-plus run without redoing work.')
    parser.add_argument('--no_resume', action='store_true',
                        help='Force re-running every task (overrides --resume).')
    # Sampling — for smoke tests / pilot runs.
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N items for testing.')
    parser.add_argument('--max_turn', type=int, default=40,
                        help='Maximum turns')
    parser.add_argument('--val_max_turn', type=int, default=60,
                        help='Maximum turns during validation')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum sessions')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum sessions during validation')
    parser.add_argument('--model_name', default='deepseek-reasoner',
                        help='Model name for API')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='Number of parallel evaluation workers')
    parser.add_argument('--local_search_url', default='http://localhost:8021',
                        help='URL of the local search server')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode')
    parser.add_argument('--save_task_logs', action='store_true', default=True,
                        help='Save individual task logs')
    parser.add_argument('--tokenizer_path', default='/data/models/Qwen3-8B',
                        help='Path to tokenizer model')
    # BrowseMaster特定参数
    parser.add_argument('--bm_max_planner_turns', type=int, default=10,
                        help='Max planner turns for BrowseMaster')
    parser.add_argument('--bm_max_executor_turns', type=int, default=5,
                        help='Max executor turns for BrowseMaster')
    parser.add_argument('--bm_search_topk', type=int, default=10,
                        help='Top-k results per search')
    parser.add_argument('--bm_max_keywords', type=int, default=8,
                        help='Max keywords for batch search')
    # HyMem特定参数
    parser.add_argument('--hymem_max_planner_turns', type=int, default=12,
                        help='Max planner turns for HyMem')
    parser.add_argument('--hymem_max_executor_turns', type=int, default=10,
                        help='Max executor turns for HyMem')
    parser.add_argument('--hymem_max_branch_turns', type=int, default=3,
                        help='Max turns per branch for HyMem (paper Sec A.2: max_iso_reasoning_turn=3)')
    parser.add_argument('--hymem_max_branches', type=int, default=5,
                        help='Max branches per session for HyMem')
    parser.add_argument('--hymem_max_folds', type=int, default=5,
                        help='Max fold operations for HyMem')
    parser.add_argument('--hymem_search_topk', type=int, default=5,
                        help='Top-k results per search for HyMem')
    parser.add_argument('--hymem_aux_model', default=None,
                        help='Auxiliary model for memory compression')
    # BrowseComp-ZH 相关参数
    parser.add_argument('--data_format', default='parquet',
                        choices=['parquet', 'json', 'browsecomp_openai'],
                        help='Data format: parquet, json (BrowseComp-ZH), or browsecomp_openai (OpenAI BrowseComp)')
    parser.add_argument('--browsecomp_cache', default=None,
                        help='Cache path for BrowseComp OpenAI dataset (optional)')
    parser.add_argument('--search_mode', default='local',
                        choices=['local', 'online', 'auto'],
                        help='Search mode: local (LocalSearch), online (OnlineSearch), or auto')
    parser.add_argument('--use_chinese_judge', action='store_true',
                        help='Use Chinese judge template for BrowseComp-ZH evaluation')
    return parser.parse_args()


# ==================== 评测函数 ====================

async def eval_one(row, config, tokenizer, model_name, logger: BMLogger, save_task_logs: bool):
    """
    评测单个实例
    
    Args:
        row: 数据行
        config: 配置
        tokenizer: 分词器
        model_name: 模型名称
        logger: 日志记录器
        save_task_logs: 是否保存单个任务日志
    
    Returns:
        评测结果字典
    """
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
        output = await bm_process_item(item, context, CallAPI)
        
        # 提取结果
        extra_data = output.non_tensor_batch.get('extra_data', [{}])[0] if output else {}
        stats = extra_data.get('stats', {})
        
        score = stats.get('score', 0)
        confidence = stats.get('confidence', 0.0)
        history = stats.get('bm_history', [])
        problem = stats.get('bm_problem', '')
        answer = stats.get('bm_answer', '')
        tool_calls_total = stats.get('tool_calls_total', 0)
        tool_calls_by_tool = stats.get('tool_calls_by_tool', {})
        
        # 格式化并保存详细日志
        if save_task_logs and history:
            log_content = logger.format_task_log(
                instance_id=instance_id,
                problem=problem,
                history=history,
                answer=answer,
                confidence=confidence,
                score=score,
                stats=stats
            )
            logger.save_task_log(instance_id, log_content)
        
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'replan_count': stats.get('replan_count', 0),
            'tool_calls_total': tool_calls_total,
            'tool_calls_by_tool': tool_calls_by_tool,
            'session_time': stats.get('session_time', 0),
            'answer': answer[:500] if answer else '',
            'problem': problem[:200] if problem else ''
        }
        
    except Exception as e:
        print(f"[ERROR] Task {instance_id} failed: {e}")
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
            'replan_count': 0,
            'tool_calls_total': 0,
            'tool_calls_by_tool': {},
            'session_time': 0,
            'answer': '',
            'problem': ''
        }
    
    # 添加到日志
    logger.add_log(result)
    
    return result


# ==================== Unified framework dispatcher ====================
# Mirrors eval_gaia.py:eval_one_gaia — one function handles all 5
# frameworks (hymem/react/resum/thread/amem). The legacy eval_one()
# above (BrowseMaster, --workflow bm_search) is kept untouched for
# backward compat.

import re as _re_eval

def _extract_answer_multi_pattern(history: List[Dict]) -> str:
    """Multi-pattern answer extraction, mirrors eval_gaia.py:274-304.

    Tries (in order):
      1. <answer>...</answer>          (HyMem)
      2. finish(VALUE)                  (Thread)
      3. <parameter=answer>VALUE</parameter>  (ReAct/env XML)
      4. Last short assistant message   (fallback)
    """
    answer = ''
    for entry in reversed(history):
        content = entry.get('content', '') or ''
        m = _re_eval.search(r'<answer>(.*?)</answer>', content, _re_eval.DOTALL)
        if m:
            return m.group(1).strip()
        m = _re_eval.search(r'finish\s*\(\s*(.+?)\s*\)\s*$', content,
                            _re_eval.MULTILINE | _re_eval.DOTALL)
        if m:
            return m.group(1).strip().strip('"\'')
        m = _re_eval.search(r'<parameter=answer>(.*?)</parameter>',
                            content, _re_eval.DOTALL)
        if m:
            return m.group(1).strip()
    # Fallback: last assistant message (short enough to be an answer)
    for entry in reversed(history):
        if entry.get('role') == 'assistant':
            c = (entry.get('content', '') or '').strip()
            if c and len(c) < 800:
                return c
    return ''


async def eval_one_framework(framework: str, row, config, tokenizer, model_name,
                              logger: HyMemLogger, save_task_logs: bool):
    """Unified per-task evaluator for {hymem, react, resum, thread, amem}.

    Schema-aligned with eval_gaia.py:eval_one_gaia so result JSONs are
    cross-comparable. Saves a per-task trace JSON to enable --resume.
    """
    instance_id = row['extra_info'].get('instance_id', 'unknown')

    context = TaskContext(
        config=config,
        global_step=0,
        server_host=model_name,
        server_port=0,
        is_train=False,
        tokenizer=tokenizer,
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
        'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn,
    }

    extra_info = row.get('extra_info', {})
    problem = extra_info.get('query', '') or extra_info.get('problem_statement', '')
    ground_truth = extra_info.get('answer', '')

    try:
        # Dispatch on framework (lazy import for amem).
        if framework == 'hymem':
            output = await hymem_process_item(item, context, CallAPI)
        elif framework == 'resum':
            output = await resum_process_item(item, context, CallAPI)
        elif framework == 'thread':
            output = await thread_process_item(item, context, CallAPI)
        elif framework == 'amem':
            from agents.amem_agent import process_item as amem_process_item
            output = await amem_process_item(item, context, CallAPI)
        else:  # 'react'  (fold_agent — covers react + summary mode via --enable_summary)
            output = await react_process_item(item, context, CallAPI)

        extra_data = output.non_tensor_batch.get('extra_data', [{}])[0] if output else {}
        stats = extra_data.get('stats', {})

        # History extraction — schema differs per framework (matches eval_gaia.py:262-270)
        if framework == 'hymem':
            history = stats.get('hymem_history', [])
        elif framework in ('thread', 'amem', 'resum', 'react'):
            history = stats.get('messages', []) or stats.get('history', [])
        else:
            history = stats.get('messages', []) or stats.get('history', []) \
                      or stats.get('hymem_history', [])

        confidence = stats.get('confidence', 0.0)
        # Prefer agent-exposed final_answer (Thread writes this since its
        # chat history is internal to _thread() and not in agent.messages()).
        # Fall back to multi-pattern extraction from history for others.
        answer = (stats.get('final_answer') or '').strip()
        if not answer:
            answer = _extract_answer_multi_pattern(history)
        score = stats.get('get_final_score', stats.get('score', 0))

        # Per-task human-readable + machine-readable logs
        if save_task_logs:
            event_log_data = stats.get('event_log')
            if event_log_data and framework == 'hymem':
                event_log_obj = HyMemEventLog()
                event_log_obj.events = event_log_data.get('events', [])
                event_log_obj.token_usage = __import__(
                    'agents.hymem_logger', fromlist=['TokenUsage']).TokenUsage()
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
                log_content = (
                    f"# {framework.upper()} Task: {instance_id}\n"
                    f"Problem: {problem[:300]}\n"
                    f"Answer: {answer}\n"
                    f"Ground Truth: {ground_truth}\n"
                    f"Score: {score}\n"
                    f"Confidence: {confidence}\n"
                )
            logger.save_task_log(instance_id, log_content)

        # ---- Result dict — schema aligned with eval_gaia.py:348-402 ----
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'prediction': answer[:500] if answer else '',
            'ground_truth': ground_truth[:500] if ground_truth else '',
            'problem': problem[:200] if problem else '',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'branch_success': stats.get('branch_success', 0),
            'fold_count': stats.get('fold_count', 0),
            'replan_count': stats.get('replan_count', 0),
            # Step A3 — unified tool-call schema
            'tool_calls_atomic': stats.get('tool_calls_atomic',
                                            stats.get('tool_calls_total', 0)),
            'agent_turns': stats.get('agent_turns', 0),
            'tool_calls_by_source': stats.get('tool_calls_by_source', {}),
            'framework': stats.get('framework', framework),
            'tool_calls_total': stats.get('tool_calls_total', 0),
            'tool_calls_by_tool': stats.get('tool_calls_by_tool', {}),
            # Step A1/A2/B1/B2
            'routing_summary': stats.get('routing_summary', {}),
            'distillation_stats': stats.get('distillation_stats', {}),
            'models_used': stats.get('models_used', {}),
            'latency_by_component': stats.get('latency_by_component', {}),
            'total_prompt_tokens': stats.get('total_prompt_tokens', 0),
            'total_completion_tokens': stats.get('total_completion_tokens', 0),
            'total_tokens': stats.get('total_tokens', 0),
            'token_usage_by_component': stats.get('token_usage_by_component', {}),
            'session_time': stats.get('session_time', 0),
            # Round 1-3 efficiency observation fields
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
            # Figure-1 dilution dump
            'planner_context_per_turn': stats.get('planner_context_per_turn', []),
            'context_size_history': stats.get('context_size_history', []),
        }

        # Save FULL per-task trace JSON (enables --resume + offline analysis)
        if save_task_logs:
            full_trace = {
                'instance_id': instance_id,
                'problem': problem,
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
                'hymem_history': stats.get('hymem_history', []) if framework == 'hymem'
                                   else stats.get('history', []),
                'messages': stats.get('messages', []),
            }
            logger.save_task_trace(instance_id, full_trace, prefix='bcplus')

    except Exception as e:
        print(f"[ERROR] {framework.upper()} Task {instance_id} failed: {e}")
        traceback.print_exc()
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': 0,
            'confidence': 0.0,
            'status': 'failed',
            'error': str(e),
            'prediction': '',
            'ground_truth': ground_truth[:500] if ground_truth else '',
            'problem': problem[:200] if problem else '',
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
            'parse_cache_hits': 0, 'parse_cache_misses': 0,
            'parse_cache_hit_rate': 0.0,
            'executor_early_exit_count': 0, 'executor_avg_inner_turns': 0.0,
            'distillation_invocations': 0, 'distillation_avg_aux_calls': 0.0,
            'distillation_chunk_concurrent_used': 0,
            'fold_invocations': 0, 'fold_avg_wall_seconds': 0.0,
            'fold_aux_parallel_seconds': 0.0,
            'parallel_dispatch_fanout_avg': 0.0,
            'parallel_dispatch_max_observed': 0,
            'planner_context_per_turn': [],
            'context_size_history': [],
        }

    logger.add_log(result)
    return result


async def eval_one_hymem(row, config, tokenizer, model_name, logger: HyMemLogger, save_task_logs: bool):
    """
    评测单个实例（HyMem框架）
    
    Args:
        row: 数据行
        config: 配置
        tokenizer: 分词器
        model_name: 模型名称
        logger: HyMem日志记录器
        save_task_logs: 是否保存单个任务日志
    
    Returns:
        评测结果字典
    """
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
        tool_calls_total = stats.get('tool_calls_total', 0)
        tool_calls_by_tool = stats.get('tool_calls_by_tool', {})
        
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
                event_log_obj.token_usage = __import__('agents.hymem_logger', fromlist=['TokenUsage']).TokenUsage()
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
            if event_log_data:
                logger.save_task_json(instance_id, event_log_data)
        
        result = {
            'instance_id': instance_id,
            'data_source': row.get('data_source', 'unknown'),
            'score': score,
            'confidence': confidence,
            'status': 'success',
            'planner_turns': stats.get('planner_turns', 0),
            'executor_calls': stats.get('executor_calls', 0),
            'branch_calls': stats.get('branch_calls', 0),
            'branch_success': stats.get('branch_success', 0),
            'fold_count': stats.get('fold_count', 0),
            'replan_count': stats.get('replan_count', 0),
            # Step A3 — unified tool-call schema (use these for cross-framework
            # comparison). Old tool_calls_total kept as alias.
            'tool_calls_atomic': stats.get('tool_calls_atomic', tool_calls_total),
            'agent_turns': stats.get('agent_turns', 0),
            'tool_calls_by_source': stats.get('tool_calls_by_source', {}),
            'framework': stats.get('framework', 'hymem'),
            'tool_calls_total': tool_calls_total,
            'tool_calls_by_tool': tool_calls_by_tool,
            # Step A1 — routing decisions
            'routing_summary': stats.get('routing_summary', {}),
            # Step A2 — distillation pipeline aggregate
            'distillation_stats': stats.get('distillation_stats', {}),
            # Step B1 — model disclosure
            'models_used': stats.get('models_used', {}),
            # Step B2 — per-component wall-clock latency
            'latency_by_component': stats.get('latency_by_component', {}),
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


async def worker(worker_id, rows, args, pbar, shared_scores, logger,
                 workflow: str = None, framework: str = 'hymem'):
    """工作线程

    Dispatch precedence:
      1. workflow='bm_search'  → legacy BrowseMaster eval_one()
      2. workflow='hymem'      → legacy eval_one_hymem() (backward compat)
      3. otherwise             → unified eval_one_framework(framework, ...)
    """
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # Map --framework → workflow string that agents/prompts.py:create_chat
    # actually understands. Unrecognized values fall to the OpenHands /
    # SWE-Bench prompt branch (else) — which gave us "execute_bash not
    # supported" loops on bcplus. The mapping below routes every non-HyMem
    # baseline to the search-agent prompt.
    if workflow:  # legacy --workflow bm_search / hymem path
        resolved_workflow = workflow
    elif framework == 'hymem':
        resolved_workflow = 'hymem'
    elif framework == 'react' and getattr(args, 'enable_summary', False):
        # Fold/ReAct with mid-trajectory summarization. create_chat still
        # treats it as 'search'; fold_agent reads enable_summary separately.
        resolved_workflow = 'search'
    elif framework in ('react', 'resum', 'amem'):
        resolved_workflow = 'search'
    elif framework == 'thread':
        # Thread has spawn capability — give it the search+branch prompt.
        resolved_workflow = 'search_branch'
    else:
        resolved_workflow = 'search'

    plugin_config = {
        # 'workflow' is what create_chat reads. Must be one of the
        # recognized values in agents/prompts.py:create_chat.
        'workflow': resolved_workflow,
        'framework': framework,
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
        # BrowseMaster legacy
        'bm_max_planner_turns': args.bm_max_planner_turns,
        'bm_max_executor_turns': args.bm_max_executor_turns,
        'bm_search_topk': args.bm_search_topk,
        'bm_max_keywords': args.bm_max_keywords,
        # BrowseComp-ZH
        'search_mode': args.search_mode,
        'use_chinese_judge': args.use_chinese_judge,
    }

    # HyMem-specific config — also propagated for amem/resum/thread/react
    # since they share the same hymem_* prompt-length / aux-model knobs.
    plugin_config.update({
        'hymem_max_planner_turns': args.hymem_max_planner_turns,
        'hymem_max_executor_turns': args.hymem_max_executor_turns,
        'hymem_max_branch_turns': args.hymem_max_branch_turns,
        'hymem_max_branches': args.hymem_max_branches,
        'hymem_max_folds': args.hymem_max_folds,
        'hymem_search_topk': args.hymem_search_topk,
        'hymem_ablation': args.ablation,
        'hymem_llm_model': args.model_name,
        'hymem_aux_model': args.hymem_aux_model,
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
    for row in rows:
        if workflow == 'bm_search':
            # Legacy BrowseMaster path (--workflow bm_search)
            result = await eval_one(row, config, tokenizer, args.model_name,
                                    logger, args.save_task_logs)
        else:
            # All 5 new framework names go through the unified evaluator
            result = await eval_one_framework(
                framework, row, config, tokenizer, args.model_name,
                logger, args.save_task_logs,
            )
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)

        pbar.set_postfix({
            'avg': f"{avg_score:.3f}",
            'id': result['instance_id'][:20],
            'score': f"{result['score']:.2f}",
            'conf': f"{result['confidence']:.2f}"
        })
        pbar.update(1)

    return results


def main():
    args = parse_args()

    # 设置环境变量
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url

    # Framework dispatch — --framework supersedes --workflow.
    # Legacy --workflow=bm_search routes to the BrowseMaster eval_one().
    legacy_workflow = args.workflow  # may be None / 'bm_search' / 'hymem'
    framework = args.framework        # 'hymem' / 'react' / 'resum' / 'thread' / 'amem'
    if legacy_workflow == 'bm_search':
        framework_name = 'BrowseMaster (legacy)'
        is_legacy_bm = True
    else:
        framework_name = framework.upper()
        is_legacy_bm = False
        if legacy_workflow and legacy_workflow != framework:
            print(f"[WARN] --workflow={legacy_workflow} ignored; "
                  f"--framework={framework} takes precedence.")

    # 日志目录与 logger — 所有 5 个 framework 共用 HyMemEvalLogger
    # （它现在有 save_task_trace），legacy bm_search 仍用 BMLogger。
    if is_legacy_bm:
        log_dir = args.log_dir
        logger = BMLogger(log_dir)
    else:
        log_dir = args.log_dir
        logger = HyMemLogger(log_dir)

    print(f"\n{'='*60}")
    print(f"BrowseComp-Plus Evaluation ({framework_name})")
    print(f"{'='*60}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Framework: {framework_name}")
    if framework == 'hymem' and not is_legacy_bm:
        print(f"Ablation: {args.ablation}")
    if legacy_workflow:
        print(f"Workflow (legacy): {legacy_workflow}")
    print(f"Search URL: {args.local_search_url}")
    print(f"Log Dir: {log_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Resume: {args.resume} (no_resume={args.no_resume})")
    print(f"Sample: {args.sample if args.sample else 'full set'}")
    if framework == 'hymem' and not is_legacy_bm:
        print(f"HyMem Config:")
        print(f"  - Max Planner Turns: {args.hymem_max_planner_turns}")
        print(f"  - Max Executor Turns: {args.hymem_max_executor_turns}")
        print(f"  - Max Branch Turns: {args.hymem_max_branch_turns}")
        print(f"  - Max Branches: {args.hymem_max_branches}")
        print(f"  - Max Folds: {args.hymem_max_folds}")
    print(f"Data Format: {args.data_format}")
    if args.data_format == 'browsecomp_openai':
        print(f"BrowseComp Cache: {args.browsecomp_cache or 'None (will download)'}")
    print(f"Search Mode: {args.search_mode}")
    print(f"Use Chinese Judge: {args.use_chinese_judge}")
    print(f"{'='*60}\n")

    # 设置搜索模式环境变量
    if args.search_mode == 'online':
        os.environ["USE_ONLINE_SEARCH"] = "true"
        os.environ["SEARCH_MODE"] = "online"
    elif args.search_mode == 'local':
        os.environ["SEARCH_MODE"] = "local"

    # 加载数据
    if args.data_format == 'browsecomp_openai':
        df = load_browsecomp_openai(cache_path=args.browsecomp_cache)
    elif args.data_format == 'json':
        df = load_browsecomp_zh_json(args.data_path)
    else:
        df = pd.read_parquet(args.data_path)

    # NOTE: Earlier attempt rewrote extra_info['prompt'] to a search-agent
    # prompt here, but turned out non-HyMem agents (fold/resum/amem/thread)
    # discard env.get_data()'s user_prompt and call agents/prompts.py
    # create_chat() instead. create_chat reads `config.plugin.workflow` to
    # pick a template — handled in worker() via the framework→workflow map.
    # _rewrite_df_prompts is kept available but disabled (would be no-op
    # for these baselines anyway).

    # Optional sub-sampling (smoke tests / pilot runs).
    # Previously hardcoded `df.sample(n=5, random_state=37)` here — removed
    # so full-set evals don't silently truncate. Use --sample N now.
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)
        print(f"[BC+] Sampled {len(df)} items for testing")

    print(f"[BC+] Loaded {len(df)} items total")

    # --- Resume support — read existing trace JSONs and skip those tasks ---
    # Mirrors eval_gaia.py:706-727. Only applies to the unified framework
    # path (--framework), not the legacy bm_search workflow.
    completed_results: List[Dict] = []
    if args.resume and not args.no_resume and not is_legacy_bm:
        existing_traces = list(Path(log_dir).glob("bcplus_*_trace.json"))
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
            df = df[df['extra_info'].apply(
                lambda x: x.get('instance_id') not in completed_ids)]
            print(f"[BC+] --resume: skipping {before - len(df)} already-completed "
                  f"tasks (loaded from {len(existing_traces)} traces in {log_dir})")
            if len(df) == 0:
                print("[BC+] All requested tasks already completed; jumping to summary.")

    if len(df) > 0:
        num_workers = max(1, min(args.num_workers, len(df)))
        chunk_size = max(1, len(df) // num_workers)
        chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < num_workers-1 else len(df)]
                  for i in range(num_workers)]

        async def run_all():
            shared_scores = []
            desc = f"Evaluating {framework_name}"
            with tqdm(total=len(df), desc=desc, unit="task") as pbar:
                tasks = [
                    worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))],
                           args, pbar, shared_scores, logger,
                           workflow=legacy_workflow, framework=framework)
                    for i in range(num_workers)
                ]
                return await asyncio.gather(*tasks)

        all_results = asyncio.run(run_all())
        new_results = [r for worker_results in all_results for r in worker_results]
    else:
        new_results = []

    # Combine resumed + newly-completed results into the aggregate summary
    results = completed_results + new_results
    if completed_results:
        print(f"[BC+] Combined {len(completed_results)} resumed + "
              f"{len(new_results)} new = {len(results)} total tasks for summary.")

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"{framework_name.upper()} EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    avg_score = np.mean([r['score'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    success_count = sum(1 for r in results if r['status'] == 'success')
    
    print(f"Overall Score: {avg_score:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Success Rate: {success_count}/{len(results)} ({100*success_count/len(results):.1f}%)")
    print(f"Average Planner Turns: {np.mean([r['planner_turns'] for r in results]):.2f}")
    print(f"Average Executor Calls: {np.mean([r['executor_calls'] for r in results]):.2f}")
    print(f"Average Tool Calls (Unified): {np.mean([r.get('tool_calls_total', 0) for r in results]):.2f}")
    
    # Branch / fold stats only meaningful for hymem / thread; print
    # if any task has them populated so other frameworks stay tidy.
    if any(r.get('branch_calls', 0) for r in results):
        print(f"Average Branch Calls: {np.mean([r.get('branch_calls', 0) for r in results]):.2f}")
        print(f"Average Branch Success: {np.mean([r.get('branch_success', 0) for r in results]):.2f}")
    if any(r.get('fold_count', 0) for r in results):
        print(f"Average Fold Count: {np.mean([r.get('fold_count', 0) for r in results]):.2f}")
    print(f"Average Replans: {np.mean([r.get('replan_count', 0) for r in results]):.2f}")

    # 按数据源统计
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        by_source[r['data_source']].append(r)

    print(f"\nBy Data Source:")
    for source in sorted(by_source.keys()):
        source_results = by_source[source]
        source_scores = [r['score'] for r in source_results]
        print(f"  {source}: avg={np.mean(source_scores):.4f}, count={len(source_scores)}")

    # 保存结果
    Path(args.output_dir).mkdir(exist_ok=True)

    # 保存详细结果JSON — prefix 体现 framework 便于多 baseline 共目录共存
    prefix = "bm" if is_legacy_bm else f"bcplus_{framework}"
    output_file = Path(args.output_dir) / f"{prefix}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary_by_source = {
        src: {
            'avg_score': float(np.mean([r['score'] for r in rs])),
            'avg_confidence': float(np.mean([r['confidence'] for r in rs])),
            'count': len(rs)
        }
        for src, rs in by_source.items()
    }
    
    summary_data = {
        'avg_score': float(avg_score),
        'avg_confidence': float(avg_confidence),
        'avg_planner_turns': float(np.mean([r['planner_turns'] for r in results])),
        'avg_executor_calls': float(np.mean([r['executor_calls'] for r in results])),
        'avg_tool_calls_total': float(np.mean([r.get('tool_calls_total', 0) for r in results])),
        'avg_replan_count': float(np.mean([r.get('replan_count', 0) for r in results])),
    }
    
    # Branch/fold stats — write them in regardless; non-applicable
    # baselines will report 0.0 which is informative.
    summary_data.update({
        'avg_branch_calls': float(np.mean([r.get('branch_calls', 0) for r in results])),
        'avg_branch_success': float(np.mean([r.get('branch_success', 0) for r in results])),
        'avg_fold_count': float(np.mean([r.get('fold_count', 0) for r in results])),
    })

    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model': args.model_name,
            'workflow': legacy_workflow,
            'framework': framework_name if is_legacy_bm else framework,
            'ablation': args.ablation if framework == 'hymem' and not is_legacy_bm else 'n/a',
            'total_items': len(results),
            'success_count': success_count,
            'cli_args': {k: v for k, v in vars(args).items()
                          if not k.startswith('_')},
        },
        'summary': summary_data,
        'by_source': summary_by_source,
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
    
    # 打印干扰注入统计（如果启用）
    print_distractor_stats()
    
    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
