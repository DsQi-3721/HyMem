"""
BrowseMaster工具函数实现

本模块实现BrowseMaster框架所需的六个核心搜索工具函数：
1. web_search - 网页搜索
2. web_parse - 网页解析
3. generate_keywords - 关键词生成
4. batch_search - 批量搜索
5. check_condition - 条件检查
6. batch_search_and_filter - 批量搜索与过滤

所有工具函数返回格式遵循BrowseMaster的executor_prompt规范。
"""

import os
import re
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI

# 复用FoldAgent的搜索客户端
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.local_search import AsyncSearchClient
from agents.distillation import run_distillation_pipeline, DistillationStats
from agents.utils import is_context_length_exceeded_error


# ==================== 配置 ====================

@dataclass
class BMToolsConfig:
    """BrowseMaster工具配置"""
    search_topk: int = 10  # 默认搜索返回数量
    max_keywords: int = 8  # 生成关键词的最大数量
    batch_search_topk: int = 5  # 批量搜索每个关键词返回数量
    max_content_length: int = 2000  # check_condition时内容截断长度
    llm_model: str = "deepseek-reasoner"  # LLM模型名称
    llm_timeout: float = 120.0  # LLM调用超时时间
    # ThreadPoolExecutor并行配置（BrowseMaster论文推荐）
    max_workers: int = 10  # 线程池最大工作线程数
    parallel_batch_size: int = 5  # 并行批次大小


# 全局线程池（用于CPU密集型操作）
_thread_pool: Optional[ThreadPoolExecutor] = None

def get_thread_pool(max_workers: int = 10) -> ThreadPoolExecutor:
    """获取或创建全局线程池"""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=max_workers)
    return _thread_pool


# ==================== LLM调用封装 ====================

async def call_llm(
    prompt: str, 
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: Optional[int] = None,
    timeout: float = 120.0
) -> str:
    """
    调用LLM生成响应 - 支持独立配置
    
    Args:
        prompt: 用户提示词
        model: 模型名称，优先级：参数 > BM_LLM_MODEL > HYMEM_EXECUTOR_MODEL > 默认值
        base_url: API 端点，优先级：参数 > BM_BASE_URL > HYMEM_EXECUTOR_BASE_URL > OPENAI_BASE_URL
        api_key: API 密钥，优先级：参数 > BM_API_KEY > HYMEM_EXECUTOR_API_KEY > OPENAI_API_KEY
        max_retries: 最大重试次数
        timeout: 超时时间
        
    Returns:
        LLM生成的响应文本
    """
    # 模型配置：默认跟随 Executor 配置
    if model is None:
        model = os.getenv("BM_LLM_MODEL") or os.getenv("HYMEM_EXECUTOR_MODEL", "Qwen3-32B")
    
    # API 配置：默认跟随 Executor 配置
    _base_url = base_url or os.getenv("BM_BASE_URL") or os.getenv("HYMEM_EXECUTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _api_key = api_key or os.getenv("BM_API_KEY") or os.getenv("HYMEM_EXECUTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    if max_retries is None:
        max_retries = max(1, int(os.getenv("BM_LLM_MAX_ATTEMPTS", "3")))
    timeout = float(os.getenv("BM_LLM_TIMEOUT", str(timeout)))

    client = AsyncOpenAI(
        api_key=_api_key,
        base_url=_base_url,
        timeout=timeout,
        max_retries=0,
    )
    
    messages = [{'role': 'user', 'content': prompt}]
    
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model, 
                messages=messages,
                temperature=0.7
            )
            message = resp.choices[0].message
            text = message.content or ""
            # DeepSeek-Reasoner: 如果content为空，尝试使用reasoning_content
            if not text and hasattr(message, 'reasoning_content') and message.reasoning_content:
                text = message.reasoning_content
            return text
        except Exception as e:
            # 检查是否是内容审核错误（不应重试）
            error_str = str(e)
            if is_context_length_exceeded_error(e):
                print(f"[BM_TOOLS] Context length exceeded, skipping retries: {error_str[:200]}")
                return f"Error: Context length exceeded - {error_str}"
            if "400" in error_str and ("Content Exists Risk" in error_str or "content_filter" in error_str.lower() or "sensitive" in error_str.lower()):
                print(f"[BM_TOOLS] Content moderation triggered, skipping: {error_str[:200]}")
                return f"Error: Content moderation - unable to process this request"
            
            if attempt == max_retries - 1:
                print(f"[BM_TOOLS] LLM call failed after {max_retries} attempts: {e}")
                return f"Error: {e}"
            # O4: exponential backoff with jitter (was: linear 1s/2s/3s).
            # Caps at 8s and adds [0, 0.5)s jitter to spread retries when
            # multiple parallel requests all hit the same 429 burst.
            import random as _rand
            delay = min(8.0, 2 ** attempt) + _rand.uniform(0.0, 0.5)
            await asyncio.sleep(delay)
    return ""


def parse_json_from_response(response: str) -> Optional[Dict]:
    """
    从LLM响应中解析JSON
    
    Args:
        response: LLM响应文本
        
    Returns:
        解析的JSON对象，失败返回None
    """
    # 尝试直接解析
    try:
        return json.loads(response)
    except:
        pass
    
    # 尝试提取JSON代码块
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # 尝试提取花括号内容
    brace_match = re.search(r'\{[\s\S]*\}', response)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except:
            pass
    
    # 尝试提取方括号内容(数组)
    bracket_match = re.search(r'\[[\s\S]*\]', response)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except:
            pass
    
    return None


# ==================== 工具函数实现 ====================

async def web_search(
    keywords: str,
    search_client: AsyncSearchClient,
    topk: int = 10
) -> Dict[str, Any]:
    """
    网页搜索工具
    
    使用LocalSearch的向量检索服务进行网页搜索。
    
    Args:
        keywords: 搜索关键词
        search_client: AsyncSearchClient实例
        topk: 返回结果数量
        
    Returns:
        {
            "tool_result": {
                "searchParameters": {"query": keywords, "topk": topk},
                "organic": [
                    {"title": str, "link": str, "snippet": str, "position": int},
                    ...
                ]
            }
        }
    """
    try:
        # 确保参数类型正确，避免422错误
        results = await search_client.search(str(keywords).strip(), k=int(topk))
        
        organic = []
        for idx, item in enumerate(results, 1):
            organic.append({
                "title": item.get('url', '').split('/')[-1] or 'Document',
                "link": item.get('url', ''),
                "snippet": item.get('text', '')[:500] if item.get('text') else '',
                "docid": item.get('docid', ''),
                "position": idx
            })
        
        return {
            "tool_result": {
                "searchParameters": {"query": keywords, "topk": topk},
                "organic": organic
            }
        }
    except Exception as e:
        print(f"[BM_TOOLS] web_search error: {e}")
        return {
            "tool_result": {
                "searchParameters": {"query": keywords, "topk": topk},
                "organic": [],
                "error": str(e)
            }
        }


async def web_parse(
    link: str,
    query: str,
    search_client: AsyncSearchClient,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    tool_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    网页解析工具 - 支持独立 LLM 配置

    获取网页内容并使用LLM根据查询提取相关信息。

    Args:
        link: 网页URL或docid
        query: 用户查询
        search_client: AsyncSearchClient实例
        llm_model: LLM模型名称（默认跟随 Executor 配置）
        llm_base_url: LLM API 端点（默认跟随 Executor 配置）
        llm_api_key: LLM API 密钥（默认跟随 Executor 配置）
        tool_manager: O1 — when given, look up + fill `_raw_page_cache` so
            that re-parsing the same URL with a different query skips the
            network fetch step.

    Returns:
        {
            "tool_result": {
                "content": str,  # LLM根据查询提取的内容
                "url": str,
                "docid": str
            }
        }
    """
    try:
        # === O1: raw-page cache for cross-query reuse on the same URL ===
        page_content = None
        page_url = link
        page_docid = ""
        cached_raw = None
        if tool_manager is not None and getattr(tool_manager, "_parse_cache_enabled", False):
            try:
                _ck = tool_manager._canonical_url(link)
                cached_raw = tool_manager._raw_page_cache.get(_ck)
                if cached_raw is not None:
                    tool_manager._raw_page_cache.move_to_end(_ck)
                    tool_manager._raw_page_hits += 1
                    page_content = cached_raw
                    page_url = link
            except Exception:
                cached_raw = None

        if page_content is None:
            # 判断是URL还是docid
            if link.startswith('http://') or link.startswith('https://'):
                pages = await search_client.open(url=link)
            else:
                pages = await search_client.open(docid=link)

            if not pages:
                return {
                    "tool_result": {
                        "content": "Failed to fetch page content.",
                        "url": link,
                        "docid": ""
                    }
                }

            page = pages[0]
            page_content = page.get('text', '')
            page_url = page.get('url', link)
            page_docid = page.get('docid', '')

            # Stash raw page for next call with different query
            if (tool_manager is not None
                    and getattr(tool_manager, "_parse_cache_enabled", False)
                    and page_content):
                try:
                    _ck = tool_manager._canonical_url(link)
                    tool_manager._raw_page_cache[_ck] = page_content
                    tool_manager._raw_page_cache.move_to_end(_ck)
                    while len(tool_manager._raw_page_cache) > tool_manager._parse_cache_max:
                        tool_manager._raw_page_cache.popitem(last=False)
                except Exception:
                    pass
        
        # 如果内容为空
        if not page_content.strip():
            return {
                "tool_result": {
                    "content": "Page content is empty.",
                    "url": page_url,
                    "docid": page_docid
                }
            }
        
        # Information Distillation (paper Sec 3.2.2): for long pages, run the
        # three-stage pipeline (rule extract + chunked LLM filter + task gate)
        # to avoid the old hard-truncation that dropped tail content.
        # Short pages bypass distillation to keep cost low.
        distillation_stats = None
        DISTILL_THRESHOLD = 4000
        HARD_CAP = 16000
        if len(page_content) > DISTILL_THRESHOLD:
            async def _checker(chunk: str, task: str):
                return await check_condition(
                    chunk, task,
                    llm_model=llm_model,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    max_content_length=3000,
                )

            async def _llm_caller(prompt: str) -> str:
                return await call_llm(
                    prompt, model=llm_model,
                    base_url=llm_base_url, api_key=llm_api_key,
                )

            page_content, distillation_stats = await run_distillation_pipeline(
                page_content[:HARD_CAP],  # outer safety cap on huge pages
                task=query,
                relevance_checker=_checker,
                llm_caller=_llm_caller,
                chunk_size=1500,
                overlap=200,
            )
        elif len(page_content) > 8000:
            # Legacy fallback path (shouldn't trigger given the 4000 threshold,
            # kept for explicit clarity).
            page_content = page_content[:8000] + "\n[Content truncated...]"

        # 使用LLM根据查询提取信息
        prompt = f"""Please analyze the following web page content and answer the user's query.
Extract and summarize the most relevant information to answer the query.
If the information is not found in the content, clearly state that.

Web Page Content:
{page_content}

User Query: {query}

Please provide a comprehensive answer based on the content above. Include specific details, facts, and evidence from the page."""

        llm_response = await call_llm(prompt, model=llm_model, base_url=llm_base_url, api_key=llm_api_key)

        result = {
            "tool_result": {
                "content": llm_response,
                "url": page_url,
                "docid": page_docid,
            }
        }
        if distillation_stats is not None:
            result["tool_result"]["distillation_stats"] = distillation_stats.to_dict()
        return result
    except Exception as e:
        print(f"[BM_TOOLS] web_parse error: {e}")
        return {
            "tool_result": {
                "content": f"Error parsing page: {str(e)}",
                "url": link,
                "docid": ""
            }
        }


async def generate_keywords(
    seed_keyword: str,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    max_keywords: int = 8
) -> Dict[str, Any]:
    """
    关键词生成工具 - 支持独立 LLM 配置
    
    基于种子关键词使用LLM生成多个搜索变体关键词。
    生成策略包括：
    1. 精确匹配变体
    2. 同义词替换
    3. 相关概念扩展
    4. 时间维度变体（如果包含时间描述）
    
    Args:
        seed_keyword: 种子关键词
        llm_model: LLM模型名称（默认跟随 Executor 配置）
        llm_base_url: LLM API 端点（默认跟随 Executor 配置）
        llm_api_key: LLM API 密钥（默认跟随 Executor 配置）
        max_keywords: 最大生成关键词数量
        
    Returns:
        {
            "tool_result": {
                "keywords": [str, str, ...],
                "seed": str
            }
        }
    """
    prompt = f"""Generate {max_keywords} diverse search query variations for the following seed keyword to maximize search coverage.

Seed Keyword: {seed_keyword}

Generate variations using these strategies:
1. Exact match variations (with quotes for key phrases)
2. Synonym replacements
3. Related concept expansions
4. Different phrasings and word orders
5. If the keyword contains time references (years, dates, periods), generate variations for specific time periods

Requirements:
- Each variation should be distinct and useful for web search
- Keep variations concise and search-engine friendly
- Include the original keyword as one of the variations
- Variations should help find information that the original keyword might miss

Return your response as a JSON object with a "keywords" array:
{{"keywords": ["keyword1", "keyword2", ...]}}

Only output the JSON, no other text."""

    try:
        response = await call_llm(prompt, model=llm_model, base_url=llm_base_url, api_key=llm_api_key)
        
        # 解析JSON响应
        parsed = parse_json_from_response(response)
        
        if parsed and isinstance(parsed, dict) and 'keywords' in parsed:
            keywords = parsed['keywords']
        elif parsed and isinstance(parsed, list):
            keywords = parsed
        else:
            # 如果解析失败，尝试简单提取
            keywords = [seed_keyword]
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip().strip('-').strip('*').strip('"').strip("'").strip()
                if line and len(line) > 2 and line not in keywords:
                    keywords.append(line)
        
        # 确保原始关键词在列表中
        if seed_keyword not in keywords:
            keywords.insert(0, seed_keyword)
        
        # 限制数量
        keywords = keywords[:max_keywords]
        
        return {
            "tool_result": {
                "keywords": keywords,
                "seed": seed_keyword
            }
        }
    except Exception as e:
        print(f"[BM_TOOLS] generate_keywords error: {e}")
        return {
            "tool_result": {
                "keywords": [seed_keyword],
                "seed": seed_keyword,
                "error": str(e)
            }
        }


async def batch_search(
    keywords: List[str],
    search_client: AsyncSearchClient,
    topk_per_keyword: int = 5,
    max_concurrent: int = 10
) -> Dict[str, Any]:
    """
    批量搜索工具（使用ThreadPoolExecutor并行优化）
    
    并行执行多个关键词的搜索并合并结果。
    使用Semaphore控制并发数，防止API限流。
    
    Args:
        keywords: 关键词列表
        search_client: AsyncSearchClient实例
        topk_per_keyword: 每个关键词返回的结果数量
        max_concurrent: 最大并发搜索数
        
    Returns:
        {
            "tool_result": {
                "organic": [...],
                "total_results": int,
                "keywords_searched": int
            }
        }
    """
    # 使用Semaphore控制并发数（BrowseMaster论文推荐的并行控制策略）
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def search_with_semaphore(kw: str):
        """带并发控制的搜索"""
        async with semaphore:
            try:
                # 确保k参数是整数类型，避免422错误
                return await search_client.search(str(kw).strip(), k=int(topk_per_keyword))
            except Exception as e:
                return e
    
    try:
        # 并行执行所有搜索（受Semaphore限制）
        tasks = [search_with_semaphore(kw) for kw in keywords]
        all_results = await asyncio.gather(*tasks)
        
        # 使用线程池进行CPU密集型的结果合并操作
        loop = asyncio.get_event_loop()
        thread_pool = get_thread_pool()
        
        def merge_results():
            """在线程池中执行结果合并（CPU密集型操作）"""
            seen_urls = set()
            organic = []
            position = 1
            
            for kw, results in zip(keywords, all_results):
                if isinstance(results, Exception):
                    print(f"[BM_TOOLS] batch_search error for keyword '{kw}': {results}")
                    continue
                
                for item in results:
                    url = item.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        organic.append({
                            "title": url.split('/')[-1] or 'Document',
                            "link": url,
                            "snippet": item.get('text', '')[:500] if item.get('text') else '',
                            "docid": item.get('docid', ''),
                            "position": position,
                            "source_keyword": kw
                        })
                        position += 1
            return organic
        
        # 在线程池中执行合并
        organic = await loop.run_in_executor(thread_pool, merge_results)
        
        return {
            "tool_result": {
                "organic": organic,
                "total_results": len(organic),
                "keywords_searched": len(keywords)
            }
        }
    except Exception as e:
        print(f"[BM_TOOLS] batch_search error: {e}")
        return {
            "tool_result": {
                "organic": [],
                "total_results": 0,
                "keywords_searched": len(keywords),
                "error": str(e)
            }
        }


async def check_condition(
    content: str,
    condition: str,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    max_content_length: int = 3000
) -> Dict[str, Any]:
    """
    条件检查工具 - 支持独立 LLM 配置

    使用LLM评估给定内容是否满足指定条件。

    Args:
        content: 待评估的内容
        condition: 需要满足的条件
        llm_model: LLM模型名称（默认跟随 Executor 配置）
        llm_base_url: LLM API 端点（默认跟随 Executor 配置）
        llm_api_key: LLM API 密钥（默认跟随 Executor 配置）
        max_content_length: 内容最大长度（提到 3000 以容纳 distillation Stage 2
            的 1500-char chunk + overlap，避免 chunk 在此处被再截断）

    Returns:
        {
            "tool_result": {
                "is_relevant": "yes" | "no" | "unknown",
                "reasoning": str  # 判断理由
            }
        }
    """
    # 截断过长的内容（保留 safety rail；分块过滤的上层已避免长内容直入此处）
    if len(content) > max_content_length:
        content = content[:max_content_length] + "... [truncated]"
    
    prompt = f"""Evaluate whether the following content satisfies the specified condition.

Content:
{content}

Condition to check:
{condition}

Instructions:
- Answer "yes" if the content clearly satisfies the condition
- Answer "no" if the content clearly does NOT satisfy the condition
- Answer "unknown" if you cannot determine from the given content

Respond in JSON format:
{{"is_relevant": "yes/no/unknown", "reasoning": "brief explanation"}}

Only output the JSON, no other text."""

    try:
        response = await call_llm(prompt, model=llm_model, base_url=llm_base_url, api_key=llm_api_key)
        
        # 解析响应
        parsed = parse_json_from_response(response)
        
        if parsed and isinstance(parsed, dict):
            is_relevant = parsed.get('is_relevant', 'unknown').lower()
            reasoning = parsed.get('reasoning', '')
        else:
            # 简单文本匹配
            response_lower = response.lower()
            if 'yes' in response_lower and 'no' not in response_lower[:20]:
                is_relevant = 'yes'
            elif 'no' in response_lower and 'yes' not in response_lower[:20]:
                is_relevant = 'no'
            else:
                is_relevant = 'unknown'
            reasoning = response
        
        # 规范化返回值
        if is_relevant not in ['yes', 'no', 'unknown']:
            is_relevant = 'unknown'
        
        return {
            "tool_result": {
                "is_relevant": is_relevant,
                "reasoning": reasoning
            }
        }
    except Exception as e:
        print(f"[BM_TOOLS] check_condition error: {e}")
        return {
            "tool_result": {
                "is_relevant": "unknown",
                "reasoning": f"Error: {str(e)}"
            }
        }


async def batch_search_and_filter(
    keyword: str,
    search_client: AsyncSearchClient,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    max_keywords: int = 6,
    topk_per_keyword: int = 5,
    max_concurrent_checks: int = 5
) -> Dict[str, Any]:
    """
    批量搜索与过滤工具 - 支持独立 LLM 配置
    
    这是一个集成工具，组合了以下步骤：
    1. 使用generate_keywords生成多个搜索关键词
    2. 使用batch_search执行批量搜索（并行）
    3. 使用check_condition过滤结果（分批并行，受Semaphore控制）
    
    Args:
        keyword: 搜索目标描述
        search_client: AsyncSearchClient实例
        llm_model: LLM模型名称（默认跟随 Executor 配置）
        llm_base_url: LLM API 端点（默认跟随 Executor 配置）
        llm_api_key: LLM API 密钥（默认跟随 Executor 配置）
        max_keywords: 生成的最大关键词数量
        topk_per_keyword: 每个关键词返回的结果数量
        max_concurrent_checks: 最大并发检查数（控制LLM API调用）
        
    Returns:
        {
            "tool_result": {
                "yes": [  # 满足条件的结果
                    {"title": str, "link": str, "snippet": str, ...},
                    ...
                ],
                "unknown": [  # 无法判断的结果
                    {"title": str, "link": str, "snippet": str, ...},
                    ...
                ],
                "no_count": int,  # 不满足条件的结果数量
                "keywords_used": [str, ...]
            }
        }
    """
    try:
        # Step 1: 生成关键词
        keywords_result = await generate_keywords(
            keyword, 
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            max_keywords=max_keywords
        )
        keywords = keywords_result["tool_result"]["keywords"]
        
        # Step 2: 批量搜索
        search_result = await batch_search(
            keywords, 
            search_client, 
            topk_per_keyword=topk_per_keyword
        )
        organic = search_result["tool_result"]["organic"]
        
        if not organic:
            return {
                "tool_result": {
                    "yes": [],
                    "unknown": [],
                    "no_count": 0,
                    "keywords_used": keywords
                }
            }
        
        # Step 3: 分批并行检查（使用Semaphore控制LLM API并发）
        check_semaphore = asyncio.Semaphore(max_concurrent_checks)
        
        async def check_with_control(item: Dict):
            """带并发控制的条件检查"""
            async with check_semaphore:
                content = f"Title: {item.get('title', '')}\nURL: {item.get('link', '')}\nSnippet: {item.get('snippet', '')}"
                try:
                    result = await check_condition(
                        content, keyword, 
                        llm_model=llm_model,
                        llm_base_url=llm_base_url,
                        llm_api_key=llm_api_key
                    )
                    return item, result
                except Exception as e:
                    return item, {"tool_result": {"is_relevant": "unknown", "reasoning": str(e)}}
        
        # 并行执行所有检查（受Semaphore限制）
        check_tasks = [check_with_control(item) for item in organic]
        check_results = await asyncio.gather(*check_tasks)
        
        # 使用线程池进行结果分类（CPU密集型操作）
        loop = asyncio.get_event_loop()
        thread_pool = get_thread_pool()
        
        def classify_results():
            yes_results = []
            unknown_results = []
            no_count = 0
            for item, check_result in check_results:
                relevance = check_result["tool_result"]["is_relevant"]
                if relevance == "yes":
                    item["relevance_reasoning"] = check_result["tool_result"].get("reasoning", "")
                    yes_results.append(item)
                elif relevance == "no":
                    no_count += 1
                else:
                    item["relevance_reasoning"] = check_result["tool_result"].get("reasoning", "")
                    unknown_results.append(item)
            return yes_results, unknown_results, no_count
        
        yes_results, unknown_results, no_count = await loop.run_in_executor(thread_pool, classify_results)
        
        return {
            "tool_result": {
                "yes": yes_results,
                "unknown": unknown_results,
                "no_count": no_count,
                "keywords_used": keywords
            }
        }
    except Exception as e:
        print(f"[BM_TOOLS] batch_search_and_filter error: {e}")
        return {
            "tool_result": {
                "yes": [],
                "unknown": [],
                "no_count": 0,
                "keywords_used": [keyword],
                "error": str(e)
            }
        }


# ==================== 工具管理器 ====================

class BMToolManager:
    """
    BrowseMaster工具管理器 - 支持独立 LLM 配置
    
    提供统一的工具调用接口，管理搜索客户端和LLM配置。
    默认跟随 Executor 配置，使用本地模型执行工具调用。
    
    支持两种搜索模式：
    - local: 使用 LocalSearch 本地向量检索服务
    - online: 使用 OnlineSearch 在线网络搜索（Google/DuckDuckGo）
    """
    
    def __init__(
        self,
        search_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        config: Optional[BMToolsConfig] = None,
        search_mode: str = "auto",
        google_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        file_cache_dir: Optional[str] = None
    ):
        """
        初始化工具管理器
        
        Args:
            search_base_url: 搜索服务器URL，默认从环境变量LOCAL_SEARCH_URL获取
            llm_model: LLM模型名称，默认跟随 Executor 配置
            llm_base_url: LLM API 端点，默认跟随 Executor 配置
            llm_api_key: LLM API 密钥，默认跟随 Executor 配置
            config: 工具配置
            search_mode: 搜索模式，可选值:
                - "local": 使用 LocalSearch
                - "online": 使用 OnlineSearch (Google/Serper/DuckDuckGo)
                - "auto": 自动选择（有 LOCAL_SEARCH_URL 用 local，否则用 online）
            google_api_key: Google Custom Search API Key（用于 online 模式）
            serper_api_key: Serper API Key（用于 online 模式）
            jina_api_key: Jina Reader API Key（用于网页抓取）
            file_cache_dir: 文件解析缓存目录
        """
        self.search_base_url = search_base_url or os.getenv("LOCAL_SEARCH_URL")
        # LLM 配置：默认跟随 Executor 配置（本地模型）
        self.llm_model = llm_model or os.getenv("BM_LLM_MODEL") or os.getenv("HYMEM_EXECUTOR_MODEL", "Qwen3-32B")
        self.llm_base_url = llm_base_url or os.getenv("BM_BASE_URL") or os.getenv("HYMEM_EXECUTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        self.llm_api_key = llm_api_key or os.getenv("BM_API_KEY") or os.getenv("HYMEM_EXECUTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.config = config or BMToolsConfig()
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_KEY") or os.getenv("SERPER_KEY_ID")
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS")
        
        # 确定搜索模式
        env_search_mode = os.getenv("SEARCH_MODE", "auto").lower()
        self.search_mode = search_mode if search_mode != "auto" else env_search_mode
        
        # 自动模式逻辑：有 LOCAL_SEARCH_URL 用 local，否则用 online
        if self.search_mode == "auto":
            if self.search_base_url:
                self.search_mode = "local"
            else:
                self.search_mode = "online"
        
        self._search_client = None
        self._search_mode_initialized = False
        
        # 初始化文件解析器
        from .file_tools import HymemFileParser
        self.file_parser = HymemFileParser(cache_dir=file_cache_dir)
        # 统一工具调用统计（BM/HyMem共用同一口径）
        self.tool_calls_total = 0
        self.tool_calls_by_tool: Dict[str, int] = {}
        self.tool_call_history: List[Dict[str, Any]] = []
        # Source attribution for atomic tool calls (Step A3 — see set_source).
        # Recorded on every _record_tool_call.
        self._tool_call_source: str = "actor"
        self.tool_calls_by_source: Dict[str, int] = {}
        # Aggregated distillation stats across all web_parse calls in a session.
        self.distillation_stats_aggregate: Dict[str, int] = {
            "chunks_total": 0,
            "chunks_kept": 0,
            "chunks_dropped": 0,
            "candidates_extracted": 0,
            "candidates_kept_after_gate": 0,
            "total_aux_calls": 0,
            "invocations": 0,
            # O3: track max chunk-concurrency observed across the session
            "chunk_concurrent_used_max": 0,
        }
        # O1: URL-level web_parse cache. Two layers:
        #   - _parse_cache: (url, query_hash) → full tool_result. Skips entire
        #     fetch+distill+LLM-extract chain on hit.
        #   - _raw_page_cache: url → raw page_content. When the same URL is
        #     parsed with a DIFFERENT query, we still skip the network fetch
        #     and only re-run distillation+LLM-extract.
        # Both LRU-capped at 256 entries to bound memory.
        # Disabled if env HYMEM_PARSE_CACHE=0.
        from collections import OrderedDict as _OD
        self._parse_cache: "_OD[Any, Dict]" = _OD()
        self._raw_page_cache: "_OD[str, str]" = _OD()
        self._parse_cache_max = 256
        self._parse_cache_enabled = os.getenv("HYMEM_PARSE_CACHE", "1") != "0"
        self._parse_cache_hits = 0
        self._parse_cache_misses = 0
        self._raw_page_hits = 0  # url-only hits (saved network fetch but re-distilled)

    def _record_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Dict[str, Any]):
        """记录一次原子工具调用，统一统计口径。

        Step A3: also attributes the call to ``self._tool_call_source``
        ('actor' | 'branch' | 'replan') so HyMem can split branch-internal
        calls from outer Actor calls in reports without losing the unified
        atomic-count semantics.
        """
        self.tool_calls_total += 1
        self.tool_calls_by_tool[tool_name] = self.tool_calls_by_tool.get(tool_name, 0) + 1
        source = self._tool_call_source or "actor"
        self.tool_calls_by_source[source] = self.tool_calls_by_source.get(source, 0) + 1

        tool_result = result.get("tool_result", {}) if isinstance(result, dict) else {}
        error = tool_result.get("error") if isinstance(tool_result, dict) else None
        success = not bool(error)

        self.tool_call_history.append({
            "tool_name": tool_name,
            "arguments": arguments,
            "success": success,
            "source": source,
            "error": str(error)[:300] if error else "",
        })

        # Aggregate per-call distillation stats (set by web_parse on long pages).
        d_stats = tool_result.get("distillation_stats") if isinstance(tool_result, dict) else None
        if d_stats:
            agg = self.distillation_stats_aggregate
            agg["invocations"] += 1
            for k in ("chunks_total", "chunks_kept", "chunks_dropped",
                      "candidates_extracted", "candidates_kept_after_gate",
                      "total_aux_calls"):
                if k in d_stats:
                    agg[k] = agg.get(k, 0) + int(d_stats[k])
            # O3: max-of-max so we can see if env override took effect
            cc = int(d_stats.get("chunk_concurrent_used", 0))
            if cc > agg.get("chunk_concurrent_used_max", 0):
                agg["chunk_concurrent_used_max"] = cc

    def set_source(self, source: str) -> None:
        """Attribute subsequent atomic tool calls to ``source`` ('actor' |
        'branch' | 'replan'). The HyMem agent toggles this at branch
        entry/exit so branch-internal tool calls are tagged accordingly."""
        if source not in ("actor", "branch", "replan"):
            source = "actor"
        self._tool_call_source = source

    def get_source(self) -> str:
        return self._tool_call_source

    def get_tool_stats(self) -> Dict[str, Any]:
        """导出工具统计信息。"""
        return {
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_by_tool": dict(self.tool_calls_by_tool),
            "tool_calls_by_source": dict(self.tool_calls_by_source),
            "tool_call_history": list(self.tool_call_history),
            "distillation_stats_aggregate": dict(self.distillation_stats_aggregate),
            # O1: web_parse cache effectiveness
            "parse_cache_hits": self._parse_cache_hits,
            "parse_cache_misses": self._parse_cache_misses,
            "parse_cache_hit_rate": (
                self._parse_cache_hits
                / max(self._parse_cache_hits + self._parse_cache_misses, 1)
            ),
            "raw_page_cache_hits": self._raw_page_hits,
            "parse_cache_enabled": self._parse_cache_enabled,
        }
    
    @property
    def search_client(self):
        """获取或创建搜索客户端"""
        if self._search_client is None:
            if self.search_mode == "online":
                # 使用在线搜索
                from envs.online_search import OnlineSearchClient
                self._search_client = OnlineSearchClient(
                    google_api_key=self.google_api_key,
                    serper_api_key=self.serper_api_key,
                    jina_api_key=self.jina_api_key,
                    use_google=False,
                    use_serper=True,
                    use_duckduckgo=False,
                    use_jina=True
                )
                if not self._search_mode_initialized:
                    # 显示可用的搜索引擎
                    engines = []
                    if self._search_client.google and self._search_client.google.is_available:
                        engines.append("Google")
                    if self._search_client.serper and self._search_client.serper.is_available:
                        engines.append("Serper")
                    if self._search_client.duckduckgo:
                        engines.append("DuckDuckGo")
                    engine_str = "/".join(engines) if engines else "None"
                    # 显示网页抓取器
                    fetcher_str = "Jina" if self._search_client.fetcher.jina and self._search_client.fetcher.jina.is_available else "Direct"
                    print(f"[BMToolManager] Using OnlineSearchClient (Search: {engine_str}, Fetcher: {fetcher_str})")
                    self._search_mode_initialized = True
            else:
                # 使用本地搜索
                if not self.search_base_url:
                    self.search_base_url = "http://localhost:8021"
                self._search_client = AsyncSearchClient(base_url=self.search_base_url)
                if not self._search_mode_initialized:
                    print(f"[BMToolManager] Using LocalSearch at {self.search_base_url}")
                    self._search_mode_initialized = True
        return self._search_client
    
    async def close(self):
        """关闭搜索客户端连接"""
        if self._search_client is not None:
            await self._search_client.close()
            self._search_client = None
    
    def is_online_mode(self) -> bool:
        """检查是否使用在线搜索模式"""
        return self.search_mode == "online"
    
    async def web_search(self, keywords: str, topk: Optional[int] = None) -> Dict[str, Any]:
        """执行网页搜索"""
        topk = topk or self.config.search_topk
        result = await web_search(keywords, self.search_client, topk=topk)
        self._record_tool_call("web_search", {"keywords": keywords, "topk": topk}, result)
        return result
    
    async def web_parse(self, link: str, query: str) -> Dict[str, Any]:
        """执行网页解析 — with O1 (url, query) cache."""
        # === O1: cache check ===
        if self._parse_cache_enabled:
            cache_key = (
                self._canonical_url(link),
                self._normalize_query_for_cache(query),
            )
            hit = self._parse_cache.get(cache_key)
            if hit is not None:
                # LRU move-to-end + bookkeeping
                self._parse_cache.move_to_end(cache_key)
                self._parse_cache_hits += 1
                # Tag the result so callers can see it was a cache hit
                # without changing the result shape (additive only).
                if isinstance(hit, dict) and "tool_result" in hit:
                    hit["tool_result"]["cache_hit"] = True
                # Still record the tool_call so atomic counter is consistent
                # (a cached call IS a tool invocation from the agent's POV).
                self._record_tool_call("web_parse",
                                       {"link": link, "query": query, "cache_hit": True},
                                       hit)
                return hit
            self._parse_cache_misses += 1

        result = await web_parse(
            link, query, self.search_client,
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            # Hand the manager so web_parse can populate _raw_page_cache
            # for cross-query reuse on the same URL.
            tool_manager=self,
        )
        self._record_tool_call("web_parse", {"link": link, "query": query}, result)

        # Store in cache (LRU eviction)
        if self._parse_cache_enabled and result.get("tool_result"):
            self._parse_cache[cache_key] = result
            self._parse_cache.move_to_end(cache_key)
            while len(self._parse_cache) > self._parse_cache_max:
                self._parse_cache.popitem(last=False)
        return result

    @staticmethod
    def _canonical_url(url: str) -> str:
        """Strip fragments + trailing slash + lowercase scheme/host so trivial
        URL variations hit the same cache entry."""
        u = (url or "").strip()
        if not u:
            return u
        # Strip fragment
        if "#" in u:
            u = u.split("#", 1)[0]
        # Strip trailing slash (but keep root)
        if u.endswith("/") and len(u) > 8:
            u = u[:-1]
        # Lowercase scheme + host (path stays case-sensitive)
        try:
            from urllib.parse import urlparse, urlunparse
            p = urlparse(u)
            if p.scheme and p.netloc:
                u = urlunparse((p.scheme.lower(), p.netloc.lower(),
                                p.path, p.params, p.query, ""))
        except Exception:
            pass
        return u

    @staticmethod
    def _normalize_query_for_cache(query: str) -> str:
        """Normalize query for cache key: lowercase, drop punct, take first 8
        meaningful tokens. Two queries that ask the same thing in slightly
        different wording will hash to the same key."""
        if not query:
            return ""
        import re as _re
        toks = _re.findall(r"[a-z0-9]+", query.lower())
        # Drop common stopwords cheaply
        stop = {"a", "an", "the", "of", "to", "in", "on", "for", "and", "or",
                "is", "are", "was", "were", "what", "who", "how", "where", "when",
                "why", "which", "this", "that", "with", "by", "as", "at", "from"}
        toks = [t for t in toks if t not in stop][:8]
        return " ".join(toks)
    
    async def generate_keywords(self, seed_keyword: str, max_keywords: Optional[int] = None) -> Dict[str, Any]:
        """生成搜索关键词"""
        max_keywords = max_keywords or self.config.max_keywords
        result = await generate_keywords(
            seed_keyword, 
            llm_model=self.llm_model, 
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            max_keywords=max_keywords
        )
        self._record_tool_call(
            "generate_keywords",
            {"seed_keyword": seed_keyword, "max_keywords": max_keywords},
            result
        )
        return result
    
    async def batch_search(self, keywords: List[str], topk_per_keyword: Optional[int] = None) -> Dict[str, Any]:
        """执行批量搜索"""
        topk_per_keyword = topk_per_keyword or self.config.batch_search_topk
        result = await batch_search(keywords, self.search_client, topk_per_keyword=topk_per_keyword)
        self._record_tool_call(
            "batch_search",
            {"keywords": keywords, "topk_per_keyword": topk_per_keyword},
            result
        )
        return result
    
    async def check_condition(self, content: str, condition: str) -> Dict[str, Any]:
        """检查条件"""
        result = await check_condition(
            content, condition, 
            llm_model=self.llm_model,
            llm_base_url=self.llm_base_url,
            llm_api_key=self.llm_api_key,
            max_content_length=self.config.max_content_length
        )
        self._record_tool_call("check_condition", {"condition": condition}, result)
        return result
    
    async def batch_search_and_filter(
        self, 
        keyword: str, 
        max_keywords: Optional[int] = None,
        topk_per_keyword: Optional[int] = None
    ) -> Dict[str, Any]:
        """执行批量搜索与过滤（内部子调用通过 self 路由，确保每个原子工具调用都被统计）"""
        max_keywords = max_keywords or self.config.max_keywords
        topk_per_keyword = topk_per_keyword or self.config.batch_search_topk

        try:
            keywords_result = await self.generate_keywords(
                keyword, max_keywords=max_keywords
            )
            keywords = keywords_result["tool_result"]["keywords"]

            search_result = await self.batch_search(
                keywords, topk_per_keyword=topk_per_keyword
            )
            organic = search_result["tool_result"]["organic"]

            if not organic:
                return {
                    "tool_result": {
                        "yes": [],
                        "unknown": [],
                        "no_count": 0,
                        "keywords_used": keywords
                    }
                }

            check_semaphore = asyncio.Semaphore(5)

            async def check_with_control(item: Dict):
                async with check_semaphore:
                    content = (
                        f"Title: {item.get('title', '')}\n"
                        f"URL: {item.get('link', '')}\n"
                        f"Snippet: {item.get('snippet', '')}"
                    )
                    try:
                        result = await self.check_condition(content, keyword)
                        return item, result
                    except Exception as e:
                        return item, {"tool_result": {"is_relevant": "unknown", "reasoning": str(e)}}

            check_tasks = [check_with_control(item) for item in organic]
            check_results = await asyncio.gather(*check_tasks)

            yes_results = []
            unknown_results = []
            no_count = 0
            for item, check_result in check_results:
                relevance = check_result["tool_result"]["is_relevant"]
                if relevance == "yes":
                    item["relevance_reasoning"] = check_result["tool_result"].get("reasoning", "")
                    yes_results.append(item)
                elif relevance == "no":
                    no_count += 1
                else:
                    item["relevance_reasoning"] = check_result["tool_result"].get("reasoning", "")
                    unknown_results.append(item)

            return {
                "tool_result": {
                    "yes": yes_results,
                    "unknown": unknown_results,
                    "no_count": no_count,
                    "keywords_used": keywords
                }
            }
        except Exception as e:
            print(f"[BM_TOOLS] batch_search_and_filter error: {e}")
            return {
                "tool_result": {
                    "yes": [],
                    "unknown": [],
                    "no_count": 0,
                    "keywords_used": [keyword],
                    "error": str(e)
                }
            }
    
    async def parse_file(self, file_path: str) -> Dict[str, Any]:
        """解析文件内容（支持PDF/CSV/XLSX/DOCX/TXT/HTML）"""
        result = await self.file_parser.parse_file(file_path)
        wrapped_result = {
            "tool_result": {
                "file_path": file_path,
                "content": result.get("content", ""),
                "metadata": result.get("metadata", {})
            }
        }
        self._record_tool_call("parse_file", {"file_path": file_path}, wrapped_result)
        return wrapped_result
    
    def get_tool_by_name(self, tool_name: str):
        """根据名称获取工具函数"""
        tools = {
            'web_search': self.web_search,
            'web_parse': self.web_parse,
            'generate_keywords': self.generate_keywords,
            'batch_search': self.batch_search,
            'check_condition': self.check_condition,
            'batch_search_and_filter': self.batch_search_and_filter,
            'parse_file': self.parse_file
        }
        return tools.get(tool_name)


# ==================== 测试函数 ====================

async def _test_tools():
    """测试工具函数"""
    print("Testing BrowseMaster Tools...")
    
    # 创建工具管理器
    manager = BMToolManager()
    
    try:
        # 测试 web_search
        print("\n1. Testing web_search...")
        result = await manager.web_search("machine learning", topk=3)
        print(f"   Found {len(result['tool_result']['organic'])} results")
        
        # 测试 generate_keywords
        print("\n2. Testing generate_keywords...")
        result = await manager.generate_keywords("deep learning models 2023", max_keywords=5)
        print(f"   Generated keywords: {result['tool_result']['keywords']}")
        
        # 测试 batch_search
        print("\n3. Testing batch_search...")
        keywords = ["neural network", "deep learning"]
        result = await manager.batch_search(keywords, topk_per_keyword=2)
        print(f"   Found {result['tool_result']['total_results']} results from {result['tool_result']['keywords_searched']} keywords")
        
        # 测试 check_condition
        print("\n4. Testing check_condition...")
        content = "GPT-4 is a large language model developed by OpenAI in 2023."
        condition = "information about large language models"
        result = await manager.check_condition(content, condition)
        print(f"   Relevance: {result['tool_result']['is_relevant']}")
        
        # 测试 batch_search_and_filter
        print("\n5. Testing batch_search_and_filter...")
        result = await manager.batch_search_and_filter(
            "transformer architecture papers", 
            max_keywords=3, 
            topk_per_keyword=2
        )
        print(f"   Yes: {len(result['tool_result']['yes'])}, Unknown: {len(result['tool_result']['unknown'])}, No: {result['tool_result']['no_count']}")
        
        print("\nAll tests completed!")
        
    finally:
        await manager.close()


if __name__ == "__main__":
    asyncio.run(_test_tools())
