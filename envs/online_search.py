"""
OnlineSearch - 在线搜索客户端

支持 Google 搜索和 DuckDuckGo 搜索，为 HyMem 框架提供真实网络检索能力。
与 LocalSearch 的 AsyncSearchClient 接口完全兼容。

使用方法:
    1. 设置环境变量（可选）:
       - GOOGLE_API_KEY + GOOGLE_SEARCH_ENGINE_ID: Google Custom Search API
    2. 如果没有 API Key，将自动使用 DuckDuckGo（免费，无需配置）
"""

import os
import re
import copy
import asyncio
import logging
import collections
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from urllib.parse import unquote, urlparse, parse_qs
import httpx

# 导入 LocalSearch 的通用函数
from .local_search import (
    GRADER_TEMPLATE, parse_judge_response, extract_citations_from_response,
    em_score, relaxed_em, call_openai_raw, judge, keep_first_n_words,
    extract_json_tool, extract_fn_call, extract_q_dict
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ==================== 搜索引擎实现 ====================

class DuckDuckGoSearch:
    """
    DuckDuckGo 搜索（免费，无需 API Key）
    
    使用 DuckDuckGo 的 HTML 版本进行搜索，无需 API 配置。
    支持中文搜索。
    """
    
    def __init__(self, timeout: float = 30.0, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    
    async def search(self, query: str, k: int = 10, region: str = "cn-zh") -> List[Dict[str, Any]]:
        """
        执行 DuckDuckGo 搜索
        
        Args:
            query: 搜索关键词
            k: 返回结果数量
            region: 地区设置，默认 cn-zh（中文）
            
        Returns:
            搜索结果列表，格式与 LocalSearch 兼容
        """
        for attempt in range(self.max_retries):
            try:
                return await self._do_search(query, k, region)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"[DuckDuckGo] Search failed after {self.max_retries} attempts: {e}")
                    return []
                await asyncio.sleep(1 * (attempt + 1))
        return []
    
    async def _do_search(self, query: str, k: int, region: str) -> List[Dict[str, Any]]:
        """执行实际搜索"""
        # 使用 DuckDuckGo HTML 版本（更稳定）
        url = "https://html.duckduckgo.com/html/"
        data = {
            "q": query,
            "kl": region,  # 地区设置
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, data=data, headers=self.headers)
            response.raise_for_status()
        
        # 解析 HTML 响应
        results = self._parse_html_response(response.text, k)
        
        if results:
            logger.info(f"[DuckDuckGo] Found {len(results)} results for: {query[:50]}...")
        
        return results
    
    def _parse_html_response(self, html: str, k: int) -> List[Dict[str, Any]]:
        """解析 DuckDuckGo HTML 响应"""
        results = []
        
        # 使用正则表达式解析（避免依赖 BeautifulSoup）
        # 匹配搜索结果块
        result_pattern = r'<div class="result[^"]*"[^>]*>.*?</div>\s*</div>'
        title_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>'
        
        # 简化的解析方式
        # 查找所有链接和标题
        link_matches = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        snippet_matches = re.findall(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        
        for i, (href, title) in enumerate(link_matches):
            if i >= k:
                break
            
            # 提取真实 URL
            real_url = self._extract_real_url(href)
            if not real_url or 'duckduckgo.com' in real_url:
                continue
            
            # 清理标题
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            
            # 获取对应的摘要
            snippet = ""
            if i < len(snippet_matches):
                snippet = re.sub(r'<[^>]+>', '', snippet_matches[i]).strip()
            
            results.append({
                "title": clean_title,
                "url": real_url,
                "text": snippet,
                "docid": f"ddg_{i+1}_{hash(real_url) % 10000}",
                "position": i + 1,
                "source": "duckduckgo"
            })
        
        return results
    
    def _extract_real_url(self, href: str) -> str:
        """从 DuckDuckGo 重定向链接中提取真实 URL"""
        if not href:
            return ""
        
        # DuckDuckGo 使用 uddg 参数存储真实 URL
        if "uddg=" in href:
            match = re.search(r'uddg=([^&]+)', href)
            if match:
                return unquote(match.group(1))
        
        # 直接链接
        if href.startswith("http"):
            return href
        
        return ""


class GoogleSearch:
    """
    Google Custom Search API 搜索
    
    需要设置环境变量:
    - GOOGLE_API_KEY: Google API Key
    - GOOGLE_SEARCH_ENGINE_ID: Custom Search Engine ID
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        search_engine_id: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        # 从环境变量获取
        self.google_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.search_engine_id = search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        self.timeout = timeout
        self.max_retries = max_retries
    
    @property
    def is_available(self) -> bool:
        """检查是否有可用的 API Key"""
        return bool(self.google_api_key) and bool(self.search_engine_id)
    
    async def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """执行 Google Custom Search"""
        if not self.is_available:
            logger.debug("[Google] No API key available")
            return []
        
        for attempt in range(self.max_retries):
            try:
                return await self._search_with_custom_search(query, k)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"[Google] Search failed after {self.max_retries} attempts: {e}")
                    return []
                await asyncio.sleep(1 * (attempt + 1))
        return []
    
    async def _search_with_custom_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """使用 Google Custom Search API"""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": self.google_api_key,
            "cx": self.search_engine_id,
            "num": min(k, 10),  # Custom Search API 每次最多返回 10 条
            "lr": "lang_zh-CN"
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        
        results = []
        for i, item in enumerate(data.get("items", [])[:k], 1):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "text": item.get("snippet", ""),
                "docid": f"google_{i}_{hash(item.get('link', '')) % 10000}",
                "position": i,
                "source": "google_custom"
            })
        
        if results:
            logger.info(f"[Google/CustomSearch] Found {len(results)} results for: {query[:50]}...")
        
        return results


class _SerperKeyPool:
    """Pool of Serper API keys with auto-rotation on quota errors.

    Reads from a JSON config file (default: scripts/serper_keys.json) so
    multiple keys can be added/removed without code changes. On 429 or
    quota-exhausted responses the current key is marked exhausted and the
    pool advances to the next available key.

    Fallback order at startup:
      1. Explicit api_key parameter
      2. JSON pool file (if exists)
      3. SERPER_KEY env var
      4. SERPER_KEY_ID env var

    Persistence: when a key is marked exhausted at runtime, we write
    `exhausted_indices` back to the pool file so a subsequent restart
    skips dead keys. (Best-effort; ignored if file is read-only.)
    """

    DEFAULT_POOL_PATH = None  # set after class def below to avoid Path issues

    def __init__(self, api_key: Optional[str] = None, pool_path=None):
        # Lazy import: this module already imports Path-style operations elsewhere;
        # keep things simple with a string path resolved here.
        import json as _json
        from pathlib import Path as _Path
        self._json = _json
        self._Path = _Path

        # Resolve pool path
        if pool_path is None:
            pool_path = _Path(__file__).parent.parent / "scripts" / "serper_keys.json"
        else:
            pool_path = _Path(pool_path)
        self.pool_path = pool_path

        self._pool: List[str] = []
        self._current_index: int = 0
        self._exhausted: set = set()

        # Single-key path (highest priority)
        explicit = api_key or os.getenv("SERPER_KEY") or os.getenv("SERPER_KEY_ID")

        # GAIA's direct-key runner can explicitly bypass the persisted JSON
        # pool so stale exhausted_indices never mask its manual SERPER_KEY.
        disable_pool = os.getenv("SERPER_DISABLE_KEY_POOL", "").lower() in (
            "1", "true", "yes"
        )

        # Load pool file if it exists and the caller did not disable it.
        if self.pool_path.exists() and not disable_pool:
            try:
                with open(self.pool_path) as f:
                    cfg = self._json.load(f)
                self._pool = [k for k in cfg.get("keys", []) if k and not k.startswith("PASTE_")]
                self._current_index = int(cfg.get("current_index", 0))
                self._exhausted = set(int(i) for i in cfg.get("exhausted_indices", []))
            except (OSError, ValueError) as e:
                logger.warning(f"[Serper] Failed to load key pool {self.pool_path}: {e}")
                self._pool = []
        elif disable_pool:
            logger.info("[Serper] JSON key pool disabled; using direct SERPER_KEY only")

        # Splice the explicit key (env / param) into the pool front so it
        # gets used first if both are configured.
        if explicit and explicit not in self._pool:
            self._pool.insert(0, explicit)

        # Skip past any keys already marked exhausted
        self._advance_to_live_key()

    def _advance_to_live_key(self) -> None:
        """Move current_index forward past any exhausted positions."""
        n = len(self._pool)
        if n == 0:
            self._current_index = -1
            return
        for _ in range(n):
            if self._current_index not in self._exhausted and 0 <= self._current_index < n:
                return
            self._current_index = (self._current_index + 1) % n
        # All exhausted
        self._current_index = -1

    @property
    def is_available(self) -> bool:
        return (self._current_index >= 0
                and self._current_index < len(self._pool)
                and self._current_index not in self._exhausted)

    @property
    def api_key(self) -> Optional[str]:
        if not self.is_available:
            return None
        return self._pool[self._current_index]

    def mark_current_exhausted(self, reason: str = "") -> bool:
        """Mark the current key dead and rotate. Returns True if a fresh
        key is now active, False if pool fully drained."""
        if self._current_index < 0 or self._current_index >= len(self._pool):
            return False
        idx = self._current_index
        self._exhausted.add(idx)
        logger.warning(
            f"[Serper] Key index {idx} marked exhausted ({reason}). "
            f"Remaining: {len(self._pool) - len(self._exhausted)} / {len(self._pool)}"
        )
        self._save_state()
        self._current_index = (idx + 1) % len(self._pool)
        self._advance_to_live_key()
        return self.is_available

    def _save_state(self) -> None:
        """Best-effort write back the exhausted set + current_index."""
        if not self.pool_path.exists():
            return
        try:
            with open(self.pool_path) as f:
                cfg = self._json.load(f)
            cfg["current_index"] = self._current_index
            cfg["exhausted_indices"] = sorted(self._exhausted)
            from datetime import datetime as _dt
            cfg["_last_update"] = _dt.now().isoformat(timespec="seconds")
            with open(self.pool_path, "w") as f:
                self._json.dump(cfg, f, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.debug(f"[Serper] Could not persist pool state: {e}")

    def status(self) -> Dict[str, Any]:
        return {
            "pool_size": len(self._pool),
            "current_index": self._current_index,
            "exhausted_indices": sorted(self._exhausted),
            "remaining": max(0, len(self._pool) - len(self._exhausted)),
        }


class SerperSearch:
    """
    Serper API 搜索 (google.serper.dev)

    需要设置环境变量: SERPER_KEY 或 SERPER_KEY_ID
    或: scripts/serper_keys.json 文件中维护多 key 池（自动 429/quota 轮换）
    """

    # Patterns indicating the current key is dead (rotate, don't just retry).
    # NOTE: Serper sometimes signals "key out of credits" via HTTP 400 with
    # body `{"message":"Not enough credits","statusCode":400}` instead of
    # 402/429. Match that explicitly so the pool rotates instead of dying
    # on the first key.
    _QUOTA_ERROR_RE = re.compile(
        r"\b(429|403|402)\b|quota|rate.?limit|exhaust|insufficient|"
        r"forbidden|not.?authorized|invalid.?api.?key|"
        r"not.?enough.?credit|out.?of.?credit|no.?credit|credits?\b",
        re.IGNORECASE,
    )

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0, max_retries: int = 3):
        self._pool = _SerperKeyPool(api_key=api_key)
        self.timeout = timeout
        self.max_retries = max_retries
        if self._pool.is_available:
            status = self._pool.status()
            if status["pool_size"] > 1:
                logger.info(f"[Serper] Multi-key pool active: {status}")

    @property
    def api_key(self) -> Optional[str]:
        return self._pool.api_key

    @property
    def is_available(self) -> bool:
        return self._pool.is_available

    def _is_quota_error(self, exc: Exception) -> bool:
        # Inspect both the exception message AND the response body —
        # httpx.HTTPStatusError.__str__ does NOT include the response body
        # by default, so plain str(exc) misses messages like
        # `{"message":"Not enough credits","statusCode":400}` that we need
        # to detect for key rotation.
        text = str(exc)
        resp = getattr(exc, "response", None)
        if resp is not None:
            body = getattr(resp, "text", "") or ""
            text = f"{text}\n{body[:300]}"
        return bool(self._QUOTA_ERROR_RE.search(text))
    
    def _has_chinese(self, text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    async def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if not self.is_available:
            logger.debug("[Serper] No API key available")
            return []
        # Serper returns HTTP 400 ("Missing query parameter") on empty/blank
        # queries — short-circuit so we don't waste retries and pollute the
        # log with confusing 400s when the agent emits an empty search.
        if not query or not query.strip():
            logger.debug("[Serper] Empty query, returning []")
            return []

        # Allow up to (max_retries \u00D7 pool_size) attempts so one bad key doesn't
        # wipe out our budget for this query.
        total_attempts = self.max_retries * max(1, len(self._pool._pool))
        attempt = 0
        while attempt < total_attempts and self.is_available:
            attempt += 1
            try:
                return await self._do_search(query, k)
            except Exception as e:
                if self._is_quota_error(e):
                    rotated = self._pool.mark_current_exhausted(
                        reason=f"quota error: {str(e)[:120]}"
                    )
                    if rotated:
                        logger.info(f"[Serper] Rotated to next key, retrying...")
                        continue
                    else:
                        logger.error(
                            f"[Serper] All {len(self._pool._pool)} keys exhausted; "
                            f"returning empty result.")
                        return []
                # Permanent (non-retryable) 400 from Serper free-tier policy:
                # the same query will get the same 400 next time — short
                # circuit to spare retries + log noise. The agent falls back
                # to the next query / engine on []. See message body
                # "Query pattern not allowed for free accounts".
                _resp = getattr(e, "response", None)
                _body = (getattr(_resp, "text", "") or "")[:300]
                if "Query pattern not allowed" in _body:
                    logger.info(
                        f"[Serper] free-tier query-pattern rejection, "
                        f"returning []  (q={query[:80]!r})"
                    )
                    return []
                # Non-quota / transient error
                if attempt >= self.max_retries:
                    logger.error(f"[Serper] Search failed after {attempt} attempts: {e}")
                    return []
                await asyncio.sleep(1 * attempt)
        return []
    
    async def _do_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        url = "https://google.serper.dev/search"

        # Serper rejects queries > ~2k chars and ones containing control
        # characters (returns 400). Agents occasionally emit huge queries
        # when an LLM dumps its full <think> as the search keyword.
        # Defensive normalization: strip control chars + truncate to 1500.
        q_clean = "".join(ch for ch in query if ord(ch) >= 32 or ch in "\t")
        q_clean = q_clean.strip()
        if len(q_clean) > 1500:
            logger.warning(
                f"[Serper] truncating long query {len(q_clean)}->1500 chars"
            )
            q_clean = q_clean[:1500]
        query = q_clean

        if self._has_chinese(query):
            payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn", "num": min(k, 20)}
        else:
            payload = {"q": query, "location": "United States", "gl": "us", "hl": "en", "num": min(k, 20)}
        
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                # Surface the real failure reason so 400/403 can be
                # diagnosed (was: silent re-raise via raise_for_status).
                logger.error(
                    f"[Serper] HTTP {response.status_code} on q={query!r:.120s} "
                    f"key_prefix={(self.api_key or '')[:8]}... "
                    f"body={response.text[:240]}"
                )
            response.raise_for_status()
            data = response.json()
        
        results = []
        for i, item in enumerate(data.get("organic", [])[:k], 1):
            snippet = item.get("snippet", "")
            if "date" in item:
                snippet = f"[{item['date']}] {snippet}"
            
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "text": snippet,
                "docid": f"serper_{i}_{hash(item.get('link', '')) % 10000}",
                "position": i,
                "source": "serper"
            })
        
        if results:
            logger.info(f"[Serper] Found {len(results)} results for: {query[:50]}...")
        
        return results


class JinaFetcher:
    """
    Jina Reader API 网页抓取器
    
    使用 Jina Reader API (r.jina.ai) 抓取网页，返回干净的 Markdown 格式内容。
    比直接抓取 HTML 更可靠，支持动态渲染页面。
    
    需要设置环境变量: JINA_API_KEY 或 JINA_API_KEYS
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: float = 50.0, max_retries: int = 3):
        self.api_key = api_key or os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS")
        self.timeout = timeout
        self.max_retries = max_retries
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def fetch(self, url: str) -> Dict[str, Any]:
        """使用 Jina Reader API 抓取网页"""
        if not self.is_available:
            return {"url": url, "text": "", "docid": url, "title": "", "error": "No Jina API key"}
        
        for attempt in range(self.max_retries):
            try:
                return await self._do_fetch(url)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"[Jina] Failed to fetch {url}: {e}")
                    return {
                        "url": url,
                        "text": f"[Jina] Failed to read page: {str(e)}",
                        "docid": url,
                        "title": ""
                    }
                await asyncio.sleep(0.5 * (attempt + 1))
        
        return {"url": url, "text": "", "docid": url, "title": ""}
    
    async def _do_fetch(self, url: str) -> Dict[str, Any]:
        """执行 Jina API 请求"""
        jina_url = f"https://r.jina.ai/{url}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()
        
        content = response.text
        
        # 从 Jina 返回的 Markdown 中提取标题
        title = ""
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        if content:
            logger.info(f"[Jina] Fetched {len(content)} chars from: {url[:50]}...")
        
        return {
            "url": url,
            "text": content,
            "docid": url,
            "title": title,
            "source": "jina"
        }


# ---------------------------------------------------------------------------
# Wikipedia REST API fetcher (front-branch in WebPageFetcher.fetch())
# ---------------------------------------------------------------------------
# Wikipedia's MediaWiki API (`/w/api.php`) is a separate program-access endpoint
# that does NOT go through Cloudflare's anti-bot rules. Direct fetching of
# `wikipedia.org/wiki/<title>` is increasingly rate-limited / 403'd for
# datacenter IPs and bot UAs, but the API endpoint accepts any UA and has
# generous rate limits (~200 req/s).
#
# Policy reference: https://www.mediawiki.org/wiki/API:Etiquette
# We add a front branch in WebPageFetcher.fetch() that detects Wikipedia URLs
# and routes them to the API; any other domain falls through to the existing
# Jina → direct-HTTP chain unchanged.

_WIKIPEDIA_HOST_RE = re.compile(
    r"^(?:[a-z]{2,3})\.wikipedia\.org$",
    re.IGNORECASE,
)


def _is_wikipedia_url(url: str) -> bool:
    """True iff `url` points at a Wikipedia article page."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if not parsed.hostname:
        return False
    if not _WIKIPEDIA_HOST_RE.match(parsed.hostname):
        return False
    # Only article pages (path starts with /wiki/), not /w/ (special pages).
    return parsed.path.startswith("/wiki/")


def _parse_wikipedia_title_and_lang(url: str) -> Tuple[str, str]:
    """Extract (page_title, language_code) from a Wikipedia article URL.

    Examples:
      https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire   -> ("A_Song_of_Ice_and_Fire", "en")
      https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD       -> ("中国", "zh")
      https://en.wikipedia.org/wiki/Foo#Section_2            -> ("Foo", "en")
    """
    parsed = urlparse(url)
    lang = parsed.hostname.split(".", 1)[0] if parsed.hostname else "en"
    # Strip "/wiki/" prefix and any "#anchor" suffix; URL-decode percent escapes.
    raw_title = parsed.path[len("/wiki/"):]
    title = unquote(raw_title)
    # Path-decoded title — keep underscores; Wikipedia API accepts both forms.
    return title, lang


class WikipediaAPIFetcher:
    """Plain-text article fetcher using the public MediaWiki API.

    Returns a result dict in the same schema as JinaFetcher / WebPageFetcher
    so the calling code is interchangeable.
    """

    # The User-Agent here follows the MediaWiki API:Etiquette recommendation
    # (identifying UA with contact info). Any UA works in practice; this just
    # avoids being flagged for rate-limiting under heavy load.
    DEFAULT_UA = "HyMemBot/1.0 (research; https://github.com/anonymous/hymem)"

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 50000,
        max_retries: int = 2,
        user_agent: Optional[str] = None,
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retries = max_retries
        self.headers = {
            "User-Agent": user_agent or self.DEFAULT_UA,
            "Accept": "application/json",
        }

    async def fetch(self, url: str) -> Dict[str, Any]:
        """Fetch via MediaWiki API. Returns dict with text/url/docid/title.
        Caller can detect failure via empty `text` or text starting with
        `[WikiAPI] Failed`."""
        title, lang = _parse_wikipedia_title_and_lang(url)
        api_url = f"https://{lang}.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "redirects": 1,        # auto-follow redirects to canonical title
            "titles": title,
            "format": "json",
            "formatversion": 2,    # cleaner JSON shape
        }

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(api_url, params=params, headers=self.headers)
                    response.raise_for_status()
                data = response.json()

                pages = (data.get("query") or {}).get("pages") or []
                if not pages:
                    return {
                        "url": url,
                        "text": f"[WikiAPI] No page found for title: {title}",
                        "docid": url,
                        "title": title,
                        "source": "wikipedia_api",
                        "error": "no_pages",
                    }

                page = pages[0]
                if page.get("missing"):
                    return {
                        "url": url,
                        "text": f"[WikiAPI] Page does not exist: {title}",
                        "docid": url,
                        "title": title,
                        "source": "wikipedia_api",
                        "error": "missing",
                    }

                extract = page.get("extract", "") or ""
                resolved_title = page.get("title", title)

                if len(extract) > self.max_content_length:
                    extract = extract[:self.max_content_length] + "\n[Content truncated...]"

                logger.info(
                    f"[WikiAPI] Fetched {len(extract)} chars: "
                    f"{lang}.wikipedia.org/wiki/{title[:40]}..."
                )
                return {
                    "url": url,
                    "text": extract,
                    "docid": url,
                    "title": resolved_title,
                    "source": "wikipedia_api",
                }

            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"[WikiAPI] Failed to fetch {url}: {e}")
                    return {
                        "url": url,
                        "text": f"[WikiAPI] Failed: {e}",
                        "docid": url,
                        "title": title,
                        "source": "wikipedia_api",
                        "error": str(e),
                    }
                await asyncio.sleep(0.5 * (attempt + 1))

        # Unreachable but keep return-shape consistent
        return {"url": url, "text": "", "docid": url, "title": title, "source": "wikipedia_api"}


# ---------------------------------------------------------------------------


class WebPageFetcher:
    """
    网页内容抓取器

    优先级: Wikipedia API (域名匹配时) > Jina Reader > 直接抓取
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_content_length: int = 50000,
        max_retries: int = 2,
        jina_api_key: Optional[str] = None,
        use_jina: bool = True,
        use_wikipedia_api: bool = True,
    ):
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.max_retries = max_retries
        self.use_jina = use_jina
        self.use_wikipedia_api = use_wikipedia_api
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        # 初始化 Jina 抓取器
        self.jina = JinaFetcher(api_key=jina_api_key, timeout=timeout) if use_jina else None
        # Wikipedia API fetcher — no key, no quota; only used for wikipedia.org URLs.
        self.wiki_api = WikipediaAPIFetcher(
            timeout=timeout,
            max_content_length=max_content_length,
            max_retries=max_retries,
        ) if use_wikipedia_api else None
        self._fetcher_logged = False

    async def fetch(self, url: str) -> Dict[str, Any]:
        """
        抓取网页内容

        优先级:
          0. Wikipedia API (域名匹配 *.wikipedia.org/wiki/*)
          1. Jina Reader (其他域名，需 JINA_API_KEY)
          2. 直接抓取 (fallback)
        """
        # 0. Wikipedia 专用 API 路径 — 完全绕开反爬
        if self.wiki_api and _is_wikipedia_url(url):
            result = await self.wiki_api.fetch(url)
            text = result.get("text", "") or ""
            # 仅当 API 返回了实际内容时使用结果；否则继续走 Jina/HTTP 兜底
            if text and not text.startswith("[WikiAPI]"):
                return result
            logger.info(f"[WikiAPI] Empty/failed for {url}; falling back to Jina/HTTP")

        # 1. 尝试 Jina（如果可用）
        if self.jina and self.jina.is_available:
            if not self._fetcher_logged:
                logger.info("[WebFetcher] Using Jina Reader API")
                self._fetcher_logged = True
            result = await self.jina.fetch(url)
            if result.get("text") and not result["text"].startswith("[Jina] Failed"):
                # 截断过长内容
                if len(result["text"]) > self.max_content_length:
                    result["text"] = result["text"][:self.max_content_length] + "\n[Content truncated...]"
                return result

        # 2. 回退到直接抓取
        if not self._fetcher_logged:
            logger.info("[WebFetcher] Using direct HTTP fetch")
            self._fetcher_logged = True

        for attempt in range(self.max_retries):
            try:
                return await self._do_fetch(url)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.warning(f"[WebFetcher] Failed to fetch {url}: {e}")
                    return {
                        "url": url,
                        "text": f"Error fetching page: {str(e)}",
                        "docid": url,
                        "title": ""
                    }
                await asyncio.sleep(0.5 * (attempt + 1))

        return {"url": url, "text": "", "docid": url, "title": ""}
    
    async def _do_fetch(self, url: str) -> Dict[str, Any]:
        """直接抓取 HTML"""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            verify=False
        ) as client:
            response = await client.get(url, headers=self.headers)
            response.raise_for_status()
        
        html = response.text
        
        # 提取标题
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""
        title = re.sub(r'<[^>]+>', '', title)
        
        # 移除脚本、样式等
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<aside[^>]*>.*?</aside>', '', html, flags=re.IGNORECASE | re.DOTALL)
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
        
        # 提取正文
        text = re.sub(r'<[^>]+>', '\n', html)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        # 截断过长内容
        if len(text) > self.max_content_length:
            text = text[:self.max_content_length] + "\n[Content truncated...]"
        
        return {
            "url": url,
            "text": text,
            "docid": url,
            "title": title,
            "source": "direct"
        }


# ==================== 在线搜索客户端（兼容 AsyncSearchClient 接口）====================

class OnlineSearchClient:
    """
    在线搜索客户端
    
    与 LocalSearch 的 AsyncSearchClient 接口完全兼容，可以直接替换使用。
    
    搜索优先级: Google Custom Search > Serper > DuckDuckGo
    
    使用示例:
        client = OnlineSearchClient()
        results = await client.search("机器学习", k=10)
        pages = await client.open(url="https://example.com")
    """
    
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        jina_api_key: Optional[str] = None,
        use_google: bool = True,
        use_serper: bool = True,
        use_duckduckgo: bool = True,
        use_jina: bool = True,
        timeout: float = 30.0,
        cache_pages: bool = True
    ):
        """
        初始化在线搜索客户端
        
        Args:
            google_api_key: Google Custom Search API Key（可选）
            serper_api_key: Serper API Key（可选，也可通过 SERPER_KEY 环境变量设置）
            jina_api_key: Jina Reader API Key（可选，也可通过 JINA_API_KEY 环境变量设置）
            use_google: 是否使用 Google Custom Search
            use_serper: 是否使用 Serper API
            use_duckduckgo: 是否使用 DuckDuckGo 搜索（免费回退）
            use_jina: 是否使用 Jina Reader API 抓取网页
            timeout: 请求超时时间
            cache_pages: 是否缓存已抓取的页面
        """
        self.use_google = use_google
        self.use_serper = use_serper
        self.use_duckduckgo = use_duckduckgo
        self.use_jina = use_jina
        self.timeout = timeout
        self.cache_pages = cache_pages
        
        # 初始化搜索引擎（优先级: Google > Serper > DuckDuckGo）
        self.google = GoogleSearch(api_key=google_api_key, timeout=timeout) if use_google else None
        self.serper = SerperSearch(api_key=serper_api_key, timeout=timeout) if use_serper else None
        self.duckduckgo = DuckDuckGoSearch(timeout=timeout) if use_duckduckgo else None
        self.fetcher = WebPageFetcher(timeout=timeout, jina_api_key=jina_api_key, use_jina=use_jina)
        
        # 页面缓存
        self._page_cache: Dict[str, Dict] = {}
        
        # docid 到 url 的映射缓存（用于 Serper 等搜索引擎返回的非 URL docid）
        self._docid_to_url: Dict[str, str] = {}
        
        # 记录使用的搜索引擎
        self._last_search_engine: str = ""
    
    async def close(self):
        """关闭客户端（兼容接口）"""
        self._page_cache.clear()
    
    async def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        执行在线搜索
        
        优先级: Google Custom Search > Serper > DuckDuckGo
        
        Args:
            query: 搜索关键词
            k: 返回结果数量
            
        Returns:
            搜索结果列表，格式与 LocalSearch 兼容:
            [{"title": str, "url": str, "text": str, "docid": str, "position": int}, ...]
        """
        results = []
        
        # 1. 尝试 Google Custom Search
        if self.google and self.google.is_available:
            results = await self.google.search(query, k)
            if results:
                self._last_search_engine = "google"
                # 保存 docid 到 url 的映射
                for item in results:
                    docid = item.get("docid", "")
                    item_url = item.get("url", "")
                    if docid and item_url:
                        self._docid_to_url[docid] = item_url
                return results
        
        # 2. 尝试 Serper API
        if self.serper and self.serper.is_available:
            results = await self.serper.search(query, k)
            if results:
                self._last_search_engine = "serper"
                # 保存 docid 到 url 的映射
                for item in results:
                    docid = item.get("docid", "")
                    item_url = item.get("url", "")
                    if docid and item_url:
                        self._docid_to_url[docid] = item_url
                return results
        
        # 3. 回退到 DuckDuckGo（免费）
        if self.duckduckgo:
            results = await self.duckduckgo.search(query, k)
            if results:
                self._last_search_engine = "duckduckgo"
                # 保存 docid 到 url 的映射
                for item in results:
                    docid = item.get("docid", "")
                    item_url = item.get("url", "")
                    if docid and item_url:
                        self._docid_to_url[docid] = item_url
                return results
        
        logger.warning("[OnlineSearch] No results found from any search engine")
        return []
    
    async def open(
        self,
        url: Optional[str] = None,
        docid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        打开网页获取内容
        
        兼容 LocalSearch 的 open 接口。
        
        Args:
            url: 网页 URL
            docid: 文档 ID（可能是真实 URL 或 Serper 等返回的内部标识符）
            
        Returns:
            页面内容列表:
            [{"url": str, "text": str, "docid": str, "title": str}]
        """
        # 确定目标 URL
        if url:
            target_url = url
        elif docid:
            # 优先从映射缓存中查找真实 URL
            if docid in self._docid_to_url:
                target_url = self._docid_to_url[docid]
            elif docid.startswith('http://') or docid.startswith('https://'):
                # docid 本身就是 URL
                target_url = docid
            else:
                logger.warning(f"[OnlineSearch] docid '{docid}' not found in cache, cannot resolve to URL")
                return []
        else:
            return []
        
        # 检查缓存
        if self.cache_pages and target_url in self._page_cache:
            return [self._page_cache[target_url]]
        
        # 抓取页面
        page = await self.fetcher.fetch(target_url)
        
        # 缓存
        if self.cache_pages:
            self._page_cache[target_url] = page
        
        return [page]
    
    async def batch_search(
        self,
        queries: List[str],
        k_per_query: int = 5,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """
        批量搜索（并行执行）
        
        Args:
            queries: 关键词列表
            k_per_query: 每个关键词返回的结果数量
            max_concurrent: 最大并发数
            
        Returns:
            合并去重后的搜索结果
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def search_with_semaphore(q: str):
            async with semaphore:
                return await self.search(q, k_per_query)
        
        tasks = [search_with_semaphore(q) for q in queries]
        all_results = await asyncio.gather(*tasks)
        
        # 合并去重
        seen_urls = set()
        merged = []
        position = 1
        
        for results in all_results:
            for item in results:
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    item["position"] = position
                    merged.append(item)
                    position += 1
        
        return merged
    
    def get_last_search_engine(self) -> str:
        """获取上次使用的搜索引擎"""
        return self._last_search_engine


# ==================== OnlineSearch 环境类（兼容 LocalSearch）====================

class OnlineSearch:
    """
    OnlineSearch 环境类
    
    与 LocalSearch 类接口完全兼容，可以在 HyMem 框架中直接替换使用。
    """
    
    def __init__(self, config, tokenizer, ability):
        self.config = config
        self.tokenizer = tokenizer
        self.ability = ability
        self.stats = collections.Counter()
        self.stats['finish'] = 0
        self.stats['search'] = 0
        self.stats['open_page'] = 0
        self.stats['change_answer'] = 0
        self.stats['is_search'] = 0
        self.stats['is_open'] = 0
        self.stats['is_finish'] = 0
        self.stats['visit_pages'] = 0
        self.stats['search_engine'] = ''
        self.env_fail = False
        
        # 使用在线搜索客户端
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self.client = OnlineSearchClient(google_api_key=google_api_key)
        
        self.question = None
        self.label_answer = None
        self.predicted_answer = None
        self.double_check = getattr(self.config.plugin, "double_check", False) if hasattr(self.config, 'plugin') else False
        self.donotgiveup = False
        self.must_search = getattr(self.config.plugin, "must_search", True) if hasattr(self.config, 'plugin') else True
        self.visited_pages = set()
        self.is_finish = False
    
    async def init_env(self, item):
        """初始化环境"""
        extra_info = item.non_tensor_batch['extra_info'][0]
        self.question = extra_info.get('query', '')
        self.label_answer = extra_info.get('answer', '')
        self.predicted_answer = None
        
        # 记录数据源，用于判断是否使用中文评判模板
        self.data_source = ''
        if 'data_source' in item.non_tensor_batch:
            ds = item.non_tensor_batch['data_source']
            self.data_source = ds[0] if isinstance(ds, (list, tuple, np.ndarray)) and len(ds) > 0 else str(ds)
        if not self.data_source:
            self.data_source = extra_info.get('data_source', '')
    
    async def get_data(self, item, context):
        """获取数据"""
        if 'prompt' in item.non_tensor_batch['extra_info'][0]:
            prompt = item.non_tensor_batch['extra_info'][0]['prompt']
            conversations = [
                {'role': 'system', 'content': prompt[0]['content']},
                {'role': 'user', 'content': prompt[-1]['content']},
            ]
        else:
            conversations = [
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': ''},
            ]
        
        self.instance_info = copy.deepcopy(item.non_tensor_batch['extra_info'][0])
        self.instance_info['problem_statement'] = self.instance_info['query']
        meta_info = copy.copy(item.meta_info)
        meta_info['uid'] = item.non_tensor_batch['uid'][0]
        meta_info['reward_model'] = item.non_tensor_batch['reward_model'][0]
        
        if "max_turn" in item.meta_info:
            max_turn = item.meta_info["max_turn"]
        else:
            if context.is_train:
                max_turn = self.config.plugin.max_turn
            else:
                if hasattr(self.config.plugin, "val_max_turn"):
                    max_turn = self.config.plugin.val_max_turn
                else:
                    max_turn = self.config.plugin.max_turn
        
        return conversations, {'max_turn': max_turn, 'meta_info': meta_info}
    
    async def run_action(self, response):
        """执行动作"""
        self.stats['action'] += 1
        fn_call = extract_fn_call(response)
        
        if fn_call is None or len(fn_call) == 0:
            return {'observation': 'No function call was detected in the model response.'}
        
        observation = ''
        for fn in fn_call:
            name = fn['function']
            
            if name == 'search':
                self.stats['search'] += 1
                self.stats['is_search'] = 1
                query = fn['arguments'].get('query', '')
                topk = (lambda v: int(v) if str(v).isdigit() else 10)(fn['arguments'].get('topk', 10))
                
                if not query:
                    observation += '[Error] The "search" function requires a "query" argument.'
                else:
                    observation += f'[Search Results for "{query}"]\n'
                    serp = await self.client.search(query, topk * 5)  # 多搜索一些，过滤后返回
                    
                    # 记录使用的搜索引擎
                    self.stats['search_engine'] = self.client.get_last_search_engine()
                    
                    show_topk = 0
                    for i, page in enumerate(serp, 1):
                        if page['url'] in self.visited_pages:
                            page['text'] = "(This page was already seen. Shorter snippet shown.) " + " ".join(page['text'].split()[:128])
                            show_topk += 0.25
                        else:
                            self.visited_pages.add(page['url'])
                            self.stats['visit_pages'] = len(self.visited_pages)
                            page['text'] = " ".join(page['text'].split()[:512])
                            show_topk += 1
                        
                        observation += (
                            f"\n--- #{i}: {page.get('docid', '')} ---\n"
                            f"docid: {page.get('docid', '')}\n"
                            f"url: {page['url']}\n"
                            f"title: {page.get('title', '')}\n"
                            f"content: {page['text']}\n"
                        )
                        
                        if show_topk >= topk:
                            break
                    observation += "\n"
            
            elif name == 'open_page':
                self.stats['open_page'] += 1
                self.stats['is_open'] = 1
                url = fn['arguments'].get('url', None)
                docid = fn['arguments'].get('docid', None)
                
                if not docid and not url:
                    observation += '[Error] The "open_page" function requires either a "docid" or a "url".'
                else:
                    open_pages = await self.client.open(url, docid)
                    for page in open_pages:
                        page['text'] = keep_first_n_words(page['text'], 4096)
                        observation += (
                            f"[Opened Page Content]\n"
                            f"docid: {page['docid']}\n"
                            f"url: {page['url']}\n"
                            f"title: {page.get('title', '')}\n"
                            f"content: {page['text']}\n"
                        )
                    observation += "\n"
            
            elif name == 'finish':
                self.stats['is_finish'] = 1
                self.is_finish = True
                answer = fn['arguments'].get('answer', "")
                explanation = fn['arguments'].get('explanation', None)
                confidence = fn['arguments'].get('confidence', None)
                
                if len(answer.strip()) == 0:
                    observation = (
                        "Fail to parse answer. Please resubmit with the correct tool call format, eg\n"
                        "<function=finish>\n"
                        "<parameter=answer>YOUR ANSWER</parameter>\n"
                        "<parameter=explanation>YOUR EXPLANATION</parameter>\n"
                        "<parameter=confidence>YOUR CONFIDENCE</parameter>\n"
                        "</function>\n"
                    )
                    return {'observation': observation.strip()}
                
                if self.predicted_answer is not None:
                    if self.predicted_answer[0] != answer:
                        self.stats['change_answer'] += 1
                
                if self.stats['search'] == 0:
                    if em_score(self.label_answer, answer) and self.must_search:
                        observation = "Answer submission failed. You MUST use the search tool to verify the answer."
                        self.must_search = False
                        return {'observation': observation.strip()}
                    answer = ""
                
                self.predicted_answer = (answer, explanation, confidence)
                
                if 'insufficient' in answer.lower() and self.donotgiveup:
                    observation = "The answer can be found through sufficient search. Do not give up."
                    self.donotgiveup = False
                    return {'observation': observation.strip()}
                
                if '<q1>' in self.label_answer:
                    label_answer_dict = extract_q_dict(self.label_answer)
                    predicted_answer_dict = extract_q_dict(self.predicted_answer[0])
                    missing = [k for k in label_answer_dict if k not in predicted_answer_dict]
                    
                    if missing:
                        observation = f"Answer submission failed. Missing: {', '.join(missing)}."
                        return {'observation': observation.strip()}
                
                if self.double_check:
                    observation = f"""Check your answer against: {self.question}
                    
Verify all conditions are met. If any gaps, search again."""
                    self.double_check = False
                    return {'observation': observation.strip()}
                
                return {'action': 'finish'}
            
            else:
                observation = f'[Error] The function "{name}" is not supported.'
        
        observation += "\n\n* Please reflect on the information obtained and keep searching if needed."
        return {'observation': observation.strip()}
    
    async def get_reward(self, item, messages, context):
        """获取奖励"""
        if self.env_fail:
            return "", 0, {}
        if self.predicted_answer is None:
            return "", 0, {}
        
        # 检查是否使用中文评判模板（用于 BrowseComp-ZH）
        use_chinese = False
        data_source = getattr(self, 'data_source', '')
        if 'browsecomp_zh' in str(data_source).lower():
            use_chinese = True
        elif 'browsecomp_zh' in str(self.ability).lower():
            use_chinese = True
        # 也可以通过配置强制指定
        if hasattr(self.config, 'plugin') and getattr(self.config.plugin, 'use_chinese_judge', False):
            use_chinese = True
        
        if '<q1>' in self.label_answer:
            label_answer_dict = extract_q_dict(self.label_answer)
            predicted_answer_dict = extract_q_dict(self.predicted_answer[0])
            all_reward = []
            for k in label_answer_dict:
                if k in predicted_answer_dict:
                    reward = await judge(self.question, label_answer_dict[k], predicted_answer_dict[k],
                                        use_chinese_template=use_chinese)
                    all_reward.append(reward)
                else:
                    all_reward.append(0)
            reward = sum(all_reward) / len(all_reward)
            return "", reward, {}
        
        reward = await judge(self.question, self.label_answer, self.predicted_answer[0],
                            use_chinese_template=use_chinese)
        return "", reward, {}
    
    async def update_dataproto(self, out, item, messages, score, reward_dict, tag='main', metrics=None):
        """更新 DataProto"""
        import numpy as np
        
        final_score = score[1]
        out.meta_info["inference_metrics"] = metrics
        out.meta_info["generation_kwargs"] = item.meta_info['generation_kwargs']
        out.non_tensor_batch = copy.deepcopy(item.non_tensor_batch)
        out.non_tensor_batch["num_of_turns"] = np.array([len(messages)], dtype=object)
        out.non_tensor_batch["turn_clipped"] = np.array([False], dtype=object)
        out.non_tensor_batch["tag"] = np.array([tag], dtype=object)
        out.non_tensor_batch["is_summary"] = np.array([int("summary" in tag)], dtype=object)
        out.non_tensor_batch["traj_cnt"] = np.array([1], dtype=object)
        
        stats = dict(self.stats)
        stats['score'] = final_score
        extra_data = {
            "score": score,
            "call_fail": self.env_fail,
            "action_fail": 0,
            "answer_reached": True,
            "stats": stats
        }
        out.non_tensor_batch['extra_data'] = np.array([extra_data], dtype=object)
        return out


# ==================== 工厂函数 ====================

def create_search_client(
    mode: str = "auto",
    base_url: Optional[str] = None,
    google_api_key: Optional[str] = None
):
    """
    创建搜索客户端
    
    Args:
        mode: "local" | "online" | "auto"
            - local: 使用 LocalSearch（需要 base_url）
            - online: 使用 OnlineSearchClient
            - auto: 如果有 base_url 用 local，否则用 online
        base_url: LocalSearch 服务地址
        google_api_key: Google 搜索 API Key
    
    Returns:
        AsyncSearchClient 或 OnlineSearchClient
    """
    if mode == "local" or (mode == "auto" and base_url):
        from .local_search import AsyncSearchClient
        return AsyncSearchClient(base_url=base_url)
    else:
        return OnlineSearchClient(google_api_key=google_api_key)


def select_search_env(ability: str, config, use_online: bool = False):
    """
    选择搜索环境类
    
    Args:
        ability: 任务能力类型
        config: 配置
        use_online: 是否使用在线搜索
        
    Returns:
        LocalSearch 或 OnlineSearch 类
    """
    if use_online or os.getenv("USE_ONLINE_SEARCH", "").lower() in ("true", "1", "yes"):
        return OnlineSearch
    else:
        from .local_search import LocalSearch
        return LocalSearch


# ==================== 测试函数 ====================

async def _test_online_search():
    """测试在线搜索"""
    print("=" * 60)
    print("Testing OnlineSearchClient...")
    print("=" * 60)
    
    client = OnlineSearchClient()
    
    try:
        # 测试中文搜索
        print("\n[Test 1] 中文搜索: '勃艮第葡萄酒产区'")
        results = await client.search("勃艮第葡萄酒产区", k=5)
        print(f"Found {len(results)} results (engine: {client.get_last_search_engine()})")
        for r in results[:3]:
            print(f"  {r['position']}. {r['title'][:50]}")
            print(f"     URL: {r['url'][:70]}")
            print(f"     Snippet: {r['text'][:80]}...")
        
        # 测试英文搜索
        print("\n[Test 2] 英文搜索: 'machine learning'")
        results = await client.search("machine learning", k=3)
        print(f"Found {len(results)} results")
        for r in results:
            print(f"  {r['position']}. {r['title'][:50]}")
        
        # 测试页面抓取
        if results:
            print("\n[Test 3] 页面抓取")
            pages = await client.open(url=results[0]["url"])
            if pages:
                print(f"  Title: {pages[0].get('title', '')[:50]}")
                print(f"  Content length: {len(pages[0].get('text', ''))} chars")
                print(f"  Content preview: {pages[0].get('text', '')[:150]}...")
        
        # 测试批量搜索
        print("\n[Test 4] 批量搜索")
        batch_results = await client.batch_search(
            ["Python programming", "深度学习"],
            k_per_query=3
        )
        print(f"Batch search found {len(batch_results)} unique results")
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_test_online_search())
