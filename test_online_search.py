#!/usr/bin/env python3
"""
OnlineSearch 功能测试脚本

测试 OnlineSearch 客户端的搜索和网页抓取功能。
无需 API Key 即可使用 DuckDuckGo 搜索。

使用方法:
    conda activate hymem
    cd /data1/test/xq/hymem/FoldAgent
    python test_online_search.py
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envs.online_search import (
    OnlineSearchClient, 
    DuckDuckGoSearch, 
    GoogleSearch, 
    WebPageFetcher,
    create_search_client
)


async def test_duckduckgo():
    """测试 DuckDuckGo 搜索"""
    print("\n" + "=" * 60)
    print("测试 DuckDuckGo 搜索")
    print("=" * 60)
    
    ddg = DuckDuckGoSearch()
    
    print("\n[1] 中文搜索: '勃艮第葡萄酒产区'")
    results = await ddg.search("勃艮第葡萄酒产区", k=5)
    print(f"    找到 {len(results)} 条结果")
    for r in results[:3]:
        print(f"    - {r['title'][:50]}")
        print(f"      URL: {r['url'][:70]}")
    
    print("\n[2] 英文搜索: 'machine learning tutorial'")
    results = await ddg.search("machine learning tutorial", k=3)
    print(f"    找到 {len(results)} 条结果")
    for r in results:
        print(f"    - {r['title'][:50]}")
    
    return len(results) > 0


async def test_google():
    """测试 Google 搜索"""
    print("\n" + "=" * 60)
    print("测试 Google 搜索")
    print("=" * 60)
    
    google = GoogleSearch()
    
    if not google.is_available:
        print("    [跳过] 未配置 GOOGLE_API_KEY 和 GOOGLE_SEARCH_ENGINE_ID")
        return True
    
    print("\n[1] 中文搜索: '机器学习入门'")
    results = await google.search("机器学习入门", k=3)
    print(f"    找到 {len(results)} 条结果")
    for r in results:
        print(f"    - {r['title'][:50]}")
    
    return len(results) > 0


async def test_web_fetcher():
    """测试网页抓取"""
    print("\n" + "=" * 60)
    print("测试网页抓取")
    print("=" * 60)
    
    fetcher = WebPageFetcher()
    test_url = "https://www.example.com"
    print(f"\n[1] 抓取: {test_url}")
    
    page = await fetcher.fetch(test_url)
    
    print(f"    标题: {page.get('title', '')[:50]}")
    print(f"    内容长度: {len(page.get('text', ''))} 字符")
    print(f"    内容预览: {page.get('text', '')[:100]}...")
    
    return len(page.get('text', '')) > 0


async def test_online_search_client():
    """测试 OnlineSearchClient"""
    print("\n" + "=" * 60)
    print("测试 OnlineSearchClient")
    print("=" * 60)
    
    client = OnlineSearchClient()
    
    try:
        print("\n[1] 搜索: '法国首都 巴黎'")
        results = await client.search("法国首都 巴黎", k=5)
        print(f"    找到 {len(results)} 条结果")
        print(f"    使用引擎: {client.get_last_search_engine()}")
        for r in results[:3]:
            print(f"    - {r['title'][:50]}")
        
        if results:
            print(f"\n[2] 打开页面: {results[0]['url'][:50]}...")
            pages = await client.open(url=results[0]["url"])
            if pages:
                print(f"    标题: {pages[0].get('title', '')[:50]}")
                print(f"    内容长度: {len(pages[0].get('text', ''))} 字符")
        
        print("\n[3] 批量搜索")
        batch_results = await client.batch_search(
            ["Python programming", "机器学习框架"],
            k_per_query=3
        )
        print(f"    找到 {len(batch_results)} 条去重结果")
        
        return len(results) > 0
        
    finally:
        await client.close()


async def main():
    """运行所有测试"""
    print("=" * 60)
    print("OnlineSearch 功能测试")
    print("=" * 60)
    
    results = {}
    
    try:
        results["DuckDuckGo"] = await test_duckduckgo()
    except Exception as e:
        print(f"    [错误] {e}")
        results["DuckDuckGo"] = False
    
    try:
        results["Google"] = await test_google()
    except Exception as e:
        print(f"    [错误] {e}")
        results["Google"] = False
    
    try:
        results["WebFetcher"] = await test_web_fetcher()
    except Exception as e:
        print(f"    [错误] {e}")
        results["WebFetcher"] = False
    
    try:
        results["OnlineSearchClient"] = await test_online_search_client()
    except Exception as e:
        print(f"    [错误] {e}")
        results["OnlineSearchClient"] = False
    
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("全部测试通过！" if all_passed else "部分测试失败"))
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
