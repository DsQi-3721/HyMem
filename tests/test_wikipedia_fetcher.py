"""Unit tests for the Wikipedia API front-branch in envs.online_search.

Mocks httpx so the tests are network-free. Covers:
  - URL detection (English / Chinese / Japanese subdomains)
  - Title + lang parsing (with percent-encoding, anchors)
  - Successful API fetch path
  - "page missing" graceful failure
  - HTTP error → fallback signal (text starts with `[WikiAPI]`)
  - Non-Wikipedia URLs → not routed to API path

Runnable two ways:
    pytest tests/test_wikipedia_fetcher.py -v
    python -m tests.test_wikipedia_fetcher
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _has_httpx_and_numpy():
    """Skip httpx-dependent tests when env doesn't have httpx/numpy.
    Pure URL/parsing tests always run."""
    try:
        import httpx, numpy   # noqa: F401
        return True
    except ImportError:
        return False


def _skip(reason: str) -> bool:
    print(f"  SKIP  (test skipped: {reason})")
    return True


# ---------- URL detection ----------

def test_is_wikipedia_english():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _is_wikipedia_url
    assert _is_wikipedia_url("https://en.wikipedia.org/wiki/Albert_Einstein") is True
    assert _is_wikipedia_url("https://en.wikipedia.org/wiki/Foo#Section") is True


def test_is_wikipedia_other_languages():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _is_wikipedia_url
    assert _is_wikipedia_url("https://zh.wikipedia.org/wiki/中国") is True
    assert _is_wikipedia_url("https://ja.wikipedia.org/wiki/日本") is True
    assert _is_wikipedia_url("https://de.wikipedia.org/wiki/Deutschland") is True


def test_is_wikipedia_excludes_special_pages():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _is_wikipedia_url
    # /w/ paths are special pages (login, recent changes, ...) — not articles
    assert _is_wikipedia_url("https://en.wikipedia.org/w/index.php?title=Special:Search") is False
    # Non-wiki paths
    assert _is_wikipedia_url("https://en.wikipedia.org/api/rest_v1/page/summary/X") is False


def test_is_wikipedia_excludes_other_domains():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _is_wikipedia_url
    assert _is_wikipedia_url("https://www.nytimes.com/2024/01/01/foo.html") is False
    assert _is_wikipedia_url("https://example.com/wiki/Foo") is False
    # wiktionary / wikidata are NOT wikipedia
    assert _is_wikipedia_url("https://en.wiktionary.org/wiki/foo") is False
    assert _is_wikipedia_url("https://www.wikidata.org/wiki/Q42") is False


def test_is_wikipedia_handles_garbage():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _is_wikipedia_url
    assert _is_wikipedia_url("") is False
    assert _is_wikipedia_url("not a url") is False
    assert _is_wikipedia_url("http://") is False


# ---------- Title + lang extraction ----------

def test_parse_title_english():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _parse_wikipedia_title_and_lang
    title, lang = _parse_wikipedia_title_and_lang(
        "https://en.wikipedia.org/wiki/A_Song_of_Ice_and_Fire"
    )
    assert title == "A_Song_of_Ice_and_Fire"
    assert lang == "en"


def test_parse_title_url_decoded():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _parse_wikipedia_title_and_lang
    # %E4%B8%AD%E5%9B%BD = "中国"
    title, lang = _parse_wikipedia_title_and_lang(
        "https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD"
    )
    assert title == "中国"
    assert lang == "zh"


def test_parse_title_strips_anchor_when_present():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _parse_wikipedia_title_and_lang
    # urlparse strips fragments anyway, so '#Section' shouldn't appear in path
    title, lang = _parse_wikipedia_title_and_lang(
        "https://en.wikipedia.org/wiki/Foo#History"
    )
    assert title == "Foo"
    assert lang == "en"


# ---------- WikipediaAPIFetcher mocked HTTP success ----------

def test_wiki_api_fetcher_success():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WikipediaAPIFetcher

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "query": {
            "pages": [{
                "pageid": 736,
                "title": "Albert Einstein",
                "extract": "Albert Einstein was a German-born theoretical physicist..."
            }]
        }
    })

    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with patch("envs.online_search.httpx.AsyncClient", return_value=fake_client):
        fetcher = WikipediaAPIFetcher()
        result = asyncio.run(fetcher.fetch(
            "https://en.wikipedia.org/wiki/Albert_Einstein"
        ))

    assert "German-born theoretical physicist" in result["text"]
    assert result["title"] == "Albert Einstein"
    assert result["source"] == "wikipedia_api"
    assert "error" not in result

    # Verify the API URL was constructed correctly
    call_args = fake_client.get.call_args
    assert call_args is not None
    api_url = call_args[0][0]
    assert api_url == "https://en.wikipedia.org/w/api.php"
    params = call_args[1].get("params", {})
    assert params["action"] == "query"
    assert params["prop"] == "extracts"
    assert params["explaintext"] == 1
    assert params["redirects"] == 1
    assert params["titles"] == "Albert_Einstein"


def test_wiki_api_fetcher_missing_page():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WikipediaAPIFetcher

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "query": {"pages": [{"missing": True, "title": "Nonexistent_Page_xyzzy"}]}
    })
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with patch("envs.online_search.httpx.AsyncClient", return_value=fake_client):
        fetcher = WikipediaAPIFetcher(max_retries=1)
        result = asyncio.run(fetcher.fetch(
            "https://en.wikipedia.org/wiki/Nonexistent_Page_xyzzy"
        ))

    # Caller sees a [WikiAPI] prefix → falls through to Jina/HTTP
    assert result["text"].startswith("[WikiAPI]")
    assert result["error"] == "missing"


def test_wiki_api_fetcher_network_error():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WikipediaAPIFetcher

    fake_client = MagicMock()
    fake_client.get = AsyncMock(side_effect=Exception("network exploded"))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with patch("envs.online_search.httpx.AsyncClient", return_value=fake_client):
        fetcher = WikipediaAPIFetcher(max_retries=1)
        result = asyncio.run(fetcher.fetch(
            "https://en.wikipedia.org/wiki/Foo"
        ))

    assert result["text"].startswith("[WikiAPI]")
    assert "network exploded" in result["text"]


def test_wiki_api_fetcher_truncates_long_extract():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WikipediaAPIFetcher

    long_text = "x" * 100000
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json = MagicMock(return_value={
        "query": {"pages": [{"pageid": 1, "title": "Big", "extract": long_text}]}
    })
    fake_client = MagicMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with patch("envs.online_search.httpx.AsyncClient", return_value=fake_client):
        fetcher = WikipediaAPIFetcher(max_content_length=5000)
        result = asyncio.run(fetcher.fetch("https://en.wikipedia.org/wiki/Big"))

    assert "Content truncated" in result["text"]
    assert len(result["text"]) <= 5100   # 5000 + truncation marker


# ---------- WebPageFetcher front-branch routing ----------

def test_webpage_fetcher_routes_wikipedia_to_api():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WebPageFetcher

    fetcher = WebPageFetcher(use_jina=False)   # disable Jina to isolate path
    fetcher.wiki_api.fetch = AsyncMock(return_value={
        "url": "x", "text": "wiki content", "docid": "x", "title": "X",
        "source": "wikipedia_api",
    })
    # Mock direct HTTP so we can detect if it's wrongly called.
    fetcher._do_fetch = AsyncMock(return_value={"text": "should not be used"})

    result = asyncio.run(fetcher.fetch("https://en.wikipedia.org/wiki/X"))

    fetcher.wiki_api.fetch.assert_awaited_once()
    fetcher._do_fetch.assert_not_called()
    assert result["text"] == "wiki content"


def test_webpage_fetcher_falls_back_when_api_fails():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WebPageFetcher

    fetcher = WebPageFetcher(use_jina=False)
    # Wikipedia API "fails" (returns text starting with [WikiAPI])
    fetcher.wiki_api.fetch = AsyncMock(return_value={
        "url": "x", "text": "[WikiAPI] Failed: rate-limited",
        "docid": "x", "title": "X", "source": "wikipedia_api",
    })
    fetcher._do_fetch = AsyncMock(return_value={
        "url": "x", "text": "fallback HTML content", "docid": "x", "title": "X",
    })

    result = asyncio.run(fetcher.fetch("https://en.wikipedia.org/wiki/X"))

    fetcher.wiki_api.fetch.assert_awaited_once()
    fetcher._do_fetch.assert_called()        # fallback fired
    assert result["text"] == "fallback HTML content"


def test_webpage_fetcher_skips_api_for_non_wiki_urls():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WebPageFetcher

    fetcher = WebPageFetcher(use_jina=False)
    fetcher.wiki_api.fetch = AsyncMock()
    fetcher._do_fetch = AsyncMock(return_value={
        "url": "x", "text": "nytimes content", "docid": "x", "title": "X",
    })

    result = asyncio.run(fetcher.fetch("https://www.nytimes.com/2024/01/foo.html"))

    fetcher.wiki_api.fetch.assert_not_called()
    fetcher._do_fetch.assert_called()
    assert result["text"] == "nytimes content"


def test_webpage_fetcher_can_disable_wiki_api():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import WebPageFetcher

    fetcher = WebPageFetcher(use_jina=False, use_wikipedia_api=False)
    assert fetcher.wiki_api is None
    fetcher._do_fetch = AsyncMock(return_value={
        "url": "x", "text": "raw html", "docid": "x", "title": "X",
    })

    result = asyncio.run(fetcher.fetch("https://en.wikipedia.org/wiki/X"))

    fetcher._do_fetch.assert_called()
    assert result["text"] == "raw html"


# ---------- Built-in runner ----------

if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e) or "AssertionError"))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"Ran {len(tests)} tests; {len(failures)} failures/errors.")
    sys.exit(1 if failures else 0)
