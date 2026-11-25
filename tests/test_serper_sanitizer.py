"""Unit tests for SerperSearch._sanitize_for_free_tier.

Goal: every query the user observed Serper free-tier rejecting must come
out of the sanitizer in a form Serper accepts (no quotes, no operators).

Reference rejection bodies (from real run logs):
    {"message":"Query pattern not allowed for free accounts.","statusCode":400}

Tests are network-free.

Runnable two ways:
    pytest tests/test_serper_sanitizer.py -v
    python -m tests.test_serper_sanitizer
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _has_httpx() -> bool:
    try:
        import httpx, numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _skip(reason: str) -> bool:
    print(f"  SKIP  (test skipped: {reason})")
    return True


def _new_sanitizer_instance():
    """Build a SerperSearch with sanitize_free_tier=True without needing
    a real key or pool init."""
    from envs.online_search import SerperSearch
    s = SerperSearch.__new__(SerperSearch)
    s.sanitize_free_tier = True
    return s


# ---------- Operator stripping ----------

def test_strips_site_operator():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('site:nature.com Article Research articles filter')
    assert 'site:' not in out
    # Operator prefix stripped, value retained as plain keyword (preserves
    # the LLM's intent that nature.com is relevant)
    assert 'nature.com' in out
    assert 'Article' in out
    assert 'filter' in out


def test_strips_multiple_operators():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier(
        'site:arxiv.org filetype:pdf inurl:abstract intitle:LLM agents'
    )
    for forbidden in ('site:', 'filetype:', 'inurl:', 'intitle:'):
        assert forbidden not in out, f"{forbidden!r} still in {out!r}"
    # Values are retained as plain keywords
    assert 'arxiv.org' in out
    assert 'pdf' in out
    assert 'abstract' in out
    assert 'LLM' in out
    assert 'agents' in out


def test_keeps_normal_words_with_colons():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('time 10:30 meeting')
    assert 'time' in out
    assert '10:30' in out
    assert 'meeting' in out


# ---------- Quoted phrase unwrap ----------

def test_unwraps_single_quoted_phrase():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('"1037" Nature 2020 research articles')
    assert '"' not in out
    assert '1037' in out
    assert 'Nature' in out


def test_unwraps_many_quoted_phrases():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier(
        '"Nature" "2020" "research articles" "published" count'
    )
    assert '"' not in out
    for w in ('Nature', '2020', 'research', 'articles', 'published', 'count'):
        assert w in out, f"lost {w!r} from {out!r}"


def test_unwraps_combined_operators_and_quotes():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier(
        'site:nature.com "Article" "Research articles" filter'
    )
    assert '"' not in out
    assert 'site:' not in out
    # Operator value kept as keyword
    assert 'nature.com' in out
    assert 'Article' in out
    assert 'filter' in out


def test_handles_quoted_phrase_with_spaces():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('the "complete works of shakespeare" online')
    assert '"' not in out
    assert 'complete works of shakespeare' in out
    assert 'online' in out


# ---------- Boolean OR + exclusions ----------

def test_strips_boolean_or():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('Python OR JavaScript tutorial')
    assert ' OR ' not in out
    assert 'Python' in out
    assert 'JavaScript' in out


def test_strips_exclusion_dash():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('Python tutorial -beginner -basic')
    assert 'beginner' in out
    assert 'basic' in out
    assert not any(tok.startswith('-') for tok in out.split())


def test_preserves_compound_words_with_dash():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('state-of-the-art transformer model')
    assert 'state-of-the-art' in out
    assert 'transformer' in out


# ---------- Whitespace + edge cases ----------

def test_collapses_whitespace():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('site:foo.com    word1     "word2"')
    assert '  ' not in out


def test_empty_after_sanitize():
    """When the query is ONLY bare operator prefixes (no values, no other
    words), result should be empty. This is rare but possible if the LLM
    emits `site: site: site:` style noise."""
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('site: filetype: inurl:')
    assert out == "" or out.strip() == ""


def test_already_clean_query_unchanged():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    inp = 'how does HyMem hierarchical context isolation work'
    assert s._sanitize_for_free_tier(inp) == inp


def test_chinese_query_unchanged():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    inp = '中国科学院深圳先进技术研究院'
    assert s._sanitize_for_free_tier(inp) == inp


def test_chinese_query_with_quotes_unwraps():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    out = s._sanitize_for_free_tier('"机器学习" 论文 2024')
    assert '"' not in out
    assert '机器学习' in out
    assert '论文' in out


# ---------- All user-observed failures pass through cleanly ----------

USER_FAILURE_QUERIES = [
    '"1037" Nature 2020 research articles',
    '"Nature" "2020" "research articles" "published" count',
    '"1037" "Nature" "2020" research',
    'site:nature.com "Research articles" "Year 2020"',
    '"Nature" "2020" "research articles" "1037"',
    '"Nature" "published" "2020" "articles" "research" "800"',
    '"Nature" "2020" "research articles" "900"',
    '"Nature" "2020" "research articles" "1000"',
    '"1037" "Nature" "research" "2020"',
    'site:nature.com "Article" "2020" "Nature" research articles count',
    '"Nature" "citable documents" 2020',
    '"Nature" "2020" "citable" SCImago',
    '"Nature" "research articles" "2020" "published" "per year"',
    'site:nature.com "Article" "Research articles" filter',
]


def test_all_user_failures_become_safe():
    if not _has_httpx(): return _skip("httpx not available")
    s = _new_sanitizer_instance()
    for q in USER_FAILURE_QUERIES:
        out = s._sanitize_for_free_tier(q)
        assert out, f"Sanitized to empty: {q!r}"
        assert '"' not in out, f"Quotes remain: {out!r}"
        # No operator prefix `op:` (but value alone is fine — `nature.com`
        # without the `site:` prefix is just a domain string Google handles).
        for op in ('site:', 'filetype:', 'inurl:', 'intitle:'):
            assert op not in out, f"Operator {op!r} remains: {out!r}"


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
