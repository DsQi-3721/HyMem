"""Unit tests for the Serper key-pool + auto-rotation logic.

Tests are network-free (mock httpx) and do not require chromadb / sentence-
transformers / numpy heavy stack. Cover:
  - pool loads from JSON correctly
  - explicit env var prepended, dedup'd
  - rotation on quota error
  - persistence (exhausted_indices written back to disk)
  - all-exhausted → returns empty + signals unavailable
  - non-quota errors do NOT rotate (transient)

Runnable two ways:
    pytest tests/test_serper_pool.py -v
    python -m tests.test_serper_pool
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _has_httpx_and_numpy() -> bool:
    try:
        import httpx, numpy  # noqa: F401
        return True
    except ImportError:
        return False


def _skip(reason: str) -> bool:
    print(f"  SKIP  (test skipped: {reason})")
    return True


# ---------- Pool loading ----------

def test_pool_loads_from_json():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["k1", "k2", "k3"],
            "current_index": 0,
            "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        os.environ.pop("SERPER_KEY_ID", None)
        pool = _SerperKeyPool(pool_path=pool_path)
        assert pool.is_available
        assert pool.api_key == "k1"
        assert pool.status()["pool_size"] == 3


def test_pool_skips_placeholder_keys():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["PASTE_PRIMARY_HERE", "real_key", "PASTE_BACKUP"],
            "current_index": 0,
            "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        pool = _SerperKeyPool(pool_path=pool_path)
        assert pool.api_key == "real_key"
        assert pool.status()["pool_size"] == 1


def test_pool_explicit_env_key_prepended():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["pool_k1", "pool_k2"], "current_index": 0,
            "exhausted_indices": [],
        }))
        os.environ["SERPER_KEY"] = "env_key"
        try:
            pool = _SerperKeyPool(pool_path=pool_path)
            assert pool.api_key == "env_key"
            assert pool.status()["pool_size"] == 3
        finally:
            os.environ.pop("SERPER_KEY", None)


def test_pool_explicit_param_takes_precedence_over_env():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    os.environ["SERPER_KEY"] = "env_key"
    try:
        pool = _SerperKeyPool(api_key="explicit_param", pool_path=Path("/nonexistent"))
        assert pool.api_key == "explicit_param"
    finally:
        os.environ.pop("SERPER_KEY", None)


def test_pool_no_keys_means_unavailable():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    os.environ.pop("SERPER_KEY", None)
    os.environ.pop("SERPER_KEY_ID", None)
    pool = _SerperKeyPool(pool_path=Path("/definitely/nonexistent"))
    assert not pool.is_available
    assert pool.api_key is None


def test_pool_skips_pre_marked_exhausted_indices():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["dead1", "live1", "live2"],
            "current_index": 0,
            "exhausted_indices": [0],
        }))
        os.environ.pop("SERPER_KEY", None)
        pool = _SerperKeyPool(pool_path=pool_path)
        assert pool.api_key == "live1"


# ---------- Rotation ----------

def test_pool_rotates_on_exhaustion():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["k1", "k2", "k3"], "current_index": 0,
            "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        pool = _SerperKeyPool(pool_path=pool_path)
        assert pool.api_key == "k1"

        rotated = pool.mark_current_exhausted("test")
        assert rotated is True
        assert pool.api_key == "k2"

        rotated = pool.mark_current_exhausted("test")
        assert rotated is True
        assert pool.api_key == "k3"

        rotated = pool.mark_current_exhausted("test")
        assert rotated is False
        assert pool.api_key is None
        assert not pool.is_available


def test_pool_persists_exhausted_to_disk():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["k1", "k2"], "current_index": 0, "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        pool = _SerperKeyPool(pool_path=pool_path)
        pool.mark_current_exhausted("test")
        cfg = json.loads(pool_path.read_text())
        assert 0 in cfg["exhausted_indices"]


# ---------- SerperSearch quota detection ----------

def test_serper_quota_error_pattern():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import SerperSearch

    s = SerperSearch.__new__(SerperSearch)
    s._pool = MagicMock()

    assert s._is_quota_error(Exception("HTTP 429 Too Many Requests"))
    assert s._is_quota_error(Exception("Quota exceeded for today"))
    assert s._is_quota_error(Exception("Rate-limited at endpoint"))
    assert s._is_quota_error(Exception("Forbidden: invalid API key"))
    assert s._is_quota_error(Exception("403 Forbidden"))

    assert not s._is_quota_error(Exception("Connection timeout"))
    assert not s._is_quota_error(Exception("DNS resolution failed"))
    assert not s._is_quota_error(Exception("internal server error 500"))


def test_serper_search_rotates_on_quota_error():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import SerperSearch, _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["dead_k1", "alive_k2"],
            "current_index": 0,
            "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        os.environ.pop("SERPER_KEY_ID", None)

        original_init = _SerperKeyPool.__init__
        def _patched(self, api_key=None):
            return original_init(self, api_key=api_key, pool_path=pool_path)

        with patch.object(_SerperKeyPool, "__init__", _patched):
            s = SerperSearch(timeout=1.0, max_retries=1)
            assert s.api_key == "dead_k1"

            call_count = {"n": 0}

            async def fake_do_search(query, k):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise Exception("HTTP 429 Too Many Requests")
                return [{"title": "ok", "url": "x", "text": "ok", "docid": "x",
                         "position": 1, "source": "serper"}]

            s._do_search = fake_do_search
            results = asyncio.run(s.search("hello", k=1))

        assert len(results) == 1
        assert results[0]["title"] == "ok"
        assert s.api_key == "alive_k2"
        assert call_count["n"] == 2


def test_serper_search_returns_empty_when_all_exhausted():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import SerperSearch, _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["k1", "k2"], "current_index": 0, "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        os.environ.pop("SERPER_KEY_ID", None)

        original_init = _SerperKeyPool.__init__
        def _patched(self, api_key=None):
            return original_init(self, api_key=api_key, pool_path=pool_path)

        with patch.object(_SerperKeyPool, "__init__", _patched):
            s = SerperSearch(timeout=1.0, max_retries=1)

            async def fake_do_search(query, k):
                raise Exception("HTTP 429")

            s._do_search = fake_do_search
            results = asyncio.run(s.search("hello", k=1))

        assert results == []
        assert not s.is_available


def test_serper_search_does_not_rotate_on_transient_error():
    if not _has_httpx_and_numpy():
        return _skip("httpx/numpy not in env")
    from envs.online_search import SerperSearch, _SerperKeyPool

    with tempfile.TemporaryDirectory() as td:
        pool_path = Path(td) / "keys.json"
        pool_path.write_text(json.dumps({
            "keys": ["k1", "k2"], "current_index": 0, "exhausted_indices": [],
        }))
        os.environ.pop("SERPER_KEY", None)
        os.environ.pop("SERPER_KEY_ID", None)

        original_init = _SerperKeyPool.__init__
        def _patched(self, api_key=None):
            return original_init(self, api_key=api_key, pool_path=pool_path)

        with patch.object(_SerperKeyPool, "__init__", _patched):
            s = SerperSearch(timeout=0.1, max_retries=2)

            call_count = {"n": 0}

            async def fake_do_search(query, k):
                call_count["n"] += 1
                raise Exception("Connection timeout")

            s._do_search = fake_do_search
            results = asyncio.run(s.search("hello", k=1))

        assert s.api_key == "k1"
        assert results == []
        assert call_count["n"] == 2


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
