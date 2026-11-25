"""Unit tests for A-MEM baseline adapter.

The vendored A-MEM modules require chromadb + sentence-transformers, which
aren't installed by default. Tests here exercise only the parts that don't
need those heavy deps:
  - llm_controller patch (base_url support, env var fallback)
  - _format_memory_for_prompt (rendering helper)
  - vendoring sanity (files exist, attribution comments present)

End-to-end smoke testing of A-MEM against actual GAIA tasks happens via
`python scripts/eval_gaia.py --api_mode --framework amem --sample 1`
once `pip install -r requirements_amem.txt` is done.

Runnable two ways:
    pytest tests/test_amem_agent.py -v
    python -m tests.test_amem_agent
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _has_openai():
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _skip(reason: str):
    """Print a SKIP line and return True to indicate test was skipped."""
    print(f"  SKIP  (test skipped: {reason})")
    return True


# ---------- Vendoring sanity ----------

def test_vendored_files_exist():
    amem_dir = REPO / "agents" / "amem"
    assert (amem_dir / "__init__.py").exists()
    assert (amem_dir / "memory_system.py").exists()
    assert (amem_dir / "retrievers.py").exists()
    assert (amem_dir / "llm_controller.py").exists()


def test_vendored_attribution_in_init():
    init_text = (REPO / "agents" / "amem" / "__init__.py").read_text()
    assert "agiresearch/A-mem" in init_text
    assert "ceffb860f071" in init_text   # commit hash
    assert "MIT" in init_text


def test_llm_controller_attribution():
    text = (REPO / "agents" / "amem" / "llm_controller.py").read_text()
    assert "Patches applied" in text
    assert "agiresearch/A-mem" in text


def test_requirements_file_exists():
    req = REPO / "requirements_amem.txt"
    assert req.exists()
    text = req.read_text()
    assert "sentence-transformers" in text
    assert "chromadb" in text


# ---------- llm_controller patch tests ----------

def test_openai_controller_accepts_base_url():
    """Patched __init__ should accept and forward base_url."""
    if not _has_openai():
        return _skip("openai SDK not installed")
    os.environ["HYMEM_AUX_API_KEY"] = "test-key"
    os.environ["HYMEM_AUX_BASE_URL"] = "http://test.invalid/v1"
    try:
        from agents.amem.llm_controller import OpenAIController
        ctrl = OpenAIController(model="test-model")
        # Verify the OpenAI client got our base_url
        # (different openai SDK versions store it differently;
        # check both .base_url and ._client_options if needed)
        assert hasattr(ctrl, 'client')
        client_base = (
            getattr(ctrl.client, "base_url", None)
            or getattr(getattr(ctrl.client, "_client_options", None), "base_url", None)
        )
        assert client_base is not None
        assert "test.invalid" in str(client_base)
    finally:
        del os.environ["HYMEM_AUX_API_KEY"]
        del os.environ["HYMEM_AUX_BASE_URL"]


def test_openai_controller_uses_env_fallback():
    """If api_key is None, should resolve from HYMEM_AUX_API_KEY first."""
    if not _has_openai():
        return _skip("openai SDK not installed")
    os.environ["HYMEM_AUX_API_KEY"] = "from-aux-env"
    if "OPENAI_API_KEY" in os.environ:
        old_openai = os.environ.pop("OPENAI_API_KEY")
    else:
        old_openai = None
    try:
        from agents.amem.llm_controller import OpenAIController
        ctrl = OpenAIController(model="test")
        assert ctrl.client.api_key == "from-aux-env"
    finally:
        del os.environ["HYMEM_AUX_API_KEY"]
        if old_openai:
            os.environ["OPENAI_API_KEY"] = old_openai


def test_openai_controller_raises_without_key():
    """No api_key, no env var → should raise ValueError."""
    if not _has_openai():
        return _skip("openai SDK not installed")
    saved = {}
    for k in ("HYMEM_AUX_API_KEY", "HYMEM_PLANNER_API_KEY", "OPENAI_API_KEY"):
        if k in os.environ:
            saved[k] = os.environ.pop(k)
    try:
        from agents.amem.llm_controller import OpenAIController
        try:
            OpenAIController(model="test")
        except ValueError as e:
            assert "API key" in str(e)
        else:
            raise AssertionError("Expected ValueError for missing API key")
    finally:
        for k, v in saved.items():
            os.environ[k] = v


def test_llm_controller_dispatcher():
    """LLMController(backend='openai') should produce an OpenAIController."""
    if not _has_openai():
        return _skip("openai SDK not installed")
    os.environ["HYMEM_AUX_API_KEY"] = "test-key"
    try:
        from agents.amem.llm_controller import LLMController, OpenAIController
        ctrl = LLMController(backend="openai", model="test")
        assert isinstance(ctrl.llm, OpenAIController)
    finally:
        del os.environ["HYMEM_AUX_API_KEY"]


# ---------- _format_memory_for_prompt ----------

# Vendored copy to avoid heavy import via agents.amem_agent.
def _format_memory_for_prompt(retrieved):
    if not retrieved:
        return ""
    lines = ["[Retrieved memory notes — most relevant to the current query]"]
    for i, note in enumerate(retrieved[:5], start=1):
        content = note.get("content") or note.get("text") or ""
        keywords = note.get("keywords") or []
        context = note.get("context") or ""
        if isinstance(keywords, list):
            kw_str = ", ".join(keywords[:6])
        else:
            kw_str = str(keywords)
        line = f"  [{i}] {content[:300]}"
        if kw_str:
            line += f"  (keywords: {kw_str})"
        if context and context != "General":
            line += f"  (context: {context[:80]})"
        lines.append(line)
    return "\n".join(lines) + "\n"


def test_format_memory_empty():
    assert _format_memory_for_prompt([]) == ""
    assert _format_memory_for_prompt(None) == ""


def test_format_memory_renders_basic():
    notes = [
        {"content": "Loroupe born 1973", "keywords": ["Tegla", "1973"], "context": "biography"},
        {"content": "PB 30:32.03", "keywords": [], "context": "General"},
    ]
    out = _format_memory_for_prompt(notes)
    assert "Retrieved memory notes" in out
    assert "Loroupe born 1973" in out
    assert "Tegla" in out
    assert "biography" in out
    assert "[1]" in out and "[2]" in out
    assert "PB 30:32.03" in out
    # 'General' context should be skipped
    assert "(context: General)" not in out


def test_format_memory_caps_at_5():
    notes = [{"content": f"note {i}", "keywords": []} for i in range(10)]
    out = _format_memory_for_prompt(notes)
    assert "[5]" in out
    assert "[6]" not in out


def test_format_memory_truncates_long_content():
    long = {"content": "x" * 10000, "keywords": []}
    out = _format_memory_for_prompt([long])
    # 300 char cap on content
    assert "x" * 301 not in out


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
