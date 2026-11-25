"""Unit tests for THREAD baseline parsing + recursion logic.

Runnable two ways:
    pytest tests/test_thread_agent.py -v
    python -m tests.test_thread_agent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.thread_prompts import (
    SPAWN_OPEN, SPAWN_CLOSE, RESOLVED_OPEN, RESOLVED_CLOSE,
    THREAD_END, THREAD_STOP_TOKENS,
    parse_action, detect_spawn, detect_end,
    extract_spawn_descriptor, extract_final_answer, wrap_child_result,
    find_unresolved_spawn, mark_resolved,
    get_thread_prompt, get_child_prompt,
)


# ---------- Markers + stop tokens ----------

def test_markers_match_upstream():
    """Spawn markers must equal upstream's `=>` and `<=` (verbatim)."""
    assert SPAWN_OPEN == "=>"
    assert SPAWN_CLOSE == "<="
    assert THREAD_END == "#END#"


def test_stop_tokens():
    # We deliberately DROPPED `=>` from stop tokens (was upstream's, but
    # forced LLM to halt mid-spawn-emission). Now only `#EN` halts so
    # complete `=>desc<=` blocks can be emitted in one shot. See bug fix
    # in this commit.
    assert "=>" not in THREAD_STOP_TOKENS, (
        "=> as stop token caused empty descriptors — must remain dropped"
    )
    assert any(s in THREAD_STOP_TOKENS for s in ["#EN", "#END#"])


# ---------- Action parser ----------

def test_parse_search_action():
    text = "Some reasoning.\nACTION: search(Eliud Kipchoge marathon pace)"
    out = parse_action(text)
    assert out is not None
    assert out[0] == "search"
    assert "Eliud Kipchoge" in out[1]


def test_parse_open_page_action():
    text = "ACTION: open_page(https://example.com/foo)"
    out = parse_action(text)
    assert out == ("open_page", "https://example.com/foo")


def test_parse_finish_action():
    text = "ACTION: finish(42)"
    out = parse_action(text)
    assert out == ("finish", "42")


def test_parse_action_takes_latest():
    text = """ACTION: search(first)
OBSERVATION: ...
ACTION: search(second)"""
    out = parse_action(text)
    assert out is not None
    assert out[1] == "second"


def test_parse_no_action():
    assert parse_action("Just thinking, no action.") is None
    assert parse_action("") is None


# ---------- Spawn detection (complete-emission protocol) ----------

def test_detect_spawn_complete_block():
    """=>desc<= is now a COMPLETE emission, not unmatched markers."""
    text = "The author is =>name of the author of Old Man and the Sea<=."
    assert detect_spawn(text) is True


def test_detect_spawn_no_blocks():
    text = "Just plain reasoning, no spawn."
    assert detect_spawn(text) is False


def test_detect_spawn_unmatched_open_returns_false():
    """Unmatched `=>` (truncated mid-emission) is treated as no-spawn-pending —
    we never recurse on partial descriptors."""
    text = "The author is =>Ernest (truncated)"
    assert detect_spawn(text) is False


def test_detect_spawn_already_resolved_skipped():
    """`〈value〉` markers represent ALREADY-RESOLVED spawns and must not
    be re-detected as new spawns."""
    text = "The author is 〈Ernest Hemingway〉, year is 〈1926〉."
    assert detect_spawn(text) is False


def test_detect_spawn_mixed_resolved_and_pending():
    """One resolved `〈〉` + one pending `=>desc<=` → still detect."""
    text = "Year was 〈1926〉. Need to find =>publisher of that novel<=."
    assert detect_spawn(text) is True


# ---------- find_unresolved_spawn / mark_resolved ----------

def test_find_unresolved_returns_offsets_and_descriptor():
    text = "Year was 〈1926〉. Need =>publisher of The Sun Also Rises<=."
    spawn = find_unresolved_spawn(text)
    assert spawn is not None
    start, end, desc = spawn
    assert text[start:end] == "=>publisher of The Sun Also Rises<="
    assert desc == "publisher of The Sun Also Rises"


def test_find_unresolved_returns_none_when_only_resolved():
    text = "All facts known: 〈Hemingway〉 wrote 〈The Sun Also Rises〉 in 〈1926〉."
    assert find_unresolved_spawn(text) is None


def test_find_unresolved_returns_none_when_unmatched():
    """Unclosed `=>` (no `<=`) → not a complete spawn block."""
    text = "Truncated: =>desc without close"
    assert find_unresolved_spawn(text) is None


def test_mark_resolved_replaces_block():
    text = "The author is =>name of author of Old Man and the Sea<=."
    spawn = find_unresolved_spawn(text)
    new_text = mark_resolved(text, spawn[0], spawn[1], "Ernest Hemingway")
    assert "=>" not in new_text or "<=" not in new_text  # original block gone
    assert "〈Ernest Hemingway〉" in new_text


def test_mark_resolved_handles_empty_result():
    text = "The author is =>desc<=."
    spawn = find_unresolved_spawn(text)
    new_text = mark_resolved(text, spawn[0], spawn[1], "")
    assert "[no result]" in new_text


def test_mark_resolved_truncates_long_results():
    text = "The fact is =>desc<=."
    spawn = find_unresolved_spawn(text)
    long_result = "x" * 5000
    new_text = mark_resolved(text, spawn[0], spawn[1], long_result)
    # Should NOT contain the full 5000 chars; truncated to <800 + ellipsis
    assert len(new_text) < 1500


def test_mark_resolved_then_no_more_pending():
    """After resolving the only spawn, find_unresolved_spawn returns None."""
    text = "Need to find =>thing<=."
    spawn = find_unresolved_spawn(text)
    new_text = mark_resolved(text, spawn[0], spawn[1], "thing-value")
    assert find_unresolved_spawn(new_text) is None


# ---------- Final answer ----------

def test_extract_final_answer_simple():
    text = "ACTION: finish(Rockhopper penguin)"
    assert extract_final_answer(text) == "Rockhopper penguin"


def test_extract_final_answer_takes_latest():
    text = "ACTION: finish(wrong)\nACTION: search(more)\nACTION: finish(right)"
    assert extract_final_answer(text) == "right"


def test_extract_final_answer_none():
    assert extract_final_answer("Just text.") is None


# ---------- End detection ----------

def test_detect_end_present():
    assert detect_end("...stuff...\n#END#") is True


def test_detect_end_absent():
    assert detect_end("just text") is False


# ---------- Prompt construction ----------

def test_thread_prompt_includes_question():
    p = get_thread_prompt("What year was X born?")
    assert "What year was X born?" in p
    assert "#START#" in p
    # Must include few-shot examples
    assert "Hemingway" in p


def test_child_prompt_focused_on_descriptor():
    p = get_child_prompt("name of the author")
    assert "name of the author" in p
    assert "#START#" in p


# ---------- Wrap child result (legacy) ----------

def test_wrap_child_result_legacy():
    """Legacy helper still exposed for backwards-compat; new code uses
    mark_resolved() (which produces 〈value〉 instead of value<=)."""
    out = wrap_child_result("42")
    assert out.endswith("<=")
    assert "42" in out


# ---------- Backwards-compat: extract_spawn_descriptor on complete blocks ----------

def test_extract_descriptor_on_complete_block():
    """New-protocol descriptor extraction: returns the contents of the
    first complete `=>desc<=` block."""
    text = "The author is =>name of the author of Old Man and the Sea<=."
    out = extract_spawn_descriptor(text)
    assert out == "name of the author of Old Man and the Sea"


def test_extract_descriptor_empty_when_none():
    assert extract_spawn_descriptor("just text") == ""


def test_extract_descriptor_unmatched_returns_empty():
    """Unmatched `=>` (no closing `<=`) → empty descriptor (not fallback
    to bogus 'unspecified sub-problem' string — that bug caused the L1
    task-1 spawn-with-empty-descriptor failure)."""
    text = "Look up =>X's birthdate (no close)"
    assert extract_spawn_descriptor(text) == ""


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
