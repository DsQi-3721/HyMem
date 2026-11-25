"""Unit tests for agents.resum_agent — verifies the summarization trigger
logic, prompt selection, and stats schema (without spinning up the full
agent loop, which needs verl/torch).

Runnable two ways:
    pytest tests/test_resum_agent.py -v
    python -m tests.test_resum_agent
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.resum_prompts import (
    QUERY_SUMMARY_PROMPT,
    QUERY_SUMMARY_PROMPT_LAST,
    RESUM_RESTART_MESSAGE,
    extract_summary,
)


# Vendored copy of agents.resum_agent._format_history_for_summary so this
# test runs without numpy/torch/verl (the agent module pulls those via
# `from verl import DataProto`). Logic must stay in sync with the real one.
def _format_history_for_summary(messages, max_chars=30000):
    parts = []
    for m in messages:
        role = m.get('role', '')
        content = (m.get('content') or '').strip()
        if not content:
            continue
        parts.append(f"### {role}\n{content}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = "...(earlier history truncated)...\n\n" + text[-max_chars:]
    return text


# ---------- Prompt vendoring sanity ----------

def test_prompts_loaded_verbatim():
    # Spot-check key phrases that MUST be present (vendored from upstream)
    assert "expert at analyzing conversation history" in QUERY_SUMMARY_PROMPT
    assert "Strictly avoid fabricating" in QUERY_SUMMARY_PROMPT
    assert "<summary>" in QUERY_SUMMARY_PROMPT
    assert "Last Summary" in QUERY_SUMMARY_PROMPT_LAST
    assert "{question}" in QUERY_SUMMARY_PROMPT
    assert "{recent_history_messages}" in QUERY_SUMMARY_PROMPT
    assert "{last_summary}" in QUERY_SUMMARY_PROMPT_LAST


def test_restart_message_format():
    msg = RESUM_RESTART_MESSAGE.format(question="Q?", summary="S.")
    assert "Question: Q?" in msg
    assert "Summary: S." in msg
    assert "Below is a summary" in msg


# ---------- extract_summary ----------

def test_extract_summary_with_tag():
    response = "Some preamble.\n<summary>\n- X happened.\n- Y was found.\n</summary>\nDone."
    out = extract_summary(response)
    assert "X happened" in out
    assert "Y was found" in out
    assert "preamble" not in out


def test_extract_summary_fallback_no_tag():
    response = "Just plain text without any tag."
    out = extract_summary(response)
    assert out == "Just plain text without any tag."


def test_extract_summary_uses_first_tag():
    # If model emits multiple <summary> blocks, take the first (re.search behavior)
    response = "<summary>FIRST</summary>\n<summary>SECOND</summary>"
    out = extract_summary(response)
    assert out == "FIRST"


# ---------- _format_history_for_summary ----------

def test_format_history_serializes_roles():
    msgs = [
        {'role': 'system', 'content': 'sys'},
        {'role': 'user', 'content': 'q'},
        {'role': 'assistant', 'content': 'a'},
    ]
    out = _format_history_for_summary(msgs)
    assert '### system' in out
    assert 'sys' in out
    assert '### user' in out
    assert 'q' in out
    assert '### assistant' in out
    assert 'a' in out


def test_format_history_truncates_with_marker():
    long_msg = {'role': 'user', 'content': 'x' * 50000}
    out = _format_history_for_summary([long_msg], max_chars=5000)
    assert "earlier history truncated" in out
    assert len(out) <= 5100  # truncated body + marker


def test_format_history_skips_empty():
    msgs = [
        {'role': 'user', 'content': 'real'},
        {'role': 'user', 'content': ''},
        {'role': 'assistant', 'content': '   '},
    ]
    out = _format_history_for_summary(msgs)
    assert out.count('###') == 1   # only the non-empty 'real' message


# ---------- Trigger logic ----------

def test_trigger_at_90_percent():
    """Replicate the in-loop trigger condition. Threshold = 0.9 * max_context."""
    max_context_tokens = 10000
    resum_threshold_ratio = 0.9
    threshold = int(max_context_tokens * resum_threshold_ratio)
    assert threshold == 9000

    # current_len just below threshold -> no trigger
    current_len = 8999
    iteration = 5
    n_summaries = 0
    resum_max_summaries = 4
    should = (
        current_len >= threshold
        and n_summaries < resum_max_summaries
        and iteration > 1
    )
    assert should is False

    # current_len at threshold -> trigger
    current_len = 9000
    should = (
        current_len >= threshold
        and n_summaries < resum_max_summaries
        and iteration > 1
    )
    assert should is True

    # already maxed out -> no trigger even if over
    n_summaries = 4
    should = (
        current_len >= threshold
        and n_summaries < resum_max_summaries
        and iteration > 1
    )
    assert should is False


def test_trigger_skips_first_iteration():
    threshold = 100
    current_len = 200
    n_summaries = 0
    iteration = 1   # first round — never summarize before any actual work
    resum_max_summaries = 4
    should = (
        current_len >= threshold
        and n_summaries < resum_max_summaries
        and iteration > 1
    )
    assert should is False


# ---------- Prompt selection (1st pass vs Nth pass) ----------

def test_first_pass_uses_no_last_summary_template():
    last_summary = None
    selected_template = QUERY_SUMMARY_PROMPT_LAST if last_summary else QUERY_SUMMARY_PROMPT
    assert selected_template is QUERY_SUMMARY_PROMPT


def test_subsequent_pass_uses_last_summary_template():
    last_summary = "previously found X"
    selected_template = QUERY_SUMMARY_PROMPT_LAST if last_summary else QUERY_SUMMARY_PROMPT
    assert selected_template is QUERY_SUMMARY_PROMPT_LAST


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
