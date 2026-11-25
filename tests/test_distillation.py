"""Unit tests for agents.distillation.

Runnable two ways:
    pytest tests/test_distillation.py -v
    python -m tests.test_distillation        # uses built-in runner, no pytest dep
"""

import asyncio
import sys
from pathlib import Path

# Allow running this file directly without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.distillation import (
    PreservationFilter,
    chunked_filter,
    task_relevance_gate,
    run_distillation_pipeline,
)


# ---------- Stage 1: PreservationFilter ----------

def test_stage1_extracts_numbers_and_units():
    text = (
        "The athlete weighed 40kg and ran 10000m in 30:32.03. "
        "Her PB improved by 2.5% over $1.2B in funding 1,234,567 dollars."
    )
    facts = PreservationFilter.extract(text)
    texts = [f.text for f in facts]
    # Must catch the unit-bearing measurements and the time format
    assert any("40kg" in t for t in texts)
    assert any("10000m" in t for t in texts)
    assert any("30:32.03" in t for t in texts)
    assert any("2.5%" in t for t in texts)
    assert any("1,234,567" in t for t in texts)


def test_stage1_extracts_dates_multiple_formats():
    text = (
        "She was born May 9, 1973 in Kenya. The race was 26 August 1999. "
        "Her career peaked in the 1990s. The 2024年1月15日 ceremony was Q3 2024."
    )
    facts = PreservationFilter.extract(text)
    types = {f.type for f in facts}
    texts = [f.text for f in facts]
    assert "date" in types
    assert any("1973" in t for t in texts)
    assert any("26 August 1999" in t or "August" in t for t in texts)
    assert any("1990s" in t for t in texts)
    assert any("2024年1月15日" in t for t in texts)
    assert any("Q3 2024" in t for t in texts)


def test_stage1_extracts_urls_and_quoted():
    text = (
        'The source is at https://example.com/page?id=42, and the song is '
        '"Bohemian Rhapsody" while the book is 《红楼梦》.'
    )
    facts = PreservationFilter.extract(text)
    types = {f.type for f in facts}
    texts = [f.text for f in facts]
    assert "url" in types
    assert any("https://example.com/page?id=42" in t for t in texts)
    assert any("Bohemian Rhapsody" in t for t in texts)
    assert any("红楼梦" in t for t in texts)


def test_stage1_extracts_proper_nouns_but_skips_sentence_starts():
    text = "Tegla Loroupe won the New York Marathon. The United Nations honored her."
    facts = PreservationFilter.extract(text)
    texts = [f.text for f in facts]
    # Multi-word proper nouns should be captured
    assert any("Tegla Loroupe" in t for t in texts)
    assert any("New York Marathon" in t for t in texts)
    assert any("United Nations" in t for t in texts)
    # Single-word "The" at sentence start should be dropped (stopword)
    assert "The" not in texts


def test_stage1_dedups_repeats():
    text = "1973 happened. Then 1973 again. And once more 1973."
    facts = PreservationFilter.extract(text)
    texts = [f.text for f in facts]
    assert texts.count("1973") <= 1  # de-duplicated


def test_stage1_recall_on_mixed_corpus():
    """Aggregated recall sanity check — must catch at least 5 facts here."""
    text = (
        "On 26 August 1999, Tegla Loroupe set a 30:32.03 personal best in the "
        "10,000m at the World Championships in Sevilla. She earned roughly $1.2B "
        "in endorsements over her career, see https://athletics.org/loroupe."
    )
    facts = PreservationFilter.extract(text)
    assert len(facts) >= 5  # number/date/url/proper_noun mix


# ---------- Stage 2: chunked_filter ----------

def _make_relevance_checker(chunk_verdicts):
    """Build a fake checker that returns verdicts in order."""
    state = {"i": 0}

    async def checker(content: str, task: str):
        i = state["i"]
        state["i"] = (i + 1) % len(chunk_verdicts)
        return {"tool_result": {"is_relevant": chunk_verdicts[i], "reasoning": ""}}

    return checker


def test_stage2_keeps_only_relevant_chunks():
    """Tail-relevant chunk must NOT be dropped (the bug we are fixing)."""
    # 4500 chars => 3 chunks at 1500
    content = "X" * 1500 + "Y" * 1500 + "Z" * 1500
    checker = _make_relevance_checker(["no", "no", "yes"])

    filtered, stats = asyncio.run(
        chunked_filter(content, "task", checker, chunk_size=1500, overlap=0)
    )
    assert stats["chunks_total"] == 3
    assert stats["chunks_kept"] == 1
    assert stats["chunks_dropped"] == 2
    assert "Z" * 100 in filtered  # tail content survived
    assert "X" * 100 not in filtered


def test_stage2_unknown_is_kept_conservatively():
    content = "A" * 1500 + "B" * 1500
    checker = _make_relevance_checker(["unknown", "no"])
    filtered, stats = asyncio.run(
        chunked_filter(content, "task", checker, chunk_size=1500, overlap=0)
    )
    assert stats["chunks_kept"] == 1
    assert "A" * 100 in filtered  # unknown was kept
    assert "B" * 100 not in filtered


def test_stage2_short_content_single_chunk():
    content = "short content"
    checker = _make_relevance_checker(["yes"])
    filtered, stats = asyncio.run(
        chunked_filter(content, "task", checker, chunk_size=1500)
    )
    assert stats["chunks_total"] == 1
    assert filtered == "short content"


# ---------- Stage 3: task_relevance_gate ----------

def _make_llm_caller(response: str):
    async def caller(prompt: str) -> str:
        return response

    return caller


def test_stage3_filters_to_relevant_indices():
    from agents.distillation import CandidateFact

    candidates = [
        CandidateFact("1973", "date", 0),
        CandidateFact("Tegla Loroupe", "proper_noun", 10),
        CandidateFact("Bohemian Rhapsody", "quoted", 20),  # irrelevant noise
    ]
    caller = _make_llm_caller('{"relevant_indices": [1, 2]}')
    relevant = asyncio.run(task_relevance_gate(candidates, "task", caller))
    texts = [f.text for f in relevant]
    assert "1973" in texts
    assert "Tegla Loroupe" in texts
    assert "Bohemian Rhapsody" not in texts


def test_stage3_fallback_keeps_numbers_and_dates_on_parse_failure():
    from agents.distillation import CandidateFact

    candidates = [
        CandidateFact("1973", "date", 0),
        CandidateFact("Tegla Loroupe", "proper_noun", 10),
        CandidateFact("$1.2B", "number", 20),
    ]
    caller = _make_llm_caller("garbage non-json response")
    relevant = asyncio.run(task_relevance_gate(candidates, "task", caller))
    types = {f.type for f in relevant}
    # Conservative fallback keeps date + number; drops proper_noun
    assert "date" in types
    assert "number" in types
    assert "proper_noun" not in types


# ---------- End-to-end pipeline ----------

def test_pipeline_end_to_end():
    """Full three-stage run on synthetic content."""
    content = (
        "Tegla Loroupe was born on May 9, 1973 in Kenya. " * 30 +
        "Random unrelated information that should be filtered. " * 30 +
        "Her 10,000m PB of 30:32.03 was set on 26 August 1999 at the World Championships. " * 30
    )

    chunk_verdicts = ["yes", "no", "yes"]
    checker = _make_relevance_checker(chunk_verdicts)

    # Gate keeps date + number facts
    caller = _make_llm_caller('{"relevant_indices": [1, 2, 3, 4, 5, 6]}')

    final, stats = asyncio.run(
        run_distillation_pipeline(
            content, "When did Loroupe set her 10,000m PB?",
            relevance_checker=checker,
            llm_caller=caller,
            chunk_size=1500,
            overlap=0,
        )
    )
    assert stats.candidates_extracted > 0
    assert stats.chunks_total >= 1
    assert "Verified facts" in final  # gate appended a fact block


# ---------- Built-in runner (no pytest required) ----------

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
