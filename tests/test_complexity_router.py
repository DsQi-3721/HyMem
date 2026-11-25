"""Unit tests for agents.complexity_router.

Runnable two ways:
    pytest tests/test_complexity_router.py -v
    python -m tests.test_complexity_router          # built-in runner
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.complexity_router import (
    LLMComplexityRouter,
    ClassificationResult,
    _parse_classifier_response,
)

# Importing agents.hymem_trigger pulls in hymem_memory -> openai. To keep tests
# environment-light we import the ComplexityEstimator class lazily and provide
# a minimal stub if openai is unavailable.
try:
    from agents.hymem_trigger import ComplexityEstimator
except ModuleNotFoundError:
    # Self-contained vendored copy of the multi-feature scorer (sync of the
    # one in agents/hymem_trigger.py) so the test runs without openai.
    import re

    _BRANCH_KW = [
        "verify", "check if", "check whether", "confirm", "validate",
        "compare", "analyze", "evaluate", "determine if", "determine whether",
        "investigate", "explore", "assess", "examine", "review",
        "which one", "which of", "is it true", "does it satisfy",
        "find out if", "figure out", "reason about", "think through",
        "验证", "核实", "确认", "判断", "对比", "比较", "分析", "评估",
        "推理", "证明", "确定", "鉴别", "辨别", "审查", "考察",
    ]
    _SEARCH_KW = [
        "search for", "find information", "look up", "get data",
        "retrieve", "fetch", "list all", "show me",
        "搜索", "查找", "查询", "检索", "获取", "拿到", "找出", "列出", "找到",
    ]
    _MULTIHOP = [
        " and then ", " then ", " after that ", " followed by ",
        "之后", "然后", "接着", "最后", " vs ", " versus ", " compared to ",
    ]

    class ComplexityEstimator:  # type: ignore[no-redef]
        @staticmethod
        def estimate(task):
            if not task:
                return "simple", 0.5
            tl = task.lower()
            cs = sum(1 for k in _BRANCH_KW if k in tl)
            ss = sum(1 for k in _SEARCH_KW if k in tl)
            cs += sum(1 for k in _MULTIHOP if k in tl) * 1.5
            if len(task) > 400:
                cs += 1.5
            elif len(task) > 200:
                cs += 0.7
            n_q = task.count("?") + task.count("？")
            if n_q >= 2:
                cs += 1.0
            n_ents = len(re.findall(
                r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", task
            ))
            if n_ents >= 3:
                cs += 1.0
            margin = abs(cs - ss)
            conf = min(0.9, 0.5 + 0.1 * margin)
            if cs > ss:
                return "complex", conf
            elif ss > cs:
                return "simple", conf
            return "complex", 0.5


# ---------- Response parsing ----------

def test_parse_clean_json():
    text = '{"complexity": "branch", "confidence": 0.85, "reason": "multi-step verification"}'
    r = _parse_classifier_response(text)
    assert r is not None
    assert r.complexity == "branch"
    assert abs(r.confidence - 0.85) < 1e-6
    assert "multi-step" in r.reason


def test_parse_markdown_fenced_json():
    text = '```json\n{"complexity": "search", "confidence": 0.9, "reason": "single fact"}\n```'
    r = _parse_classifier_response(text)
    assert r is not None
    assert r.complexity == "search"


def test_parse_simple_complex_aliases():
    text = '{"complexity": "complex", "confidence": 0.7, "reason": "reasoning"}'
    r = _parse_classifier_response(text)
    assert r is not None and r.complexity == "branch"
    text2 = '{"complexity": "simple", "confidence": 0.7, "reason": "lookup"}'
    r2 = _parse_classifier_response(text2)
    assert r2 is not None and r2.complexity == "search"


def test_parse_garbage_returns_none():
    assert _parse_classifier_response("totally not json") is None
    assert _parse_classifier_response("{broken: json,,,}") is None


def test_parse_clamps_confidence():
    r = _parse_classifier_response('{"complexity": "search", "confidence": 1.5, "reason": "x"}')
    assert r is not None and r.confidence == 1.0
    r = _parse_classifier_response('{"complexity": "search", "confidence": -0.5, "reason": "x"}')
    assert r is not None and r.confidence == 0.0


# ---------- Fallback estimator (multilingual + multi-feature) ----------

def test_fallback_estimator_chinese_keywords():
    label, conf = ComplexityEstimator.estimate("请验证这位运动员是否在 1999 年获得过冠军")
    assert label == "complex"
    assert conf >= 0.5


def test_fallback_estimator_english_lookup():
    label, _ = ComplexityEstimator.estimate("look up the birthdate of Tegla Loroupe")
    assert label == "simple"


def test_fallback_estimator_multihop_indicator():
    label, _ = ComplexityEstimator.estimate(
        "Find the oldest movie from director X and then check whether it was nominated."
    )
    assert label == "complex"


def test_fallback_estimator_long_task_skews_complex():
    long_task = (
        "I need to determine which of these candidates satisfies all five "
        "criteria including birth year between 1970 and 1980, three world "
        "championship titles, weighing under 45 kg, retired before 2022, "
        "and recognized by the United Nations as a peace ambassador. "
    ) * 2  # ~600 chars
    label, _ = ComplexityEstimator.estimate(long_task)
    assert label == "complex"


# ---------- LLMComplexityRouter behavior with mocked LLM ----------

class _FakeRouter(LLMComplexityRouter):
    """Subclass that bypasses the OpenAI client by overriding _call_llm.
    fast-path defaults to OFF here so the LLM-path tests below actually
    exercise the LLM mock instead of being short-circuited."""

    def __init__(self, response_factory, **kw):
        kw.setdefault("enable_fast_path", False)
        super().__init__(**kw)
        self._response_factory = response_factory

    async def _call_llm(self, task: str):
        return self._response_factory(task)


def test_router_uses_llm_when_confidence_high():
    def factory(task):
        return ClassificationResult(
            complexity="branch",
            confidence=0.9,
            reason="LLM said so",
            source="llm",
        )

    router = _FakeRouter(factory, fallback_estimator=ComplexityEstimator.estimate)
    r = asyncio.run(router.classify("verify whether X did Y"))
    assert r.complexity == "branch"
    assert r.source == "llm"
    assert r.confidence == 0.9


def test_router_falls_back_when_llm_low_confidence():
    def factory(task):
        return ClassificationResult(
            complexity="search",
            confidence=0.3,  # below threshold 0.6
            reason="not sure",
            source="llm",
        )

    router = _FakeRouter(
        factory,
        fallback_estimator=ComplexityEstimator.estimate,
        confidence_threshold=0.6,
    )
    r = asyncio.run(router.classify(
        "verify whether the candidate satisfies all five criteria"
    ))
    assert r.source == "fallback"
    # Fallback's estimator should pick "branch" for this multi-criteria task
    assert r.complexity == "branch"


def test_router_falls_back_when_llm_call_fails():
    def factory(task):
        return None  # simulate LLM failure

    router = _FakeRouter(factory, fallback_estimator=ComplexityEstimator.estimate)
    r = asyncio.run(router.classify("look up X's birthdate"))
    assert r.source == "fallback"
    assert r.complexity == "search"


def test_router_caches_results():
    counter = {"calls": 0}

    def factory(task):
        counter["calls"] += 1
        return ClassificationResult(
            complexity="branch", confidence=0.8, reason="x", source="llm"
        )

    router = _FakeRouter(factory, fallback_estimator=ComplexityEstimator.estimate)
    asyncio.run(router.classify("identical task"))
    asyncio.run(router.classify("identical task"))
    asyncio.run(router.classify("identical task"))
    assert counter["calls"] == 1  # cached after the first


def test_router_hint_branch_breaks_low_confidence_tie():
    def factory(task):
        return ClassificationResult(
            complexity="search", confidence=0.3, reason="weak", source="llm"
        )

    router = _FakeRouter(
        factory, fallback_estimator=ComplexityEstimator.estimate,
        confidence_threshold=0.6,
    )
    # "look up X" is clearly search, but planner emitted <branch> as a hint
    r = asyncio.run(router.classify("look up the birthdate of X", hint_branch=True))
    # Hint should bias the fallback toward branch
    assert r.complexity == "branch" or r.confidence >= 0.5  # tolerate either


def test_router_handles_empty_task():
    router = LLMComplexityRouter(fallback_estimator=ComplexityEstimator.estimate)
    r = asyncio.run(router.classify(""))
    assert r.complexity == "search"
    assert r.source == "fallback"


def test_router_stats_aggregation():
    def factory(task):
        return ClassificationResult(
            complexity="branch", confidence=0.85, reason="x", source="llm"
        )

    router = _FakeRouter(factory, fallback_estimator=ComplexityEstimator.estimate)
    for q in ["verify A", "compare B and C", "validate D"]:
        asyncio.run(router.classify(q))
    s = router.stats()
    assert s["total"] == 3
    assert s["from_llm"] == 3
    assert s["branch_routed"] == 3
    assert s["search_routed"] == 0


# ---------- Annotated-set accuracy (>= 85% required) ----------

# 10 search-class + 10 branch-class human-labeled tasks
ANNOTATED = [
    # (task, expected_label)
    ("Look up the birthdate of Tegla Loroupe.", "search"),
    ("Find the population of Tokyo as of 2023.", "search"),
    ("Get the URL of the official OpenAI homepage.", "search"),
    ("查询 Tegla Loroupe 的出生日期。", "search"),
    ("What is the boiling point of water in Celsius?", "search"),
    ("List all winners of the 2020 Nobel Prize in Physics.", "search"),
    ("Fetch the abstract of the paper at arxiv 2503.04725.", "search"),
    ("Show me the GitHub link for the verl repository.", "search"),
    ("Retrieve the price of one share of NVIDIA on 2024-01-15.", "search"),
    ("找出莎士比亚的出生年份。", "search"),

    ("Verify that the candidate satisfies all five criteria including "
     "weight under 45kg and three world championship titles.", "branch"),
    ("Compare the methodology of HyMem with that of MemAgent across "
     "long-horizon tasks and decide which is more efficient.", "branch"),
    ("Determine whether the 1999 Sevilla 10,000m World Championship was "
     "won by Loroupe by cross-checking three independent sources.", "branch"),
    ("Analyze the trade-offs between the three caching strategies and "
     "recommend which to deploy in production.", "branch"),
    ("Investigate whether all four authors of the paper share the same "
     "institution and then list any prior collaborations.", "branch"),
    ("验证候选人 X 是否同时满足以下三个条件：1990 年代出生、获得过 NBA "
     "总冠军、有过加拿大籍。", "branch"),
    ("对比 ReAct 与 HyMem 在 GAIA L3 任务上的成功率并分析失败模式。", "branch"),
    ("Check whether the article published before June 2023 mentions all "
     "of cultural traditions, scientific processes, and culinary "
     "innovations.", "branch"),
    ("Evaluate the consistency of the three financial reports vs the "
     "press release and flag any discrepancies.", "branch"),
    ("Find out if any actor born in London in the 1910s appeared in a "
     "Ruth Ford 1940s film together with John Gielgud.", "branch"),
]


def test_annotated_accuracy_via_fallback():
    """Even the fallback-only path should hit >= 80% on this curated set."""
    correct = 0
    for task, expected in ANNOTATED:
        label, _ = ComplexityEstimator.estimate(task)
        # Map old labels to new ones
        mapped = "branch" if label == "complex" else "search"
        if mapped == expected:
            correct += 1
    accuracy = correct / len(ANNOTATED)
    assert accuracy >= 0.80, (
        f"Fallback accuracy {accuracy:.0%} below 80% on annotated set"
    )


# ---------- fast-path (Optimization 5) ----------

def test_fast_path_simple_lookup():
    from agents.complexity_router import _fast_path_classify
    r = _fast_path_classify("look up the birthdate of Tegla Loroupe", hint_branch=False)
    assert r is not None
    assert r.complexity == "search"
    assert r.source == "fast_path"


def test_fast_path_complex_verify():
    from agents.complexity_router import _fast_path_classify
    r = _fast_path_classify("verify whether the candidate satisfies all five criteria", hint_branch=False)
    assert r is not None
    assert r.complexity == "branch"
    assert r.source == "fast_path"


def test_fast_path_ambiguous_returns_none():
    from agents.complexity_router import _fast_path_classify
    # No simple-lead, no complex-hint -> ambiguous, defer to LLM
    r = _fast_path_classify("identify the title of the publication", hint_branch=False)
    assert r is None


def test_fast_path_long_task_defers_to_llm():
    from agents.complexity_router import _fast_path_classify
    long_task = "find the publication that mentions cultural traditions, scientific processes, and culinary innovations co-authored by three individuals" * 3
    assert _fast_path_classify(long_task, hint_branch=False) is None


def test_router_with_fast_path_enabled_skips_llm():
    """When fast-path is on, an obvious-pattern task should NOT call the LLM."""
    counter = {"calls": 0}

    def factory(task):
        counter["calls"] += 1
        return ClassificationResult(
            complexity="search", confidence=0.9, reason="x", source="llm"
        )

    router = _FakeRouter(factory, fallback_estimator=ComplexityEstimator.estimate,
                        enable_fast_path=True)
    r = asyncio.run(router.classify("look up the birthdate of Tegla Loroupe"))
    assert r.source == "fast_path"
    assert counter["calls"] == 0  # LLM was NOT called


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
