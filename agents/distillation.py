"""
Information Distillation pipeline (HyMem paper Sec 3.2.2).

Three-stage pipeline that addresses two prior issues:
  (1) hard truncation at 2000 chars used to drop tail content
  (2) keeping every regex-extracted entity floods the Actor with task-irrelevant noise

Stages:
  Stage 1 (PreservationFilter): rule-based extraction of candidate critical facts
                                (numbers, dates, URLs, quoted entities, capitalized
                                NER) from the FULL raw content, BEFORE any truncation.
  Stage 2 (chunked_filter):     split full content into overlapping chunks; run the
                                existing yes/no/unknown LLM relevance check on each
                                chunk; concatenate kept chunks. Replaces hard truncation.
  Stage 3 (task_relevance_gate): one batch LLM call decides which Stage-1 candidate
                                facts are actually task-relevant; only those are
                                appended to the final filtered output as "Verified
                                facts".

Reuses HYMEM_AUX_MODEL / HYMEM_EXECUTOR_* env vars — no new endpoint configuration.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Stage 1: PreservationFilter
# ---------------------------------------------------------------------------

CandidateType = str  # "number" | "date" | "url" | "quoted" | "proper_noun"


@dataclass
class CandidateFact:
    text: str
    type: CandidateType
    position: int  # offset in raw content


# Regex patterns. Compiled once at import.
# IMPORTANT: tried in order; later patterns skip ranges already covered by
# earlier ones (see PreservationFilter.extract — offset-overlap guard).
_NUMBER_PATTERNS = [
    # 1) Unit-bearing measurements: "40kg", "10000m", "5 km", "100ms"
    re.compile(
        r"\b\d+(?:\.\d+)?\s?(?:kg|km|cm|mm|lb|oz|ft|in|ms|hz|mhz|ghz|m|g)(?=\b|[\s.,;:!?]|$)",
        re.IGNORECASE,
    ),
    # 2) Sport / clock time: "30:32.03", "1:23:45"
    re.compile(r"\b\d{1,2}:\d{2}(?:[:.]\d{1,3})?\b"),
    # 3) Currency / suffix-magnitude: "$1.2B", "5K", "12.5M"
    re.compile(r"\$\d+(?:,\d{3})*(?:\.\d+)?[KMB]?(?=[^\w]|$)"),
    re.compile(r"\b\d+(?:\.\d+)?[KMB](?=[^\w]|$)"),
    # 4) Percent: "2.5%", "100%"
    re.compile(r"\b\d+(?:\.\d+)?%(?=[^\w]|$)"),
    # 5) Comma-grouped numbers: "1,234", "1,234,567.89"
    re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"),
    # 6) Plain numbers — must NOT be adjacent to digit/dot (avoids re-capturing
    #    fragments of comma-numbers or decimals already matched above).
    re.compile(r"(?<![\d.,])\d+(?:\.\d+)?(?![\d.,])"),
]

_DATE_PATTERNS = [
    # 2024-01-15 / 2024/1/15 / 2024年1月15日
    re.compile(r"\b\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?\b"),
    # 1 January 2024 / January 1, 2024 / Jan. 1 2024
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b"
    ),
    re.compile(
        r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b"
    ),
    # 2024 alone is too noisy; pair year with quarter / decade markers
    re.compile(r"\b(?:Q[1-4]|H[12])\s?\d{4}\b"),
    re.compile(r"\b\d{4}s\b"),  # "1990s"
]

_URL_PATTERN = re.compile(r"https?://\S+")

_QUOTED_PATTERNS = [
    re.compile(r'"([^"\n]{2,80})"'),
    re.compile(r"'([^'\n]{2,80})'"),
    re.compile(r"《([^》\n]+)》"),
    re.compile(r"「([^」\n]+)」"),
]

# Sequences of capitalized words: "Tegla Loroupe", "United Nations"
_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")

# Words that are technically capitalized but rarely meaningful as entities.
_PROPER_NOUN_STOPWORDS = {
    "The", "This", "That", "These", "Those", "However", "Therefore",
    "Although", "Because", "Indeed", "Moreover", "Furthermore", "Note",
    "See", "From", "Such", "While", "When", "Where", "Why", "What", "Which",
    "Who", "Whom", "Whose", "How", "If", "And", "Or", "But", "So", "Then",
}


class PreservationFilter:
    """Stage 1 — rule-based candidate fact extraction over the FULL raw content."""

    @staticmethod
    def extract(raw_content: str, max_per_type: int = 80) -> List[CandidateFact]:
        if not raw_content:
            return []

        candidates: List[CandidateFact] = []
        seen: set = set()  # de-dup by (text_lower, type)
        # Track byte ranges captured per type, to suppress later patterns from
        # re-extracting fragments of an already-captured span (e.g. "1" out of
        # "1,234,567", or "30:32" out of "30:32.03").
        covered_by_type: dict = {"number": [], "date": []}

        def overlaps(start: int, end: int, ranges: list) -> bool:
            for s, e in ranges:
                if start < e and end > s:
                    return True
            return False

        def add(text: str, type_: CandidateType, pos: int, end: int) -> None:
            key = (text.lower(), type_)
            if key in seen:
                return
            seen.add(key)
            candidates.append(CandidateFact(text=text, type=type_, position=pos))
            if type_ in covered_by_type:
                covered_by_type[type_].append((pos, end))

        # Numbers (patterns tried in order; later ones skip ranges already
        # covered by earlier ones).
        for pat in _NUMBER_PATTERNS:
            for m in pat.finditer(raw_content):
                if overlaps(m.start(), m.end(), covered_by_type["number"]):
                    continue
                add(m.group(0).strip(), "number", m.start(), m.end())

        # Dates
        for pat in _DATE_PATTERNS:
            for m in pat.finditer(raw_content):
                if overlaps(m.start(), m.end(), covered_by_type["date"]):
                    continue
                add(m.group(0).strip(), "date", m.start(), m.end())

        # URLs
        for m in _URL_PATTERN.finditer(raw_content):
            url = m.group(0).rstrip(").,;:!?\"'")
            add(url, "url", m.start(), m.start() + len(url))

        # Quoted strings
        for pat in _QUOTED_PATTERNS:
            for m in pat.finditer(raw_content):
                inner = m.group(1).strip()
                if inner:
                    add(inner, "quoted", m.start(), m.end())

        # Proper-noun sequences (light-weight NER)
        for m in _PROPER_NOUN_PATTERN.finditer(raw_content):
            text = m.group(0).strip()
            if text in _PROPER_NOUN_STOPWORDS:
                continue
            words = text.split()
            if len(words) == 1:
                # Heuristic: drop single-word capitals at sentence starts —
                # these are usually just "The"/"After"/etc. that pass the
                # stopword filter due to dictionary gaps.
                start = m.start()
                preceding = raw_content[max(0, start - 2):start]
                if preceding in ("", ". ", "! ", "? ", "\n"):
                    continue
            add(text, "proper_noun", m.start(), m.end())

        # Cap per-type to avoid candidate explosion on huge docs.
        by_type: dict = {}
        for c in candidates:
            by_type.setdefault(c.type, []).append(c)
        capped: List[CandidateFact] = []
        for type_, lst in by_type.items():
            capped.extend(lst[:max_per_type])
        capped.sort(key=lambda c: c.position)
        return capped


# ---------------------------------------------------------------------------
# Stage 2: chunked_filter
# ---------------------------------------------------------------------------

# Type alias for the relevance-checker callable. It mirrors the public signature
# of `bm_tools.check_condition` so we do not import that module here (avoids
# circular import; the caller passes a bound function).
RelevanceChecker = Callable[[str, str], Awaitable[dict]]


def _split_into_chunks(content: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [content]
    if overlap < 0 or overlap >= chunk_size:
        overlap = max(0, chunk_size // 10)
    chunks: List[str] = []
    n = len(content)
    if n <= chunk_size:
        return [content]
    step = chunk_size - overlap
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(content[start:end])
        if end == n:
            break
        start += step
    return chunks


async def chunked_filter(
    raw_content: str,
    task: str,
    relevance_checker: RelevanceChecker,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_concurrent: int = 8,
) -> Tuple[str, dict]:
    """Stage 2 — split content into overlapping chunks, keep chunks judged
    relevant (or unknown) by `relevance_checker`. Replaces hard truncation.

    Optimization (O3): chunk relevance checks fan out concurrently with a
    semaphore (default 8 — was 4). For an 8-chunk page this drops ~16s
    serial to ~2s, with no extra LLM tokens. Override via env var
    HYMEM_DISTILL_CHUNK_CONCURRENCY=N.

    Returns (filtered_main, stats) where stats has:
      {chunks_total, chunks_kept, chunks_dropped, max_concurrent_used}
    """
    chunks = _split_into_chunks(raw_content, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return "", {"chunks_total": 0, "chunks_kept": 0, "chunks_dropped": 0,
                    "max_concurrent_used": 0}

    # O3: env override allows operator to dial down on rate-limited APIs
    # without re-deploying. Caller-supplied max_concurrent is the default.
    try:
        env_concurrency = int(os.getenv("HYMEM_DISTILL_CHUNK_CONCURRENCY", "0"))
    except ValueError:
        env_concurrency = 0
    effective_concurrent = env_concurrency if env_concurrency > 0 else max_concurrent
    sem = asyncio.Semaphore(effective_concurrent)

    async def _check(chunk: str) -> str:
        async with sem:
            try:
                result = await relevance_checker(chunk, task)
                if isinstance(result, dict):
                    return result.get("tool_result", {}).get("is_relevant", "unknown")
            except Exception:
                pass
            return "unknown"

    verdicts = await asyncio.gather(*[_check(c) for c in chunks])

    kept: List[str] = []
    dropped = 0
    for chunk, verdict in zip(chunks, verdicts):
        if verdict == "no":
            dropped += 1
        else:  # yes or unknown — keep (conservative)
            kept.append(chunk)

    filtered_main = "\n\n[chunk separator]\n\n".join(kept)
    stats = {
        "chunks_total": len(chunks),
        "chunks_kept": len(kept),
        "chunks_dropped": dropped,
        "max_concurrent_used": effective_concurrent,
    }
    return filtered_main, stats


# ---------------------------------------------------------------------------
# Stage 3: task_relevance_gate
# ---------------------------------------------------------------------------

# Caller passes a bound function with the same signature as
# `bm_tools.call_llm` (prompt -> str). Avoids hard-coupling at import time.
LLMCaller = Callable[[str], Awaitable[str]]


def _format_candidates_for_gate(candidates: List[CandidateFact]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        # Truncate very long candidate text for prompt economy.
        text = c.text if len(c.text) <= 120 else c.text[:117] + "..."
        lines.append(f"[{i}] ({c.type}) {text}")
    return "\n".join(lines)


def _parse_relevant_indices(response: str, n_candidates: int) -> List[int]:
    # Try to grab a JSON object with a "relevant_indices" array.
    m = re.search(r'\{[^{}]*"relevant_indices"\s*:\s*\[([^\]]*)\][^{}]*\}', response)
    if m:
        body = m.group(1)
    else:
        # Bare array fallback
        m = re.search(r"\[([\d,\s]+)\]", response)
        if not m:
            return []
        body = m.group(1)
    try:
        nums = [int(x.strip()) for x in body.split(",") if x.strip().isdigit()]
    except Exception:
        return []
    return [n for n in nums if 1 <= n <= n_candidates]


async def task_relevance_gate(
    candidates: List[CandidateFact],
    task: str,
    llm_caller: LLMCaller,
    max_candidates_per_call: int = 60,
    max_concurrent_batches: int = 4,
) -> List[CandidateFact]:
    """Stage 3 — one (or few) batch LLM calls to filter candidates by task
    relevance. Conservative fallback on parse failure: keep all `number` /
    `date` candidates (the most paper-critical types).

    O2: batches dispatch CONCURRENTLY (semaphore-bounded) instead of
    serially. For 200-candidate documents this saves ~10-15s.
    Disabled if env HYMEM_DISTILL_GATE_PARALLEL=0 (falls back to serial).
    """
    if not candidates:
        return []

    # Build all batches upfront, with their absolute slice for ordering.
    batches: List[List[CandidateFact]] = []
    for batch_start in range(0, len(candidates), max_candidates_per_call):
        batches.append(candidates[batch_start:batch_start + max_candidates_per_call])

    parallel_enabled = os.getenv("HYMEM_DISTILL_GATE_PARALLEL", "1") != "0"

    async def _process_batch(batch: List[CandidateFact]) -> List[CandidateFact]:
        prompt = (
            f"You are filtering candidate facts for task-relevance.\n\n"
            f"Task: {task}\n\n"
            f"Below are facts extracted from a source document. "
            f"List the indices of facts likely relevant to answering the task.\n\n"
            f"{_format_candidates_for_gate(batch)}\n\n"
            f'Respond ONLY in JSON: {{"relevant_indices": [1, 3, ...]}}'
        )
        try:
            response = await llm_caller(prompt)
            indices = _parse_relevant_indices(response, len(batch))
            if not indices:
                return [c for c in batch if c.type in ("number", "date")]
            return [batch[i - 1] for i in indices]
        except Exception:
            return [c for c in batch if c.type in ("number", "date")]

    if parallel_enabled and len(batches) > 1:
        sem = asyncio.Semaphore(max_concurrent_batches)

        async def _gated(b: List[CandidateFact]) -> List[CandidateFact]:
            async with sem:
                return await _process_batch(b)

        # Order is preserved by gather over a list comprehension.
        per_batch_results = await asyncio.gather(*[_gated(b) for b in batches])
    else:
        per_batch_results = []
        for b in batches:
            per_batch_results.append(await _process_batch(b))

    # Flatten in batch-emit order (matches the original serial behavior).
    relevant: List[CandidateFact] = []
    for r in per_batch_results:
        relevant.extend(r)
    return relevant


# ---------------------------------------------------------------------------
# Pipeline glue
# ---------------------------------------------------------------------------

@dataclass
class DistillationStats:
    chunks_total: int = 0
    chunks_kept: int = 0
    chunks_dropped: int = 0
    candidates_extracted: int = 0
    candidates_kept_after_gate: int = 0
    total_aux_calls: int = 0  # chunked_filter calls + gate calls (rough estimate)
    # O3: actual semaphore width used in chunked_filter (env-overridable)
    chunk_concurrent_used: int = 0

    def to_dict(self) -> dict:
        return {
            "chunks_total": self.chunks_total,
            "chunks_kept": self.chunks_kept,
            "chunks_dropped": self.chunks_dropped,
            "candidates_extracted": self.candidates_extracted,
            "candidates_kept_after_gate": self.candidates_kept_after_gate,
            "total_aux_calls": self.total_aux_calls,
            "chunk_concurrent_used": self.chunk_concurrent_used,
        }


async def run_distillation_pipeline(
    raw_content: str,
    task: str,
    relevance_checker: RelevanceChecker,
    llm_caller: LLMCaller,
    chunk_size: int = 1500,
    overlap: int = 200,
    max_candidates_per_call: int = 60,
) -> Tuple[str, DistillationStats]:
    """Run all three stages and return the final filtered content + stats.

    The output format is:

        <filtered chunks separated by [chunk separator]>

        --- Verified facts (preserved via rule-based extraction +
        task-relevance filter): ---
        - fact 1
        - fact 2
        ...

    If raw_content is empty or short enough to skip chunking, behavior degrades
    gracefully (Stage 1 still runs, Stage 2 returns the content as-is, Stage 3
    still runs).
    """
    stats = DistillationStats()
    if not raw_content:
        return "", stats

    # Stage 1
    candidates = PreservationFilter.extract(raw_content)
    stats.candidates_extracted = len(candidates)

    # Stage 2
    filtered_main, chunk_stats = await chunked_filter(
        raw_content, task, relevance_checker,
        chunk_size=chunk_size, overlap=overlap,
    )
    stats.chunks_total = chunk_stats["chunks_total"]
    stats.chunks_kept = chunk_stats["chunks_kept"]
    stats.chunks_dropped = chunk_stats["chunks_dropped"]
    stats.chunk_concurrent_used = chunk_stats.get("max_concurrent_used", 0)
    stats.total_aux_calls += stats.chunks_total

    # Stage 3
    relevant_facts = await task_relevance_gate(
        candidates, task, llm_caller,
        max_candidates_per_call=max_candidates_per_call,
    )
    stats.candidates_kept_after_gate = len(relevant_facts)
    # rough estimate: 1 LLM call per `max_candidates_per_call` batch
    if candidates:
        n_batches = (len(candidates) + max_candidates_per_call - 1) // max_candidates_per_call
        stats.total_aux_calls += n_batches

    # Compose final output.
    if relevant_facts:
        facts_block = (
            "\n\n--- Verified facts (preserved via rule-based extraction + "
            "task-relevance filter): ---\n"
            + "\n".join(f"- [{f.type}] {f.text}" for f in relevant_facts)
        )
    else:
        facts_block = ""

    return filtered_main + facts_block, stats
