"""Figure-1 dilution experiment — Step 2.

Extract key facts per GAIA task from (question, gold_answer) ONLY.
Never reads any agent trace — avoids circular reasoning.

Usage:
    source scripts/env_eval_local.sh
    python scripts/analysis/extract_key_facts.py \
        --data_path /Users/hoi/project/HyMem2026/GAIA/2023/validation/metadata.parquet \
        --levels 1,2 --text_only \
        --out data/key_facts/gaia.jsonl

Schema (one JSON per line):
    {"task_id": "...", "question": "...", "gold_answer": "...",
     "level": 2, "facts": [{"fact": str, "type": str, "evidence_source": str}],
     "extraction": {"attempts": int, "discarded_invalid": int}}

Judge: JUDGE_MODEL (env) via JUDGE_BASE_URL/JUDGE_API_KEY. Temperature 0.
Substring rule: every `fact` MUST be a literal substring of either the
question or the gold answer. Invalid facts get one retry round; if the
re-extraction is also invalid we keep the valid subset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from openai import AsyncOpenAI


PROMPT_TMPL = """You are extracting the minimal set of atomic facts a researcher MUST retain in working memory to verify the GOLD ANSWER from the QUESTION.

QUESTION: {question}
GOLD ANSWER: {gold_answer}

Rules:
1. Each fact MUST be a literal substring of either the question or gold answer.
2. Atomic only: entities, numbers, dates, URLs, quoted strings, proper nouns.
3. NO intermediate reasoning facts. NO synonyms. NO reformulations.
4. 3 <= |facts| <= 10.
5. `type` must be one of {{number, date, url, quoted, proper_noun, other}}.
6. `evidence_source` must be exactly "question" or "gold".
7. Output ONLY a JSON array of {{"fact": str, "type": str, "evidence_source": "question"|"gold"}}. No prose."""


RETRY_PROMPT_TMPL = """Your previous extraction had {n_invalid} fact(s) that were NOT literal substrings of the source. Invalid facts:
{invalid_list}

Re-extract, obeying Rule 1 strictly. Output ONLY the JSON array."""


VALID_TYPES = {"number", "date", "url", "quoted", "proper_noun", "other"}
VALID_SOURCES = {"question", "gold"}


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Pull the first JSON array from a possibly-wrapped LLM response."""
    if not text:
        return []
    # Strip code fences if any
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                  flags=re.MULTILINE)
    # Find first [...] block (greedy across newlines)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _validate_facts(facts: List[Dict[str, Any]], question: str,
                    gold: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split facts into (valid, invalid). A fact is valid iff:
       - has fact/type/evidence_source keys
       - type is in VALID_TYPES
       - evidence_source is 'question' or 'gold'
       - fact is a literal substring of the named source"""
    valid, invalid = [], []
    for f in facts:
        if not isinstance(f, dict):
            invalid.append({"_raw": str(f)[:100], "reason": "not_dict"})
            continue
        fact = (f.get("fact") or "").strip()
        ftype = (f.get("type") or "").strip().lower()
        src = (f.get("evidence_source") or "").strip().lower()
        if not fact or ftype not in VALID_TYPES or src not in VALID_SOURCES:
            invalid.append({"fact": fact, "type": ftype,
                            "evidence_source": src, "reason": "bad_schema"})
            continue
        source_text = question if src == "question" else gold
        if fact not in source_text:
            invalid.append({"fact": fact, "type": ftype,
                            "evidence_source": src, "reason": "not_substring"})
            continue
        valid.append({"fact": fact, "type": ftype, "evidence_source": src})
    return valid, invalid


def _make_client() -> AsyncOpenAI:
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: set JUDGE_API_KEY (or OPENAI_API_KEY) env")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def _call_judge(client: AsyncOpenAI, model: str, prompt: str,
                      max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  [judge] failed after {max_retries}: {e}")
                return ""
            await asyncio.sleep(1.5 * (attempt + 1))
    return ""


async def extract_for_task(client: AsyncOpenAI, model: str,
                           task_id: str, question: str, gold: str
                           ) -> Dict[str, Any]:
    prompt = PROMPT_TMPL.format(question=question, gold_answer=gold)
    raw = await _call_judge(client, model, prompt)
    facts = _extract_json_array(raw)
    valid, invalid = _validate_facts(facts, question, gold)

    attempts = 1
    discarded = len(invalid)

    # Retry once if we have any invalid facts AND < 3 valid
    if invalid and len(valid) < 3:
        invalid_list = "\n".join(
            f"- {repr(i.get('fact', i.get('_raw', '')))} "
            f"(reason: {i.get('reason')})"
            for i in invalid[:5]
        )
        retry_prompt = (
            prompt + "\n\n" +
            RETRY_PROMPT_TMPL.format(n_invalid=len(invalid),
                                      invalid_list=invalid_list)
        )
        raw2 = await _call_judge(client, model, retry_prompt)
        facts2 = _extract_json_array(raw2)
        valid2, invalid2 = _validate_facts(facts2, question, gold)
        attempts = 2
        if len(valid2) > len(valid):
            valid, invalid = valid2, invalid2
            discarded = len(invalid)

    # Dedupe valid facts on (fact, type, src)
    seen = set()
    deduped = []
    for f in valid:
        key = (f["fact"], f["type"], f["evidence_source"])
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    return {
        "task_id": task_id,
        "question": question,
        "gold_answer": gold,
        "facts": deduped,
        "extraction": {"attempts": attempts, "discarded_invalid": discarded},
    }


async def main_async(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.data_path)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else None

    rows = []
    for _, r in df.iterrows():
        if levels and int(r.get("Level", 0)) not in levels:
            continue
        if args.text_only and r.get("file_name") and not pd.isna(r.get("file_name")):
            continue
        rows.append({
            "task_id": r.get("task_id"),
            "question": r.get("Question", ""),
            "gold_answer": r.get("Final answer", ""),
            "level": int(r.get("Level", 0)),
        })

    if args.limit:
        rows = rows[: args.limit]
    print(f"[extract_key_facts] {len(rows)} tasks (levels={args.levels}, "
          f"text_only={args.text_only})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip task_ids already in output file
    done_ids = set()
    if args.resume and out_path.exists():
        for ln in out_path.open():
            try:
                done_ids.add(json.loads(ln)["task_id"])
            except Exception:
                pass
        print(f"[extract_key_facts] resume: {len(done_ids)} already done")

    client = _make_client()
    model = os.getenv("JUDGE_MODEL", "deepseek-reasoner")
    print(f"[extract_key_facts] judge model: {model}")

    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()

    async def _one(row):
        async with sem:
            if row["task_id"] in done_ids:
                return None
            out = await extract_for_task(
                client, model, row["task_id"],
                row["question"], row["gold_answer"],
            )
            out["level"] = row["level"]
            return out

    written = 0
    with out_path.open("a") as fout:
        for fut in asyncio.as_completed([_one(r) for r in rows]):
            res = await fut
            if res is None:
                continue
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if written % 10 == 0 or written < 5:
                print(f"  [{written}/{len(rows)}] {res['task_id']}  "
                      f"facts={len(res['facts'])}  "
                      f"discarded={res['extraction']['discarded_invalid']}  "
                      f"attempts={res['extraction']['attempts']}")

    elapsed = time.time() - t0
    print(f"\n[extract_key_facts] wrote {written} entries to {out_path} "
          f"in {elapsed:.1f}s")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True,
                   help="GAIA metadata.parquet path")
    p.add_argument("--levels", default="1,2",
                   help='Comma-separated level filter, e.g. "1,2"')
    p.add_argument("--text_only", action="store_true",
                   help="Exclude tasks with file attachments")
    p.add_argument("--out", default="data/key_facts/gaia.jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap task count (for smoke tests)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Parallel judge calls")
    p.add_argument("--resume", action="store_true",
                   help="Skip task_ids already present in --out")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
