"""O5: executor diminishing-returns detector.

Standalone module so unit tests can import the detection logic without
pulling in heavy deps (torch, verl, openai). Used by HyMemAgent.run_executor.
"""

import re
from typing import List, Optional


def _is_low_info_obs(s: str) -> bool:
    """Short response (<200 chars) with neither digits nor a Capitalized
    multi-word entity. Used as a 'no useful signal' heuristic."""
    if not s or len(s.strip()) >= 200:
        return False
    t = s.strip()
    has_digit = any(ch.isdigit() for ch in t)
    has_entity = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", t))
    return not (has_digit or has_entity)


def detect_executor_diminishing(window: List[str]) -> Optional[str]:
    """Inspect the last 2 executor observations and return an exit reason
    if diminishing-returns are detected, else None.

    Reasons:
      - "loop"                 — first 200 chars identical (with len >= 50)
      - "low_info"             — both short + no digits/entities
      - "consecutive_errors"   — both contain "Error"
    """
    if len(window) < 2:
        return None
    a, b = window[-2], window[-1]
    if a[:200] == b[:200] and len(a) >= 50:
        return "loop"
    if _is_low_info_obs(a) and _is_low_info_obs(b):
        return "low_info"
    if "Error" in a and "Error" in b:
        return "consecutive_errors"
    return None
