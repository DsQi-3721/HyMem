"""Triage tool: find trace files that look "completed" but actually failed
due to API balance / Serper quota / other infra problems.

Symptoms we detect (in order of severity):
  1. "Insufficient Balance" / "Error code: 402"  → DeepSeek balance ran out
  2. "401" / "invalid_api_key"                   → API key was rejected
  3. "429" + empty/error tool results            → Serper quota exhausted
  4. "rate limit" / "quota exceeded"             → generic rate-limit
  5. Trace marked "failed" with explicit error
  6. Empty prediction with low planner activity  → agent never got going

Output:
  - Dry-run by default: prints a categorized report + list of bad traces.
  - --quarantine moves bad traces (+ matching .log files) into a backup dir.
    Once moved, `--resume` will re-execute those tasks.
  - --delete (DANGEROUS) deletes the bad traces in place.

Usage:
    # Just look
    python scripts/triage_failed_traces.py --log-dir logs/gaia_hymem/pass@3_run2

    # Move bad ones into a backup dir so --resume re-runs them
    python scripts/triage_failed_traces.py --log-dir logs/gaia_hymem/pass@3_run2 \\
        --quarantine

    # Also flag low-confidence/quick-fail tasks (more aggressive cleanup)
    python scripts/triage_failed_traces.py --log-dir logs/gaia_hymem/pass@3_run2 \\
        --aggressive --quarantine
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------- Detection patterns -------------------------------------------

# Patterns that mark a trace as definitively-bad. Each entry:
#   (category, regex, severity)
# severity: "hard" = definitely API/infra failure, must redo
#           "soft" = suspicious, redo unless --keep-soft
DETECTORS: List[Tuple[str, re.Pattern, str]] = [
    ("deepseek_402_balance",
     re.compile(r"Insufficient\s*Balance|Error\s*code:\s*402", re.IGNORECASE),
     "hard"),
    ("invalid_api_key",
     re.compile(r"invalid_api_key|invalid api key|incorrect api key|Error\s*code:\s*401",
                re.IGNORECASE),
     "hard"),
    ("serper_quota_429",
     re.compile(r"\[Serper\].*?(429|rate.?limit|quota)|"
                r"google\.serper\.dev.*?(429|403)",
                re.IGNORECASE),
     "soft"),  # soft because we want to also check tool results were empty
    ("generic_rate_limit",
     re.compile(r"\brate.?limited\b|\bquota.?exceeded\b|\btoo many requests\b",
                re.IGNORECASE),
     "soft"),
    ("internal_error_marker",
     re.compile(r"\(error[^)]*\)|\(timeout\)|\(no answer\)|\(child timed out\)",
                re.IGNORECASE),
     "soft"),
    ("empty_observation_all_calls",
     re.compile(r"\[Serper\] No API key available", re.IGNORECASE),
     "hard"),
]


@dataclass
class TraceVerdict:
    instance_id: str
    trace_path: Path
    log_path: Optional[Path]
    score: float
    status: str
    has_answer: bool
    # Detected pattern categories with hit counts
    hits: Counter = field(default_factory=Counter)
    severity: str = "ok"   # ok / suspicious / bad
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "trace_path": str(self.trace_path),
            "log_path": str(self.log_path) if self.log_path else None,
            "score": self.score,
            "status": self.status,
            "has_answer": self.has_answer,
            "hits": dict(self.hits),
            "severity": self.severity,
            "reason": self.reason,
        }


# ---------- Core classifier ---------------------------------------------

def load_trace_text(trace_path: Path) -> Tuple[Dict, str]:
    """Read trace JSON; return (data, full_serialized_text). Full text
    is what we pattern-match against, since errors can be anywhere
    (event_log, hymem_history, messages, tool_call_history)."""
    with open(trace_path) as f:
        data = json.load(f)
    # Serialize the WHOLE trace back to JSON for pattern scanning. This
    # catches errors buried in nested arrays / dict values.
    text = json.dumps(data, ensure_ascii=False)
    return data, text


def classify(
    trace_path: Path,
    log_path: Optional[Path],
    aggressive: bool = False,
) -> TraceVerdict:
    """Classify one trace. Returns a TraceVerdict."""
    try:
        data, trace_text = load_trace_text(trace_path)
    except (OSError, json.JSONDecodeError) as e:
        # Malformed trace itself is suspicious — redo
        return TraceVerdict(
            instance_id=trace_path.stem.replace("gaia_", "").replace("_trace", ""),
            trace_path=trace_path, log_path=log_path,
            score=0.0, status="malformed", has_answer=False,
            severity="bad", reason=f"trace JSON malformed: {e}",
        )

    instance_id = data.get("instance_id") or \
                  trace_path.stem.replace("gaia_", "").replace("_trace", "")
    score = float(data.get("score", 0) or 0)
    status = (data.get("stats", {}) or {}).get("status", "unknown")
    prediction = data.get("prediction") or ""
    has_answer = bool(prediction and str(prediction).strip()
                      and "(no answer" not in str(prediction).lower()
                      and "error" not in str(prediction).lower()[:20])

    # Also fold in .log file contents (sometimes errors appear in human-readable
    # log but not in the JSON history).
    combined_text = trace_text
    if log_path and log_path.exists():
        try:
            combined_text += "\n" + log_path.read_text(errors="replace")
        except OSError:
            pass

    verdict = TraceVerdict(
        instance_id=instance_id, trace_path=trace_path, log_path=log_path,
        score=score, status=status, has_answer=has_answer,
    )

    # Pattern scan
    hard_hits = 0
    soft_hits = 0
    for category, pat, severity in DETECTORS:
        matches = pat.findall(combined_text)
        if matches:
            verdict.hits[category] = len(matches)
            if severity == "hard":
                hard_hits += 1
            else:
                soft_hits += 1

    # Decision tree:
    #   - HARD hit (DeepSeek 402, 401, "No API key") → BAD: must redo
    #   - SOFT hit + score==0 + no good answer → BAD: redo
    #   - SOFT hit + score==1 → SUSPICIOUS but keep (model still answered correctly)
    #   - No hits, score==1, has_answer → OK
    #   - No hits, score==0 with reasonable prediction → OK (real wrong answer)
    #   - Aggressive mode: also flag score==0 + very low planner_turns
    if hard_hits > 0:
        verdict.severity = "bad"
        cats = sorted(c for c in verdict.hits if any(s == "hard" for n, _, s in DETECTORS if n == c))
        verdict.reason = f"hard API failure: {', '.join(cats)}"
    elif soft_hits > 0 and (score == 0 or not has_answer):
        verdict.severity = "bad"
        cats = sorted(verdict.hits.keys())
        verdict.reason = f"infra error patterns + no good answer: {', '.join(cats)}"
    elif soft_hits > 0 and score == 1:
        verdict.severity = "suspicious"
        verdict.reason = "infra errors but got correct answer anyway — keep"
    elif aggressive and score == 0 and not has_answer:
        # Aggressive: empty prediction + no errors usually means the agent
        # crashed silently early. Worth redoing.
        stats = data.get("stats", {}) or {}
        planner_turns = int(stats.get("planner_turns", 0) or 0)
        if planner_turns <= 1:
            verdict.severity = "bad"
            verdict.reason = f"aggressive: no answer + only {planner_turns} planner turn(s)"
        else:
            verdict.severity = "ok"
            verdict.reason = "agent worked but got wrong answer — keep"
    else:
        verdict.severity = "ok"
        if score == 1:
            verdict.reason = "score=1, clean trace"
        else:
            verdict.reason = "score=0 but no infra errors — real wrong answer"

    return verdict


# ---------- Main report + action -----------------------------------------

def build_report(verdicts: List[TraceVerdict]) -> str:
    by_sev = Counter(v.severity for v in verdicts)
    lines = []
    lines.append(f"=" * 70)
    lines.append(f"TRACE TRIAGE REPORT  ({len(verdicts)} traces)")
    lines.append(f"=" * 70)
    lines.append(f"  ok          : {by_sev.get('ok', 0):3d}  (keep — real answer)")
    lines.append(f"  suspicious  : {by_sev.get('suspicious', 0):3d}  (kept by default; --strict to remove)")
    lines.append(f"  bad         : {by_sev.get('bad', 0):3d}  (REMOVE — will be redone by --resume)")
    lines.append("")

    # Aggregate hit categories across bad/suspicious
    cat_counts: Counter = Counter()
    for v in verdicts:
        if v.severity in ("bad", "suspicious"):
            cat_counts.update(v.hits)
    if cat_counts:
        lines.append("Failure category breakdown (bad + suspicious):")
        for cat, n in cat_counts.most_common():
            lines.append(f"  {cat:30s}: {n}")
        lines.append("")

    # Score breakdown
    score1 = sum(1 for v in verdicts if v.score == 1 and v.severity == "ok")
    score0 = sum(1 for v in verdicts if v.score == 0 and v.severity == "ok")
    lines.append(f"Clean traces (kept): score=1: {score1}, score=0: {score0}")
    if (score1 + score0) > 0:
        lines.append(f"  Effective Pass@1 (after triage): {score1 / (score1 + score0):.3f}")
    lines.append("")
    return "\n".join(lines)


def write_quarantine(verdicts: List[TraceVerdict], log_dir: Path, severities_to_move: List[str]) -> Tuple[int, Path]:
    """Move trace + log files for the given severities into a backup dir."""
    quarantine = log_dir / "_quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    moved = 0
    for v in verdicts:
        if v.severity not in severities_to_move:
            continue
        try:
            shutil.move(str(v.trace_path), str(quarantine / v.trace_path.name))
            moved += 1
        except OSError as e:
            print(f"[WARN] could not move {v.trace_path}: {e}")
            continue
        if v.log_path and v.log_path.exists():
            try:
                shutil.move(str(v.log_path), str(quarantine / v.log_path.name))
            except OSError as e:
                print(f"[WARN] could not move {v.log_path}: {e}")
    # Write a manifest listing what was moved + why
    manifest_path = quarantine / "manifest.json"
    manifest_data = []
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifest_data = []
    manifest_data.append({
        "moved_count": moved,
        "verdicts": [v.to_dict() for v in verdicts if v.severity in severities_to_move],
    })
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False))
    return moved, quarantine


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Directory containing per-task *_trace.json + *.log files")
    parser.add_argument("--aggressive", action="store_true",
                        help="Also flag tasks with no answer + ≤1 planner turn even "
                             "without explicit error patterns (usually silent crashes)")
    parser.add_argument("--strict", action="store_true",
                        help="Also move 'suspicious' traces (e.g. correct answer "
                             "despite mid-run rate-limit). Default: keep them.")
    parser.add_argument("--quarantine", action="store_true",
                        help="Actually move bad traces to _quarantine/ subdir. "
                             "Without this flag, only prints the report (dry-run).")
    parser.add_argument("--delete", action="store_true",
                        help="DANGEROUS: delete bad traces in-place instead of "
                             "quarantining. Use only if you've backed things up.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write full verdict list as JSON to this path")
    parser.add_argument("--list-bad", action="store_true",
                        help="Print the list of bad trace instance_ids for piping")
    args = parser.parse_args(argv)

    log_dir: Path = args.log_dir.resolve()
    if not log_dir.exists():
        print(f"[FATAL] log dir not found: {log_dir}")
        return 2

    traces = sorted(log_dir.glob("*_trace.json"))
    if not traces:
        print(f"[FATAL] no *_trace.json files under {log_dir}")
        return 2

    print(f"Scanning {len(traces)} traces in {log_dir} ...")

    verdicts: List[TraceVerdict] = []
    for tp in traces:
        # Matching .log file (best-effort)
        log_path = tp.with_name(tp.name.replace("_trace.json", ".log"))
        if not log_path.exists():
            log_path = None
        verdicts.append(classify(tp, log_path, aggressive=args.aggressive))

    print(build_report(verdicts))

    bad_verdicts = [v for v in verdicts if v.severity == "bad"]
    if args.strict:
        bad_verdicts += [v for v in verdicts if v.severity == "suspicious"]

    if args.list_bad:
        for v in bad_verdicts:
            print(v.instance_id)

    if args.output:
        args.output.write_text(json.dumps(
            [v.to_dict() for v in verdicts], indent=2, ensure_ascii=False
        ))
        print(f"\nFull verdict list -> {args.output}")

    if not bad_verdicts:
        print("\n✅ Nothing to clean up — all traces look fine.")
        return 0

    # Show first few bad ones for sanity
    print(f"\nFirst 5 BAD traces (of {len(bad_verdicts)}):")
    for v in bad_verdicts[:5]:
        hits_str = ", ".join(f"{k}×{n}" for k, n in v.hits.items())
        print(f"  {v.instance_id}  score={v.score:.0f}  hits=[{hits_str}]")
        print(f"    -> {v.reason}")

    if args.delete and args.quarantine:
        print("\n[FATAL] Use either --quarantine OR --delete, not both.")
        return 2

    if args.delete:
        confirm = input(f"\nAbout to DELETE {len(bad_verdicts)} trace + matching log files. "
                        f"Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return 0
        for v in bad_verdicts:
            try:
                v.trace_path.unlink()
            except OSError as e:
                print(f"[WARN] could not delete {v.trace_path}: {e}")
            if v.log_path and v.log_path.exists():
                try:
                    v.log_path.unlink()
                except OSError as e:
                    pass
        print(f"Deleted {len(bad_verdicts)} bad traces. --resume will redo them.")
        return 0

    if args.quarantine:
        severities = ["bad"]
        if args.strict:
            severities.append("suspicious")
        moved, q_dir = write_quarantine(verdicts, log_dir, severities)
        print(f"\nMoved {moved} bad traces -> {q_dir}/")
        print(f"Manifest of moves: {q_dir}/manifest.json")
        print(f"\nNow re-run your eval with --resume; --resume will see those tasks")
        print(f"as not-yet-done and re-execute them.")
        print(f"\nIf you want to undo: mv {q_dir}/*.json {log_dir}/ "
              f"&& mv {q_dir}/*.log {log_dir}/")
        return 0

    # Default: just report
    print("\n(Dry-run: no files moved. Re-run with --quarantine to actually clean up.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
