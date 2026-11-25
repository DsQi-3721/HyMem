"""Live monitor for an eval_gaia.py run.

Watches a log directory for fatal API errors and takes automated actions:
  - DeepSeek "Insufficient Balance" / 402  → SIGTERM the eval process
  - Serper 429 / quota exhausted           → counts; SIGTERM after threshold
  - OpenAI / DeepSeek 401 invalid key      → SIGTERM the eval process

Designed to attach to a run that's ALREADY in flight — no restart, no env
changes, no patches to the running process. Just tail the per-task .log
files and fire SIGTERM via OS-level pgrep when a fatal pattern shows up.

Usage:
    # In a separate terminal from your eval:
    python scripts/monitor_eval_run.py \\
        --log-dir logs/gaia_hymem/pass@3_run1 \\
        --process-name eval_gaia.py

The script keeps running until the eval process exits (or you Ctrl-C the
monitor, which leaves the eval alone). Detected events are also written
to `<log-dir>/_monitor.jsonl` for forensics.

Exit codes:
    0  monitored process exited normally
    1  monitor itself crashed
    2  killed eval due to fatal pattern (DeepSeek balance / invalid key)
    3  killed eval due to Serper exhaustion
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ----- Fatal pattern table -------------------------------------------------

# Each pattern: (label, regex, severity)
#   severity = 'fatal_kill' → SIGTERM immediately on first match
#   severity = 'count'      → count occurrences; trigger SIGTERM if rate exceeded
PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # DeepSeek balance / OpenAI hard quota
    ("deepseek_insufficient_balance",
     re.compile(r"Insufficient\s*Balance|Error\s*code:\s*402", re.IGNORECASE),
     "fatal_kill"),

    # Invalid API key — also fatal because no point continuing
    ("invalid_api_key",
     re.compile(r"invalid_api_key|invalid api key|incorrect api key|Error code:\s*401",
                re.IGNORECASE),
     "fatal_kill"),

    # Serper rate-limit / quota — recoverable per-call but if many in window, kill
    ("serper_rate_limited",
     re.compile(
         r"\[Serper\].*?(429|rate.?limit|quota)|"
         r"google\.serper\.dev.*?(429|403)",
         re.IGNORECASE),
     "count"),

    # Generic 5xx from DeepSeek — info only
    ("deepseek_5xx",
     re.compile(r"deepseek\.com.*?5\d{2}|Error code:\s*5\d{2}", re.IGNORECASE),
     "info"),
]


# ----- Helpers --------------------------------------------------------------

def find_eval_pids(process_name: str) -> List[int]:
    """Return PIDs of running processes whose cmdline contains process_name."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", process_name], text=True, stderr=subprocess.DEVNULL,
        )
        return [int(p) for p in out.split() if p.strip().isdigit()]
    except subprocess.CalledProcessError:
        return []


def kill_eval(pids: List[int], reason: str) -> None:
    """SIGTERM the listed PIDs; escalate to SIGKILL after 30s if still alive."""
    print(f"\n[MONITOR] !!! Sending SIGTERM to PIDs {pids} — reason: {reason}",
          flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            print(f"[MONITOR] PID {pid} already gone")
            continue
        except PermissionError:
            print(f"[MONITOR] PID {pid}: PermissionError on SIGTERM "
                  f"(uid mismatch?). Skipping.")
            continue

    # Grace period
    deadline = time.time() + 30
    while time.time() < deadline:
        alive = [p for p in pids if _pid_alive(p)]
        if not alive:
            print(f"[MONITOR] All PIDs exited.")
            return
        time.sleep(1)

    # Escalate
    survivors = [p for p in pids if _pid_alive(p)]
    if survivors:
        print(f"[MONITOR] PIDs {survivors} ignored SIGTERM — escalating to SIGKILL")
        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ----- Log scanner ---------------------------------------------------------

class LogScanner:
    """Tracks file → byte-offset for incremental scans, so we don't re-read
    log content on every pass."""

    def __init__(self, log_dir: Path, monitor_log: Path):
        self.log_dir = log_dir
        self.monitor_log = monitor_log
        self.offsets: Dict[Path, int] = {}
        self.events: List[Dict] = []
        # Rolling counts for "count" severity patterns
        self.counts: Dict[str, deque] = {p[0]: deque() for p in PATTERNS}

    def fast_forward_to_eof(self) -> int:
        """Initialize byte offsets to current file sizes so we only catch
        NEW content. Called at startup to avoid re-detecting historical
        errors from previous failed runs. Returns # files fast-forwarded."""
        if not self.log_dir.exists():
            return 0
        n = 0
        for path in list(self.log_dir.glob("*.log")) + list(self.log_dir.glob("*_trace.json")):
            try:
                self.offsets[path] = path.stat().st_size
                n += 1
            except OSError:
                continue
        return n

    def scan_once(self) -> List[Tuple[str, Path, str]]:
        """Scan all .log/.json files in log_dir for new content; match patterns.
        Returns list of (label, file, snippet)."""
        hits: List[Tuple[str, Path, str]] = []
        if not self.log_dir.exists():
            return hits

        # Look at .log files (per-task) AND _trace.json files. The trace JSON
        # captures stats including any error strings recorded in the agent's
        # event log.
        for path in list(self.log_dir.glob("*.log")) + list(self.log_dir.glob("*_trace.json")):
            try:
                size = path.stat().st_size
            except OSError:
                continue
            offset = self.offsets.get(path, 0)
            if size <= offset:
                continue
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    data = f.read(size - offset).decode("utf-8", errors="replace")
            except OSError:
                continue
            self.offsets[path] = size

            for label, pat, severity in PATTERNS:
                for m in pat.finditer(data):
                    snippet_start = max(0, m.start() - 80)
                    snippet_end = min(len(data), m.end() + 80)
                    snippet = data[snippet_start:snippet_end].replace("\n", " ")
                    hits.append((label, path, snippet))
                    if severity == "count":
                        self.counts[label].append(time.time())

        return hits

    def record_event(self, kind: str, label: str, file: str, snippet: str) -> None:
        evt = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "kind": kind,            # detected | action | info
            "label": label,
            "file": str(file),
            "snippet": snippet[:300],
        }
        self.events.append(evt)
        try:
            with open(self.monitor_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt, ensure_ascii=False) + "\n")
        except OSError:
            pass
        print(f"[MONITOR] {evt['ts']} {kind.upper()} {label}: "
              f"{Path(file).name} :: {snippet[:120]}", flush=True)

    def trim_count_window(self, window_seconds: float) -> None:
        """Drop count entries older than the window so the rate is rolling."""
        threshold = time.time() - window_seconds
        for q in self.counts.values():
            while q and q[0] < threshold:
                q.popleft()

    def count(self, label: str) -> int:
        return len(self.counts.get(label, ()))


# ----- Main loop -----------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", required=True, type=Path,
                        help="Directory containing per-task .log + _trace.json files "
                             "(--log_dir of the eval_gaia.py run)")
    parser.add_argument("--process-name", default="eval_gaia.py",
                        help="String to grep for in the eval process's cmdline")
    parser.add_argument("--poll-seconds", type=float, default=3.0,
                        help="How often to scan logs (default 3s)")
    parser.add_argument("--serper-window-seconds", type=float, default=120.0,
                        help="Sliding window for Serper rate-limit count (default 120s)")
    parser.add_argument("--serper-window-threshold", type=int, default=5,
                        help="If Serper rate-limit hits ≥ this many times in the "
                             "window, treat as exhausted and SIGTERM eval (default 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Detect patterns and log but do NOT send SIGTERM")
    parser.add_argument("--wait-for-eval", type=int, default=600,
                        metavar="SECONDS",
                        help="If no eval process is found at startup, wait up to this "
                             "many seconds for one to appear (default 600 = 10 min). "
                             "0 = exit immediately (legacy behavior).")
    args = parser.parse_args(argv)

    log_dir: Path = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    monitor_log = log_dir / "_monitor.jsonl"

    print(f"[MONITOR] Watching {log_dir}", flush=True)
    print(f"[MONITOR] Process: '{args.process_name}', poll={args.poll_seconds}s, "
          f"serper window={args.serper_window_seconds}s/"
          f"≥{args.serper_window_threshold} hits", flush=True)
    print(f"[MONITOR] Events log: {monitor_log}", flush=True)
    print(f"[MONITOR] Dry-run: {args.dry_run}", flush=True)
    print()

    scanner = LogScanner(log_dir, monitor_log)
    # Skip past pre-existing log content so we don't re-flag historical
    # errors from previous failed runs (the user already triaged those).
    n_ff = scanner.fast_forward_to_eof()
    if n_ff:
        print(f"[MONITOR] Fast-forwarded past existing content in {n_ff} files. "
              f"Will only react to NEW errors from now on.", flush=True)

    seen_fatal_keys: Set[str] = set()  # dedupe fatal events
    last_pid_check = 0.0
    target_pids: List[int] = []
    ever_attached: bool = False   # have we ever seen a matching PID?
    wait_start: float = time.time()

    try:
        while True:
            now = time.time()

            # Refresh PIDs every ~10s (cheap)
            if now - last_pid_check > 10:
                target_pids = find_eval_pids(args.process_name)
                last_pid_check = now

                if target_pids:
                    if not ever_attached:
                        print(f"[MONITOR] Attached to PIDs {target_pids} "
                              f"matching '{args.process_name}'.", flush=True)
                    ever_attached = True
                elif ever_attached:
                    # Was attached previously; now gone → eval finished
                    print(f"[MONITOR] Eval process(es) exited — monitor done.",
                          flush=True)
                    return 0
                else:
                    # Never attached; in wait-for-eval mode
                    elapsed = now - wait_start
                    if args.wait_for_eval <= 0:
                        print(f"[MONITOR] No process matching '{args.process_name}' "
                              f"found — eval may have ended. Exiting "
                              f"(--wait-for-eval=0).", flush=True)
                        return 0
                    if elapsed > args.wait_for_eval:
                        print(f"[MONITOR] No '{args.process_name}' process appeared "
                              f"within {args.wait_for_eval}s — giving up.",
                              flush=True)
                        return 0
                    # Quiet status update every 30s
                    if int(elapsed) % 30 == 0:
                        remaining = args.wait_for_eval - int(elapsed)
                        print(f"[MONITOR] Waiting for '{args.process_name}' to "
                              f"start ... ({int(elapsed)}s elapsed, "
                              f"{remaining}s remaining)", flush=True)

            # Scan logs
            hits = scanner.scan_once()
            scanner.trim_count_window(args.serper_window_seconds)

            for label, path, snippet in hits:
                pat_severity = next(s for n, _, s in PATTERNS if n == label)

                if pat_severity == "fatal_kill":
                    key = f"{label}::{path}"
                    if key in seen_fatal_keys:
                        continue
                    seen_fatal_keys.add(key)
                    scanner.record_event("detected", label, path, snippet)
                    if args.dry_run:
                        scanner.record_event("action", label, path,
                                             "[dry-run] would SIGTERM")
                    else:
                        scanner.record_event("action", label, path,
                                             f"SIGTERM PIDs {target_pids}")
                        kill_eval(target_pids, reason=label)
                        return 2

                elif pat_severity == "count":
                    scanner.record_event("detected", label, path, snippet)
                    n = scanner.count(label)
                    if n >= args.serper_window_threshold:
                        msg = (f"{label}: {n} hits in last "
                               f"{args.serper_window_seconds:.0f}s — "
                               f"exceeds threshold {args.serper_window_threshold}")
                        if args.dry_run:
                            scanner.record_event("action", label, path,
                                                 f"[dry-run] would SIGTERM ({msg})")
                        else:
                            scanner.record_event("action", label, path,
                                                 f"SIGTERM (rate-exhaust): {msg}")
                            kill_eval(target_pids, reason=msg)
                            return 3

                else:  # info
                    scanner.record_event("info", label, path, snippet)

            time.sleep(args.poll_seconds)

    except KeyboardInterrupt:
        print(f"\n[MONITOR] Ctrl-C received — exiting WITHOUT killing eval", flush=True)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[MONITOR] ERROR: {e}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
