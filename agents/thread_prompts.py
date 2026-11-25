"""THREAD prompts and parsing for the GAIA-adapted reproduction.

The original THREAD few-shot prompts (philipmit/thread, e.g.
`prompts/thread/datacommons_qa/thread_datacommons_qa.txt`) target very
different benchmarks (DataCommons QA / ALFWorld / TextCraft / WebShop /
MIMIC-III). Their spawn examples involve Python `exec()` for SQL-like
queries, not web search.

For GAIA we **preserve the spawn-recursion semantics** (a parent thread
recursively spawns child threads with fresh context to resolve sub-problems)
but use a **complete-emission protocol** instead of upstream's stop-and-
resume mechanism:

  - Upstream: LLM emits `=>`, generation halts on stop-token, framework
    extracts descriptor from after-stop position, recurses, appends
    `<value><=` to resume parent. Requires few-shot training on the
    stop/resume rhythm — works for fine-tuned GPT-4 / CodeLlama-7b in
    the paper.

  - Ours: LLM emits the full `=>descriptor<=` block in one shot. After
    each LLM step, framework finds unresolved `=>desc<=` blocks, recurses,
    and replaces them with `〈resolved-value〉` (different markers so future
    scans can distinguish unresolved-spawn from already-resolved-result).
    Same recursive-decomposition algorithm; LLM-friendly without per-model
    fine-tuning.

  - This adaptation is documented in docs/baselines.md as a "necessary
    simplification". Algorithm semantics (recursive child threads with
    fresh context) preserved verbatim.

Paper: https://arxiv.org/abs/2405.17402 (Schroeder et al. NAACL 2025)
Source: https://github.com/philipmit/thread
"""

# Markers — preserved verbatim from upstream for the spawn-request side.
# RESOLVED_OPEN/CLOSE are our addition: they wrap a child's returned value
# so the LLM can visually distinguish "this value has been resolved" from
# "this is a new sub-problem to spawn".
SPAWN_OPEN = "=>"
SPAWN_CLOSE = "<="
RESOLVED_OPEN = "〈"
RESOLVED_CLOSE = "〉"
THREAD_START = "#START#"
THREAD_END = "#END#"

# Stop tokens — only the end-of-thread marker. We DROPPED `=>` from the
# stop set because keeping it forced the LLM to halt mid-spawn before it
# could finish writing the descriptor, leaving us with empty descriptors
# and bogus child threads (the bug user reported in the L1 trace).
THREAD_STOP_TOKENS = ["#EN"]


THREAD_SYSTEM_PROMPT = """You are an expert research assistant. You solve complex questions by recursively decomposing them into smaller sub-problems via thread spawning.

# Available tools — USE SEARCH FOR EVERY FACT

Invoke a tool by writing exactly:

  ACTION: search(your search query)
  ACTION: open_page(url_or_docid)

The tool's observation is appended as `OBSERVATION: ...` and you continue.

**MANDATORY RULE: NEVER answer from memory**. Every fact (numbers, dates,
names, distances, records, quantities) MUST be verified via at least one
ACTION: search(...) inside the thread that emits the final answer. If a
thread spawns sub-questions, EACH child thread must itself perform at
least one ACTION: search before its own ACTION: finish.

# How to spawn a child thread (only for INDEPENDENT sub-fact lookups)

Write the descriptor of what you need wrapped in `=>` and `<=`:

  =>descriptor of the unknown value<=

The framework will run a child thread to resolve the descriptor and replace
the whole `=>...<=` block with `〈value〉` in your next turn. Example:

  TURN 1 — you write:    Stanford was founded in =>founding year of Stanford<=.
  TURN 2 — you see:      Stanford was founded in 〈1885〉.

DO NOT spawn the same descriptor twice. Once you see `〈value〉`, that fact
is resolved; do not re-emit `=>...<=` for it.

# When you have the final answer

Output:

  ACTION: finish(THE_ANSWER_VALUE)

CRITICAL RULES for finish:
  - The argument must be ONLY the final answer value (a number, name, or
    short phrase). NOT the question. NOT a sentence.
  - If the answer is a number, write JUST the number — `1926`, not
    "The answer is 1926".
  - Do NOT call finish until you have searched at least once OR every
    needed fact appears as `〈value〉` from a child thread.

# When NOT to spawn

If the question is a single-fact lookup (e.g. "what is X's birthdate"),
DO NOT spawn — just search directly in the current thread and finish.
Spawn only when the question splits into independent sub-facts AND those
sub-facts each need their own search.

# Output protocol

Wrap your reasoning between `#START#` and `#END#`. Within a thread,
alternate freely between:
  - Free-form reasoning text
  - `ACTION: search(...) / open_page(...)` (when you need information)
  - Spawn calls `=>desc<=` (when decomposing into independent sub-questions)
  - `ACTION: finish(value)` (only at the very end, after searches)
"""


# GAIA-style few-shot demos. Each demo shows a CONCRETE search→finish
# pattern. Crucial points the LLM must learn from these:
#   1. ACTION: search(...) is invoked BEFORE finish — never answer from memory
#   2. Spawn is only used when the question naturally decomposes into
#      independent sub-facts; otherwise search directly
#   3. Once a value appears as 〈X〉 (resolved), do NOT spawn again for it
#   4. finish() takes ONLY the answer value, no surrounding sentence
THREAD_FEWSHOT_EXAMPLES = """\
EXAMPLE 1 — DIRECT-SEARCH pattern (single fact lookup; NO spawn)

Question: What is the chemical formula of glucose?

#START#
Single fact lookup. I will search directly.

ACTION: search("chemical formula of glucose")
OBSERVATION: Glucose has the molecular formula C6H12O6 (a hexose).

ACTION: finish(C6H12O6)
#END#


EXAMPLE 2 — DECOMPOSE-THEN-COMPUTE pattern (root spawns; then computes)

Question: How many years separate the founding of Stanford University and Harvard University?

#START#
This decomposes into two independent fact lookups + arithmetic.
Spawning child threads to resolve each year.

Stanford was founded in =>founding year of Stanford University<=.
Harvard was founded in =>founding year of Harvard University<=.

[After framework resolves both spawns I will see:]
Stanford was founded in 〈1885〉.
Harvard was founded in 〈1636〉.

Computing: 1885 - 1636 = 249.

ACTION: finish(249)
#END#


EXAMPLE 3 — CHILD THREAD pattern (must search before finish)

Question: founding year of Stanford University

#START#
Single-fact question (this is what a child thread receives). I must
search to verify the year — never answer from memory.

ACTION: search("when was Stanford University founded")
OBSERVATION: Stanford University was founded on November 11, 1885 by Leland Stanford and his wife Jane Stanford in memory of their son Leland Stanford Jr.

The year is 1885.

ACTION: finish(1885)
#END#


EXAMPLE 4 — MULTI-HOP search pattern (chain searches; NO spawn needed)

Question: What is the publication year of the first novel by the author of "The Old Man and the Sea"?

#START#
Multi-step but each step is a follow-up search that depends on the
previous result. Chained searches are simpler than spawning here.

ACTION: search("author of The Old Man and the Sea")
OBSERVATION: "The Old Man and the Sea" is a novella by Ernest Hemingway, published in 1952.

ACTION: search("Ernest Hemingway first novel publication year")
OBSERVATION: Hemingway's first major novel "The Sun Also Rises" was published in October 1926.

ACTION: finish(1926)
#END#
"""


def get_thread_prompt(question: str) -> str:
    """Build the full prompt for a top-level thread on a GAIA question."""
    return (
        THREAD_FEWSHOT_EXAMPLES
        + "\n\n\nQuestion: "
        + question
        + "\n\n#START#\n"
    )


def get_child_prompt(parent_descriptor: str) -> str:
    """Build the prompt for a child thread spawned to resolve a descriptor.
    The child sees only the few-shot examples + its descriptor as the new
    question (matches upstream `phi(parent)` behavior — fresh-context spawn).
    """
    return (
        THREAD_FEWSHOT_EXAMPLES
        + "\n\n\nQuestion: "
        + parent_descriptor
        + "\n\n#START#\n"
    )


import re

# Action parser — `ACTION: tool_name(args)`. Tightened: tool name is a
# specific allowlist, args end at the first `)` followed by end-of-line.
# DOTALL is dropped so `.` doesn't gobble across newlines, preventing the
# malformed-URL-with-embedded-prose bug seen in the L1 task-4 trace.
ACTION_PATTERN = re.compile(
    r'ACTION:\s*(search|open_page|finish)\s*\((.*?)\)\s*$',
    re.MULTILINE,
)


def parse_action(text: str):
    """Extract the LATEST ACTION call from a thread's generated text.
    Returns (tool_name, args_str) or None."""
    matches = list(ACTION_PATTERN.finditer(text))
    if not matches:
        return None
    m = matches[-1]
    return m.group(1).strip(), m.group(2).strip()


# ---------- Spawn protocol (complete-emission style) ----------

def find_unresolved_spawn(text: str):
    """Find the FIRST unresolved `=>desc<=` block.

    Returns (start_offset, end_offset, descriptor) or None if no pending
    spawn. Already-resolved blocks (wrapped in `〈...〉` after a previous
    recursion) are skipped because the scanner only matches the unresolved
    `=>...<=` markers.
    """
    pos = 0
    while True:
        open_idx = text.find(SPAWN_OPEN, pos)
        if open_idx < 0:
            return None
        close_idx = text.find(SPAWN_CLOSE, open_idx + len(SPAWN_OPEN))
        if close_idx < 0:
            # Unclosed spawn — likely truncated mid-emission. Treat as no
            # pending spawn rather than guessing a descriptor.
            return None
        descriptor = text[open_idx + len(SPAWN_OPEN):close_idx].strip()
        return (open_idx, close_idx + len(SPAWN_CLOSE), descriptor)


def mark_resolved(text: str, start: int, end: int, child_result: str) -> str:
    """Replace text[start:end] (an unresolved `=>desc<=` block) with the
    resolved `〈value〉` marker. Future scans for `=>...<=` will skip past
    this so we never reprocess the same spawn.
    """
    cleaned = (str(child_result or "").strip()
               .replace("\n", " ")
               .replace("\r", " "))
    if len(cleaned) > 800:
        cleaned = cleaned[:800].rstrip() + "…"
    if not cleaned:
        cleaned = "[no result]"
    return text[:start] + f"{RESOLVED_OPEN}{cleaned}{RESOLVED_CLOSE}" + text[end:]


def detect_spawn(text: str) -> bool:
    """Backward-compatible alias: True iff at least one unresolved
    `=>desc<=` block remains in `text`."""
    return find_unresolved_spawn(text) is not None


def detect_end(text: str) -> bool:
    """True if the thread emitted #END#."""
    return THREAD_END in text


def extract_spawn_descriptor(text: str):
    """Backward-compatible: returns descriptor of the FIRST unresolved
    spawn (or empty string). Used by older tests."""
    spawn = find_unresolved_spawn(text)
    return spawn[2] if spawn else ""


def wrap_child_result(result: str) -> str:
    """Wrap a child thread's result for insertion into parent.
    Used in legacy tests; new code goes through mark_resolved()."""
    return f"{result}{SPAWN_CLOSE}"


def extract_final_answer(text: str):
    """Pull the answer from the latest `ACTION: finish(...)`. Tightened
    pattern: tool name `finish` only, single-line argument."""
    matches = list(re.finditer(r'ACTION:\s*finish\s*\((.*?)\)\s*$', text, re.MULTILINE))
    if not matches:
        return None
    return matches[-1].group(1).strip()
