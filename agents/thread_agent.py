"""THREAD baseline (training-free reproduction, GAIA-adapted).

Algorithm (mirrors `external/thread/datacommons_qa/run_datacommons_qa.py:706`):
    A thread is one invocation of the LLM that may either:
      (a) run to completion (emit #END# with a final ACTION: finish(...))
      (b) emit `=>desc` (spawn open) — generation stops; framework recursively
          calls another thread() with `desc` as its question; the child's
          finish-answer is wrapped `=>answer<=` and appended to the parent's
          token stream; parent then continues.
      (c) emit `ACTION: tool(args)` — framework dispatches the tool, appends
          `OBSERVATION: ...` to the parent's stream, and continues.

    All threads share a global recursion budget (t_max).

Faithful to:
    Schroeder et al., "THREAD: Thinking Deeper with Recursive Spawning",
    arXiv 2405.17402, NAACL 2025.

Adaptation notes (documented in docs/baselines.md):
    - Original benchmarks (DataCommons / ALFWorld / WebShop) use Python
      exec() for tool actions. We retarget to GAIA web tools — search,
      open_page, finish — dispatched via BMToolManager.
    - Few-shot prompts re-authored for GAIA (paper-original prompts target
      e.g. data-commons SQL queries and don't transfer).
    - Spawn syntax (`=>...<=`), recursion, and shared budget kept verbatim.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # noqa: F401
import torch  # noqa: F401

from verl import DataProto
from .utils import CallLLM, Agent, select_env, TaskContext, CallAPI
from .prompts import create_chat
from .tool_metrics import UnifiedToolCounter, write_baseline_stats
from .thread_prompts import (
    THREAD_SYSTEM_PROMPT,
    THREAD_STOP_TOKENS,
    get_thread_prompt,
    get_child_prompt,
    parse_action,
    detect_end,
    extract_final_answer,
    find_unresolved_spawn,
    mark_resolved,
    RESOLVED_OPEN,
    RESOLVED_CLOSE,
)


DEFAULT_SESSION_TIMEOUT = 150 * 60
DEFAULT_THREAD_T_MAX = 25         # paper Appendix-suggested order
DEFAULT_MAX_DEPTH = 3             # bound recursion (paper: max ~3-4 typical)
DEFAULT_MAX_NEW_TOKENS_PER_STEP = 1024


class _ThreadState:
    """Shared mutable state across all (parent + child) threads in one task.
    Tracks the global recursion budget (t_max), tool-call atomic counts
    (so we can populate env.stats consistently), token usage, and the
    spawn tree for debugging."""

    def __init__(self, t_max: int):
        self.t_max = t_max
        self.t = 0                       # global thread step counter
        self.tool_call_atomic = 0
        self.tool_calls_by_tool: Dict[str, int] = {}
        self.tool_calls_by_source: Dict[str, int] = {"actor": 0, "child_thread": 0}
        self.spawn_count = 0             # how many child threads were spawned
        # G3 — successful spawns. A spawn counts as success when its child
        # returned a non-empty, non-fallback answer (i.e. it actually
        # produced a useful value, not "(unknown)" / "(child timed out)").
        # Symmetric with HyMem's branch_success — enables direct §C
        # paired comparison.
        self.spawn_success = 0
        self.spawn_depth_max = 0
        self.finished = False
        self.final_answer: Optional[str] = None
        # Token accounting across all threads (parent + children)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # G4 — per-step context size trace (manually maintained because
        # Thread bypasses the Agent class). Each entry =
        #   (t_step, depth, context_token_len)
        # so analysis can separate root vs child trajectories.
        self.context_size_history: list = []

    def step(self) -> bool:
        """Increment global counter; return True if budget remains."""
        self.t += 1
        return self.t < self.t_max

    def record_tool(self, tool_name: str, source: str):
        self.tool_call_atomic += 1
        self.tool_calls_by_tool[tool_name] = self.tool_calls_by_tool.get(tool_name, 0) + 1
        if source not in self.tool_calls_by_source:
            self.tool_calls_by_source[source] = 0
        self.tool_calls_by_source[source] += 1


async def _dispatch_tool(env, tool_name: str, args: str) -> str:
    """Dispatch a tool call by emitting the env's expected XML format and
    routing through env.run_action. This keeps env.stats in sync with the
    unified counter (env.stats['search'] etc. are populated by env.run_action).
    """
    if tool_name == "search":
        # query may be quoted; strip quotes
        q = args.strip().strip('"\'')
        xml = f"<function=search>\n<parameter=query>{q}</parameter>\n<parameter=topk>5</parameter>\n</function>"
    elif tool_name == "open_page":
        url = args.strip().strip('"\'')
        if url.startswith("http"):
            xml = f"<function=open_page>\n<parameter=url>{url}</parameter>\n</function>"
        else:
            xml = f"<function=open_page>\n<parameter=docid>{url}</parameter>\n</function>"
    elif tool_name == "finish":
        ans = args.strip().strip('"\'')
        xml = (
            f"<function=finish>\n"
            f"<parameter=answer>{ans}</parameter>\n"
            f"<parameter=explanation>Thread completed via spawn-recursive decomposition.</parameter>\n"
            f"<parameter=confidence>0.8</parameter>\n"
            f"</function>"
        )
    else:
        return f"[Error] Unknown tool: {tool_name}"

    from .utils import run_action
    obs = await run_action(env, xml)
    if obs is None:
        return "[finish]"  # signal upward to terminate
    return str(obs) if obs else "[empty observation]"


async def _llm_step(llm_client, tokenizer, config, chat_messages: list,
                    max_new_tokens: int):
    """Single LLM call given a chat-history message list. Returns
    (text, usage_dict) or (None, None) on failure.

    REWORKED (was: take raw full_prompt). Original implementation passed
    `[system, user(few_shot+question+assistant_so_far)]`, which made the
    LLM treat the assistant's prior output as part of the user-side
    context — leading it to RE-EMIT the same `=>desc<=` blocks each turn
    (observed in user's L1 trace: 6 spawns of 2 unique descriptors,
    indicating duplicate emissions). Proper chat-message alternation
    (system / user / assistant / user / assistant / ...) gives the LLM
    clear "this is what I already said" boundaries.
    """
    try:
        input_ids = tokenizer.apply_chat_template(
            chat_messages, add_generation_prompt=True, tokenize=True
        )
        completion = await llm_client.create_completion(
            input_ids, messages=chat_messages,
            max_new_tokens=max_new_tokens,
        )
        if completion is None:
            return None, None
        msg = completion["choices"][0]["message"]
        text = msg.get("content") or ""
        usage = msg.get("metrics", {}).get("usage", {})
        for stop in THREAD_STOP_TOKENS:
            idx = text.find(stop)
            if idx >= 0:
                text = text[:idx + len(stop)]
                break
        return text, usage
    except Exception as e:
        print(f"[Thread] LLM call failed: {e}")
        return None, None


async def _thread(
    parent_prompt: str,
    state: _ThreadState,
    env,
    llm_client,
    tokenizer,
    config,
    *,
    depth: int,
    max_depth: int,
    max_new_tokens: int,
    is_root: bool,
) -> str:
    """Run one thread to completion or until the global budget is exhausted.
    Returns the thread's accumulated text (root) or the finish-answer string
    (child).

    REWRITTEN — uses proper chat-message history (system / user / assistant
    / user / assistant / ...) instead of single growing user-text. Each
    LLM step appends an assistant message; spawn resolutions and tool
    observations are appended as USER messages so the LLM sees clear turn
    boundaries. Without this the LLM re-emitted duplicate `=>desc<=`
    blocks each turn (observed bug).
    """
    state.spawn_depth_max = max(state.spawn_depth_max, depth)
    source_label = "actor" if is_root else "child_thread"

    # Initialize chat with system prompt + few-shot + question
    chat: List[Dict[str, str]] = [
        {"role": "system", "content": THREAD_SYSTEM_PROMPT},
        {"role": "user", "content": parent_prompt},
    ]
    # Track every assistant emission concatenated for final answer scan
    Y = ""
    # Per-thread dedup: descriptors we've already spawned for. Prevents
    # the LLM from re-asking the same sub-question across turns (observed
    # in user's L1 task 1: same descriptor spawned 3×).
    seen_descriptors: set = set()
    # Per-thread guardrail state — every thread (root or child) must do
    # at least one useful action (search / open_page / spawn-that-actually-
    # spawned) before its own ACTION: finish is allowed. This matches
    # Thread's design intent: every thread does work, doesn't just answer
    # from memory. Without this, deepseek-reasoner-class models emit
    # ACTION: finish on the first turn straight from parametric memory and
    # the env's must_search rule zeros their answer (observed L1 trace,
    # tasks 3/4/6/8/9/10).
    did_useful_action = False
    finish_reject_count = 0      # how many times we blocked a premature finish
    MAX_FINISH_REJECTS = 2       # after this many rejects, allow even if no action

    while True:
        if not state.step():
            print(f"[Thread] Global budget t_max={state.t_max} reached at depth={depth}")
            break
        if state.finished:
            break

        # G4: record context size BEFORE this LLM step
        approx_chars = sum(len(m.get("content", "")) for m in chat)
        state.context_size_history.append((state.t, depth, approx_chars))

        gen, usage = await _llm_step(
            llm_client, tokenizer, config, chat, max_new_tokens,
        )
        if gen is None:
            break
        if usage:
            state.total_prompt_tokens += int(usage.get('prompt_tokens', 0) or 0)
            state.total_completion_tokens += int(usage.get('completion_tokens', 0) or 0)

        # Append assistant turn to chat history (verbatim — never modified)
        chat.append({"role": "assistant", "content": gen})
        Y = Y + "\n" + gen

        if detect_end(gen):
            ans = extract_final_answer(gen) or extract_final_answer(Y)
            if is_root and ans is not None:
                await _dispatch_tool(env, "finish", ans)
                state.record_tool("finish", source_label)
                state.finished = True
                state.final_answer = ans
            return ans if ans is not None else gen

        # Resolve all unresolved spawn blocks in THIS turn's response.
        # Each resolution becomes a USER message reporting the result, so
        # the LLM sees its previous spawn requests as "asked, got reply"
        # and won't re-emit them.
        spawn_results: List[str] = []
        response_with_resolved = gen
        while True:
            spawn = find_unresolved_spawn(response_with_resolved)
            if spawn is None:
                break
            sp_start, sp_end, descriptor = spawn

            if not descriptor:
                response_with_resolved = mark_resolved(
                    response_with_resolved, sp_start, sp_end,
                    "[empty spawn skipped]",
                )
                spawn_results.append("[empty spawn skipped]")
                continue

            if depth >= max_depth:
                response_with_resolved = mark_resolved(
                    response_with_resolved, sp_start, sp_end,
                    "[max depth — not spawned]",
                )
                spawn_results.append(f"max depth reached, not spawned: {descriptor}")
                continue

            # Dedup: if we've already spawned for this descriptor (or a
            # near-duplicate after lowercasing/whitespace normalization),
            # skip recursion and just mark resolved with a hint.
            normalized = " ".join(descriptor.lower().split())
            if normalized in seen_descriptors:
                response_with_resolved = mark_resolved(
                    response_with_resolved, sp_start, sp_end,
                    "[duplicate descriptor — see prior spawn result above]",
                )
                spawn_results.append(
                    f"duplicate (already-spawned descriptor): {descriptor}"
                )
                continue
            seen_descriptors.add(normalized)

            state.spawn_count += 1
            did_useful_action = True   # spawn counts as useful work
            print(f"[Thread] Spawn @ depth {depth+1}: {descriptor[:80]}")
            child_prompt = get_child_prompt(descriptor)
            child_result = await _thread(
                child_prompt, state, env, llm_client, tokenizer, config,
                depth=depth + 1, max_depth=max_depth,
                max_new_tokens=max_new_tokens, is_root=False,
            )
            usable = False
            if child_result is not None:
                _r = str(child_result).strip()
                if (_r and not _r.startswith("(")
                        and not _r.startswith("[")
                        and _r.lower() not in ("unknown", "n/a", "")):
                    state.spawn_success += 1
                    usable = True
            if child_result is None or not usable:
                child_result = str(child_result) if child_result else "unknown"

            response_with_resolved = mark_resolved(
                response_with_resolved, sp_start, sp_end, child_result,
            )
            spawn_results.append(f"`{descriptor}` → {child_result}")

        if spawn_results:
            # Report spawn resolutions back to LLM as a user message.
            # Phrasing tells the LLM these are now-known facts and not to
            # spawn the same descriptors again.
            obs_text = (
                "Spawn results (already-resolved facts; do NOT spawn these "
                "descriptors again, use the values directly):\n"
                + "\n".join(f"  - {r}" for r in spawn_results)
                + "\n\nContinue toward the final answer. Remember: you must "
                "still ACTION: search(...) for any unverified fact before "
                "ACTION: finish(value)."
            )
            chat.append({"role": "user", "content": obs_text})
            Y = Y + "\n[spawn-results]\n" + "\n".join(spawn_results)
            continue

        # ACTION dispatch (tool call)
        action = parse_action(gen)
        if action:
            tool_name, args = action

            # Per-thread guardrail (kept; faithful to upstream Thread's
            # implicit "every thread does sub-work" assumption):
            # refuse ACTION: finish if THIS thread has not yet done any
            # useful action (search / open_page / actual spawn). Each
            # thread must engage with the tool layer (or delegate via
            # spawn) before committing to an answer.
            #
            # NOTE: We deliberately do NOT enforce a stronger "global
            # search must have run" rule. Earlier experiments showed it
            # over-constrains Thread's natural behavior on GAIA — when
            # children graceful-degrade without searching the GAIA env's
            # must_search rule will zero the answer downstream, but that
            # is a fair signal (Thread genuinely struggled), not something
            # we should hide with extra agent-side gymnastics.
            #
            # Rejection refunds state.t (no budget charge). After
            # MAX_FINISH_REJECTS retries on the same thread, allow anyway
            # (graceful degrade prevents infinite loop on stubborn LLMs).
            if (tool_name == "finish"
                    and not did_useful_action
                    and finish_reject_count < MAX_FINISH_REJECTS):
                state.t -= 1   # refund this LLM call from global budget
                finish_reject_count += 1
                print(
                    f"[Thread] Reject premature finish "
                    f"({finish_reject_count}/{MAX_FINISH_REJECTS}) at depth={depth}: "
                    f"thread has no search/open_page/spawn yet"
                )
                chat.append({
                    "role": "user",
                    "content": (
                        "Your `ACTION: finish(...)` was REJECTED — this "
                        "thread has not yet performed any search, open_page, "
                        "or spawn. You may NOT answer from memory.\n\n"
                        "Required next step: emit one of:\n"
                        "  - `ACTION: search(query)` to verify the answer "
                        "with a web search, OR\n"
                        "  - `=>sub-question<=` to delegate a sub-fact to a "
                        "child thread (which will itself search).\n\n"
                        "Then you can call ACTION: finish(value)."
                    ),
                })
                continue

            obs = await _dispatch_tool(env, tool_name, args)
            state.record_tool(tool_name, source_label)
            if tool_name in ("search", "open_page"):
                did_useful_action = True
            if tool_name == "finish":
                state.finished = True
                state.final_answer = args.strip().strip('"\'')
                return state.final_answer
            chat.append({"role": "user", "content": f"OBSERVATION: {obs}"})
            Y = Y + f"\nOBSERVATION: {obs}\n"
            continue

        # No spawn, no action, no end — degenerate. Nudge the LLM.
        chat.append({
            "role": "user",
            "content": (
                "I see no ACTION or spawn in your last response. "
                "If you have the answer, emit `ACTION: finish(value)`. "
                "If you need information, emit `ACTION: search(query)`."
            ),
        })
        # Don't break — give one more chance
        continue

    # Fallback returns
    if is_root:
        return state.final_answer or extract_final_answer(Y) or "(no answer)"
    return extract_final_answer(Y) or "(child timed out)"


async def process_item(
    item: DataProto,
    context: TaskContext,
    LLMClass=CallAPI,
) -> DataProto:
    """THREAD process_item — same signature as react/hymem/resum agents."""
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    ability = item.non_tensor_batch['ability'][0]
    EnvClass = select_env(ability, config)
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Thread] Environment init error: {e}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) \
               or getattr(config.plugin, "workflow", "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)

    host = context.server_host
    port = context.server_port
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))

    # Need an Agent wrapper for the dataproto export at the end. We mostly
    # bypass it during the actual thread run, but use it for messages capture.
    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=len(user_prompt))

    question = env.instance_info.get('problem_statement', '') \
               or item.non_tensor_batch.get('extra_info', [{}])[0].get('query', '')

    t_max = int(getattr(config.plugin, 'thread_t_max', DEFAULT_THREAD_T_MAX))
    max_depth = int(getattr(config.plugin, 'thread_max_depth', DEFAULT_MAX_DEPTH))
    max_new_tokens = int(getattr(config.plugin, 'thread_max_new_tokens', DEFAULT_MAX_NEW_TOKENS_PER_STEP))
    session_timeout = getattr(config.plugin, 'session_timeout', DEFAULT_SESSION_TIMEOUT)

    state = _ThreadState(t_max=t_max)
    start_time = time.time()

    try:
        root_prompt = get_thread_prompt(question)
        # Wrap the recursive thread in a session timeout
        final_answer = await asyncio.wait_for(
            _thread(
                root_prompt, state, env, llm_client, tokenizer, config,
                depth=0, max_depth=max_depth,
                max_new_tokens=max_new_tokens, is_root=True,
            ),
            timeout=session_timeout,
        )
    except asyncio.TimeoutError:
        print(f"[Thread] Session timeout after {session_timeout/60:.1f} min")
        final_answer = state.final_answer or "(timeout)"
    except Exception as e:
        print(f"[Thread] Unhandled error: {e}")
        final_answer = state.final_answer or f"(error: {e})"

    print(f"[Thread] Done. Final answer: {str(final_answer)[:120]}; "
          f"spawns={state.spawn_count}, max_depth={state.spawn_depth_max}, "
          f"t={state.t}/{t_max}")

    session_time = time.time() - start_time

    print('[Thread] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Thread] reward error: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    # === Unified stats schema (paper-consistent across all baselines) ===
    # Pre-write our richer tool_calls_by_tool so write_baseline_stats won't
    # overwrite it with the env-derived version.
    env.stats['tool_calls_by_tool'] = dict(state.tool_calls_by_tool)
    write_baseline_stats(
        env.stats,
        framework="thread",
        score=score[1],
        iterations=state.t,         # each thread "step" = one agent_turn
        total_prompt_tokens=state.total_prompt_tokens,
        total_completion_tokens=state.total_completion_tokens,
        session_time=session_time,
    )
    # Override tool_calls_by_source with our richer accounting (actor +
    # child_thread split — Thread-specific structural insight)
    env.stats['tool_calls_by_source'] = dict(state.tool_calls_by_source)
    env.stats['get_final_score'] = score[1]
    env.stats['messages'] = agent.messages()
    # Expose Thread's final answer through stats so cross-framework
    # eval scripts (eval_bc / eval_gaia) can extract a prediction without
    # parsing the chat history (Thread's chat is internal to _thread()
    # and agent.messages() only holds system+user).
    env.stats['final_answer'] = final_answer if isinstance(final_answer, str) else str(final_answer or '')
    # Thread-specific extras
    env.stats['thread_spawn_count'] = state.spawn_count
    # G3 — spawn success count (analog to HyMem branch_success). Lets §C
    # analysis compare HyMem branch success rate vs Thread spawn success rate.
    env.stats['thread_spawn_success'] = state.spawn_success
    env.stats['thread_max_depth_reached'] = state.spawn_depth_max
    env.stats['thread_t_used'] = state.t
    env.stats['thread_t_max'] = t_max
    # G4 — per-step context size trace (manually maintained — Thread
    # bypasses Agent class). Tuples = (t_step, depth, context_chars).
    env.stats['context_size_history'] = list(state.context_size_history)

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(
        out, item, messages, score, reward_dict,
        tag='main', metrics=agent.get_metrics(),
    )
    return DataProto.concat([out])
