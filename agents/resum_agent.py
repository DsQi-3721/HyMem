"""ReSum baseline (training-free reproduction).

Algorithm: ReAct loop + periodic mid-trajectory summarization that *resets* the
chat history. When the context approaches the window limit (default 90%),
the summarizer condenses the entire interaction so far into a compact summary;
the agent then restarts with `[system_prompt, restart_message_with_summary]`
as its only context, freeing up budget for further exploration.

Faithful to:
    Wu et al., "ReSum: Unlocking Long-Horizon Search Intelligence via
    Context Summarization", arXiv 2509.13313, Sep 2025.

Differences from upstream (documented in docs/baselines.md):
    - Summarizer model: HYMEM_PLANNER_MODEL (deepseek-reasoner) instead of
      ReSumTool-30B (their distilled summarizer). Paper notes generic LLMs
      also work, just with smaller gain.
    - Search tools: BMToolManager (our wrapper) instead of Serper+Jina —
      same set of capabilities (web_search, web_parse).
    - No ReSum-GRPO RL fine-tuning (training-free path only).

The vendored summarizer prompts live in agents/resum_prompts.py with full
verbatim text from upstream and a "do not modify" note.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np  # noqa: F401 — required by env stack
import torch  # noqa: F401 — required by env stack

from verl import DataProto
from .utils import CallLLM, Agent, select_env, TaskContext, run_action, CallAPI
from .prompts import create_chat
from .tool_metrics import UnifiedToolCounter, write_baseline_stats
from .resum_prompts import (
    QUERY_SUMMARY_PROMPT,
    QUERY_SUMMARY_PROMPT_LAST,
    RESUM_RESTART_MESSAGE,
    extract_summary,
)


DEFAULT_SESSION_TIMEOUT = 150 * 60  # 150 min
DEFAULT_MIN_RESPONSE_TOKENS = 4096
DEFAULT_RESUM_THRESHOLD_RATIO = 0.9   # paper default
DEFAULT_RESUM_MAX_SUMMARIES = 4       # bound runaway loops


def _format_history_for_summary(messages: List[Dict[str, str]], max_chars: int = 30000) -> str:
    """Render a chat history into a single text block for the summarizer.
    Keeps the original messages format (role: content), tail-truncated.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get('role', '')
        content = (m.get('content') or '').strip()
        if not content:
            continue
        parts.append(f"### {role}\n{content}")
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        # Keep the tail (most recent) since older content tends to be already
        # superseded; ReSum's anti-hallucination prompt strictly forbids
        # making things up, so partial history is safe.
        text = "...(earlier history truncated)...\n\n" + text[-max_chars:]
    return text


async def _call_summarizer(
    summarizer_client,
    tokenizer,
    config,
    *,
    question: str,
    history_messages: List[Dict[str, str]],
    last_summary: Optional[str],
):
    """Run one summarization pass. Returns (summary_text_or_None, usage_dict).
    Picks the prompt template based on whether there's a prior summary.
    """
    history_str = _format_history_for_summary(history_messages)
    if last_summary:
        prompt = QUERY_SUMMARY_PROMPT_LAST.format(
            question=question,
            last_summary=last_summary,
            recent_history_messages=history_str,
        )
    else:
        prompt = QUERY_SUMMARY_PROMPT.format(
            question=question,
            recent_history_messages=history_str,
        )

    # Build a one-shot Agent dedicated to summarization. The Agent class wraps
    # llm_client.create_completion with the same calling convention used by
    # the main planner — keeps us inside the harness's stats accounting.
    summarizer_messages = [
        {'role': 'system', 'content': 'You are a careful summarizer. Output only the requested format.'},
        {'role': 'user', 'content': prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            summarizer_messages, add_generation_prompt=True, tokenize=True
        )
        completion = await summarizer_client.create_completion(
            input_ids, messages=summarizer_messages,
            max_new_tokens=getattr(config.plugin, 'resum_summary_max_tokens', 1024),
        )
        if completion is None:
            return None, None
        text = completion["choices"][0]["message"].get("content") or ""
        if not text:
            msg = completion["choices"][0]["message"]
            text = getattr(msg, "reasoning_content", "") or msg.get("reasoning_content", "") or ""
        usage = completion["choices"][0]["message"].get("metrics", {}).get("usage", {})
        return (extract_summary(text) if text else None), usage
    except Exception as e:
        print(f"[ReSum] summarizer call failed: {e}")
        return None, None


async def process_item(
    item: DataProto,
    context: TaskContext,
    LLMClass=CallAPI,
) -> DataProto:
    """ReSum process_item — same signature as react_agent / hymem_agent."""
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
        print(f"[ReSum] Environment init error: {e}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) \
               or getattr(config.plugin, "workflow", "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = agent_config.get("max_turn", 64)

    host = context.server_host
    port = context.server_port
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))
    # Summarizer reuses the same client — it's another OpenAI-compat call.
    # Could in principle point at a different model via RESUM_SUMMARIZER_MODEL,
    # but we keep parity with the planner per "no new env vars" rule.
    summarizer_client = llm_client

    prompt_turn = len(user_prompt)
    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)

    # Original question for the summarizer prompt template.
    question = env.instance_info.get('problem_statement', '') \
               or item.non_tensor_batch.get('extra_info', [{}])[0].get('query', '')

    # Configuration
    session_timeout = getattr(config.plugin, 'session_timeout', DEFAULT_SESSION_TIMEOUT)
    max_model_len = config.prompt_length + config.response_length
    min_response_tokens = getattr(config.plugin, 'min_response_tokens', DEFAULT_MIN_RESPONSE_TOKENS)
    max_context_tokens = getattr(config.plugin, 'max_context_tokens', None) or (max_model_len - min_response_tokens)
    resum_threshold_ratio = float(getattr(config.plugin, 'resum_threshold_ratio', DEFAULT_RESUM_THRESHOLD_RATIO))
    resum_threshold_tokens = int(max_context_tokens * resum_threshold_ratio)
    resum_max_summaries = int(getattr(config.plugin, 'resum_max_summaries', DEFAULT_RESUM_MAX_SUMMARIES))

    # State for summarization tracking
    last_summary: Optional[str] = None
    n_summaries = 0
    summary_events: List[Dict] = []  # {round, tokens_before, tokens_after_reset}

    iteration = 0
    start_time = time.time()
    # Tokens auto-accumulated by Agent.append (see agents/utils.py
    # Agent.cumulative_usage). Summarizer LLM calls happen outside the
    # Agent — we tally those separately.
    summarizer_prompt_tokens = 0
    summarizer_completion_tokens = 0

    while iteration < max_turn:
        iteration += 1

        # Timeout
        elapsed = time.time() - start_time
        if elapsed > session_timeout:
            print(f'[ReSum] [TIMEOUT] Session exceeded {session_timeout/60:.1f} min')
            break

        # Context check
        current_context = agent.context()
        current_len = len(current_context)
        remaining = max_model_len - current_len

        # === ReSum trigger: context approaching limit ===
        # Paper: 90% of context window. We also bound to resum_max_summaries
        # to avoid runaway loops on pathological tasks.
        should_summarize = (
            current_len >= resum_threshold_tokens
            and n_summaries < resum_max_summaries
            and iteration > 1   # never summarize before any rounds
        )

        if should_summarize:
            tokens_before = current_len
            print(f"[ReSum] Trigger: context={tokens_before} >= "
                  f"{resum_threshold_tokens} (90% of {max_context_tokens}), "
                  f"summarization #{n_summaries + 1}")

            history_msgs = agent.messages()
            summary, summary_usage = await _call_summarizer(
                summarizer_client, tokenizer, config,
                question=question,
                history_messages=history_msgs,
                last_summary=last_summary,
            )
            # Summarizer LLM call counts toward token budget too
            if summary_usage:
                summarizer_prompt_tokens += int(summary_usage.get('prompt_tokens', 0) or 0)
                summarizer_completion_tokens += int(summary_usage.get('completion_tokens', 0) or 0)
            if summary:
                # Reset agent's chat to [system, restart_message_with_summary]
                # — paper Algorithm 1 line 20-21 (compressed state q' = (q, s))
                restart_user_msg = RESUM_RESTART_MESSAGE.format(
                    question=question, summary=summary,
                )
                # The Agent class doesn't expose a reset(); we mutate .chat
                # directly. The first message is the system prompt (set at
                # construction); keep it, drop everything else, append restart.
                if hasattr(agent, 'chat'):
                    sys_msgs = [m for m in agent.chat if m.get('role') == 'system']
                    if not sys_msgs and len(user_prompt) > 0:
                        sys_msgs = [user_prompt[0]] if user_prompt[0].get('role') == 'system' else []
                    agent.chat = list(sys_msgs) + [{'role': 'user', 'content': restart_user_msg}]

                tokens_after = len(agent.context())
                summary_events.append({
                    'round': iteration,
                    'tokens_before': tokens_before,
                    'tokens_after_reset': tokens_after,
                    'summary_len_chars': len(summary),
                })
                last_summary = summary
                n_summaries += 1
                print(f"[ReSum] Reset done: {tokens_before} -> {tokens_after} tokens")
            else:
                # Summarizer failed — log and continue without resetting
                # (better to overflow than to lose the trace).
                print(f"[ReSum] Summarizer returned None; continuing without reset")

        # Re-check after potential summarization
        current_context = agent.context()
        current_len = len(current_context)
        remaining = max_model_len - current_len

        if remaining < min_response_tokens:
            print(f'[ReSum] [TOKEN LIMIT] Context {current_len} tokens, '
                  f'remaining {remaining} < need {min_response_tokens}; ending.')
            break

        # Standard ReAct step. Token usage auto-accumulated into
        # agent.cumulative_usage by Agent.append (see utils.py).
        response = await agent.step()
        if response is None:
            break
        observation = await run_action(env, response)
        if observation is None:
            break
        agent.append({'role': 'user', 'content': observation})

    session_time = time.time() - start_time

    print('[ReSum] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[ReSum] reward error: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    # === Unified stats schema (every baseline writes the same fields so
    # the paper's cross-framework Tables / cost analysis / efficiency plot
    # all work uniformly without per-baseline shims). ===
    # Total = main agent steps (auto-accumulated) + summarizer LLM calls
    total_pt = agent.cumulative_usage["prompt_tokens"] + summarizer_prompt_tokens
    total_ct = agent.cumulative_usage["completion_tokens"] + summarizer_completion_tokens
    write_baseline_stats(
        env.stats,
        framework="resum",
        score=score[1],
        iterations=iteration,
        total_prompt_tokens=total_pt,
        total_completion_tokens=total_ct,
        session_time=session_time,
    )
    env.stats['get_final_score'] = score[1]
    env.stats['total_turns'] = len(agent.messages())
    env.stats['messages'] = agent.messages()
    # G4 — per-turn context size trace for cross-baseline Figure 3a-revised
    env.stats['context_size_history'] = list(agent.context_size_history)
    # ReSum-specific extras
    env.stats['resum_summaries_fired'] = n_summaries
    env.stats['resum_summary_events'] = summary_events
    env.stats['resum_threshold_tokens'] = resum_threshold_tokens

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(
        out, item, messages, score, reward_dict,
        tag='main', metrics=agent.get_metrics(),
    )
    return DataProto.concat([out])
