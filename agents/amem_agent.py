"""A-MEM baseline (training-free reproduction).

A-MEM is a Zettelkasten-style agentic memory subsystem (NOT a full agent).
Each new piece of information becomes a structured note (description /
keywords / tags), notes are connected via top-k embedding similarity, and
the LLM occasionally "evolves" old notes (rewriting their context/tags) in
light of new evidence.

Upstream provides only the memory layer; we wrap it in a ReAct-style agent
loop that:
  - Before each step: retrieve top-k relevant notes for the current query,
    inject them into the prompt as a "Memory:" block.
  - After each observation: ingest the observation into A-MEM via add_note().
  - On finish action: terminate.

Faithful to:
    Xu et al., "A-MEM: Agentic Memory for LLM Agents",
    arXiv 2502.12110, Feb 2025.
    Code: https://github.com/agiresearch/A-mem (MIT, vendored)

Adaptation notes (documented in docs/baselines.md):
    - Vendored modules in agents/amem/ (memory_system.py, retrievers.py,
      llm_controller.py — last patched for base_url support).
    - LLM backend: OpenAI-compatible via HYMEM_AUX_MODEL / HYMEM_AUX_BASE_URL
      (no new env vars introduced).
    - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (upstream
      default, ~80MB, downloads to HF cache on first use).
    - Vector DB: ChromaDB in-memory mode, fresh instance per task (no
      cross-task leak — A-MEM is "per-query" memory in our setup).
    - Original A-MEM paper benchmarks are dialogue-memory tasks (LoCoMo);
      no GAIA numbers reported. Our reproduction is a faithful port of the
      memory layer, plugged into a ReAct loop with our BMToolManager search
      tools.

Heavy deps (chromadb, sentence-transformers) installed via:
    pip install -r requirements_amem.txt
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np  # noqa: F401
import torch  # noqa: F401

from verl import DataProto
from .utils import CallLLM, Agent, select_env, TaskContext, run_action, CallAPI
from .prompts import create_chat
from .tool_metrics import UnifiedToolCounter, write_baseline_stats


DEFAULT_SESSION_TIMEOUT = 150 * 60
DEFAULT_MIN_RESPONSE_TOKENS = 4096
DEFAULT_TOP_K = 5
DEFAULT_EVO_THRESHOLD = 100   # paper default
DEFAULT_AMEM_LLM_MODEL = None  # falls back to HYMEM_AUX_MODEL / HYMEM_PLANNER_MODEL


def _build_amem_system(model_name: str, llm_model: str, evo_threshold: int,
                       collection_name: Optional[str] = None):
    """Lazily import + initialize A-MEM. Heavy import (chromadb,
    sentence-transformers) is done here so plain HyMem runs aren't slowed.
    Raises with a clear message if amem deps aren't installed.

    `collection_name` should be UNIQUE per task — the upstream A-MEM
    hardcoded "memories" + `client.reset()` clobbers sibling workers'
    collections when eval_gaia runs N workers in parallel.
    """
    try:
        from .amem import AgenticMemorySystem
    except ImportError as e:
        raise RuntimeError(
            "A-MEM dependencies missing. Install with:\n"
            "  pip install -r requirements_amem.txt\n"
            f"Original error: {e}"
        )
    return AgenticMemorySystem(
        model_name=model_name,
        llm_backend="openai",
        llm_model=llm_model,
        evo_threshold=evo_threshold,
        api_key=os.getenv("HYMEM_AUX_API_KEY") or os.getenv("OPENAI_API_KEY"),
        collection_name=collection_name,
    )


def _format_memory_for_prompt(retrieved: List[Dict[str, Any]]) -> str:
    """Render A-MEM's search_agentic results as a compact text block to
    inject before the next planner step. Skips empty fields."""
    if not retrieved:
        return ""
    lines = ["[Retrieved memory notes — most relevant to the current query]"]
    for i, note in enumerate(retrieved[:5], start=1):
        # The exact dict shape from search_agentic varies across A-MEM
        # versions. Be defensive about field names.
        content = note.get("content") or note.get("text") or ""
        keywords = note.get("keywords") or []
        context = note.get("context") or ""
        if isinstance(keywords, list):
            kw_str = ", ".join(keywords[:6])
        else:
            kw_str = str(keywords)
        line = f"  [{i}] {content[:300]}"
        if kw_str:
            line += f"  (keywords: {kw_str})"
        if context and context != "General":
            line += f"  (context: {context[:80]})"
        lines.append(line)
    return "\n".join(lines) + "\n"


async def process_item(
    item: DataProto,
    context: TaskContext,
    LLMClass=CallAPI,
) -> DataProto:
    """A-MEM process_item — same signature as react/hymem/resum agents."""
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
        print(f"[A-MEM] Environment init error: {e}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) \
               or getattr(config.plugin, "workflow", "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = agent_config.get("max_turn", 64)

    host = context.server_host
    port = context.server_port
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))

    prompt_turn = len(user_prompt)
    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)

    question = env.instance_info.get('problem_statement', '') \
               or item.non_tensor_batch.get('extra_info', [{}])[0].get('query', '')

    # === Initialize A-MEM memory system (per-task, in-memory) ===
    amem_llm_model = (
        getattr(config.plugin, 'amem_llm_model', None)
        or os.getenv('HYMEM_AUX_MODEL')
        or os.getenv('HYMEM_PLANNER_MODEL')
        or 'gpt-4o-mini'
    )
    amem_embedding_model = getattr(
        config.plugin, 'amem_embedding_model',
        'sentence-transformers/all-MiniLM-L6-v2',
    )
    amem_evo_threshold = int(getattr(config.plugin, 'amem_evo_threshold', DEFAULT_EVO_THRESHOLD))
    top_k = int(getattr(config.plugin, 'amem_top_k', DEFAULT_TOP_K))

    # Build a UNIQUE chromadb collection name per task. Combines task
    # instance_id + a short uuid suffix so even running the same task in
    # parallel workers (re-runs) doesn't clobber. Avoids the sibling-worker
    # collection-clobber bug when --num_workers >= 2.
    import uuid as _uuid
    _instance_id = (
        item.non_tensor_batch.get('extra_info', [{}])[0].get('instance_id', 'task')
    )
    _safe_id = ''.join(c if c.isalnum() else '_' for c in str(_instance_id))[:24]
    _collection = f"amem_{_safe_id}_{_uuid.uuid4().hex[:8]}"

    try:
        memory = _build_amem_system(
            amem_embedding_model, amem_llm_model, amem_evo_threshold,
            collection_name=_collection,
        )
        amem_available = True
        print(f"[A-MEM] Initialized: embedding={amem_embedding_model}, "
              f"llm={amem_llm_model}, evo_threshold={amem_evo_threshold}, "
              f"collection={_collection}")
    except RuntimeError as e:
        print(f"[A-MEM] FATAL: {e}")
        memory = None
        amem_available = False

    # Track A-MEM-specific metrics
    n_notes_added = 0
    n_retrievals = 0

    # Configuration
    session_timeout = getattr(config.plugin, 'session_timeout', DEFAULT_SESSION_TIMEOUT)
    max_model_len = config.prompt_length + config.response_length
    min_response_tokens = getattr(config.plugin, 'min_response_tokens', DEFAULT_MIN_RESPONSE_TOKENS)

    iteration = 0
    start_time = time.time()
    # Tokens auto-accumulated by Agent.append (utils.py). The A-MEM memory
    # layer also calls the LLM internally for note extraction / evolution;
    # those calls go through OpenAI SDK directly (vendored llm_controller),
    # not via Agent — they are NOT in agent.cumulative_usage. We don't have
    # a hook to capture them precisely; for paper purposes, we report
    # 'agent-side' tokens here and note this limitation in docs/baselines.md.

    while iteration < max_turn:
        iteration += 1

        elapsed = time.time() - start_time
        if elapsed > session_timeout:
            print(f'[A-MEM] [TIMEOUT] Session exceeded {session_timeout/60:.1f} min')
            break

        current_context = agent.context()
        current_len = len(current_context)
        remaining = max_model_len - current_len
        if remaining < min_response_tokens:
            print(f'[A-MEM] [TOKEN LIMIT] Context {current_len} too long, ending.')
            break

        response = await agent.step()
        if response is None:
            break
        observation = await run_action(env, response)
        if observation is None:
            break

        # === Memory ingestion: store this turn's observation as a note ===
        if amem_available and observation:
            try:
                obs_for_note = str(observation)[:2000]
                memory.add_note(content=obs_for_note)
                n_notes_added += 1
            except Exception as e:
                print(f"[A-MEM] add_note failed: {e}")

        # === Memory retrieval: prepare memory snippet for the NEXT step ===
        # Inject memory INLINE into the observation message (NOT as a
        # separate user message — consecutive user messages break Qwen's
        # apply_chat_template and crash agent.dataproto()).
        memory_block = ""
        if amem_available and n_notes_added > 0:
            try:
                retrieved = memory.search_agentic(question, k=top_k) or []
                n_retrievals += 1
                memory_block = _format_memory_for_prompt(retrieved)
            except Exception as e:
                print(f"[A-MEM] retrieval failed: {e}")
                memory_block = ""

        if memory_block:
            augmented_obs = (
                f"{memory_block}\n"
                f"[Use the notes above if relevant to plan your next action.]\n\n"
                f"OBSERVATION:\n{observation}"
            )
        else:
            augmented_obs = observation
        agent.append({'role': 'user', 'content': augmented_obs})

    session_time = time.time() - start_time

    print('[A-MEM] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[A-MEM] reward error: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    # === Unified stats schema (paper-consistent across all baselines) ===
    write_baseline_stats(
        env.stats,
        framework="amem",
        score=score[1],
        iterations=iteration,
        total_prompt_tokens=agent.cumulative_usage["prompt_tokens"],
        total_completion_tokens=agent.cumulative_usage["completion_tokens"],
        session_time=session_time,
    )
    env.stats['get_final_score'] = score[1]
    env.stats['total_turns'] = len(agent.messages())
    env.stats['messages'] = agent.messages()
    # G4 — per-turn context size trace for cross-baseline Figure 3a-revised
    env.stats['context_size_history'] = list(agent.context_size_history)
    # A-MEM-specific extras
    env.stats['amem_notes_created'] = n_notes_added
    env.stats['amem_retrievals'] = n_retrievals
    env.stats['amem_available'] = amem_available

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(
        out, item, messages, score, reward_dict,
        tag='main', metrics=agent.get_metrics(),
    )
    return DataProto.concat([out])
