# Baseline Reproduction Notes — HyMem ACL Rebuttal

This document captures the survey of three reviewer-requested baselines (Thread,
ReSum, A-MEM) before reimplementing them in our harness. For each baseline we
recorded the algorithm, the prompts, the upstream code path, the published
numbers, and the necessary simplifications for our training-free reproduction.

The goal is **honest, faithful reproduction** — not a strawman. Where our
re-implementation differs from the paper, the difference is documented here and
will be carried into the appendix of the camera-ready paper.

---

## Survey Summary Table

| 调研项 | Thread | A-MEM | ReSum |
|---|---|---|---|
| Paper | [arxiv 2405.17402](https://arxiv.org/abs/2405.17402), NAACL 2025 | [arxiv 2502.12110](https://arxiv.org/abs/2502.12110) | [arxiv 2509.13313](https://arxiv.org/abs/2509.13313) |
| Source | [philipmit/thread](https://github.com/philipmit/thread) | [agiresearch/A-mem](https://github.com/agiresearch/A-mem), MIT | [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) `WebAgent/WebResummer/` |
| 训练 | ❌ Few-shot | ❌ Embedding-based | ❌ training-free 模式（GRPO 可选） |
| 论文报告 GAIA | **没有** — 只在 ALFWorld/TextCraft/WebShop/DataCommons QA/MIMIC | **没有** — 主要在 LoCoMo/对话记忆benchmark | **有** — WebSailor-30B+ReSum: 47.3% (ReAct: 45.0%) text-only 103-sample subset |
| 是否需要 BMToolManager 替换 | 是（原版 action = Python exec；GAIA 需 web 工具） | 否（A-MEM 是 memory layer，agent 自己写） | 是（原版 = Serper+Jina；可直接换 BMToolManager） |
| Embedding 依赖 | 无 | `sentence-transformers/all-MiniLM-L6-v2` + `chromadb` | 无 |
| 复用我们 `CallAPI` | 较易（自己写 wrapper） | 中等（需改 `OpenAIController`） | 极易（已用 OpenAI SDK） |
| 与 HyMem 核心区别 | spawn 是 fresh-context 递归（only descriptor），HyMem branch 是 selective-inheritance | A-MEM 是 LLM 驱动的 note evolve（最近邻），HyMem 是固定结构 episode/working/tool memory | ReSum 周期触发 mid-trajectory summary（90% context），HyMem 用多触发 fold 机制 |
| 复现 sanity 目标（GAIA L1 with deepseek-reasoner，HyMem ≈ 71.7%） | 无 paper 数字参照，预期与 ReAct 接近 ± 5% | 无 paper 数字参照，预期与 ReAct 接近 ± 3% | ~75% 估算（按 paper 4.5% gap 比例外推） |
| 必要简化 | 用 GAIA few-shot 替换 DataCommons few-shot；action 路径换 web search | 用本地 sentence-transformers 替换可能的远程 embedding；用 `CallAPI` 替换 OpenAI/litellm 双路径 | 用 `HYMEM_PLANNER_MODEL` 替换 ReSumTool-30B；用 BMToolManager 替换 Serper+Jina |

---

## ReSum

### Method core

ReSum periodically invokes an external **summarizer** to compress accumulated
interaction history into a compact summary when the context approaches the
window limit (≥ 90% × 32K tokens) or after a fixed number of rounds. After the
summarizer fires, the agent restarts from a "compressed query" `q' = (original_q,
summary)` with a **fresh context** containing only the system prompt and the
new compressed user message. This is *orthogonal* to "summary at end" (our
existing Summary baseline) — it's mid-trajectory, repeatable, and resets the
context window.

### Trigger (verbatim from `external/DeepResearch/WebAgent/WebResummer/src/react_agent.py:135`)

```python
should_summarize = ((RESUM and token_count >= max_tokens * 0.9)
                    or round % summary_iteration == 0) \
                   and num_llm_calls_available
```

Two triggers OR'd:
- **Token-based**: `token_count >= 0.9 * 32K = 28.8K`
- **Iteration-based**: every `summary_iteration` rounds (configurable; main.py:22 default very high)

### Summarizer prompt

`external/DeepResearch/WebAgent/WebResummer/src/prompt.py:97-126` —
`QUERY_SUMMARY_PROMPT`. Asks the summarizer to extract only "explicitly
present" information from the conversation, output `<summary>...</summary>`.
Anti-hallucination instruction is heavy.

For subsequent (2nd+) summaries: `QUERY_SUMMARY_PROMPT_LAST` (lines 129-169)
adds "you saw a previous summary, here it is" + identifies information gaps.

### Summary injection (verbatim from `react_agent.py:146-153`)

```python
if summary_response:
    last_summary = summary_response
    new_observation = ("Question: " + question
        + "\nBelow is a summary of the previous conversation. ..."
        + "\nSummary: " + summary_response + "\n")
    messages = [
        {"role": "system", "content": self.system_message},
        {"role": "user", "content": new_observation}
    ]
```

**Key**: messages array is hard-reset to `[system, new_user_obs]`. Full prior
history is discarded. This is what makes it "restartable".

### Reported numbers (paper Table 1, Sec 4)

- **Setting**: GAIA 103-sample text-only subset, BrowseComp, BrowseComp-zh
- **Models**: WebSailor-30B base
- **Training-free ReSum vs ReAct on GAIA**: 47.3% vs 45.0% Pass@1 → **+2.3% absolute**
- Paper headlines "+4.5% across 3 benchmarks"
- **With ReSum-GRPO RL training (NOT our path)**: 48.5% on GAIA → +8.2% relative

### Our adaptation

| Aspect | Upstream | Our reimplementation |
|---|---|---|
| Summarizer model | ReSumTool-30B (distilled) | `HYMEM_PLANNER_MODEL` (deepseek-reasoner) |
| Search backend | Serper + Jina | `BMToolManager` (current online stack) |
| Trigger | 90% × 32K + every N rounds | Same (use 90% × `MAX_CONTEXT` from our config) |
| Prompt | Verbatim from `prompt.py` | Vendored to `agents/resum_prompts.py` with attribution |
| LLM client | OpenAI SDK | `CallAPI` (already OpenAI compatible) |
| RL fine-tuning | Optional (ReSum-GRPO) | Skipped — training-free only |

### Compatibility verdict

⭐⭐⭐⭐⭐ Cleanest of the three. ~150-200 LOC.

### Reproduction sanity target

Given HyMem on GAIA L1 ≈ 71.7% with deepseek-reasoner, ReSum on the same
backbone should land **at or modestly above ReAct** (which we have not yet
measured on this setup but estimate ~65%). If ReSum < 60%, something is wrong
(prompt mismatch / summary not firing / context corrupted on reset).

---

## Thread

### Method core

Thread frames LLM generation as a **thread of execution** where any thread can
recursively `spawn` child threads to offload sub-work. Children execute in
**fresh-ish context** (just the few-shot examples + the parent's spawn
descriptor) and return only the tokens needed by the parent. Spawning is
in-line (no XML tag) — the parent emits a special spawn open token (`=>`),
generation stops, the framework recursively calls itself for the child, the
child's output is wrapped in `=>...<=` and inserted into the parent's stream.

### Spawn syntax (verbatim from `external/thread/datacommons_qa/run_datacommons_qa.py:287`)

```python
stop_tokens_=['=>','#EN']
close_token='<='
```

When the LLM emits `=>`, generation halts. Example from a few-shot
(`external/thread/prompts/thread/datacommons_qa/thread_datacommons_qa.txt:7`):

```
Here is a table showing the percentage of people with diabetes
during 2018 to 2021 for Willows and Villa Ridge: =>{table_dir}<=.

Based on the data in {table_dir} from 2018 to 2021 the slope of
the percentage of people with diabetes in Willows is =>0.3357<=
and in Villa Ridge is =>0.1642<=.
```

So `=>{table_dir}<=` is a *spawn slot*: the descriptor `{table_dir}` is the
parent's request to a child (here: "find the table dir"); the child returns
the value (here: a path string, or for a sub-fact, a number like `0.3357`).
**Multiple spawns can appear in one parent response** (the table example has 2
spawns for slope-Willows / slope-VillaRidge).

### Recursion machinery (verbatim from `run_datacommons_qa.py:706-723`)

```python
def thread(c, Y):
    global t
    t = t+1
    if t < t_max:
        c_plus_Y = prompt0_examples + '\n\n\n\n\n\nPrompt:\n'+ c \
                   + '\n\n\nResponse:\n'+start_+'\n' + Y
        Y = Y + llm(c_plus_Y, stop=stop_tokens_, max_new_tokens_=max_new_tokens_task)
        thread_status = get_thread_status(Y)
        if thread_status == 'spawn':
            Y = Y + psi( thread( phi(Y), '') )    # ← recursive call
            if not done:
                return thread(c, Y)
        elif thread_status == 'action':
            action, observation = get_action_and_observation(Y)
            Y = Y + observation
            if not done:
                return thread(c, Y)
        else:
            return Y
```

`phi(parent)` extracts the spawn descriptor (the last line of parent's tokens
ending in `=>`). `psi(child_result)` wraps the child's output in `=>...<=`
markers. There's a **shared global recursion budget** `t_max` (no per-depth
limit; if any thread exceeds total budget, all halt).

### Child context (key paper-vs-HyMem contrast)

**Child sees only**:
1. The few-shot examples (`prompt0_examples`)
2. The parent's spawn descriptor (last line of parent's response, ending in `=>`)

**Child does NOT see**:
- The full parent reasoning above the spawn point
- Sibling spawns' results
- Any "memory store"

This is **fundamentally different from HyMem `<branch>`**, which inherits the
last 3-5 planner exchanges (selective inheritance) AND has full access to the
shared episode/working/tool memory store. **This contrast is the LMtp #1
core comparison point** — Thread's spawn is leaner / more isolated, HyMem's
branch is richer / more contextualized.

### Action vs Spawn

Thread also supports "action" — i.e., calling external tools (in upstream this
is Python `exec()`; the model emits Python code, framework executes it,
appends observation). For our GAIA reimplementation, we replace the Python
exec path with our `BMToolManager` (web_search / web_parse / parse_file).

### Reported numbers

**Thread paper does NOT report GAIA**. Benchmarks tested (paper Sec 4):

| Benchmark | THREAD GPT-4 | THREAD GPT-3.5 | THREAD Llama-3-8b |
|---|---|---|---|
| ALFWorld | 98.5% | 95.5% | 71.6% |
| TextCraft | — | 93.5% | 92.0% |
| WebShop SR | — | 49 | 47 |
| DataCommons QA | n/a (paper-only benchmark) | — | — |
| MIMIC-III ICU QA | n/a (paper-only) | — | — |

Headline: 10-50% absolute over ReAct on these benchmarks; on GAIA we have no
paper number to anchor against.

### Our adaptation

| Aspect | Upstream | Our reimplementation |
|---|---|---|
| Spawn markers | `=>` / `<=` inline | Same — preserves Thread's defining mechanism |
| Few-shot prompts | Per-benchmark (DataCommons / ALFWorld / etc.) | Write GAIA-specific spawn examples (~6-8 demos) |
| Action backend | Python `exec()` | `BMToolManager` (web tools) |
| Recursion budget | Shared global `t_max` | Same; default `t_max=20` (paper Appendix value) |
| LLM client | HuggingFace local + OpenAI | `CallAPI` (OpenAI SDK) |

### Compatibility verdict

⭐⭐⭐ Medium — algorithm itself is clean recursion, but few-shot prompts must
be re-authored for GAIA. ~250-300 LOC.

### Reproduction sanity target

No paper anchor. Expectation: Thread should hit roughly ReAct level ± 5pp on
GAIA L1 — Thread's strength is decomposable benchmarks (ALFWorld), and GAIA's
cross-source-verification questions may not benefit from fresh-context spawn.
If Thread < 50%, the spawn machinery probably isn't firing correctly.

---

## A-MEM

### Method core

A-MEM provides a **Zettelkasten-style agentic memory** as a CRUD layer (NOT an
agent). Three phases:
1. **Note creation**: each new piece of information becomes a `MemoryNote` with
   LLM-extracted metadata (`description`, `keywords`, `tags`).
2. **Linking**: ChromaDB top-k similarity (cosine) connects each new note to
   semantic neighbors via a `links` field.
3. **Evolution** (LLM-judged + threshold-triggered): every N notes (default
   100), an LLM looks at the new note + nearest neighbors and either
   `strengthen` (extend `links`/tags) or `update_neighbor` (rewrite neighbor
   contexts/tags in-place).

### Note schema (verbatim from `external/A-mem/agentic_memory/memory_system.py:24-81`)

```python
class MemoryNote:
    content: str
    id: str  # uuid
    timestamp: str  # YYYYMMDDHHmm
    keywords: List[str]
    context: str
    category: str
    tags: List[str]
    links: List[str]  # neighbor IDs
    last_accessed: str
    retrieval_count: int
    evolution_history: List
```

### Linking (verbatim from `retrievers.py:93-109`)

```python
class ChromaRetriever:
    def search(self, query: str, k: int = 5):
        return self.collection.query(query_texts=[query], n_results=k)
```

Uses `SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')` —
local embedding, no API calls.

### Evolution (verbatim from `memory_system.py:590-727`)

```python
def process_memory(self, note):
    neighbors = self.find_related_memories(note, top_k=5)
    response = self.llm.get_completion(EVOLUTION_PROMPT.format(...))
    if response['should_evolve']:
        if 'strengthen' in response['actions']:
            note.links.extend(response['suggested_connections'])
            note.tags.extend(response['suggested_tags'])
        if 'update_neighbor' in response['actions']:
            for nid in response['neighbors_to_update']:
                # rewrite neighbor's context/tags in-memory
```

### LLM client (verbatim from `llm_controller.py:13-37`)

```python
class OpenAIController(BaseLLMController):
    def __init__(self, model_name="gpt-4", api_key=None):
        self.client = openai.OpenAI(api_key=api_key)

    def get_completion(self, prompt, response_format, temperature=0.7):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
            temperature=temperature,
        ).choices[0].message.content
```

`openai.OpenAI(api_key=...)` — **no `base_url` parameter** by default. We need
to pass it through to point at our DeepSeek endpoint.

### A-MEM is a memory layer, NOT an agent

The package provides `add_note`, `read`, `update`, `delete`, `search_agentic`.
**There is no agent loop in upstream**. To use A-MEM on GAIA, we must:
- Write our own ReAct-style agent loop
- After each observation: call `amem.add_note(observation)`
- Before each action: call `amem.search_agentic(query, k=5)` and inject the
  returned notes into context

### Reported numbers

**A-MEM paper does NOT report GAIA**. Their benchmark is **LoCoMo** (long-form
conversation memory), reporting QA accuracy on multi-session dialogue tasks.
The GitHub README lists 6 LLM backbones tested, but no GAIA / agent-task
numbers in either paper or repo.

### Dependencies (verbatim from `external/A-mem/pyproject.toml:18-28`)

```
sentence-transformers>=2.2.2
chromadb>=0.4.22
rank_bm25>=0.2.2
nltk>=3.8.1
litellm>=1.16.11
numpy>=1.24.3
scikit-learn>=1.3.2
openai>=1.3.7
```

**New deps for our harness**: `chromadb`, `sentence-transformers`, `litellm`,
`rank_bm25`, `nltk`. Of these, `chromadb` is the heaviest (vector DB; we'll
use in-memory mode to avoid disk I/O). `litellm` we can skip (we use our own
`CallAPI`). `rank_bm25` and `nltk` are imported but not central; safe to keep.

### Module list to vendor (~1.1 KLOC)

| File | LOC | Required |
|---|---|---|
| `agentic_memory/memory_system.py` | 727 | ✅ Required (core) |
| `agentic_memory/llm_controller.py` | 97 | ✅ Required (will adapt) |
| `agentic_memory/retrievers.py` | 303 | ✅ Required (Chroma wrapper) |

### Our adaptation

| Aspect | Upstream | Our reimplementation |
|---|---|---|
| LLM client | OpenAI / Ollama dual-path | Adapter to `CallAPI` (OpenAI compat with custom `base_url`) |
| Embedding | `all-MiniLM-L6-v2` (sentence-transformers) | Same — local, ~80MB, no API |
| Vector DB | ChromaDB persistent | ChromaDB in-memory (per task; no cross-task leak) |
| Agent loop | Not provided | Write ReAct-style on top of vendored memory layer |
| Tool layer | Not provided | `BMToolManager` |
| Evolution threshold | 100 notes default | Same |
| Top-k retrieval | k=5 default | Same |

### Compatibility verdict

⭐⭐ Most complex of the three. Upstream is ~1.1KLoC vendored + ~200 LOC
adapter + ~200 LOC ReAct-with-A-MEM agent loop. Total ~1.5KLOC.

### Reproduction sanity target

No paper anchor for GAIA. A-MEM is memory-organization-focused; on agent tasks
where reasoning (not memory recall) is the bottleneck, expected gain over ReAct
is small. **Realistic prediction: A-MEM on GAIA L1 should land within ±3pp of
ReAct.** If A-MEM significantly under-performs ReAct, embedding/Chroma is
broken. If it over-performs by >5pp, suspicious — check for memory leak across
tasks.

---

## Why we're NOT implementing Mem1 / MemAgent

Both methods' **core contribution is RL fine-tuning** (Mem1: RL on consolidated
single-state buffer; MemAgent: RL on overwrite-policy across segments). Without
the trained weights, "prompt-only Mem1" is just a rolling-state buffer with
ReAct, and "prompt-only MemAgent" is just chunked summarization — both reduce
to weak versions of what ReSum already does in our experiments.

Doing them as prompt-only baselines would (a) misrepresent the methods to
reviewers and (b) provide no signal beyond what ReSum already gives. We will
note this in an appendix paragraph titled "Methods we explicitly do not
include and why."

---

## Implementation precedence (planned execution order)

1. **ReSum first** (cheapest, cleanest harness fit, reproducible numbers)
2. **Thread next** (medium — algorithm clean but new few-shot prompts needed)
3. **A-MEM last** (largest — vendoring + adapter + new agent loop)

After each baseline: smoke test 1 GAIA task, confirm stats schema is unified,
proceed to next.

---

## Paired metrics (HyMem ↔ each baseline) — for §A/§B/§C subsection analyses

Each baseline is a near-neighbor of one HyMem module. The "paired metrics"
below are the field-level handles that make per-pair comparison possible
when running the post-experiment analysis scripts. Every metric here is
recorded **automatically** during `eval_gaia.py` runs — no extra logging
plumbing needed at experiment time.

### §A — HyMem fold ↔ ReSum mid-trajectory summary

| Metric | HyMem field | ReSum field | Comparable? |
|---|---|---|---|
| Reset count | `fold_count` | `resum_summaries_fired` | ✅ same semantic |
| Per-event token snapshot | `hymem_fold_events[].{tokens_before, tokens_after_reset, trigger_reason}` | `resum_summary_events[].{round, tokens_before, tokens_after_reset, summary_len_chars}` | ✅ field-shape aligned |
| Trigger threshold | `TriggerConfig.planner_context_threshold` | `resum_threshold_tokens` | ✅ both = % of context cap |
| Context growth between resets | from `context_size_history` | from `context_size_history` | ✅ G4 |

§A claim to support: "HyMem 分层隔离 vs ReSum 周期 reset — HyMem 减少 context 不丢 evidence chain"
→ analysis: plot `tokens_before` per fold/summary event, side-by-side; compare accuracy on tasks where multiple resets fired.

### §B — HyMem typed memory ↔ A-MEM agentic notes

| Metric | HyMem field | A-MEM field | Comparable? |
|---|---|---|---|
| Memory writes | `hymem_memory_ops.{episode_writes, working_writes, tool_writes}` (sum) | `amem_notes_created` | ✅ both = "memory write events" |
| Memory retrievals | `hymem_memory_ops.retrievals` | `amem_retrievals` | ✅ aligned |
| Tool-call records | `hymem_memory_ops.tool_call_records` | (implicit in note content) | ⚠ HyMem-only granularity |
| Final memory size | `len(episode_memory) + len(working_memory) + len(tool_memory)` | `len(notes)` (in trace) | ✅ approximate |

§B claim to support: "HyMem 类型化确定性 memory ≥ A-MEM note 图，且 memory cost 更低"
→ analysis: scatter `(memory_ops, total_tokens) vs Pass@1` for both frameworks; show HyMem reaches same Pass@1 with lower memory_ops.

### §C — HyMem branch ↔ Thread spawn

| Metric | HyMem field | Thread field | Comparable? |
|---|---|---|---|
| Sub-problem invocations | `branch_calls` | `thread_spawn_count` | ✅ same semantic |
| Successful invocations | `branch_success` | `thread_spawn_success` | ✅ aligned (G3) |
| Recursion depth | always 1 (HyMem branches don't nest) | `thread_max_depth_reached` | ⚠ different by design |
| Source attribution of tool calls | `tool_calls_by_source.branch` | `tool_calls_by_source.child_thread` | ✅ aligned |

§C claim to support: "HyMem branch (selective inheritance + memory access) > Thread spawn (fresh context) on cross-source verification tasks"
→ analysis: subset Pass@1 on tasks needing parent context; HyMem branch_success rate vs Thread spawn_success rate.

### Cross-baseline — Context-growth dynamics (Figure 3a-revised)

Every framework now writes `context_size_history` to `env.stats`:
- HyMem: `[(turn_idx, context_token_len), ...]` for the Coordination Layer (planner_messages)
- ReAct/ReSum/A-MEM: `[(chat_msg_idx, context_token_len), ...]` from `Agent.context_size_history`
- Fold: `[{agent: name, history: [...]}]` per main + each branch sub-agent
- Thread: `[(t_step, depth, context_chars), ...]` (manual — bypasses Agent class)

Analysis: plot 6 lines on one figure; HyMem-coord should be flat-ish (≈ 1/2 ReAct slope per the original Figure 3a claim). Confirms "hierarchical isolation keeps strategic context clean" with full-baseline comparison.

---

## Unified stats schema (cross-baseline comparison contract)

Every baseline (HyMem, ReAct, Fold/Summary, ReSum, Thread, A-MEM) writes the
SAME 14 universally-comparable fields to `env.stats`. This is enforced by
`tests/test_baseline_schema.py` and ensured by the shared helper
`agents/tool_metrics.write_baseline_stats()` (which all 5 non-HyMem agents
call once at end-of-task).

### Universal fields — every baseline emits these

| Field | Type | Source |
|---|---|---|
| `get_final_score` | float ∈ {0, 1} | LLM-as-judge verdict (env.get_reward) |
| `tool_calls_atomic` | int | `UnifiedToolCounter` derives from env.stats search/open_page/etc. |
| `agent_turns` | int | LLM iteration count |
| `tool_calls_by_source` | dict | `{actor, branch, child_thread, replan, ...}` → atomic count |
| `tool_calls_by_tool` | dict | `{search: 5, open_page: 3, ...}` per-tool breakdown |
| `framework` | str | `hymem` / `react` / `fold` / `resum` / `thread` / `amem` |
| `tool_calls_total` | int | Legacy alias of `tool_calls_atomic` |
| `total_prompt_tokens` | int | All LLM prompt tokens (auto-accumulated by `Agent.cumulative_usage`) |
| `total_completion_tokens` | int | All LLM completion tokens |
| `total_tokens` | int | Sum of above two |
| `token_usage_by_component` | dict | HyMem: `{planner, executor, branch, memory}`; others: `{actor}` (single bucket) |
| `latency_by_component` | dict | HyMem: per-component LLM time; others: `{actor: session_time, total: session_time}` |
| `models_used` | dict | `{planner, executor, aux, judge}` resolved from env vars |
| `session_time` | float | Wall-clock seconds end-to-end |

### Architecture-specific fields — HyMem-only

These default to `0` / `{}` for non-HyMem baselines (correctly — those
concepts don't exist in those frameworks). `eval_gaia.py` reads them via
`stats.get(field, default)` with the appropriate default. Paper Table
caption uses `—` to denote "not applicable to this baseline architecture".

| Field | Default | What it means in HyMem |
|---|---|---|
| `planner_turns` | 0 | Coordination Layer iterations |
| `executor_calls` | 0 | Actor Layer dispatches |
| `branch_calls` | 0 | Isolated Reasoning Module invocations |
| `branch_success` | 0 | Branches that returned a `<return>` |
| `fold_count` | 0 | Memory compression triggers |
| `replan_count` | 0 | Confidence-driven replanning events |
| `routing_summary` | `{}` | LLMComplexityRouter stats (LLM vs fallback shares) |
| `distillation_stats` | `{}` | Information distillation pipeline aggregate |

### Why this matters for the rebuttal

- **LMtp Q4 (tool calls counting)**: All 6 frameworks now use `UnifiedToolCounter`
  with the same atomic-call definition. Cross-Table comparison is apples to apples.
- **LMtp comment (aux model size disclosure)**: All 6 frameworks emit `models_used`
  resolved from the same env vars. Reviewer sees `{planner: deepseek-reasoner,
  aux: Qwen3-32B, ...}` for every baseline, not just HyMem.
- **Bh2U W2 (efficiency metrics)**: All 6 frameworks emit `total_tokens`,
  `token_usage_by_component`, `latency_by_component`, and `session_time`.
  `cost_calculator.py` and `efficiency_analysis.py` produce LaTeX tables for
  every baseline without per-baseline shims.

### Helper API summary

```python
# Auto-accumulator (every Agent step contributes)
agent.cumulative_usage  # {prompt_tokens, completion_tokens, total_tokens, call_count}

# One-shot stats writer for non-HyMem baselines
from agents.tool_metrics import write_baseline_stats
write_baseline_stats(
    env.stats,
    framework="resum",                # | "react" | "fold" | "thread" | "amem"
    score=score[1],
    iterations=iteration,
    total_prompt_tokens=agent.cumulative_usage["prompt_tokens"],
    total_completion_tokens=agent.cumulative_usage["completion_tokens"],
    session_time=time.time() - start_time,
)

# HyMem-side equivalent (preserves per-component splits)
from agents.tool_metrics import merge_hymem_tool_stats
merge_hymem_tool_stats(env.stats, bm_tool_stats, agent_turns=agent_turns)
# HyMem ALSO writes models_used, latency_by_component (per-component),
# token_usage_by_component (per-component) directly — they override the
# coarser baseline defaults.
```
