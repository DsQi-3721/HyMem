#!/usr/bin/env bash
# Qwen3.6-27B API + GAIA HyMem ablation runner.
# Uses every text-only item from the combined GAIA validation metadata.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE="${ENV_FILE:-scripts/env_eval_local.sh}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing environment file: $ENV_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

# Keep online-search credentials out of the runner. On the server this file
# is populated separately with mode 0600; a caller may override its location.
SEARCH_ENV_FILE="${SEARCH_ENV_FILE:-.scratch/ablation_search_env.sh}"
if [[ -f "$SEARCH_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SEARCH_ENV_FILE"
fi

# Fill this value manually before launching GAIA. This runner intentionally
# ignores scripts/serper_keys.json and uses only this direct Serper key.
export SERPER_KEY="${SERPER_KEY:-}"
export SERPER_DISABLE_KEY_POOL="1"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# All HyMem generation roles use the same OpenAI-compatible Qwen3.6-27B API.
# Qwen3-VL is loaded by a separate OpenAI-compatible inference server. vLLM
# uses the model path as its default model ID, so keep MODEL_NAME and
# MODEL_PATH aligned unless --served-model-name is explicitly overridden.
# Keep the base URL at /v1: the OpenAI client appends /chat/completions.
GAIA_MODEL_PATH="${GAIA_MODEL_PATH:-/data/models/Qwen3-VL-30B-A3B-Instruct-FP8}"
GAIA_MODEL_NAME="${GAIA_MODEL_NAME:-$GAIA_MODEL_PATH}"
GAIA_MODEL_BASE_URL="${GAIA_MODEL_BASE_URL:-http://127.0.0.1:8022/v1}"
GAIA_MODEL_API_KEY="${GAIA_MODEL_API_KEY:-EMPTY}"

if [[ "$GAIA_MODEL_BASE_URL" == */chat/completions ]]; then
    echo "GAIA_MODEL_BASE_URL must end at /v1, not /chat/completions" >&2
    exit 1
fi

export HYMEM_PLANNER_MODEL="$GAIA_MODEL_NAME"
export HYMEM_EXECUTOR_MODEL="$GAIA_MODEL_NAME"
export HYMEM_AUX_MODEL="$GAIA_MODEL_NAME"
export HYMEM_PLANNER_BASE_URL="$GAIA_MODEL_BASE_URL"
export HYMEM_EXECUTOR_BASE_URL="$GAIA_MODEL_BASE_URL"
export HYMEM_AUX_BASE_URL="$GAIA_MODEL_BASE_URL"
export HYMEM_PLANNER_API_KEY="$GAIA_MODEL_API_KEY"
export HYMEM_EXECUTOR_API_KEY="$GAIA_MODEL_API_KEY"
export HYMEM_AUX_API_KEY="$GAIA_MODEL_API_KEY"

# Bypass the server's HTTP proxy for the private model endpoint while leaving
# the proxy available to the external judge and online-search services.
MODEL_AUTHORITY="${GAIA_MODEL_BASE_URL#*://}"
MODEL_HOSTPORT="${MODEL_AUTHORITY%%/*}"
MODEL_HOST="${MODEL_HOSTPORT%%:*}"
PROXY_BYPASS_HOSTS="${MODEL_HOST},localhost,127.0.0.1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${PROXY_BYPASS_HOSTS}"
export no_proxy="$NO_PROXY"

export SEARCH_MODE="online"
export USE_ONLINE_SEARCH="true"

GAIA_ROOT="${GAIA_ROOT:-/home/wangxinqi/datasets/GAIA}"
DATA_PATH="${DATA_PATH:-${GAIA_ROOT}/2023/validation/metadata.parquet}"
CONDA_ENV="${CONDA_ENV:-verl}"
NUM_WORKERS="${NUM_WORKERS:-4}"
# eval_gaia/CallAPI treats prompt_length + response_length as one total
# request budget. Keep the existing conservative 14336-token total by default;
# CTX_LEN may be raised only when the Qwen3-VL server is launched with a
# matching --max-model-len.
CTX_LEN="${CTX_LEN:-7168}"
# Keep retry behavior bounded and observable. The OpenAI SDK's hidden retry
# layer is disabled in the clients; these values control the single retry
# layer owned by HyMem.
export HYMEM_API_TIMEOUT="${HYMEM_API_TIMEOUT:-120}"
export HYMEM_API_MAX_ATTEMPTS="${HYMEM_API_MAX_ATTEMPTS:-2}"
export HYMEM_AUX_TIMEOUT="${HYMEM_AUX_TIMEOUT:-120}"
export HYMEM_AUX_MAX_ATTEMPTS="${HYMEM_AUX_MAX_ATTEMPTS:-2}"
export BM_LLM_TIMEOUT="${BM_LLM_TIMEOUT:-120}"
export BM_LLM_MAX_ATTEMPTS="${BM_LLM_MAX_ATTEMPTS:-2}"
GAIA_TOKENIZER_PATH="${GAIA_TOKENIZER_PATH:-$GAIA_MODEL_PATH}"
TAG="${TAG:-gaia_qwen3vl_30b_a3b_fp8_text_only}"
RUN_ID="${RUN_ID:-run1}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing GAIA metadata: $DATA_PATH" >&2
    exit 1
fi
if [[ -z "${JUDGE_API_KEY:-}" ]]; then
    echo "JUDGE_API_KEY is not set by $ENV_FILE" >&2
    exit 1
fi
if [[ -z "${SERPER_KEY:-}" ]]; then
    echo "Set SERPER_KEY in scripts/run_gaia_ablation.sh before launching" >&2
    exit 1
fi

run_one() {
    local ablation="$1"
    local output_dir="results/${TAG}/${RUN_ID}/${ablation}"
    local log_dir="logs/${TAG}/${RUN_ID}/${ablation}"

    echo "Preparing GAIA ablation: ${ablation}"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python scripts/eval_gaia.py \
        --framework hymem \
        --ablation "$ablation" \
        --data_path "$DATA_PATH" \
        --gaia_root "$GAIA_ROOT" \
        --model_name "$HYMEM_PLANNER_MODEL" \
        --aux_model "$HYMEM_AUX_MODEL" \
        --tokenizer_path "$GAIA_TOKENIZER_PATH" \
        --num_workers "$NUM_WORKERS" \
        --prompt_length "$CTX_LEN" \
        --response_length "$CTX_LEN" \
        --max_planner_turns 12 \
        --max_executor_turns 10 \
        --max_branch_turns 3 \
        --max_branches 5 \
        --max_folds 7 \
        --search_topk 5 \
        --search_mode online \
        --text_only \
        --output_dir "$output_dir" \
        --log_dir "$log_dir" \
        --resume
}

run_requested() {
    case "$1" in
        full|no_isolated_reasoning|no_memory_management)
            run_one "$1"
            ;;
        *)
            echo "Unknown ablation '$1'. Expected: full, no_isolated_reasoning, or no_memory_management." >&2
            exit 2
            ;;
    esac
}

if [[ "$#" -gt 1 ]]; then
    echo "Usage: bash $0 [full|no_isolated_reasoning|no_memory_management]" >&2
    exit 2
elif [[ "$#" -eq 1 ]]; then
    run_requested "$1"
else
    run_one full
    run_one no_isolated_reasoning
    run_one no_memory_management
fi

echo "Requested GAIA ablation run(s) complete: results/${TAG}/${RUN_ID}/"
