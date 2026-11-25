#!/usr/bin/env bash
# DeepSeek + BrowseComp-Plus HyMem ablation runner.
# Uses the complete balanced 150-task BrowseComp-Plus test set.
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

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Freeze the model-role assignment used by the DeepSeek main experiment.
export HYMEM_PLANNER_MODEL="deepseek-reasoner"
export HYMEM_EXECUTOR_MODEL="deepseek-chat"
export HYMEM_AUX_MODEL="deepseek-chat"
export JUDGE_MODEL="deepseek-reasoner"
export SEARCH_MODE="local"
export USE_ONLINE_SEARCH="false"

DATA_PATH="${DATA_PATH:-/home/wangxinqi/datasets/bc_test.parquet}"
LOCAL_SEARCH_URL="${LOCAL_SEARCH_URL:-http://localhost:8021}"
CONDA_ENV="${CONDA_ENV:-verl}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TAG="${TAG:-bcplus_deepseek}"
RUN_ID="${RUN_ID:-run1}"

if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing BrowseComp-Plus dataset: $DATA_PATH" >&2
    exit 1
fi
for required_var in HYMEM_PLANNER_BASE_URL HYMEM_PLANNER_API_KEY \
                    HYMEM_EXECUTOR_BASE_URL HYMEM_EXECUTOR_API_KEY \
                    HYMEM_AUX_BASE_URL HYMEM_AUX_API_KEY \
                    JUDGE_BASE_URL JUDGE_API_KEY; do
    if [[ -z "${!required_var:-}" ]]; then
        echo "$required_var is not set by $ENV_FILE" >&2
        exit 1
    fi
done

run_one() {
    local ablation="$1"
    local output_dir="results/${TAG}/${RUN_ID}/${ablation}"
    local log_dir="logs/${TAG}/${RUN_ID}/${ablation}"

    echo "Preparing BrowseComp-Plus ablation: ${ablation}"
    conda run --no-capture-output -n "$CONDA_ENV" \
        python scripts/eval_bc.py \
        --framework hymem \
        --ablation "$ablation" \
        --data_path "$DATA_PATH" \
        --data_format parquet \
        --model_name "$HYMEM_PLANNER_MODEL" \
        --hymem_aux_model "$HYMEM_AUX_MODEL" \
        --num_workers "$NUM_WORKERS" \
        --prompt_length 65536 \
        --response_length 65536 \
        --hymem_max_planner_turns 12 \
        --hymem_max_executor_turns 10 \
        --hymem_max_branch_turns 3 \
        --hymem_max_branches 5 \
        --hymem_max_folds 5 \
        --hymem_search_topk 5 \
        --search_mode local \
        --local_search_url "$LOCAL_SEARCH_URL" \
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

echo "Requested BrowseComp-Plus ablation run(s) complete: results/${TAG}/${RUN_ID}/"
