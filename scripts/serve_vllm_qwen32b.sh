#!/usr/bin/env bash
# ============================================================================
# Launch vLLM serving Qwen3-32B for HyMem GAIA eval.
#
# The served-model-name MUST be "Qwen3-32B" and the port MUST be 8022 to match
# scripts/env_eval_local.sh (HYMEM_EXECUTOR_BASE_URL=http://localhost:8022/v1,
# HYMEM_EXECUTOR_MODEL=Qwen3-32B).
#
# Usage:
#   bash scripts/serve_vllm_qwen32b.sh              # foreground
#   nohup bash scripts/serve_vllm_qwen32b.sh > vllm.log 2>&1 &   # background
#
# Override via env, e.g.:
#   MODEL_PATH=/data/models/Qwen3-32B TP=4 MAX_LEN=131072 bash scripts/serve_vllm_qwen32b.sh
# ============================================================================
set -euo pipefail

# HF repo id, or a local path to downloaded weights.
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
PORT="${PORT:-8022}"
TP="${TP:-2}"                       # tensor-parallel = number of GPUs
MAX_LEN="${MAX_LEN:-40960}"         # Qwen3-32B native max (max_position_embeddings=40960).
                                    # For >40960 you must enable YaRN — see the
                                    # ROPE_SCALING block below.
GPU_UTIL="${GPU_UTIL:-0.90}"
API_KEY="${VLLM_API_KEY:-dummy}"    # matches HYMEM_EXECUTOR_API_KEY=dummy

# Pick which GPUs to use (edit as needed).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:0}"

echo "[vLLM] model=$MODEL_PATH  served-name=Qwen3-32B  port=$PORT  TP=$TP  max_len=$MAX_LEN"
echo "[vLLM] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Optional YaRN long-context extension. Only needed if MAX_LEN > 40960.
# Enable with:  USE_YARN=1 MAX_LEN=65536 bash scripts/serve_vllm_qwen32b.sh
ROPE_ARGS=()
if [ "${USE_YARN:-0}" = "1" ]; then
    # factor must cover MAX_LEN/40960 (65536/40960≈1.6 → 2.0 is safe headroom).
    YARN_FACTOR="${YARN_FACTOR:-2.0}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    ROPE_ARGS=(--rope-scaling "{\"rope_type\":\"yarn\",\"factor\":${YARN_FACTOR},\"original_max_position_embeddings\":40960}")
    echo "[vLLM] YaRN enabled: factor=$YARN_FACTOR (extends 40960 → $(python3 -c "print(int(40960*${YARN_FACTOR}))" 2>/dev/null || echo '?'))"
fi

exec vllm serve "$MODEL_PATH" \
  --served-model-name Qwen3-32B \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --api-key "$API_KEY" \
  "${ROPE_ARGS[@]}"
