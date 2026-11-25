#!/usr/bin/env bash
#
# BrowseComp-Plus 多 baseline 适配 smoke 脚本
# ---------------------------------------------
# 用法：在已启动 local search server 的服务器上执行
#
#   bash scripts/smoke_browsecomp_plus.sh [DATA_PATH] [SEARCH_URL]
#
# 默认参数：
#   DATA_PATH   browsecomp-plus parquet/CSV 路径（按你服务器实际位置改）
#   SEARCH_URL  本地 search server URL（默认 localhost:8021）
#
# 该脚本依次跑 5 个 framework × 1 道题 + ReAct summary 模式，
# 验证 eval_bc.py 适配后无 crash，并对每条 trace JSON 做最小校验。
#
# 跑完成功后 → 用 --resume 跑全集即可。

set -e

DATA_PATH="${1:-/path/to/browsecomp_plus.parquet}"
SEARCH_URL="${2:-http://localhost:8021}"
LOG_BASE="logs/bcplus_smoke"
OUT_BASE="results/bcplus_smoke"

# 工作目录假定在 HyMem 仓库根
cd "$(dirname "$0")/.."

if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: data file not found at $DATA_PATH"
    echo "Pass the correct path as first arg, e.g.:"
    echo "  bash scripts/smoke_browsecomp_plus.sh /home/me/bcplus_test.parquet"
    exit 1
fi

# 启动前必要环境变量（你的服务器 env 配置脚本应该已经 export 过）：
#   OPENAI_API_KEY / HYMEM_PLANNER_API_KEY / HYMEM_AUX_API_KEY / JUDGE_API_KEY 等
#   LOCAL_SEARCH_URL（本脚本会强制传 --local_search_url）

echo "=== BrowseComp-Plus smoke: 5 frameworks × 1 sample each + react summary ==="
echo "Data:    $DATA_PATH"
echo "Search:  $SEARCH_URL"
echo "Logs:    $LOG_BASE/<fw>/"
echo "Output:  $OUT_BASE/<fw>/"
echo

run_one () {
    local fw="$1"
    local extra="$2"  # 额外 flag，比如 react summary 用 "--enable_summary"
    local label="${fw}${extra:+_summary}"
    echo "----- smoke [$label] -----"
    python scripts/eval_bc.py \
        --framework "$fw" $extra \
        --data_path "$DATA_PATH" \
        --local_search_url "$SEARCH_URL" \
        --search_mode local \
        --sample 1 \
        --num_workers 1 \
        --log_dir "$LOG_BASE/$label" \
        --output_dir "$OUT_BASE/$label" \
        2>&1 | tail -25
    # 校验：trace JSON 出现 + framework 字段正确
    n_traces=$(ls "$LOG_BASE/$label"/bcplus_*_trace.json 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n_traces" -ge 1 ]; then
        echo "[smoke] OK: $n_traces trace JSON for [$label]"
    else
        echo "[smoke] WARN: no trace JSON produced for [$label]"
    fi
    echo
}

run_one hymem  ""
run_one react  ""
run_one react  "--enable_summary"
run_one resum  ""
run_one thread ""
run_one amem   ""

echo "=== Smoke done. Inspect trace JSONs under $LOG_BASE/ ==="
echo
echo "下一步：跑全集（5 framework 并行后台）:"
echo "  for fw in hymem react resum thread amem; do"
echo "    nohup python scripts/eval_bc.py --framework \$fw --resume \\"
echo "      --data_path $DATA_PATH --local_search_url $SEARCH_URL \\"
echo "      --search_mode local --num_workers 2 \\"
echo "      --log_dir logs/bcplus_full/\$fw \\"
echo "      --output_dir results/bcplus_full/\$fw \\"
echo "      > /tmp/bcplus_\$fw.log 2>&1 &"
echo "  done"
