#!/usr/bin/env bash
set -euo pipefail

# 公共参数
INDEX_PATH="benchmarks/data/indices/rpj_wiki/rpj_wiki"
NUM_QUERIES=20
BATCH_SIZE=128
LLM_MODEL="qwen3:4b"
TOP_K=3

# 日志目录（带时间戳）
LOG_DIR="logs/eval_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# dataset -> ef 列表
declare -A EF_MAP=(
  [nq_open.jsonl]="32 62 190"
  [trivia_qa.jsonl]="77 150 249"
  [gpqa.jsonl]="41 72 124"
  [hotpot_qa.jsonl]="137 299 1199"
)

# 按指定顺序遍历
ORDERED_DATASETS=(nq_open.jsonl trivia_qa.jsonl gpqa.jsonl hotpot_qa.jsonl)

for dataset in "${ORDERED_DATASETS[@]}"; do
  for ef in ${EF_MAP[$dataset]}; do
    log_file="${LOG_DIR}/${dataset%.jsonl}_ef${ef}.log"

    # 展示并记录将要执行的命令
    cmd=(python benchmarks/run_evaluation.py "$INDEX_PATH" \
      --num-queries "$NUM_QUERIES" \
      --ef "$ef" \
      --batch-size "$BATCH_SIZE" \
      --llm-model "$LLM_MODEL" \
      --top-k "$TOP_K" \
      --queries-file "$dataset")

    echo "=== Running dataset=${dataset} ef=${ef} ===" | tee -a "$log_file"
    printf 'CMD: '; printf '%q ' "${cmd[@]}" | tee -a "$log_file"; echo | tee -a "$log_file"

    # 同时输出到命令行和日志文件
    "${cmd[@]}" 2>&1 | tee -a "$log_file"

    echo | tee -a "$log_file"
  done
done

echo "All runs completed. Logs in: $LOG_DIR"