#!/usr/bin/env bash
set -euo pipefail

# Absolute paths (adjust if needed)
PROMPTS_DIR="/home/tony/yichuan/leann/benchmarks/data/prompts_g5"
SCRIPT_PATH="/home/tony/yichuan/leann/benchmarks/generation_speed_bench.py"

# Common args
MAX_TOKENS=2048
OLLAMA_MODEL="qwen3:4b"
HF_MODEL="Qwen/Qwen3-4B"

# Logs
LOG_DIR="/home/tony/yichuan/leann/logs/speed_bench_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Scanning: $PROMPTS_DIR"

# Iterate all .txt files under PROMPTS_DIR
while IFS= read -r -d '' file; do
  base_name=$(basename "$file")
  stem_name="${base_name%.*}"

  # 1) Ollama
  log_ollama="${LOG_DIR}/${stem_name}_ollama.log"
  cmd_ollama=(python "$SCRIPT_PATH" \
    --path "$file" \
    --type ollama \
    --model "$OLLAMA_MODEL" \
    --max_tokens "$MAX_TOKENS")

  echo "=== Running (ollama) file=${file} model=${OLLAMA_MODEL} ===" | tee -a "$log_ollama"
  printf 'CMD: '; printf '%q ' "${cmd_ollama[@]}" | tee -a "$log_ollama"; echo | tee -a "$log_ollama"
  "${cmd_ollama[@]}" 2>&1 | tee -a "$log_ollama"
  echo | tee -a "$log_ollama"

  # 2) HF
  log_hf="${LOG_DIR}/${stem_name}_hf.log"
  cmd_hf=(python "$SCRIPT_PATH" \
    --path "$file" \
    --type hf \
    --model "$HF_MODEL" \
    --max_tokens "$MAX_TOKENS")

  echo "=== Running (hf) file=${file} model=${HF_MODEL} ===" | tee -a "$log_hf"
  printf 'CMD: '; printf '%q ' "${cmd_hf[@]}" | tee -a "$log_hf"; echo | tee -a "$log_hf"
  "${cmd_hf[@]}" 2>&1 | tee -a "$log_hf"
  echo | tee -a "$log_hf"

done < <(find "$PROMPTS_DIR" -type f -name '*.txt' -print0)


echo "All runs completed. Logs in: $LOG_DIR"


