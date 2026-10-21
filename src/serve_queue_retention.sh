#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
policy="${1:?vllm, lfu or queue}"
mode="${2:?trace or bench}"
profile="${3:?targeted or ay}"
case "$policy" in vllm|lfu|queue) ;; *) exit 2 ;; esac
case "$mode" in trace|bench) ;; *) exit 2 ;; esac
case "$profile" in
  targeted) model_len=384; slots=1; blocks=32 ;;
  ay) model_len=512; slots=3; blocks=96 ;;
  *) exit 2 ;;
esac
model="${STUDY_MODEL:-Qwen/Qwen3-0.6B}"
export STUDY_WAITING_POLICY=fcfs
export STUDY_EVICTION_POLICY="$policy"
export STUDY_QUEUE_LOOKAHEAD=16
export STUDY_PARTIAL_PREFIX=0
export STUDY_SKIP_CONTINUE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_STUDY_TRACE=0
export VLLM_STUDY_TRACE_STDOUT=0
export VLLM_SERVER_DEV_MODE=0
result_root="${STUDY_QUEUE_RESULTS_ROOT:-$experiment_root/results/queue-retention}"
mkdir -p "$result_root/$profile/$policy"
run_dir="$(mktemp -d "$result_root/$profile/$policy/server-$mode-XXXXXX")"
echo "Server results: $run_dir"
export VLLM_STUDY_TRACE_DIR="$run_dir"
if [ "$mode" = trace ]; then export VLLM_STUDY_TRACE=1; fi
"$experiment_root/.venv/bin/vllm" serve "${STUDY_MODEL_PATH:-$model}"   --served-model-name "$model" --host 127.0.0.1 --port 8001 --seed 0   --worker-cls auto --max-model-len "$model_len" --max-num-seqs "$slots"   --max-num-batched-tokens 128 --block-size 16   --num-gpu-blocks-override "$blocks" --prefix-match-unit 16   --enable-prefix-caching --enable-chunked-prefill --no-async-scheduling   --enable-prompt-tokens-details 2>&1 | tee "$run_dir/server.log"
