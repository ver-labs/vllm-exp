#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
policy="${1:-}"
mode="${2:-trace}"
case "$policy" in vllm|lfu) ;; *) echo "Usage: bash src/serve_cache_eviction.sh vllm|lfu [trace|bench]"; exit 2 ;; esac
case "$mode" in trace|bench) ;; *) echo "Mode must be trace or bench"; exit 2 ;; esac
mkdir -p "results/cache-eviction/$policy"
run_dir="$(mktemp -d "$experiment_root/results/cache-eviction/$policy/server-$mode-XXXXXX")"
echo "Server results: $run_dir"
export STUDY_WAITING_POLICY=fcfs
export STUDY_EVICTION_POLICY="$policy"
export VLLM_LOGGING_LEVEL=INFO
export VLLM_STUDY_TRACE_DIR="$run_dir"
export VLLM_STUDY_TRACE_STDOUT=0
export VLLM_STUDY_TRACE=0
if [ "$mode" = trace ]; then export VLLM_STUDY_TRACE=1; fi
"$experiment_root/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8001 --seed 0 \
  --max-model-len 384 --max-num-seqs 3 --max-num-batched-tokens 128 \
  --block-size 16 --num-gpu-blocks-override 32 \
  --enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --enable-prompt-tokens-details \
  2>&1 | tee "$run_dir/server.log"
