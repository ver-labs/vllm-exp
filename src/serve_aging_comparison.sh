#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
policy="${1:?vllm, lfu or aged_lfu}"
mode="${2:?trace or bench}"
profile="${3:?shift or stable}"
case "$policy" in vllm|lfu|aged_lfu) ;; *) exit 2 ;; esac
case "$mode" in trace|bench) ;; *) exit 2 ;; esac
case "$profile" in shift|stable) ;; *) exit 2 ;; esac
export STUDY_WAITING_POLICY=fcfs
export STUDY_EVICTION_POLICY="$policy"
export STUDY_LFU_HALF_LIFE_BLOCKS=64
export STUDY_PARTIAL_PREFIX=0
export STUDY_SKIP_CONTINUE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_STUDY_TRACE=0
export VLLM_STUDY_TRACE_STDOUT=0
export VLLM_SERVER_DEV_MODE=0
mkdir -p "results/aging-lfu/$profile/$policy"
run_dir="$(mktemp -d "$experiment_root/results/aging-lfu/$profile/$policy/server-$mode-XXXXXX")"
echo "Server results: $run_dir"
export VLLM_STUDY_TRACE_DIR="$run_dir"
if [ "$mode" = trace ]; then export VLLM_STUDY_TRACE=1; fi
"$experiment_root/.venv/bin/vllm" serve "$STUDY_MODEL_PATH" \
  --served-model-name Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8001 --seed 0 \
  --worker-cls auto --max-model-len 384 --max-num-seqs 1 \
  --max-num-batched-tokens 128 --block-size 16 --num-gpu-blocks-override 32 \
  --prefix-match-unit 16 --enable-prefix-caching --enable-chunked-prefill \
  --no-async-scheduling --enable-prompt-tokens-details 2>&1 | tee "$run_dir/server.log"
