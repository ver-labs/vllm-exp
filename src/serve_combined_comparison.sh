#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
variant="${1:?baseline or combined}"
mode="${2:?trace or bench}"
profile="${3:?ay, targeted or temporal}"
case "$variant" in baseline|combined) ;; *) exit 2 ;; esac
case "$mode" in trace|bench) ;; *) exit 2 ;; esac
case "$profile" in
  ay) model_len=512; slots=3; blocks=96; batch=512 ;;
  targeted) model_len=512; slots=8; blocks=256; batch=512 ;;
  temporal) model_len=384; slots=3; blocks=32; batch=384 ;;
  *) exit 2 ;;
esac
export STUDY_WAITING_POLICY=fcfs
export STUDY_EVICTION_POLICY=vllm
export STUDY_PARTIAL_PREFIX=0
export STUDY_SKIP_CONTINUE=0
export STUDY_SKIP_LOOKAHEAD=16
export STUDY_SKIP_MAX_BYPASSES=4
export STUDY_SKIP_MAX_WAIT_S=30
unit=16
if [ "$variant" = combined ]; then
  export STUDY_EVICTION_POLICY=lfu
  export STUDY_PARTIAL_PREFIX=1
  export STUDY_SKIP_CONTINUE=1
  unit=1
fi
mkdir -p "results/combined/$profile/$variant"
run_dir="$(mktemp -d "$experiment_root/results/combined/$profile/$variant/server-$mode-XXXXXX")"
echo "Server results: $run_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
export VLLM_STUDY_TRACE_DIR="$run_dir"
export VLLM_STUDY_TRACE_STDOUT=0
export VLLM_STUDY_TRACE=0
export VLLM_SERVER_DEV_MODE=0
if [ "$mode" = trace ]; then export VLLM_STUDY_TRACE=1; fi
"$experiment_root/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8001 --seed 0 --worker-cls auto \
  --served-model-name Qwen/Qwen3-0.6B \
  --max-model-len "$model_len" --max-num-seqs "$slots" \
  --max-num-batched-tokens "$batch" \
  --block-size 16 --num-gpu-blocks-override "$blocks" \
  --prefix-match-unit "$unit" \
  --enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling \
  --enable-prompt-tokens-details \
  2>&1 | tee "$run_dir/server.log"
