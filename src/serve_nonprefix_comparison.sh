#!/usr/bin/env bash
set -euo pipefail
experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
variant="${1:?baseline or repair}"
mode="${2:-bench}"
case "$variant" in baseline|repair) ;; *) exit 2 ;; esac
case "$mode" in trace|bench) ;; *) exit 2 ;; esac
mkdir -p "results/nonprefix-vllm/$variant"
run_dir="$(mktemp -d "$experiment_root/results/nonprefix-vllm/$variant/server-$mode-XXXXXX")"
echo "Server results: $run_dir"
export PYTHONPATH="$experiment_root/src${PYTHONPATH:+:$PYTHONPATH}"
export STUDY_WAITING_POLICY=fcfs
export STUDY_EVICTION_POLICY=vllm
export VLLM_LOGGING_LEVEL=INFO
export VLLM_STUDY_TRACE_DIR="$run_dir"
export VLLM_STUDY_TRACE_STDOUT=0
export VLLM_STUDY_TRACE=0
if [ "$mode" = trace ]; then export VLLM_STUDY_TRACE=1; fi
worker_args=(--worker-cls auto)
if [ "$variant" = repair ]; then
  worker_args=(--worker-cls study_nonprefix_worker.StudyNonPrefixWorker)
  export STUDY_NONPREFIX_RATIO="${STUDY_NONPREFIX_RATIO:-0.1}"
  export STUDY_NONPREFIX_EVENTS="$run_dir/nonprefix-events.jsonl"
fi
"$experiment_root/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 --port 8001 --seed 0 \
  --max-model-len 512 --max-num-seqs 3 --max-num-batched-tokens 128 \
  --block-size 16 --num-gpu-blocks-override 96 \
  --enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --enable-prompt-tokens-details "${worker_args[@]}" \
  2>&1 | tee "$run_dir/server.log"
