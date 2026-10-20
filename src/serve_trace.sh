#!/usr/bin/env bash
set -euo pipefail

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
mkdir -p results/source-checkout
trace_dir="$(mktemp -d "$experiment_root/results/source-checkout/trace-XXXXXX")"

echo "Trace directory: $trace_dir"
echo "Stop any existing server on port 8001 before launching this script."
echo "This launches the editable checkout; it does not submit a workload."

VLLM_STUDY_TRACE=1 \
VLLM_STUDY_TRACE_DIR="$trace_dir" \
VLLM_STUDY_TRACE_STDOUT=1 \
"$experiment_root/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 2048 \
  --max-num-seqs 3 \
  --enable-logging-iteration-details \
  "$@" \
  2>&1 | tee "$trace_dir/server.log"
