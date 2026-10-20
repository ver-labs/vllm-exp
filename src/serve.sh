#!/usr/bin/env bash
set -euo pipefail

exec "$HOME/.venv-vllm-metal/bin/vllm" serve Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 2048 \
  --max-num-seqs 3