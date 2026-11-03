#!/usr/bin/env bash
set -euo pipefail
work="$HOME/vllm-research-20261024"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$work/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export VLLM_PLUGINS=''
test -f "$work/serving-extension-results-20261103/smol-native-gate.json"
"$work/.venv-cuda/bin/python" "$work/serving-extension-kit-20261103/serve_driver.py"
"$work/.venv-cuda/bin/python" "$work/serving-extension-kit-20261103/summarize.py" "$work/serving-extension-results-20261103"
"$work/.venv-cuda/bin/python" "$work/serving-extension-kit-20261103/archive_results.py"
