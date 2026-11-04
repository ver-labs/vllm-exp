#!/usr/bin/env bash
set -euo pipefail
work="$HOME/vllm-research-20261024"
kit="$work/serving-extension-kit-20261103"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export HF_HOME="$work/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export VLLM_PLUGINS=''
"$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/stage_models.py"
"$work/.venv-cuda/bin/python" "$kit/stage_model.py"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$work/.venv-cuda/bin/python" "$kit/serve_driver.py" --gate
