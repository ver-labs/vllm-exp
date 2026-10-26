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
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
"$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/stage_models.py"
"$work/tools/uv-x86_64-unknown-linux-gnu/uv" pip freeze --python "$work/.venv-cuda/bin/python" > "$work/benchmark-results-20261026/dependencies.txt"
"$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/run_matrix.py"
"$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/archive_results.py"
