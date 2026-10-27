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
"$work/.venv-cuda/bin/python" "$work/attention-proof-kit-20261026/copy_proof_driver.py"
tar -C "$work" -czf "$work/attention-copy-proof-20261026.tar.gz" attention-copy-proof-20261026 attention-proof-kit-20261026
sha256sum "$work/attention-copy-proof-20261026.tar.gz" > "$work/attention-copy-proof-20261026.sha256"
