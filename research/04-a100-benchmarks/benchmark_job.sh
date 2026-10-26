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
export STUDY_PARTIAL_PREFIX=0
export STUDY_EVICTION_POLICY=vllm
export STUDY_LFU_HALF_LIFE_BLOCKS=4
export STUDY_PARTIAL_MIN_TOKENS=0
export STUDY_CACHE_IMPL=incremental
export PYTHONPATH="$work/source-git"
"$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/stage_models.py"
exec timeout 1200 "$work/.venv-cuda/bin/python" "$work/benchmark-kit-20261026/run_benchmark.py" \
    --model Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
    --source "$work/source-git" --output "$work/benchmark-results-20261026/pilot-stock-r2" \
    --arm stock --repeat 0
