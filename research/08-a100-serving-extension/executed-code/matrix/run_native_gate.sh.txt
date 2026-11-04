#!/usr/bin/env bash
set -euo pipefail
work="$HOME/vllm-research-20261024"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$work/attention-src"
export VLLM_PLUGINS=''
export VLLM_BATCH_INVARIANT=1
export STUDY_FA2_SINGLE_QUERY_CAUSAL=1
export ATTENTION_GATE_OUTPUT="$work/serving-extension-results-20261103/smol-native-gate.json"
"$work/tools/uv-x86_64-unknown-linux-gnu/uv" pip freeze --python "$work/.venv-cuda/bin/python" > "$work/serving-extension-results-20261103/dependencies.txt"
"$work/.venv-cuda/bin/python" "$work/serving-extension-kit-20261103/analyze_gates.py"
"$work/.venv-cuda/bin/python" "$work/serving-extension-kit-20261103/smol_attention_gate.py"
