#!/usr/bin/env bash
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
work="$HOME/vllm-research-20261024"
state="$work/source-build-state.txt"
trap 'code=$?; printf "exit_code=%s\n" "$code" >> "$state"' EXIT
printf 'job=%s\nhost=%s\nphase=build-dependencies\n' "$SLURM_JOB_ID" "$(hostname)" > "$state"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$work/tools/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR="/tmp/vllm-source-cache-$SLURM_JOB_ID"
export TMPDIR="/tmp/vllm-source-build-$SLURM_JOB_ID"
export VLLM_PLUGINS=''
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
venv_python="$work/.venv-cuda/bin/python"
uv pip install --python "$venv_python" --torch-backend=cu130 -r "$work/source-git/requirements/build/cuda.txt"
uv pip freeze --python "$venv_python" > "$work/dependencies-with-build-tools.txt"
"$venv_python" "$work/preflight-kit/native_smoke.py" "$work/.venv-cuda" > "$work/native-official.json"
printf 'phase=verified-release-wheel\n' >> "$state"
wheel=$("$venv_python" "$work/preflight-kit/download_release_wheel.py" "$work/wheels")
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$wheel"
for arm in stock candidate; do
    source_dir="$work/source-git"
    if [[ $arm == candidate ]]; then source_dir="$work/candidate-src"; fi
    printf 'phase=editable-build-%s\n' "$arm" >> "$state"
    uv pip install --python "$venv_python" --no-deps --no-build-isolation --editable "$source_dir" 2>&1 | tee "$work/source-install-$arm.txt"
    printf 'phase=native-smoke-%s\n' "$arm" >> "$state"
    "$venv_python" "$work/preflight-kit/native_smoke.py" "$source_dir" > "$work/native-$arm.json"
    git -C "$source_dir" status --porcelain --untracked-files=no > "$work/tracked-status-$arm.txt"
    test ! -s "$work/tracked-status-$arm.txt"
done
uv pip check --python "$venv_python" > "$work/source-dependency-check.txt"
uv pip freeze --python "$venv_python" > "$work/dependencies-final.txt"
printf 'phase=complete\n' >> "$state"
