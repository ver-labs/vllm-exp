#!/usr/bin/env bash
set -euo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || exit 2
work="$HOME/vllm-research-20261024"
state="$work/environment-state.txt"
trap 'code=$?; printf "exit_code=%s\n" "$code" >> "$state"' EXIT
printf 'job=%s\nhost=%s\nphase=toolkit-smoke\n' "$SLURM_JOB_ID" "$(hostname)" > "$state"
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CUDA_HOME/bin:$work/tools/uv-x86_64-unknown-linux-gnu:$PATH"
export UV_CACHE_DIR="/tmp/vllm-uv-cache-$SLURM_JOB_ID"
export UV_PYTHON_INSTALL_DIR="$work/tools/python"
export TMPDIR="/tmp/vllm-build-$SLURM_JOB_ID"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
bash "$work/preflight-kit/collect_environment.sh" "$work/receipts-toolkit"
printf 'phase=source-preparation\n' >> "$state"
git -C "$work/source-git" fetch "$work/candidate-source.bundle" refs/heads/research/cache-baseline-20261024:refs/heads/frozen-candidate
test "$(git -C "$work/source-git" rev-parse HEAD)" = db9527a46873454610df6dbedf79a36d6bf1a7f6
git -C "$work/source-git" worktree add --detach "$work/candidate-src" 08d2c99e8fe5117605d0ca8eaf6b6fd07672d822
printf 'phase=python-environment\n' >> "$state"
uv venv --python 3.12 "$work/.venv-cuda"
printf 'phase=official-release-install\n' >> "$state"
uv pip install --python "$work/.venv-cuda/bin/python" --torch-backend=cu130 'vllm==0.31.0' 2>&1 | tee "$work/dependency-install.txt"
uv pip freeze --python "$work/.venv-cuda/bin/python" > "$work/dependencies-resolved.txt"
uv pip check --python "$work/.venv-cuda/bin/python" > "$work/dependency-check.txt"
printf 'phase=torch-smoke\n' >> "$state"
"$work/.venv-cuda/bin/python" "$work/preflight-kit/torch_smoke.py" > "$work/torch-smoke.json"
printf 'phase=complete\n' >> "$state"
