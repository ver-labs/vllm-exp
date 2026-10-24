#!/usr/bin/env bash
# Run only inside a Slurm GPU allocation. No installation or model download.
set -u
set -o pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Run this collector inside a Slurm GPU allocation, not the login node.' >&2
    exit 2
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo 'Usage: bash collect_environment.sh NEW_OUTPUT_DIR [VENV_PYTHON]' >&2
    exit 2
fi
output=$1
if [[ -e $output ]]; then
    echo 'Output directory already exists; use a fresh directory.' >&2
    exit 2
fi
mkdir -p "$output" || exit 2
output=$(cd "$output" && pwd)
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

capture() {
    local label=$1
    shift
    "$@" > "$output/$label.txt" 2>&1
    local code=$?
    printf '%s\t%s\n' "$label" "$code" >> "$output/command-status.tsv"
}

printf 'command\texit_code\n' > "$output/command-status.tsv"
{
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    hostname
    uname -a
    printf 'SLURM_JOB_ID=%s\n' "$SLURM_JOB_ID"
    printf 'SLURM_JOB_PARTITION=%s\n' "${SLURM_JOB_PARTITION:-}"
    printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK:-}"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
    printf 'CUDA_HOME=%s\n' "${CUDA_HOME:-}"
    printf 'LOADEDMODULES=%s\n' "${LOADEDMODULES:-}"
} > "$output/allocation.txt"
capture os-release cat /etc/os-release
capture gpu nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,compute_cap --format=csv
capture gpu-topology nvidia-smi -L
capture nvcc nvcc --version
capture cxx c++ --version
capture glibc getconf GNU_LIBC_VERSION
capture cmake cmake --version
capture ninja ninja --version
capture rust rustc --version
capture uv uv --version
capture memory free -h
capture disk df -h . /tmp

if command -v nvcc >/dev/null 2>&1; then
    capture cuda-compile nvcc -std=c++20 -gencode arch=compute_80,code=sm_80 \
        "$script_dir/cuda_smoke.cu" -o "$output/cuda-smoke"
    if [[ -x $output/cuda-smoke ]]; then
        capture cuda-run "$output/cuda-smoke"
    fi
fi
if [[ $# -eq 2 ]]; then
    if [[ -x $2 && -f $(dirname "$2")/../pyvenv.cfg ]]; then
        capture python "$2" "$script_dir/torch_smoke.py"
    else
        echo 'Explicit Python must belong to a virtual environment.' > "$output/python.txt"
        printf 'python\t2\n' >> "$output/command-status.tsv"
    fi
else
    echo 'Not run: provide the isolated CUDA virtualenv Python as argument 2.' > "$output/python.txt"
    printf 'python\tskipped\n' >> "$output/command-status.tsv"
fi
printf 'Survey saved to %s. Inspect every exit code; completion is not a compatibility PASS.\n' "$output"
