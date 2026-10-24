Executed results: see RESULTS.txt and validation-a100.json.
The preparation notes below describe the original preflight plan.

A100 environment and CUDA compatibility preflight
================================================
Prepared 24 October 2026. This kit installs nothing and downloads no models.

Confirmed from supplied terminal output
--------------------------------------
The rg-nextgen-hpc partition granted one A100 through Slurm job 174876.
The allocated device reported NVIDIA A100-PCIE-40GB, 40960 MiB and driver
610.57.04. This is historical observed output, not a fresh execution receipt.
The login node itself did not have nvidia-smi; do not install drivers there.

Next live checks
----------------
Run collect_environment.sh only inside a new Slurm GPU allocation. It records
OS, compiler/toolkit, build tools, GPU allocation and available storage. If nvcc
is available it compiles a C++20 SM80 kernel and checks launch/synchronization/
device-to-host copy. It does not build vLLM or prove candidate correctness.
Every command's exit code is recorded; collector completion is not a PASS.

Example after copying this directory to the cluster:

  srun --partition=rg-nextgen-hpc --gres=gpu:a100:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
    --time=00:05:00 --immediate=10 \
    bash /path/to/kit/collect_environment.sh /path/to/new-receipts

If an isolated CUDA environment exists, append its .venv/bin/python as the
second collector argument. The Python smoke checks torch package provenance,
CUDA initialization and small FP16/BF16 matrix results. It does not load a model
or validate vLLM native extensions. If nvcc is absent, inspect available CUDA
modules or an approved development container before installing anything.

Pinned source requirements
--------------------------
The exact stock release and candidate use official v0.31.0 base
 db9527a46873454610df6dbedf79a36d6bf1a7f6.
The new candidate is 08d2c99e8fe5117605d0ca8eaf6b6fd07672d822.
The pinned source declares Python >=3.10,<3.15 and torch==2.13.0, including
in build requirements. Its Dockerfile default is CUDA 13.0.3 / Python 3.12;
that default alone is not an instruction to replace the cluster toolkit.
CMake explicitly includes SM80, and uses C++20 for CUDA. The actual toolkit,
compiler, PyTorch CUDA ABI and dependencies must be checked together.
Do not copy the Mac native binaries or Mac constraints into this environment.

Remaining acceptance gates
--------------------------
1. Fresh environment receipt from an allocated A100 and usable scratch space.
2. Resolve/build identical isolated stock and candidate CUDA environments.
3. Import real Linux CUDA native extensions and exercise their operators.
4. Run metadata tests and actual KV copy/logit/cancellation/reset/preemption
   correctness checks, before any serving performance claims.

This kit is preparation, not a claim that any of these gates has passed.
