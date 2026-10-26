#!/usr/bin/env bash
set -euo pipefail
work="$HOME/vllm-research-20261024"
"$work/.venv-cuda/bin/python" "$work/profiling-analysis-kit-20261026/analyze_profiles.py" "$work/profiling-results-20261026"
"$work/.venv-cuda/bin/python" "$work/profiling-analysis-kit-20261026/archive_analysis.py"
