#!/usr/bin/env bash
set -euo pipefail
work="$HOME/vllm-research-20261024"
"$work/.venv-cuda/bin/python" "$work/correctness-kit/hash_models.py"
"$work/.venv-cuda/bin/python" - <<'PY'
from pathlib import Path
import hashlib, json
work = Path.home() / 'vllm-research-20261024'
files = {}
for path in sorted((work / 'correctness-results').rglob('*')):
    if path.is_file():
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files[str(path.relative_to(work))] = {'sha256': digest, 'bytes': path.stat().st_size}
(work / 'correctness-file-inventory.json').write_text(json.dumps(files, indent=2) + '\n')
PY
cd "$work"
tar -cf - correctness-results correctness-kit correctness-model-file-sha256.json correctness-source-state.json correctness-file-inventory.json correctness-pilot-*.txt correctness-matrix-*.txt | gzip -1 > "$work/correctness-evidence-20261024.tar.gz"
sha256sum "$work/correctness-evidence-20261024.tar.gz" > "$work/correctness-archive-sha256.txt"
cat "$work/correctness-archive-sha256.txt"
