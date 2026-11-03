"""Seal completed serving evidence while excluding the actively written job log."""

import hashlib
import json
import os
import tarfile
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
root = work / "serving-extension-results-20261103"
active_log = f"matrix-job-{os.environ['SLURM_JOB_ID']}.log"
files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != active_log)
manifest = {}
for path in files:
    with path.open("rb") as handle:
        manifest[str(path.relative_to(root))] = {
            "size": path.stat().st_size,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }
index = root / "archive-manifest.json"
index.write_text(json.dumps(manifest, indent=2) + "\n")
files.append(index)
archives = {}
for kind in ("full", "compact"):
    target = work / f"serving-extension-{kind}-20261103.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for path in files:
            if kind == "compact" and path.name == "full-logprobs.json":
                continue
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    with target.open("rb") as handle:
        archives[kind] = {
            "file": str(target),
            "size": target.stat().st_size,
            "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
        }
(root / "archives.json").write_text(json.dumps(archives, indent=2) + "\n")
print(json.dumps(archives, indent=2), flush=True)
