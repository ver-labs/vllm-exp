"""Seal raw traces separately from the compact analyzed evidence."""

import hashlib
import json
import tarfile
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
root = work / "profiling-results-20261026"
files = []
for folder in (
    root,
    work / "profiling-kit-20261026",
    work / "profiling-analysis-kit-20261026",
):
    files.extend(
        p
        for p in sorted(folder.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    )
files.append(work / "profiling-job-175116.log")
manifest = {}
for path in files:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest[str(path.relative_to(work))] = {
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
inventory = work / "profiling-inventory-20261026.json"
inventory.write_text(json.dumps(manifest, indent=2) + "\n")
for name, compact in (
    ("profiling-analyzed-20261026.tar.gz", False),
    ("profiling-compact-20261026.tar.gz", True),
):
    archive = work / name
    with tarfile.open(archive, "w:gz") as bundle:
        for path in files:
            if compact and (
                path.name.endswith(".trace.json.gz") or path.suffix == ".log"
            ):
                continue
            bundle.add(path, arcname=str(path.relative_to(work)))
        bundle.add(inventory, arcname=inventory.name)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (work / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    print("SEALED", name, archive.stat().st_size, digest, flush=True)
