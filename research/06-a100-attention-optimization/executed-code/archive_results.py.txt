"""Seal complete evidence and a compact version without raw tensor dumps."""

import hashlib
import json
import tarfile
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
files = []
for directory in (work / "attention-kit-20261026", work / "attention-results-20261026"):
    files.extend(
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
    )
files.extend(work.glob("attention-gate-*.log"))
files.extend(work.glob("attention-benchmark-*.log"))
manifest = {}
for path in files:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest[str(path.relative_to(work))] = {
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
inventory = work / "attention-inventory-20261026.json"
inventory.write_text(json.dumps(manifest, indent=2) + "\n")
for name, compact in (
    ("attention-evidence-20261026.tar.gz", False),
    ("attention-compact-20261026.tar.gz", True),
):
    with tarfile.open(work / name, "w:gz", compresslevel=1) as bundle:
        for path in files:
            if compact and path.suffix in (".pt", ".log"):
                continue
            bundle.add(path, arcname=str(path.relative_to(work)))
        bundle.add(inventory, arcname=inventory.name)
    with (work / name).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    (work / (name + ".sha256")).write_text(f"{digest}  {name}\n")
    print("SEALED_ATTENTION", name, (work / name).stat().st_size, digest, flush=True)
