"""Seal sweep evidence and a compact archive without raw tensor dumps."""

import hashlib
import json
import tarfile
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
files = []
for directory in (
    work / "attention-sweep-kit-20261027",
    work / "attention-sweep-results-20261027",
):
    files.extend(
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    )
files.extend(work.glob("attention-sweep-*.log"))
manifest = {}
for path in files:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest[str(path.relative_to(work))] = {
        "bytes": path.stat().st_size,
        "sha256": digest,
    }
inventory = work / "attention-sweep-inventory-20261027.json"
inventory.write_text(json.dumps(manifest, indent=2) + "\n")
for name, compact in (
    ("attention-sweep-evidence-20261027.tar.gz", False),
    ("attention-sweep-compact-20261027.tar.gz", True),
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
    print("SEALED_SWEEP", name, (work / name).stat().st_size, digest, flush=True)
