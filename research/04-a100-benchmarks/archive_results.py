"""Seal benchmark records and code without model weights or native binaries."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
root = work / "benchmark-results-20261026"
shutil.copyfile(
    work / "benchmark-models-20261026/staging-receipt.json",
    root / "model-staging-receipt.json",
)
for path in work.glob("benchmark-pilot-*.txt"):
    shutil.copyfile(path, root / path.name)
for path in work.glob("benchmark-matrix-*.txt"):
    shutil.copyfile(path, root / path.name)
manifest = {}
for directory in (root, work / "benchmark-kit-20261026"):
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest[str(path.relative_to(work))] = {
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
(work / "benchmark-file-inventory-20261026.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
archive = work / "benchmark-evidence-20261026.tar.gz"
with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
    bundle.add(root, arcname=root.name)
    bundle.add(work / "benchmark-kit-20261026", arcname="benchmark-kit-20261026")
    bundle.add(
        work / "benchmark-file-inventory-20261026.json",
        arcname="benchmark-file-inventory-20261026.json",
    )
with archive.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
(work / "benchmark-archive-sha256-20261026.txt").write_text(
    f"{digest}  {archive.name}\n"
)
print("SEALED_BENCHMARK", archive, archive.stat().st_size, digest, flush=True)
