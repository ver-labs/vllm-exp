"""Assemble complete local snapshots from previously verified model-cache files."""

import hashlib
import json
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
manifest_path = work / "benchmark-kit-20261026/frozen-model-files.json"
manifest = json.loads(manifest_path.read_text())
records = []
for item in manifest:
    source = work / item["path"]
    assert source.stat().st_size == item["bytes"]
    with source.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    relative = Path(item["path"].split("/snapshots/" + item["revision"] + "/")[1])
    assert ".." not in relative.parts
    destination = (
        work / "benchmark-models-20261026" / item["model"].split("/")[-1] / relative
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
    else:
        destination.symlink_to(source.resolve())
    records.append({**item, "staged_path": str(destination.relative_to(work))})
(work / "benchmark-models-20261026/staging-receipt.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
print("Verified and staged", len(records), "pinned model files", flush=True)
