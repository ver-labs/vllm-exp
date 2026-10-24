"""Fetch and verify only the exact release's Linux x86_64 CUDA wheel."""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

with urllib.request.urlopen("https://pypi.org/pypi/vllm/0.31.0/json", timeout=60) as response:
    metadata = json.load(response)
candidates = [item for item in metadata["urls"] if item["filename"].endswith("x86_64.whl") and "manylinux" in item["filename"]]
if len(candidates) != 1:
    raise RuntimeError(f"Expected one official Linux x86_64 wheel, found {len(candidates)}")
item = candidates[0]
folder = Path(sys.argv[1])
folder.mkdir(parents=True, exist_ok=True)
path = folder / item["filename"]
if not path.exists():
    urllib.request.urlretrieve(item["url"], path)
with path.open("rb") as stream:
    actual = hashlib.file_digest(stream, "sha256").hexdigest()
if actual != item["digests"]["sha256"]:
    raise RuntimeError("Official wheel SHA256 mismatch")
(folder / "wheel-provenance.json").write_text(json.dumps({"filename": item["filename"], "url": item["url"], "sha256": actual, "bytes": path.stat().st_size}, indent=2) + "\n")
print(path)
