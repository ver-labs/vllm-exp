"""Record the fallback installation without importing GPU libraries."""

import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
site = Path(sys.prefix) / "lib/python3.12/site-packages"
native = {}
for package in ("vllm", "vllm_metal"):
    for path in sorted((site / package).rglob("*")):
        if path.is_file() and (
            path.suffix in {".so", ".dylib", ".metallib"}
            or path.name == "vllm-rs"
        ):
            native[str(path.relative_to(site))] = hashlib.sha256(path.read_bytes()).hexdigest()
record = {
    "environment": sys.prefix,
    "packages": {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in metadata.distributions()},
    "native_sha256": native,
}
destination = root / "configs/metal-baseline.json"
if destination.exists():
    raise SystemExit(f"Refusing to replace baseline: {destination}")
destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
constraints = root / "configs/metal-constraints.txt"
constraints.write_text("".join(f"{name}=={version}\n"
                               for name, version in sorted(record["packages"].items())))
print(f"Saved {len(record['packages'])} package versions and {len(native)} native hashes")
