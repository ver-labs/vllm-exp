"""Stage the pinned additional family without altering installed packages."""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

root = Path(__file__).resolve().parent
pins = json.loads((root / "pins.json").read_text())
model = "HuggingFaceTB/SmolLM3-3B"
work = Path.home() / "vllm-research-20261024"
dest = work / "serving-extension-assets-20261103" / "SmolLM3-3B"
os.environ.pop("HF_HUB_OFFLINE", None)
snapshot_download(
    model,
    revision=pins[model],
    local_dir=dest,
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "LICENSE*"],
    max_workers=2,
)
assets = {}
for path in sorted(dest.rglob("*")):
    if path.is_file() and ".cache" not in path.parts:
        with path.open("rb") as handle:
            assets[str(path.relative_to(dest))] = {
                "size": path.stat().st_size,
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
            }
(work / "serving-extension-results-20261103/model-assets.json").write_text(
    json.dumps({"model": model, "revision": pins[model], "assets": assets}, indent=2)
    + "\n"
)
print(dest)
