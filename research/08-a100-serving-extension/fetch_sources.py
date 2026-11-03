"""Freeze upstream evidence and public model/dataset revisions."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def fetch(url):
    request = Request(url, headers={"User-Agent": "vllm-attention-case-study"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def api(url):
    return json.loads(fetch(url))


def main():
    dest = ROOT / "upstream"
    dest.mkdir(exist_ok=True)
    pins = {}
    for repo in (
        "vllm-project/vllm",
        "vllm-project/flash-attention",
        "Dao-AILab/flash-attention",
    ):
        data = api(f"https://api.github.com/repos/{repo}/commits/main")
        pins[repo] = data["sha"]
        (dest / (repo.replace("/", "--") + "-commit.json")).write_text(
            json.dumps(data, indent=2) + "\n"
        )
    paths = {
        "vllm-project/vllm": [
            "vllm/vllm_flash_attn/flash_attn_interface.py",
            "cmake/external_projects/vllm_flash_attn.cmake",
            "docs/features/batch_invariance.md",
        ],
        "vllm-project/flash-attention": ["csrc/flash_attn/flash_api.cpp"],
        "Dao-AILab/flash-attention": ["csrc/flash_attn/flash_api.cpp"],
    }
    records = []
    for repo, files in paths.items():
        for path in files:
            url = f"https://raw.githubusercontent.com/{repo}/{pins[repo]}/{path}"
            try:
                raw = fetch(url)
            except OSError as error:
                records.append({"url": url, "error": str(error)})
                continue
            name = repo.replace("/", "--") + "--" + path.replace("/", "--")
            (dest / name).write_bytes(raw)
            records.append(
                {"url": url, "file": name, "sha256": hashlib.sha256(raw).hexdigest()}
            )
    for kind, repo in (
        ("models", "HuggingFaceTB/SmolLM3-3B"),
        ("datasets", "allenai/WildChat"),
    ):
        info = api(f"https://huggingface.co/api/{kind}/{repo}")
        pins[repo] = info["sha"]
        (dest / (repo.replace("/", "--") + "-info.json")).write_text(
            json.dumps(info, indent=2) + "\n"
        )
    (ROOT / "pins.json").write_text(json.dumps(pins, indent=2) + "\n")
    (dest / "sources.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(pins, indent=2))


if __name__ == "__main__":
    main()
