"""Inspect vLLM's actual current FA dependency and preserve duplicate searches."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import quote

from fetch_sources import fetch

root = Path(__file__).resolve().parent
dest = root / "upstream"
cmake = (
    dest / "vllm-project--vllm--cmake--external_projects--vllm_flash_attn.cmake"
).read_text()
revision = re.search(r"GIT_TAG ([a-f0-9]{40})", cmake).group(1)
url = f"https://raw.githubusercontent.com/vllm-project/flash-attention/{revision}/csrc/flash_attn/flash_api.cpp"
raw = fetch(url)
name = "vllm-pinned-dependency-flash_api.cpp"
(dest / name).write_bytes(raw)
report = {
    "dependency_revision": revision,
    "url": url,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "searches": [],
}
queries = [
    'repo:vllm-project/vllm "FA2" "clone"',
    'repo:vllm-project/vllm "attention" "reshape"',
    'repo:vllm-project/flash-attention "seqlenq_ngroups_swapped"',
    'repo:Dao-AILab/flash-attention "seqlenq_ngroups_swapped"',
    'repo:Dao-AILab/flash-attention "GQA" "copy"',
]
for query in queries:
    try:
        result = json.loads(
            fetch(
                "https://api.github.com/search/issues?q="
                + quote(query)
                + "&per_page=100"
            )
        )
        report["searches"].append(
            {
                "query": query,
                "total_count": result["total_count"],
                "items": [
                    {
                        k: item[k]
                        for k in ("html_url", "title", "state", "created_at", "body")
                    }
                    for item in result["items"]
                ],
            }
        )
    except OSError as error:
        report["searches"].append({"query": query, "error": str(error)})
(dest / "duplicate-search.json").write_text(json.dumps(report, indent=2) + "\n")
print("Pinned dependency", revision)
for query in report["searches"]:
    print(query["query"], query.get("total_count", query.get("error")))
