"""Freeze a deterministic, disjoint WildChat evaluation and warmup sample."""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent


def main():
    pins = json.loads((ROOT / "pins.json").read_text())
    revision = pins["allenai/WildChat"]
    path = hf_hub_download(
        "allenai/WildChat",
        "data/train-00000-of-00006.parquet",
        repo_type="dataset",
        revision=revision,
    )
    rows, seen = [], set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            turns = row.get("conversation") or []
            if not turns:
                continue
            first = turns[0]
            text = first.get("content", "").strip()
            if (
                first.get("role") != "user"
                or not text
                or first.get("language") != "English"
                or first.get("toxic")
                or first.get("redacted")
                or not 80 <= len(text) <= 20000
                or text in seen
            ):
                continue
            seen.add(text)
            # One independently selected conversation, one first user turn.
            rows.append({"id": str(row["conversation_id"]), "text": text})
            if len(rows) == 512:
                break
        if len(rows) == 512:
            break
    assert len(rows) == 512
    assert len({r["id"] for r in rows}) == 512
    assert len({r["text"] for r in rows}) == 512
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    result = {
        "dataset": "allenai/WildChat",
        "revision": revision,
        "source_shard": "data/train-00000-of-00006.parquet",
        "shard_sha256": digest,
        "selection": "first 512 eligible rows in pinned shard order",
        "warmup": rows[:32],
        "evaluation": rows[32:],
        "attribution": "Contains information from allenai/WildChat, available "
        "under the ODC Attribution License: https://opendatacommons.org/licenses/by/1-0/",
    }
    (ROOT / "wildchat-sample.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Frozen", len(rows), "disjoint conversation IDs")


if __name__ == "__main__":
    main()
