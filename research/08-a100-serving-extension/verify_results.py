"""Independently verify recorded counts, timing identities and token parity."""

import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
files = list((args.root / "matrix").glob("*/*/repeat-*/*/result.json"))
assert len(files) == 36
requests = tokens = cells = 0
outputs, gpus = {}, set()
for path in files:
    record = json.loads(path.read_text())
    assert record["status"] == "passed"
    gpus.add(record["gpu"])
    model, mode, repeat, arm = path.relative_to(args.root / "matrix").parts[:4]
    bank_path = args.root / "matrix" / f"{model}-workload.json"
    bank = json.loads(bank_path.read_text())
    assert (
        bank_path.read_bytes()
        == (args.root / "gate" / f"{model}-workload.json").read_bytes()
    )
    expected_ids = {r["id"] for r in bank["evaluation"]}
    assert len(record["cases"]) == 4
    for case, data in record["cases"].items():
        limit = int(case.split("-O")[1])
        rows = data["requests"]
        assert len(rows) == 96 and {r["id"] for r in rows} == expected_ids
        outputs[model, repeat, arm, case] = {r["id"]: r["token_ids"] for r in rows}
        for row in rows:
            count = len(row["token_ids"])
            chunks = row["chunks"]
            arrivals = [c["ns"] for c in chunks]
            assert count == limit and sum(c["tokens"] for c in chunks) == count
            assert arrivals == sorted(arrivals)
            assert row["start_ns"] <= arrivals[0] <= arrivals[-1] <= row["end_ns"]
            expected = {
                "ttft_ms": (arrivals[0] - row["start_ns"]) / 1e6,
                "latency_ms": (row["end_ns"] - row["start_ns"]) / 1e6,
                "tpot_ms": (arrivals[-1] - arrivals[0]) / 1e6 / (count - 1),
            }
            for name, value in expected.items():
                assert math.isclose(row[name], value, rel_tol=1e-12, abs_tol=1e-9)
            requests += 1
            tokens += count
        assert math.isclose(
            data["requests_per_s"] * data["elapsed_s"], 96, rel_tol=1e-12
        )
        assert math.isclose(
            data["output_tokens_per_s"] * data["elapsed_s"], 96 * limit, rel_tol=1e-12
        )
        cells += 1
for (model, repeat, arm, case), ids in outputs.items():
    assert ids == outputs[model, repeat, "stock", case]
    assert ids == outputs[model, "repeat-1", "stock", case]
assert len(gpus) == 1
assert (requests, tokens, cells) == (13824, 1105920, 144)
report = {
    "status": "passed",
    "fresh_servers": 36,
    "paired_repeats": 6,
    "timed_cells": cells,
    "requests": requests,
    "actual_output_tokens": tokens,
    "all_output_ids_match_across_arms_and_repeats": True,
    "timing_and_count_identities_verified": True,
    "same_gpu_query_across_servers": True,
}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
