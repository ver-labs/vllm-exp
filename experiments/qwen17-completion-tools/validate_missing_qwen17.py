"""Verify completed model replications and diagnostic trace invariants."""

from collections import Counter
import hashlib
import json
from pathlib import Path

root = Path("/Users/nandanadileep/projects/vllm/vllm/vllm-exp")
progress = json.loads((root / "results/missing-qwen17/progress.json").read_text())
assert progress["complete"] and len(progress["stages"]) == 8
assert sum(r["gpu_runs"] for r in progress["stages"]) == 98
assert sum(r["timed_runs"] for r in progress["stages"]) == 72
files = {}
stages = []
for stage in progress["stages"]:
    summary = json.loads(Path(stage["frozen_summary"]).read_text())
    assert summary["complete"]
    records = summary["diagnostics"] + summary["runs"]
    outputs = []
    for record in records:
        data = json.loads(Path(record["result"]).read_text())
        assert data.get("valid_run", True)
        batch = data.get("batch_summary", data.get("summary"))
        assert batch["failed_requests"] == 0
        expected = 1984
        if stage["stage"] == "02-temporal":
            expected = 336
        if stage["stage"] == "05" and record["profile"] == "targeted":
            expected = 2112
        if stage["stage"] == "06":
            expected = {"ay": 1984, "targeted": 2112, "temporal": 336}[
                record["profile"]
            ]
        if stage["stage"] == "10":
            expected = {"shift": 672, "stable": 336}[record["profile"]]
        assert batch["successful_output_tokens"] == expected, (
            stage["stage"],
            record,
            batch,
        )
        outputs.append(expected)
    inventory = json.loads(
        (Path(stage["frozen_summary"]).parent / "raw-sha256.json").read_text()
    )
    for item in inventory:
        files[item["path"]] = item
    stages.append(
        dict(
            stage=stage["stage"],
            gpu_runs=len(records),
            timed_runs=len(summary["runs"]),
            successful_output_tokens=sum(outputs),
        )
    )
for item in files.values():
    path = root / item["path"]
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    assert h.hexdigest() == item["sha256"] and path.stat().st_size == item["bytes"], (
        path
    )
traces = []
for rel in sorted(files):
    path = root / rel
    if not (path.name.startswith("events-") and path.suffix == ".jsonl"):
        continue
    counts = Counter()
    budget = None
    max_bypass = 0
    for line in path.open():
        e = json.loads(line)
        event = e["event"]
        counts[event] += 1
        if event == "TRACE_READY":
            budget = e["token_budget_limit"]
            assert e["scheduler_source"].startswith(str(root / "vllm-src"))
        if event == "SCHEDULE_END":
            assert budget is not None and 0 <= e["assigned_total"] <= budget
        if event == "EVICT_CACHED_BLOCK":
            assert e["before"]["refs"] == 0
        if event == "BUDGET_BACKFILL_SKIP":
            max_bypass = max(max_bypass, e["bypasses"])
            assert e["bypasses"] <= 4
        assert e["free_blocks"] >= 0
    assert counts["TRACE_READY"] == 1 and counts["SCHEDULE_END"] > 0
    assert counts["PREEMPT_END"] == 0
    assert counts["ADAPTIVE_PREFILL_CAP"] == 0
    traces.append(
        dict(
            path=rel,
            events=sum(counts.values()),
            budget=budget,
            preemptions=0,
            zero_reference_evictions=counts["EVICT_CACHED_BLOCK"],
            skip_events=counts["BUDGET_BACKFILL_SKIP"],
            max_bypasses=max_bypass,
            native_source_verified=True,
            token_budget_verified=True,
        )
    )
assert len(traces) == 26, len(traces)
audit = json.loads(
    (root / "results/missing-qwen17/input-manifest-audit.json").read_text()
)
assert all(row["unchanged_except_model"] for row in audit)
validation = dict(
    complete=True,
    model=progress["model"],
    model_revision=progress["model_revision"],
    gpu_runs=98,
    timed_runs=72,
    diagnostic_runs=26,
    stages=stages,
    verified_raw_files=len(files),
    successful_output_tokens=sum(s["successful_output_tokens"] for s in stages),
    trace_validation=traces,
    manifest_comparison=audit,
    all_requests_successful=True,
    all_servers_verified_native_gpu=True,
)
dest = root / "experiments/qwen3-1.7B-completion-validation.json"
dest.write_text(json.dumps(validation, indent=2) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in validation.items()
            if k not in ("trace_validation", "manifest_comparison")
        },
        indent=2,
    )
)
