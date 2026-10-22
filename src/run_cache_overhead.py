"""Compare implementation overhead with identical LFU and partial-prefix policy."""

import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT


def decisions(record):
    events = next(Path(record["run_dir"]).glob("events-*.jsonl"))
    result = []
    for line in events.open():
        e = json.loads(line)
        kind = e["event"]
        if kind == "ALLOCATE_PHYSICAL_END":
            result.append((kind, e["allocated"]))
        elif kind == "PREFIX_LOOKUP_END":
            result.append((kind, e["hit_tokens"], e["hit_blocks"]))
        elif kind == "EVICT_CACHED_BLOCK":
            result.append((kind, e["before"], e["old_hash"]))
        elif kind == "SCHEDULE_END" and e["assigned_total"]:
            result.append((kind, e["assigned_total"], e["kv_block_copies"]))
    return result


def load(record):
    return json.loads(Path(record["result"]).read_text())


def main():
    os.environ["STUDY_MODEL"] = "Qwen/Qwen3-0.6B"
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        os.environ["STUDY_MODEL"],
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    folder = ROOT / "results/cache-overhead" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    summary = dict(
        experiment="10-cache-overhead", diagnostics=[], runs=[], complete=False
    )
    path = folder / "summary.json"

    def save():
        path.write_text(json.dumps(summary, indent=2) + "\n")

    def run(profile, impl, mode, label):
        os.environ["STUDY_AGING_PROFILE"] = profile
        os.environ["STUDY_AGING_RESULTS_ROOT"] = str(
            ROOT / "results/cache-overhead" / impl
        )
        record = run_one(
            folder,
            impl,
            mode,
            f"{profile}-{label}",
            launcher="src/serve_cache_overhead.sh",
            launcher_args=[profile],
            client_script="workloads/compare_cache_overhead.py",
            client_eviction_policy="lfu",
        )
        record.update(profile=profile, round=label)
        cpu_path = Path(record["run_dir"]) / "cpu-profile.json"
        record["cpu_profile"] = json.loads(cpu_path.read_text())
        record["cpu_profile_path"] = str(cpu_path)
        assert (
            record["cpu_profile"]["timings"]["request_finish"]["calls"]
            == len(load(record)["requests"]) + 1
        )
        return record

    try:
        for profile in ("shift", "stable"):
            for impl in ("reference", "incremental"):
                summary["diagnostics"].append(run(profile, impl, "trace", "diagnostic"))
                save()
            left, right = summary["diagnostics"][-2:]
            assert decisions(left) == decisions(right), "Diagnostic decisions changed"
            for index in range(3):
                order = (
                    ("reference", "incremental")
                    if index % 2 == 0
                    else ("incremental", "reference")
                )
                for impl in order:
                    summary["runs"].append(run(profile, impl, "bench", index + 1))
                    save()
        summary["aggregate"] = {}
        for profile in ("shift", "stable"):
            rows = [r for r in summary["runs"] if r["profile"] == profile]
            summary["aggregate"][profile] = {}
            for impl in ("reference", "incremental"):
                records = [r for r in rows if r["variant"] == impl]
                data = [load(r) for r in records]
                cpu = [r["cpu_profile"] for r in records]
                summary["aggregate"][profile][impl] = dict(
                    duration_ms=[d["batch_summary"]["duration_ms"] for d in data],
                    batch_median_ms=statistics.median(
                        d["batch_summary"]["duration_ms"] for d in data
                    ),
                    median_request_latency_ms=statistics.median(
                        d["batch_summary"]["total_ms"]["median"] for d in data
                    ),
                    median_throughput=statistics.median(
                        d["batch_summary"]["output_tokens_per_second"] for d in data
                    ),
                    cpu_median_ms={
                        key: statistics.median(
                            c["timings"][key]["ns"] / 1e6 for c in cpu
                        )
                        for key in cpu[0]["timings"]
                    },
                    visited_blocks=[c["partial_blocks_visited"] for c in cpu],
                    order_rebuilds=[c["order_rebuilds"] for c in cpu],
                )
            for round_id in (1, 2, 3):
                pair = [
                    load(
                        next(
                            r
                            for r in rows
                            if r["round"] == round_id and r["variant"] == impl
                        )
                    )
                    for impl in ("reference", "incremental")
                ]
                assert pair[0]["manifest"] == pair[1]["manifest"]
                assert [
                    (r["request_id"], r["cached_prompt_tokens"], r["answer"])
                    for r in pair[0]["requests"]
                ] == [
                    (r["request_id"], r["cached_prompt_tokens"], r["answer"])
                    for r in pair[1]["requests"]
                ]
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        save()
        print(f"SUMMARY {path}", flush=True)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
