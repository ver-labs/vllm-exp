"""Matched stock/LFU/queue-retention benchmarks with separate traced checks."""

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT

POLICIES = ("vllm", "lfu", "queue")
REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}


def load(record):
    return json.loads(Path(record["result"]).read_text())


def retention_trace(record):
    events = Path(record["run_dir"]).glob("events-*.jsonl")
    totals = dict(
        selections=0, prevented_upstream_choices=0, forced_demanded=0, selection_ns=0
    )
    for path in events:
        for line in path.open():
            event = json.loads(line)
            if event["event"] == "QUEUE_CACHE_RETENTION":
                totals["selections"] += 1
                totals["prevented_upstream_choices"] += len(
                    event["retained_instead_of_upstream"]
                )
                totals["forced_demanded"] += event["selected_demanded"]
                totals["selection_ns"] += event["selection_ns"]
    return totals


def aggregate(records, policy):
    data = [load(r) for r in records if r["variant"] == policy]
    result = dict(
        duration_ms=[d["candidate_summary"]["duration_ms"] for d in data],
        median_duration_ms=statistics.median(
            d["candidate_summary"]["duration_ms"] for d in data
        ),
        median_batch_ms=statistics.median(
            d["batch_summary"]["duration_ms"] for d in data
        ),
        median_throughput=statistics.median(
            d["candidate_summary"]["output_tokens_per_second"] for d in data
        ),
        median_first_text_ms=statistics.median(
            d["candidate_summary"]["first_text_ms"]["median"] for d in data
        ),
        median_p95_first_text_ms=statistics.median(
            d["candidate_summary"]["first_text_ms"]["p95_nearest_rank"] for d in data
        ),
        median_request_ms=statistics.median(
            d["candidate_summary"]["total_ms"]["median"] for d in data
        ),
        cached_tokens=[d["candidate_cached_prompt_tokens"] for d in data],
    )
    if groups := data[0]["manifest"].get("prompt_groups"):
        result["groups"] = {}
        for name, labels in groups.items():
            per_run = [
                [r for r in d["requests"] if r["request_id"] in labels] for d in data
            ]
            result["groups"][name] = dict(
                median_first_text_ms=statistics.median(
                    statistics.median(r["first_text_ms"] for r in rows)
                    for rows in per_run
                ),
                cached_tokens=[
                    sum(r["cached_prompt_tokens"] for r in rows) for rows in per_run
                ],
                first_request_hits=[
                    next(
                        r["cached_prompt_tokens"]
                        for r in rows
                        if r["request_id"] == labels[0]
                    )
                    for rows in per_run
                ],
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("targeted", "ay"), required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--model", choices=tuple(REVISIONS), default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("rounds must be nonnegative")
    os.environ["STUDY_MODEL"] = args.model
    revision = REVISIONS[args.model]
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        args.model, revision=revision, local_files_only=True
    )
    result_root = ROOT / "results/queue-retention"
    if args.model != "Qwen/Qwen3-0.6B":
        result_root /= args.model.replace("/", "--")
    os.environ["STUDY_QUEUE_RESULTS_ROOT"] = str(result_root)
    folder = result_root / args.profile / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    destination = folder / "summary.json"
    summary = dict(
        experiment="08-queue-retention",
        profile=args.profile,
        model=args.model,
        model_revision=revision,
        model_path=os.environ["STUDY_MODEL_PATH"],
        rounds=args.rounds,
        diagnostics=[],
        runs=[],
        complete=False,
    )

    def run(policy, mode, label):
        record = run_one(
            folder,
            policy,
            mode,
            label,
            launcher="src/serve_queue_retention.sh",
            launcher_args=[args.profile],
            client_script=(
                "workloads/compare_queue_retention.py"
                if args.profile == "targeted"
                else "workloads/compare_queue_retention_ay.py"
            ),
            client_eviction_policy=policy,
        )
        record["round"] = label
        if mode == "trace":
            record["queue_retention"] = retention_trace(record)
        return record

    try:
        for policy in POLICIES:
            summary["diagnostics"].append(run(policy, "trace", "diagnostic"))
            destination.write_text(json.dumps(summary, indent=2) + "\n")
        diagnostic = {r["variant"]: load(r) for r in summary["diagnostics"]}
        if args.profile == "targeted":
            returning = diagnostic["queue"]["manifest"]["prompt_groups"]["returning"][0]
            hits = {
                p: next(
                    r["cached_prompt_tokens"]
                    for r in d["requests"]
                    if r["request_id"] == returning
                )
                for p, d in diagnostic.items()
            }
            summary["diagnostic_first_return_hits"] = hits
            if hits["queue"] <= max(hits["vllm"], hits["lfu"]):
                raise RuntimeError(
                    "Targeted diagnostic did not demonstrate additional "
                    "queued-prefix retention; do not time this workload yet"
                )
        for index in range(args.rounds):
            order = POLICIES[index % 3 :] + POLICIES[: index % 3]
            for policy in order:
                summary["runs"].append(run(policy, "bench", index + 1))
                destination.write_text(json.dumps(summary, indent=2) + "\n")
        data = [load(r) for r in summary["diagnostics"] + summary["runs"]]
        assert all(d["manifest"] == data[0]["manifest"] for d in data)
        assert all(d["valid_run"] for d in data)
        assert all(d["manifest"]["model"] == args.model for d in data)
        summary["manifest"] = data[0]["manifest"]
        if args.rounds:
            summary["aggregate"] = {p: aggregate(summary["runs"], p) for p in POLICIES}
            summary["paired"] = []
            for index in range(1, args.rounds + 1):
                pair = {
                    p: load(
                        next(
                            r
                            for r in summary["runs"]
                            if r["variant"] == p and r["round"] == index
                        )
                    )
                    for p in POLICIES
                }
                base = pair["vllm"]["candidate_summary"]["duration_ms"]
                row = dict(
                    round=index, duration_change_pct={}, answers_matching_baseline={}
                )
                left = {
                    r["request_id"]: r["answer"]
                    for r in pair["vllm"]["requests"]
                    if r["request_id"] in pair["vllm"]["manifest"]["candidate_order"]
                }
                for policy in ("lfu", "queue"):
                    row["duration_change_pct"][policy] = round(
                        (pair[policy]["candidate_summary"]["duration_ms"] / base - 1)
                        * 100,
                        3,
                    )
                    right = {
                        r["request_id"]: r["answer"] for r in pair[policy]["requests"]
                    }
                    row["answers_matching_baseline"][policy] = sum(
                        text == right[label] for label, text in left.items()
                    )
                summary["paired"].append(row)
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SUMMARY {destination}", flush=True)
    print(json.dumps(summary.get("aggregate", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
