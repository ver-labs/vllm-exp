"""Fresh-server stock/LFU/aging-LFU comparisons at a fixed decay interval."""

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT

POLICIES = ("vllm", "lfu", "aged_lfu")


def load(record):
    return json.loads(Path(record["result"]).read_text())


def aggregate(records, policy):
    data = [load(r) for r in records if r["variant"] == policy]
    return dict(
        duration_ms=[d["batch_summary"]["duration_ms"] for d in data],
        median_duration_ms=statistics.median(
            d["batch_summary"]["duration_ms"] for d in data
        ),
        median_throughput=statistics.median(
            d["batch_summary"]["output_tokens_per_second"] for d in data
        ),
        median_probe_first_text_ms=statistics.median(
            d["phases"]["probe"]["first_text_ms"]["median"] for d in data
        ),
        median_probe_latency_ms=statistics.median(
            d["phases"]["probe"]["total_ms"]["median"] for d in data
        ),
        cached_tokens=[
            sum(r["cached_prompt_tokens"] for r in d["requests"]) for d in data
        ],
        probe_hits=[d["probe_hits"] for d in data],
        adaptation_cycles=[d["first_sustained_full_hit_cycle"] for d in data],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("shift", "stable"), required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("rounds must be nonnegative")
    os.environ["STUDY_MODEL"] = "Qwen/Qwen3-0.6B"
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        os.environ["STUDY_MODEL"],
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        local_files_only=True,
    )
    os.environ["STUDY_AGING_PROFILE"] = args.profile
    folder = (
        ROOT
        / "results/aging-lfu"
        / args.profile
        / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    path = folder / "summary.json"
    summary = dict(
        experiment="09-aging-lfu",
        profile=args.profile,
        half_life_blocks=64,
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
            launcher="src/serve_aging_comparison.sh",
            launcher_args=[args.profile],
            client_script="workloads/compare_aging_lfu.py",
            client_eviction_policy=policy,
        )
        record["round"] = label
        return record

    try:
        for policy in POLICIES:
            summary["diagnostics"].append(run(policy, "trace", "diagnostic"))
            path.write_text(json.dumps(summary, indent=2) + "\n")
        for index in range(args.rounds):
            for policy in POLICIES[index % 3 :] + POLICIES[: index % 3]:
                summary["runs"].append(run(policy, "bench", index + 1))
                path.write_text(json.dumps(summary, indent=2) + "\n")
        data = [load(r) for r in summary["diagnostics"] + summary["runs"]]
        assert all(d["manifest"] == data[0]["manifest"] for d in data)
        assert all(d["valid_run"] for d in data)
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
                base = pair["vllm"]["batch_summary"]["duration_ms"]
                answers = {
                    p: {r["request_id"]: r["answer"] for r in d["requests"]}
                    for p, d in pair.items()
                }
                summary["paired"].append(
                    dict(
                        round=index,
                        duration_change_pct={
                            p: round(
                                (d["batch_summary"]["duration_ms"] / base - 1) * 100, 3
                            )
                            for p, d in pair.items()
                        },
                        answers_matching_baseline={
                            p: sum(
                                text == answers[p][label]
                                for label, text in answers["vllm"].items()
                            )
                            for p in POLICIES
                        },
                    )
                )
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SUMMARY {path}", flush=True)
    print(json.dumps(summary.get("aggregate", {}), indent=2), flush=True)


if __name__ == "__main__":
    main()
