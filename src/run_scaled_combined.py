"""Run matched baseline/combined pairs at 25, 50 or 100 queued requests."""

import argparse
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download
from run_combined_batches import trace_counts
from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT


def group_metrics(data, labels):
    wanted = set(labels)
    rows = [r for r in data["requests"] if r["request_id"] in wanted]
    assert len(rows) == len(labels)
    return dict(
        first_text_median_ms=statistics.median(r["first_text_ms"] for r in rows),
        latency_median_ms=statistics.median(r["total_ms"] for r in rows),
        first_text_max_ms=max(r["first_text_ms"] for r in rows),
    )


def aggregate(records, variant):
    data = [
        json.loads(Path(r["result"]).read_text())
        for r in records
        if r["variant"] == variant
    ]
    groups = data[0]["manifest"]["prompt_groups"]
    return dict(
        duration_median_ms=statistics.median(
            r["candidate_summary"]["duration_ms"] for r in data
        ),
        duration_ms=[r["candidate_summary"]["duration_ms"] for r in data],
        throughput_median_tokens_s=statistics.median(
            r["candidate_summary"]["output_tokens_per_second"] for r in data
        ),
        first_text_median_ms=statistics.median(
            r["candidate_summary"]["first_text_ms"]["median"] for r in data
        ),
        latency_median_ms=statistics.median(
            r["candidate_summary"]["total_ms"]["median"] for r in data
        ),
        cached_tokens=[r["candidate_cached_prompt_tokens"] for r in data],
        output_tokens=[r["batch_summary"]["successful_output_tokens"] for r in data],
        groups={
            name: {
                key: statistics.median(group_metrics(r, labels)[key] for r in data)
                for key in (
                    "first_text_median_ms",
                    "latency_median_ms",
                    "first_text_max_ms",
                )
            }
            for name, labels in groups.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", help="Cached Hugging Face commit to pin")
    parser.add_argument("--candidates", type=int, choices=(25, 50, 100), default=50)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    revision = snapshot_download(
        args.model, revision=args.revision, local_files_only=True
    )
    os.environ["STUDY_MODEL"] = args.model
    os.environ["STUDY_MODEL_PATH"] = revision
    if args.revision:
        os.environ["STUDY_MODEL_REVISION"] = args.revision
    else:
        os.environ.pop("STUDY_MODEL_REVISION", None)
    name = args.model.replace("/", "--")
    folder = (
        ROOT
        / "results/scaled-combined"
        / name
        / f"candidates-{args.candidates}"
        / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = dict(
        experiment="07-scaled-combined",
        model=args.model,
        local_model_revision=Path(revision).name,
        candidates=args.candidates,
        rounds=args.rounds,
        diagnostics=[],
        runs=[],
        complete=False,
    )
    path = folder / "summary.json"

    def run(variant, mode, label):
        record = run_one(
            folder,
            variant,
            mode,
            label,
            launcher="src/serve_combined_comparison.sh",
            launcher_args=["targeted"],
            client_script="workloads/compare_scaled_combined.py",
            client_eviction_policy="lfu" if variant == "combined" else "vllm",
            client_candidates=args.candidates,
        )
        record["round"] = label
        if mode == "trace":
            record["trace"] = trace_counts(record)
        return record

    try:
        for variant in ("baseline", "combined"):
            summary["diagnostics"].append(run(variant, "trace", "diagnostic"))
            path.write_text(json.dumps(summary, indent=2) + "\n")
        for index in range(args.rounds):
            order = (
                ("baseline", "combined") if index % 2 == 0 else ("combined", "baseline")
            )
            for variant in order:
                summary["runs"].append(run(variant, "bench", index + 1))
                path.write_text(json.dumps(summary, indent=2) + "\n")
        records = summary["diagnostics"] + summary["runs"]
        data = [json.loads(Path(r["result"]).read_text()) for r in records]
        assert all(r["manifest"] == data[0]["manifest"] for r in data)
        assert all(r["valid_run"] for r in data)
        assert all(r["manifest"]["model"] == args.model for r in data)
        expected_tokens = args.candidates * 64 + 8 * 128
        assert all(
            r["batch_summary"]["successful_output_tokens"] == expected_tokens
            for r in data
        )
        groups = data[0]["manifest"]["prompt_groups"]
        assert len(groups["long"]) + len(groups["short"]) == args.candidates
        for r in data:
            for row in r["requests"]:
                if row["request_id"] in groups["long"]:
                    assert row["usage"]["prompt_tokens"] == 320
                if row["request_id"] in groups["short"]:
                    assert row["usage"]["prompt_tokens"] == 64
        summary["aggregate"] = {
            variant: aggregate(summary["runs"], variant)
            for variant in ("baseline", "combined")
        }
        summary["paired_duration_changes_pct"] = []
        summary["answer_match"] = []
        for index in range(1, args.rounds + 1):
            pair = {}
            for variant in ("baseline", "combined"):
                record = next(
                    r
                    for r in summary["runs"]
                    if r["variant"] == variant and r["round"] == index
                )
                pair[variant] = json.loads(Path(record["result"]).read_text())
            baseline = pair["baseline"]
            combined = pair["combined"]
            left = baseline["candidate_summary"]["duration_ms"]
            right = combined["candidate_summary"]["duration_ms"]
            summary["paired_duration_changes_pct"].append(
                round((right / left - 1) * 100, 3)
            )
            answers = [
                {r["request_id"]: r["answer"] for r in pair[name]["requests"]}
                for name in ("baseline", "combined")
            ]
            assert answers[0].keys() == answers[1].keys()
            summary["answer_match"].append(
                sum(answers[0][key] == answers[1][key] for key in answers[0])
            )
        summary["manifest"] = data[0]["manifest"]
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SUMMARY {path}", flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
