"""Compare unchunked FCFS and bounded skip/continue, with earlier experiments off."""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path

from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    folder = ROOT / "results/skip-continue" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    d = dict(experiment="05-skip-continue", diagnostics=[], runs=[], complete=False)
    path = folder / "summary.json"

    def run(variant, mode, label, profile):
        record = run_one(
            folder,
            variant,
            mode,
            f"{profile}-{label}",
            launcher="src/serve_skip_continue.sh",
            launcher_args=[profile],
            client_script="workloads/compare_skip_continue.py"
            if profile == "targeted"
            else "workloads/compare_prefix_policies.py",
        )
        record["profile"] = profile
        record["round"] = label
        raw = json.loads(Path(record["result"]).read_text())
        if mode == "trace":
            stats = dict(
                budget_skips=0,
                budget_stops=0,
                scan_limits=0,
                admission_order=[],
                steps=[],
            )
            affected = set()
            events = next(Path(record["run_dir"]).glob("events-*.jsonl"))
            schedules = []
            for line in events.read_text().splitlines():
                e = json.loads(line)
                req = e.get("request") or {}
                related = raw["run_id"] in req.get("id", "")
                if e["event"] == "BUDGET_BACKFILL_SKIP" and related:
                    stats["budget_skips"] += 1
                    affected.add(e["step"])
                if (
                    e["event"] == "STOP_WAITING_ADMISSION"
                    and related
                    and e.get("reason") == "unchunked_prompt_exceeds_budget"
                ):
                    stats["budget_stops"] += 1
                    affected.add(e["step"])
                if e["event"] == "BACKFILL_SCAN_LIMIT":
                    stats["scan_limits"] += 1
                if e["event"] == "ADMIT_OR_RESUME" and related:
                    stats["admission_order"].append(req["id"])
                if e["event"] == "SCHEDULE_END" and any(
                    raw["run_id"] in key for key in e.get("assigned_positions", {})
                ):
                    schedules.append(e)
            stats["steps"] = [
                dict(
                    step=e["step"],
                    assigned=e["assigned_total"],
                    unused=512 - e["assigned_total"],
                )
                for e in schedules
                if e["step"] in affected
            ]
            record["backfill_trace"] = stats
        return record

    try:
        for profile in ["original", "targeted"]:
            for variant in ["baseline", "skip"]:
                d["diagnostics"].append(run(variant, "trace", "diagnostic", profile))
                path.write_text(json.dumps(d, indent=2) + "\n")
        for index in range(args.rounds):
            for variant in (
                ["baseline", "skip"] if index % 2 == 0 else ["skip", "baseline"]
            ):
                d["runs"].append(run(variant, "bench", index + 1, "targeted"))
                path.write_text(json.dumps(d, indent=2) + "\n")
        d["aggregate"] = {}
        d["per_request"] = {}
        manifests = {}
        for run_ in d["diagnostics"] + d["runs"]:
            raw = json.loads(Path(run_["result"]).read_text())
            profile = run_["profile"]
            assert manifests.setdefault(profile, raw["manifest"]) == raw["manifest"]
        d["manifests"] = manifests
        for variant in ["baseline", "skip"]:
            runs = [r for r in d["runs"] if r["variant"] == variant]
            ds = [json.loads(Path(r["result"]).read_text()) for r in runs]
            d["aggregate"][variant] = dict(
                duration_median_ms=statistics.median(
                    r["candidate_summary"]["duration_ms"] for r in runs
                ),
                throughput_median_tokens_s=statistics.median(
                    r["candidate_summary"]["output_tokens_per_second"] for r in runs
                ),
                first_text_median_ms=statistics.median(
                    r["candidate_summary"]["first_text_ms"]["median"] for r in runs
                ),
                latency_median_ms=statistics.median(
                    r["candidate_summary"]["total_ms"]["median"] for r in runs
                ),
            )
            d["per_request"][variant] = {}
            for label in manifests["targeted"]["candidate_order"]:
                rows = [
                    next(r for r in x["requests"] if r["request_id"] == label)
                    for x in ds
                ]
                d["per_request"][variant][label] = dict(
                    first_text_median_ms=statistics.median(
                        r["first_text_ms"] for r in rows
                    ),
                    latency_median_ms=statistics.median(r["total_ms"] for r in rows),
                    answers=[r["answer"] for r in rows],
                )
        d["complete"] = True
    except Exception as exc:
        d["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(d, indent=2) + "\n")
        print(f"SUMMARY {path}", flush=True)
    print(json.dumps(d["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
