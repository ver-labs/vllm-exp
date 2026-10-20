"""A-Y concurrent workload; vary eviction alone with FCFS scheduling."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from run_prefix_batches import ROOT, port_open, stop_server


def run_one(folder, policy, mode, round_number):
    if port_open():
        raise RuntimeError("Port 8001 is occupied")
    server_log = folder / f"{round_number}-{policy}-{mode}-server.log"
    client_log = folder / f"{round_number}-{policy}-{mode}-client.log"
    print(f"START {round_number}: {policy} ({mode})", flush=True)
    with server_log.open("w") as log:
        process = subprocess.Popen(
            ["bash", "src/serve_cache_eviction.sh", policy, mode, "ay"],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited; see {server_log}")
                try:
                    with urlopen("http://127.0.0.1:8001/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server readiness timed out; see {server_log}")
                time.sleep(0.5)
            command = [
                sys.executable,
                "workloads/compare_prefix_policies.py",
                "--policy",
                "fcfs",
                "--candidates",
                "25",
                "--eviction-policy",
                policy,
            ]
            if mode == "trace":
                command.append("--trace-latest")
            with client_log.open("w") as client:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=client,
                    stderr=subprocess.STDOUT,
                    timeout=240,
                )
            saved = [
                line.removeprefix("Saved: ").strip()
                for line in client_log.read_text().splitlines()
                if line.startswith("Saved: ")
            ]
            if completed.returncode or not saved:
                raise RuntimeError(f"Workload failed; see {client_log}")
            path = Path(saved[-1])
            data = json.loads(path.read_text())
            if not data["valid_run"]:
                raise RuntimeError(f"Invalid workload setup; see {path}")
            print(
                f"DONE {round_number}: {policy}; A-Y "
                f"{data['candidate_summary']['duration_ms']} ms; "
                f"cached tokens {data['candidate_cached_prompt_tokens']}",
                flush=True,
            )
            return {
                "round": round_number,
                "policy": policy,
                "mode": mode,
                "result": str(path),
                "server_log": str(server_log),
                "client_log": str(client_log),
                "candidate_summary": data["candidate_summary"],
                "batch_summary": data["batch_summary"],
                "cached_prompt_tokens": data["candidate_cached_prompt_tokens"],
                "trace_analysis": data["trace_analysis"],
            }
        finally:
            stop_server(process)
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    folder = (
        ROOT / "results/cache-eviction-ay" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = {
        "workload": "controlled-prefix-v2-ay",
        "rounds_requested": args.rounds,
        "scheduling": "fcfs",
        "diagnostics": [],
        "runs": [],
        "aggregate": {},
        "per_request": {},
        "complete": False,
    }
    destination = folder / "summary.json"
    try:
        for policy in ["vllm", "lfu"]:
            summary["diagnostics"].append(
                run_one(folder, policy, "trace", "diagnostic")
            )
            destination.write_text(json.dumps(summary, indent=2) + "\n")
        # Time both policies even if the diagnostic reveals no retention benefit.
        for index in range(args.rounds):
            for policy in ["vllm", "lfu"] if index % 2 == 0 else ["lfu", "vllm"]:
                summary["runs"].append(run_one(folder, policy, "bench", index + 1))
                destination.write_text(json.dumps(summary, indent=2) + "\n")
        for policy in ["vllm", "lfu"]:
            runs = [run for run in summary["runs"] if run["policy"] == policy]
            summary["aggregate"][policy] = {
                "candidate_duration_median_ms": statistics.median(
                    run["candidate_summary"]["duration_ms"] for run in runs
                ),
                "candidate_duration_range_ms": [
                    min(run["candidate_summary"]["duration_ms"] for run in runs),
                    max(run["candidate_summary"]["duration_ms"] for run in runs),
                ],
                "candidate_throughput_median_tokens_s": statistics.median(
                    run["candidate_summary"]["output_tokens_per_second"] for run in runs
                ),
                "median_first_text_ms": statistics.median(
                    run["candidate_summary"]["first_text_ms"]["median"] for run in runs
                ),
                "median_request_latency_ms": statistics.median(
                    run["candidate_summary"]["total_ms"]["median"] for run in runs
                ),
                "cached_prompt_tokens_per_run": [
                    run["cached_prompt_tokens"] for run in runs
                ],
            }
            results = [json.loads(Path(run["result"]).read_text()) for run in runs]
            summary["per_request"][policy] = {}
            for label in results[0]["manifest"]["candidate_order"]:
                rows = [
                    next(row for row in d["requests"] if row["request_id"] == label)
                    for d in results
                ]
                summary["per_request"][policy][label] = {
                    "median_first_text_ms": statistics.median(
                        row["first_text_ms"] for row in rows
                    ),
                    "median_total_ms": statistics.median(
                        row["total_ms"] for row in rows
                    ),
                    "cached_tokens_per_run": [
                        row["cached_prompt_tokens"] for row in rows
                    ],
                }
        with (folder / "per-request.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "request",
                    "vllm_first_text_s",
                    "lfu_first_text_s",
                    "vllm_completion_s",
                    "lfu_completion_s",
                ]
            )
            for label in summary["per_request"]["vllm"]:
                a = summary["per_request"]["vllm"][label]
                b = summary["per_request"]["lfu"][label]
                writer.writerow(
                    [
                        label,
                        a["median_first_text_ms"] / 1000,
                        b["median_first_text_ms"] / 1000,
                        a["median_total_ms"] / 1000,
                        b["median_total_ms"] / 1000,
                    ]
                )
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        destination.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"SUMMARY {destination}", flush=True)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
