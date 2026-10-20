"""Fresh-server, matched baseline-versus-LFU cache eviction comparisons."""

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from run_prefix_batches import ROOT, port_open, stop_server


def run_one(folder, policy, mode, label):
    if port_open():
        raise RuntimeError("Port 8001 is occupied; stop the previous server first")
    server_log = folder / f"{label}-{policy}-{mode}-server.log"
    client_log = folder / f"{label}-{policy}-{mode}-client.log"
    print(f"START {label}: {policy} ({mode})", flush=True)
    with server_log.open("w") as log:
        process = subprocess.Popen(
            ["bash", "src/serve_cache_eviction.sh", policy, mode],
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
                "workloads/compare_cache_eviction.py",
                "--policy",
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
                raise RuntimeError(f"Invalid workload; see {path}")
            probes = [row for row in data["requests"] if row["phase"] == "hot_probe"]
            first_ms = data["phases"]["hot_probe"]["first_text_ms"]["median"]
            print(
                f"DONE {label}: {policy}; probe cache hits "
                f"{[row['cached_prompt_tokens'] for row in probes]}; "
                f"median first-text {first_ms} ms",
                flush=True,
            )
            return {
                "round": label,
                "policy": policy,
                "mode": mode,
                "result": str(path),
                "server_log": str(server_log),
                "client_log": str(client_log),
                "phases": data["phases"],
                "summary": data["summary"],
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
    folder = ROOT / "results/cache-eviction" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    summary = {
        "rounds_requested": args.rounds,
        "diagnostics": [],
        "runs": [],
        "aggregate": {},
        "complete": False,
    }
    destination = folder / "summary.json"
    try:
        for policy in ["vllm", "lfu"]:
            summary["diagnostics"].append(
                run_one(folder, policy, "trace", "diagnostic")
            )
            destination.write_text(json.dumps(summary, indent=2) + "\n")
        baseline, candidate = summary["diagnostics"]
        summary["mechanism_demonstrated"] = (
            candidate["phases"]["hot_probe"]["cached_prompt_tokens"]
            > baseline["phases"]["hot_probe"]["cached_prompt_tokens"]
        )
        if not summary["mechanism_demonstrated"]:
            raise RuntimeError(
                "Diagnostic did not show preserved hot prefixes; inspect before timing"
            )
        for index in range(args.rounds):
            for policy in ["vllm", "lfu"] if index % 2 == 0 else ["lfu", "vllm"]:
                summary["runs"].append(run_one(folder, policy, "bench", index + 1))
                destination.write_text(json.dumps(summary, indent=2) + "\n")
        for policy in ["vllm", "lfu"]:
            runs = [run for run in summary["runs"] if run["policy"] == policy]
            summary["aggregate"][policy] = {}
            for phase in ["no_reuse_control", "hot_probe"]:
                values = [run["phases"][phase] for run in runs]
                summary["aggregate"][policy][phase] = {
                    "median_first_text_ms": statistics.median(
                        v["first_text_ms"]["median"] for v in values
                    ),
                    "median_total_ms": statistics.median(
                        v["total_ms"]["median"] for v in values
                    ),
                    "cached_tokens_per_run": [
                        v["cached_prompt_tokens"] for v in values
                    ],
                    "first_text_median_range_ms": [
                        min(v["first_text_ms"]["median"] for v in values),
                        max(v["first_text_ms"]["median"] for v in values),
                    ],
                }
            summary["aggregate"][policy]["median_batch_ms"] = statistics.median(
                run["summary"]["duration_ms"] for run in runs
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
