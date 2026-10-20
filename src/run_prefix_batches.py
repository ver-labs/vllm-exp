"""Run fresh-server comparisons sequentially and save their provenance."""

import argparse
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["fcfs", "prefix_first", "fcfs_head_prefix"]


def port_open():
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", 8001)) == 0


def stop_server(process):
    for sig, timeout in [
        (signal.SIGINT, 15),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ]:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        deadline = time.monotonic() + timeout
        while port_open() and time.monotonic() < deadline:
            time.sleep(0.5)
        if not port_open():
            return
    raise RuntimeError("Server did not release port 8001")


def run_one(folder, policy, mode, label):
    if port_open():
        raise RuntimeError("Port 8001 is occupied; stop the existing server first")
    server_log = folder / f"{label}-{policy}-{mode}-launcher.log"
    client_log = folder / f"{label}-{policy}-{mode}-client.log"
    print(f"START {label}: {policy} ({mode}); starting fresh server", flush=True)
    with server_log.open("w") as log:
        process = subprocess.Popen(
            ["bash", "src/serve_prefix_comparison.sh", policy, mode],
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
            result = json.loads(path.read_text())
            if not result["valid_run"]:
                raise RuntimeError(f"Invalid workload setup; see {path}")
            print(
                f"DONE {label}: {policy}; candidates "
                f"{result['candidate_summary']['duration_ms']} ms; {path.name}",
                flush=True,
            )
            return {
                "round": label,
                "policy": policy,
                "mode": mode,
                "result": str(path),
                "launcher_log": str(server_log),
                "client_log": str(client_log),
                "candidate_summary": result["candidate_summary"],
                "batch_summary": result["batch_summary"],
                "trace_analysis": result["trace_analysis"],
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
        ROOT / "results/prefix-comparison" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = {
        "rounds_requested": args.rounds,
        "diagnostic": None,
        "runs": [],
        "aggregate": {},
        "complete": False,
    }
    destination = folder / "summary.json"
    try:
        summary["diagnostic"] = run_one(
            folder, "fcfs_head_prefix", "trace", "diagnostic"
        )
        for index in range(args.rounds):
            # Rotate order to reduce a consistent first/last-policy timing bias.
            order = POLICIES[index % 3 :] + POLICIES[: index % 3]
            for policy in order:
                summary["runs"].append(run_one(folder, policy, "bench", index + 1))
                destination.write_text(json.dumps(summary, indent=2) + "\n")
        for policy in POLICIES:
            runs = [r for r in summary["runs"] if r["policy"] == policy]
            summary["aggregate"][policy] = {
                "candidate_duration_median_ms": statistics.median(
                    r["candidate_summary"]["duration_ms"] for r in runs
                ),
                "candidate_duration_range_ms": [
                    min(r["candidate_summary"]["duration_ms"] for r in runs),
                    max(r["candidate_summary"]["duration_ms"] for r in runs),
                ],
                "candidate_throughput_median_tokens_s": statistics.median(
                    r["candidate_summary"]["output_tokens_per_second"] for r in runs
                ),
                "median_first_text_median_ms": statistics.median(
                    r["candidate_summary"]["first_text_ms"]["median"] for r in runs
                ),
                "median_request_latency_median_ms": statistics.median(
                    r["candidate_summary"]["total_ms"]["median"] for r in runs
                ),
            }
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
