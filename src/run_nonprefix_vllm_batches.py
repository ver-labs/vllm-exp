"""Fresh-server experiment03 against unchanged vLLM on the original A-Y workload."""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from run_prefix_batches import ROOT, port_open, stop_server


def run_one(
    folder,
    variant,
    mode,
    label,
    *,
    launcher="src/serve_nonprefix_comparison.sh",
    probe=None,
):
    if port_open():
        raise RuntimeError("Port 8001 is occupied")
    server_log = folder / f"{label}-{variant}-{mode}-server.log"
    client_log = folder / f"{label}-{variant}-{mode}-client.log"
    print(f"START {label}: {variant} ({mode})", flush=True)
    with server_log.open("w") as stream:
        server = subprocess.Popen(
            ["bash", launcher, variant, mode],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {server_log}")
                try:
                    with urlopen("http://127.0.0.1:8001/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Server readiness: {server_log}")
                time.sleep(0.5)
            run_dir = Path(
                next(
                    line.removeprefix("Server results: ").strip()
                    for line in server_log.read_text().splitlines()
                    if line.startswith("Server results: ")
                )
            )
            probe_result = probe(variant) if probe else None
            cmd = [
                sys.executable,
                "workloads/compare_prefix_policies.py",
                "--policy",
                "fcfs",
                "--candidates",
                "25",
                "--eviction-policy",
                "vllm",
            ]
            if mode == "trace":
                cmd += ["--trace", str(next(run_dir.glob("events-*.jsonl")))]
            with client_log.open("w") as stream:
                client = subprocess.run(
                    cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=240
                )
            if client.returncode:
                raise RuntimeError(f"Client failed: {client_log}")
            raw = Path(
                next(
                    line.removeprefix("Saved: ").strip()
                    for line in client_log.read_text().splitlines()
                    if line.startswith("Saved: ")
                )
            )
            data = json.loads(raw.read_text())
            if not data["valid_run"]:
                raise RuntimeError(f"Invalid run: {raw}")
            copied = folder / f"{label}-{variant}-{mode}.json"
            copied.write_bytes(raw.read_bytes())
            events = run_dir / "nonprefix-events.jsonl"
            records = (
                [json.loads(line) for line in events.read_text().splitlines()]
                if events.exists()
                else []
            )
            if variant == "repair" and not records:
                print("NOTE: no eligible middle-block reuse observed", flush=True)
            result = dict(
                label=label,
                probe=probe_result,
                variant=variant,
                mode=mode,
                result=str(copied),
                original_result=str(raw),
                server_log=str(server_log),
                client_log=str(client_log),
                run_dir=str(run_dir),
                candidate_summary=data["candidate_summary"],
                batch_summary=data["batch_summary"],
                trace_analysis=data["trace_analysis"],
                nonprefix_totals=records[-1] if records else {},
                sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
            )
            print(
                f"DONE {label}: {variant}; "
                f"A-Y {data['candidate_summary']['duration_ms']}ms; "
                f"nonprefix skipped "
                f"{result['nonprefix_totals'].get('skipped_rows_total', 0)}",
                flush=True,
            )
            return result
        finally:
            stop_server(server)
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("rounds must be nonnegative")
    folder = ROOT / "results/nonprefix-vllm" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    summary = dict(
        experiment="03-nonprefix-vllm-ay",
        rounds_requested=args.rounds,
        diagnostics=[],
        runs=[],
        complete=False,
    )
    path = folder / "summary.json"
    try:
        for variant in ["repair", "baseline"]:
            summary["diagnostics"].append(
                run_one(folder, variant, "trace", "diagnostic")
            )
            path.write_text(json.dumps(summary, indent=2) + "\n")
        for index in range(args.rounds):
            order = ["baseline", "repair"] if index % 2 == 0 else ["repair", "baseline"]
            for variant in order:
                summary["runs"].append(run_one(folder, variant, "bench", index + 1))
                path.write_text(json.dumps(summary, indent=2) + "\n")
        all_runs = summary["diagnostics"] + summary["runs"]
        data = [json.loads(Path(run["result"]).read_text()) for run in all_runs]
        assert all(d["manifest"] == data[0]["manifest"] for d in data)
        assert all(d["batch_summary"]["successful_output_tokens"] == 1984 for d in data)
        summary["manifest"] = data[0]["manifest"]
        summary["aggregate"] = {}
        for variant in ["baseline", "repair"]:
            runs = [r for r in summary["runs"] if r["variant"] == variant]
            if not runs:
                continue
            summary["aggregate"][variant] = {
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
                "median_first_text_ms": statistics.median(
                    r["candidate_summary"]["first_text_ms"]["median"] for r in runs
                ),
                "median_request_latency_ms": statistics.median(
                    r["candidate_summary"]["total_ms"]["median"] for r in runs
                ),
                "nonprefix_skipped_rows_per_run": [
                    r["nonprefix_totals"].get("skipped_rows_total", 0) for r in runs
                ],
            }
        per_request = []
        for label in summary["manifest"]["candidate_order"]:
            answers = {}
            for variant in ["baseline", "repair"]:
                ds = [
                    json.loads(Path(r["result"]).read_text())
                    for r in summary["runs"]
                    if r["variant"] == variant
                ]
                rows = [
                    next(x for x in d["requests"] if x["request_id"] == label)
                    for d in ds
                ]
                answers[variant] = [r["answer"] for r in rows]
                for i, r in enumerate(rows, 1):
                    per_request.append(
                        dict(
                            request=label,
                            variant=variant,
                            round=i,
                            first_text_ms=r["first_text_ms"],
                            total_ms=r["total_ms"],
                            cached_prompt_tokens=r["cached_prompt_tokens"],
                            answer=r["answer"],
                        )
                    )
            if summary["runs"]:
                summary.setdefault("answer_exact_agreement_by_request", {})[label] = [
                    a == b for a, b in zip(answers["baseline"], answers["repair"])
                ]
        if per_request:
            with (folder / "per-request.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(per_request[0]))
                writer.writeheader()
                writer.writerows(per_request)
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
