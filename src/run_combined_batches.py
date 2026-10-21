"""Compare experiments 2+4+5 enabled together with native vLLM."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT, port_open, stop_server


def run_temporal(folder, variant, mode, label):
    if port_open():
        raise RuntimeError("Port 8001 is occupied")
    server_log = folder / f"temporal-{label}-{variant}-{mode}-server.log"
    client_log = folder / f"temporal-{label}-{variant}-{mode}-client.log"
    print(f"START temporal {label}: {variant} ({mode})", flush=True)
    with server_log.open("w") as stream:
        server = subprocess.Popen(
            ["bash", "src/serve_combined_comparison.sh", variant, mode, "temporal"],
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
            policy = "lfu" if variant == "combined" else "vllm"
            with client_log.open("w") as stream:
                client = subprocess.run(
                    [
                        sys.executable,
                        "workloads/compare_cache_eviction.py",
                        "--policy",
                        policy,
                    ],
                    cwd=ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=240,
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
            copied = folder / f"temporal-{label}-{variant}-{mode}.json"
            copied.write_bytes(raw.read_bytes())
            hot_hits = [
                r["cached_prompt_tokens"]
                for r in data["requests"]
                if r["phase"] == "hot_probe"
            ]
            print(
                f"DONE temporal {label}: {variant}; "
                f"{data['summary']['duration_ms']}ms; "
                f"hot hits {hot_hits}",
                flush=True,
            )
            return dict(
                profile="temporal",
                label=label,
                variant=variant,
                mode=mode,
                result=str(copied),
                original_result=str(raw),
                server_log=str(server_log),
                client_log=str(client_log),
                run_dir=str(run_dir),
                sha256=hashlib.sha256(copied.read_bytes()).hexdigest(),
            )
        finally:
            stop_server(server)
            time.sleep(3)


def trace_counts(record):
    path = next(Path(record["run_dir"]).glob("events-*.jsonl"))
    counts = dict(skip=0, budget_stops=0, cached_evictions=0, preemptions=0)
    header = None
    for line in path.read_text().splitlines():
        event = json.loads(line)
        kind = event["event"]
        if kind == "TRACE_READY":
            header = {
                key: event.get(key)
                for key in [
                    "scheduler_source",
                    "eviction_policy",
                    "skip_continue",
                    "chunked_prefill",
                    "block_size",
                    "token_budget_limit",
                ]
            }
        if kind == "BUDGET_BACKFILL_SKIP":
            counts["skip"] += 1
        if (
            kind == "STOP_WAITING_ADMISSION"
            and event.get("reason") == "unchunked_prompt_exceeds_budget"
        ):
            counts["budget_stops"] += 1
        if kind == "EVICT_CACHED_BLOCK":
            counts["cached_evictions"] += 1
        if kind == "PREEMPT_END":
            counts["preemptions"] += 1
    assert header is not None
    assert "vllm-src/vllm/v1/core/sched/scheduler.py" in header["scheduler_source"]
    assert header["eviction_policy"] == (
        "lfu" if record["variant"] == "combined" else "vllm"
    )
    assert header["skip_continue"] == (record["variant"] == "combined")
    assert header["chunked_prefill"] is False
    return dict(header=header, counts=counts, events=str(path))


def aggregate(records, profile, variant):
    runs = [
        record
        for record in records
        if record["profile"] == profile and record["variant"] == variant
    ]
    ds = [json.loads(Path(record["result"]).read_text()) for record in runs]
    if profile == "temporal":
        return dict(
            batch_median_ms=statistics.median(d["summary"]["duration_ms"] for d in ds),
            probe_first_text_median_ms=statistics.median(
                d["phases"]["hot_probe"]["first_text_ms"]["median"] for d in ds
            ),
            probe_cached_tokens=[
                [
                    r["cached_prompt_tokens"]
                    for r in d["requests"]
                    if r["phase"] == "hot_probe"
                ]
                for d in ds
            ],
            output_tokens=[
                sum(r["usage"]["completion_tokens"] for r in d["requests"]) for d in ds
            ],
            durations_ms=[d["summary"]["duration_ms"] for d in ds],
        )
    result = dict(
        batch_median_ms=statistics.median(
            d["candidate_summary"]["duration_ms"] for d in ds
        ),
        throughput_median_tokens_s=statistics.median(
            d["candidate_summary"]["output_tokens_per_second"] for d in ds
        ),
        first_text_median_ms=statistics.median(
            d["candidate_summary"]["first_text_ms"]["median"] for d in ds
        ),
        request_latency_median_ms=statistics.median(
            d["candidate_summary"]["total_ms"]["median"] for d in ds
        ),
        cached_tokens=[d["candidate_cached_prompt_tokens"] for d in ds],
        output_tokens=[d["batch_summary"]["successful_output_tokens"] for d in ds],
        durations_ms=[d["candidate_summary"]["duration_ms"] for d in ds],
    )
    if profile == "targeted":
        result["groups"] = {}
        for group, labels in [("long", "ABCDEFGHIJKLM"), ("short", "NOPQRSTUVWXY")]:
            result["groups"][group] = dict(
                first_text_median_ms=statistics.median(
                    statistics.median(
                        r["first_text_ms"]
                        for r in d["requests"]
                        if r["request_id"] in labels
                    )
                    for d in ds
                ),
                latency_median_ms=statistics.median(
                    statistics.median(
                        r["total_ms"]
                        for r in d["requests"]
                        if r["request_id"] in labels
                    )
                    for d in ds
                ),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    folder = ROOT / "results/combined" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    summary = dict(
        experiment="06-combined",
        rounds=args.rounds,
        diagnostics=[],
        runs=[],
        complete=False,
    )
    path = folder / "summary.json"

    def run(profile, variant, mode, label):
        if profile == "temporal":
            record = run_temporal(folder, variant, mode, label)
        else:
            record = run_one(
                folder,
                variant,
                mode,
                f"{profile}-{label}",
                launcher="src/serve_combined_comparison.sh",
                launcher_args=[profile],
                client_script=(
                    "workloads/compare_skip_continue.py"
                    if profile == "targeted"
                    else "workloads/compare_prefix_policies.py"
                ),
                client_eviction_policy="lfu" if variant == "combined" else "vllm",
            )
            record["profile"] = profile
        record["label"] = label
        if mode == "trace":
            record["trace"] = trace_counts(record)
        return record

    try:
        for profile in ["ay", "targeted", "temporal"]:
            for variant in ["baseline", "combined"]:
                summary["diagnostics"].append(
                    run(profile, variant, "trace", "diagnostic")
                )
                path.write_text(json.dumps(summary, indent=2) + "\n")
            for index in range(args.rounds):
                order = (
                    ["baseline", "combined"]
                    if index % 2 == 0
                    else ["combined", "baseline"]
                )
                for variant in order:
                    summary["runs"].append(run(profile, variant, "bench", index + 1))
                    path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["aggregate"] = {
            profile: {
                variant: aggregate(summary["runs"], profile, variant)
                for variant in ["baseline", "combined"]
            }
            for profile in ["ay", "targeted", "temporal"]
        }
        summary["manifests_match"] = {}
        summary["answer_match"] = {}
        for profile in ["ay", "targeted", "temporal"]:
            records = [
                r
                for r in summary["diagnostics"] + summary["runs"]
                if r["profile"] == profile
            ]
            ds = [json.loads(Path(record["result"]).read_text()) for record in records]
            summary["manifests_match"][profile] = all(
                d["manifest"] == ds[0]["manifest"] for d in ds
            )
            assert summary["manifests_match"][profile]
            baseline = [
                json.loads(Path(r["result"]).read_text())
                for r in summary["runs"]
                if r["profile"] == profile and r["variant"] == "baseline"
            ]
            combined = [
                json.loads(Path(r["result"]).read_text())
                for r in summary["runs"]
                if r["profile"] == profile and r["variant"] == "combined"
            ]
            agreement = []
            for left, right in zip(baseline, combined):
                a = {r["request_id"]: r["answer"] for r in left["requests"]}
                b = {r["request_id"]: r["answer"] for r in right["requests"]}
                assert set(a) == set(b)
                agreement.append(sum(a[key] == b[key] for key in a))
            summary["answer_match"][profile] = dict(
                exact_matches_per_round=agreement,
                requests_per_round=len(baseline[0]["requests"]),
            )
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
