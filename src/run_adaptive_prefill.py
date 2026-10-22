"""Open-loop long-prompt bursts during generation: prefill policies."""

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from run_prefix_batches import ROOT, port_open, stop_server
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
VARIANTS = ("stock", "existing_adaptive", "decode_adaptive")


def workload(tokenizer):
    rows = []
    for index in range(24):
        ongoing = index < 4
        n = 64 if ongoing else (512 if index % 2 == 0 else 1024)
        tokens = tokenizer.encode(
            f"Case {index}: Explain the following observations carefully. "
            + "The library records book reservations and sends reminders to members. "
            * 130,
            add_special_tokens=False,
        )[:n]
        assert len(tokens) == n
        rows.append(
            dict(
                request_id=f"T{index:02}",
                tokens=tokens,
                prompt_length=n,
                output_tokens=128 if ongoing else 16,
                cohort="ongoing" if ongoing else "burst",
                arrival_s=0.0 if ongoing else (0.8 if index < 14 else 2.4),
            )
        )
    return dict(
        model=MODEL,
        revision=REVISION,
        requests=rows,
        arrival_model="open loop: four at 0s; ten at 0.8s; ten at 2.4s",
        running_slots=8,
        block_size=16,
        physical_blocks=1024,
        max_model_len=1280,
        token_budget=128,
        age_boost_seconds=0.75,
        age_boost_cap=64,
        adaptive_formula=(
            "min(128, max(16, 128//(1+2*decoders), "
            "64 if oldest prompt age>=0.75 else 0))"
        ),
        prefix_cache=False,
        skip_continue=False,
    )


def send(tokens, label, limit, origin):
    payload = dict(
        model=MODEL,
        prompt=tokens,
        max_tokens=limit,
        temperature=0,
        ignore_eos=True,
        stream=True,
        stream_options=dict(include_usage=True),
    )
    request = Request(
        "http://127.0.0.1:8001/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": label},
    )
    start = time.perf_counter()
    first, usage, finish, text, done = None, {}, None, "", False
    text_times = []
    with urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                done = True
                break
            event = json.loads(value)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("text") or ""
                if delta and first is None:
                    first = time.perf_counter()
                if delta:
                    text_times.append(time.perf_counter())
                text += delta
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    assert done and finish and usage["completion_tokens"] == limit
    assert first is not None
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert 0 <= cached < len(tokens)
    return dict(
        trace_request_id=label,
        text_chunk_times_ms=[(t - start) * 1000 for t in text_times],
        chunk_gaps_ms=[(b - a) * 1000 for a, b in zip(text_times, text_times[1:])],
        tpot_ms=(text_times[-1] - text_times[0]) * 1000 / max(1, limit - 1),
        answer=text,
        usage=usage,
        cached_tokens=cached,
        started_ms=(start - origin) * 1000,
        finished_ms=(end - origin) * 1000,
        first_text_ms=(first - start) * 1000,
        total_ms=(end - start) * 1000,
    )


def stats(rows, duration_ms):
    result = dict(
        requests=len(rows),
        cached_tokens=sum(r["cached_tokens"] for r in rows),
        output_tokens=sum(r["usage"]["completion_tokens"] for r in rows),
    )
    for field in ("first_text_ms", "total_ms", "tpot_ms"):
        values = sorted(r[field] for r in rows)
        result[field] = dict(
            median=statistics.median(values),
            p95=values[math.ceil(0.95 * len(values)) - 1],
        )
    gaps = sorted(g for r in rows for g in r["chunk_gaps_ms"])
    result["text_chunk_gap_p95_ms"] = (
        gaps[math.ceil(0.95 * len(gaps)) - 1] if gaps else None
    )
    if duration_ms is not None:
        result.update(
            duration_ms=duration_ms,
            output_tokens_per_second=result["output_tokens"] / (duration_ms / 1000),
            requests_per_second=len(rows) / (duration_ms / 1000),
        )
    return result


def inspect_trace(path, rows, prefix):
    preemptions = mixed = max_running = max_waiting = 0
    caps, decoders, boosts, first_scheduled = [], [], 0, {}
    start_states = {}
    for line in path.open():
        event = json.loads(line)
        kind = event["event"]
        running = event.get("running", [])
        waiting = event.get("waiting", [])
        if kind == "SCHEDULE_BEGIN":
            start_states = {r["id"]: r for r in running + waiting}
        if not any(
            prefix in r["id"] for r in running + waiting
        ) and prefix not in event.get("request", {}).get("id", ""):
            continue
        max_running = max(max_running, len(running))
        max_waiting = max(max_waiting, len(waiting))
        preemptions += kind == "PREEMPT_END"
        if kind == "ADAPTIVE_PREFILL_CAP":
            caps.append(event["cap"])
            decoders.append(event["decoders"])
            boosts += event["age_boost"]
        if kind == "SCHEDULE_END":
            assigned = event["assigned_positions"]
            assert sum(assigned.values()) <= 128
            states = [start_states[r] for r in assigned if r in start_states]
            mixed += any(r["computed"] < r["prompt"] for r in states) and any(
                r["output"] > 0 and r["computed"] >= r["prompt"] for r in states
            )
            for rid in assigned:
                first_scheduled.setdefault(
                    rid, event["time"] if "time" in event else event.get("time_ns")
                )
    assert not preemptions and max_running >= 4 and max_waiting > 0 and mixed > 0
    assert all(
        any(r["trace_request_id"] in key for key in first_scheduled) for r in rows
    )
    if caps:
        assert min(caps) < max(caps) and boosts > 0 and max(decoders) >= 2
    return dict(
        preemptions=preemptions,
        mixed_steps=mixed,
        max_running=max_running,
        max_waiting=max_waiting,
        caps=sorted(set(caps)),
        age_boost_steps=boosts,
        maximum_decoders=max(decoders, default=0),
        all_requests_scheduled=True,
    )


def run(folder, manifest, variant, mode, round_id):
    concurrency = 24
    assert not port_open()
    name = f"{variant}-{mode}-r{round_id}"
    run_dir = folder / name
    run_dir.mkdir()
    env = os.environ.copy()
    env.update(
        STUDY_PARTIAL_PREFIX="0",
        STUDY_ADAPTIVE_PREFILL="1" if variant == "decode_adaptive" else "0",
        STUDY_PARTIAL_MIN_TOKENS="0",
        STUDY_CACHE_IMPL="incremental",
        STUDY_EVICTION_POLICY="vllm",
        STUDY_WAITING_POLICY="fcfs",
        STUDY_SKIP_CONTINUE="0",
        STUDY_CACHE_PROFILE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_STUDY_TRACE="1" if mode == "trace" else "0",
        VLLM_STUDY_TRACE_DIR=str(run_dir),
        VLLM_STUDY_TRACE_STDOUT="0",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_LOGGING_LEVEL="INFO",
    )
    command = [
        str(ROOT / ".venv/bin/vllm"),
        "serve",
        os.environ["STUDY_MODEL_PATH"],
        "--served-model-name",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--seed",
        "0",
        "--worker-cls",
        "auto",
        "--max-model-len",
        "1280",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "128",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "1024",
        "--prefix-match-unit",
        "16",
        "--no-enable-prefix-caching",
        "--long-prefill-token-threshold",
        "16" if variant == "existing_adaptive" else "0",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--enable-prompt-tokens-details",
    ]
    if variant == "existing_adaptive":
        command.append("--long-prefill-token-threshold-adaptive")
    record = dict(
        variant=variant,
        concurrency=concurrency,
        mode=mode,
        round=round_id,
        run_dir=str(run_dir),
        command=command,
        rows=[],
        complete=False,
    )
    path = run_dir / "result.json"
    print("START", name, flush=True)
    with (run_dir / "server.log").open("w") as log:
        server = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited: {run_dir}")
                try:
                    with urlopen("http://127.0.0.1:8001/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Startup")
                time.sleep(0.5)
            send(
                manifest["requests"][0]["tokens"][:32],
                name + "-warmup",
                1,
                time.perf_counter(),
            )
            origin = time.perf_counter()
            prefix = name + "-timed-"

            def task(item):
                time.sleep(max(0, origin + item["arrival_s"] - time.perf_counter()))
                result = send(
                    item["tokens"],
                    prefix + item["request_id"],
                    item["output_tokens"],
                    origin,
                )
                result["dispatch_lag_ms"] = (
                    result["started_ms"] - item["arrival_s"] * 1000
                )
                result.update({k: v for k, v in item.items() if k != "tokens"})
                return result

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                jobs = [executor.submit(task, item) for item in manifest["requests"]]
                for job in as_completed(jobs):
                    record["rows"].append(job.result())
            duration = (time.perf_counter() - origin) * 1000
            record["rows"].sort(key=lambda row: row["request_id"])
            rows = record["rows"]
            assert len(rows) == 24
            record["summary"] = stats(rows, duration)
            record["groups"] = {
                cohort: stats([r for r in rows if r["cohort"] == cohort], None)
                for cohort in ("ongoing", "burst")
            }
            record["length_groups"] = {
                str(n): stats([r for r in rows if r["prompt_length"] == n], None)
                for n in (64, 512, 1024)
            }
            intervals = [(r["started_ms"], 1) for r in rows] + [
                (r["finished_ms"], -1) for r in rows
            ]
            active = peak = 0
            for _, delta in sorted(intervals):
                active += delta
                peak = max(peak, active)
            record["client_peak_inflight"] = peak
            assert peak >= 12
            assert record["summary"]["output_tokens"] == 832
            assert all(r["cached_tokens"] == 0 for r in rows)
            record["max_dispatch_lag_ms"] = max(r["dispatch_lag_ms"] for r in rows)
            record["complete"] = True
        except Exception as exc:
            record["error"] = repr(exc)
            raise
        finally:
            stop_server(server)
            path.write_text(json.dumps(record, indent=2) + "\n")
            time.sleep(3)
    if mode == "trace":
        record["diagnostic"] = inspect_trace(
            next(run_dir.glob("events-*.jsonl")), record["rows"], prefix
        )
        path.write_text(json.dumps(record, indent=2) + "\n")
    print("DONE", name, round(record["summary"]["duration_ms"], 1), "ms", flush=True)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        MODEL, revision=REVISION, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["STUDY_MODEL_PATH"], local_files_only=True
    )
    manifest = workload(tokenizer)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    folder = (
        ROOT / "results/adaptive-prefill" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = dict(manifest=manifest, diagnostics=[], runs=[], complete=False)
    path = folder / "summary.json"
    try:
        for variant in VARIANTS:
            summary["diagnostics"].append(run(folder, manifest, variant, "trace", 0))
            path.write_text(json.dumps(summary, indent=2) + "\n")
        variants = list(VARIANTS)
        for round_id in range(1, 4):
            order = variants[round_id - 1 :] + variants[: round_id - 1]
            for variant in order:
                summary["runs"].append(
                    run(folder, manifest, variant, "bench", round_id)
                )
                path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["complete"] = True
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", path, flush=True)


if __name__ == "__main__":
    main()
