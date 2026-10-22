"""Mixed-length closed-loop load for exact partial-prefix reuse."""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from run_partial_threshold import post
from run_prefix_batches import ROOT, port_open, stop_server
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
VARIANTS = {"stock": (16, 0), "partial": (1, 0), "minimum8": (1, 8)}


def workload(tokenizer):
    seeds, rows = [], []
    lengths = (64, 128, 256, 512)
    outputs = (16, 32, 64)
    replacement = tokenizer.encode(" volcano", add_special_tokens=False)[-1]
    for family in range(16):
        length = lengths[family % 4]
        limit = outputs[family % 3]
        document = tokenizer.encode(
            f"Document {family}: A library team manages reservations and reminders. "
            + "Members search the catalogue and collect reserved books on time. " * 80,
            add_special_tokens=False,
        )[:length]
        cold = tokenizer.encode(
            f"Unrelated case {family}: A coastal team records water temperature. "
            + "Observers compare reef photographs and record fish at different sites. "
            * 80,
            add_special_tokens=False,
        )[:length]
        assert len(document) == len(cold) == length
        seeds.append(document)
        offset = 1 + family * 7 % 15
        partial = document.copy()
        assert partial[length - 16 + offset] != replacement
        partial[length - 16 + offset] = replacement
        for cohort, tokens in (
            ("repeat", document),
            ("partial", partial),
            ("unrelated", cold),
        ):
            rows.append(
                dict(
                    family=family,
                    cohort=cohort,
                    tokens=tokens,
                    prompt_length=length,
                    output_tokens=limit,
                    divergence_offset=offset if cohort == "partial" else None,
                )
            )
    random.Random(1202).shuffle(rows)
    for index, row in enumerate(rows):
        row["request_id"] = f"T{index:02}"
    return dict(
        model=MODEL,
        revision=REVISION,
        seeds=seeds,
        requests=rows,
        client_concurrency_levels=[4, 12],
        running_slots=4,
        block_size=16,
        physical_blocks=256,
        max_model_len=768,
        token_budget=128,
        temperature=0,
        ignore_eos=True,
        arrival_model="closed loop; fixed submission order; replacement on completion",
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
                text += delta
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    assert done and finish and usage["completion_tokens"] == limit
    assert first is not None
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert 0 <= cached < len(tokens)
    return dict(
        trace_request_id=label,
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
    for field in ("first_text_ms", "total_ms"):
        values = sorted(r[field] for r in rows)
        result[field] = dict(
            median=statistics.median(values),
            p95=values[math.ceil(0.95 * len(values)) - 1],
        )
    if duration_ms is not None:
        result.update(
            duration_ms=duration_ms,
            output_tokens_per_second=result["output_tokens"] / (duration_ms / 1000),
            requests_per_second=len(rows) / (duration_ms / 1000),
        )
    return result


def inspect_trace(path, rows, prefix):
    lookups, copies = {}, 0
    max_running = max_waiting = preemptions = evictions = invalid_refs = 0
    ids = {r["trace_request_id"]: r for r in rows}
    for line in path.open():
        event = json.loads(line)
        timed = any(
            prefix in r["id"]
            for r in event.get("running", []) + event.get("waiting", [])
        )
        rid = event.get("request", {}).get("id", "")
        timed = timed or prefix in rid
        if not timed:
            continue
        max_running = max(max_running, len(event.get("running", [])))
        max_waiting = max(max_waiting, len(event.get("waiting", [])))
        kind = event["event"]
        if kind == "PREEMPT_END":
            preemptions += 1
        if kind == "EVICT_CACHED_BLOCK":
            evictions += 1
            invalid_refs += event["before"]["refs"] != 0
        if kind == "SCHEDULE_END":
            copies += len(event.get("kv_block_copies", []))
        if kind == "PREFIX_LOOKUP_END":
            for key in ids:
                if key in rid:
                    assert key not in lookups
                    lookups[key] = event["hit_tokens"]
    assert len(lookups) == len(rows)
    assert all(lookups[r["trace_request_id"]] == r["cached_tokens"] for r in rows)
    assert not preemptions and not invalid_refs
    assert max_running >= 2
    expected_copies = sum(r["cached_tokens"] % 16 != 0 for r in rows)
    assert copies == expected_copies, (copies, expected_copies)
    return dict(
        max_running=max_running,
        max_waiting=max_waiting,
        preemptions=preemptions,
        cached_evictions=evictions,
        invalid_eviction_refs=invalid_refs,
        copies=copies,
        expected_copies=expected_copies,
        lookup_matches_api=True,
    )


def run(folder, manifest, variant, concurrency, mode, round_id):
    assert not port_open()
    name = f"c{concurrency}-{variant}-{mode}-r{round_id}"
    run_dir = folder / name
    run_dir.mkdir()
    g, minimum = VARIANTS[variant]
    env = os.environ.copy()
    env.update(
        STUDY_PARTIAL_PREFIX="0" if variant == "stock" else "1",
        STUDY_PARTIAL_MIN_TOKENS=str(minimum),
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
        "768",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "128",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "256",
        "--prefix-match-unit",
        str(g),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--enable-prompt-tokens-details",
    ]
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
            send(manifest["seeds"][0][:32], name + "-warmup", 1, time.perf_counter())
            assert post("reset_prefix_cache")["success"]
            seed_start = time.perf_counter()
            record["seeds"] = [
                send(tokens, f"{name}-seed-{i:02}", 1, seed_start)
                for i, tokens in enumerate(manifest["seeds"])
            ]
            record["seed_wall_ms"] = (time.perf_counter() - seed_start) * 1000
            origin = time.perf_counter()
            prefix = name + "-timed-"

            def task(item):
                result = send(
                    item["tokens"],
                    prefix + item["request_id"],
                    item["output_tokens"],
                    origin,
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
            assert len(rows) == 48
            record["summary"] = stats(rows, duration)
            record["groups"] = {
                cohort: stats([r for r in rows if r["cohort"] == cohort], None)
                for cohort in ("repeat", "partial", "unrelated")
            }
            record["length_groups"] = {
                str(n): stats([r for r in rows if r["prompt_length"] == n], None)
                for n in (64, 128, 256, 512)
            }
            intervals = [(r["started_ms"], 1) for r in rows] + [
                (r["finished_ms"], -1) for r in rows
            ]
            active = peak = 0
            for _, delta in sorted(intervals):
                active += delta
                peak = max(peak, active)
            record["client_peak_inflight"] = peak
            assert peak == concurrency
            assert record["summary"]["output_tokens"] == 1728
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
        if concurrency == 12:
            assert record["diagnostic"]["max_waiting"] >= 1
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
        ROOT / "results/concurrent-partial" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = dict(manifest=manifest, diagnostics=[], runs=[], complete=False)
    path = folder / "summary.json"
    try:
        for concurrency in (4, 12):
            for variant in VARIANTS:
                summary["diagnostics"].append(
                    run(folder, manifest, variant, concurrency, "trace", 0)
                )
                path.write_text(json.dumps(summary, indent=2) + "\n")
            variants = list(VARIANTS)
            for round_id in range(1, 4):
                order = variants[round_id - 1 :] + variants[: round_id - 1]
                for variant in order:
                    summary["runs"].append(
                        run(folder, manifest, variant, concurrency, "bench", round_id)
                    )
                    path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["complete"] = True
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", path, flush=True)


if __name__ == "__main__":
    main()
