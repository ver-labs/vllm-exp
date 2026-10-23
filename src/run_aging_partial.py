"""Experiment 14: factorial aging retention and exact partial-prefix reuse."""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from run_partial_threshold import post
from run_prefix_batches import ROOT, port_open, stop_server
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
VARIANTS = {"stock": (16, 0), "aging": (16, 0), "partial": (1, 0), "combined": (1, 0)}
REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}


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
    assert max_running >= 1
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
        STUDY_PARTIAL_PREFIX="1" if g == 1 else "0",
        STUDY_ADAPTIVE_PREFILL="0",
        STUDY_LFU_HALF_LIFE_BLOCKS="64",
        STUDY_PARTIAL_MIN_TOKENS=str(minimum),
        STUDY_CACHE_IMPL="incremental",
        STUDY_EVICTION_POLICY="aged_lfu"
        if variant in ("aging", "combined")
        else "vllm",
        STUDY_WAITING_POLICY="fcfs",
        STUDY_SKIP_CONTINUE="0",
        STUDY_CACHE_PROFILE="1" if mode == "trace" else "0",
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
        str(manifest["max_model_len"]),
        "--max-num-seqs",
        str(manifest["running_slots"]),
        "--max-num-batched-tokens",
        "128",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        str(manifest["physical_blocks"]),
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
        study_environment={
            k: v
            for k, v in env.items()
            if k.startswith("STUDY_") or k.startswith("VLLM_STUDY_")
        },
        manifest_sha256=hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
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
            assert len(rows) == len(manifest["requests"])
            record["summary"] = stats(rows, duration)
            record["groups"] = {
                cohort: stats([r for r in rows if r["cohort"] == cohort], None)
                for cohort in sorted({r["cohort"] for r in rows})
            }
            record["length_groups"] = {
                str(n): stats([r for r in rows if r["prompt_length"] == n], None)
                for n in sorted({r["prompt_length"] for r in rows})
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
            assert record["summary"]["output_tokens"] == sum(
                r["output_tokens"] for r in manifest["requests"]
            )
            with urlopen("http://127.0.0.1:8001/metrics", timeout=10) as response:
                (run_dir / "metrics-after.txt").write_bytes(response.read())
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
        validate_diagnostic(run_dir, record, manifest, variant)
        if concurrency > manifest["running_slots"]:
            assert record["diagnostic"]["max_waiting"] >= 1
        path.write_text(json.dumps(record, indent=2) + "\n")
    print("DONE", name, round(record["summary"]["duration_ms"], 1), "ms", flush=True)
    return str(path)


def validate_diagnostic(run_dir, record, manifest, variant):
    decays = schedules = 0
    ready = None
    for line in next(run_dir.glob("events-*.jsonl")).open():
        event = json.loads(line)
        kind = event["event"]
        if kind == "TRACE_READY":
            ready = event
        if kind == "CACHE_FREQUENCY_DECAY":
            decays += 1
            assert event["counts_after"] == [
                n >> event["elapsed_epochs"] for n in event["counts_before"]
            ]
        if kind == "SCHEDULE_END":
            schedules += 1
            # Verify the observer's per-request and total scheduled work.
            counts = event["assigned_positions"]
            if isinstance(counts, dict):
                assert sum(counts.values()) == event["assigned_total"]
                assert event["assigned_total"] <= manifest["token_budget"]
        if kind == "EVICT_CACHED_BLOCK":
            assert event["before"]["refs"] == 0
    assert ready and ready["waiting_policy"] == "fcfs"
    assert ready["eviction_policy"] == (
        "aged_lfu" if variant in ("aging", "combined") else "vllm"
    )
    assert ready["chunked_prefill"] and not ready["skip_continue"]
    assert "vllm-src/vllm/v1/core/sched/scheduler.py" in ready["scheduler_source"]
    if variant in ("aging", "combined"):
        assert decays > 0 and ready["aging_half_life_blocks"] == 64
    if variant in ("stock", "aging"):
        assert record["diagnostic"]["copies"] == 0
    else:
        assert record["diagnostic"]["copies"] > 0
    assert "MLX device set to: Device(gpu, 0)" in (run_dir / "server.log").read_text()
    record["diagnostic"].update(
        decays=decays, schedules=schedules, ready=ready, metal_gpu=True
    )


def shift_workload(tokenizer):
    sys.path.insert(0, str(ROOT / "workloads"))
    from compare_aging_lfu import workload as old_workload

    sequence, threshold = old_workload("shift")
    rows = []
    for index, item in enumerate(sequence):
        rendered = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Explain clearly and concisely."
                    ),
                },
                {"role": "user", "content": item["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = tokenizer.encode(rendered, add_special_tokens=False)
        assert 256 <= len(tokens) < 376
        rows.append(
            dict(
                request_id=f"T{index:02}",
                tokens=tokens,
                prompt_length=len(tokens),
                output_tokens=8,
                cohort=item["phase"],
                cycle=item["cycle"],
            )
        )
    return dict(
        model=MODEL,
        revision=REVISION,
        seeds=[rows[0]["tokens"][:32]],
        requests=rows,
        client_concurrency_levels=[1],
        running_slots=1,
        physical_blocks=32,
        block_size=16,
        max_model_len=384,
        token_budget=128,
        temperature=0,
        ignore_eos=True,
        full_probe_hit_tokens=threshold,
        arrival_model="sequential; original experiment 09 traffic shift",
    )


def aggregate(records):
    by = {
        v: [
            json.loads(Path(p).read_text())
            for p in records
            if json.loads(Path(p).read_text())["variant"] == v
        ]
        for v in VARIANTS
    }
    out = {}
    for variant, runs in by.items():
        out[variant] = dict(
            duration_ms=[r["summary"]["duration_ms"] for r in runs],
            median_duration_ms=statistics.median(
                r["summary"]["duration_ms"] for r in runs
            ),
            median_output_tokens_per_second=statistics.median(
                r["summary"]["output_tokens_per_second"] for r in runs
            ),
            median_first_text_ms=statistics.median(
                r["summary"]["first_text_ms"]["median"] for r in runs
            ),
            median_p95_request_ms=statistics.median(
                r["summary"]["total_ms"]["p95"] for r in runs
            ),
            cached_tokens=[r["summary"]["cached_tokens"] for r in runs],
            groups={
                g: {
                    metric: statistics.median(
                        r["groups"][g][metric]["median"] for r in runs
                    )
                    for metric in ("first_text_ms", "total_ms")
                }
                for g in runs[0]["groups"]
            },
        )
    paired = []
    for index in sorted({r["round"] for r in by["stock"]}):
        match = {
            v: next(r for r in rows if r["round"] == index) for v, rows in by.items()
        }
        base = match["stock"]["summary"]["duration_ms"]
        texts = {r["request_id"]: r["answer"] for r in match["stock"]["rows"]}
        paired.append(
            dict(
                round=index,
                duration_change_pct={
                    v: (r["summary"]["duration_ms"] / base - 1) * 100
                    for v, r in match.items()
                },
                exact_answer_matches={
                    v: sum(x["answer"] == texts[x["request_id"]] for x in r["rows"])
                    for v, r in match.items()
                },
            )
        )
    return dict(variants=out, paired=paired)


def main():
    global MODEL, REVISION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS, required=True)
    parser.add_argument("--profile", choices=("mixed", "shift"), required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    assert args.rounds > 0
    MODEL, REVISION = args.model, REVISIONS[args.model]
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        MODEL, revision=REVISION, local_files_only=True
    )
    os.environ["STUDY_MODEL"] = MODEL
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["STUDY_MODEL_PATH"], local_files_only=True
    )
    manifest = (
        workload(tokenizer) if args.profile == "mixed" else shift_workload(tokenizer)
    )
    manifest.update(profile=args.profile, revision=REVISION, model=MODEL)
    concurrency = 12 if args.profile == "mixed" else 1
    manifest["client_concurrency_levels"] = [concurrency]
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    folder = (
        ROOT
        / "results/aging-partial"
        / MODEL.replace("/", "--")
        / args.profile
        / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = dict(
        experiment="14-aging-partial",
        manifest=manifest,
        diagnostics=[],
        runs=[],
        complete=False,
        source_commit=subprocess.check_output(
            ["git", "-C", str(ROOT / "vllm-src"), "rev-parse", "HEAD"], text=True
        ).strip(),
        rounds=args.rounds,
        half_life_blocks=64,
        partial_min_tokens=0,
        limitations=[
            "Three timing rounds are exploratory.",
            "Mixed traffic uses completion-dependent closed-loop arrivals.",
            (
                "Seed publication and startup are excluded; "
                "shift uses sequential arrivals."
            ),
            "Exact text agreement is not a semantic quality evaluation.",
        ],
    )
    path = folder / "summary.json"
    try:
        for variant in VARIANTS:
            summary["diagnostics"].append(
                run(folder, manifest, variant, concurrency, "trace", 0)
            )
            path.write_text(json.dumps(summary, indent=2) + "\n")
        variants = list(VARIANTS)
        for index in range(args.rounds):
            order = variants[index:] + variants[:index]
            for variant in order:
                summary["runs"].append(
                    run(folder, manifest, variant, concurrency, "bench", index + 1)
                )
                summary["aggregate_partial"] = len(summary["runs"])
                path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["aggregate"] = aggregate(summary["runs"])
        summary["complete"] = True
    except Exception as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", path, flush=True)


if __name__ == "__main__":
    main()
