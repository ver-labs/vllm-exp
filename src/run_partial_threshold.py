"""Matched partial-hit thresholds across actual token divergence offsets."""

import json
import os
import random
import subprocess
import time
from datetime import datetime
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from run_prefix_batches import ROOT, port_open, stop_server
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
CONFIGS = [(16, 16)] + [(g, m) for g in (1, 4, 8) for m in (0, 8, 16)]


def post(endpoint, payload=None):
    with urlopen(
        Request(
            "http://127.0.0.1:8001/" + endpoint,
            data=b"" if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=120,
    ) as response:
        return json.load(response)


def send(tokens, label):
    payload = dict(
        model=MODEL,
        prompt=tokens,
        max_tokens=1,
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
    started = time.perf_counter_ns()
    first = None
    text = ""
    usage = {}
    finish = None
    done = False
    with urlopen(request, timeout=120) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("text") or ""
                if content and first is None:
                    first = time.perf_counter_ns()
                text += content
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter_ns()
    assert done and finish and usage["completion_tokens"] == 1
    return dict(
        label=label,
        text=text,
        usage=usage,
        cached_tokens=usage["prompt_tokens_details"]["cached_tokens"],
        first_text_ms=None if first is None else (first - started) / 1e6,
        total_ms=(end - started) / 1e6,
    )


def run(folder, source, targets, g, minimum, mode, round_id):
    assert not port_open()
    name = f"g{g}-m{minimum}-{mode}-r{round_id}"
    run_dir = folder / name
    run_dir.mkdir()
    env = os.environ.copy()
    env.update(
        STUDY_PARTIAL_MIN_TOKENS=str(minimum),
        STUDY_PARTIAL_PREFIX="0" if g == 16 else "1",
        STUDY_CACHE_IMPL="incremental",
        STUDY_EVICTION_POLICY="vllm",
        STUDY_WAITING_POLICY="fcfs",
        STUDY_SKIP_CONTINUE="0",
        STUDY_CACHE_PROFILE="0",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_SERVER_DEV_MODE="1",
        VLLM_STUDY_TRACE="1" if mode == "trace" else "0",
        VLLM_STUDY_TRACE_DIR=str(run_dir),
        VLLM_STUDY_TRACE_STDOUT="0",
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
        "256",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "128",
        "--block-size",
        "16",
        "--num-gpu-blocks-override",
        "96",
        "--prefix-match-unit",
        str(g),
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--no-async-scheduling",
        "--enable-prompt-tokens-details",
    ]
    record = dict(
        config=f"g{g}-m{minimum}",
        granularity=g,
        minimum=minimum,
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
                    raise TimeoutError("Server startup")
                time.sleep(0.5)
            send(source[:32], f"{name}-warmup")
            cases = [(rep, offset) for rep in range(3) for offset in range(1, 16)]
            random.Random(302 + round_id).shuffle(cases)
            started = time.perf_counter_ns()
            for rep, offset in cases:
                assert post("reset_prefix_cache")["success"]
                a = send(source, f"{name}-d{offset}-n{rep}-A")
                b = send(targets[str(offset)], f"{name}-d{offset}-n{rep}-B")
                extra = offset // g * g
                expected = 112 + (extra if extra >= minimum else 0)
                assert a["cached_tokens"] == 0
                assert b["cached_tokens"] == expected, (
                    name,
                    offset,
                    b["cached_tokens"],
                    expected,
                )
                assert b["first_text_ms"] is not None
                record["rows"].append(
                    dict(
                        offset=offset,
                        repeat=rep,
                        extra_tokens=expected - 112,
                        source=a,
                        target=b,
                        pair_ms=a["total_ms"] + b["total_ms"],
                    )
                )
            record["wall_ms_including_resets"] = (
                time.perf_counter_ns() - started
            ) / 1e6
            record["complete"] = True
        finally:
            stop_server(server)
            path.write_text(json.dumps(record, indent=2) + "\n")
            time.sleep(3)
    if mode == "trace":
        events = [
            json.loads(line) for line in next(run_dir.glob("events-*.jsonl")).open()
        ]
        copies = sum(
            len(e.get("kv_block_copies", []))
            for e in events
            if e["event"] == "SCHEDULE_END"
        )
        expected = sum(r["extra_tokens"] > 0 for r in record["rows"])
        assert copies == expected, (name, copies, expected)
        assert not any(e["event"] == "PREEMPT_END" for e in events)
        record["diagnostic"] = dict(
            copies=copies, expected_copies=expected, preemptions=0
        )
        for row in record["rows"]:
            label = row["target"]["label"]
            lookups = [
                e
                for e in events
                if e["event"] == "PREFIX_LOOKUP_END"
                and label in e.get("request", {}).get("id", "")
            ]
            assert (
                len(lookups) == 1
                and lookups[0]["hit_tokens"] == row["target"]["cached_tokens"]
            )
        path.write_text(json.dumps(record, indent=2) + "\n")
    print("DONE", name, flush=True)
    return str(path)


def main():
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        MODEL, revision=REVISION, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["STUDY_MODEL_PATH"], local_files_only=True
    )
    source = tokenizer.encode(
        "The library records book reservations and reminds members to collect them. "
        * 30,
        add_special_tokens=False,
    )[:128]
    alternative = tokenizer.encode(" volcano", add_special_tokens=False)[-1]
    targets = {}
    for offset in range(1, 16):
        target = source.copy()
        assert alternative != target[112 + offset]
        target[112 + offset] = alternative
        targets[str(offset)] = target
    folder = (
        ROOT / "results/partial-threshold" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    )
    folder.mkdir(parents=True)
    summary = dict(
        model=MODEL,
        revision=REVISION,
        source=source,
        targets=targets,
        configs=CONFIGS,
        diagnostics=[],
        runs=[],
        complete=False,
    )
    path = folder / "summary.json"
    try:
        for g, minimum in CONFIGS:
            summary["diagnostics"].append(
                run(folder, source, targets, g, minimum, "trace", 0)
            )
            path.write_text(json.dumps(summary, indent=2) + "\n")
        for round_id in range(1, 4):
            order = CONFIGS[round_id - 1 :] + CONFIGS[: round_id - 1]
            for g, minimum in order:
                summary["runs"].append(
                    run(folder, source, targets, g, minimum, "bench", round_id)
                )
                path.write_text(json.dumps(summary, indent=2) + "\n")
        summary["complete"] = True
    finally:
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY", path, flush=True)


if __name__ == "__main__":
    main()
