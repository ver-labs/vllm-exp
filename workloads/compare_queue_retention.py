"""Queued demand versus historical popularity under a fixed KV-page limit."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event

from compare_prefix_policies import MODEL, SYSTEM, inspect_trace, send, summarize
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def build_workload():
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("STUDY_MODEL_PATH")
        or snapshot_download(MODEL, local_files_only=True),
        local_files_only=True,
    )

    def tokens(prompt):
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tokenizer.encode(rendered, add_special_tokens=False)

    def padded(prompt, target):
        for _ in range(512):
            if len(tokens(prompt)) >= target:
                break
            prompt += " detail"
        if len(tokens(prompt)) != target:
            raise ValueError("Could not create the exact target prompt length")
        return prompt

    old = padded(
        "Historical handbook: A mountain weather team records wind and rainfall. "
        "Explain how to organize observations and check measurements.",
        192,
    )
    current = padded(
        "Current handbook: A library manages reservations and collection notices. "
        "Explain how staff can record requests and improve service.",
        192,
    )
    blocker = padded("List household objects and explain their uses.", 48)
    prompts, groups = {}, {"cold": [], "returning": []}
    for index, label in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXY"):
        if index % 2 == 0:
            prompt = padded(
                f"Independent case {label}: A community project needs a plan. "
                "Explain how to compare costs, observations and practical changes.",
                256,
            )
            group = "cold"
        else:
            prompt = current + f"\nQuestion {label}: Suggest one practical improvement."
            group = "returning"
        prompts[label] = prompt
        groups[group].append(label)
    common = []
    seed_ids = tokens(current)
    for label in groups["returning"]:
        ids = tokens(prompts[label])
        count = 0
        for left, right in zip(seed_ids, ids):
            if left != right:
                break
            count += 1
        common.append(count // 16 * 16)
    return dict(
        workload="queue-retention-v1",
        model=MODEL,
        system=SYSTEM,
        historical_prompt=old,
        current_prompt=current,
        blocker=blocker,
        historical_repeats=3,
        training_output_tokens=8,
        blocker_output_tokens=128,
        candidate_output_tokens=32,
        candidate_interval_ms=10,
        maximum_client_concurrency=26,
        candidates=prompts,
        candidate_order=list(prompts),
        prompt_groups=groups,
        candidate_prompt_lengths={k: len(tokens(v)) for k, v in prompts.items()},
        shared_current_prefix_tokens=common,
        ignore_eos=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fcfs",), required=True)
    parser.add_argument(
        "--eviction-policy", choices=("vllm", "lfu", "queue"), required=True
    )
    parser.add_argument("--candidates", type=int, choices=(25,), default=25)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    manifest = build_workload()
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    folder = Path("results/queue-retention/targeted") / args.eviction_policy
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"controlled-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    run_id = f"queue-study-{time.time_ns()}"
    warmup = send(
        args.url, run_id, "WARMUP", "Name a household object.", 1, time.perf_counter()
    )
    if warmup["error"]:
        raise RuntimeError(warmup["error"])
    origin = time.perf_counter()
    rows = []
    for index in range(3):
        rows.append(
            send(
                args.url,
                run_id,
                f"OLD{index}",
                manifest["historical_prompt"],
                8,
                origin,
            )
        )
    rows.append(
        send(args.url, run_id, "CURRENT", manifest["current_prompt"], 8, origin)
    )
    if any(r["error"] for r in rows):
        raise RuntimeError("Cache priming failed")
    first = Event()
    setup_error = None
    release_ms = None
    overlap = False
    with ThreadPoolExecutor(max_workers=26) as executor:
        holder = executor.submit(
            send, args.url, run_id, "H1", manifest["blocker"], 128, origin, first
        )
        jobs = [holder]
        if first.wait(60) and not holder.done():
            release = time.perf_counter()
            release_ms = (release - origin) * 1000
            for index, (label, prompt) in enumerate(manifest["candidates"].items()):
                delay = release + index * 0.010 - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                jobs.append(
                    executor.submit(send, args.url, run_id, label, prompt, 32, origin)
                )
            overlap = not holder.done()
        else:
            setup_error = "Blocker did not hold the running slot"
        rows.extend(job.result() for job in as_completed(jobs))
    duration_ms = (time.perf_counter() - origin) * 1000
    candidates = [r for r in rows if r["request_id"] in manifest["candidates"]]
    for row in rows:
        row["cached_prompt_tokens"] = (
            row["usage"].get("prompt_tokens_details") or {}
        ).get("cached_tokens")
    trace = (
        inspect_trace(args.trace, run_id, manifest["candidates"])
        if args.trace
        else None
    )
    valid = (
        overlap
        and len(rows) == 30
        and len(candidates) == 25
        and all(
            r["error"] is None and isinstance(r["cached_prompt_tokens"], int)
            for r in rows
        )
        and all(
            r["usage"]["prompt_tokens"]
            == manifest["candidate_prompt_lengths"][r["request_id"]]
            for r in candidates
        )
        and (
            trace is None
            or (
                trace["eviction_policy"] == args.eviction_policy
                and trace["server_policy"] == "fcfs"
                and trace["max_candidates_waiting"] == 25
                and trace["candidate_admission_order"] == manifest["candidate_order"]
                and trace["preemptions"] == 0
            )
        )
    )
    span = (
        max(r["finished_ms"] for r in candidates) - release_ms if candidates else None
    )
    result = dict(
        manifest=manifest,
        run_id=run_id,
        valid_run=valid,
        eviction_policy=args.eviction_policy,
        warmup=warmup,
        blockers_unfinished_after_all_submissions=overlap,
        setup_error=setup_error,
        candidate_release_ms=release_ms,
        candidate_cached_prompt_tokens=sum(
            r["cached_prompt_tokens"] or 0 for r in candidates
        ),
        candidate_summary=summarize(candidates, span) if candidates else None,
        batch_summary=summarize(rows, duration_ms),
        trace_analysis=trace,
        requests=sorted(rows, key=lambda r: r["request_id"]),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid_run": valid,
                "candidate_summary": result["candidate_summary"],
                "trace_analysis": trace,
            },
            indent=2,
        )
    )
    print(f"Saved: {output.resolve()}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
