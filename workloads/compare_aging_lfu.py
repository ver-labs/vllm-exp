"""Measure response to a changing hot document, with a stable-hot control."""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from compare_cache_eviction import build_sequence
from compare_prefix_policies import MODEL, SYSTEM, send, summarize
from transformers import AutoTokenizer


def workload(profile):
    if profile == "stable":
        rows = build_sequence()
        for row in rows:
            row["phase"] = "probe" if row["phase"] == "hot_probe" else row["phase"]
        return rows, 272
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["STUDY_MODEL_PATH"], local_files_only=True
    )

    def padded(text):
        for _ in range(512):
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            count = len(tokenizer.encode(rendered, add_special_tokens=False))
            if count >= 256:
                break
            text += " detail"
        if count != 256:
            raise ValueError("Expected exactly256 prompt tokens")
        return text

    a = padded(
        "Document A: A mountain weather team records rainfall and wind. "
        "Explain how it organizes observations."
    )
    b = padded(
        "Document B: A library team records reservations and sends collection notices. "
        "Explain how it organizes its service."
    )
    rows = []
    for index in range(12):
        rows.append(
            dict(
                label=f"A{index:02}",
                phase="train_old",
                cycle=0,
                prompt=a + f"\nSummarize note {index}.",
            )
        )
    for cycle in range(1, 9):
        for index in range(4):
            rows.append(
                dict(
                    label=f"B{cycle}-T{index}",
                    phase="train_new",
                    cycle=cycle,
                    prompt=b + f"\nSummarize note {cycle}.{index}.",
                )
            )
        for index in range(4):
            text = padded(
                f"Independent scan {cycle}.{index}: A community project needs "
                "observations, a plan and practical measurements. Explain its first steps."
            )
            rows.append(
                dict(
                    label=f"S{cycle}-{index}",
                    phase="cold_scan",
                    cycle=cycle,
                    prompt=text,
                )
            )
        rows.append(
            dict(
                label=f"B{cycle}-P",
                phase="probe",
                cycle=cycle,
                prompt=b + f"\nSuggest an improvement. Note {cycle}.",
            )
        )
    return rows, 240


def inspect(path, rows):
    report = dict(
        eviction_policy=None,
        half_life_blocks=None,
        decays=0,
        epochs=[],
        preemptions=0,
        evictions=0,
        invalid_eviction_refs=0,
        lookup_hits={},
    )
    ids = {r["trace_request_id"]: r["request_id"] for r in rows}
    for line in path.open():
        event = json.loads(line)
        name = event["event"]
        if name == "TRACE_READY":
            report["eviction_policy"] = event["eviction_policy"]
            report["half_life_blocks"] = event.get("aging_half_life_blocks")
        if name == "CACHE_FREQUENCY_DECAY":
            report["decays"] += 1
            report["epochs"].append(event["epoch"])
            assert event["counts_after"] == [
                v >> event["elapsed_epochs"] for v in event["counts_before"]
            ]
        if name == "PREEMPT_END":
            report["preemptions"] += 1
        if name == "EVICT_CACHED_BLOCK":
            report["evictions"] += 1
            report["invalid_eviction_refs"] += event["before"]["refs"] != 0
        if name == "PREFIX_LOOKUP_END":
            rid = event.get("request", {}).get("id", "")
            for key, label in ids.items():
                if rid == key or key + "-" in rid:
                    report["lookup_hits"][label] = event["hit_tokens"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("fcfs",), required=True)
    parser.add_argument(
        "--eviction-policy", choices=("vllm", "lfu", "aged_lfu"), required=True
    )
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    profile = os.environ["STUDY_AGING_PROFILE"]
    sequence, full_hit = workload(profile)
    manifest = dict(
        workload="aging-lfu-v1",
        profile=profile,
        model=MODEL,
        system=SYSTEM,
        output_tokens=8,
        client_concurrency=1,
        full_probe_hit_tokens=full_hit,
        sequence=sequence,
        ignore_eos=True,
    )
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    folder = Path("results/aging-lfu") / profile / args.eviction_policy
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"temporal-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    url = "http://127.0.0.1:8001/v1/chat/completions"
    run_id = f"aging-study-{time.time_ns()}"
    warmup = send(
        url, run_id, "WARMUP", "Name a household object.", 1, time.perf_counter()
    )
    if warmup["error"]:
        raise RuntimeError(warmup["error"])
    origin = time.perf_counter()
    rows = []
    for item in sequence:
        row = send(url, run_id, item["label"], item["prompt"], 8, origin)
        row.update({k: item[k] for k in ("phase", "cycle")})
        row["cached_prompt_tokens"] = (
            row["usage"].get("prompt_tokens_details") or {}
        ).get("cached_tokens")
        rows.append(row)
        if row["error"]:
            break
    duration = (time.perf_counter() - origin) * 1000
    phases = {}
    for phase in sorted({r["phase"] for r in rows}):
        part = [r for r in rows if r["phase"] == phase]
        phases[phase] = summarize(part, sum(r["total_ms"] for r in part))
        phases[phase]["cached_prompt_tokens"] = sum(
            r["cached_prompt_tokens"] or 0 for r in part
        )
    probes = [r for r in rows if r["phase"] == "probe"]
    hits = [r["cached_prompt_tokens"] for r in probes]
    adaptation = next(
        (
            probes[i]["cycle"]
            for i in range(len(probes) - 1)
            if hits[i] >= full_hit and hits[i + 1] >= full_hit
        ),
        None,
    )
    trace = inspect(args.trace, rows) if args.trace else None
    valid = len(rows) == len(sequence) and all(
        r["error"] is None and isinstance(r["cached_prompt_tokens"], int) for r in rows
    )
    if trace:
        valid = (
            valid
            and trace["eviction_policy"] == args.eviction_policy
            and not trace["preemptions"]
            and not trace["invalid_eviction_refs"]
        )
        valid = valid and all(
            trace["lookup_hits"].get(r["request_id"]) == r["cached_prompt_tokens"]
            for r in rows
        )
        if args.eviction_policy == "aged_lfu":
            valid = valid and trace["half_life_blocks"] == 64 and trace["decays"] > 0
    summary = summarize(rows, duration)
    result = dict(
        manifest=manifest,
        run_id=run_id,
        valid_run=valid,
        warmup=warmup,
        candidate_summary=summary,
        batch_summary=summary,
        phases=phases,
        probe_hits=hits,
        first_sustained_full_hit_cycle=adaptation,
        trace_analysis=trace,
        requests=rows,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "valid_run": valid,
                "probe_hits": hits,
                "adaptation_cycle": adaptation,
                "phases": phases,
            },
            indent=2,
        )
    )
    print(f"Saved: {output.resolve()}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
