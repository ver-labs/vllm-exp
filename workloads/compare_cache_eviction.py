"""Temporal prefix-reuse experiment with FCFS scheduling and fixed output lengths."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from compare_prefix_policies import DOCUMENT, MODEL, SYSTEM, send, summarize

HOT_DOCUMENT = DOCUMENT + (
    "The manager plans a small pilot before changing the entire service. "
    "One volunteer will check reminder messages and another will record whether "
    "members collect books on time. Every week the team will compare the number "
    "of notices sent, cancelled reservations, and books returned to the shelf. "
    "They will ask members whether the messages are clear and whether collection "
    "hours are convenient. Changes must preserve the original order of the "
    "waiting list and provide the same service to telephone and online members. "
)
TOPICS = [
    "Coral reef monitoring",
    "Mountain weather records",
    "Composting food scraps",
    "Bicycle gear maintenance",
    "Urban bus reliability",
    "Solar battery storage",
    "Earthquake preparedness",
    "Honeybee pollination",
    "Museum audio guides",
    "River water filtration",
    "School meal planning",
    "Bird migration tracking",
    "Public park accessibility",
    "Household recycling",
    "Stellar cloud observations",
    "Community gardening",
    "Apartment insulation",
    "Coastal tide measurements",
    "Hospital appointment reminders",
    "Rainwater collection",
    "Ocean current mapping",
    "Railway timetable planning",
    "Forest fire monitoring",
    "Wind turbine inspections",
    "Local theatre ticketing",
    "Sports facility maintenance",
    "Footpath repairs",
    "Historic building restoration",
    "Air quality sampling",
    "Mountain trail safety",
]
OUTPUT_TOKENS = 8


def build_sequence():
    rows = []
    topic_index = 0

    def cold(label, phase, cycle):
        nonlocal topic_index
        topic = TOPICS[topic_index]
        topic_index += 1
        prompt = topic + (
            ": A small community team wants to improve its work over the next "
            "three months. It has a limited budget and several volunteers. "
            "The team can collect observations every week, interview residents, "
            "and compare records from different locations. The locations differ "
            "in size, activity, weather exposure, and access to transport. "
            "Some records are incomplete and several volunteers are new. "
            "The team cannot hire more staff or buy expensive equipment. "
            "It wants practical changes that can be checked with simple measurements. "
            "Suggest a useful first step and explain what observation would "
            "show whether that step is helping. Keep the answer brief."
        )
        rows.append({"label": label, "phase": phase, "cycle": cycle, "prompt": prompt})

    for index in range(12):
        cold(f"N{index:02}", "no_reuse_control", 0)
    for cycle in range(1, 4):
        for index in range(3):
            rows.append(
                {
                    "label": f"C{cycle}-T{index}",
                    "phase": "train_hot",
                    "cycle": cycle,
                    "prompt": HOT_DOCUMENT
                    + f"\nGive a brief summary. Note {cycle}.{index}.",
                }
            )
        for index in range(6):
            cold(f"C{cycle}-S{index}", "cold_scan", cycle)
        rows.append(
            {
                "label": f"C{cycle}-P",
                "phase": "hot_probe",
                "cycle": cycle,
                "prompt": HOT_DOCUMENT
                + f"\nSuggest one useful improvement. Note {cycle}.",
            }
        )
    return rows


def analyze_trace(path, results):
    report = {
        "eviction_policy": None,
        "waiting_policy": None,
        "evicted_cached_blocks": 0,
        "evicted_reuse_counts": [],
        "preemptions": 0,
        "probe_hit_tokens": {},
    }
    probes = {
        row["trace_request_id"]: row["request_id"]
        for row in results
        if row["phase"] == "hot_probe"
    }
    with Path(path).open() as source:
        for line in source:
            event = json.loads(line)
            if event["event"] == "TRACE_READY":
                report["eviction_policy"] = event.get("eviction_policy")
                report["waiting_policy"] = event.get("waiting_policy")
            if event["event"] == "EVICT_CACHED_BLOCK":
                report["evicted_cached_blocks"] += 1
                report["evicted_reuse_counts"].append(
                    event["before"].get("reuse_count")
                )
            if event["event"] == "PREEMPT_END":
                report["preemptions"] += 1
            if event["event"] == "PREFIX_LOOKUP_END":
                rid = event.get("request", {}).get("id", "")
                for key, label in probes.items():
                    if rid == key or key + "-" in rid:
                        report["probe_hit_tokens"][label] = event["hit_tokens"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("vllm", "lfu"),
        required=True,
        help="Label only; real eviction policy is set by the server",
    )
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--trace-latest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    sequence = build_sequence()
    manifest = {
        "workload": "cache-eviction-v1",
        "model": MODEL,
        "system": SYSTEM,
        "output_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "client_concurrency": 1,
        "sequence": sequence,
    }
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    folder = Path("results/cache-eviction") / args.policy
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / f"temporal-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    trace_path = None
    if args.trace_latest:
        paths = list(folder.glob("server-trace-*/events-*.jsonl"))
        if not paths:
            raise SystemExit("No diagnostic trace found")
        trace_path = max(paths, key=lambda path: path.stat().st_mtime_ns)
    run_id = f"eviction-study-{time.time_ns()}"
    warmup = send(
        args.url,
        run_id,
        "WARMUP",
        "Name a common household object.",
        OUTPUT_TOKENS,
        time.perf_counter(),
    )
    if warmup["error"]:
        raise SystemExit(f"Warm-up failed: {warmup['error']}")
    origin = time.perf_counter()
    results = []
    for item in sequence:
        row = send(
            args.url, run_id, item["label"], item["prompt"], OUTPUT_TOKENS, origin
        )
        row.update({key: item[key] for key in ["phase", "cycle"]})
        row["cached_prompt_tokens"] = (
            row["usage"].get("prompt_tokens_details") or {}
        ).get("cached_tokens")
        results.append(row)
        if row["error"]:
            break
    duration = (time.perf_counter() - origin) * 1000
    phases = {}
    for phase in ["no_reuse_control", "train_hot", "cold_scan", "hot_probe"]:
        rows = [row for row in results if row["phase"] == phase]
        if rows:
            phases[phase] = summarize(rows, sum(row["total_ms"] for row in rows))
            phases[phase]["cached_prompt_tokens"] = sum(
                row["cached_prompt_tokens"] or 0 for row in rows
            )
    trace = analyze_trace(trace_path, results) if trace_path else None
    valid = (
        len(results) == len(sequence)
        and all(row["error"] is None for row in results)
        and all(isinstance(row["cached_prompt_tokens"], int) for row in results)
        and (
            trace is None
            or (
                trace["eviction_policy"] == args.policy
                and trace["waiting_policy"] == "fcfs"
            )
        )
    )
    result = {
        "manifest": manifest,
        "policy_label": args.policy,
        "run_id": run_id,
        "valid_run": valid,
        "warmup": warmup,
        "summary": summarize(results, duration),
        "phases": phases,
        "trace_analysis": trace,
        "requests": results,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"valid_run": valid, "phases": phases, "trace_analysis": trace}, indent=2
        )
    )
    print(f"Saved: {output.resolve()}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
