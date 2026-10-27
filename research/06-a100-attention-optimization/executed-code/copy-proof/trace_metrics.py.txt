"""Attribute real CUDA events through launch correlations, excluding annotations."""

import bisect
import gzip
import json
from collections import Counter, defaultdict


def union_us(intervals):
    end, total = float("-inf"), 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze_trace(path):
    with gzip.open(path, "rt") as stream:
        events = json.load(stream)["traceEvents"]
    scopes = defaultdict(list)
    launches = {}
    gpu_events = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "user_annotation" and e["name"].startswith("study/"):
            scopes[(e["pid"], e["tid"], e["name"])].append(e)
        elif cat in ("cuda_runtime", "cuda_driver"):
            correlation = e.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = e
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_events.append(e)
    assert gpu_events
    indexed = {}
    for key, values in scopes.items():
        values.sort(key=lambda e: e["ts"])
        indexed[key] = ([e["ts"] for e in values], values)
    attributed = defaultdict(list)
    per_call = defaultdict(lambda: defaultdict(list))
    kernels = defaultdict(lambda: [0, 0.0])
    matched = 0
    for e in gpu_events:
        if e["cat"] == "kernel":
            kernels[e["name"]][0] += 1
            kernels[e["name"]][1] += e["dur"]
        launch = launches.get(e.get("args", {}).get("correlation"))
        if launch is None:
            continue
        matched += 1
        for (pid, tid, name), (starts, values) in indexed.items():
            if (pid, tid) != (launch["pid"], launch["tid"]):
                continue
            i = bisect.bisect_right(starts, launch["ts"]) - 1
            if i < 0 or launch["ts"] > values[i]["ts"] + values[i]["dur"]:
                continue
            attributed[name].append(e)
            per_call[name][i].append(e)
    assert matched == len(gpu_events), (matched, len(gpu_events))
    result = {}
    for (_, _, name), (_, values) in indexed.items():
        actual = attributed[name]
        calls = []
        for i, scope in enumerate(values):
            selected = per_call[name][i]
            calls.append(
                {
                    "host_range_ms": scope["dur"] / 1000,
                    "gpu_event_count": len(selected),
                    "gpu_event_sum_ms": sum(e["dur"] for e in selected) / 1000,
                    "gpu_busy_union_ms": union_us(
                        [(e["ts"], e["ts"] + e["dur"]) for e in selected]
                    )
                    / 1000,
                }
            )
            if i == 0 and name == "study/model.forward":
                first_kernels = defaultdict(lambda: [0, 0.0])
                for e in selected:
                    if e["cat"] == "kernel":
                        first_kernels[e["name"]][0] += 1
                        first_kernels[e["name"]][1] += e["dur"] / 1000
                calls[-1]["kernels"] = {
                    k: {"calls": v[0], "total_ms": v[1]}
                    for k, v in first_kernels.items()
                }
                calls[-1]["cpu_ops"] = dict(
                    Counter(
                        e["name"]
                        for e in events
                        if e.get("cat") == "cpu_op"
                        and scope["ts"] <= e["ts"] <= scope["ts"] + scope["dur"]
                    )
                )
        result[name] = {
            "calls": len(values),
            "host_range_total_ms": sum(e["dur"] for e in values) / 1000,
            "gpu_event_count": len(actual),
            "gpu_event_sum_ms": sum(e["dur"] for e in actual) / 1000,
            "gpu_busy_union_ms": union_us(
                [(e["ts"], e["ts"] + e["dur"]) for e in actual]
            )
            / 1000,
            "per_call": calls,
        }
    return {
        "method": (
            "Real kernel/memcpy/memset events mapped to CPU scopes by launch "
            "correlation and thread/time containment; GPU annotations excluded."
        ),
        "actual_gpu_events": len(gpu_events),
        "matched_launches": matched,
        "all_gpu_busy_union_ms": union_us(
            [(e["ts"], e["ts"] + e["dur"]) for e in gpu_events]
        )
        / 1000,
        "scopes": result,
        "top_kernels": sorted(
            [
                {"name": k, "calls": v[0], "total_ms": v[1] / 1000}
                for k, v in kernels.items()
            ],
            key=lambda x: x["total_ms"],
            reverse=True,
        )[:20],
    }
