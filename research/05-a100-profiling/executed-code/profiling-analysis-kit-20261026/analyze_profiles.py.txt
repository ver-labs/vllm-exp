"""Describe measured host spans and device attribution without speed claims."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from trace_metrics import analyze_trace

root = Path(sys.argv[1])
records = json.loads((root / "execution.json").read_text())
assert len(records) == 6 and all(r["returncode"] == 0 for r in records)
assert json.loads((root / "validation.json").read_text())["status"] == "passed"
rows, summary = [], {}
for model in ("Qwen3-0.6B", "Qwen3-1.7B"):
    summary[model] = {}
    for arm in ("stock", "off", "partial"):
        folder = root / model / arm
        report = json.loads((folder / "result.json").read_text())
        summary[model][arm] = {}
        for profile in ("partial_reuse", "popularity_shift", "unique_control"):
            cpu = json.loads((folder / f"cpu-{profile}.json").read_text())
            trace = json.loads((folder / f"trace-{profile}.json").read_text())
            actual = analyze_trace(folder / f"trace-{profile}.trace.json.gz")
            (folder / f"kernels-{profile}.json").write_text(
                json.dumps(actual, indent=2) + "\n"
            )
            control = report["profiles"]["control/" + profile]
            probed = report["profiles"]["cpu/" + profile]
            n = len(probed["requests"])
            shape_counts = Counter(tuple(b["query_lengths"]) for b in cpu["batches"])
            detail = {
                "control_ttft_p50_ms": control["ttft_ms"]["p50"],
                "probed_ttft_p50_ms": probed["ttft_ms"]["p50"],
                "control_elapsed_s": control["elapsed_s"],
                "probed_elapsed_s": probed["elapsed_s"],
                "cpu_probe_elapsed_change_pct": 100
                * (probed["elapsed_s"] / control["elapsed_s"] - 1),
                "scheduled_copy_pairs": sum(b["copy_pairs"] for b in cpu["batches"]),
                "scheduled_tokens": sum(b["total_tokens"] for b in cpu["batches"]),
                "batch_shapes": {str(k): v for k, v in shape_counts.items()},
                "cpu_wall_ms_per_request": {
                    name: {
                        "inclusive": v["inclusive_ms"] / n,
                        "exclusive": v["exclusive_ms"] / n,
                        "calls_per_request": v["calls"] / n,
                        "mean_call_ms": v["inclusive_ms"] / v["calls"],
                    }
                    for name, v in cpu["cpu_wall"].items()
                },
                "trace_scope_ms_per_request": {
                    e["name"]: {
                        "host_inclusive": e["cpu_total_us"] / 3000,
                        "device_inclusive": actual["scopes"][e["name"]][
                            "gpu_event_sum_ms"
                        ]
                        / 3,
                        "calls_per_request": e["calls"] / 3,
                        "mean_host_call_ms": e["cpu_total_us"] / e["calls"] / 1000,
                        "mean_device_call_ms": actual["scopes"][e["name"]][
                            "gpu_event_sum_ms"
                        ]
                        / e["calls"],
                    }
                    for e in trace["operators"]
                    if e["name"].startswith("study/") and e["cpu_total_us"] > 0
                },
                "trace_actual_gpu_events": actual["actual_gpu_events"],
                "trace_matched_launches": actual["matched_launches"],
                "trace_gpu_busy_union_ms": actual["all_gpu_busy_union_ms"],
                "trace_first_model_forward": actual["scopes"]["study/model.forward"][
                    "per_call"
                ][0],
                "trace_top_kernels": actual["top_kernels"],
            }
            summary[model][arm][profile] = detail
            for name, v in detail["cpu_wall_ms_per_request"].items():
                rows.append(
                    {"model": model, "arm": arm, "profile": profile, "scope": name, **v}
                )
(root / "profile-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "cpu-breakdown.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "A100 diagnostic profiling — 26 October 2026",
    "==========================================",
    "",
    "One diagnostic engine per model/arm, not a repeated performance benchmark.",
    "CPU totals below are host wall time across a whole request, divided by 96.",
    "Inclusive spans overlap; do not sum them. They include waits and probe cost.",
    "GPU trace scope totals overlap too and cover only the first three requests.",
    "The prior phase04 paired benchmark remains the performance evidence.",
    "",
]
for model, arms in summary.items():
    for profile in ("partial_reuse", "popularity_shift", "unique_control"):
        lines += ["", model + " / " + profile]
        for arm, profiles in arms.items():
            d = profiles[profile]
            lines.append(
                f"{arm}: control/probed TTFT {d['control_ttft_p50_ms']:.3f}/"
                f"{d['probed_ttft_p50_ms']:.3f} ms; probe elapsed change "
                f"{d['cpu_probe_elapsed_change_pct']:+.2f}%; "
                f"copy pairs {d['scheduled_copy_pairs']}; "
                f"scheduled tokens {d['scheduled_tokens']}"
            )
            for name in (
                "engine.add_request",
                "scheduler.schedule",
                "cache_manager.get_computed_blocks",
                "partial_index.publish",
                "copy.dispatch",
                "model.forward",
                "runner.sample_tokens",
            ):
                v = d["cpu_wall_ms_per_request"].get(name)
                if v:
                    lines.append(
                        f"  {name}: inclusive {v['inclusive']:.4f} / "
                        f"exclusive {v['exclusive']:.4f} ms/request; "
                        f"{v['calls_per_request']:.2f} calls/request"
                    )
            for name in ("study/copy.dispatch", "study/model.forward"):
                v = d["trace_scope_ms_per_request"].get(name)
                if v:
                    lines.append(
                        f"  TRACE {name}: host {v['host_inclusive']:.4f} / "
                        f"device {v['device_inclusive']:.4f} ms/request"
                    )
(root / "PROFILE-RESULTS.txt").write_text("\n".join(lines) + "\n")
print(
    "PROFILE_ANALYSIS_PASSED",
    hashlib.sha256((root / "profile-summary.json").read_bytes()).hexdigest(),
)
