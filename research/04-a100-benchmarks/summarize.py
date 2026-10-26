"""Summarize paired repeat-level effects; never pool requests as repeats."""

import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = ["stock", "off", "partial", "aging", "combined"]
profiles = ["partial_reuse", "popularity_shift", "unique_control"]
metrics = {
    "output_tokens_per_s": True,
    "requests_per_s": True,
    "latency_ms.p50": False,
    "latency_ms.p95": False,
    "ttft_ms.p50": False,
    "ttft_ms.p95": False,
    "tpot_ms.p50": False,
}


def median(values):
    return statistics.median(values)


def quantile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        index - low
    )


def value(record, metric):
    for field in metric.split("."):
        record = record[field]
    return record


def paired(samples, baseline, higher):
    improvements = [
        100 * (a / b - 1) if higher else 100 * (1 - a / b)
        for a, b in zip(samples, baseline)
    ]
    rng = random.Random(20261026)
    bootstrap = [
        median([rng.choice(improvements) for _ in improvements]) for _ in range(5000)
    ]
    return {
        "median_improvement_pct": median(improvements),
        "ci95_low": quantile(bootstrap, 0.025),
        "ci95_high": quantile(bootstrap, 0.975),
        "wins": sum(x > 0 for x in improvements),
        "repeats": len(improvements),
        "paired_improvements_pct": improvements,
    }


records = json.loads((root / "execution.json").read_text())
assert len(records) == 50 and all(x["returncode"] == 0 for x in records)
summary = {
    "status": "passed",
    "configuration": "BF16 batch-invariant eager engine, concurrency 3, 16 KV pages",
    "repeats": 5,
    "requests_per_profile": 96,
    "timed_requests": 14400,
    "actual_output_tokens": 0,
    "models": {},
    "statistical_unit": "paired repeat, n=5",
}
csv_rows = []
for model in sorted(p for p in root.iterdir() if p.is_dir()):
    results = {
        arm: [
            json.loads((model / f"repeat-{r + 1}" / arm / "result.json").read_text())
            for r in range(5)
        ]
        for arm in arms
    }
    golden = {}
    for arm, runs in results.items():
        for run in runs:
            assert run["status"] == "passed"
            for profile, data in run["profiles"].items():
                assert len(data["requests"]) == 96
                actual = {r["id"]: r["output_token_ids"] for r in data["requests"]}
                assert all(len(ids) == 32 for ids in actual.values())
                golden.setdefault(profile, actual)
                assert actual == golden[profile], (
                    model.name,
                    arm,
                    profile,
                    "output mismatch",
                )
                assert (
                    abs(data["output_tokens_per_s"] - 32 * data["requests_per_s"])
                    < 1e-6
                )
                for row in data["requests"]:
                    assert len(row["token_arrivals_offset_ns"]) == 32
                    assert row["latency_ms"] >= row["ttft_ms"] > 0
                    assert (
                        abs(row["latency_ms"] - row["ttft_ms"] - 31 * row["tpot_ms"])
                        < 1e-6
                    )
                summary["actual_output_tokens"] += 3072
    model_summary = {
        "profiles": {},
        "output_parity": True,
        "output_fingerprint_sha256": hashlib.sha256(
            json.dumps(golden, sort_keys=True).encode()
        ).hexdigest(),
    }
    for profile in profiles:
        cells = {}
        for arm in arms:
            data = [x["profiles"][profile] for x in results[arm]]
            baseline = [x["profiles"][profile] for x in results["stock"]]
            cell = {
                "metrics": {},
                "median_cached_tokens": median([x["cached_tokens"] for x in data]),
                "median_preemptions": median([x["preemptions"] for x in data]),
            }
            for metric, higher in metrics.items():
                samples = [value(x, metric) for x in data]
                bases = [value(x, metric) for x in baseline]
                effect = paired(samples, bases, higher)
                detail = {
                    "median": median(samples),
                    "min": min(samples),
                    "max": max(samples),
                    "run_values": samples,
                    "paired_vs_stock": effect,
                }
                cell["metrics"][metric] = detail
                csv_rows.append(
                    {
                        "model": model.name,
                        "profile": profile,
                        "arm": arm,
                        "metric": metric,
                        "median": detail["median"],
                        "min": detail["min"],
                        "max": detail["max"],
                        **{
                            k: effect[k]
                            for k in (
                                "median_improvement_pct",
                                "ci95_low",
                                "ci95_high",
                                "wins",
                                "repeats",
                            )
                        },
                    }
                )
            if profile == "popularity_shift":
                phase_data = {}
                for name, lo, hi in (
                    ("hot_A", 0, 32),
                    ("hot_B", 32, 64),
                    ("return_A", 64, 96),
                ):
                    per_run = []
                    for run in data:
                        selected = [
                            r
                            for r in run["requests"]
                            if lo <= int(r["id"].rsplit("-", 1)[1]) < hi
                        ]
                        per_run.append(
                            {
                                "ttft_p50_ms": median([r["ttft_ms"] for r in selected]),
                                "cached_tokens": sum(
                                    r["cached_tokens"] or 0 for r in selected
                                ),
                            }
                        )
                    phase_data[name] = {
                        "median_ttft_p50_ms": median(
                            [x["ttft_p50_ms"] for x in per_run]
                        ),
                        "median_cached_tokens": median(
                            [x["cached_tokens"] for x in per_run]
                        ),
                    }
                cell["phases"] = phase_data
            cells[arm] = cell
        model_summary["profiles"][profile] = cells
    summary["models"][model.name] = model_summary
assert summary["actual_output_tokens"] == 460800
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
lines = [
    "A100 latency/throughput results — 26 October 2026",
    "=================================================",
    "",
    "Five balanced paired repeats; fresh engine per arm/repeat; all runs on one",
    "A100 40 GB full-size MIG instance. BF16, batch-invariant mode, eager, concurrency",
    "three, 16 total KV pages. These are token-ID engine timings, not HTTP latency",
    "or peak vLLM capacity. 14,400 requests / 460,800 actual generated tokens.",
    "All generated token IDs matched stock across paired arms and repetitions.",
    "",
    "Effects are paired repeat medians; positive means improvement. 95% bootstrap",
    "intervals resample five repeats and are coarse. Intervals including zero",
    "do not establish an improvement. No default-BF16 or production claim follows.",
]
for model, model_data in summary["models"].items():
    for profile, cells in model_data["profiles"].items():
        lines += ["", model + " / " + profile, "-" * (len(model) + 3 + len(profile))]
        for arm, cell in cells.items():
            m = cell["metrics"]
            lines.append(
                f"{arm}: {m['output_tokens_per_s']['median']:.2f} output tok/s; "
                f"{m['requests_per_s']['median']:.3f} req/s; "
                f"E2E p50/p95 {m['latency_ms.p50']['median']:.2f}/"
                f"{m['latency_ms.p95']['median']:.2f} ms; "
                f"TTFT p50/p95 {m['ttft_ms.p50']['median']:.2f}/"
                f"{m['ttft_ms.p95']['median']:.2f} ms; "
                f"TPOT p50 {m['tpot_ms.p50']['median']:.2f} ms"
            )
            if arm != "stock":
                for metric, label in (
                    ("output_tokens_per_s", "Throughput"),
                    ("latency_ms.p95", "p95 latency"),
                    ("ttft_ms.p50", "p50 TTFT"),
                ):
                    e = m[metric]["paired_vs_stock"]
                    lines.append(
                        f"  {label} improvement: {e['median_improvement_pct']:+.2f}% "
                        f"[95% interval {e['ci95_low']:+.2f}, "
                        f"{e['ci95_high']:+.2f}], wins {e['wins']}/5"
                    )
        lines.append(
            "Native cached-token / preemption count medians: "
            + ", ".join(
                f"{a} {d['median_cached_tokens']}/{d['median_preemptions']}"
                for a, d in cells.items()
            )
        )
lines += [
    "",
    "See PROTOCOL.txt, comparison.csv, raw per-request records and summary.json.",
    "Workloads, model hashes, source pins, repeated-run order and output parity",
    "checks are retained. Speed results apply only to this declared test scope.",
]
(root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
print(
    "BENCHMARK_SUMMARY",
    summary["status"],
    "50 runs / 150 profiles / 14400 requests",
    flush=True,
)
