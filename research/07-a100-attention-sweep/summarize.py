"""Repeat-level paired statistics for the fixed nine-cell attention sweep."""

import csv
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
arms = ("stock", "off", "attention")
cases = [
    f"L{length}-C{concurrency}"
    for length in (64, 512, 2048)
    for concurrency in (3, 8, 16)
]
metrics = {
    "output_tokens_per_s": True,
    "requests_per_s": True,
    "latency_ms.p50": False,
    "latency_ms.p95": False,
    "ttft_ms.p50": False,
    "ttft_ms.p95": False,
    "tpot_ms.p50": False,
    "tpot_ms.p95": False,
}


def value(record, metric):
    for part in metric.split("."):
        record = record[part]
    return record


def quantile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    low = int(index)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        index - low
    )


def paired(left, right, higher):
    effects = [
        100 * (a / b - 1) if higher else 100 * (1 - a / b) for a, b in zip(left, right)
    ]
    assert len(effects) == 6
    rng = random.Random(20261027)
    boot = [
        statistics.median([rng.choice(effects) for _ in effects]) for _ in range(5000)
    ]
    return {
        "median_improvement_pct": statistics.median(effects),
        "ci95_low": quantile(boot, 0.025),
        "ci95_high": quantile(boot, 0.975),
        "wins": sum(x > 0 for x in effects),
        "repeats": 6,
        "paired_improvements_pct": effects,
    }


execution = json.loads((root / "execution.json").read_text())
assert len(execution) == 36 and all(r["returncode"] == 0 for r in execution)
assert json.loads((root / "validation.json").read_text())["status"] == "passed"
summary = {
    "status": "passed",
    "repeats": 6,
    "fresh_engines": 36,
    "timed_requests": 0,
    "actual_output_tokens": 0,
    "statistical_unit": "paired repeat, n=6",
    "models": {},
}
rows = []
for model in ("Qwen3-0.6B", "Qwen3-1.7B"):
    results = {
        arm: [
            json.loads(
                (root / model / f"repeat-{r + 1}" / arm / "result.json").read_text()
            )
            for r in range(6)
        ]
        for arm in arms
    }
    gold = {}
    for arm, runs in results.items():
        for run in runs:
            assert run["status"] == "passed" and set(run["cases"]) == set(cases)
            for case, data in run["cases"].items():
                requests = data["requests"]
                assert (
                    len(requests) == 96
                    and data["cached_tokens"] == data["preemptions"] == 0
                )
                actual = {r["id"]: r["output_token_ids"] for r in requests}
                assert len(actual) == 96 and all(
                    len(ids) == 32 for ids in actual.values()
                )
                gold.setdefault(case, actual)
                assert actual == gold[case], (model, arm, case, "output mismatch")
                assert (
                    abs(data["output_tokens_per_s"] - 32 * data["requests_per_s"])
                    < 1e-6
                )
                for r in requests:
                    assert len(r["token_arrivals_offset_ns"]) == 32
                    assert r["latency_ms"] >= r["ttft_ms"] > 0
                    assert (
                        abs(r["latency_ms"] - r["ttft_ms"] - 31 * r["tpot_ms"]) < 1e-6
                    )
                summary["timed_requests"] += len(requests)
                summary["actual_output_tokens"] += sum(map(len, actual.values()))
    model_summary = {
        "cases": {},
        "output_parity": True,
        "output_sha256": hashlib.sha256(
            json.dumps(gold, sort_keys=True).encode()
        ).hexdigest(),
    }
    for case in cases:
        cells = {}
        for arm in arms:
            cell = {"metrics": {}}
            for metric, higher in metrics.items():
                samples = [value(x["cases"][case], metric) for x in results[arm]]
                detail = {
                    "median": statistics.median(samples),
                    "min": min(samples),
                    "max": max(samples),
                    "run_values": samples,
                }
                for reference in ("stock", "off") if arm == "attention" else ("stock",):
                    baseline = [
                        value(x["cases"][case], metric) for x in results[reference]
                    ]
                    effect = paired(samples, baseline, higher)
                    detail["paired_vs_" + reference] = effect
                    rows.append(
                        {
                            "model": model,
                            "case": case,
                            "arm": arm,
                            "reference": reference,
                            "metric": metric,
                            "median": detail["median"],
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
                cell["metrics"][metric] = detail
            cells[arm] = cell
        model_summary["cases"][case] = cells
    summary["models"][model] = model_summary
assert summary["timed_requests"] == 31104 and summary["actual_output_tokens"] == 995328
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = [
    "Longer-prompt / higher-concurrency attention sweep",
    "=" * 50,
    "",
    "Six balanced paired repeats; unique requests; zero cache hits/preemptions.",
    "BF16 batch-invariant FA2 eager, token budget 2048, 4096 KV pages, "
    "max16 sequences.",
    "31,104 measured requests / 995,328 actual output tokens. All IDs match stock.",
    "Effects are paired-repeat medians. Positive means improvement. Coarse 95%",
    "Intervals resample six repeats; no multiple-comparison correction. "
    "Not SLO capacity.",
]
for model, data in summary["models"].items():
    for case, cells in data["cases"].items():
        lines += ["", model + " / " + case]
        for arm, cell in cells.items():
            m = cell["metrics"]
            lines.append(
                f"{arm}: {m['output_tokens_per_s']['median']:.2f} tok/s; "
                f"p95 E2E {m['latency_ms.p95']['median']:.2f} ms; "
                f"TTFT p50/p95 {m['ttft_ms.p50']['median']:.2f}/"
                f"{m['ttft_ms.p95']['median']:.2f} ms; "
                f"TPOT p50 {m['tpot_ms.p50']['median']:.2f} ms"
            )
            if arm == "attention":
                for reference in ("stock", "off"):
                    for metric in (
                        "output_tokens_per_s",
                        "latency_ms.p95",
                        "ttft_ms.p50",
                        "tpot_ms.p50",
                    ):
                        e = m[metric]["paired_vs_" + reference]
                        lines.append(
                            f"  vs {reference}, {metric}: "
                            f"{e['median_improvement_pct']:+.2f}% "
                            f"[{e['ci95_low']:+.2f}, {e['ci95_high']:+.2f}], "
                            f"wins {e['wins']}/6"
                        )
(root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
print(
    "SWEEP_SUMMARY_PASSED",
    summary["timed_requests"],
    summary["actual_output_tokens"],
    flush=True,
)
