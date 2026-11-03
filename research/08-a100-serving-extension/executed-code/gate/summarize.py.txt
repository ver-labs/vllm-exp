"""Report every paired serving cell; reject output-incomparable speed claims."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def value(result, case, metric):
    data = result["cases"][case]
    if metric == "output_tokens_per_s":
        return data[metric]
    name, quantile = metric.rsplit("_", 1)
    rows = [row[name] for row in data["requests"]]
    return float(np.percentile(rows, int(quantile[1:])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    gates = json.loads((args.root / "gate/validation.json").read_text())
    rng = np.random.default_rng(20261103)
    records = []
    for key, gate in gates.items():
        if gate["status"] != "passed":
            continue
        model, mode = key.split("/")
        repeats = []
        for repeat in range(1, 7):
            root = args.root / "matrix" / model / mode / f"repeat-{repeat}"
            arms = {
                arm: json.loads((root / arm / "result.json").read_text())
                for arm in ("stock", "attention")
            }
            assert all(r["status"] == "passed" for r in arms.values())
            assert arms["stock"]["gpu"] == arms["attention"]["gpu"]
            repeats.append(arms)
        cases = repeats[0]["stock"]["cases"]
        for case in cases:
            matched = True
            for pair in repeats:
                a = {
                    r["id"]: r["token_ids"]
                    for r in pair["stock"]["cases"][case]["requests"]
                }
                b = {
                    r["id"]: r["token_ids"]
                    for r in pair["attention"]["cases"][case]["requests"]
                }
                matched &= a == b
            for metric in (
                "output_tokens_per_s",
                "latency_ms_p50",
                "latency_ms_p95",
                "ttft_ms_p50",
                "ttft_ms_p95",
                "tpot_ms_p50",
                "tpot_ms_p95",
            ):
                a = np.array([value(p["stock"], case, metric) for p in repeats])
                b = np.array([value(p["attention"], case, metric) for p in repeats])
                effects = (
                    (b / a - 1) * 100
                    if metric == "output_tokens_per_s"
                    else (1 - b / a) * 100
                )
                boot = np.median(rng.choice(effects, size=(5000, 6)), axis=1)
                low, high = np.percentile(boot, (2.5, 97.5))
                records.append(
                    {
                        "model": model,
                        "mode": mode,
                        "case": case,
                        "metric": metric,
                        "output_ids_match": matched,
                        "stock_median": float(np.median(a)),
                        "attention_median": float(np.median(b)),
                        "improvement_percent": float(np.median(effects)),
                        "ci95_low": float(low),
                        "ci95_high": float(high),
                        "wins": int(sum(effects > 0)),
                        "repeats": 6,
                        "paired_effects": effects.tolist(),
                    }
                )
    (args.root / "summary.json").write_text(
        json.dumps({"gates": gates, "cells": records}, indent=2) + "\n"
    )
    if records:
        keys = [k for k in records[0] if k != "paired_effects"]
        with (args.root / "comparison.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "CUDA-graph HTTP serving extension",
        "================================",
        "",
    ]
    for key, gate in gates.items():
        lines.append(f"Gate {key}: {gate['status']}")
    lines += [
        "",
        "Positive percentages mean improvement. Intervals use six paired repeats.",
        "Output mismatches invalidate an output-equivalent speed claim.",
        "",
    ]
    for row in records:
        if row["metric"] == "output_tokens_per_s":
            lines.append(
                f"{row['model']} {row['mode']} {row['case']}: "
                f"{row['improvement_percent']:+.2f}% "
                f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}], "
                f"{row['wins']}/6 wins; IDs match={row['output_ids_match']}"
            )
    (args.root / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
