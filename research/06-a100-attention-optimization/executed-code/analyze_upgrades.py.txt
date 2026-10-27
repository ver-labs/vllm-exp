"""Paired contrasts isolate attention changes from cache policy effects."""

import csv
import json
import random
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
rows = []
for model, model_data in summary["models"].items():
    for profile, cells in model_data["profiles"].items():
        for arm, reference in (
            ("attention", "off"),
            ("partial_attention", "partial"),
            ("partial_attention", "attention"),
        ):
            for metric, result in cells[arm]["metrics"].items():
                left = result["run_values"]
                right = cells[reference]["metrics"][metric]["run_values"]
                higher = metric in ("output_tokens_per_s", "requests_per_s")
                effects = [
                    100 * (a / b - 1) if higher else 100 * (1 - a / b)
                    for a, b in zip(left, right)
                ]
                rng = random.Random(20261026)
                boot = sorted(
                    statistics.median([rng.choice(effects) for _ in effects])
                    for _ in range(5000)
                )

                def quantile(p):
                    i = (len(boot) - 1) * p
                    lo = int(i)
                    return boot[lo] + (boot[min(lo + 1, len(boot) - 1)] - boot[lo]) * (
                        i - lo
                    )

                rows.append(
                    {
                        "model": model,
                        "profile": profile,
                        "arm": arm,
                        "reference": reference,
                        "metric": metric,
                        "median_improvement_pct": statistics.median(effects),
                        "ci95_low": quantile(0.025),
                        "ci95_high": quantile(0.975),
                        "wins": sum(x > 0 for x in effects),
                        "repeats": 5,
                    }
                )
with (root / "upgrade-comparison.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "upgrade-comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
print("UPGRADE_COMPARISONS_PASSED", len(rows), flush=True)
