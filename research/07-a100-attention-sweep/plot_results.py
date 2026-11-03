"""Heatmaps retain every length/load cell, including regressions and uncertainty."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
summary = json.loads(source.read_text())
assert summary["status"] == "passed" and summary["repeats"] == 6
models = ("Qwen3-0.6B", "Qwen3-1.7B")
for metric, title, filename, reference in (
    ("output_tokens_per_s", "Output throughput", "throughput-vs-stock.png", "stock"),
    ("latency_ms.p95", "p95 total request latency", "latency-vs-stock.png", "stock"),
    ("ttft_ms.p50", "Median time to first token", "ttft-vs-stock.png", "stock"),
    (
        "tpot_ms.p50",
        "Median average time per output token",
        "tpot-vs-stock.png",
        "stock",
    ),
    ("output_tokens_per_s", "Output throughput", "throughput-vs-off.png", "off"),
):
    cells = {}
    extreme = 1.0
    for model in models:
        grid = []
        for length in (64, 512, 2048):
            row = []
            for concurrency in (3, 8, 16):
                effect = summary["models"][model]["cases"][f"L{length}-C{concurrency}"][
                    "attention"
                ]["metrics"][metric]["paired_vs_" + reference]
                row.append(effect)
                extreme = max(extreme, abs(effect["median_improvement_pct"]))
            grid.append(row)
        cells[model] = grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, model in zip(axes, models):
        grid = cells[model]
        values = np.array([[x["median_improvement_pct"] for x in row] for row in grid])
        chart = axis.imshow(values, cmap="RdYlGn", vmin=-extreme, vmax=extreme)
        axis.set_xticks(range(3), [3, 8, 16])
        axis.set_yticks(range(3), [64, 512, 2048])
        axis.set_xlabel("Closed-loop concurrency")
        axis.set_ylabel("Prompt tokens")
        axis.set_title(model)
        for i, row in enumerate(grid):
            for j, effect in enumerate(row):
                axis.text(
                    j,
                    i,
                    f"{values[i, j]:+.2f}%\n"
                    f"[{effect['ci95_low']:+.2f}, {effect['ci95_high']:+.2f}]\n"
                    f"{effect['wins']}/6 wins",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#111111",
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none"},
                )
    fig.suptitle(
        f"{title}: attention optimization vs {reference}\n"
        "Positive = improvement; negative = regression",
        fontsize=16,
    )
    fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.3, right=0.88)
    cax = fig.add_axes([0.91, 0.26, 0.018, 0.47])
    fig.colorbar(chart, cax=cax, label="Paired median improvement (%)")
    fig.text(
        0.08,
        0.12,
        "Each cell: six-repeat median effect, coarse 95% repeat-bootstrap interval, "
        "wins/6. "
        "Intervals spanning zero are inconclusive.",
        fontsize=9,
    )
    fig.text(
        0.08,
        0.065,
        "A100, BF16 batch-invariant eager; 4096 KV pages; no cache hits/preemptions. "
        "Synthetic requests; not peak or SLO capacity.",
        fontsize=9,
    )
    fig.savefig(output / filename, dpi=160, facecolor="white")
    plt.close(fig)
(output / "provenance.json").write_text(
    json.dumps(
        {
            "summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
        },
        indent=2,
    )
    + "\n"
)
