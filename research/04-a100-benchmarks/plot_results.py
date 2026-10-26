"""Plot paired repeat effects with coarse bootstrap intervals."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

summary_path = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
summary = json.loads(summary_path.read_text())
assert summary["status"] == "passed" and summary["repeats"] == 5
models = ["Qwen3-0.6B", "Qwen3-1.7B"]
profiles = ["partial_reuse", "popularity_shift", "unique_control"]
profile_labels = [
    "Shared-prefix reuse",
    "Popularity shift + pressure",
    "Unique-prompt control",
]
arms = ["off", "partial", "aging", "combined"]
arm_labels = ["Flags off", "Partial", "Aging", "Combined"]
colors = ["#8b949e", "#2672ac", "#d17c21", "#7252a3"]
for metric, title, unit, filename in (
    (
        "output_tokens_per_s",
        "Output throughput compared with stock",
        "tok/s",
        "throughput-comparison.png",
    ),
    (
        "latency_ms.p95",
        "p95 request latency compared with stock",
        "ms",
        "latency-p95-comparison.png",
    ),
    (
        "ttft_ms.p50",
        "Median time to first token compared with stock",
        "ms",
        "ttft-comparison.png",
    ),
):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    lows, highs = [0.0], [0.0]
    for row, model in enumerate(models):
        for column, profile in enumerate(profiles):
            axis = axes[row, column]
            cells = summary["models"][model]["profiles"][profile]
            effects = [cells[arm]["metrics"][metric]["paired_vs_stock"] for arm in arms]
            values = np.array([effect["median_improvement_pct"] for effect in effects])
            lower = np.array([effect["ci95_low"] for effect in effects])
            upper = np.array([effect["ci95_high"] for effect in effects])
            lows.extend(lower.tolist())
            highs.extend(upper.tolist())
            x = np.arange(4)
            axis.bar(x, values, color=colors, width=0.66, zorder=3)
            axis.errorbar(
                x,
                values,
                yerr=np.vstack((values - lower, upper - values)),
                fmt="none",
                color="#222222",
                capsize=4,
                linewidth=1.2,
                zorder=4,
            )
            axis.axhline(0, color="#39424c", linewidth=1)
            axis.grid(axis="y", color="#dfe4e8", linewidth=0.7, zorder=0)
            axis.set_axisbelow(True)
            axis.set_xticks(x, arm_labels, fontsize=9)
            stock = cells["stock"]["metrics"][metric]["median"]
            axis.set_title(
                f"{profile_labels[column]}\nStock median: {stock:.2f} {unit}",
                fontsize=11,
                pad=12,
            )
            if column == 0:
                axis.set_ylabel(f"{model}\nImprovement vs stock (%)", fontsize=11)
            for index, effect in enumerate(effects):
                axis.annotate(
                    f"{values[index]:+.1f}%\n{effect['wins']}/5 wins",
                    (index, upper[index] if values[index] >= 0 else lower[index]),
                    xytext=(0, 7 if values[index] >= 0 else -9),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if values[index] >= 0 else "top",
                    fontsize=8,
                )
            axis.spines[["top", "right"]].set_visible(False)
    low, high = min(lows), max(highs)
    span = max(high - low, 6)
    for axis in axes.flat:
        axis.set_ylim(low - 0.38 * span, high + 0.38 * span)
    fig.suptitle(
        title + "\nPositive = improvement; negative = regression", fontsize=18, y=0.98
    )
    fig.subplots_adjust(top=0.82, bottom=0.14, hspace=0.62, wspace=0.2)
    fig.text(
        0.06,
        0.055,
        "Five paired repeats; error bars are 95% repeat-bootstrap intervals "
        "(coarse at n=5).",
        fontsize=10,
    )
    fig.text(
        0.06,
        0.025,
        "A100 40 GB full-size MIG; BF16 batch-invariant eager engine; "
        "concurrency 3; 16 KV pages. Not peak/default-mode performance.",
        fontsize=9,
        color="#45505c",
    )
    fig.savefig(output / filename, dpi=160, facecolor="white")
    plt.close(fig)
metadata = {
    "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    "matplotlib": matplotlib.__version__,
    "numpy": np.__version__,
    "figure_note": (
        "Effect is median paired percent change, not ratio of unpaired medians. "
        "Positive means improvement for every plotted metric."
    ),
}
(output / "plot-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
