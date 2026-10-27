"""Plot paired throughput contrasts that separate attention and cache effects."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
rows = json.loads((root / "upgrade-comparison.json").read_text())
contrasts = [
    ("attention", "off"),
    ("partial_attention", "partial"),
    ("partial_attention", "attention"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
low, high = 0, 0
for r, model in enumerate(("Qwen3-0.6B", "Qwen3-1.7B")):
    for c, profile in enumerate(
        ("partial_reuse", "popularity_shift", "unique_control")
    ):
        selected = []
        for arm, reference in contrasts:
            selected.append(
                next(
                    x
                    for x in rows
                    if x["model"] == model
                    and x["profile"] == profile
                    and x["arm"] == arm
                    and x["reference"] == reference
                    and x["metric"] == "output_tokens_per_s"
                )
            )
        values = np.array([x["median_improvement_pct"] for x in selected])
        lows = np.array([x["ci95_low"] for x in selected])
        highs = np.array([x["ci95_high"] for x in selected])
        low, high = min(low, lows.min()), max(high, highs.max())
        axis = axes[r, c]
        axis.bar(range(3), values, color=["#209489", "#7252a3", "#2672ac"], zorder=3)
        axis.errorbar(
            range(3),
            values,
            yerr=[values - lows, highs - values],
            fmt="none",
            color="#222222",
            capsize=4,
            zorder=4,
        )
        axis.axhline(0, color="#39424c", linewidth=1)
        axis.set_xticks(
            range(3),
            ["Attention\nchange", "Attention\nwith partial", "Extra partial\nreuse"],
            fontsize=9,
        )
        axis.set_title(profile.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        if c == 0:
            axis.set_ylabel(model + "\nThroughput improvement (%)")
        for i, x in enumerate(selected):
            axis.annotate(
                f"{values[i]:+.2f}%\n{x['wins']}/5 wins",
                (i, highs[i] if values[i] >= 0 else lows[i]),
                xytext=(0, 7 if values[i] >= 0 else -9),
                textcoords="offset points",
                ha="center",
                va="bottom" if values[i] >= 0 else "top",
                fontsize=8,
            )
span = max(high - low, 6)
for axis in axes.flat:
    axis.set_ylim(low - 0.38 * span, high + 0.38 * span)
fig.suptitle("Separating attention changes from extra cache reuse", fontsize=19, y=0.98)
fig.subplots_adjust(top=0.86, bottom=0.23, hspace=0.55, wspace=0.2)
fig.text(
    0.07,
    0.13,
    "Attention change: attention vs features off. "
    "Attention with partial: new combined vs old partial.",
    fontsize=10,
)
fig.text(
    0.07,
    0.09,
    "Extra partial reuse: combined vs attention only. Positive = gain; "
    "negative = loss. Five paired repeats; coarse 95% intervals.",
    fontsize=9,
)
fig.text(
    0.07,
    0.045,
    "A100, BF16 batch-invariant eager execution, concurrency 3, 16 KV pages; "
    "synthetic short prompts. Not peak/default serving.",
    fontsize=9,
)
fig.savefig(output / "isolated-throughput-effects.png", dpi=160, facecolor="white")
