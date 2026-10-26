"""Plot diagnostic overhead and first-forward device activity with explicit units."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
data = json.loads((root / "profile-summary.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
arms = ["stock", "off", "partial"]
colors = ["#697582", "#a7afb7", "#2672ac"]
for row, model in enumerate(("Qwen3-0.6B", "Qwen3-1.7B")):
    cells = {a: data[model][a]["partial_reuse"] for a in arms}
    scopes = [
        "engine.add_request",
        "cache_manager.get_computed_blocks",
        "partial_index.publish",
        "copy.dispatch",
    ]
    for i, arm in enumerate(arms):
        values = [
            cells[arm]["cpu_wall_ms_per_request"].get(s, {}).get("inclusive", 0)
            for s in scopes
        ]
        axes[row, 0].bar(
            np.arange(4) + (i - 1) * 0.24,
            values,
            width=0.23,
            color=colors[i],
            label=arm,
        )
    axes[row, 0].set_xticks(
        np.arange(4),
        ["Add\nrequest", "Cache\nlookup", "Publish\npartials", "Dispatch\nKV copy"],
    )
    axes[row, 0].set_ylabel(f"{model}\nHost wall ms per request")
    axes[row, 0].set_title("Host cost over the full request (amortized)")
    axes[row, 0].legend(frameon=False)
    values = [cells[a]["trace_first_model_forward"]["gpu_event_sum_ms"] for a in arms]
    bars = axes[row, 1].bar(arms, values, color=colors, width=0.6)
    for bar, arm, value in zip(bars, arms, values):
        item = cells[arm]["trace_first_model_forward"]
        axes[row, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f} ms\n{item['gpu_event_count']} events",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[row, 1].set_ylim(0, max(values) * 1.3)
    axes[row, 1].set_ylabel("Real GPU event time per batch (ms)")
    axes[row, 1].set_title("First model forward: 3 shared-prefix requests")
for axis in axes.flat:
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
fig.suptitle("Where partial reuse adds work", fontsize=20, y=0.98)
fig.subplots_adjust(top=0.87, bottom=0.17, hspace=0.52, wspace=0.30)
fig.text(
    0.07,
    0.08,
    "Host: low-overhead CPU spans over 96 requests. GPU: one traced batch; "
    "actual kernels/memory events mapped to their launches.",
    fontsize=9,
)
fig.text(
    0.07,
    0.045,
    "Diagnostic captures have overhead and no repeat-level intervals. "
    "The five-repeat benchmark remains the speed evidence.",
    fontsize=9,
)
fig.savefig(root / "profiling-breakdown.png", dpi=160, facecolor="white")
