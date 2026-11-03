"""Plot every serving comparison cell with paired-repeat uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")
parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()
root = Path(args.root)
data = json.loads((root / "summary.json").read_text())
rows = data["cells"]
models = ["Qwen3-0.6B", "Qwen3-1.7B", "SmolLM3-3B"]
cases = ["C3-O32", "C16-O32", "C3-O128", "C16-O128"]
colors = ["#167D8D", "#3B5BA9", "#9653A8"]
metrics = [
    ("output_tokens_per_s", "Output throughput"),
    ("latency_ms_p95", "P95 total latency"),
    ("ttft_ms_p95", "P95 time to first token"),
    ("tpot_ms_p50", "Median time per output token"),
]
for mode in sorted({r["mode"] for r in rows}):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, (metric, title) in zip(axes.flat, metrics, strict=True):
        for i, model in enumerate(models):
            selected = [
                next(
                    r
                    for r in rows
                    if r["model"] == model
                    and r["mode"] == mode
                    and r["case"] == case
                    and r["metric"] == metric
                )
                for case in cases
            ]
            values = np.array([r["improvement_percent"] for r in selected])
            lower = values - np.array([r["ci95_low"] for r in selected])
            upper = np.array([r["ci95_high"] for r in selected]) - values
            bars = ax.bar(
                np.arange(4) + (i - 1) * 0.24,
                values,
                width=0.23,
                color=colors[i],
                label=model,
                zorder=3,
            )
            ax.errorbar(
                np.arange(4) + (i - 1) * 0.24,
                values,
                yerr=[lower, upper],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2,
                zorder=4,
            )
            for bar, row in zip(bars, selected, strict=True):
                if not row["output_ids_match"]:
                    bar.set_hatch("///")
        ax.axhline(0, color="#333333", linewidth=0.9)
        ax.set_xticks(np.arange(4), ["C3 / 32", "C16 / 32", "C3 / 128", "C16 / 128"])
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Improvement over stock (%)")
        ax.set_xlabel("Concurrency / output tokens")
        ax.grid(axis="y", color="#DFE4E8", linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.suptitle(
        "Attention routing with CUDA graphs and human prompts",
        fontsize=16,
        weight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        "A100 · batch-invariant BF16 FA2 · six paired repeats · "
        "95% bootstrap intervals\n"
        "Positive is better. Hatching: output differences. Default BF16 excluded.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.89))
    dest = root / "figures"
    dest.mkdir(exist_ok=True)
    fig.savefig(dest / f"{mode}-comparison.png", dpi=180)
    fig.savefig(dest / f"{mode}-comparison.pdf")
    plt.close(fig)
