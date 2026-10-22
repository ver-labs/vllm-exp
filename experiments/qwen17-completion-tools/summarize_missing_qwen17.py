"""Freeze readable comparison metrics for each completed replication stage."""

from collections import defaultdict
import json
import statistics
from pathlib import Path

ROOT = Path("/Users/nandanadileep/projects/vllm/vllm/vllm-exp")
progress = json.loads((ROOT / "results/missing-qwen17/progress.json").read_text())
med = statistics.median
controls = {
    "01": "fcfs",
    "02-temporal": "vllm",
    "02-ay": "vllm",
    "03": "baseline",
    "04": "baseline",
    "05": "baseline",
    "06": "baseline",
    "10": "reference",
}
changes = {
    "01": "Rank waiting requests by cached prefix; hybrid protects the first3 FCFS requests and ranks the remaining tail. FCFS ties and10-second wait safeguard retained.",
    "02": "Prefer retaining frequently reused zero-reference cached pages. Active pages cannot be evicted. FCFS order unchanged. Temporal and concurrent A-Y profiles measured separately.",
    "03": "Approximate reuse of matching middle pages despite different preceding context, with boundary-row recomputation and a custom Metal attention path. This is NOT exact prefix reuse; semantic quality is not established.",
    "04": "Exact causal-prefix reuse inside16-token physical pages, using finer hashes and native copy-on-write before continuation. Fine-hashes-only control isolates indexing from actual partial reuse.",
    "05": "With chunked prefill OFF, bypass a waiting prompt that exceeds the remaining token budget to admit later fitting requests. Bounds:16 scanned,4 bypasses per request,30-second age protection. Restore original skipped order.",
    "06": "Enable frequency retention(02), exact partial-prefix reuse(04) and bounded unchunked skip/continue(05) together; approximate repair(03) remains OFF. Test A-Y, targeted packing and sequential temporal profiles.",
    "10": "Preserve LFU plus exact partial-prefix decisions while replacing full publisher rescans with dirty/new-range updates and repeated free-list sorting with a lazy LFU heap. Compare study reference vs incremental implementations, not stock vLLM.",
}
experiments = {
    "01": "01-prefix-scheduling",
    "02": "02-frequency-eviction",
    "03": "03-nonprefix-repair",
    "04": "04-partial-prefix",
    "05": "05-skip-continue",
    "06": "06-combined",
    "10": "10-cache-overhead",
}
by_experiment = defaultdict(list)
for stage in progress["stages"]:
    label = stage["stage"]
    data = json.loads(Path(stage["frozen_summary"]).read_text())
    records = data["runs"]
    groups = defaultdict(list)
    for record in records:
        raw = json.loads(Path(record["result"]).read_text())
        profile = record.get(
            "profile",
            "temporal"
            if label == "02-temporal"
            else "ay"
            if label != "05"
            else "targeted",
        )
        variant = record.get("variant", record.get("policy"))
        round_id = record.get("round", record.get("label"))
        summary = raw.get(
            "candidate_summary", raw.get("summary", raw.get("batch_summary"))
        )
        assert summary
        row = dict(
            profile=profile,
            variant=variant,
            round=round_id,
            duration_ms=summary["duration_ms"],
            throughput_tokens_s=summary["output_tokens_per_second"],
            first_text_ms=summary["first_text_ms"]["median"],
            request_ms=summary["total_ms"]["median"],
            p95_request_ms=summary["total_ms"].get(
                "p95", summary["total_ms"].get("p95_nearest_rank")
            ),
            result=record["result"],
        )
        cached = [
            r.get(
                "cached_prompt_tokens",
                r.get("usage", {})
                .get("prompt_tokens_details", {})
                .get("cached_tokens"),
            )
            for r in raw["requests"]
        ]
        row["cached_tokens"] = sum(x for x in cached if x is not None)
        if label == "05":
            row["groups"] = {}
            for name, ids in [
                ("long", set("ABCDEFGHIJKLM")),
                ("short", set("NOPQRSTUVWXY")),
            ]:
                rows = [r for r in raw["requests"] if r["request_id"] in ids]
                row["groups"][name] = dict(
                    first_text_ms=med(r["first_text_ms"] for r in rows),
                    total_ms=med(r["total_ms"] for r in rows),
                )
        if label == "02-temporal":
            row["probe_first_text_ms"] = raw["phases"]["hot_probe"]["first_text_ms"][
                "median"
            ]
            row["probe_cached_tokens"] = [
                r["cached_prompt_tokens"]
                for r in raw["requests"]
                if r["phase"] == "hot_probe"
            ]
        groups[(profile, variant)].append((row, raw))
    normalized = []
    for (profile, variant), pairs in groups.items():
        metrics = [x[0] for x in pairs]
        assert len(metrics) == 3
        output = dict(
            profile=profile,
            variant=variant,
            **{
                f"median_{key}": med(r[key] for r in metrics)
                for key in (
                    "duration_ms",
                    "throughput_tokens_s",
                    "first_text_ms",
                    "request_ms",
                    "p95_request_ms",
                )
            },
            duration_samples_ms=[r["duration_ms"] for r in metrics],
            cached_tokens_per_run=[r["cached_tokens"] for r in metrics],
        )
        paired_changes, answers = [], []
        for row, raw in pairs:
            baseline = next(
                x
                for x in records
                if x.get("variant", x.get("policy")) == controls[label]
                and x.get("profile", profile) == profile
                and x.get("round", x.get("label")) == row["round"]
            )
            b = json.loads(Path(baseline["result"]).read_text())
            bs = b.get("candidate_summary", b.get("summary", b.get("batch_summary")))
            paired_changes.append(100 * (row["duration_ms"] / bs["duration_ms"] - 1))
            expected = {
                (r.get("phase", ""), r["request_id"]): r["answer"]
                for r in b["requests"]
            }
            answers.append(
                dict(
                    matches=sum(
                        r["answer"] == expected[(r.get("phase", ""), r["request_id"])]
                        for r in raw["requests"]
                    ),
                    total=len(raw["requests"]),
                )
            )
        output["paired_duration_changes_pct"] = paired_changes
        output["answer_matches_all_requests"] = answers
        baseline_rows = [x[0] for x in groups[(profile, controls[label])]]
        output["duration_group_median_change_pct"] = 100 * (
            output["median_duration_ms"] / med(x["duration_ms"] for x in baseline_rows)
            - 1
        )
        if label == "05":
            output["groups"] = {
                g: {
                    k: med(x["groups"][g][k] for x in metrics)
                    for k in ("first_text_ms", "total_ms")
                }
                for g in ("long", "short")
            }
        if label == "02-temporal":
            output["median_probe_first_text_ms"] = med(
                x["probe_first_text_ms"] for x in metrics
            )
            output["probe_cached_tokens"] = [x["probe_cached_tokens"] for x in metrics]
        normalized.append(output)
    evidence = Path(stage["frozen_summary"]).parent
    (evidence / "replication-metrics.json").write_text(
        json.dumps(normalized, indent=2) + "\n"
    )
    by_experiment[label.split("-")[0]].append((stage, data, normalized))
for number, stages in by_experiment.items():
    folder = ROOT / "experiments" / experiments[number] / "qwen3-1.7B"
    complete = number != "02" or len(stages) == 2
    text = [
        f"Experiment{number}: pinned Qwen3-1.7B replication",
        "=" * 65,
        "COMPLETED" if complete else "TEMPORAL COMPLETE; A-Y FOLLOW-UP PENDING",
        "",
        "Change: " + changes[number],
        "",
        "Model:Qwen/Qwen3-1.7B, revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e.",
        "Native MLX GPU verified in every physical server log.",
        "Same request/tokenizer files and original per-profile serving settings.",
        "New model measurement session on source "
        + progress["source_checkpoint"]
        + ".",
        "Experiments01-06 use reference cache mode;10 compares both modes explicitly.",
        "Original0.6B evidence remains unchanged. No historical cross-session causal timing claim.",
        "",
        "Three fresh-server untraced timing rounds per policy/profile.",
        "Separate mechanism diagnostics are excluded from the timing numbers.",
        "Negative time change means faster; group-median and paired changes both reported.",
        "Answer agreement is exact text, not a semantic quality score.",
        "",
    ]
    for stage, data, normalized in stages:
        text.extend(
            [
                f"Stage:{stage['stage']}; {stage['gpu_runs']} GPU runs; {stage['timed_runs']} timed.",
                f"Raw summary:{stage['summary']}",
                "Profile | Variant | Batch seconds | Output tokens/s | First text ms | Request ms | p95 request ms | Change%",
            ]
        )
        for row in normalized:
            text.append(
                f"{row['profile']} | {row['variant']} | {row['median_duration_ms'] / 1000:.3f} | {row['median_throughput_tokens_s']:.3f} | {row['median_first_text_ms']:.3f} | {row['median_request_ms']:.3f} | {row['median_p95_request_ms']:.3f} | {row['duration_group_median_change_pct']:+.3f}"
            )
            text.append("  Batch samples ms:" + str(row["duration_samples_ms"]))
            text.append(
                "  Paired duration changes%:" + str(row["paired_duration_changes_pct"])
            )
            text.append(
                "  Cached prompt tokens, all requests:"
                + str(row["cached_tokens_per_run"])
            )
            text.append(
                "  Exact answer agreement, all requests:"
                + str(row["answer_matches_all_requests"])
            )
            for key in ("groups", "median_probe_first_text_ms", "probe_cached_tokens"):
                if key in row:
                    text.append("  " + key + ":" + json.dumps(row[key]))
        text.append("")
    text.extend(
        [
            "Controls, raw logs, per-request data and trace/probe outcomes are frozen in evidence.",
            "SHA256 inventories index original raw artifacts without deleting failed attempts.",
            "Three repeats are exploratory; do not equate more cache hits with overall speed.",
            "All request-time measurements include scheduler waiting; startup/warmup excluded.",
            "Reproduce the missing-model suite:.venv/bin/python src/run_missing_qwen17.py",
        ]
    )
    (folder / "RESULTS.txt").write_text("\n".join(text) + "\n")
    (folder / "manifest.json").write_text(
        json.dumps(
            dict(
                model=progress["model"],
                model_revision=progress["model_revision"],
                source_checkpoint=progress["source_checkpoint"],
                complete=complete,
                stages=[s[0] for s in stages],
            ),
            indent=2,
        )
        + "\n"
    )
print("Readable reports saved:", list(by_experiment))
