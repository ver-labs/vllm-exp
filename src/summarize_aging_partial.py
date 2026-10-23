"""Validate and archive the four experiment-14 comparisons."""

import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

from run_aging_partial import REVISIONS, ROOT, VARIANTS

DEST = ROOT / "experiments/14-aging-partial"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    comparisons, audit, files = [], [], []
    for model, revision in REVISIONS.items():
        for profile in ("shift", "mixed"):
            candidates = sorted(
                (
                    ROOT / "results/aging-partial" / model.replace("/", "--") / profile
                ).glob("batches-*/summary.json")
            )
            assert candidates
            source = candidates[-1]
            summary = read(source)
            assert summary["complete"] and "error" not in summary
            assert summary["manifest"]["model"] == model
            assert summary["manifest"]["revision"] == revision
            assert len(summary["diagnostics"]) == 4
            assert len(summary["runs"]) == 12
            part = DEST / model.split("/")[-1] / profile / "evidence"
            part.mkdir(parents=True, exist_ok=True)
            records = [read(p) for p in summary["diagnostics"] + summary["runs"]]
            assert len({r["manifest_sha256"] for r in records}) == 1
            for record in records:
                assert record["complete"] and "error" not in record
                assert len(record["rows"]) == len(summary["manifest"]["requests"])
                assert all(
                    r["usage"]["prompt_tokens"] == r["prompt_length"]
                    for r in record["rows"]
                )
                env = record["study_environment"]
                assert env["STUDY_ADAPTIVE_PREFILL"] == "0"
                assert env["STUDY_SKIP_CONTINUE"] == "0"
                assert env["STUDY_WAITING_POLICY"] == "fcfs"
                assert env["STUDY_CACHE_IMPL"] == "incremental"
                traced = record["mode"] == "trace"
                assert env["VLLM_STUDY_TRACE"] == str(int(traced))
                assert env["STUDY_CACHE_PROFILE"] == str(int(traced))
                if traced:
                    diag = record["diagnostic"]
                    assert diag["lookup_matches_api"] and diag["metal_gpu"]
                    assert diag["max_running"] == summary["manifest"]["running_slots"]
                    assert not diag["preemptions"]
                    assert not diag["invalid_eviction_refs"]
                run_dir = Path(record["run_dir"])
                assert (
                    "MLX device set to: Device(gpu, 0)"
                    in (run_dir / "server.log").read_text()
                )
                destination = part / run_dir.name
                destination.mkdir(exist_ok=True)
                for path in sorted(run_dir.iterdir()):
                    if not path.is_file() or path.suffix == ".tmp":
                        continue
                    target = destination / path.name
                    if (
                        path.suffix in (".jsonl", ".log")
                        or path.name == "metrics-after.txt"
                    ):
                        target = target.with_name(target.name + ".gz")
                        with (
                            path.open("rb") as src,
                            gzip.open(target, "wb", compresslevel=1) as dst,
                        ):
                            shutil.copyfileobj(src, dst)
                        with gzip.open(target, "rb") as check:
                            assert hashlib.sha256(check.read()).hexdigest() == digest(
                                path
                            )
                    else:
                        shutil.copy2(path, target)
                        assert digest(path) == digest(target)
                    files.append(
                        dict(
                            path=str(target.relative_to(DEST)),
                            sha256=digest(target),
                            source=str(path),
                            source_sha256=digest(path),
                        )
                    )
            shutil.copy2(source, part / "summary.json")
            files.append(
                dict(
                    path=str((part / "summary.json").relative_to(DEST)),
                    sha256=digest(part / "summary.json"),
                    source=str(source),
                )
            )
            aggregate = summary["aggregate"]
            baseline = aggregate["variants"]["stock"]["median_duration_ms"]
            row = dict(
                model=model,
                profile=profile,
                source_summary=str(source),
                archived_summary=str((part / "summary.json").relative_to(DEST)),
                aggregate=aggregate,
            )
            comparisons.append(row)
            audit.append(
                dict(
                    model=model,
                    profile=profile,
                    runs=len(records),
                    requests_per_run=len(summary["manifest"]["requests"]),
                    measured_output_tokens=sum(
                        r["summary"]["output_tokens"]
                        for r in records
                        if r["mode"] == "bench"
                    ),
                    diagnostic_output_tokens=sum(
                        r["summary"]["output_tokens"]
                        for r in records
                        if r["mode"] == "trace"
                    ),
                    source_commit=summary["source_commit"],
                    manifest_sha256=records[0]["manifest_sha256"],
                    valid=True,
                )
            )
            lines = [
                f"Experiment 14 | {model} | {profile}",
                "=" * 72,
                "Medians of three fresh-server untraced timing runs.",
                "Lower duration/latency is better; higher throughput is better.",
                "",
                (
                    "Variant     Batch s   Output tok/s  "
                    "First text ms  p95 request ms  vs stock"
                ),
            ]
            for variant, values in aggregate["variants"].items():
                change = (values["median_duration_ms"] / baseline - 1) * 100
                lines.append(
                    f"{variant:<10} {values['median_duration_ms'] / 1000:8.3f} "
                    f"{values['median_output_tokens_per_second']:12.3f} "
                    f"{values['median_first_text_ms']:14.3f} "
                    f"{values['median_p95_request_ms']:15.3f} {change:+8.2f}%"
                )
                lines.append(f"  cached tokens/run: {values['cached_tokens']}")
                lines.append(
                    "  cohort latency medians: "
                    + json.dumps(values["groups"], sort_keys=True)
                )
            lines += [
                "",
                "Paired rounds (preserve alongside ratios of group medians):",
                json.dumps(aggregate["paired"], indent=2),
                "",
                "Combination compared directly with each component:",
            ]
            both = aggregate["variants"]["combined"]["median_duration_ms"]
            for component in ("aging", "partial"):
                value = aggregate["variants"][component]["median_duration_ms"]
                wins = sum(
                    next(
                        r
                        for r in records
                        if r["mode"] == "bench"
                        and r["round"] == i
                        and r["variant"] == "combined"
                    )["summary"]["duration_ms"]
                    < next(
                        r
                        for r in records
                        if r["mode"] == "bench"
                        and r["round"] == i
                        and r["variant"] == component
                    )["summary"]["duration_ms"]
                    for i in (1, 2, 3)
                )
                lines.append(
                    f"  vs {component}: {(both / value - 1) * 100:+.2f}% batch median; "
                    f"{wins}/3 paired rounds faster"
                )
            lines += [
                "",
                "Cautions: only three timing rounds; synthetic workload; closed-loop",
                "completion order affects arrivals/cache churn in mixed traffic.",
                "Shift is sequential.",
                "Seed publication excluded. Exact text agreement is not a",
                "semantic quality score. Traced/profiled runs excluded from timing.",
            ]
            (part.parent / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    # Both models must receive identical token requests within each profile.
    for profile in ("shift", "mixed"):
        manifests = [
            read(c["source_summary"])["manifest"]
            for c in comparisons
            if c["profile"] == profile
        ]
        assert manifests[0]["requests"] == manifests[1]["requests"]
        assert manifests[0]["seeds"] == manifests[1]["seeds"]
    report = dict(
        experiment=14,
        comparisons=comparisons,
        validation=audit,
        diagnostic_runs=16,
        timed_runs=48,
        all_valid=True,
        source_files={
            name: digest(ROOT / "src" / name)
            for name in ("run_aging_partial.py", "summarize_aging_partial.py")
        },
    )
    (DEST / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (DEST / "evidence-sha256.json").write_text(json.dumps(files, indent=2) + "\n")
    with (DEST / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "model",
                "profile",
                "variant",
                "batch_seconds",
                "batch_change_pct",
                "output_tokens_per_second",
                "first_text_ms",
                "p95_request_ms",
                "cached_tokens_runs",
                "paired_faster_than_stock",
            ],
        )
        writer.writeheader()
        for c in comparisons:
            base = c["aggregate"]["variants"]["stock"]["median_duration_ms"]
            for v, d in c["aggregate"]["variants"].items():
                writer.writerow(
                    dict(
                        model=c["model"],
                        profile=c["profile"],
                        variant=v,
                        batch_seconds=d["median_duration_ms"] / 1000,
                        batch_change_pct=(d["median_duration_ms"] / base - 1) * 100,
                        output_tokens_per_second=d["median_output_tokens_per_second"],
                        first_text_ms=d["median_first_text_ms"],
                        p95_request_ms=d["median_p95_request_ms"],
                        cached_tokens_runs=json.dumps(d["cached_tokens"]),
                        paired_faster_than_stock=sum(
                            p["duration_change_pct"][v] < 0
                            for p in c["aggregate"]["paired"]
                        ),
                    )
                )
    lines = [
        "Experiment 14: aging frequency retention + exact partial-prefix reuse",
        "=" * 72,
        "Both models; four configurations; shifting-prefix and concurrent profiles.",
        "64 fresh-server GPU runs: 48 untraced timed, 16 traced diagnostics.",
        "Three timing rounds/configuration. See PROTOCOL.txt for fixed settings.",
        "",
        "Batch medians in seconds (lower is better):",
        "Model          Profile       Stock     Aging   Partial  Combined  vs stock",
    ]
    for c in comparisons:
        a = c["aggregate"]
        values = a["variants"]
        times = [values[v]["median_duration_ms"] / 1000 for v in VARIANTS]
        change = (times[-1] / times[0] - 1) * 100
        wins = sum(p["duration_change_pct"]["combined"] < 0 for p in a["paired"])
        lines.append(
            f"{c['model'].split('/')[-1]:<14} {c['profile']:<10} "
            + " ".join(f"{n:9.3f}" for n in times)
            + f" {change:+8.2f}%"
        )
        lines.append(f"  Combined beat stock in {wins}/3 paired rounds.")
        lines.append(
            "  Exact combined/stock answer matches by round: "
            + str([p["exact_answer_matches"]["combined"] for p in a["paired"]])
        )
        if c["profile"] == "shift":
            before = values["stock"]["groups"]["probe"]["first_text_ms"]
            after = values["combined"]["groups"]["probe"]["first_text_ms"]
            lines.append(
                f"  Returning-prefix first text: {before:.3f} -> {after:.3f} ms "
                f"({(after / before - 1) * 100:+.2f}%)."
            )
    lines += [
        "",
        "Interpretation:",
        "The policies work together and increase useful reuse on shifting traffic.",
        "More cached tokens do not guarantee lower complete-batch latency.",
        "Use the individual-policy controls and paired rounds, not only medians.",
        "No generally faster combined configuration is established by these runs.",
        "",
        "Validation:",
        "All runs used the local source and Apple Metal GPU. Diagnostic checks",
        "validated allocation budgets, cache API/trace agreement, aging, copies,",
        "running/waiting queues and zero-reference eviction safety.",
        "Existing unittest suites: 11 tests passed; 7 partial-prefix tests also",
        "passed with aging enabled and a four-allocation decay interval.",
        "Both new scripts pass Ruff lint and formatting checks.",
        "",
        "Limits:",
        "Synthetic traffic; three rounds and incomplete launch-position balance.",
        "Only a short warmup, not all request shapes prewarmed. Mixed traffic is",
        "closed-loop with 12 clients, and seed publication is outside the timer.",
        "Shift is sequential. CPU diagnostic profiles include trace overhead,",
        "warmup and seeds. Exact answer agreement is not semantic quality scoring.",
        "Three of 84 small-model shift answers differed with partial reuse; see",
        "raw requests and paired answer-match counts. No quality parity claimed.",
        "",
        "Files:",
        "comparison.csv: compact results; comparison.json: paired/cohort detail.",
        "Qwen3-*/PROFILE/RESULTS.txt: individual-policy comparisons.",
        "evidence-sha256.json: archived raw evidence checksums and original paths.",
    ]
    (DEST / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(audit, indent=2))
    print("Saved", DEST)


if __name__ == "__main__":
    main()
