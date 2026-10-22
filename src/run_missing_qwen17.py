"""Run the missing pinned Qwen3-1.7B replications, checkpointing each stage."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
STAGES = [
    (
        "01",
        "src/run_prefix_batches.py",
        ["--rounds", "3", "--candidates", "25"],
        "01-prefix-scheduling",
        "evidence",
    ),
    (
        "02-temporal",
        "src/run_cache_eviction_batches.py",
        ["--rounds", "3"],
        "02-frequency-eviction",
        "evidence",
    ),
    (
        "02-ay",
        "src/run_cache_eviction_ay.py",
        ["--rounds", "3"],
        "02-frequency-eviction",
        "evidence-ay",
    ),
    (
        "03",
        "src/run_nonprefix_vllm_batches.py",
        ["--rounds", "3"],
        "03-nonprefix-repair",
        "evidence-vllm-ay",
    ),
    (
        "04",
        "src/run_partial_prefix_batches.py",
        ["--rounds", "3"],
        "04-partial-prefix",
        "evidence",
    ),
    (
        "05",
        "src/run_skip_continue_batches.py",
        ["--rounds", "3"],
        "05-skip-continue",
        "evidence",
    ),
    ("06", "src/run_combined_batches.py", ["--rounds", "3"], "06-combined", "evidence"),
    ("10", "src/run_cache_overhead.py", [], "10-cache-overhead", "evidence"),
]


def archive_stage(stage, summary_path, provenance):
    label, _, _, experiment, evidence_name = stage
    data = json.loads(summary_path.read_text())
    assert data["complete"]
    records = data.get("diagnostics", []) + data["runs"]
    assert records
    raw_paths = set(summary_path.parent.rglob("*"))

    def collect(value):
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, str) and value.startswith(str(ROOT)):
            path = Path(value)
            if path.is_dir():
                raw_paths.update(path.rglob("*"))
            elif path.is_file():
                raw_paths.add(path)

    collect(data)
    for record in records:
        path = Path(record["result"])
        raw = json.loads(path.read_text())
        assert raw.get("valid_run", True)
        manifest = raw.get("manifest", {})
        assert manifest.get("model", raw.get("model")) == MODEL, (
            path,
            manifest.get("model"),
            raw.get("model"),
        )
        collect(raw)
        # Launchers print their separate physical server directory.
        log_path = Path(record.get("server_log", record.get("launcher_log", "")))
        assert log_path.is_file()
        log = log_path.read_text()
        assert "MLX device set to: Device(gpu, 0)" in log
        assert REVISION in log, "Server did not load the pinned model path"
        for line in log.splitlines():
            if line.startswith("Server results: "):
                raw_paths.update(
                    Path(line.removeprefix("Server results: ").strip()).rglob("*")
                )
    dest = ROOT / "experiments" / experiment / "qwen3-1.7B" / evidence_name
    dest.mkdir(parents=True, exist_ok=True)
    data["replication_provenance"] = provenance
    data["original_summary"] = str(summary_path)
    (dest / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    for name in ("per-request.csv", "per-request-latency.csv"):
        path = summary_path.parent / name
        if path.exists():
            shutil.copy2(path, dest / name)
    inventory = []
    for path in sorted(raw_paths):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        inventory.append(
            dict(
                path=str(path.relative_to(ROOT)),
                bytes=path.stat().st_size,
                sha256=digest.hexdigest(),
            )
        )
    (dest / "raw-sha256.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(
        f"ARCHIVED {label}: {len(records)} GPU runs, {len(inventory)} raw files",
        flush=True,
    )
    return dict(
        stage=label,
        summary=str(summary_path),
        frozen_summary=str(dest / "summary.json"),
        gpu_runs=len(records),
        timed_runs=len(data["runs"]),
        raw_files=len(inventory),
    )


def main():
    os.environ.update(
        STUDY_MODEL=MODEL,
        STUDY_MODEL_REVISION=REVISION,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        STUDY_ADAPTIVE_PREFILL="0",
        STUDY_PARTIAL_PREFIX="0",
        STUDY_PARTIAL_MIN_TOKENS="0",
        STUDY_SKIP_CONTINUE="0",
        STUDY_WAITING_POLICY="fcfs",
        STUDY_EVICTION_POLICY="vllm",
        STUDY_CACHE_IMPL="reference",
        STUDY_CACHE_PROFILE="0",
    )
    os.environ["STUDY_MODEL_PATH"] = snapshot_download(
        MODEL, revision=REVISION, local_files_only=True
    )
    folder = ROOT / "results/missing-qwen17"
    folder.mkdir(parents=True, exist_ok=True)
    progress_path = folder / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        assert progress["model_revision"] == REVISION
    else:
        progress = dict(
            model=MODEL,
            model_revision=REVISION,
            started=datetime.now().isoformat(),
            source_checkpoint=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT / "vllm-src", text=True
            ).strip(),
            implementation_mode=(
                "reference for experiments01-06; reference/incremental in10"
            ),
            stages=[],
            complete=False,
        )
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    for stage in STAGES:
        label, script, args, _, _ = stage
        if any(s["stage"] == label for s in progress["stages"]):
            continue
        log_path = folder / f"{label}-{datetime.now():%Y%m%d-%H%M%S}.log"
        print(f"STAGE {label} BEGIN", flush=True)
        summary_path = None
        with log_path.open("w") as log:
            child = subprocess.Popen(
                [sys.executable, script, *args],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("START", "DONE", "SUMMARY", "NOTE")):
                    print(f"[{label}] {line.rstrip()}", flush=True)
                if line.startswith("SUMMARY "):
                    summary_path = Path(line.removeprefix("SUMMARY ").strip())
            code = child.wait()
        if code:
            progress["last_failure"] = dict(
                stage=label,
                log=str(log_path),
                code=code,
                summary=str(summary_path) if summary_path else None,
            )
            progress_path.write_text(json.dumps(progress, indent=2) + "\n")
            raise RuntimeError(f"Stage{label} failed; see {log_path}")
        assert summary_path
        record = archive_stage(
            stage,
            summary_path,
            {k: v for k, v in progress.items() if k not in ("stages", "last_failure")},
        )
        progress["stages"].append(record)
        progress.pop("last_failure", None)
        progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    progress["complete"] = True
    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
    print("ALL MISSING QWEN3-1.7B COMPARISONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
