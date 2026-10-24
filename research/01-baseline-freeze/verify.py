"""Verify isolated source imports, controls, metadata behavior and reset state."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def execute(name, tree, command, settings):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUDY_")}
    env.update(
        PYTHONPATH=os.pathsep.join((str(tree), str(HERE))),
        PYTHONDONTWRITEBYTECODE="1",
        VLLM_PLUGINS="",
        STUDY_PARTIAL_PREFIX="0",
        STUDY_EVICTION_POLICY="vllm",
        STUDY_CACHE_IMPL="incremental",
    )
    env.update(settings)
    completed = subprocess.run(
        [sys.executable, *command],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    log = HERE / "receipts" / f"{name}.txt"
    log.write_text(
        json.dumps(dict(tree=str(tree), command=command, settings=settings))
        + "\nSTDOUT\n"
        + completed.stdout
        + "\nSTDERR\n"
        + completed.stderr
        + f"\nEXIT {completed.returncode}\n"
    )
    if completed.returncode:
        raise RuntimeError(f"{name} failed; see {log}")
    print("PASS", name, flush=True)
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    probes = {}
    for name, tree, settings in (
        ("stock", args.stock, {}),
        ("candidate-off", args.candidate, {}),
        ("candidate-aging", args.candidate, {"STUDY_EVICTION_POLICY": "aged_lfu"}),
        ("candidate-partial", args.candidate, {"STUDY_PARTIAL_PREFIX": "1"}),
        (
            "candidate-combined",
            args.candidate,
            {"STUDY_PARTIAL_PREFIX": "1", "STUDY_EVICTION_POLICY": "aged_lfu"},
        ),
    ):
        tree = tree.resolve()
        text = execute(name, tree, [str(HERE / "probe.py")], settings)
        result = json.loads(
            next(x[11:] for x in text.splitlines() if x.startswith("PROBE_JSON "))
        )
        assert Path(result["source"]).is_relative_to(tree)
        probes[name] = result
    native = {k: v for k, v in probes["stock"].items() if k != "source"}
    disabled = {k: v for k, v in probes["candidate-off"].items() if k != "source"}
    assert native == disabled, (native, disabled)
    for name, settings in (
        ("candidate-tests", {}),
        (
            "combined-tests",
            {"STUDY_EVICTION_POLICY": "aged_lfu", "STUDY_LFU_HALF_LIFE_BLOCKS": "4"},
        ),
    ):
        execute(
            name,
            args.candidate.resolve(),
            ["-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"],
            settings,
        )
    result = dict(
        stock_equals_disabled_candidate=True,
        probes=probes,
        cuda_validated=False,
        model_quality_validated=False,
        gpu_benchmarks_run=False,
    )
    result["receipt_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((HERE / "receipts").glob("*.txt"))
    }
    (HERE / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("VALIDATED", HERE / "validation.json")


if __name__ == "__main__":
    main()
