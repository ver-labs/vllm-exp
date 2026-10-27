"""Run five balanced paired repetitions on one GPU allocation."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
assert (
    json.loads((work / "attention-results-20261026/validation-gate.json").read_text())[
        "status"
    ]
    == "passed"
)
root = work / "attention-results-20261026/benchmark"
root.mkdir(parents=True, exist_ok=False)
lock = json.loads((work / "benchmark-kit-20261026/model-lock.json").read_text())
arms = ["stock", "off", "partial", "attention", "partial_attention"]
plan = []
for repeat in range(5):
    models = list(lock) if repeat % 2 == 0 else list(reversed(lock))
    order = arms[repeat:] + arms[:repeat]
    for model in models:
        for arm in order:
            plan.append({"repeat": repeat, "model": model, "arm": arm})
(root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
assert (
    hashlib.sha256(
        (work / "benchmark-kit-20261026/run_benchmark.py").read_bytes()
    ).hexdigest()
    == "9c173763073791342363e1f84f6095f4f8e5fda523b90721d31de182b8a6d284"
)
source_state = {}
for arm, folder, expected in (
    ("stock", "source-git", "db9527a46873454610df6dbedf79a36d6bf1a7f6"),
    ("candidate", "candidate-src", "08d2c99e8fe5117605d0ca8eaf6b6fd07672d822"),
    ("attention", "attention-src", "ee6e17c5cc817efd06f3a3340b82a49f586321f2"),
):
    source = work / folder
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert commit == expected and not dirty
    with (source / "vllm/_C_stable_libtorch.abi3.so").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == "c5b2ba47283119cf12b3bdbc8cb79a940fecb4fed38ba71c70cccd84b27e20c5"
    with (source / "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so").open("rb") as stream:
        fa2_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert (
        fa2_digest == "db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9"
    )
    source_state[arm] = {
        "commit": commit,
        "tracked_dirty": False,
        "native_sha256": digest,
        "fa2_native_sha256": fa2_digest,
    }
(root / "source-state.json").write_text(json.dumps(source_state, indent=2) + "\n")
records, gpu_uuid, workloads = [], None, {}
for task in plan:
    repeat, model, arm = task["repeat"], task["model"], task["arm"]
    source = work / (
        "source-git"
        if arm == "stock"
        else "attention-src"
        if arm in ("attention", "partial_attention")
        else "candidate-src"
    )
    destination = root / model.split("/")[-1] / f"repeat-{repeat + 1}" / arm
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUDY_")}
    env.update(
        PYTHONPATH=str(source),
        STUDY_PARTIAL_PREFIX=str(int(arm in ("partial", "partial_attention"))),
        STUDY_EVICTION_POLICY="vllm",
        STUDY_LFU_HALF_LIFE_BLOCKS="4",
        STUDY_PARTIAL_MIN_TOKENS="0",
        STUDY_CACHE_IMPL="incremental",
        STUDY_FA2_SINGLE_QUERY_CAUSAL=str(
            int(arm in ("attention", "partial_attention"))
        ),
    )
    command = [
        sys.executable,
        str(work / "benchmark-kit-20261026/run_benchmark.py"),
        "--model",
        model,
        "--revision",
        lock[model]["revision"],
        "--source",
        str(source),
        "--output",
        str(destination),
        "--repeat",
        str(repeat),
        "--arm",
        arm,
    ]
    print("START", repeat + 1, model, arm, flush=True)
    try:
        with destination.with_suffix(".txt").open("w") as log:
            completed = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200
            )
        status = completed.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    records.append({**task, "returncode": status, "output": str(destination)})
    (root / "execution.json").write_text(json.dumps(records, indent=2) + "\n")
    print("END", repeat + 1, model, arm, status, flush=True)
    assert status == 0, f"Benchmark failed; stop and preserve {destination}"
    report = json.loads((destination / "result.json").read_text())
    assert report["status"] == "passed"
    gpu_uuid = gpu_uuid or report["device"]["uuid"]
    assert report["device"]["uuid"] == gpu_uuid
    workloads.setdefault(model, report["workload_sha256"])
    assert workloads[model] == report["workload_sha256"]
    if len(records) % len(arms) != 0:
        continue
    # At the end of each model/repeat block, compare all generated token IDs.
    reference = json.loads((destination.parent / "stock/result.json").read_text())
    for other_arm in arms:
        other = json.loads((destination.parent / other_arm / "result.json").read_text())
        for profile, values in reference["profiles"].items():
            expected = {
                row["id"]: row["output_token_ids"] for row in values["requests"]
            }
            actual = {
                row["id"]: row["output_token_ids"]
                for row in other["profiles"][profile]["requests"]
            }
            assert actual == expected, (
                model,
                repeat,
                other_arm,
                profile,
                "output mismatch",
            )
subprocess.run(
    [sys.executable, str(work / "attention-kit-20261026/summarize.py"), str(root)],
    check=True,
)
