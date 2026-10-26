"""Run a fixed six-engine diagnostic plan; preserve failures and output parity."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
kit = work / "profiling-kit-20261026"
root = work / "profiling-results-20261026"
root.mkdir(exist_ok=False)
lock = json.loads((work / "benchmark-kit-20261026/model-lock.json").read_text())
plan = [
    {"model": model, "arm": arm}
    for model, arms in (
        ("Qwen/Qwen3-0.6B", ("stock", "off", "partial")),
        ("Qwen/Qwen3-1.7B", ("partial", "off", "stock")),
    )
    for arm in arms
]
(root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
state = {}
for folder, expected in (
    ("source-git", "db9527a46873454610df6dbedf79a36d6bf1a7f6"),
    ("candidate-src", "08d2c99e8fe5117605d0ca8eaf6b6fd07672d822"),
):
    path = work / folder
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    assert commit == expected and not dirty
    with (path / "vllm/_C_stable_libtorch.abi3.so").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == "c5b2ba47283119cf12b3bdbc8cb79a940fecb4fed38ba71c70cccd84b27e20c5"
    state[folder] = {"commit": commit, "native_sha256": digest, "tracked_dirty": False}
(root / "source-state.json").write_text(json.dumps(state, indent=2) + "\n")
records, uuid = [], None
for task in plan:
    model, arm = task["model"], task["arm"]
    source = work / ("source-git" if arm == "stock" else "candidate-src")
    dest = root / model.split("/")[-1] / arm
    dest.parent.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUDY_")}
    env.update(
        PYTHONPATH=str(source),
        STUDY_PARTIAL_PREFIX=str(int(arm == "partial")),
        STUDY_EVICTION_POLICY="vllm",
        STUDY_LFU_HALF_LIFE_BLOCKS="4",
        STUDY_PARTIAL_MIN_TOKENS="0",
        STUDY_CACHE_IMPL="incremental",
    )
    command = [
        sys.executable,
        str(kit / "run_diagnostic.py"),
        "--model",
        model,
        "--revision",
        lock[model]["revision"],
        "--source",
        str(source),
        "--output",
        str(dest),
        "--repeat",
        "0",
        "--arm",
        arm,
    ]
    print("PROFILE_START", model, arm, flush=True)
    with dest.with_suffix(".log").open("w") as log:
        completed = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200
        )
    records.append({**task, "returncode": completed.returncode})
    (root / "execution.json").write_text(json.dumps(records, indent=2) + "\n")
    assert completed.returncode == 0, dest
    report = json.loads((dest / "result.json").read_text())
    assert report["status"] == "passed"
    uuid = uuid or report["device"]["uuid"]
    assert report["device"]["uuid"] == uuid
    reference = json.loads(
        (
            work
            / "benchmark-results-20261026/matrix-r1"
            / model.split("/")[-1]
            / "repeat-1/stock/result.json"
        ).read_text()
    )
    assert report["workload_sha256"] == reference["workload_sha256"]
    for name, data in report["profiles"].items():
        mode, profile = name.split("/")
        gold = {
            r["id"]: r["output_token_ids"]
            for r in reference["profiles"][profile]["requests"]
        }
        assert len(data["requests"]) == (3 if mode == "trace" else 96)
        assert all(
            row["output_token_ids"] == gold[row["id"]] for row in data["requests"]
        )
    print("PROFILE_END", model, arm, "passed output parity", flush=True)
(root / "validation.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "engines": 6,
            "same_gpu_uuid": uuid,
            "all_outputs_match_frozen_stock": True,
            "diagnostic_only": True,
        },
        indent=2,
    )
    + "\n"
)
print("ALL_PROFILES_PASSED", flush=True)
