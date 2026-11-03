"""Gate longer contexts, then run a fixed six-repeat balanced A100 sweep."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gate", action="store_true")
args = parser.parse_args()
work = Path.home() / "vllm-research-20261024"
kit = work / "attention-sweep-kit-20261027"
base = work / "attention-sweep-results-20261027"
root = base / ("gate" if args.gate else "matrix")
root.mkdir(parents=True, exist_ok=False)
lock = json.loads((work / "benchmark-kit-20261026/model-lock.json").read_text())
folders = {"stock": "source-git", "off": "candidate-src", "attention": "attention-src"}
expected = {
    "stock": "db9527a46873454610df6dbedf79a36d6bf1a7f6",
    "off": "08d2c99e8fe5117605d0ca8eaf6b6fd07672d822",
    "attention": "ee6e17c5cc817efd06f3a3340b82a49f586321f2",
}
states = {}
for arm, folder in folders.items():
    source = work / folder
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == expected[arm]
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    native = {}
    for relative, digest in (
        (
            "vllm/_C_stable_libtorch.abi3.so",
            "c5b2ba47283119cf12b3bdbc8cb79a940fecb4fed38ba71c70cccd84b27e20c5",
        ),
        (
            "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
            "db8b4687ba2824445e40b574886408a010596e7bfa1fbe1af4d4cf8d40af8cf9",
        ),
    ):
        with (source / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == digest
        native[relative] = actual
    states[arm] = {"commit": commit, "tracked_dirty": False, "native": native}
(root / "source-state.json").write_text(json.dumps(states, indent=2) + "\n")
runtime = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in kit.glob("*.py")}
(root / "kit-sha256.json").write_text(json.dumps(runtime, indent=2) + "\n")


def environment(arm):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STUDY_")}
    env.update(
        PYTHONPATH=str(work / folders[arm]),
        STUDY_PARTIAL_PREFIX="0",
        STUDY_EVICTION_POLICY="vllm",
        STUDY_LFU_HALF_LIFE_BLOCKS="4",
        STUDY_PARTIAL_MIN_TOKENS="0",
        STUDY_CACHE_IMPL="incremental",
        STUDY_FA2_SINGLE_QUERY_CAUSAL=str(int(arm == "attention")),
    )
    return env


if args.gate:
    env = environment("attention")
    env["ATTENTION_GATE_OUTPUT"] = str(root / "attention-gate.json")
    subprocess.run(
        [sys.executable, str(kit / "attention_gate.py")], env=env, check=True
    )
    plan = [
        {"repeat": 0, "model": model, "arm": arm}
        for model in lock
        for arm in ("stock", "attention")
    ]
else:
    assert json.loads((base / "gate/validation.json").read_text())["status"] == "passed"
    orders = list(itertools.permutations(("stock", "off", "attention")))
    plan = [
        {"repeat": repeat, "model": model, "arm": arm}
        for repeat, order in enumerate(orders)
        for model in (list(lock) if repeat % 2 == 0 else list(reversed(lock)))
        for arm in order
    ]
(root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
records, uuid, workloads = [], None, {}
for task in plan:
    for name, digest in runtime.items():
        assert hashlib.sha256((kit / name).read_bytes()).hexdigest() == digest
    model, arm, repeat = task["model"], task["arm"], task["repeat"]
    dest = root / model.split("/")[-1] / f"repeat-{repeat + 1}" / arm
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(kit / "run_case_matrix.py"),
        "--model",
        model,
        "--revision",
        lock[model]["revision"],
        "--source",
        str(work / folders[arm]),
        "--output",
        str(dest),
        "--repeat",
        str(repeat),
        "--arm",
        arm,
    ]
    if args.gate:
        command.append("--gate")
    print("START_SWEEP", task, flush=True)
    try:
        with dest.with_suffix(".log").open("w") as log:
            result = subprocess.run(
                command,
                env=environment(arm),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=2400,
            )
        status = result.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    records.append({**task, "returncode": status})
    (root / "execution.json").write_text(json.dumps(records, indent=2) + "\n")
    assert status == 0, dest
    report = json.loads((dest / "result.json").read_text())
    assert report["status"] == "passed"
    uuid = uuid or report["device"]["uuid"]
    assert uuid == report["device"]["uuid"]
    workloads.setdefault(model, report["workload_sha256"])
    assert workloads[model] == report["workload_sha256"]
    print("END_SWEEP", task, "passed", flush=True)
    if not args.gate and len(records) % 3 == 0:
        gold = json.loads((dest.parent / "stock/result.json").read_text())
        for other in ("stock", "off", "attention"):
            candidate = json.loads((dest.parent / other / "result.json").read_text())
            assert set(candidate["cases"]) == set(gold["cases"])
            for case in gold["cases"]:
                expected_ids = {
                    r["id"]: r["output_token_ids"]
                    for r in gold["cases"][case]["requests"]
                }
                actual_ids = {
                    r["id"]: r["output_token_ids"]
                    for r in candidate["cases"][case]["requests"]
                }
                assert actual_ids == expected_ids, (
                    task,
                    other,
                    case,
                    "output mismatch",
                )
if args.gate:
    import torch

    checks = {}
    for model in lock:
        folder = root / model.split("/")[-1] / "repeat-1"
        gold = torch.load(
            folder / "stock/raw-logits.pt", weights_only=True, map_location="cpu"
        )
        actual = torch.load(
            folder / "attention/raw-logits.pt", weights_only=True, map_location="cpu"
        )
        assert set(gold) == set(actual) and len(gold) == 27
        left = json.loads((folder / "stock/result.json").read_text())
        right = json.loads((folder / "attention/result.json").read_text())
        for case in left["cases"]:
            assert {
                r["id"]: r["output_token_ids"] for r in left["cases"][case]["requests"]
            } == {
                r["id"]: r["output_token_ids"] for r in right["cases"][case]["requests"]
            }
        count, different, max_abs, max_rms = 0, 0, 0.0, 0.0
        for label in gold:
            assert set(gold[label]) == set(actual[label]) and len(gold[label]) == 4
            for position, expected_values in gold[label].items():
                values = actual[label][position]
                assert (
                    torch.isfinite(values).all()
                    and torch.isfinite(expected_values).all()
                )
                delta = expected_values - values
                rms = float(delta.square().mean().sqrt())
                assert (
                    torch.allclose(values, expected_values, atol=0.0625, rtol=0.005)
                    and rms <= 0.02
                )
                assert int(values.argmax()) == int(expected_values.argmax())
                count += 1
                different += int(not torch.equal(values, expected_values))
                max_abs, max_rms = (
                    max(max_abs, float(delta.abs().max())),
                    max(max_rms, rms),
                )
        assert count == 108
        checks[model] = {
            "vectors": count,
            "non_exact_value_vectors": different,
            "max_abs": max_abs,
            "max_rms": max_rms,
            "outputs_match": True,
        }
    (root / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "models": checks,
                "same_gpu_uuid": uuid,
                "tolerance": {"atol": 0.0625, "rtol": 0.005, "rms": 0.02},
            },
            indent=2,
        )
        + "\n"
    )
else:
    assert len(records) == 36
    (root / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "fresh_engines": 36,
                "same_gpu_uuid": uuid,
                "outputs_match_stock": True,
            },
            indent=2,
        )
        + "\n"
    )
print("SWEEP_COMPLETE", "gate" if args.gate else "matrix", flush=True)
