"""Require direct attention, model logits, KV and lifecycle checks before timing."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

work = Path.home() / "vllm-research-20261024"
kit = work / "attention-kit-20261026"
root = work / "attention-results-20261026"
root.mkdir(exist_ok=False)
source = work / "attention-src"
expected = "ee6e17c5cc817efd06f3a3340b82a49f586321f2"
assert (
    subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    == expected
)
assert not subprocess.check_output(
    ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
    text=True,
)
env = {k: v for k, v in os.environ.items() if not k.startswith("STUDY_")}
env.update(
    PYTHONPATH=str(source),
    STUDY_FA2_SINGLE_QUERY_CAUSAL="1",
    ATTENTION_GATE_OUTPUT=str(root / "attention-gate.json"),
)
subprocess.run([sys.executable, str(kit / "attention_gate.py")], env=env, check=True)
lock = json.loads((work / "benchmark-kit-20261026/model-lock.json").read_text())
inventory = json.loads((work / "correctness-file-inventory.json").read_text())
refs = {}
for model in lock:
    model_root = root / "correctness" / model.split("/")[-1]
    model_root.mkdir(parents=True)
    reference = work / "correctness-results/matrix-r2" / model.split("/")[-1] / "stock"
    for name in ("result.json", "raw-logits.pt"):
        path = reference / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        relative = str(path.relative_to(work))
        assert digest == inventory[relative]["sha256"]
        refs[relative] = digest
    (model_root / "stock").symlink_to(reference, target_is_directory=True)
    for arm in ("attention", "partial_attention"):
        local_env = dict(env)
        local_env.update(
            STUDY_PARTIAL_PREFIX=str(int(arm == "partial_attention")),
            STUDY_EVICTION_POLICY="vllm",
            STUDY_LFU_HALF_LIFE_BLOCKS="4",
            STUDY_PARTIAL_MIN_TOKENS="0",
            STUDY_CACHE_IMPL="incremental",
        )
        command = [
            sys.executable,
            str(kit / "run_correctness.py"),
            "--model",
            model,
            "--revision",
            lock[model]["revision"],
            "--source",
            str(source),
            "--output",
            str(model_root / arm),
            "--download-dir",
            str(work / "model-cache"),
        ]
        print("GATE_START", model, arm, flush=True)
        with (model_root / (arm + ".log")).open("w") as log:
            subprocess.run(
                command,
                env=local_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                check=True,
            )
        print("GATE_END", model, arm, flush=True)
subprocess.run(
    [sys.executable, str(kit / "compare_arms.py"), str(root / "correctness")],
    check=True,
)
(root / "validation-gate.json").write_text(
    json.dumps(
        {
            "status": "passed",
            "optimized_source_commit": expected,
            "reference_hashes_verified": refs,
            "tolerances_unchanged": True,
        },
        indent=2,
    )
    + "\n"
)
print("ALL_ATTENTION_GATES_PASSED", flush=True)
