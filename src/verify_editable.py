"""Verify editable scheduler imports and preservation of the Metal fallback."""

import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[1]
checkout = root / "vllm-src"
baseline = json.loads((root / "configs/metal-baseline.json").read_text())
fallback_site = Path(baseline["environment"]) / "lib/python3.12/site-packages"
site = Path(sys.prefix) / "lib/python3.12/site-packages"


def versions(distributions):
    return {d.metadata["Name"].lower().replace("_", "-"): d.version
            for d in distributions}


expected = baseline["packages"]
fallback_versions = versions(metadata.distributions(path=[str(fallback_site)]))
assert fallback_versions == expected, "Fallback package versions changed"
experiment_versions = versions(metadata.distributions())
for name, version in expected.items():
    assert experiment_versions[name] == version, (name, version, experiment_versions[name])

for relative, digest in baseline["native_sha256"].items():
    fallback = fallback_site / relative
    assert hashlib.sha256(fallback.read_bytes()).hexdigest() == digest, fallback
    target = checkout / relative if relative.startswith("vllm/") else site / relative
    assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, target

import vllm

scheduler = importlib.import_module("vllm.v1.core.sched.scheduler")
expected_scheduler = checkout / "vllm/v1/core/sched/scheduler.py"
assert Path(vllm.__file__).resolve() == checkout / "vllm/__init__.py"
assert Path(scheduler.__file__).resolve() == expected_scheduler
assert Path(inspect.getfile(scheduler.Scheduler)).resolve() == expected_scheduler
original = subprocess.check_output(
    ["git", "-C", str(checkout), "show", "HEAD:vllm/v1/core/sched/scheduler.py"]
)
assert expected_scheduler.read_bytes() == original, "Scheduler source changed"
source_diff = subprocess.check_output(["git", "-C", str(checkout), "diff", "HEAD"])
assert not source_diff, "Tracked checkout files changed"
distribution = metadata.distribution("vllm")
provenance = json.loads(distribution.read_text("direct_url.json"))
assert provenance["dir_info"]["editable"] is True
assert provenance["url"] == checkout.as_uri()
report = {
    "environment": sys.prefix,
    "fallback_environment": baseline["environment"],
    "source": str(checkout),
    "tag": subprocess.check_output(
        ["git", "-C", str(checkout), "describe", "--tags", "--exact-match"], text=True
    ).strip(),
    "commit": subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip(),
    "version": distribution.version,
    "editable_provenance": provenance,
    "scheduler_source": str(expected_scheduler),
    "scheduler_sha256": hashlib.sha256(original).hexdigest(),
    "original_packages_preserved": len(expected),
    "native_files_preserved": baseline["native_sha256"],
    "added_build_packages": {name: version for name, version in experiment_versions.items()
                             if name not in expected},
    "tracked_source_changes": False,
    "experiments_run": False,
    "build_settings": {
        "VLLM_USE_PRECOMPILED": "1",
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_PRECOMPILED_WHEEL_LOCATION": str(
            root / ".artifacts/vllm-0.31.0+cpu-cp312-cp312-macosx_11_0_arm64.whl"
        ),
        "dependency_resolution": False,
        "build_isolation": False,
    },
}
(root / "configs/editable-setup.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(report, indent=2, sort_keys=True))
