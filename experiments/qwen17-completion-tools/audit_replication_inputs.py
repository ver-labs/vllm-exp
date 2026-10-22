from pathlib import Path
import json

root = Path("/Users/nandanadileep/projects/vllm/vllm/vllm-exp")
progress = json.loads((root / "results/missing-qwen17/progress.json").read_text())
paths = {
    "01": "01-prefix-scheduling/evidence/batches-20261020-195946/summary.json",
    "02-temporal": "02-frequency-eviction/evidence/summary.json",
    "02-ay": "02-frequency-eviction/evidence-ay/summary.json",
    "03": "03-nonprefix-repair/evidence-vllm-ay/summary.json",
    "04": "04-partial-prefix/evidence/summary.json",
    "05": "05-skip-continue/evidence/summary.json",
    "06": "06-combined/evidence/summary.json",
    "10": "10-cache-overhead/evidence/summary.json",
}


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += differences(a.get(k), b.get(k), prefix + "." + k)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [prefix + ".length"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += differences(x, y, prefix + f"[{i}]")
        return out
    return [] if a == b else [prefix]


rows = []
for stage in progress["stages"]:
    old = json.loads((root / "experiments" / paths[stage["stage"]]).read_text())
    new = json.loads(Path(stage["frozen_summary"]).read_text())
    old_records = old["runs"] + old.get("diagnostics", [])
    new_records = new["runs"] + new.get("diagnostics", [])
    profiles = {r.get("profile", "default") for r in new_records}
    for profile in sorted(profiles):
        left = next(r for r in old_records if r.get("profile", "default") == profile)
        right = next(r for r in new_records if r.get("profile", "default") == profile)
        a = json.loads(Path(left["result"]).read_text())
        b = json.loads(Path(right["result"]).read_text())
        assert "manifest" in a and "manifest" in b
        delta = differences(a["manifest"], b["manifest"])
        rows.append(
            dict(
                stage=stage["stage"],
                profile=profile,
                differing_manifest_fields=delta,
                unchanged_except_model=all(d.endswith(".model") for d in delta),
            )
        )
dest = root / "results/missing-qwen17/input-manifest-audit.json"
dest.write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
