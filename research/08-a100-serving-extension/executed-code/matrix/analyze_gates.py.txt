"""Describe all distribution differences without overriding the frozen gate."""

import json
from pathlib import Path

import numpy as np

work = Path.home() / "vllm-research-20261024"
root = work / "serving-extension-results-20261103"
gates = json.loads((root / "gate/validation.json").read_text())
report = {}
for key, gate in gates.items():
    dest = root / "gate" / key
    left_path, right_path = (
        dest / a / "full-logprobs.json" for a in ("stock", "attention")
    )
    if not left_path.exists() or not right_path.exists():
        report[key] = {"gate_status": gate["status"], "data_complete": False}
        continue
    before, after = (
        json.loads(left_path.read_text()),
        json.loads(right_path.read_text()),
    )
    vectors = []
    for a, b in zip(before, after, strict=True):
        assert a["id"] == b["id"]
        for step, (x, y) in enumerate(
            zip(
                a["logprobs"]["top_logprobs"],
                b["logprobs"]["top_logprobs"],
                strict=True,
            )
        ):
            assert x.keys() == y.keys() and len(x) == a["vocab_size"]
            keys = sorted(x)
            p, q = np.array([x[k] for k in keys]), np.array([y[k] for k in keys])
            diff = p - q
            vectors.append(
                {
                    "id": a["id"],
                    "step": step,
                    "max_abs": float(np.max(np.abs(diff))),
                    "rms": float(np.sqrt(np.mean(diff * diff))),
                    "exact": bool(np.array_equal(p, q)),
                    "argmax_match": bool(np.argmax(p) == np.argmax(q)),
                    "same_generated_history": a["token_ids"][:step]
                    == b["token_ids"][:step],
                }
            )
    natural = {}
    for arm in ("stock", "attention"):
        result = json.loads((dest / arm / "result.json").read_text())
        natural[arm] = {
            r["id"]: r for r in result.get("natural", {}).get("requests", [])
        }
    common = natural["stock"].keys() & natural["attention"].keys()
    natural_pairs = [(natural["stock"][k], natural["attention"][k]) for k in common]
    report[key] = {
        "gate_status": gate["status"],
        "data_complete": True,
        "greedy_ids_match": all(
            a["token_ids"] == b["token_ids"] for a, b in zip(before, after, strict=True)
        ),
        "vectors": vectors,
        "natural_response_pairs": len(natural_pairs),
        "natural_data_complete": len(common) == 8,
        "natural_exact_output_pairs": sum(
            a["token_ids"] == b["token_ids"] for a, b in natural_pairs
        ),
        "note": "Later steps compare distributions on each arm's own history; "
        "if token IDs diverge, they are not teacher-forced comparisons.",
    }
(root / "gate-diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
print("All-vector diagnostic saved; no frozen gate decisions were changed.", flush=True)
