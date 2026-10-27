"""Compare identical-context logits across policy arms, retaining numerical failures."""

import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
summary = {
    "status": "passed",
    "models": {},
    "tolerance": {"atol": 0.0625, "rtol": 0.005, "rms_max": 0.02},
}
for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    stock = json.loads((model_dir / "stock/result.json").read_text())
    baseline = torch.load(
        model_dir / "stock/raw-logits.pt", weights_only=True, map_location="cpu"
    )
    model = {"arms": {}, "model_revision": stock["revision"]}
    for arm in ("stock", "attention", "partial_attention"):
        report = json.loads((model_dir / arm / "result.json").read_text())
        logits = torch.load(
            model_dir / arm / "raw-logits.pt", weights_only=True, map_location="cpu"
        )
        result = {
            "status": "passed",
            "vectors_compared": 0,
            "non_bitwise_vectors": 0,
            "argmax_differences": 0,
            "max_abs": 0.0,
            "max_rms": 0.0,
            "failures": [],
        }
        if report["status"] != "passed" or set(logits) != set(baseline):
            result["failures"].append("Arm failed or trace-label coverage differs")
        for label in sorted(set(logits) & set(baseline)):
            if set(logits[label]) != set(baseline[label]):
                result["failures"].append(
                    {"label": label, "error": "context-position mismatch"}
                )
                continue
            left_ids = stock["outputs"].get(label, {}).get("tokens")
            right_ids = report["outputs"].get(label, {}).get("tokens")
            if left_ids != right_ids:
                result["failures"].append(
                    {
                        "label": label,
                        "error": "token IDs differ; later contexts are not comparable",
                    }
                )
                continue
            for position, actual in logits[label].items():
                expected = baseline[label][position]
                delta = expected - actual
                maximum = float(delta.abs().max())
                rms = float(delta.square().mean().sqrt())
                result["vectors_compared"] += 1
                result["non_bitwise_vectors"] += int(not torch.equal(expected, actual))
                result["argmax_differences"] += int(
                    int(expected.argmax()) != int(actual.argmax())
                )
                result["max_abs"] = max(result["max_abs"], maximum)
                result["max_rms"] = max(result["max_rms"], rms)
                if (
                    not torch.allclose(expected, actual, atol=0.0625, rtol=0.005)
                    or rms > 0.02
                ):
                    result["failures"].append(
                        {
                            "label": label,
                            "position": position,
                            "max_abs": maximum,
                            "rms": rms,
                        }
                    )
        if result["failures"]:
            result["status"] = summary["status"] = "failed"
        result["copy_events"] = len(report["copies"])
        result["copy_views_bitwise_checked"] = sum(
            e["tensor_views_checked"] for e in report["copies"]
        )
        result["preemption_events"] = len(report["preemptions"])
        result["lifecycle"] = report["lifecycle"]
        result["kv_recompute_comparison"] = report.get("kv_recompute_comparison")
        model["arms"][arm] = result
    summary["models"][model_dir.name] = model
(root / "cross-arm-validation.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
raise SystemExit(0 if summary["status"] == "passed" else 1)
