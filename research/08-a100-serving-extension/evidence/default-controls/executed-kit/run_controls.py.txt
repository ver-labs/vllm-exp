"""Exploratory untimed controls for default-mode numerical differences."""

import json
import subprocess
import traceback

import numpy as np
from serve_driver import MODELS, OUT, ROOT, WORK, session, write


def compare(left, right):
    before = json.loads((left / "full-logprobs.json").read_text())
    after = json.loads((right / "full-logprobs.json").read_text())
    vectors = []
    same_ids = True
    for a, b in zip(before, after, strict=True):
        assert a["id"] == b["id"]
        same_ids &= a["token_ids"] == b["token_ids"]
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
            rms = float(np.sqrt(np.mean(diff * diff)))
            vectors.append(
                {
                    "id": a["id"],
                    "step": step,
                    "same_history": a["token_ids"][:step] == b["token_ids"][:step],
                    "exact": bool(np.array_equal(p, q)),
                    "max_abs": float(np.max(np.abs(diff))),
                    "rms": rms,
                    "within_limits": bool(
                        np.allclose(p, q, atol=0.0625, rtol=0.005) and rms <= 0.02
                    ),
                }
            )
    natural = {}
    for name, path in (("left", left), ("right", right)):
        data = json.loads((path / "result.json").read_text())
        natural[name] = {r["id"]: r["token_ids"] for r in data["natural"]["requests"]}
    assert natural["left"].keys() == natural["right"].keys()
    return {
        "greedy_four_step_ids_match": same_ids,
        "natural_pairs": len(natural["left"]),
        "natural_exact_pairs": sum(
            natural["left"][k] == natural["right"][k] for k in natural["left"]
        ),
        "vectors": vectors,
        "passes_declared_limits": same_ids and all(v["within_limits"] for v in vectors),
    }


def main():
    root = OUT / "default-controls"
    root.mkdir(exist_ok=False)
    snapshot = root / "executed-kit"
    snapshot.mkdir()
    for name in ("run_controls.py", "serve_driver.py", "http_workload.py"):
        (snapshot / (name + ".txt")).write_bytes((ROOT / name).read_bytes())
    for folder, commit in (
        ("source-git", "db9527a46873454610df6dbedf79a36d6bf1a7f6"),
        ("attention-src", "ee6e17c5cc817efd06f3a3340b82a49f586321f2"),
    ):
        actual = subprocess.check_output(
            ["git", "-C", str(WORK / folder), "rev-parse", "HEAD"], text=True
        ).strip()
        assert actual == commit
    report = {}
    for model in MODELS:
        report[model] = {}
        bank = json.loads((OUT / "gate" / f"{model}-workload.json").read_text())
        for name, arm in (("stock-repeat", "stock"), ("features-off", "off")):
            try:
                session(
                    model, "default_graph", arm, root / model / name, bank, gate=True
                )
            except Exception:  # noqa: BLE001 -- retain diagnostic failures
                report[model][name] = {"error": traceback.format_exc()}
        paths = {
            "stock": OUT / "gate" / model / "default_graph/stock",
            "optimized": OUT / "gate" / model / "default_graph/attention",
            "stock-repeat": root / model / "stock-repeat",
            "features-off": root / model / "features-off",
        }
        for a, b in (
            ("stock", "stock-repeat"),
            ("stock", "features-off"),
            ("features-off", "optimized"),
        ):
            try:
                report[model][f"{a}-vs-{b}"] = compare(paths[a], paths[b])
            except Exception:  # noqa: BLE001 -- no change to primary gate
                report[model][f"{a}-vs-{b}"] = {"error": traceback.format_exc()}
        write(root / "diagnostics.json", report)
        print("DEFAULT_CONTROL", model, "complete", flush=True)
    print(
        "Exploratory controls complete; original gate decisions unchanged.", flush=True
    )


if __name__ == "__main__":
    main()
