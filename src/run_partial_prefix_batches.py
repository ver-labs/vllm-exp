"""Experiment04: exact partial-prefix reuse against the original vLLM A-Y baseline."""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from huggingface_hub import snapshot_download
from run_nonprefix_vllm_batches import run_one
from run_prefix_batches import ROOT
from transformers import AutoTokenizer


def probe(variant):
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
        local_files_only=True,
    )
    prefix = tokenizer.encode(
        (
            "The library records book reservations and reminds members "
            "to collect them. "
            * 8
        ),
        add_special_tokens=False,
    )[:31]
    a = prefix + [tokenizer.encode(" Yes", add_special_tokens=False)[-1]]
    b = prefix + [tokenizer.encode(" No", add_special_tokens=False)[-1]]
    assert len(a) == len(b) == 32 and a[-1] != b[-1]

    def reset():
        with urlopen(
            Request(
                "http://127.0.0.1:8001/reset_prefix_cache", data=b"", method="POST"
            ),
            timeout=30,
        ) as response:
            assert json.load(response)["success"], "Cache reset failed"

    def send(tokens, limit):
        data = json.dumps(
            dict(
                model="Qwen/Qwen3-0.6B",
                prompt=tokens,
                max_tokens=limit,
                temperature=0,
                ignore_eos=True,
            )
        ).encode()
        start = time.perf_counter()
        with urlopen(
            Request(
                "http://127.0.0.1:8001/v1/completions",
                data=data,
                headers={"Content-Type": "application/json"},
            ),
            timeout=120,
        ) as response:
            result = json.load(response)
        return dict(
            usage=result["usage"],
            text=result["choices"][0]["text"],
            total_ms=(time.perf_counter() - start) * 1000,
        )

    reset()
    source = send(a, 1)
    reused = send(b, 8)
    reset()
    cold = send(b, 8)
    reset()
    cached = reused["usage"]["prompt_tokens_details"]["cached_tokens"]
    expected = 31 if variant == "partial" else 16
    assert cached == expected, (variant, cached, expected)
    assert cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    return dict(
        source_token_ids=a,
        target_token_ids=b,
        source=source,
        reused=reused,
        cold=cold,
        expected_reused_tokens=expected,
        answers_identical=reused["text"] == cold["text"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 0:
        parser.error("rounds must be nonnegative")
    folder = ROOT / "results/partial-prefix" / f"batches-{datetime.now():%Y%m%d-%H%M%S}"
    folder.mkdir(parents=True)
    d = dict(experiment="04-partial-prefix", diagnostics=[], runs=[], complete=False)
    variants = ["baseline", "fine_hash", "partial"]
    path = folder / "summary.json"

    def run(variant, mode, label):
        return run_one(
            folder,
            variant,
            mode,
            label,
            launcher="src/serve_partial_prefix_comparison.sh",
            probe=probe if mode == "trace" else None,
        )

    try:
        for variant in ["partial", "baseline", "fine_hash"]:
            d["diagnostics"].append(run(variant, "trace", "diagnostic"))
            path.write_text(json.dumps(d, indent=2) + "\n")
        for index in range(args.rounds):
            for variant in variants[index % 3 :] + variants[: index % 3]:
                d["runs"].append(run(variant, "bench", index + 1))
                path.write_text(json.dumps(d, indent=2) + "\n")
        all_data = [
            json.loads(Path(r["result"]).read_text())
            for r in d["diagnostics"] + d["runs"]
        ]
        assert all(x["manifest"] == all_data[0]["manifest"] for x in all_data)
        assert all(
            x["batch_summary"]["successful_output_tokens"] == 1984 for x in all_data
        )
        d["manifest"] = all_data[0]["manifest"]
        d["aggregate"] = {}
        d["per_request"] = {}
        csv_rows = []
        for variant in variants:
            runs = [r for r in d["runs"] if r["variant"] == variant]
            if not runs:
                continue
            ds = [json.loads(Path(r["result"]).read_text()) for r in runs]
            d["aggregate"][variant] = dict(
                candidate_duration_median_ms=statistics.median(
                    r["candidate_summary"]["duration_ms"] for r in runs
                ),
                candidate_duration_range_ms=[
                    min(r["candidate_summary"]["duration_ms"] for r in runs),
                    max(r["candidate_summary"]["duration_ms"] for r in runs),
                ],
                throughput_median_tokens_s=statistics.median(
                    r["candidate_summary"]["output_tokens_per_second"] for r in runs
                ),
                median_first_text_ms=statistics.median(
                    r["candidate_summary"]["first_text_ms"]["median"] for r in runs
                ),
                median_request_latency_ms=statistics.median(
                    r["candidate_summary"]["total_ms"]["median"] for r in runs
                ),
                cached_tokens_per_run=[x["candidate_cached_prompt_tokens"] for x in ds],
            )
            d["per_request"][variant] = {}
            for label in d["manifest"]["candidate_order"]:
                rows = [
                    next(r for r in x["requests"] if r["request_id"] == label)
                    for x in ds
                ]
                d["per_request"][variant][label] = dict(
                    answers=[r["answer"] for r in rows],
                    cached_tokens=[r["cached_prompt_tokens"] for r in rows],
                )
                for i, r in enumerate(rows, 1):
                    csv_rows.append(
                        dict(
                            variant=variant,
                            round=i,
                            request=label,
                            first_text_ms=r["first_text_ms"],
                            total_ms=r["total_ms"],
                            cached_tokens=r["cached_prompt_tokens"],
                            answer=r["answer"],
                        )
                    )
        if csv_rows:
            with (folder / "per-request.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
        d["complete"] = True
    except Exception as exc:
        d["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path.write_text(json.dumps(d, indent=2) + "\n")
        print(f"SUMMARY {path}", flush=True)
    print(json.dumps(d["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
