"""Experiment03: independent MLX non-prefix reuse with rule-based sparse repair.

Model evaluation runs only when explicitly invoked without --prepare-only.
No vLLM scheduler, allocator or installed model implementation is patched.
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nonprefix_repair import NonPrefixRepair

MODEL = "Qwen/Qwen3-0.6B"
SYSTEM = "Read the definition and report. Answer the final question with one word only."
DOCUMENT = (
    "Inspection report: the register records ALPHA as the final selection. "
    "Several alternatives were discussed, but the committee did not choose them. "
    "The team checked that the selection appears on the signed sheet, the "
    "meeting notes, and the delivery record. All three records agree on ALPHA. "
    "The report uses the label from the definition supplied before this report. "
    "The label has no universal meaning and must be interpreted for this case. "
    "The committee reviewed its records twice and found no later amendment. "
)
MARKER = "<<<INSERT_SHARED_REPORT_HERE>>>"


def cases(count=8):
    rows = []
    for index, (old, new) in enumerate(
        [
            ("Paris", "Tokyo"),
            ("red", "blue"),
            ("cat", "dog"),
            ("north", "south"),
            ("London", "Rome"),
            ("green", "yellow"),
            ("boat", "train"),
            ("east", "west"),
            ("apple", "pear"),
            ("Monday", "Friday"),
            ("silver", "gold"),
            ("small", "large"),
            ("winter", "summer"),
            ("tea", "coffee"),
            ("river", "forest"),
            ("copper", "iron"),
            ("circle", "square"),
            ("left", "right"),
            ("paper", "glass"),
            ("quiet", "loud"),
            ("early", "late"),
            ("mouse", "horse"),
            ("rain", "snow"),
            ("Berlin", "Madrid"),
            ("open", "closed"),
        ]
    ):
        if index >= count:
            break
        extra = (
            ""
            if index < 4
            else (
                "This revised definition takes priority over earlier conventions. "
                "Use it consistently throughout this case. "
            )
        )
        rows.append(
            {
                "id": chr(65 + index) if count == 25 else f"case-{index + 1:02}",
                "source_context": f"Definition for this case: ALPHA means {old}. ",
                "target_context": extra
                + f"Definition for this case: ALPHA means {new}. ",
                "document": DOCUMENT,
                "question": (
                    "What is the final selection according to this case's definition?"
                ),
                "expected": new,
            }
        )
    return rows


def segments(tokenizer, context, document, question):
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": context + "\n\n" + MARKER + "\n\n" + question},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count(MARKER) != 1:
        raise ValueError("Chat template did not preserve the document marker")
    before, after = rendered.split(MARKER)
    # Explicit token segments make the shared document IDs identical in all arms.
    # Dense reference consumes exactly the same concatenated IDs as sparse repair.
    return (
        tuple(tokenizer.encode(before, add_special_tokens=False)),
        tuple(tokenizer.encode(document, add_special_tokens=False)),
        tuple(tokenizer.encode(after, add_special_tokens=False)),
    )


def generate(model, prefill, limit, eos_ids):
    ids = []
    logits = prefill.logits
    first = None
    for index in range(limit):
        token = int(mx.argmax(logits[0], axis=-1).item())
        if first is None:
            mx.synchronize()
            first = time.perf_counter()
        ids.append(token)
        if token in eos_ids or index + 1 == limit:
            break
        logits = model(mx.array([[token]]), cache=prefill.cache)[:, -1, :]
        mx.eval(logits, [item.state for item in prefill.cache])
    mx.synchronize()
    return ids, first


def normalized_answer(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def compare_logits(reference, candidate):
    a, b = reference[0].astype(mx.float32), candidate[0].astype(mx.float32)
    log_a = a - mx.logsumexp(a)
    log_b = b - mx.logsumexp(b)
    return {
        "first_token_kl_dense_to_candidate": max(
            0.0, float(mx.sum(mx.exp(log_a) * (log_a - log_b)).item())
        ),
        "max_abs_first_logit_error": float(mx.max(mx.abs(a - b)).item()),
        "first_token_agrees_with_dense": bool((mx.argmax(a) == mx.argmax(b)).item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--ratios", default="0,0.1,0.25,0.5,1")
    parser.add_argument(
        "--selector", choices=("boundary", "uniform"), default="boundary"
    )
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cases", type=int, choices=(8, 25), default=8)
    parser.add_argument(
        "--order-offset",
        type=int,
        default=0,
        help="Rotate measured arms; compare to dense after all arms",
    )
    args = parser.parse_args()
    try:
        ratios = sorted(set(float(value) for value in args.ratios.split(",")))
    except ValueError:
        parser.error("ratios must be comma-separated numbers")
    if not ratios or any(not 0 <= value <= 1 for value in ratios):
        parser.error("ratios must be finite values in [0,1]")
    if args.max_output_tokens < 1:
        parser.error("max-output-tokens must be positive")
    manifest = {
        "experiment": "03-nonprefix-sparse-repair",
        "implementation": "own-rule-based-mlx-prototype",
        "model": str(args.model_path) if args.model_path else MODEL,
        "selector": args.selector,
        "repair_ratios": ratios,
        "max_output_tokens": args.max_output_tokens,
        "cases": cases(args.cases),
        "order_offset": args.order_offset,
        "semantics": (
            "approximate; not the CacheBlend paper's token-selection algorithm"
        ),
        "segment_tokenization": (
            "render chat template, split marker, encode "
            "three pieces separately; identical IDs in all arms"
        ),
    }
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    path = args.model_path or Path(snapshot_download(MODEL, local_files_only=True))
    mx.set_default_device(mx.gpu)
    model, tokenizer = load(str(path))
    model.eval()
    mx.eval(model.parameters())
    mx.synchronize()
    adapter = NonPrefixRepair(model)
    eos = tokenizer.eos_token_ids
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    folder = Path("results/nonprefix-repair")
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"repair-{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    result = {
        "manifest": manifest,
        "model_path": str(path.resolve()),
        "versions": {name: version(name) for name in ["mlx", "mlx-lm"]},
        "device": str(mx.default_device()),
        "cases": [],
        "complete": False,
        "note": (
            "offline cache preparation is separately recorded; "
            "no output-quality equivalence is assumed"
        ),
    }
    try:
        # Warm each arm on the first case; exclude this work from recorded times.
        first_case = manifest["cases"][0]
        source, doc, _ = segments(
            tokenizer, first_case["source_context"], DOCUMENT, first_case["question"]
        )
        target, _, suffix = segments(
            tokenizer, first_case["target_context"], DOCUMENT, first_case["question"]
        )
        warm = adapter.prepare(source, doc)
        adapter.dense(target, doc, suffix).evaluate()
        for ratio in ratios:
            adapter.repair(warm, target, doc, suffix, ratio, args.selector).evaluate()
        for case in manifest["cases"]:
            source, doc, _ = segments(
                tokenizer, case["source_context"], case["document"], case["question"]
            )
            target, target_doc, suffix = segments(
                tokenizer, case["target_context"], case["document"], case["question"]
            )
            if (
                len(target + doc + suffix) + args.max_output_tokens
                > model.args.max_position_embeddings
            ):
                raise ValueError("Prompt plus requested output exceeds model positions")
            if target_doc != doc:
                raise ValueError("Shared token IDs differ")
            mx.synchronize()
            start = time.perf_counter()
            chunk = adapter.prepare(source, doc)
            prepare_ms = (time.perf_counter() - start) * 1000
            case_result = {
                "id": case["id"],
                "expected": case["expected"],
                "source_prefix_tokens": len(source),
                "target_prefix_tokens": len(target),
                "document_tokens": len(doc),
                "suffix_tokens": len(suffix),
                "position_shift": len(target) - len(source),
                "offline_cache_prepare_ms": prepare_ms,
                "cached_document_bytes": chunk.nbytes,
                "target_token_ids": list(target + doc + suffix),
                "variants": [],
            }
            result["cases"].append(case_result)
            observations = []
            arms = [None] + ratios
            offset = args.order_offset % len(arms)
            arms = arms[offset:] + arms[:offset]
            for ratio in arms:
                mx.synchronize()
                start = time.perf_counter()
                prefill = (
                    adapter.dense(target, doc, suffix)
                    if ratio is None
                    else adapter.repair(
                        chunk, target, doc, suffix, ratio, args.selector
                    )
                )
                prefill.evaluate()
                prefill_ms = (time.perf_counter() - start) * 1000
                first_logits = prefill.logits
                ids, first = generate(model, prefill, args.max_output_tokens, eos_ids)
                end = time.perf_counter()
                text = tokenizer.decode(ids, skip_special_tokens=True)
                row = {
                    "mode": "dense" if ratio is None else "reuse_repair",
                    "ratio": ratio,
                    "prefill_ms": prefill_ms,
                    "first_token_ms": (first - start) * 1000,
                    "total_ms": (end - start) * 1000,
                    "output_tokens": ids,
                    "answer": text,
                    "answer_exact_match": normalized_answer(text)
                    == normalized_answer(case["expected"]),
                    "repaired_doc_positions": list(prefill.repaired_document_positions),
                    "recomputed_rows_per_layer": prefill.recomputed_rows_per_layer,
                    "reused_rows_per_layer": prefill.reused_rows_per_layer,
                }
                case_result["variants"].append(row)
                observations.append((row, first_logits, ids))
                print(
                    f"{case['id']} {row['mode']} ratio={ratio}: "
                    f"correct={row['answer_exact_match']} prefill={prefill_ms:.2f}ms",
                    flush=True,
                )
                destination.write_text(json.dumps(result, indent=2) + "\n")
            _, dense_logits, dense_ids = next(
                item for item in observations if item[0]["ratio"] is None
            )
            for row, first_logits, ids in observations:
                row.update(compare_logits(dense_logits, first_logits))
                row["matches_dense_generation"] = ids == dense_ids
            destination.write_text(json.dumps(result, indent=2) + "\n")
        result["aggregate"] = []
        for ratio in [None] + ratios:
            rows = [
                row
                for case in result["cases"]
                for row in case["variants"]
                if row["ratio"] == ratio
            ]
            result["aggregate"].append(
                {
                    "mode": "dense" if ratio is None else "reuse_repair",
                    "ratio": ratio,
                    "cases": len(rows),
                    "answer_accuracy": sum(row["answer_exact_match"] for row in rows)
                    / len(rows),
                    "first_token_agreement": sum(
                        row["first_token_agrees_with_dense"] for row in rows
                    )
                    / len(rows),
                    "mean_first_token_kl": statistics.mean(
                        row["first_token_kl_dense_to_candidate"] for row in rows
                    ),
                    "median_prefill_ms": statistics.median(
                        row["prefill_ms"] for row in rows
                    ),
                    "median_first_token_ms": statistics.median(
                        row["first_token_ms"] for row in rows
                    ),
                    "output_token_counts": [len(row["output_tokens"]) for row in rows],
                }
            )
        result["complete"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
