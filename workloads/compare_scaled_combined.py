"""Run the same unchunked long/short workload at 25, 50 or 100 requests."""

import argparse
import os
import sys

import compare_prefix_policies as base
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def build_prompts(count):
    if count not in (25, 50, 100):
        raise ValueError("Scaled workload supports 25, 50 or 100 candidates")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(
            base.MODEL,
            revision=os.environ.get("STUDY_MODEL_REVISION"),
            local_files_only=True,
        ),
        local_files_only=True,
    )
    long_count = (count + 1) // 2
    labels = [f"R{index:03}" for index in range(1, count + 1)]
    base.PROMPT_GROUPS = {
        "long": labels[:long_count],
        "short": labels[long_count:],
    }
    prompts = {}
    for index, label in enumerate(labels):
        target = 320 if index < long_count else 64
        prompt = (
            f"Case {label}: Explain three useful steps for planning a community "
            "project. Consider cost, practical work, and how to measure progress."
        )
        for _ in range(512):
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": base.SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) >= target:
                break
            prompt += " detail"
        if len(ids) != target:
            raise ValueError(f"{label} has {len(ids)} tokens; expected {target}")
        prompts[label] = prompt
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidates", type=int, required=True)
    args, _ = parser.parse_known_args()
    if args.candidates not in (25, 50, 100):
        raise SystemExit("--candidates must be 25, 50 or 100")
    base.build_prompts = build_prompts
    base.WORKLOAD_NAME = f"scaled-combined-{args.candidates}"
    base.CANDIDATE_INTERVAL_S = 0.010
    base.BLOCKERS = {
        f"H{i}": (
            f"For workshop number {i}, write a detailed numbered list of forty "
            "different household objects. For each object explain its purpose and "
            "how it is made. Continue the list without an introduction."
        )
        for i in range(1, 9)
    }
    base.BLOCKER_TOKENS = 128
    sys.argv[0] = "compare_scaled_combined.py"
    base.main()
