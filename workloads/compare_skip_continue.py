"""Twenty-five queued requests: thirteen long prompts followed by twelve short ones."""

import compare_prefix_policies as base
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def build_prompts(count):
    assert count == 25
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_download(base.MODEL, local_files_only=True), local_files_only=True
    )
    prompts = {}
    for index, label in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXY"):
        target = 320 if index < 13 else 64
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
        assert len(ids) == target, (label, len(ids), target)
        prompts[label] = prompt
    return prompts


if __name__ == "__main__":
    base.build_prompts = build_prompts
    base.BLOCKERS = {
        f"H{i}": (
            f"For workshop number {i}, write a detailed numbered list of forty "
            "different household objects. For each object explain its purpose and "
            "how it is made. Continue the list without an introduction."
        )
        for i in range(1, 9)
    }
    base.BLOCKER_TOKENS = 64
    base.main()
