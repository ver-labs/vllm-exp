import json
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

# Select GPU before importing the generation utilities.
mx.set_default_device(mx.gpu)

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = "Qwen/Qwen3-0.6B"
model_path = snapshot_download(MODEL)

SYSTEM = "You are a helpful assistant. Explain clearly and concisely."

PROMPTS = {
    "A": "Explain how a library manages book reservations.",
    "B": "Explain how rising ocean temperatures affect coral reefs.",
    "C": "Explain how a star forms and evolves.",
    "D": "Suggest ways to improve a library's membership process.",
    "E": "Describe useful features for an online library catalogue.",
    "F": "Explain how bees pollinate flowering plants.",
    "G": "Describe how rain forms in the atmosphere.",
    "H": "Suggest ways to reduce food waste at home.",
    "I": "Explain how vaccines help the immune system.",
    "J": "Describe how a bicycle gear system works.",
    "K": "Suggest ways to make public parks more accessible.",
    "L": "Explain how composting turns waste into soil.",
    "M": "Describe useful features for a city bus tracking app.",
    "N": "Explain how earthquakes occur along fault lines.",
    "O": "Suggest ways to improve recycling in apartment buildings.",
    "P": "Explain how photosynthesis produces oxygen.",
    "Q": "Describe how a thermostat regulates room temperature.",
    "R": "Suggest ways to organize a community book swap.",
    "S": "Explain how tides are caused by the moon.",
    "T": "Describe useful features for a museum audio guide.",
    "U": "Explain how batteries store and release energy.",
    "V": "Suggest ways to reduce traffic congestion downtown.",
    "W": "Explain how birds migrate across continents.",
    "X": "Describe how a water filter removes impurities.",
    "Y": "Suggest ways to improve a school's lunch program.",
}

# Reuse the cached original checkpoint, without downloading or quantizing.
model_path = snapshot_download(MODEL, local_files_only=True)

print(f"Device: {mx.default_device()}")
print(f"Model: {model_path}")

load_start = time.perf_counter()
model, tokenizer = load(model_path)
mx.eval(model.parameters())
mx.synchronize()
load_ms = (time.perf_counter() - load_start) * 1000

print(f"Model load: {load_ms:.2f} ms (excluded from request timings)")


def send_request(label, prompt):
    print(f"\n{label}: {prompt}")

    mx.synchronize()
    start = time.perf_counter()

    prompt_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    answer = ""
    first_text_time = None
    last_response = None

    # No prompt cache is passed: each request gets fresh KV state.
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt_ids,
        max_tokens=64,
        sampler=make_sampler(temp=0.0),
    ):
        last_response = response
        if response.text:
            if first_text_time is None:
                first_text_time = time.perf_counter()
            answer += response.text
            print(response.text, end="", flush=True)

    mx.synchronize()
    end = time.perf_counter()

    result = {
        "request_id": label,
        "prompt": prompt,
        "answer": answer,
        "first_text_ms": (
            round((first_text_time - start) * 1000, 2)
            if first_text_time is not None else None
        ),
        "total_ms": round((end - start) * 1000, 2),
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": (
                last_response.generation_tokens if last_response else 0
            ),
        },
        "finish_reason": (
            last_response.finish_reason if last_response else None
        ),
    }

    print(
        f"\nFirst text: {result['first_text_ms']} ms"
        f" | Total: {result['total_ms']} ms"
    )
    print(f"Token usage: {result['usage']}")
    return result


results = [
    send_request(label, prompt)
    for label, prompt in PROMPTS.items()
]

folder = Path("results/direct-mlx")
folder.mkdir(parents=True, exist_ok=True)
output = folder / f"chat-{datetime.now():%Y%m%d-%H%M%S-%f}.json"

output.write_text(json.dumps({
    "model": MODEL,
    "model_revision": Path(model_path).name,
    "mlx_version": version("mlx"),
    "mlx_lm_version": version("mlx-lm"),
    "system_prompt": SYSTEM,
    "max_output_tokens": 64,
    "execution": "sequential",
    "model_load_ms": round(load_ms, 2),
    "cross_request_prefix_reuse": False,
    "requests": results,
}, indent=2))

print(f"\nSaved results to {output.resolve()}")