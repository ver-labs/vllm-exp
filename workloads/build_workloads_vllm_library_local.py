import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed



URL = "http://127.0.0.1:8001/v1/chat/completions"
INSTALLATION = "source-checkout"
MODEL = "Qwen/Qwen3-0.6B"

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

CONCURRENCY = 5

print_lock = Lock()


def send_request(label, prompt, batch_start):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    first_text_time = None
    answer = ""
    usage = {}
    finish_reason = None
    error = None
    stream_done = False

    with print_lock:
        print(f"[START] {label}: {prompt}", flush=True)

    start = time.perf_counter()

    try:
        with urlopen(request, timeout=180) as response:
            for raw_line in response:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue

                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    stream_done = True
                    break

                event = json.loads(data)
                if event.get("error"):
                    raise RuntimeError(event["error"])

                if event.get("usage"):
                    usage = event["usage"]

                for choice in event.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        if first_text_time is None:
                            first_text_time = time.perf_counter()
                        answer += text

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

        if not stream_done:
            raise RuntimeError("Response ended without [DONE]")

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    end = time.perf_counter()

    return {
        "request_id": label,
        "prompt": prompt,
        "answer": answer,
        "started_ms": round((start - batch_start) * 1000, 2),
        "finished_ms": round((end - batch_start) * 1000, 2),
        "first_text_ms": (
            round((first_text_time - start) * 1000, 2)
            if first_text_time is not None else None
        ),
        "total_ms": round((end - start) * 1000, 2),
        "usage": usage,
        "finish_reason": finish_reason,
        "error": error,
    }


batch_start = time.perf_counter()
results = []

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = {
        executor.submit(send_request, label, prompt, batch_start): label
        for label, prompt in PROMPTS.items()
    }

    for future in as_completed(futures):
        result = future.result()
        results.append(result)

        with print_lock:
            if result["error"]:
                print(
                    f"[FAILED] {result['request_id']}: {result['error']}",
                    flush=True,
                )
            else:
                print(
                    f"[DONE] {result['request_id']} | "
                    f"First text: {result['first_text_ms']} ms | "
                    f"Total: {result['total_ms']} ms",
                    flush=True,
                )

batch_ms = (time.perf_counter() - batch_start) * 1000

# Save in original prompt order; retain actual start/finish timestamps.
order = {label: index for index, label in enumerate(PROMPTS)}
results.sort(key=lambda result: order[result["request_id"]])

successful = [r for r in results if r["error"] is None]
output_tokens = sum(
    r["usage"].get("completion_tokens", 0)
    for r in successful
)

folder = Path("results/source-checkout")
folder.mkdir(parents=True, exist_ok=True)
output = folder / f"concurrent-{datetime.now():%Y%m%d-%H%M%S-%f}.json"

output.write_text(json.dumps({
    "installation": INSTALLATION,
    "model": MODEL,
    "url": URL,
    "system_prompt": SYSTEM,
    "max_output_tokens": 64,
    "execution": "concurrent",
    "client_concurrency": CONCURRENCY,
    "request_count": len(results),
    "successful_requests": len(successful),
    "failed_requests": len(results) - len(successful),
    "batch_total_ms": round(batch_ms, 2),
    "successful_output_tokens": output_tokens,
    "output_tokens_per_second": round(
        output_tokens / (batch_ms / 1000), 2
    ),
    "requests": results,
}, indent=2))

print(f"\nCompleted: {len(successful)}/{len(results)}")
print(f"Client concurrency: {CONCURRENCY}")
print(f"Batch duration: {batch_ms:.2f} ms")
print(f"Saved results to {output.resolve()}")