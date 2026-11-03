"""Loopback streaming workload, with separate untimed distribution gates."""

import concurrent.futures
import json
import time
from urllib.request import Request, urlopen


def call(base, route, payload=None):
    request = Request(
        base + route,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=600) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def one(base, item, limit=32, natural=False, gate=False):
    payload = {
        "model": "study",
        "prompt": item["tokens"],
        "max_tokens": limit,
        "temperature": 0,
        "seed": 0,
        "ignore_eos": not natural,
        "return_token_ids": True,
        "stream": not gate,
    }
    if gate:
        payload.update(logprobs=item["vocab_size"], return_tokens_as_token_ids=True)
        result = call(base, "/v1/completions", payload)
        choice = result["choices"][0]
        return {
            "id": item["id"],
            "vocab_size": item["vocab_size"],
            "token_ids": choice["token_ids"],
            "logprobs": choice["logprobs"],
            "usage": result["usage"],
        }
    request = Request(
        base + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter_ns()
    tokens, arrivals, text = [], [], ""
    with urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if value == "[DONE]":
                break
            event = json.loads(value)
            if "error" in event:
                raise RuntimeError(event["error"])
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids:
                    arrivals.append({"ns": time.perf_counter_ns(), "tokens": len(ids)})
                    tokens.extend(ids)
                text += choice.get("text", "")
    end = time.perf_counter_ns()
    assert tokens and (natural or len(tokens) == limit)
    return {
        "id": item["id"],
        "token_ids": tokens,
        "text": text,
        "start_ns": start,
        "end_ns": end,
        "chunks": arrivals,
        "ttft_ms": (arrivals[0]["ns"] - start) / 1e6,
        "latency_ms": (end - start) / 1e6,
        "tpot_ms": (arrivals[-1]["ns"] - arrivals[0]["ns"]) / 1e6 / (len(tokens) - 1)
        if len(tokens) > 1
        else None,
    }


def wave(base, items, concurrency, limit=32, natural=False):
    start = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, base, item, limit, natural) for item in items]
        rows = [future.result() for future in futures]
    end = time.perf_counter_ns()
    elapsed = (end - start) / 1e9
    return {
        "elapsed_s": elapsed,
        "requests": rows,
        "requests_per_s": len(rows) / elapsed,
        "output_tokens_per_s": sum(len(r["token_ids"]) for r in rows) / elapsed,
    }


def compare_gate(before, after):
    import numpy as np

    assert [r["id"] for r in before] == [r["id"] for r in after]
    errors, count, exact = [], 0, 0
    for left, right in zip(before, after):
        assert left["token_ids"] == right["token_ids"], "Greedy gate mismatch"
        for a, b in zip(
            left["logprobs"]["top_logprobs"],
            right["logprobs"]["top_logprobs"],
            strict=True,
        ):
            assert a.keys() == b.keys() and len(a) == left["vocab_size"], (
                "Incomplete vocabulary",
                len(a),
                left["vocab_size"],
            )
            keys = sorted(a)
            x = np.array([a[k] for k in keys])
            y = np.array([b[k] for k in keys])
            assert np.isfinite(x).all() and np.isfinite(y).all()
            diff = x - y
            rms = float(np.sqrt(np.mean(diff * diff)))
            maximum = float(np.max(np.abs(diff)))
            count += 1
            exact += int(np.array_equal(x, y))
            errors.append({"max_abs": maximum, "rms": rms})
            assert np.allclose(x, y, atol=0.0625, rtol=0.005) and rms <= 0.02, errors[
                -1
            ]
    return {
        "status": "passed",
        "full_logprob_vectors": count,
        "exact_value_vectors": exact,
        "errors": errors,
    }
