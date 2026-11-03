"""Shared fixed workload and engine timing primitives for the attention sweep."""

import time

LENGTHS = (64, 512, 2048)
CONCURRENCIES = (3, 8, 16)
REQUESTS = 96
OUTPUT = 32
PAGES = 4096


def workload(tokenizer):
    text = (
        "The library stores books in numbered shelves. Readers reserve books "
        "online and librarians update the catalogue. Ocean scientists collect "
        "observations and explain changes in ecosystems. "
    )
    base = tokenizer.encode(text * 128, add_special_tokens=False)
    assert len(base) >= max(LENGTHS)
    cases, profiles, warmups = {}, {}, {}
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            case = f"L{length}-C{concurrency}"
            index = len(cases)
            cases[case] = {"prompt_tokens": length, "concurrency": concurrency}
            profiles[case] = [
                {
                    "id": f"{case}-{i:03d}",
                    "family": "unique",
                    "tokens": [10000 + index * 128 + i] + base[1:length],
                }
                for i in range(REQUESTS)
            ]
            warmups[case] = [
                {
                    "id": f"warm-{case}-{i:03d}",
                    "family": "warm",
                    "tokens": [50000 + index * 32 + i] + base[1:length],
                }
                for i in range(concurrency)
            ]
    assert tokenizer.vocab_size > 51000
    return {
        "cases": cases,
        "profiles": profiles,
        "warmups": warmups,
        "output_tokens": OUTPUT,
        "requests_per_case": REQUESTS,
    }


def engine_settings(local_model):
    return dict(
        model=str(local_model),
        tokenizer=str(local_model),
        dtype="bfloat16",
        seed=0,
        max_model_len=2304,
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        block_size=16,
        num_gpu_blocks_override=PAGES,
        kv_cache_memory_bytes=8 * 1024**3,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        async_scheduling=False,
        enforce_eager=True,
        distributed_executor_backend="uni",
        disable_custom_all_reduce=True,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        compilation_config={"mode": 0},
        disable_log_stats=False,
    )


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def describe(values):
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "p99_exploratory": percentile(values, 0.99),
    }


def stream(engine, pool, items, concurrency, measure=False, limit=32):
    prepared = []
    for item in items:
        label = item["id"]
        prepared.append((label, item, {"prompt_token_ids": item["tokens"]}))
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    local_params = SamplingParams(
        temperature=0,
        seed=0,
        max_tokens=limit,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    active, finished = {}, []
    cursor = 0
    phase_start = None
    while cursor < len(prepared) or active:
        while len(active) < concurrency and cursor < len(prepared):
            label, item, prompt = prepared[cursor]
            cursor += 1
            now = time.perf_counter_ns()
            if phase_start is None:
                phase_start = now
            active[label] = {
                "item": item,
                "submit_ns": now,
                "first_ns": None,
                "last_ns": None,
                "previous_count": 0,
                "token_arrivals_ns": [],
            }
            engine.add_request(label, prompt, local_params)
        batch = engine.step()
        now = time.perf_counter_ns()
        for output in batch:
            state = active[output.request_id]
            ids = output.outputs[0].token_ids
            count = len(ids)
            increase = count - state["previous_count"]
            assert increase in (0, 1), "Unexpected stream token coalescing"
            if increase:
                if state["first_ns"] is None:
                    state["first_ns"] = now
                state["last_ns"] = now
                state["token_arrivals_ns"].append(now - phase_start)
            state["previous_count"] = count
            if not output.finished:
                continue
            assert count == limit and state["first_ns"] is not None
            metrics = output.metrics
            assert metrics is not None
            finished.append(
                {
                    "id": state["item"]["id"],
                    "family": state["item"]["family"],
                    "submit_offset_ns": state["submit_ns"] - phase_start,
                    "finish_offset_ns": now - phase_start,
                    "ttft_ms": (state["first_ns"] - state["submit_ns"]) / 1e6,
                    "latency_ms": (now - state["submit_ns"]) / 1e6,
                    "tpot_ms": (state["last_ns"] - state["first_ns"])
                    / 1e6
                    / (count - 1)
                    if count > 1
                    else None,
                    "cached_tokens": output.num_cached_tokens,
                    "preemptions": metrics.num_preemptions,
                    "output_token_ids": list(ids),
                    "token_arrivals_offset_ns": state["token_arrivals_ns"],
                }
            )
            del active[output.request_id]
    assert not engine.has_unfinished_requests()
    assert all(block.ref_cnt == 0 for block in pool.blocks if not block.is_null)
    assert len(finished) == len(items)
    elapsed = max(row["finish_offset_ns"] for row in finished) / 1e9
    result = {"elapsed_s": elapsed, "requests": finished}
    if measure:
        result.update(
            requests_per_s=len(finished) / elapsed,
            output_tokens_per_s=len(finished) * limit / elapsed,
            ttft_ms=describe([r["ttft_ms"] for r in finished]),
            latency_ms=describe([r["latency_ms"] for r in finished]),
            tpot_ms=describe([r["tpot_ms"] for r in finished]),
            cached_tokens=sum(r["cached_tokens"] or 0 for r in finished),
            preemptions=sum(r["preemptions"] for r in finished),
        )
    return result
