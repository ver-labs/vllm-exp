"""Diagnostic derivative of the frozen benchmark; timings include probe overhead."""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from profile_tools import Capture


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "model": args.model,
        "revision": args.revision,
        "arm": args.arm,
        "repeat": args.repeat,
        "profiles": {},
        "diagnostic_only": True,
        "scope": (
            "In-process token-ID engine latency, closed-loop concurrency "
            "3; not HTTP or maximum serving capacity"
        ),
    }
    try:
        import torch
        from transformers import AutoTokenizer

        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import RequestOutputKind

        assert os.environ["VLLM_BATCH_INVARIANT"] == "1"
        assert Path(vllm.__file__).resolve().is_relative_to(args.source.resolve())
        work = Path.home() / "vllm-research-20261024"
        props = torch.cuda.get_device_properties(0)
        assert "A100" in props.name and torch.cuda.get_device_capability(0) == (8, 0)
        report["device"] = {
            "name": props.name,
            "uuid": str(props.uuid),
            "memory_bytes": props.total_memory,
            "hostname": os.uname().nodename,
            "slurm_job": os.environ.get("SLURM_JOB_ID"),
        }
        report["gpu_before"] = subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,te"
                    "mperature.gpu,power.draw"
                ),
                "--format=csv",
            ],
            text=True,
        )
        local_model = work / "benchmark-models-20261026" / args.model.split("/")[-1]
        tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
        documents = [
            (
                "The library stores books in carefully numbered shelves. Read"
                "ers reserve books online and collect them from librarians. E"
                "ach catalogue record describes a title and its available cop"
                "ies. "
            ),
            (
                "Astronomers observe distant stars and compare their temperat"
                "ures across the night sky. Telescopes collect light and reve"
                "al how galaxies form over many billions of years. "
            ),
            (
                "Ocean scientists study coral reefs and measure changes in wa"
                "ter temperature. Healthy ecosystems support many species and"
                " help communities understand their natural environment. "
            ),
        ]
        base = [
            tokenizer.encode(text * 8, add_special_tokens=False)[:64]
            for text in documents
        ]
        assert len({tokens[0] for tokens in base}) == 3
        profiles = {}
        for profile in ("partial_reuse", "popularity_shift", "unique_control"):
            items = []
            for i in range(96):
                if profile == "partial_reuse":
                    family, tokens = "A", base[0][:63] + [1000 + i]
                elif profile == "popularity_shift":
                    family = "A" if i < 32 or i >= 64 else "B"
                    if i % 5 == 4:
                        family, tokens = "unique", [3000 + i] + base[2][1:]
                    else:
                        tokens = base[0 if family == "A" else 1][:63] + [2000 + i]
                else:
                    family, tokens = "unique", [4000 + i] + base[2][1:]
                assert len(tokens) == 64 and max(tokens) < tokenizer.vocab_size
                items.append(
                    {"id": f"{profile}-{i:03d}", "family": family, "tokens": tokens}
                )
            profiles[profile] = items
        workload = {
            "documents": base,
            "profiles": profiles,
            "output_tokens": 32,
            "concurrency": 3,
            "request_count_per_profile": 96,
        }
        payload = json.dumps(workload, sort_keys=True).encode()
        report["workload_sha256"] = hashlib.sha256(payload).hexdigest()
        (args.output / "workload.json").write_bytes(payload)
        settings = dict(
            model=str(local_model),
            tokenizer=str(local_model),
            download_dir=str(work / "model-cache"),
            dtype="bfloat16",
            seed=0,
            max_model_len=192,
            max_num_seqs=3,
            max_num_batched_tokens=64,
            block_size=16,
            num_gpu_blocks_override=16,
            kv_cache_memory_bytes=268435456,
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
        if os.environ.get("STUDY_PARTIAL_PREFIX") == "1":
            settings["prefix_match_unit"] = 1
        report["settings"] = settings
        report["policy_environment"] = {
            k: v for k, v in os.environ.items() if k.startswith("STUDY_")
        }
        llm = LLM(**settings)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        pool = core.scheduler.kv_cache_manager.block_pool
        runner = engine.model_executor.driver_worker.worker.model_runner
        assert len(pool.blocks) == 16
        report["resolved"] = {
            "vllm_path": vllm.__file__,
            "runner": inspect.getfile(type(runner)),
            "cache_config": str(engine.vllm_config.cache_config),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        }
        params = SamplingParams(
            temperature=0,
            seed=0,
            max_tokens=32,
            ignore_eos=True,
            detokenize=False,
            output_kind=RequestOutputKind.CUMULATIVE,
        )
        sequence = [0]

        def stream(items, measure=False, limit=32):
            prepared = []
            for item in items:
                sequence[0] += 1
                label = f"bench-{sequence[0]}"
                prepared.append((label, item, {"prompt_token_ids": item["tokens"]}))
            local_params = (
                params
                if limit == 32
                else SamplingParams(
                    temperature=0,
                    seed=0,
                    max_tokens=limit,
                    ignore_eos=True,
                    detokenize=False,
                )
            )
            active, finished = {}, []
            cursor = 0
            phase_start = None
            while cursor < len(prepared) or active:
                while len(active) < 3 and cursor < len(prepared):
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

        for i, match in enumerate((1, 7, 15, 17, 31, 47, 63)):
            assert engine.reset_prefix_cache()
            stream([{"id": "warm-seed", "family": "A", "tokens": base[0]}], limit=1)
            tail = base[1][: 64 - match]
            if tail[0] == base[0][match]:
                tail[0] = 5000 + i
            stream(
                [
                    {
                        "id": f"warm-{i}",
                        "family": "warm",
                        "tokens": base[0][:match] + tail,
                    }
                ]
            )
        assert engine.reset_prefix_cache()
        stream(
            [
                {
                    "id": f"warm-batch-{i}",
                    "family": "warm",
                    "tokens": [6000 + i] + base[2][1:],
                }
                for i in range(3)
            ]
        )
        order = ["partial_reuse", "popularity_shift", "unique_control"]
        order = order[args.repeat % 3 :] + order[: args.repeat % 3]
        report["profile_order"] = order
        for mode, profile in [
            (m, p) for m in ("control", "cpu", "trace") for p in order
        ]:
            assert engine.reset_prefix_cache()
            policy = getattr(pool, "_study_eviction", None)
            if policy is not None:
                assert not any(policy.reuse_counts) and policy.epochs == 0
            if profile != "unique_control":
                stream([{"id": "seed", "family": "A", "tokens": base[0]}], limit=1)
            torch.cuda.synchronize()
            items = profiles[profile][:3] if mode == "trace" else profiles[profile]
            with Capture(mode, engine, core, runner, args.output / f"{mode}-{profile}"):
                result = stream(items, measure=True)
            report["profiles"][f"{mode}/{profile}"] = result
            (args.output / "result.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        report["gpu_after"] = subprocess.check_output(
            [
                "nvidia-smi",
                (
                    "--query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,te"
                    "mperature.gpu,power.draw"
                ),
                "--format=csv",
            ],
            text=True,
        )
        report["status"] = "passed"
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("BENCHMARK_RESULT", report["status"], args.output, flush=True)
    if report["status"] != "passed":
        print(report["error"], flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
