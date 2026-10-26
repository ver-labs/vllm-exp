"""Diagnostic scopes: CPU wall spans and a separate CPU/CUDA trace pass."""

import functools
import gzip
import importlib
import json
import time
from collections import defaultdict


class Capture:
    def __init__(self, mode, engine, core, runner, output):
        self.mode, self.engine, self.core = mode, engine, core
        self.runner, self.output = runner, output
        self.originals = []
        self.stats = defaultdict(lambda: [0, 0, 0])
        self.stack, self.batches = [], []
        self.profiler = None
        self.available = []

    def wrap(self, obj, name, label, after=None):
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)
        self.originals.append((obj, name, original))
        self.available.append(label)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self.mode == "trace":
                with self.torch.profiler.record_function("study/" + label):
                    result = original(*args, **kwargs)
            else:
                start = time.perf_counter_ns()
                frame = [0]
                self.stack.append(frame)
                try:
                    result = original(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter_ns() - start
                    self.stack.pop()
                    values = self.stats[label]
                    values[0] += 1
                    values[1] += elapsed
                    values[2] += elapsed - frame[0]
                    if self.stack:
                        self.stack[-1][0] += elapsed
            if after:
                after(result, args)
            return result

        setattr(obj, name, wrapped)

    def scheduled(self, result, args):
        sizes = list(result.num_scheduled_tokens.values())
        self.batches.append(
            {
                "query_lengths": sizes,
                "total_tokens": result.total_num_scheduled_tokens,
                "copy_pairs": len(result.kv_cache_block_copies or []),
                "new_requests": len(result.scheduled_new_reqs),
            }
        )

    def __enter__(self):
        if self.mode == "control":
            return self
        import torch

        self.torch = torch
        scheduler = self.core.scheduler
        manager = scheduler.kv_cache_manager
        pool = manager.block_pool
        self.wrap(self.engine, "add_request", "engine.add_request")
        self.wrap(self.engine, "step", "engine.step")
        self.wrap(scheduler, "schedule", "scheduler.schedule", self.scheduled)
        self.wrap(scheduler, "update_from_output", "scheduler.update_from_output")
        for name in ("get_computed_blocks", "allocate_slots", "cache_blocks", "free"):
            self.wrap(manager, name, "cache_manager." + name)
        for name in (
            "get_new_blocks",
            "cache_full_blocks",
            "cache_partial_block",
            "touch",
            "free_blocks",
            "_get_partial_block_hash",
        ):
            self.wrap(pool, name, "block_pool." + name)
        partial = getattr(manager, "_study_partial_prefix", None)
        if partial is not None:
            for name in ("publish", "block_changed", "forget"):
                self.wrap(partial, name, "partial_index." + name)
        for name in (
            "execute_model",
            "update_requests",
            "prepare_inputs",
            "prepare_attn",
            "sample_tokens",
            "sample",
        ):
            self.wrap(self.runner, name, "runner." + name)
        self.wrap(self.runner.get_model(), "forward", "model.forward")
        module = importlib.import_module(type(self.runner).__module__)
        self.wrap(module, "copy_kv_cache_blocks_inplace", "copy.dispatch")
        assert {"scheduler.schedule", "model.forward", "copy.dispatch"}.issubset(
            self.available
        )
        if self.mode == "trace":
            self.profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.profiler:
            self.profiler.__exit__(exc_type, exc_value, tb)
        for obj, name, original in reversed(self.originals):
            setattr(obj, name, original)
        if exc_type or self.mode == "control":
            return
        result = {
            "mode": self.mode,
            "scopes": self.available,
            "batches": self.batches,
            "cpu_wall": {
                name: {
                    "calls": v[0],
                    "inclusive_ms": v[1] / 1e6,
                    "exclusive_ms": v[2] / 1e6,
                }
                for name, v in self.stats.items()
            },
        }
        if self.profiler:
            trace = self.output.with_suffix(".trace.json")
            self.profiler.export_chrome_trace(str(trace))
            with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                import shutil

                shutil.copyfileobj(src, dst)
            trace.unlink()
            result["operators"] = [
                {
                    "name": e.key,
                    "calls": e.count,
                    "cpu_total_us": e.cpu_time_total,
                    "cpu_self_us": e.self_cpu_time_total,
                    "device_total_us": e.device_time_total,
                    "device_self_us": e.self_device_time_total,
                }
                for e in self.profiler.key_averages()
            ]
            assert any(e["device_total_us"] > 0 for e in result["operators"])
        self.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
