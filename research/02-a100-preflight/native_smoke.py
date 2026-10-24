"""Check source import provenance and small native CUDA operators, without a model."""

import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path


def main():
    expected = Path(sys.argv[1]).resolve()
    report = {"expected_source": str(expected), "status": "failed"}
    try:
        import torch

        import vllm
        from vllm import _custom_ops as ops

        actual = Path(vllm.__file__).resolve()
        if not actual.is_relative_to(expected):
            raise RuntimeError(f"Imported {actual}, expected source under {expected}")
        report["vllm_path"] = str(actual)
        report["torch_version"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0)
        report["native_modules"] = {}
        for name in ("vllm._C_stable_libtorch",):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            report["native_modules"][name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
            }
        torch.manual_seed(1)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn((4, 128), device="cuda", dtype=dtype)
            weight = torch.ones(128, device="cuda", dtype=dtype)
            out = torch.empty_like(x)
            ops.rms_norm(out, x, weight, 1e-6)
            reference = x.float() * torch.rsqrt(
                x.float().square().mean(-1, keepdim=True) + 1e-6
            )
            torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.02)
            key = (
                torch.arange(256, device="cuda", dtype=torch.float32)
                .reshape(2, 2, 64)
                .to(dtype)
            )
            value = key + 1
            key_cache = torch.zeros((2, 16, 2, 64), device="cuda", dtype=dtype)
            value_cache = torch.zeros_like(key_cache)
            slots = torch.tensor([0, 17], device="cuda", dtype=torch.int64)
            scale = torch.tensor(1.0, device="cuda")
            ops.reshape_and_cache_flash(
                key, value, key_cache, value_cache, slots, "auto", scale, scale
            )
            expected_key = torch.zeros_like(key_cache)
            expected_value = torch.zeros_like(value_cache)
            expected_key[0, 0], expected_key[1, 1] = key[0], key[1]
            expected_value[0, 0], expected_value[1, 1] = value[0], value[1]
            torch.testing.assert_close(key_cache, expected_key, rtol=0, atol=0)
            torch.testing.assert_close(value_cache, expected_value, rtol=0, atol=0)
            copied = torch.zeros_like(key_cache)
            mapping = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
            ops.swap_blocks(
                key_cache,
                copied,
                key_cache[0].numel() * key_cache.element_size(),
                mapping,
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(copied[1], key_cache[0], rtol=0, atol=0)
            torch.testing.assert_close(copied[0], key_cache[1], rtol=0, atol=0)
        report["checks"] = [
            "native_imports",
            "fp16_bf16_rms_norm",
            "kv_insertion",
            "device_block_copy",
        ]
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
