"""Inspect one allocated GPU from an explicit isolated CUDA environment."""

import importlib.metadata
import json
import platform
import sys
import traceback


def main():
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "status": "failed",
        "vllm_native_extension": "not_tested",
        "full_vllm_build": "not_tested",
    }
    for name in ("torch", "vllm", "triton", "flashinfer-python", "setuptools"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        report["torch_path"] = torch.__file__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_home"] = CUDA_HOME
        report["cuda_available"] = torch.cuda.is_available()
        if not report["cuda_available"]:
            raise RuntimeError("PyTorch cannot use the allocated CUDA GPU.")
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["capability"] = list(torch.cuda.get_device_capability(0))
        report["torch_arch_list"] = torch.cuda.get_arch_list()
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        if report["capability"] != [8, 0] or "A100" not in report["gpu_name"]:
            raise RuntimeError("This study requires the allocated A100 device.")
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.ones((32, 32), device="cuda", dtype=dtype)
            result = x @ x
            torch.cuda.synchronize()
            if not bool(torch.all(result == 32).item()):
                raise RuntimeError(f"Incorrect matrix result with {dtype}.")
        report["fp16_bf16_matmul"] = "passed"
        report["status"] = "torch_cuda_smoke_passed"
    except Exception:
        report["error"] = traceback.format_exc()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "torch_cuda_smoke_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
