"""Check guarded FA2 routing against original attention and an FP32 reference."""

import itertools
import json
import os
from pathlib import Path

import torch

from vllm.vllm_flash_attn import flash_attn_interface as fa

torch.manual_seed(20261026)
results = []
dispatch = []
native = torch.ops._vllm_fa2_C.varlen_fwd


def checked(*args):
    dispatch.append({"max_q": args[10], "window": [args[16], args[17]]})
    return native(*args)


torch.ops._vllm_fa2_C.varlen_fwd = checked
for batch, context, dtype, groups, query in itertools.product(
    (3, 8, 16), (512, 2112), (torch.bfloat16,), (1, 2), (1, 2)
):
    heads, dim, blocks, block = 16, 128, 256, 16
    kv_heads = heads // groups
    q = torch.randn(batch * query, heads, dim, device="cuda", dtype=dtype) * 0.2
    k = torch.randn(blocks, block, kv_heads, dim, device="cuda", dtype=dtype) * 0.2
    v = torch.randn_like(k)
    table_cpu = torch.stack([torch.randperm(blocks)[:132] for _ in range(batch)])
    table = table_cpu.to(device="cuda", dtype=torch.int32)
    lengths = [context - i for i in range(batch)]
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * query
    inputs = dict(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=query,
        cu_seqlens_q=cu,
        max_seqlen_k=context,
        seqused_k=used,
        block_table=table,
        causal=True,
        return_softmax_lse=True,
        fa_version=2,
        num_splits=1,
    )
    fa._STUDY_FA2_SINGLE_QUERY_CAUSAL = False
    before, lse_before = fa.flash_attn_varlen_func(**inputs)
    fa._STUDY_FA2_SINGLE_QUERY_CAUSAL = True
    after, lse_after = fa.flash_attn_varlen_func(**inputs)
    changed = query == 1 and groups > 1
    if changed and batch == 3 and context == 512 and dtype == torch.bfloat16:
        guard_inputs = inputs.copy()
    assert dispatch[-2]["window"] == [-1, -1]
    assert dispatch[-1]["window"] == ([-1, 0] if changed else [-1, -1])
    torch.testing.assert_close(after, before, atol=0.015625, rtol=0.005)
    torch.testing.assert_close(lse_after, lse_before, atol=0.015625, rtol=0.005)
    reference = []
    for i, length in enumerate(lengths):
        positions = torch.arange(length)
        kk = k.cpu()[table_cpu[i, positions // block], positions % block].float()
        vv = v.cpu()[table_cpu[i, positions // block], positions % block].float()
        kk = kk.repeat_interleave(groups, dim=1)
        vv = vv.repeat_interleave(groups, dim=1)
        qq = q[i * query : (i + 1) * query].cpu().float()
        scores = torch.einsum("qhd,khd->hqk", qq, kk) * (dim**-0.5)
        mask = positions[None, :] <= torch.arange(length - query, length)[:, None]
        scores.masked_fill_(~mask[None, :, :], -float("inf"))
        reference.append(torch.einsum("hqk,khd->qhd", scores.softmax(-1), vv))
    expected = torch.cat(reference).to(after.device)
    torch.testing.assert_close(after.float(), expected, atol=0.015625, rtol=0.005)
    results.append(
        {
            "batch": batch,
            "context": context,
            "dtype": str(dtype),
            "groups": groups,
            "query": query,
            "changed": changed,
            "max_abs": float((after.float() - before.float()).abs().max()),
            "reference_max_abs": float((after.float() - expected).abs().max()),
        }
    )
guards = []
for change in (
    {"causal": False},
    {"window_size": [4, 0]},
    {"alibi_slopes": torch.linspace(0.01, 0.1, 16, device="cuda")},
):
    args = {**guard_inputs, **change}
    fa._STUDY_FA2_SINGLE_QUERY_CAUSAL = False
    before, _ = fa.flash_attn_varlen_func(**args)
    fa._STUDY_FA2_SINGLE_QUERY_CAUSAL = True
    after, _ = fa.flash_attn_varlen_func(**args)
    assert dispatch[-1] == dispatch[-2]
    assert torch.equal(before, after)
    guards.append(next(iter(change)))
clones = {}
for enabled in (False, True):
    fa._STUDY_FA2_SINGLE_QUERY_CAUSAL = enabled
    args = {**guard_inputs, "out": torch.empty_like(guard_inputs["q"])}
    fa.flash_attn_varlen_func(**args)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        fa.flash_attn_varlen_func(**args)
    clones[str(enabled)] = sum(
        e.count for e in p.key_averages() if e.key == "aten::clone"
    )
assert clones["True"] == 0 and clones["False"] > 0, clones
out = Path(os.environ["ATTENTION_GATE_OUTPUT"])
out.write_text(
    json.dumps(
        {
            "status": "passed",
            "cases": results,
            "fallback_guards": guards,
            "attention_clone_counts": clones,
        },
        indent=2,
    )
    + "\n"
)
print("ATTENTION_GATE_PASSED", len(results), flush=True)
