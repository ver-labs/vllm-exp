"""Exercise real cache metadata without loading a model or executing GPU work."""

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


def make_cache():
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                ),
            )
        ],
    )
    settings = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16, prefix_match_unit=1, enable_prefix_caching=True
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    sizes = resolve_kv_cache_block_sizes(config, settings)
    return KVCacheManager(config, 128, *sizes), sizes


def request(label, tokens, unit, salt=None):
    return Request(
        label,
        tokens,
        SamplingParams(max_tokens=8),
        None,
        cache_salt=salt,
        block_hasher=get_request_block_hasher(unit, sha256),
    )


def main():
    source = Path(inspect.getfile(KVCacheManager)).resolve()
    tree = source.parents[3]
    for name, module in list(sys.modules.items()):
        if name.startswith("vllm.v1.core") and getattr(module, "__file__", None):
            assert Path(module.__file__).resolve().is_relative_to(tree), name
    init_none_hash(sha256)
    cache, sizes = make_cache()
    a = request("A", list(range(1, 33)), sizes[1])
    b = request("B", list(range(1, 32)) + [999], sizes[1])
    other = request("context", [888] + list(range(2, 33)), sizes[1])
    isolated = request("isolated", list(b.prompt_token_ids), sizes[1], "tenant-2")
    cache.allocate_slots(a, 32)
    a.num_computed_tokens = 32
    partial = getattr(cache, "_study_partial_prefix", None)
    if partial is not None:
        partial.publish(a)
    hit, count, _ = cache.get_computed_blocks(b)
    assert cache.get_computed_blocks(other)[1] == 0
    assert cache.get_computed_blocks(isolated)[1] == 0
    expected = 31 if partial is not None else 16
    assert count == expected, (count, expected)
    cache.allocate_slots(b, 32 - count, count, hit)
    copies, retained = cache.take_kv_cache_block_copies()
    assert len(copies) == int(partial is not None)
    endpoints = [(c.src_block_id, c.dst_block_id) for c in copies]
    cache.free(a)
    cache.free(b)
    if endpoints:
        assert all(
            cache.block_pool.blocks[i].ref_cnt > 0 for pair in endpoints for i in pair
        )
        assert not cache.reset_prefix_cache()
    cache.block_pool.free_blocks(retained)
    assert cache.reset_prefix_cache()
    assert cache.get_computed_blocks(b)[1] == 0
    policy = getattr(cache.block_pool, "_study_eviction", None)
    if policy is not None:
        assert not any(policy.reuse_counts)
        assert policy.allocated_blocks == policy.epochs == 0

    # A separate full-block allocator trace is comparable across the flags-off arms.
    pool = BlockPool(5, True, 16)
    hot, cold, active = pool.get_new_blocks(3)
    for block, value in ((hot, b"hot"), (cold, b"cold")):
        block.set_block_hash(value)
        pool.cached_block_hash_to_block.insert(value, block)
    for _ in range(3):
        pool.touch([hot])
        pool.free_blocks([hot])
    pool.free_blocks([hot, cold])
    victims = pool.get_new_blocks(2)
    result = dict(
        source=str(Path(inspect.getfile(KVCacheManager)).resolve()),
        core_imports_local=True,
        coordinator=type(cache.coordinator).__name__,
        scheduler_block_size=sizes[0],
        hash_block_size=sizes[1],
        recovered_tokens=count,
        copies=len(copies),
        different_prefix_misses=True,
        tenant_salt_misses=True,
        pending_copy_refs_protected=True,
        cache_reset=True,
        allocator_victims=[b.block_id for b in victims],
        active_ref=active.ref_cnt,
        kv_tensor_copy_executed=False,
        model_loaded=False,
        cuda_validated=False,
    )
    print("PROBE_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
