"""Ensure threshold rejection keeps full blocks and avoids partial CoW."""

import os
import unittest
from unittest.mock import patch

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request


class ThresholdTests(unittest.TestCase):
    def test_all_offsets_match_rounded_savings_and_only_accepted_hits_copy(self):
        init_none_hash(sha256)
        for granularity in (1, 4, 8):
            for minimum in (0, 8, 16):
                spec = FullAttentionSpec(
                    block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32
                )
                config = KVCacheConfig(
                    num_blocks=64,
                    kv_cache_tensors=[],
                    kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
                )
                with patch.dict(
                    os.environ,
                    STUDY_PARTIAL_PREFIX="1",
                    STUDY_PARTIAL_MIN_TOKENS=str(minimum),
                ):
                    cache = KVCacheManager(config, 256, 16, granularity)

                def request(label, tokens, granularity=granularity):
                    return Request(
                        label,
                        tokens,
                        SamplingParams(max_tokens=1),
                        None,
                        block_hasher=get_request_block_hasher(granularity, sha256),
                    )

                tokens = list(range(1, 129))
                a = request("A", tokens)
                cache.allocate_slots(a, 128)
                a.num_computed_tokens = 128
                cache._study_partial_prefix.publish(a)
                cache.free(a)
                for offset in range(1, 16):
                    b_tokens = tokens.copy()
                    b_tokens[112 + offset] = 999
                    b = request(str(offset), b_tokens)
                    blocks, count, _ = cache.get_computed_blocks(b)
                    extra = offset // granularity * granularity
                    expected = 112 + (extra if extra >= minimum else 0)
                    self.assertEqual(count, expected, (granularity, minimum, offset))
                    cache.allocate_slots(
                        b,
                        128 - count,
                        new_computed_blocks=blocks,
                        num_new_computed_tokens=count,
                    )
                    copies, retained = cache.take_kv_cache_block_copies()
                    self.assertEqual(len(copies), int(expected > 112))
                    cache.free(b)
                    cache.block_pool.free_blocks(retained)


if __name__ == "__main__":
    unittest.main()
