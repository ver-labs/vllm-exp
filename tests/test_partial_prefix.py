"""Check exact prefix lookup, copy ownership and alias invalidation."""

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


class PartialPrefixTests(unittest.TestCase):
    def setUp(self):
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32
                    ),
                )
            ],
        )
        with patch.dict(os.environ, STUDY_PARTIAL_PREFIX="1"):
            self.cache = KVCacheManager(config, 64, 4, 1)
        self.a = self.request("A", list(range(1, 9)))
        self.b = self.request("B", list(range(1, 8)) + [99])

    def request(self, label, tokens, salt=None):
        return Request(
            label,
            tokens,
            SamplingParams(max_tokens=8),
            None,
            cache_salt=salt,
            block_hasher=get_request_block_hasher(1, sha256),
        )

    def compute_a(self, count=8):
        self.cache.allocate_slots(self.a, count)
        self.a.num_computed_tokens = count
        self.cache._study_partial_prefix.publish(self.a)

    def test_only_completed_kv_is_published_then_seven_tokens_hit(self):
        self.cache.allocate_slots(self.a, 8)
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 4)
        self.a.num_computed_tokens = 8
        self.cache._study_partial_prefix.publish(self.a)
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 7)
        c = self.request("C", [1, 2, 3, 4, 5, 6, 99, 98])
        self.assertEqual(self.cache.get_computed_blocks(c)[1], 6)
        other = self.request("salt", list(self.b.prompt_token_ids), salt="isolated")
        self.assertEqual(self.cache.get_computed_blocks(other)[1], 0)

    def test_partial_hit_gets_private_tail_and_retains_copy_endpoints(self):
        self.compute_a()
        hit, count, _ = self.cache.get_computed_blocks(self.b)
        source = hit.blocks[0][-1]
        self.cache.allocate_slots(self.b, 1, count, hit)
        target = self.cache.coordinator.single_type_managers[0].req_to_blocks["B"][-1]
        self.assertNotEqual(source.block_id, target.block_id)
        copies, retained = self.cache.take_kv_cache_block_copies()
        self.assertEqual(
            [(c.src_block_id, c.dst_block_id) for c in copies],
            [(source.block_id, target.block_id)],
        )
        self.cache.free(self.a)
        self.cache.free(self.b)
        self.assertGreater(source.ref_cnt, 0)
        self.assertGreater(target.ref_cnt, 0)
        self.cache.block_pool.free_blocks(retained)
        self.assertEqual(source.ref_cnt, 0)
        self.assertEqual(target.ref_cnt, 0)

    def test_eviction_and_reset_remove_every_interior_alias(self):
        self.compute_a()
        tail = self.cache.coordinator.single_type_managers[0].req_to_blocks["A"][-1]
        self.cache.free(self.a)
        self.cache.evict_blocks({tail.block_id})
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 4)
        self.assertTrue(self.cache.reset_prefix_cache())
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 0)
        self.assertFalse(self.cache._study_partial_prefix.progress)

    def test_partial_to_full_promotion_restores_shorter_aliases(self):
        self.compute_a(6)
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 6)
        self.cache.allocate_slots(self.a, 2)
        self.a.num_computed_tokens = 8
        self.cache._study_partial_prefix.publish(self.a)
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 7)
        short = self.request("short", [1, 2, 3, 4, 5, 99])
        self.assertEqual(self.cache.get_computed_blocks(short)[1], 5)

    def test_configuration_resolves_one_token_matching_with_four_token_pages(self):
        from types import SimpleNamespace

        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=4, prefix_match_unit=1, enable_prefix_caching=True
            ),
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        )
        self.assertEqual(
            resolve_kv_cache_block_sizes(self.cache.kv_cache_config, config), (4, 1)
        )
        config.cache_config.prefix_match_unit = 3
        with self.assertRaises(ValueError):
            resolve_kv_cache_block_sizes(self.cache.kv_cache_config, config)

    def test_disabled_index_keeps_full_block_matching_and_deferred_free_cleans_up(self):
        with patch.dict(os.environ, STUDY_PARTIAL_PREFIX="0"):
            stock = KVCacheManager(self.cache.kv_cache_config, 64, 4, 1)
        stock.allocate_slots(self.a, 8)
        self.assertIsNone(stock._study_partial_prefix)
        self.assertEqual(stock.get_computed_blocks(self.b)[1], 4)
        self.compute_a()
        self.assertIn("A", self.cache._study_partial_prefix.progress)
        blocks = self.cache.pop_blocks_for_free(self.a)
        self.assertNotIn("A", self.cache._study_partial_prefix.progress)
        self.cache.block_pool.free_blocks(blocks)
