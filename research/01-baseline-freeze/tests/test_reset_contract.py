"""Guard defined cold policy state and failed-reset atomicity."""

import os
import unittest
from unittest.mock import patch

from vllm.v1.core.block_pool import BlockPool


class ResetContractTests(unittest.TestCase):
    def make_pool(self):
        with patch.dict(
            os.environ,
            STUDY_EVICTION_POLICY="aged_lfu",
            STUDY_LFU_HALF_LIFE_BLOCKS="4",
        ):
            return BlockPool(16, True, 16)

    def test_successful_reset_restarts_the_allocation_epoch(self):
        pool = self.make_pool()
        blocks = pool.get_new_blocks(5)
        pool.free_blocks(blocks)
        policy = pool._study_eviction
        policy.reuse_counts[:] = [7] * 16
        self.assertEqual((policy.allocated_blocks, policy.epochs), (5, 1))
        self.assertTrue(pool.reset_prefix_cache())
        self.assertEqual((policy.allocated_blocks, policy.epochs), (0, 0))
        self.assertFalse(any(policy.reuse_counts))
        pool.get_new_blocks(3)
        self.assertEqual(policy.epochs, 0)
        pool.get_new_blocks(1)
        self.assertEqual(policy.epochs, 1)

    def test_failed_reset_preserves_live_ownership_and_policy_state(self):
        pool = self.make_pool()
        blocks = pool.get_new_blocks(5)
        policy = pool._study_eviction
        policy.reuse_counts[:] = [7] * 16
        before = (policy.allocated_blocks, policy.epochs, policy.reuse_counts[:])
        self.assertFalse(pool.reset_prefix_cache())
        self.assertEqual(
            (policy.allocated_blocks, policy.epochs, policy.reuse_counts), before
        )
        self.assertTrue(all(b.ref_cnt == 1 for b in blocks))

    def test_combined_reset_removes_publication_metadata_and_clock(self):
        from probe import make_cache, request

        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import init_none_hash

        init_none_hash(sha256)
        with patch.dict(
            os.environ,
            STUDY_PARTIAL_PREFIX="1",
            STUDY_EVICTION_POLICY="aged_lfu",
            STUDY_LFU_HALF_LIFE_BLOCKS="2",
        ):
            cache, sizes = make_cache()
        a = request("A", list(range(1, 33)), sizes[1])
        cache.allocate_slots(a, 32)
        a.num_computed_tokens = 32
        index = cache._study_partial_prefix
        index.publish(a)
        self.assertTrue(index.progress)
        self.assertGreater(index.visited_blocks, 0)
        cache.free(a)
        self.assertTrue(cache.reset_prefix_cache())
        self.assertFalse(index.progress or index.owners or index.valid or index.dirty)
        self.assertEqual(index.visited_blocks, 0)
        policy = cache.block_pool._study_eviction
        self.assertEqual((policy.allocated_blocks, policy.epochs), (0, 0))


if __name__ == "__main__":
    unittest.main()
