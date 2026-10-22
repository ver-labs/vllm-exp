"""Check synchronized decay and reference-safe cache selection."""

import os
import unittest
from unittest.mock import patch

from vllm.v1.core.study_aging_eviction import StudyAgingLFUEviction

from vllm.v1.core.block_pool import BlockPool


class AgingLFUTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(
            os.environ, STUDY_EVICTION_POLICY="aged_lfu", STUDY_LFU_HALF_LIFE_BLOCKS="4"
        ):
            self.pool = BlockPool(12, True, 4)
        self.policy = self.pool._study_eviction

    def test_decay_uses_allocated_blocks_and_applies_to_every_count(self):
        live = self.pool.get_new_blocks(3)
        self.policy.reuse_counts[:] = [8] * 12
        self.assertEqual(self.policy.epochs, 0)
        self.pool.get_new_blocks(1)
        self.assertEqual(self.policy.epochs, 1)
        self.assertTrue(all(self.policy.reuse_counts[b.block_id] == 4 for b in live))
        self.assertTrue(all(b.ref_cnt == 1 for b in live))
        self.pool.get_new_blocks(4)
        self.assertEqual(self.policy.epochs, 2)
        self.assertTrue(all(self.policy.reuse_counts[b.block_id] == 2 for b in live))

    def test_multiple_epochs_are_applied_together_and_zero_allocations_do_not_age(self):
        self.policy.reuse_counts[0] = 15
        self.pool.get_new_blocks(8)
        self.assertEqual(self.policy.reuse_counts[0], 3)
        self.assertEqual(self.policy.epochs, 2)
        self.pool.get_new_blocks(0)
        self.assertEqual(self.policy.allocated_blocks, 8)

    def test_stale_hot_score_can_fall_below_a_recently_used_score(self):
        self.policy.reuse_counts[0] = 8
        blocks = self.pool.get_new_blocks(4)
        self.pool.free_blocks(blocks)
        live = self.pool.get_new_blocks(4)
        self.assertEqual(self.policy.reuse_counts[0], 2)
        recent = live[0].block_id
        self.policy.reuse_counts[recent] = 3
        self.pool.get_new_blocks(1)
        self.assertGreater(
            self.policy.reuse_counts[recent], self.policy.reuse_counts[0]
        )

    def test_factory_is_opt_in_and_rejects_invalid_interval(self):
        with patch.dict(os.environ, STUDY_EVICTION_POLICY="vllm"):
            self.assertIsNone(BlockPool(4, True, 4)._study_eviction)
        with (
            patch.dict(os.environ, STUDY_LFU_HALF_LIFE_BLOCKS="0"),
            self.assertRaises(ValueError),
        ):
            StudyAgingLFUEviction(4)


if __name__ == "__main__":
    unittest.main()
