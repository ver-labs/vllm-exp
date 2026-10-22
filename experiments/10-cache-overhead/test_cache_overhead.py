"""Differential checks for unchanged reuse decisions and invalidation."""

import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


class FreeOrderingTests(unittest.TestCase):
    def test_identical_victims_after_mixed_pool_operations_and_decay(self):
        for policy in ("lfu", "aged_lfu"):
            pools = []
            for mode in ("reference", "incremental"):
                with patch.dict(
                    os.environ,
                    STUDY_EVICTION_POLICY=policy,
                    STUDY_CACHE_IMPL=mode,
                    STUDY_LFU_HALF_LIFE_BLOCKS="23",
                ):
                    pools.append(BlockPool(65, True, 4))
            rng = random.Random(81)
            for step in range(1500):
                ref = pools[0]
                active = [b.block_id for b in ref.blocks if b.ref_cnt and not b.is_null]
                free = [b.block_id for b in ref.free_block_queue.get_all_free_blocks()]
                cached = [
                    b.block_id for b in ref.blocks if b.block_hash and not b.is_null
                ]
                op = rng.randrange(7)
                if op == 0 and free:
                    count = rng.randint(1, min(5, len(free)))
                    chosen = [
                        [b.block_id for b in p.get_new_blocks(count)] for p in pools
                    ]
                    self.assertEqual(*chosen, (policy, step))
                elif op == 1 and active:
                    ids = rng.sample(active, min(len(active), rng.randint(1, 5)))
                    for pool in pools:
                        pool.free_blocks([pool.blocks[i] for i in ids])
                elif op == 2 and active:
                    bid = rng.choice(active)
                    for pool in pools:
                        b = pool.blocks[bid]
                        if b.block_hash is None:
                            key = make_block_hash_with_group_id(
                                step.to_bytes(32, "big"), 0
                            )
                            pool._insert_block_hash(key, b, num_tokens=4)
                elif op == 3 and cached:
                    bid = rng.choice(cached)
                    for pool in pools:
                        pool.touch([pool.blocks[bid]])
                elif op == 4 and free:
                    bid = rng.choice(free)
                    for pool in pools:
                        pool.evict_blocks({bid})
                elif op == 5 and active:
                    bid = rng.choice(active)
                    for pool in pools:
                        pool.unpin_blocks([pool.blocks[bid]], lambda b: None)
                elif op == 6 and not active:
                    self.assertTrue(all(pool.reset_prefix_cache() for pool in pools))
                self.assertEqual(
                    [
                        b.block_id
                        for b in pools[0].free_block_queue.get_all_free_blocks()
                    ],
                    [
                        b.block_id
                        for b in pools[1].free_block_queue.get_all_free_blocks()
                    ],
                )
                self.assertEqual(
                    pools[0]._study_eviction.reuse_counts,
                    pools[1]._study_eviction.reuse_counts,
                )
                order = pools[1]._study_eviction.order
                self.assertLessEqual(len(order.heap), 8 * len(pools[1].blocks))


# Reuse the existing integration checks for exact lookup, copy retention and salt.
namespace = {}
exec(
    compile(
        Path("experiments/04-partial-prefix/test_partial_prefix.py.txt").read_text(),
        "existing_partial_tests",
        "exec",
    ),
    namespace,
)
PartialPrefixTests = namespace["PartialPrefixTests"]


class IncrementalPartialTests(PartialPrefixTests):
    def setUp(self):
        self.env = patch.dict(os.environ, STUDY_CACHE_IMPL="incremental")
        self.env.start()
        self.addCleanup(self.env.stop)
        super().setUp()

    def test_decode_does_not_rescan_old_prompt_blocks(self):
        self.compute_a()
        publisher = self.cache._study_partial_prefix
        publisher.publish(self.a)  # Drain primary-hash invalidations.
        before = publisher.visited_blocks
        for _ in range(100):
            publisher.publish(self.a)
        self.assertEqual(publisher.visited_blocks, before)

    def test_promotion_copy_move_and_preemption_forget_are_republished(self):
        self.compute_a()
        publisher = self.cache._study_partial_prefix
        mgr = publisher.manager
        block = mgr.req_to_blocks[self.a.request_id][1]
        copy = mgr.block_pool.get_new_blocks(1)[0]
        mgr.block_pool.move_block_hashes(block, copy)
        self.assertIn(1, publisher.dirty[self.a.request_id])
        publisher.publish(self.a)
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 7)
        self.cache.free(self.a)  # Same free/forget path used by scheduler preemption.
        self.assertNotIn(self.a.request_id, publisher.progress)
        self.assertFalse(publisher.owners)
        self.a.num_computed_tokens = 0
        self.compute_a()
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 7)


if __name__ == "__main__":
    unittest.main()
