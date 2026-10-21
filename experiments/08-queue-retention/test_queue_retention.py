"""Selection must retain useful prefixes without pinning or blocking allocation."""

import os
import unittest
from types import SimpleNamespace
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
from vllm.v1.request import Request, RequestStatus


class QueueRetentionTests(unittest.TestCase):
    def setUp(self):
        init_none_hash(sha256)
        config = KVCacheConfig(
            num_blocks=9,
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
        with patch.dict(
            os.environ, STUDY_EVICTION_POLICY="queue", STUDY_PARTIAL_PREFIX="0"
        ):
            self.cache = KVCacheManager(config, 64, 4, 4)
        self.pool = self.cache.block_pool
        self.policy = self.pool._study_eviction
        self.policy.scheduler = SimpleNamespace(
            waiting=[], kv_cache_manager=self.cache, _study_trace=None
        )
        self.a = self.request("A", list(range(1, 9)))
        self.c = self.request("C", list(range(20, 44)))
        self.cache.allocate_slots(self.a, 8)
        self.cache.allocate_slots(self.c, 24)
        self.a_ids = {b.block_id for b in self.cache.get_blocks("A").blocks[0]}
        self.cache.free(self.a)
        self.cache.free(self.c)
        self.b = self.request("B", list(range(1, 9)) + [99])
        self.policy.scheduler.waiting = [self.b]

    def request(self, label, tokens, salt=None):
        return Request(
            label,
            tokens,
            SamplingParams(max_tokens=8),
            None,
            cache_salt=salt,
            block_hasher=get_request_block_hasher(4, sha256),
        )

    def test_queued_prefix_is_retained_without_touching_or_reordering(self):
        before = [b.ref_cnt for b in self.pool.blocks]
        demand, _ = self.policy.collect_demand()
        self.assertEqual(set(demand), self.a_ids)
        self.assertEqual(before, [b.ref_cnt for b in self.pool.blocks])
        allocated = self.pool.get_new_blocks(3)
        self.assertFalse(self.a_ids & {b.block_id for b in allocated})
        self.assertEqual(self.cache.get_computed_blocks(self.b)[1], 8)
        self.assertEqual(self.policy.scheduler.waiting, [self.b])
        self.assertTrue(all(self.pool.blocks[i].ref_cnt == 0 for i in self.a_ids))

    def test_soft_retention_yields_when_all_free_blocks_are_needed(self):
        allocated = self.pool.get_new_blocks(8)
        self.assertEqual(len(allocated), 8)
        self.assertTrue(self.a_ids <= {b.block_id for b in allocated})
        self.assertTrue(all(b.ref_cnt == 1 for b in allocated))

    def test_different_preceding_context_and_salt_do_not_create_demand(self):
        for request in [
            self.request("different", [90, 91, 92, 93, 5, 6, 7, 8, 99]),
            self.request("salted", list(range(1, 9)) + [99], "other"),
        ]:
            self.policy.scheduler.waiting = [request]
            self.assertFalse(self.policy.collect_demand()[0])

    def test_evicted_cache_is_not_protected_by_stale_page_ids(self):
        self.assertTrue(self.policy.collect_demand()[0])
        self.cache.evict_blocks(self.a_ids)
        self.assertFalse(self.policy.collect_demand()[0])

    def test_blocked_requests_and_requests_beyond_lookahead_do_not_pin(self):
        blocked = self.request("blocked", list(range(1, 9)) + [99])
        blocked.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
        self.policy.lookahead = 1
        self.policy.scheduler.waiting = [blocked, self.b]
        self.assertFalse(self.policy.collect_demand()[0])

    def test_deferred_reader_retains_last_resort_priority(self):
        self.policy.scheduler.waiting = []
        watched = self.pool.free_block_queue.get_all_free_blocks()[0]
        self.pool._reuse_watchers[watched.block_id] = lambda block: None
        allocated = self.pool.get_new_blocks(7)
        self.assertNotIn(watched.block_id, {b.block_id for b in allocated})


if __name__ == "__main__":
    unittest.main()
