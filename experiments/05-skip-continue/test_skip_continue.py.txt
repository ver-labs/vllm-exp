"""Guard real unchunked admission, queue order, bounded scanning and fairness."""

import os
import unittest
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from vllm.v1.core.sched.study_skip_continue import StudySkipContinue

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager


def request(label, length):
    return Request(label, [1] * length, SamplingParams(max_tokens=1), None)


class SkipContinueTests(unittest.TestCase):
    def scheduler(self, enabled, lookahead=16):
        with patch.dict(
            os.environ,
            STUDY_SKIP_CONTINUE=str(int(enabled)),
            STUDY_SKIP_LOOKAHEAD=str(lookahead),
            STUDY_WAITING_POLICY="fcfs",
            STUDY_PARTIAL_PREFIX="0",
            VLLM_STUDY_TRACE="0",
            HF_HUB_OFFLINE="1",
        ):
            model = ModelConfig(
                model=snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
                max_model_len=512,
                skip_tokenizer_init=True,
            )
            cache = CacheConfig(block_size=16, enable_prefix_caching=False)
            cache.num_gpu_blocks = 256
            config = VllmConfig(
                model_config=model,
                cache_config=cache,
                scheduler_config=SchedulerConfig(
                    is_encoder_decoder=False,
                    max_model_len=512,
                    max_num_seqs=8,
                    max_num_batched_tokens=512,
                    enable_chunked_prefill=False,
                    async_scheduling=False,
                ),
            )
            kv = KVCacheConfig(
                num_blocks=256,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(
                        ["layer"],
                        FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=8,
                            dtype=torch.float32,
                        ),
                    )
                ],
            )
            from vllm.v1.core.single_type_kv_cache_manager import (
                register_all_kvcache_specs,
            )

            register_all_kvcache_specs(config)
            return Scheduler(config, kv, StructuredOutputManager(config), 16)

    def test_actual_scheduler_skips_c_admits_d_and_restores_waiting_order(self):
        for enabled, expected in [(False, ["A"]), (True, ["A", "D"])]:
            s = self.scheduler(enabled)
            for label, n in [("A", 400), ("C", 400), ("D", 80), ("E", 400)]:
                s.add_request(request(label, n))
            out = s.schedule()
            self.assertEqual(list(out.num_scheduled_tokens), expected)
            self.assertEqual(
                [r.request_id for r in s.waiting],
                ["C", "E"] if enabled else ["C", "D", "E"],
            )
            self.assertEqual(
                sum(out.num_scheduled_tokens.values()), 480 if enabled else 400
            )
            self.assertEqual(s.requests["C"].num_computed_tokens, 0)

    def test_actual_scheduler_honors_scan_limit(self):
        s = self.scheduler(True, lookahead=2)
        for label, n in [("A", 400), ("C", 400), ("E", 400), ("D", 80)]:
            s.add_request(request(label, n))
        out = s.schedule()
        self.assertEqual(list(out.num_scheduled_tokens), ["A"])
        self.assertEqual([r.request_id for r in s.waiting], ["C", "E", "D"])

    def test_age_and_bypass_limit_protect_head_and_admission_clears_history(self):
        p = StudySkipContinue(max_bypasses=2, max_wait_s=5)
        r = request("C", 10)
        now = r.arrival_time
        self.assertTrue(p.can_skip(r, ordinary_waiting=True, now=now))
        p.begin_step()
        self.assertTrue(p.can_skip(r, ordinary_waiting=True, now=now))
        p.begin_step()
        self.assertFalse(p.can_skip(r, ordinary_waiting=True, now=now))
        p.admitted(r)
        self.assertNotIn(r, p.bypasses)
        self.assertFalse(p.can_skip(r, ordinary_waiting=True, now=now + 5))

    def test_preempted_or_kv_holding_requests_cannot_be_bypassed(self):
        p = StudySkipContinue()
        r = request("C", 10)
        self.assertFalse(p.can_skip(r, ordinary_waiting=False))
        r.status = RequestStatus.PREEMPTED
        self.assertFalse(p.can_skip(r, ordinary_waiting=True))
        r.status = RequestStatus.WAITING
        r.num_preemptions = 1
        self.assertFalse(p.can_skip(r, ordinary_waiting=True))

    def test_actual_scheduler_stops_at_age_or_bypass_protected_head(self):
        for protection in ["age", "bypasses"]:
            scheduler = self.scheduler(True)
            a, c, d = request("A", 400), request("C", 400), request("D", 80)
            if protection == "age":
                c.arrival_time -= 31
            else:
                scheduler._study_skip_continue.bypasses[c] = 4
            for r in [a, c, d]:
                scheduler.add_request(r)
            out = scheduler.schedule()
            self.assertEqual(list(out.num_scheduled_tokens), ["A"])
            self.assertEqual([r.request_id for r in scheduler.waiting], ["C", "D"])
