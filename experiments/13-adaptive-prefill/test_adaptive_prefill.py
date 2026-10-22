"""Policy contract: load/age caps and opt-in configuration boundaries."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace as R
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "policy", ROOT / "vllm-src/vllm/v1/core/sched/study_adaptive_prefill.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestAdaptivePrefill(unittest.TestCase):
    def setUp(self):
        self.policy = module.StudyAdaptivePrefill()
        self.decoder = R(
            num_output_tokens=2,
            num_computed_tokens=65,
            num_prompt_tokens=64,
            arrival_time=0,
        )
        self.prompt = R(
            num_output_tokens=0,
            num_computed_tokens=0,
            num_prompt_tokens=1024,
            arrival_time=9.9,
        )

    def test_load_shrinks_cap_and_prompt_age_restores_progress(self):
        self.assertEqual(self.policy.choose([], [self.prompt], 128, 10)[0], 128)
        self.assertEqual(
            self.policy.choose([self.decoder] * 4, [self.prompt], 128, 10)[0], 16
        )
        self.assertEqual(
            self.policy.choose([self.decoder] * 4, [self.prompt], 128, 11)[0], 64
        )

    def test_budget_is_never_exceeded_and_no_prompt_has_no_age_boost(self):
        for budget in (1, 8, 16, 128):
            cap = self.policy.choose([self.decoder] * 4, [self.prompt], budget, 11)[0]
            self.assertTrue(0 < cap <= budget)
        self.assertFalse(self.policy.choose([self.decoder] * 4, [], 128, 11)[3])

    def test_clock_boundary_and_uncomputed_prompt_are_not_decoders(self):
        cap, decoders, age, _ = self.policy.choose([self.prompt], [], 128, 9)
        self.assertEqual((cap, decoders, age), (128, 0, 0))

    def test_disabled_path_and_incompatible_modes_fail_early(self):
        scheduler = R(
            scheduler_config=R(enable_chunked_prefill=True),
            _study_skip_continue=None,
            adaptive_long_prefill_threshold=False,
        )
        with patch.dict(os.environ, {"STUDY_ADAPTIVE_PREFILL": "0"}):
            self.assertIsNone(module.StudyAdaptivePrefill.from_environment(scheduler))
        with patch.dict(os.environ, {"STUDY_ADAPTIVE_PREFILL": "1"}):
            self.assertIsNotNone(
                module.StudyAdaptivePrefill.from_environment(scheduler)
            )
            for field, value in (
                ("_study_skip_continue", object()),
                ("adaptive_long_prefill_threshold", True),
            ):
                old = getattr(scheduler, field)
                setattr(scheduler, field, value)
                with self.assertRaises(ValueError):
                    module.StudyAdaptivePrefill.from_environment(scheduler)
                setattr(scheduler, field, old)
            scheduler.scheduler_config.enable_chunked_prefill = False
            with self.assertRaises(ValueError):
                module.StudyAdaptivePrefill.from_environment(scheduler)


if __name__ == "__main__":
    unittest.main()
