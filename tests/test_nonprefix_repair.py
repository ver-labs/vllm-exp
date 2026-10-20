"""Tiny random Qwen3 correctness checks; no pretrained model or benchmark run."""

import sys
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3 import Model, ModelArgs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nonprefix_repair import NonPrefixRepair, _rotate, select_rows


class NonPrefixRepairTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.cpu)
        mx.random.seed(7)
        self.model = Model(
            ModelArgs(
                model_type="qwen3",
                hidden_size=32,
                num_hidden_layers=3,
                intermediate_size=64,
                num_attention_heads=4,
                rms_norm_eps=1e-6,
                vocab_size=128,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_theta=1000000.0,
                head_dim=8,
                tie_word_embeddings=True,
            )
        )
        self.adapter = NonPrefixRepair(self.model)
        self.old = [1, 2, 3]
        self.doc = [11, 12, 13, 14, 15, 16]
        self.suffix = [21, 22]
        self.chunk = self.adapter.prepare(self.old, self.doc)

    def close(self, a, b, tolerance=2e-4):
        error = float(
            mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        )
        self.assertLess(error, tolerance)

    def test_full_repair_matches_native_dense_and_cache_at_shifted_position(self):
        prefix = [4, 5, 6, 7, 8]
        dense = self.adapter.dense(prefix, self.doc, self.suffix).evaluate()
        repaired = self.adapter.repair(
            self.chunk, prefix, self.doc, self.suffix, 1
        ).evaluate()
        self.close(repaired.logits, dense.logits)
        for a, b in zip(repaired.cache, dense.cache):
            self.close(a.state[0], b.state[0])
            self.close(a.state[1], b.state[1])
        self.assertEqual(repaired.reused_rows_per_layer, 0)

    def test_identical_prefix_zero_repair_matches_dense(self):
        dense = self.adapter.dense(self.old, self.doc, self.suffix).evaluate()
        reused = self.adapter.repair(
            self.chunk, self.old, self.doc, self.suffix, 0
        ).evaluate()
        self.close(reused.logits, dense.logits)
        self.assertEqual(reused.reused_rows_per_layer, len(self.doc))

    def test_partial_uniform_repair_is_causal_and_does_not_mutate_source(self):
        prefix = [4, 5, 6, 7]
        before = [(k.tolist(), v.tolist()) for k, v in self.chunk.layers]
        repaired = self.adapter.repair(
            self.chunk, prefix, self.doc, self.suffix, 0.5, "uniform"
        ).evaluate()
        prefix_cache = self.adapter.prepare([], prefix)
        for actual, (expected_k, expected_v) in zip(
            repaired.cache, prefix_cache.layers
        ):
            self.close(actual.state[0][..., : len(prefix), :], expected_k)
            self.close(actual.state[1][..., : len(prefix), :], expected_v)
        self.assertEqual(
            before, [(k.tolist(), v.tolist()) for k, v in self.chunk.layers]
        )
        self.assertEqual(repaired.repaired_document_positions, (0, 2, 4))
        self.assertEqual(repaired.recomputed_rows_per_layer, 9)
        self.assertEqual(repaired.reused_rows_per_layer, 3)

    def test_zero_repair_under_changed_context_can_differ_from_dense(self):
        prefix = [90, 91, 92]
        dense = self.adapter.dense(prefix, self.doc, self.suffix).evaluate()
        reused = self.adapter.repair(
            self.chunk, prefix, self.doc, self.suffix, 0
        ).evaluate()
        error = float(mx.max(mx.abs(dense.logits - reused.logits)).item())
        self.assertGreater(error, 1e-6)

    def test_positional_shift_matches_native_rope_including_negative_shift(self):
        rope = self.model.layers[0].self_attn.rope
        x = mx.random.normal((1, 2, 6, 8))
        self.close(_rotate(x, range(4, 10), rope), rope(x, offset=4), 2e-5)
        moved = _rotate(rope(x, offset=7), [-4] * 6, rope)
        self.close(moved, rope(x, offset=3), 2e-5)

    def test_decode_after_full_repair_matches_dense_decode(self):
        prefix = [4, 5]
        dense = self.adapter.dense(prefix, self.doc, self.suffix).evaluate()
        repaired = self.adapter.repair(
            self.chunk, prefix, self.doc, self.suffix, 1
        ).evaluate()
        token = mx.argmax(dense.logits, axis=-1).reshape(1, 1)
        a = self.model(token, cache=dense.cache)
        b = self.model(token, cache=repaired.cache)
        self.close(a, b)

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_tiny_metal_forward_and_sparse_repair(self):
        mx.set_default_device(mx.gpu)
        chunk = self.adapter.prepare(self.old, self.doc)
        prefix = [7, 8, 9, 10]
        dense = self.adapter.dense(prefix, self.doc, self.suffix).evaluate()
        full = self.adapter.repair(chunk, prefix, self.doc, self.suffix, 1).evaluate()
        self.close(full.logits, dense.logits)
        sparse = self.adapter.repair(
            chunk, prefix, self.doc, self.suffix, 0.5, "uniform"
        ).evaluate()
        self.assertTrue(bool(mx.all(mx.isfinite(sparse.logits)).item()))

    @unittest.skipUnless(mx.metal.is_available(), "Metal is unavailable")
    def test_tiny_bfloat16_metal_full_repair(self):
        mx.set_default_device(mx.gpu)
        self.model.set_dtype(mx.bfloat16)
        adapter = NonPrefixRepair(self.model)
        chunk = adapter.prepare(self.old, self.doc)
        prefix = [7, 8, 9, 10]
        dense = adapter.dense(prefix, self.doc, self.suffix).evaluate()
        repaired = adapter.repair(chunk, prefix, self.doc, self.suffix, 1).evaluate()
        self.close(dense.logits, repaired.logits, tolerance=0.03)

    def test_invalid_inputs_and_cross_session_cache_are_rejected(self):
        for ratio in [-1, 1.01, float("nan")]:
            with self.assertRaises(ValueError):
                select_rows(6, ratio)
        self.assertEqual(select_rows(6, 0, "uniform"), ())
        with self.assertRaises(ValueError):
            self.adapter.repair(self.chunk, self.old, [99], self.suffix, 0.5)
        with self.assertRaises(ValueError):
            NonPrefixRepair(self.model).repair(
                self.chunk, self.old, self.doc, self.suffix, 0.5
            )
        with self.assertRaises(ValueError):
            self.adapter.repair(self.chunk, self.old, self.doc, [], 0.5)


if __name__ == "__main__":
    unittest.main()
