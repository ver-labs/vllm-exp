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


class PagedRepairTests(unittest.TestCase):
    """Catch stale-page reuse and verify writes feed normal native decode."""

    def setUp(self):
        from types import SimpleNamespace

        from study_nonprefix_worker import PagedRepair
        from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
        from vllm_metal.metal import get_ops

        mx.set_default_device(mx.gpu)
        mx.random.seed(9)
        self.model = Model(
            ModelArgs(
                model_type="qwen3",
                hidden_size=128,
                num_hidden_layers=2,
                intermediate_size=256,
                num_attention_heads=2,
                rms_norm_eps=1e-6,
                vocab_size=128,
                num_key_value_heads=2,
                max_position_embeddings=128,
                rope_theta=1000000.0,
                head_dim=64,
                tie_word_embeddings=True,
            )
        )
        self.model.set_dtype(mx.float16)
        self.model.eval()
        self.adapter = NonPrefixRepair(self.model)
        self.doc = list(range(20, 36))
        self.target = [3, 4] + self.doc + [8, 9]
        self.dense = self.adapter.dense([], self.target, [10]).evaluate()
        self.source = self.adapter.prepare([], self.doc)
        self.paged = MetalPagedKVCache(2, 2, 64, 8, 16, mx.float16)
        for i, (k, v) in enumerate(self.source.layers):
            nk, nv = get_ops().reshape_and_cache(
                mx.contiguous(k[0].transpose(1, 0, 2)),
                mx.contiguous(v[0].transpose(1, 0, 2)),
                self.paged.key_caches[i],
                self.paged.value_caches[i],
                mx.array(list(range(32, 48)), dtype=mx.int64),
            )
            self.paged.replace_layer_cache(i, nk, nv)
        mx.eval(*self.paged.key_caches, *self.paged.value_caches)
        self.study = PagedRepair(SimpleNamespace(_forward_model=self.model), 1)
        self.study.cache = self.paged
        for layer in self.model.layers:
            object.__setattr__(layer.self_attn, "_inner", layer.self_attn)

    def tearDown(self):
        for layer in self.model.layers:
            object.__delattr__(layer.self_attn, "_inner")

    def test_full_paged_repair_matches_dense_and_continuation(self):
        from study_nonprefix_worker import ResidentBlock

        source = ResidentBlock(2, tuple(self.doc), ())
        hidden = self.study.forward(
            [("test", self.target, 0, [[4, 5]])], {"test": [(2, source)]}
        )
        # Continue from the written paged KV, without rebuilding a private cache.
        hidden = self.study.forward([("test", [10], len(self.target), [[4, 5]])], {})
        logits = self.model.model.embed_tokens.as_linear(hidden)[:, -1, :]
        mx.eval(logits)
        error = float(mx.max(mx.abs(logits - self.dense.logits)).item())
        self.assertLess(error, 0.08)

    def test_sparse_paged_repair_preserves_source_and_prefix_causality(self):
        from study_nonprefix_worker import ResidentBlock

        self.study.ratio = 0.1
        before = [self.study.gather(i, list(range(32, 48))).tolist() for i in range(2)]
        source = ResidentBlock(2, tuple(self.doc), ())
        hidden = self.study.forward(
            [("test", self.target, 0, [[4, 5]])], {"test": [(2, source)]}
        )
        mx.eval(hidden, *self.paged.key_caches, *self.paged.value_caches)
        for i in range(2):
            self.assertEqual(
                before[i], self.study.gather(i, list(range(32, 48))).tolist()
            )
        # Early keys cannot depend on the later reused document.
        reference = NonPrefixRepair(self.model).prepare([], self.target)
        for i in range(2):
            actual = self.study.gather(i, [64, 65])
            error = float(
                mx.max(mx.abs(actual - reference.layers[i][0][..., :2, :])).item()
            )
            self.assertLess(error, 0.02)

    def test_block_index_rejects_written_and_identical_prefix_pages(self):
        from study_nonprefix_worker import BlockIndex, ResidentBlock

        index = BlockIndex()
        index.blocks[7] = ResidentBlock(7, (1, 2, 3, 4), (8,))
        self.assertEqual(index.matches([1, 2, 3, 4], [8], set(), 4), [])
        self.assertEqual(index.matches([1, 2, 3, 4], [9], {7}, 4), [])
        self.assertEqual(len(index.matches([1, 2, 3, 4], [9], set(), 4)), 1)
        index.invalidate({7})
        self.assertEqual(index.matches([1, 2, 3, 4], [9], set(), 4), [])
