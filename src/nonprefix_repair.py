"""Session-local, approximate non-prefix KV reuse for unscaled MLX Qwen3.

This is a rule-based research prototype, not a CacheBlend implementation.
Only selected document rows pass through attention and MLP. Unselected document
KV rows retain their old-context values; causal queries use absolute positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Model


@dataclass(frozen=True)
class CachedChunk:
    owner: object
    token_ids: tuple[int, ...]
    source_prefix_length: int
    layers: tuple[tuple[mx.array, mx.array], ...]

    @property
    def nbytes(self):
        return sum(k.nbytes + v.nbytes for k, v in self.layers)


@dataclass
class Prefill:
    logits: mx.array
    cache: list[KVCache]
    repaired_document_positions: tuple[int, ...]
    recomputed_rows_per_layer: int
    reused_rows_per_layer: int

    def evaluate(self):
        mx.eval(self.logits, [item.state for item in self.cache])
        mx.synchronize()
        return self


def select_rows(length: int, ratio: float, strategy: str = "boundary"):
    if length <= 0 or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("Need a nonempty document and a finite ratio in [0, 1]")
    count = math.ceil(length * ratio)
    if strategy == "boundary":
        return tuple(range(count))
    if strategy == "uniform":
        return tuple(index * length // count for index in range(count))
    raise ValueError("Selection strategy must be boundary or uniform")


def _rotate(x, positions, rope):
    # Qwen3 uses nontraditional, unscaled, full-head RoPE. Compute phases in f32.
    dims = x.shape[-1]
    frequency = mx.exp(
        -mx.arange(0, dims, 2, dtype=mx.float32) * math.log(rope.base) / dims
    )
    phase = mx.array(tuple(positions), dtype=mx.float32)[:, None] * frequency[None, :]
    cosine, sine = mx.cos(phase)[None, None], mx.sin(phase)[None, None]
    left, right = mx.split(x.astype(mx.float32), 2, axis=-1)
    return mx.concatenate(
        [left * cosine - right * sine, right * cosine + left * sine], axis=-1
    ).astype(x.dtype)


class NonPrefixRepair:
    def __init__(self, model):
        if not isinstance(model, Model) or model.args.rope_scaling is not None:
            raise ValueError("Prototype supports only MLX Qwen3 with unscaled RoPE")
        for layer in model.layers:
            rope = layer.self_attn.rope
            if (
                type(rope) is not nn.RoPE
                or rope.traditional
                or rope.scale != 1
                or rope.dims != model.args.head_dim
                or rope.dims % 2
            ):
                raise ValueError("Unsupported positional encoding")
        self.model = model
        self._owner = object()
        model.eval()

    def _ids(self, values, *, allow_empty=False):
        values = tuple(values)
        if not values and not allow_empty:
            raise ValueError("Token segment must not be empty")
        if any(
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or not 0 <= value < self.model.args.vocab_size
            for value in values
        ):
            raise ValueError("Token IDs must be integers inside the model vocabulary")
        return tuple(int(value) for value in values)

    def _check_length(self, length):
        if length > self.model.args.max_position_embeddings:
            raise ValueError("Input exceeds model positional limit")

    def _head(self, hidden):
        if self.model.args.tie_word_embeddings:
            return self.model.model.embed_tokens.as_linear(hidden)
        return self.model.lm_head(hidden)

    def dense(self, prefix, document, suffix):
        prefix = self._ids(prefix, allow_empty=True)
        document, suffix = self._ids(document), self._ids(suffix)
        self._check_length(len(prefix + document + suffix))
        cache = [KVCache() for _ in self.model.layers]
        hidden = self.model.model(mx.array([prefix + document + suffix]), cache=cache)
        logits = self._head(hidden[:, -1:, :])[:, -1, :]
        return Prefill(
            logits,
            cache,
            tuple(range(len(document))),
            len(prefix) + len(document) + len(suffix),
            0,
        )

    def prepare(self, source_prefix, document):
        prefix = self._ids(source_prefix, allow_empty=True)
        document = self._ids(document)
        cache = [KVCache() for _ in self.model.layers]
        self._check_length(len(prefix + document))
        # No suffix is required: causal document KV cannot depend on future text.
        self.model.model(mx.array([prefix + document]), cache=cache)
        layers = tuple(
            (c.state[0][..., len(prefix) :, :], c.state[1][..., len(prefix) :, :])
            for c in cache
        )
        mx.eval(layers)
        mx.synchronize()
        return CachedChunk(self._owner, document, len(prefix), layers)

    def repair(self, chunk, prefix, document, suffix, ratio, strategy="boundary"):
        if chunk.owner is not self._owner:
            raise ValueError("Cached chunk belongs to another adapter/model session")
        prefix = self._ids(prefix, allow_empty=True)
        document, suffix = self._ids(document), self._ids(suffix)
        if document != chunk.token_ids:
            raise ValueError("Shared document token IDs must match exactly")
        selected = select_rows(len(document), ratio, strategy)
        p, d, total = len(prefix), len(document), len(prefix + document + suffix)
        self._check_length(total)
        positions = tuple(range(p)) + tuple(p + index for index in selected)
        positions += tuple(range(p + d, total))
        indices = mx.array(positions, dtype=mx.int32)
        tokens = mx.array([prefix + document + suffix])
        hidden = self.model.model.embed_tokens(tokens[:, indices])
        mask = indices[:, None] >= mx.arange(total)[None, :]
        cache = []
        shift = p - chunk.source_prefix_length
        for layer, (old_keys, old_values) in zip(self.model.layers, chunk.layers):
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            length = len(positions)
            q = attention.q_norm(
                attention.q_proj(normalized).reshape(1, length, attention.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            k = attention.k_norm(
                attention.k_proj(normalized).reshape(
                    1, length, attention.n_kv_heads, -1
                )
            ).transpose(0, 2, 1, 3)
            v = (
                attention.v_proj(normalized)
                .reshape(1, length, attention.n_kv_heads, -1)
                .transpose(0, 2, 1, 3)
            )
            q, k = (
                _rotate(q, positions, attention.rope),
                _rotate(k, positions, attention.rope),
            )
            moved_keys = (
                old_keys
                if shift == 0
                else _rotate(old_keys, [shift] * d, attention.rope)
            )
            # Fresh assembled arrays: never overwrite the reusable source chunk.
            shape_p = (1, attention.n_kv_heads, p, k.shape[-1])
            shape_s = (1, attention.n_kv_heads, len(suffix), k.shape[-1])
            keys = mx.concatenate(
                [mx.zeros(shape_p, k.dtype), moved_keys, mx.zeros(shape_s, k.dtype)],
                axis=2,
            )
            values = mx.concatenate(
                [mx.zeros(shape_p, v.dtype), old_values, mx.zeros(shape_s, v.dtype)],
                axis=2,
            )
            keys[..., indices, :] = k
            values[..., indices, :] = v
            output = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=attention.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(1, length, -1)
            hidden = hidden + attention.o_proj(output)
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            item = KVCache()
            item.state = (keys, values)
            cache.append(item)
        logits = self._head(self.model.model.norm(hidden[:, -1:, :]))[:, -1, :]
        return Prefill(logits, cache, selected, len(positions), d - len(selected))
