"""Opt-in Qwen3/Metal non-prefix reuse within vLLM's allocated KV pool."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3 import Model
from nonprefix_repair import _rotate, select_rows
from vllm_metal.attention.impls.bidi_prefill import gather_kv
from vllm_metal.metal import get_ops
from vllm_metal.v1.model_adapter import (
    IntermediateForwardOutput,
    TargetModelForwardOutput,
)
from vllm_metal.v1.worker import MetalWorker

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class ResidentBlock:
    block: int
    tokens: tuple[int, ...]
    prefix: tuple[int, ...]


class BlockIndex:
    """CPU metadata only; no KV tensors are retained outside the paged pool."""

    def __init__(self):
        self.blocks = {}

    def invalidate(self, written):
        for block in written:
            self.blocks.pop(block, None)

    def matches(self, tokens, prefix, excluded, size):
        matches = []
        offset = 0
        while offset + size <= len(tokens):
            window = tuple(tokens[offset : offset + size])
            match = next(
                (
                    row
                    for block, row in self.blocks.items()
                    if block not in excluded
                    and row.tokens == window
                    and row.prefix != tuple(prefix) + tuple(tokens[:offset])
                ),
                None,
            )
            if match is None:
                offset += 1
            else:
                matches.append((offset, match))
                offset += size
        return matches


class PagedRepair:
    def __init__(self, runner, ratio, event_path=None):
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("Repair ratio must be finite in [0,1]")
        self.runner, self.ratio = runner, ratio
        self.index = BlockIndex()
        self.event_path = Path(event_path) if event_path else None
        self.steps = self.hit_steps = self.matched = self.skipped = 0
        self.validated = False

    def validate(self):
        r = self.runner
        m = r._forward_model
        if (
            not isinstance(m, Model)
            or m.args.rope_scaling is not None
            or r._drafter is not None
            or r._is_pooling
            or r._is_vlm
            or r.vllm_config.lora_config is not None
            or r.vllm_config.parallel_config.tensor_parallel_size != 1
            or r.vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Study worker supports plain single-device Qwen3 only")
        for layer in m.layers:
            a = layer.self_attn._inner
            if (
                a.rope.traditional
                or a.rope.scale != 1
                or a.rope.dims != m.args.head_dim
            ):
                raise ValueError("Unsupported RoPE")
        self.cache = m.layers[0].self_attn._mk_kv_cache
        if self.cache.turboquant or self.cache.group_index_for_layer(0) != 0:
            raise ValueError("Study worker requires one unquantized KV group")
        self.validated = True

    def start(self, original, batch, prefills, decodes, output):
        if not self.validated:
            self.validate()
        r, cache = self.runner, self.cache
        size = cache.block_size
        segments = []
        for req_id, state in decodes:
            start = r._paged_request_seq_lens[req_id]
            segments.append((req_id, [state.token_ids[-1]], start, state.block_ids))
        segments += [
            (p.req_id, p.token_ids, p.start_pos, p.block_ids) for p in prefills
        ]
        written = {
            blocks[0][pos // size]
            for _, tokens, start, blocks in segments
            for pos in range(start, start + len(tokens))
        }
        written.update(output.new_block_ids_to_zero or [])
        written.update(dest for _, dest in (output.kv_cache_block_copies or []))
        if any(p.sampling_params.prompt_logprobs is not None for p in prefills):
            raise ValueError("Study worker does not support prompt logprobs")
        plans = {}
        for p in prefills:
            state = r._request_states.get(p.req_id)
            full = (
                state.token_ids
                if state
                else batch.new_reqs_by_id[p.req_id].prompt_token_ids
            )
            matches = self.index.matches(
                p.token_ids, full[: p.start_pos], written, size
            )
            if matches:
                plans[p.req_id] = matches
        self.index.invalidate(written)
        self.steps += 1
        skipped = 0
        for req_id, tokens, _, _ in segments:
            active = set(range(len(tokens)))
            for offset, _ in plans.get(req_id, []):
                selected = set(select_rows(size, self.ratio))
                active.difference_update(
                    offset + i for i in range(size) if i not in selected
                )
            active.add(len(tokens) - 1)
            skipped += len(tokens) - len(active)
        target = r._target_forward
        adapter = r._model_adapter
        intermediate = adapter.intermediate_forward
        if skipped:

            def target_forward(
                input_ids,
                *,
                cache=None,
                collect_hidden_states=False,
                logits_indices=None,
            ):
                if collect_hidden_states:
                    raise ValueError(
                        "Hidden-state collection unsupported in study path"
                    )
                hidden = self.forward(segments, plans)
                if logits_indices is not None:
                    hidden = hidden[:, logits_indices, :]
                model = r._forward_model
                logits = (
                    model.model.embed_tokens.as_linear(hidden)
                    if model.args.tie_word_embeddings
                    else model.lm_head(hidden)
                )
                return TargetModelForwardOutput(logits=logits)

            def intermediate_forward(model, input_ids, *, cache=None):
                return IntermediateForwardOutput(
                    hidden_states=self.forward(segments, plans)
                )

            r._target_forward = target_forward
            adapter.intermediate_forward = intermediate_forward
        try:
            original(batch, prefills, decodes, output)
        finally:
            r._target_forward = target
            adapter.intermediate_forward = intermediate
        # A physical page's metadata is replaced only after its new content is written.
        for p in prefills:
            state = r._request_states.get(p.req_id)
            full = (
                state.token_ids
                if state
                else batch.new_reqs_by_id[p.req_id].prompt_token_ids
            )
            end = p.start_pos + len(p.token_ids)
            for start in range((p.start_pos // size) * size, end - size + 1, size):
                block = p.block_ids[0][start // size]
                self.index.blocks[block] = ResidentBlock(
                    block, tuple(full[start : start + size]), tuple(full[:start])
                )
        if plans:
            matched = sum(len(rows) * size for rows in plans.values())
            self.hit_steps += 1
            self.matched += matched
            self.skipped += skipped
            event = dict(
                step=self.steps,
                hit_steps=self.hit_steps,
                matched_tokens_total=self.matched,
                skipped_rows_total=self.skipped,
                matched_tokens=matched,
                skipped_rows=skipped,
                requests=list(plans),
                matches=[
                    dict(
                        request=req_id,
                        target_position=start + offset,
                        source_position=len(source.prefix),
                        source_block=source.block,
                        repaired_rows=len(select_rows(size, self.ratio)),
                    )
                    for req_id, _, start, _ in segments
                    for offset, source in plans.get(req_id, [])
                ],
                ratio=self.ratio,
            )
            logger.info("STUDY_NONPREFIX %s", json.dumps(event))
            if self.event_path:
                with self.event_path.open("a") as stream:
                    stream.write(json.dumps(event) + "\n")

    def gather(self, layer, slots, values=False):
        cache = self.cache
        array = cache.value_caches[layer] if values else cache.key_caches[layer]
        return gather_kv(
            array,
            mx.array(slots, dtype=mx.int32),
            cache.head_dim,
            cache.block_stride_for_layer(layer),
        ).transpose(1, 0, 2)[None]

    def forward(self, segments, plans):
        model, cache = self.runner._forward_model, self.cache
        size = cache.block_size
        outputs = []
        for req_id, tokens, start, blocks in segments:
            n = len(tokens)
            active = set(range(n))
            for offset, _ in plans.get(req_id, []):
                selected = set(select_rows(size, self.ratio))
                active.difference_update(
                    offset + i for i in range(size) if i not in selected
                )
            active.add(n - 1)
            active = sorted(active)
            positions = [start + i for i in active]
            indices = mx.array(active, dtype=mx.int32)
            hidden = model.model.embed_tokens(mx.array([[tokens[i] for i in active]]))
            prefix_slots = [
                blocks[0][i // size] * size + i % size for i in range(start)
            ]
            slots = mx.array(
                [
                    blocks[0][i // size] * size + i % size
                    for i in range(start, start + n)
                ],
                dtype=mx.int64,
            )
            mask = mx.array(positions)[:, None] >= mx.arange(start + n)[None, :]
            for layer_id, layer in enumerate(model.layers):
                a = layer.self_attn._inner
                norm = layer.input_layernorm(hidden)
                q = a.q_norm(
                    a.q_proj(norm).reshape(1, len(active), a.n_heads, -1)
                ).transpose(0, 2, 1, 3)
                k = a.k_norm(
                    a.k_proj(norm).reshape(1, len(active), a.n_kv_heads, -1)
                ).transpose(0, 2, 1, 3)
                v = (
                    a.v_proj(norm)
                    .reshape(1, len(active), a.n_kv_heads, -1)
                    .transpose(0, 2, 1, 3)
                )
                q, k = _rotate(q, positions, a.rope), _rotate(k, positions, a.rope)
                shape = (1, a.n_kv_heads, n, k.shape[-1])
                chunk_k, chunk_v = mx.zeros(shape, k.dtype), mx.zeros(shape, v.dtype)
                for offset, source in plans.get(req_id, []):
                    source_slots = list(
                        range(source.block * size, (source.block + 1) * size)
                    )
                    old_k = self.gather(layer_id, source_slots)
                    shift = start + offset - len(source.prefix)
                    if shift:
                        old_k = _rotate(old_k, [shift] * size, a.rope)
                    chunk_k[..., offset : offset + size, :] = old_k
                    chunk_v[..., offset : offset + size, :] = self.gather(
                        layer_id, source_slots, True
                    )
                chunk_k[..., indices, :] = k
                chunk_v[..., indices, :] = v
                keys, values = chunk_k, chunk_v
                if start:
                    keys = mx.concatenate(
                        [self.gather(layer_id, prefix_slots), keys], axis=2
                    )
                    values = mx.concatenate(
                        [self.gather(layer_id, prefix_slots, True), values], axis=2
                    )
                attended = mx.fast.scaled_dot_product_attention(
                    q, keys, values, scale=a.scale, mask=mask
                )
                hidden = hidden + a.o_proj(
                    attended.transpose(0, 2, 1, 3).reshape(1, len(active), -1)
                )
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
                new_k, new_v = get_ops().reshape_and_cache(
                    mx.contiguous(chunk_k[0].transpose(1, 0, 2).astype(cache.dtype)),
                    mx.contiguous(chunk_v[0].transpose(1, 0, 2).astype(cache.dtype)),
                    cache.key_caches[layer_id],
                    cache.value_caches[layer_id],
                    slots,
                )
                cache.replace_layer_cache(layer_id, new_k, new_v)
            normalized = model.model.norm(hidden)
            packed = mx.zeros((1, n, normalized.shape[-1]), normalized.dtype)
            packed[:, indices, :] = normalized
            outputs.append(packed)
        return mx.concatenate(outputs, axis=1)


class StudyNonPrefixWorker(MetalWorker):
    def init_device(self):
        super().init_device()
        runner = self.model_runner
        original = runner._start_paged_forward
        study = PagedRepair(
            runner,
            float(os.environ.get("STUDY_NONPREFIX_RATIO", "0.1")),
            os.environ.get("STUDY_NONPREFIX_EVENTS"),
        )
        logger.info("STUDY_NONPREFIX_ENABLED ratio=%s paged_pool_only=1", study.ratio)
        runner._study_nonprefix = study
        runner._start_paged_forward = lambda *args: study.start(original, *args)
