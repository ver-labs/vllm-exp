"""Host-only scaling checks, separate from GPU end-to-end measurements."""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id


def replay(mode, size):
    with patch.dict(os.environ, STUDY_EVICTION_POLICY="lfu", STUDY_CACHE_IMPL=mode):
        pool = BlockPool(size + 1, True, 4)
    blocks = pool.get_new_blocks(size)
    for block in blocks:
        key = make_block_hash_with_group_id(block.block_id.to_bytes(32, "big"), 0)
        pool._insert_block_hash(key, block, num_tokens=4)
    pool.free_blocks(blocks)
    decisions = []
    selection_ns = 0
    started = None
    for index in range(450):
        if index == 50:
            started = time.perf_counter_ns()
        before = time.perf_counter_ns()
        block = pool.get_new_blocks(1)[0]
        if index >= 50:
            selection_ns += time.perf_counter_ns() - before
            decisions.append(block.block_id)
        key = make_block_hash_with_group_id((index + size + 1).to_bytes(32, "big"), 0)
        pool._insert_block_hash(key, block, num_tokens=4)
        touches = index % 4
        pool.touch([block] * touches)
        pool.free_blocks([block] * (touches + 1))
    return dict(
        total_ms=(time.perf_counter_ns() - started) / 1e6,
        allocation_ms=selection_ns / 1e6,
        decisions=decisions,
        free_blocks=size,
        measured_allocations=400,
    )


def main():
    runs = []
    for size in (32, 4096):
        for round_id in range(1, 4):
            order = (
                ("reference", "incremental")
                if round_id % 2
                else ("incremental", "reference")
            )
            pair = {}
            for mode in order:
                result = replay(mode, size)
                pair[mode] = result.pop("decisions")
                result.update(implementation=mode, round=round_id)
                runs.append(result)
            assert pair["reference"] == pair["incremental"]
    path = Path("experiments/10-cache-overhead/evidence/host-scaling.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                kind="host-only synthetic allocation/cache/touch/free replay",
                identical_victims=True,
                runs=runs,
            ),
            indent=2,
        )
        + "\n"
    )
    print(path.read_text())


if __name__ == "__main__":
    main()
