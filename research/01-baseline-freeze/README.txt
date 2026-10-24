Research step 1: pinned baseline and isolated cache candidate
============================================================
Prepared 24 October 2026. Source preparation and CPU metadata validation only.

Outcome
-------
A clean stock/candidate pair is frozen at the official v0.31.0 release base.
This is the historical-release bridge for initial CUDA replication, not an
assertion that this release is the newest or strongest contemporary baseline.
A separate clean upstream snapshot dated 24 October 2026 is pinned for the
capability audit. It has substantial core changes; the bridge patch does not
apply there. A contemporary comparison requires a separately tested port.

Files
-----
baseline-lock.json    Official base, candidate and upstream-reference pins;
                      patch hash, changed paths, constraints, native-file hashes.
capabilities.csv      Native versus candidate capability table, with evidence level.
source-evidence.json  Immutable source/function anchors and hashes.
candidate.patch      Standalone 10-file cache-only patch against v0.31.0.
arms.json            Stock, aging, partial and combined definitions; not run-ready.
validation.json      CPU probe results and receipt hashes.
receipts/            Exact commands, outputs and expected negative controls.
prepare.py           Recreate isolated source trees; refuses existing destinations.
probe.py, verify.py   CPU metadata and source-isolation checks.
tests/               Copied/adapted metadata tests plus new reset-contract tests.

Frozen copies on this machine
-----------------------------
Experiment root: /Users/nandanadileep/projects/vllm/vllm/vllm-exp
Stock:     .artifacts/research-baselines/stock-v0.31.0
Candidate: .artifacts/research-baselines/candidate-v0.31.0
Current:   .artifacts/research-baselines/upstream-20261024
Historical vllm-src and its editable installation remain untouched.
These are Git worktrees; recreate them on another machine rather than copying
worktree .git links. No model server or inference benchmark was started.

What we inherit, and what the study adds
---------------------------------------
Native: causal-prefix hashes, eligible partial-tail lookup, private-tail CoW,
source/destination retention, alias invalidation and GPU copy paths.
Study: opt-in full-attention fine-grained path, publication of already-computed
interior prompt boundaries, alias preservation, LFU/allocation-clock aging and
incremental metadata/ordering. The copy kernels are not a study contribution.
The native final prompt-tail entry is not arbitrary interior publication.
Native behavior differs by model/cache-group configuration. Do not generalize
this one full-attention-group probe to every vLLM architecture/backend.

The candidate excludes request reordering, queue-aware retention, skip/continue,
adaptive prefill, trace/profile hooks and approximate middle-block reuse.
It retains the previous minimum-hit setting (zero) and incremental cache mode.

Deliberate changes from historical experiment 14
-----------------------------------------------
1. When partial mode is off, fine-match configuration and coordinator selection
   follow the release's native behavior. The pilot had configuration changes
   outside the opt-in flag. The flags-off candidate now matches clean stock in
   the executable metadata/allocator probe; this is not end-to-end equivalence.
2. Successful reset clears study frequency counts, allocation count, aging epoch
   and request publication state/visit counter, then rebuilds ordering against
   the native free queue. Failed reset leaves live refs and policy state intact.
   Native free-queue order is retained, so reset is not a new allocator instance.
3. This is a NEW candidate. It does not retroactively change pilot code/results.

Validation and limits
---------------------
Five real cache-metadata probes cover stock, candidate off, aging, partial and
combined. Clean stock and candidate off have identical normalized probe results.
For 16-token physical pages and a source [1..32], a request matching 31 tokens
reuses 16 in stock and 31 with interior publication. Different earlier context
and tenant salt miss. Private-tail endpoints remain protected until release.
This checks metadata/refcounts; no actual KV tensor copy or model ran.

Fourteen tests pass in ordinary mode and with aging enabled at a four-allocation
half-life: six partial-prefix, one threshold, four aging and three new reset
contracts. Historical reset code produces two expected failures under the new
reset contracts. Receipts preserve the initial test failures caused by old
ungated-configuration assumptions; copied tests were adapted to explicit opt-in.
Historical test files are unchanged. The previously reported seven partial-
related tests were six in test_partial_prefix plus one threshold test.

The CPU checks use the existing Mac virtualenv with plugins disabled and verified
ignored links to the same release's native components. Those files are not in
candidate.patch and must never be treated as NVIDIA artifacts. No package install
or global editable-path switch was performed. Current-upstream is source-audited
only; it was not imported with incompatible historical native binaries.

Recreate source copies
----------------------
Use an existing Git clone containing the two exact commits in baseline-lock.json.
Missing objects can be fetched from the official public repository by immutable
SHA. Use your project virtualenv Python, never system Python or bare pip.

  .venv/bin/python research/01-baseline-freeze/prepare.py \
    --repository /path/to/vllm-git \
    --destination /path/to/new-isolated-copies

For local Mac CPU metadata tests only, the optional --cpu-native-from may point
to the preserved vllm-src. Its required native component hashes are checked.
On NVIDIA, build each source tree against the actual approved CUDA environment;
do not use that Mac option. CUDA build instructions remain a separate step
until the compute-node runtime and dependencies are verified.

Run local metadata validation
-----------------------------
  .venv/bin/python research/01-baseline-freeze/verify.py \
    --stock .artifacts/research-baselines/stock-v0.31.0 \
    --candidate .artifacts/research-baselines/candidate-v0.31.0

Next milestone
--------------
Verify A100 environment and build compatibility; enforce declared supported
configurations; test actual copied KV/logits and cancellation/eviction/reset/
forced-preemption paths. Pin the model/tokenizer/backend and memory budget.
The declared scope is one full-attention group, TP=PP=DCP=PCP=1, synchronous
scheduling and no speculative decoding. These limits are not all startup-guarded
by the legacy implementation; numerical/distributed safety is not established.
A current-upstream port/baseline and supported native page-granularity controls
are required before broad comparative claims.
No latency, throughput, quality-parity or CUDA-correctness claim follows from
this preparation or its green CPU tests.
