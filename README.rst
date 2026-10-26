vLLM cache and scheduler experiments
====================================

Exploratory Apple Silicon / Metal experiments with Qwen3-0.6B and Qwen3-1.7B.
This repository preserves positive, mixed and negative results. The Mac pilot
measurements do not establish CUDA correctness or a general serving speedup.

Start here
----------
* experiments/INDEX.txt: pilot experiment index.
* experiments/: plans, changes, manifests, summaries and retained evidence for
  14 pilot experiments. Consult each experiment's results and limitations.
* output/pdf/vllm-experiment-comparison.pdf: comparison report for experiments
  1 through 13; experiment 14 is documented separately under experiments/.
* research/01-baseline-freeze/README.txt: the first research milestone, with
  pinned stock/candidate source, CPU validation and outstanding CUDA checks.
* src/ and workloads/: launchers, measurement tools and workload drivers.
* configs/: historical example settings and Mac dependency constraints.

Source and reproduction
-----------------------
The original custom vLLM checkout, environments, models and raw run directories
are intentionally ignored. experiments/historical-study.patch preserves the
historical checkout changes against official vLLM v0.31.0 commit
 db9527a46873454610df6dbedf79a36d6bf1a7f6.
Apply that patch only to a clean checkout of that exact release. This is the
historical pilot implementation, including exploratory features, not the new
isolated research candidate.

research/01-baseline-freeze/candidate.patch is a separate cache-only candidate.
Its README describes preparation, validation, inherited native capabilities,
fixes relative to the pilot, supported scope and remaining work. The source
preparation helper does not install dependencies or validate NVIDIA execution.
Historical Mac constraints must not be used as a CUDA environment specification.

Evidence storage
----------------
Scientific summaries, per-request records, tests and source patches remain
versioned. During repository preparation, 144 compressed diagnostic logs and
traces were removed from tracking, with their original local files retained and
an additional hash-verified local archive. They are not included in a fresh
clone. research/public-prep/evidence-storage.json records every affected path,
size and checksum. Original evidence inventories remain unchanged. Those
inventories include local-only files; this checkout alone is not the complete
raw diagnostic archive. research/public-prep/README.txt explains the boundary.

Generated machine setup receipts, virtual environments, model weights, native
binaries, credentials and distribution archives are ignored. Generate local
setup receipts with the setup helpers rather than committing machine paths.

Research status
---------------
The frozen baseline has source and metadata checks. A100 environment and source
installation checks are in research/02-a100-preflight/RESULTS.txt. Model-level
checks for both Qwen sizes pass in the declared synchronous batch-invariant
configuration; see research/03-a100-correctness/RESULTS.txt for logits, KV copies,
eviction, cancellation, reset and preemption checks. The default non-invariant
BF16 stock numerical gate failed and is preserved separately. Task quality,
broader deployment configurations, complete native source compilation and the
current-upstream port remain unvalidated. A controlled A100 latency/throughput benchmark is now saved in
research/04-a100-benchmarks/INTERPRETATION.txt and RESULTS.txt: 50 fresh-engine
runs across both models, with five paired repeats and matching output token IDs.
It does not establish a throughput speedup; partial reuse worsens first-token
latency on the shared-prefix workloads despite extra cache hits. The benchmark
uses synthetic short prompts, concurrency three, 16 KV pages and eager
batch-invariant mode; it is not a peak-throughput or production claim.
Historical pilot results and corrected candidate results remain separate.

Diagnostic profiling
--------------------
research/05-a100-profiling/INTERPRETATION.txt records stock, features-off and
partial-reuse CPU/CUDA diagnostics on both models. They identify extra copy
dispatch, publication and one-query attention-path tensor copies; they do not
establish a new speedup. Raw traces are preserved in the cluster archive.
