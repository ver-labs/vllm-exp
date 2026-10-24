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
The research baseline has source and CPU metadata checks. Actual KV tensor
copies, model quality, NVIDIA execution and current-upstream compatibility still
need validation. Historical pilot results and corrected candidate results must
be reported separately. No new inference benchmark was run during cleanup.
