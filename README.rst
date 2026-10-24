vLLM scheduler study — historical snapshot
==========================================

Exploratory request-scheduling and KV-cache work using vLLM with the Apple
Silicon Metal plugin. This branch is a historical development snapshot, not a
complete production package or a validated NVIDIA implementation.

Layout
------
* src/: server launchers, environment checks and measurement helpers.
* workloads/: request drivers and comparison tools.
* configs/: example settings and historical Mac package constraints.

Reproduction limits
-------------------
The local modified vLLM checkout, environments, models and raw results are not
bundled. Some tools require that custom checkout; this snapshot alone does not
provide full reproduction. Consult individual manifests where present. The
historical Mac dependency constraints are not a CUDA setup specification.

Generated configs/editable-setup.json and configs/metal-baseline.json are local
machine receipts and are intentionally ignored. The verification and snapshot
helpers can generate local receipts. Use paths appropriate to your machine.
Scientific summaries and existing evidence have not been removed in cleanup.

Newer unpublished research is maintained separately; this cleanup does not
publish it or make a new speedup, quality or CUDA-correctness claim.
