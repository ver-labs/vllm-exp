Repository preparation — 24 October 2026
=========================================

Preserved
---------
All scientific result summaries, per-request records, source patches, tests and
original evidence inventories are retained. No result was removed because it
was negative or mixed. The nested vllm-src checkout and active environments were
not changed. The complete pre-cleanup Git history was saved as a verified bundle.

Removed from current Git tracking, retained locally
--------------------------------------------------
144 compressed generated diagnostics (53,697,243 bytes) and two generated
machine receipts: configs/editable-setup.json and configs/metal-baseline.json.
The originals still exist at their old paths. A second, checksum-verified copy
of the diagnostics lives in the local archive listed in evidence-storage.json.
The machine receipts and history bundle are stored alongside that archive.
The ignored .artifacts directory is not part of a fresh clone.

Existing evidence inventories still reference these diagnostics. Use the
sidecar evidence-storage.json to distinguish locally retained files from files
available in a fresh clone. This is a storage change, not evidence deletion.
A public archival deposit of the complete diagnostics remains future work.

Publication boundary
--------------------
GitHub already listed ver-labs/vllm-exp as public when this cleanup began.
The published branches contained older snapshots. Their cleanup is based only
on those published tips; it does not merge the newer local research commits.
The current maintenance/public-prep-cleanup branch is prepared locally for a
later reviewed release. Repository visibility is not changed by this cleanup.

Security check and limits
-------------------------
credential-pattern-scan.json records a scan of all locally reachable historical
text and gzip blobs for common token, private-key and credential-URL patterns.
No matches were found. This is not proof that all sensitive information is
absent; the binary PDF was not text-scanned. Historical evidence may contain
machine paths and runtime metadata. Ordinary file removal preserves older Git
commits. No history rewrite, branch deletion or force push is part of cleanup.

Source preservation
-------------------
experiments/historical-study.patch captures the historical custom vLLM source
against official v0.31.0. The corrected cache-only research candidate is frozen
separately in research/01-baseline-freeze; neither patch replaces the other.
