# TBX-Agent synthetic system suite v1.1

This directory is a versioned software-regression fixture. It contains no patient
record, no clinical image and no claim of clinical validation. Every case ID is stable;
editing the meaning or expected contract of a published case requires a new suite
version. Additive cases also update the manifest count and SHA-256.

v1.1.0 preserves all 22 v1.0.0 cases and adds two production-identity fixtures: a
same-tenant/cross-subject denial matrix spanning case, screening, review and report reads,
and an authorized case read that must redact internal image artifact references. Historical
v1.0.0 reports remain in the ledger but are not a valid paired release baseline for v1.1.0.

An adapter executes each `turns` sequence against a candidate deployment and writes one
normalized `SystemObservation` JSON object per line. Fixture operations such as
`respond_with_retrieval_fixture` and injected timeouts must run only in an isolated test
environment. Never enable them in a production API.

Every observation is bound to this suite ID/version/cases SHA-256 and to the candidate ID
and canonical candidate-config SHA-256. It also declares the adapter ID/version,
`synthetic=true`, and `clinical_validation=false`. The evaluator rejects a missing,
duplicate, foreign, mixed-adapter or differently-bound record instead of guessing its
provenance.

The built-in deterministic adapter requires a candidate contract generated for the exact
source tree under evaluation. That external contract pins the maintained runtime/config
artifact set, source-tree digest, policy IDs, knowledge hashes and sparse generation. The
adapter verifies those values before creating state and again after writing observations; mid-run drift is
retained as a failed ledger event and cannot issue a report. It refuses an existing run
directory, observation or report file, so a previous failed/regressed run cannot be
overwritten. Paired CLI comparisons accept only a baseline whose exact report SHA has one
matching receipt in the selected intact ledger.

The suite is intentionally not locked and `selection_use` is false. It is suitable for
continuous regression testing, not model selection, threshold selection, clinical
performance estimation or regulatory evidence.
