# BM25 engineering smoke v5

This is the immutable v7-corpus successor to `smoke_v4`. It preserves seed
`20260901`, split ID `fixed-engineering-smoke-v4`, and all 31 query IDs, texts,
and labels. Only the suite version and reviewed corpus binding changed.
Historical `smoke_v4` files and results remain valid only for the v6 corpus.

## Identity record

- Suite: `tbx-rag-engineering-smoke` / `5.0.0`
- Snapshot: `tbx-guidelines-curated-2026-09-01-v7`
- Corpus generation: `sparse-1b305b3eb573708ed7989a3e`
- Queries SHA-256: `ba5437889740e96a8011ec0af1cbf3a8e2b765b2d74548bdb38ca6889423c3c9`
- Qrels SHA-256: `7450e52b1c69a5c7d26227b74fe0c51cf55dee248d633e08b73cada48c108fb1`
- Config SHA-256: `79d79c2b28353311351551f0669d287bffb31ef4eed930aec250e87fe071a235`
- Parent config SHA-256: `60a51de16b48f1f6a897fdaa104287bf73fe68c929b0f01103835134961b19f0`

This fixture is an engineering smoke test, not a clinical validation or a
retriever-selection benchmark.

## Recorded BM25 verification

Run date: 2026-09-01. Status `completed`; suite validation `verified`; no backend
failures. Source-tree SHA-256 was
`55ec5fd09e85df7a0f7d0b9d1c055add3e7412dfad26b671e0804b591ceb00da`.
Wall time was 0.416832 s, peak RSS was 49,795,072 bytes, and peak VRAM was
unavailable because torch was not loaded.

| K | Recall | MRR | nDCG | Citation hit | Source recall | Hard-negative rejection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.801075 | 0.935484 | 0.917051 | 0.967742 | 0.897849 | 1.000000 |
| 3 | 0.903226 | 0.951613 | 0.911191 | 1.000000 | 0.973118 | 0.750000 |
| 5 | 0.962366 | 0.959677 | 0.933901 | 1.000000 | 0.973118 | 0.687500 |
| 10 | 0.962366 | 0.959677 | 0.933901 | 1.000000 | 0.973118 | 0.593750 |

The report retains 16 per-query error rows at maximum K (three incomplete-recall
and 13 hard-negative observations); they are not silently removed or promoted
to clinical failures.
