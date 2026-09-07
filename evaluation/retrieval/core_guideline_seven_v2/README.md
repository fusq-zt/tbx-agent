# Core guideline seven v2

This is the immutable v7-corpus successor to `core_guideline_seven_v1`. It
preserves seed `20260901`, split ID `fixed-reviewed-corpus-v1`, and all seven
query IDs, texts, and answerability labels. The mask query's applicability scope
is narrowed from the parent suite's `infection_control OR respiratory_protection`
to `respiratory_protection`: the v7 corpus authorizes unrelated shared-object
transmission evidence under generic infection control, which must not satisfy a
respiratory-protection question.

## Identity record

- Suite: `tbx-core-guideline-seven` / `2.0.0`
- Snapshot: `tbx-guidelines-curated-2026-09-01-v7`
- Corpus generation: `sparse-1b305b3eb573708ed7989a3e`
- Queries SHA-256: `cec79c75462244f46922ce530f13b3e0fa4519f9b2694084e3413f036fdaace3`
- Qrels SHA-256: `ce01f2d1b327dd9ee99ab0d87dc64f82f2efa435c7f6a9297bb228a48f48c778`
- Config SHA-256: `7fedd2c4e775f945651aa6b61576cce19d8a0535c0b03741454820a6368fff70`
- Parent config SHA-256: `39927807b6f8a1dea71c01e8228a3130edd83c3df2c95181598327d67982aebf`

This fixture measures scoped engineering retrieval behavior and must not be
used for clinical conclusions or retriever/model selection.

## Recorded BM25 verification

Run date: 2026-09-01. Status `completed`; suite validation `verified`; no backend
failures. The no-answer abstention accuracy is 1.000000. Source-tree SHA-256 was
`55ec5fd09e85df7a0f7d0b9d1c055add3e7412dfad26b671e0804b591ceb00da`.
Wall time was 0.466434 s, peak RSS was 49,561,600 bytes, and peak VRAM was
unavailable because torch was not loaded.

| K | Recall | MRR | nDCG | Citation hit | Source recall | Hard-negative rejection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.333333 | 0.666667 | 0.380952 | 1.000000 | 1.000000 | 1.000000 |
| 3 | 0.638889 | 0.722222 | 0.546490 | 1.000000 | 1.000000 | 0.833333 |
| 5 | 0.805556 | 0.755556 | 0.594472 | 1.000000 | 1.000000 | 0.833333 |
| 10 | 1.000000 | 0.755556 | 0.688646 | 1.000000 | 1.000000 | 0.833333 |

The report retains one hard-negative-at-maximum-K observation for the standard
regimen duration query.
