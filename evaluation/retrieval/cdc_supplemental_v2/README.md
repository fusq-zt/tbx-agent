# CDC supplemental retrieval fixture v2

This is the immutable v7-corpus successor to `cdc_supplemental_v1`. It preserves
seed `20260901`, split ID `fixed-cdc-supplemental-v1`, and all 19 query IDs,
texts, scopes, and relevance labels. Only the suite version and reviewed corpus
binding changed.

## Identity record

- Suite: `tbx-cdc-supplemental` / `2.0.0`
- Snapshot: `tbx-guidelines-curated-2026-09-01-v7`
- Corpus generation: `sparse-1b305b3eb573708ed7989a3e`
- Queries SHA-256: `336ca7a9eb93f0231df5be3b307a3154de3fa77a505aeb703e8209d1052d5d56`
- Qrels SHA-256: `8d9267e643a0ab3cf1f220afc6c6182734bc6fec363f1d70fb626935a89a0bdf`
- Config SHA-256: `b00944ed0f957c5761e78a88e8dade8e844a811ad25e7ba7443d6cd9f22155eb`
- Parent config SHA-256: `bc3825c1b7a14539efc023069f57464cd92c48ce19dc04b9dc6cbeeae6b8dacb`

The US CDC evidence remains supplemental to China and WHO guidance. This
fixture is not a clinical validation set or selection benchmark.

## Recorded BM25 verification

Run date: 2026-09-01. Status `completed`; suite validation `verified`; no backend
failures. Source-tree SHA-256 was
`55ec5fd09e85df7a0f7d0b9d1c055add3e7412dfad26b671e0804b591ceb00da`.
Wall time was 0.134782 s, peak RSS was 49,680,384 bytes, and peak VRAM was
unavailable because torch was not loaded.

| K | Recall | MRR | nDCG | Citation hit | Source recall | Hard-negative rejection |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.736842 | 0.789474 | 0.789474 | 0.947368 | 0.921053 | 0.913043 |
| 3 | 0.894737 | 0.859649 | 0.859531 | 1.000000 | 1.000000 | 0.652174 |
| 5 | 1.000000 | 0.872807 | 0.902965 | 1.000000 | 1.000000 | 0.565217 |
| 10 | 1.000000 | 0.872807 | 0.902965 | 1.000000 | 1.000000 | 0.521739 |

The report retains nine hard-negative-at-maximum-K observations.
