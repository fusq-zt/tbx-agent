# Engineering retrieval smoke v2

This fixture is the current deterministic BM25 smoke suite for knowledge snapshot
`tbx-guidelines-curated-2026-08-31-v4`. It preserves the v1 query set and relevance
labels while recording a new split identity, query hash, qrels hash, and corpus
generation after the reviewed active-screening chunk correction.

`smoke_v1` remains byte-frozen for historical schema and digest replay. It is not
silently pointed at the v4 corpus.
