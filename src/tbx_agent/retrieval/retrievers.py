from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict

from .contracts import (
    EmbeddingProvenance,
    MetadataFilter,
    RankedItem,
    RetrievalDocument,
    RetrievedChunk,
)
from .embeddings import EmbeddingAdapter
from .errors import ContractError, RetrievalError, StaleIndexError
from .fusion import weighted_reciprocal_rank_fusion
from .indexing import IndexManifest, build_index_manifest, validate_index_manifest
from .sparse import BM25Index
from .vectorstores import VectorStore


def _ordered_documents(
    documents: Iterable[RetrievalDocument],
) -> tuple[RetrievalDocument, ...]:
    ordered = tuple(sorted(documents, key=lambda item: item.chunk_id))
    if not ordered:
        raise ContractError("retriever corpus cannot be empty")
    if len({item.chunk_id for item in ordered}) != len(ordered):
        raise ContractError("retriever corpus contains duplicate chunk_id values")
    return ordered


def _unit_vector(vector: Sequence[float], dimensions: int) -> tuple[float, ...]:
    converted = tuple(float(value) for value in vector)
    if len(converted) != dimensions or any(not math.isfinite(value) for value in converted):
        raise ContractError("dense vector violates the embedding dimension contract")
    norm = math.sqrt(sum(value * value for value in converted))
    if not math.isfinite(norm) or norm == 0:
        raise ContractError("dense vector must have a finite non-zero norm")
    return tuple(value / norm for value in converted)


class DenseRetriever:
    """Metadata-gated dense retriever with an injectable embedding adapter.

    Tests can supply ``LocalCallableEmbeddingAdapter`` and in-memory vectors;
    deployments can supply any promoted ``VectorStore`` (including Qdrant
    local).  No model import or download occurs inside this class.
    """

    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        embedder: EmbeddingAdapter,
        vector_store: VectorStore | None = None,
        document_vectors: Mapping[str, Sequence[float]] | None = None,
        index_manifest: IndexManifest | None = None,
    ) -> None:
        if vector_store is not None and document_vectors is not None:
            raise ContractError("choose a vector_store or document_vectors, not both")
        self._ordered = _ordered_documents(documents)
        self.documents = {item.chunk_id: item for item in self._ordered}
        self.embedder = embedder
        self.vector_store = vector_store
        provenance = embedder.provenance
        if not provenance.normalized:
            raise ContractError("dense retrieval requires normalized embedding provenance")

        if vector_store is not None:
            promoted = vector_store.manifest
            mismatches = []
            expected = build_index_manifest(
                self._ordered,
                provenance,
                index_backend=promoted.store_type,
            )
            if promoted.corpus_sha256 != expected.corpus_sha256:
                mismatches.append("corpus_sha256")
            if promoted.embedding_fingerprint != provenance.fingerprint:
                mismatches.append("embedding_fingerprint")
            if promoted.dimensions != provenance.dimensions:
                mismatches.append("dimensions")
            if promoted.vector_count >= 0 and promoted.vector_count != len(self._ordered):
                mismatches.append("chunk_count")
            if mismatches:
                raise StaleIndexError("stale vector index: " + ", ".join(mismatches))
            self.index_manifest = IndexManifest(
                schema_version=1,
                index_backend=promoted.store_type,
                generation_id=promoted.generation_id,
                corpus_sha256=promoted.corpus_sha256,
                embedding_fingerprint=promoted.embedding_fingerprint,
                embedding_provenance=asdict(provenance),
                dimensions=promoted.dimensions,
                chunk_count=len(self._ordered),
            )
            self._vectors: dict[str, tuple[float, ...]] | None = None
            return

        if document_vectors is None:
            batch = embedder.embed(
                tuple(document.embedding_text for document in self._ordered),
                purpose="document",
            )
            if batch.provenance.fingerprint != provenance.fingerprint:
                raise StaleIndexError("document embedding provenance changed in-flight")
            document_vectors = {
                document.chunk_id: vector
                for document, vector in zip(self._ordered, batch.vectors, strict=True)
            }
        if set(document_vectors) != set(self.documents):
            raise ContractError("document_vectors must contain every chunk_id exactly once")
        self._vectors = {
            chunk_id: _unit_vector(vector, provenance.dimensions)
            for chunk_id, vector in document_vectors.items()
        }
        if index_manifest is None:
            index_manifest = build_index_manifest(
                self._ordered,
                provenance,
                index_backend="in_memory_exact",
            )
        validate_index_manifest(index_manifest, self._ordered, provenance)
        self.index_manifest = index_manifest

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self.embedder.provenance

    def ranked_items(
        self,
        query: str,
        *,
        filters: MetadataFilter | None = None,
        limit: int = 20,
    ) -> tuple[RankedItem, ...]:
        if not query.strip():
            return ()
        if limit < 1:
            raise ContractError("dense retrieval limit must be positive")
        validate_index_manifest(self.index_manifest, self._ordered, self.provenance)
        policy = filters or MetadataFilter()
        batch = self.embedder.embed((query,), purpose="query")
        if batch.provenance.fingerprint != self.provenance.fingerprint:
            raise StaleIndexError("query embedding provenance changed in-flight")
        query_vector = _unit_vector(batch.vectors[0], self.provenance.dimensions)
        if self.vector_store is not None:
            return self.vector_store.search(query_vector, filters=policy, limit=limit)
        assert self._vectors is not None
        scored = [
            (
                document.chunk_id,
                sum(
                    left * right
                    for left, right in zip(
                        query_vector,
                        self._vectors[document.chunk_id],
                        strict=True,
                    )
                ),
            )
            for document in self._ordered
            if policy.admits(document)
        ]
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            RankedItem(chunk_id=chunk_id, score=score, backend="dense", rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        )

    def retrieve(
        self,
        query: str,
        *,
        filters: MetadataFilter | None = None,
        top_k: int = 5,
    ) -> tuple[RetrievedChunk, ...]:
        items = self.ranked_items(query, filters=filters, limit=top_k)
        return tuple(
            RetrievedChunk(
                rank=item.rank,
                score=item.score,
                document=self.documents[item.chunk_id],
                backend_ranks={"dense": item.rank},
                backend_scores={"dense": item.score},
            )
            for item in items
        )


class HybridRetriever:
    """BM25 + dense retrieval fused with deterministic weighted RRF."""

    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        dense_retriever: DenseRetriever | None = None,
        candidate_limit: int = 40,
        sparse_weight: float = 1.0,
        dense_weight: float = 1.0,
        rrf_rank_constant: int = 60,
        sparse_minimum_score: float = 0.0,
        fallback_to_sparse: bool = True,
    ) -> None:
        self._ordered = _ordered_documents(documents)
        self.documents = {item.chunk_id: item for item in self._ordered}
        if dense_retriever is not None and set(dense_retriever.documents) != set(self.documents):
            raise ContractError("dense and sparse retrievers must use the same chunk set")
        if not 1 <= candidate_limit <= 1000:
            raise ContractError("candidate_limit must be between 1 and 1000")
        if sparse_weight <= 0 or dense_weight <= 0 or rrf_rank_constant < 1:
            raise ContractError("hybrid weights and RRF rank constant must be positive")
        if sparse_minimum_score < 0:
            raise ContractError("sparse_minimum_score must be non-negative")
        self.sparse = BM25Index(self._ordered)
        self.dense = dense_retriever
        self.candidate_limit = candidate_limit
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_rank_constant = rrf_rank_constant
        self.sparse_minimum_score = sparse_minimum_score
        self.fallback_to_sparse = fallback_to_sparse
        self.last_backend_status: dict[str, str] = {}
        self.last_fallback_reason: str | None = None

    def retrieve(
        self,
        query: str,
        *,
        filters: MetadataFilter | None = None,
        top_k: int = 5,
    ) -> tuple[RetrievedChunk, ...]:
        if not query.strip():
            return ()
        if not 1 <= top_k <= 100:
            raise ContractError("hybrid top_k must be between 1 and 100")
        policy = filters or MetadataFilter()
        sparse_items = self.sparse.search(
            query,
            filters=policy,
            limit=self.candidate_limit,
            minimum_score=self.sparse_minimum_score,
        )
        rankings: dict[str, tuple[RankedItem, ...]] = {"bm25": sparse_items}
        self.last_backend_status = {
            "bm25": "used",
            "dense": "disabled" if self.dense is None else "not_queried",
        }
        self.last_fallback_reason = None
        if self.dense is not None:
            try:
                rankings["dense"] = self.dense.ranked_items(
                    query,
                    filters=policy,
                    limit=self.candidate_limit,
                )
                self.last_backend_status["dense"] = "used"
            except RetrievalError as exc:
                self.last_backend_status["dense"] = "unavailable"
                self.last_fallback_reason = f"dense:{type(exc).__name__}"
                if not self.fallback_to_sparse:
                    raise
        fused = weighted_reciprocal_rank_fusion(
            rankings,
            weights={"bm25": self.sparse_weight, "dense": self.dense_weight},
            rank_constant=self.rrf_rank_constant,
        )
        return tuple(
            RetrievedChunk(
                rank=rank,
                score=item.score,
                document=self.documents[item.chunk_id],
                backend_ranks=item.backend_ranks,
                backend_scores=item.backend_scores,
            )
            for rank, item in enumerate(fused[:top_k], start=1)
        )
