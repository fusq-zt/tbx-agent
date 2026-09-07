from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import (
    BackendStatus,
    RetrievalDocument,
    RetrievalHit,
    RetrievalQuery,
    RetrievalReceipt,
    RetrievalResult,
    canonical_json,
    sha256_text,
)
from .embeddings import EmbeddingAdapter
from .errors import (
    BackendUnavailableError,
    ContractError,
    GenerationMismatchError,
    RetrievalError,
)
from .fusion import FusedCandidate, deduplicate_candidates, weighted_reciprocal_rank_fusion
from .indexing import corpus_sha256
from .rerank import Reranker
from .sparse import BM25Index
from .vectorstores import VectorStore


@dataclass(frozen=True, slots=True)
class RetrievalEngineConfig:
    retrieval_version: str = "tbx-retrieval-v1"
    retrieval_mode: Literal["sparse", "dense", "hybrid"] = "hybrid"
    candidate_limit: int = 40
    sparse_minimum_score: float = 0.0
    rrf_rank_constant: int = 60
    sparse_weight: float = 1.0
    dense_weight: float = 1.0
    fallback_to_sparse: bool = True
    near_duplicate_threshold: float = 0.92
    near_duplicates_across_sources: bool = False
    rerank_top_n: int = 20
    reranker_required: bool = False
    require_verified_model_hashes: bool = True

    def __post_init__(self) -> None:
        if not self.retrieval_version.strip():
            raise ContractError("retrieval_version must be non-empty")
        if self.retrieval_mode not in {"sparse", "dense", "hybrid"}:
            raise ContractError("retrieval_mode must be sparse, dense, or hybrid")
        if not 1 <= self.candidate_limit <= 1000:
            raise ContractError("candidate_limit must be between 1 and 1000")
        if self.sparse_minimum_score < 0:
            raise ContractError("sparse_minimum_score must be non-negative")
        if self.rrf_rank_constant < 1:
            raise ContractError("rrf_rank_constant must be positive")
        if self.sparse_weight <= 0 or self.dense_weight <= 0:
            raise ContractError("retrieval backend weights must be positive")
        if not 0 <= self.near_duplicate_threshold <= 1:
            raise ContractError("near_duplicate_threshold must be in [0, 1]")
        if not 0 <= self.rerank_top_n <= self.candidate_limit:
            raise ContractError("rerank_top_n must be between 0 and candidate_limit")

    @property
    def sha256(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


class RetrievalEngine:
    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        config: RetrievalEngineConfig | None = None,
        embedder: EmbeddingAdapter | None = None,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        ordered = tuple(sorted(documents, key=lambda item: item.chunk_id))
        if not ordered:
            raise ContractError("retrieval corpus cannot be empty")
        if len({document.chunk_id for document in ordered}) != len(ordered):
            raise ContractError("retrieval corpus contains duplicate chunk_id values")
        if (embedder is None) != (vector_store is None):
            raise ContractError("dense retrieval requires both an embedder and vector store")
        self.documents = {document.chunk_id: document for document in ordered}
        self.config = config or RetrievalEngineConfig()
        self.sparse = BM25Index(ordered)
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = reranker
        self._closed = False
        if self.config.retrieval_mode == "dense" and self.embedder is None:
            raise ContractError("dense retrieval mode requires an embedder and vector store")
        if (
            self.config.require_verified_model_hashes
            and self.embedder is not None
            and self.embedder.provenance.model_sha256 is None
        ):
            raise ContractError("dense retrieval requires a verified embedding model_sha256")
        if (
            self.config.require_verified_model_hashes
            and self.reranker is not None
            and self.reranker.fingerprint.endswith("@UNVERIFIED")
        ):
            raise ContractError("reranking requires a verified model_sha256")
        self.corpus_sha256 = corpus_sha256(ordered)
        self.sparse_generation_id = f"sparse-{self.corpus_sha256[:24]}"
        self.sparse_manifest_sha256 = sha256_text(
            canonical_json(
                {
                    "schema_version": 1,
                    "backend": "bm25",
                    "generation_id": self.sparse_generation_id,
                    "corpus_sha256": self.corpus_sha256,
                }
            )
        )

    def close(self) -> None:
        """Release an owned vector backend without initializing optional models."""

        if self._closed:
            return
        self._closed = True
        close = getattr(self.vector_store, "close", None)
        if callable(close):
            close()

    def _validate_dense_items(self, items: tuple, query: RetrievalQuery) -> None:
        seen: set[str] = set()
        for expected_rank, item in enumerate(items, start=1):
            document = self.documents.get(item.chunk_id)
            if (
                item.backend != "dense"
                or item.rank != expected_rank
                or item.chunk_id in seen
                or not math.isfinite(item.score)
                or document is None
                or not query.filters.admits(document)
            ):
                raise BackendUnavailableError("dense backend violated the retrieval contract")
            seen.add(item.chunk_id)

    @staticmethod
    def _rerank_candidates(
        candidates: tuple[FusedCandidate, ...],
        scores: dict[str, float],
        top_n: int,
    ) -> tuple[FusedCandidate, ...]:
        head = candidates[:top_n]
        tail = candidates[top_n:]
        reranked = sorted(head, key=lambda item: (-scores[item.chunk_id], item.chunk_id))
        return tuple([*reranked, *tail])

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if self._closed:
            raise BackendUnavailableError("retrieval engine is closed")
        mode = self.config.retrieval_mode
        use_sparse = mode in {"sparse", "hybrid"}
        use_dense = mode in {"dense", "hybrid"} and self.embedder is not None
        backend_status: dict[str, BackendStatus] = {
            "bm25": "used" if use_sparse else "not_queried",
            "dense": "not_queried" if use_dense else "disabled",
            "reranker": "disabled" if self.reranker is None else "not_queried",
        }
        fallback_reasons: list[str] = []
        sparse_items = ()
        rankings = {}
        if use_sparse:
            sparse_items = self.sparse.search(
                query.text,
                filters=query.filters,
                limit=self.config.candidate_limit,
                minimum_score=self.config.sparse_minimum_score,
            )
            rankings["bm25"] = sparse_items
        generation_id = self.sparse_generation_id
        manifest_sha256 = self.sparse_manifest_sha256
        embedding_fingerprint: str | None = None
        dense_items = ()
        if use_dense and self.embedder is not None and self.vector_store is not None:
            try:
                manifest = self.vector_store.manifest
                generation_id = manifest.generation_id
                manifest_sha256 = manifest.manifest_sha256
                embedding_fingerprint = self.embedder.provenance.fingerprint
                if manifest.corpus_sha256 != self.corpus_sha256:
                    raise GenerationMismatchError("dense corpus does not match sparse corpus")
                if manifest.embedding_fingerprint != embedding_fingerprint:
                    backend_status["dense"] = "mismatch"
                    raise GenerationMismatchError("query embedder does not match vector generation")
                batch = self.embedder.embed((query.text,), purpose="query")
                if batch.provenance.fingerprint != embedding_fingerprint:
                    raise GenerationMismatchError("embedding response provenance changed in-flight")
                dense_items = self.vector_store.search(
                    batch.vectors[0],
                    filters=query.filters,
                    limit=self.config.candidate_limit,
                )
                self._validate_dense_items(dense_items, query)
                rankings["dense"] = dense_items
                backend_status["dense"] = "used"
            except RetrievalError as exc:
                if isinstance(exc, GenerationMismatchError):
                    backend_status["dense"] = "mismatch"
                elif backend_status["dense"] != "mismatch":
                    backend_status["dense"] = "unavailable"
                fallback_reasons.append(f"dense:{type(exc).__name__}")
                if not self.config.fallback_to_sparse:
                    raise
                if not use_sparse:
                    sparse_items = self.sparse.search(
                        query.text,
                        filters=query.filters,
                        limit=self.config.candidate_limit,
                        minimum_score=self.config.sparse_minimum_score,
                    )
                    rankings["bm25"] = sparse_items
                    backend_status["bm25"] = "used"
                generation_id = self.sparse_generation_id
                manifest_sha256 = self.sparse_manifest_sha256
            except Exception as exc:  # noqa: BLE001 - optional backend isolation boundary
                backend_status["dense"] = "unavailable"
                fallback_reasons.append(f"dense:{type(exc).__name__}")
                if not self.config.fallback_to_sparse:
                    raise
                if not use_sparse:
                    sparse_items = self.sparse.search(
                        query.text,
                        filters=query.filters,
                        limit=self.config.candidate_limit,
                        minimum_score=self.config.sparse_minimum_score,
                    )
                    rankings["bm25"] = sparse_items
                    backend_status["bm25"] = "used"
                generation_id = self.sparse_generation_id
                manifest_sha256 = self.sparse_manifest_sha256
        weights = {"bm25": self.config.sparse_weight, "dense": self.config.dense_weight}
        fused = weighted_reciprocal_rank_fusion(
            rankings,
            weights=weights,
            rank_constant=self.config.rrf_rank_constant,
        )
        deduplicated = deduplicate_candidates(
            fused,
            self.documents,
            near_duplicate_threshold=self.config.near_duplicate_threshold,
            near_duplicates_across_sources=self.config.near_duplicates_across_sources,
        )
        candidates = deduplicated.candidates
        rerank_scores: dict[str, float] = {}
        if self.reranker is not None and candidates and self.config.rerank_top_n:
            top_n = min(self.config.rerank_top_n, len(candidates))
            rerank_documents = [self.documents[item.chunk_id] for item in candidates[:top_n]]
            try:
                result = self.reranker.score(query.text, rerank_documents)
                rerank_scores = {item.chunk_id: item.score for item in result}
                if set(rerank_scores) != {item.chunk_id for item in candidates[:top_n]}:
                    raise BackendUnavailableError("reranker omitted or invented a chunk_id")
                candidates = self._rerank_candidates(candidates, rerank_scores, top_n)
                backend_status["reranker"] = "used"
            except BackendUnavailableError as exc:
                backend_status["reranker"] = "unavailable"
                fallback_reasons.append(f"reranker:{type(exc).__name__}")
                if self.config.reranker_required:
                    raise
        hits = tuple(
            RetrievalHit(
                rank=rank,
                score=item.score,
                document=self.documents[item.chunk_id],
                backend_ranks=item.backend_ranks,
                backend_scores=item.backend_scores,
                rerank_score=rerank_scores.get(item.chunk_id),
            )
            for rank, item in enumerate(candidates[: query.top_k], start=1)
        )
        receipt_body = {
            "schema_version": 2,
            "retrieval_version": self.config.retrieval_version,
            "config_sha256": self.config.sha256,
            "query_id": query.query_id,
            "query_sha256": query.query_sha256,
            "top_k": query.top_k,
            "filters": query.filters.to_dict(),
            "request_sha256": query.request_sha256,
            "returned_chunks": tuple(
                {
                    "chunk_id": hit.document.chunk_id,
                    "source_id": hit.document.source_id,
                    "content_sha256": hit.document.content_sha256,
                    "source_sha256": hit.document.source_sha256,
                }
                for hit in hits
            ),
            "corpus_generation_id": generation_id,
            "corpus_manifest_sha256": manifest_sha256,
            "backend_status": backend_status,
            "embedding_fingerprint": embedding_fingerprint,
            "fallback_reason": ";".join(fallback_reasons) or None,
            "suppressed_duplicate_ids": deduplicated.suppressed_chunk_ids,
        }
        receipt = RetrievalReceipt(
            retrieval_id=sha256_text(canonical_json(receipt_body))[:32],
            **receipt_body,
        )
        return RetrievalResult(hits=hits, receipt=receipt)
