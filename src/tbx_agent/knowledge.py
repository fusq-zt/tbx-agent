from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

from .retrieval import (
    BackendUnavailableError,
    BgeM3LocalEmbeddingAdapter,
    ContractError,
    EmbeddingProvenance,
    GenerationMismatchError,
    OpenAICompatibleEmbeddingAdapter,
    OptionalDependencyError,
    QdrantLocalConfig,
    QdrantLocalVectorStore,
    RetrievalDocument,
    RetrievalEngine,
    RetrievalFilter,
    RetrievalQuery,
    StaleIndexError,
    corpus_sha256,
    load_index_manifest,
    load_retrieval_config,
    sha256_text,
    validate_index_manifest,
)
from .retrieval.contracts import canonical_json
from .retrieval.corpus import load_curated_corpus
from .schemas import Citation

_EXPLICIT_SOURCE_TITLE_RE = re.compile(r"《([^\r\n《》]{2,200})》")
_EXPLICIT_PAGE_RE = re.compile(r"第\s*(\d{1,4})\s*页")
_LOCATOR_PAGE_SPAN_RE = re.compile(r"(?:PDF\s*)?第\s*(\d{1,4})\s*(?:[-–—~～至]\s*(\d{1,4}))?\s*页")
_LOCATOR_PAGE_SUFFIX_RE = re.compile(r"(?:PDF\s*)?页\s*(\d{1,4})")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    score: float
    lexical_score: float
    semantic_score: float
    citation: Citation
    topics: tuple[str, ...]
    jurisdiction: str
    allowed_claim_scope: tuple[str, ...]
    treatment_details_allowed: bool
    claim_type: str
    recommendation_strength: str | None
    certainty: str | None
    text: str


@dataclass(frozen=True, slots=True)
class RetrievalAttestation:
    """Result of re-validating hits at the knowledge-to-agent trust boundary."""

    hits: tuple[RetrievalHit, ...]
    rejected_count: int
    rejection_reasons: tuple[str, ...]


class RetrievalStatusCode(StrEnum):
    """Stable online retrieval outcomes suitable for receipts and telemetry."""

    NOT_RUN = "not_run"
    SPARSE_OK = "sparse_ok"
    DENSE_OK = "dense_ok"
    HYBRID_OK = "hybrid_ok"
    DENSE_DISABLED_FALLBACK_SPARSE = "dense_disabled_fallback_sparse"
    DENSE_INIT_STALE_FALLBACK_SPARSE = "dense_init_stale_fallback_sparse"
    DENSE_INIT_DEPENDENCY_FALLBACK_SPARSE = "dense_init_dependency_fallback_sparse"
    DENSE_INIT_UNAVAILABLE_FALLBACK_SPARSE = "dense_init_unavailable_fallback_sparse"
    DENSE_INIT_CONTRACT_FALLBACK_SPARSE = "dense_init_contract_fallback_sparse"
    DENSE_QUERY_MISMATCH_FALLBACK_SPARSE = "dense_query_mismatch_fallback_sparse"
    DENSE_QUERY_UNAVAILABLE_FALLBACK_SPARSE = "dense_query_unavailable_fallback_sparse"


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeState:
    requested_mode: str
    effective_mode: str
    status_code: RetrievalStatusCode
    dense_initialized: bool
    fallback_reason: str | None


def _normalized_source_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _locator_covers_page(locator: str, page: int) -> bool:
    for match in _LOCATOR_PAGE_SPAN_RE.finditer(locator):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if min(start, end) <= page <= max(start, end):
            return True
    return any(int(match.group(1)) == page for match in _LOCATOR_PAGE_SUFFIX_RE.finditer(locator))


class GuidelineRetriever:
    """Reviewed offline corpus backed by the versioned retrieval subsystem.

    The checked-in profile runs metadata-gated BM25. A configured dense/hybrid
    profile is initialized lazily on its first query, after immutable corpus,
    embedding, and Qdrant generation identities have been verified.
    """

    def __init__(
        self,
        knowledge_dir: Path,
        *,
        retrieval_config_path: Path | None = None,
        embedding_adapter: Any | None = None,
        vector_store: Any | None = None,
    ):
        if (embedding_adapter is None) != (vector_store is None):
            raise ValueError("embedding_adapter and vector_store must be supplied together")
        self.manifest_path = knowledge_dir / "source_manifest.json"
        self.chunks_path = knowledge_dir / "chunks.jsonl"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        snapshot = load_curated_corpus(knowledge_dir)
        self.snapshot_id = snapshot.snapshot_id
        self.manifest_sha256 = snapshot.source_manifest_sha256
        self.chunks_sha256 = snapshot.chunks_sha256
        self.sources = snapshot.sources
        self.chunks = list(snapshot.chunks)
        documents = list(snapshot.documents)
        self._documents = tuple(documents)
        self._snapshot = snapshot
        self._chunks_by_id = {str(chunk["chunk_id"]): chunk for chunk in snapshot.chunks}
        self._trusted_hit_identities: dict[str, str] = {}
        if not documents:
            raise ValueError("guideline knowledge snapshot contains no chunks")

        retrieval_config_path = retrieval_config_path or (
            knowledge_dir.parent / "configs" / "retrieval.yaml"
        )
        runtime = load_retrieval_config(retrieval_config_path)
        if runtime.reranker.enabled:
            raise ValueError("GuidelineRetriever online reranking is not enabled")
        self.retrieval_config = runtime
        self._retrieval_config_path = retrieval_config_path
        self._requested_mode = runtime.engine.retrieval_mode
        sparse_config = replace(runtime.engine, retrieval_mode="sparse")
        self._sparse_engine = RetrievalEngine(documents, config=sparse_config)
        self.engine = self._sparse_engine
        self._dense_engine: RetrievalEngine | None = None
        self._dense_initialization_attempted = False
        self._dense_initialization_status: RetrievalStatusCode | None = None
        self._dense_initialization_error: Exception | None = None
        self._dense_initialization_lock = Lock()
        self._closed = False
        self._injected_embedding_adapter = embedding_adapter
        self._injected_vector_store = vector_store
        for document in documents:
            chunk = self._chunks_by_id[document.chunk_id]
            source = self.sources[document.source_id]
            canonical_hit = self._build_hit(
                chunk=chunk,
                document=document,
                source=source,
                score=0.0,
                lexical_score=0.0,
                semantic_score=0.0,
            )
            self._trusted_hit_identities[document.chunk_id] = self._hit_identity(canonical_hit)
        self.retrieval_version = runtime.engine.retrieval_version
        self.corpus_generation_id = self._sparse_engine.sparse_generation_id
        self.last_receipt = None
        self.last_runtime_state = RetrievalRuntimeState(
            requested_mode=self._requested_mode,
            effective_mode="not_run",
            status_code=RetrievalStatusCode.NOT_RUN,
            dense_initialized=False,
            fallback_reason=None,
        )

    def close(self) -> None:
        """Close initialized retrieval resources; never trigger lazy loading."""

        with self._dense_initialization_lock:
            if self._closed:
                return
            self._closed = True
            self._sparse_engine.close()
            if self._dense_engine is not None:
                self._dense_engine.close()
            else:
                close = getattr(self._injected_vector_store, "close", None)
                if callable(close):
                    close()

    def _embedding_provenance(self) -> EmbeddingProvenance:
        dense = self.retrieval_config.dense
        return EmbeddingProvenance(
            provider=dense.adapter,
            model_id=dense.model_id,
            dimensions=dense.dimensions,
            normalized=dense.normalized,
            model_sha256=dense.model_sha256,
            revision=dense.revision,
            query_prefix=dense.query_prefix,
            document_prefix=dense.document_prefix,
        )

    def _default_embedding_adapter(self) -> Any:
        dense = self.retrieval_config.dense
        provenance = self._embedding_provenance()
        if dense.adapter == "bge_m3_local":
            if not dense.model_path_env or not dense.cache_dir_env or not dense.device_env:
                raise ContractError("BGE-M3 environment variable names are not configured")
            return BgeM3LocalEmbeddingAdapter(
                provenance=provenance,
                model_path_env=dense.model_path_env,
                cache_dir_env=dense.cache_dir_env,
                device_env=dense.device_env,
                query_prefix=dense.query_prefix,
                document_prefix=dense.document_prefix,
            )
        if dense.adapter == "openai_compatible_http" and dense.endpoint:
            return OpenAICompatibleEmbeddingAdapter(
                endpoint=dense.endpoint,
                provenance=provenance,
                timeout_seconds=dense.timeout_seconds,
                require_loopback=dense.require_loopback,
                query_prefix=dense.query_prefix,
                document_prefix=dense.document_prefix,
            )
        raise ContractError("online retrieval cannot construct the configured dense adapter")

    def _default_vector_store(self) -> QdrantLocalVectorStore:
        vector = self.retrieval_config.vector_store
        if (
            vector.backend != "qdrant_local"
            or vector.qdrant_path is None
            or not vector.qdrant_collection
        ):
            raise ContractError("online dense retrieval requires qdrant_local")
        return QdrantLocalVectorStore(
            QdrantLocalConfig(
                root=vector.qdrant_path,
                collection=vector.qdrant_collection,
            )
        )

    def _validate_dense_generation(self, embedder: Any, store: Any) -> None:
        manifest = store.manifest
        expected_corpus = corpus_sha256(self._documents)
        mismatches: list[str] = []
        if manifest.corpus_sha256 != expected_corpus:
            mismatches.append("corpus_sha256")
        if manifest.embedding_fingerprint != embedder.provenance.fingerprint:
            mismatches.append("embedding_fingerprint")
        if manifest.dimensions != embedder.provenance.dimensions:
            mismatches.append("dimensions")
        if manifest.vector_count >= 0 and manifest.vector_count != len(self._documents):
            mismatches.append("chunk_count")
        if mismatches:
            raise StaleIndexError("stale vector index: " + ", ".join(mismatches))
        if self._injected_vector_store is not None:
            return
        vector_root = self.retrieval_config.vector_store.qdrant_path
        assert vector_root is not None
        portable = load_index_manifest(vector_root / "index-manifest.json")
        validate_index_manifest(
            portable,
            self._documents,
            embedder.provenance,
            source_manifest_sha256=self._snapshot.source_manifest_sha256,
            chunks_sha256=self._snapshot.chunks_sha256,
        )
        if portable.generation_id != manifest.generation_id:
            raise StaleIndexError("stale vector index: promoted generation_id")

    def _initialize_dense_engine(self) -> RetrievalEngine | None:
        # Concurrent first queries must observe one completed initialization, not
        # interpret another request's in-progress attempt as a sparse fallback.
        with self._dense_initialization_lock:
            return self._initialize_dense_engine_locked()

    def _initialize_dense_engine_locked(self) -> RetrievalEngine | None:
        if self._closed:
            raise BackendUnavailableError("guideline retriever is closed")
        if self._dense_initialization_attempted:
            if (
                self._dense_initialization_error is not None
                and not self.retrieval_config.engine.fallback_to_sparse
            ):
                raise self._dense_initialization_error.with_traceback(None)
            return self._dense_engine
        self._dense_initialization_attempted = True
        try:
            embedder = self._injected_embedding_adapter or self._default_embedding_adapter()
            store = self._injected_vector_store or self._default_vector_store()
            self._validate_dense_generation(embedder, store)
            self._dense_engine = RetrievalEngine(
                self._documents,
                config=self.retrieval_config.engine,
                embedder=embedder,
                vector_store=store,
            )
            self._dense_initialization_status = None
            return self._dense_engine
        except Exception as exc:  # noqa: BLE001 - optional backend isolation boundary
            self._dense_initialization_error = exc
            if isinstance(exc, GenerationMismatchError):
                status = RetrievalStatusCode.DENSE_INIT_STALE_FALLBACK_SPARSE
            elif isinstance(exc, OptionalDependencyError):
                status = RetrievalStatusCode.DENSE_INIT_DEPENDENCY_FALLBACK_SPARSE
            elif isinstance(exc, ContractError):
                status = RetrievalStatusCode.DENSE_INIT_CONTRACT_FALLBACK_SPARSE
            else:
                status = RetrievalStatusCode.DENSE_INIT_UNAVAILABLE_FALLBACK_SPARSE
            self._dense_initialization_status = status
            if not self.retrieval_config.engine.fallback_to_sparse:
                raise
        return None

    @staticmethod
    def _receipt_with_runtime_fallback(receipt: Any, status: RetrievalStatusCode) -> Any:
        backend_status = dict(receipt.backend_status)
        backend_status["dense"] = (
            "mismatch"
            if status == RetrievalStatusCode.DENSE_INIT_STALE_FALLBACK_SPARSE
            else "unavailable"
        )
        if status == RetrievalStatusCode.DENSE_DISABLED_FALLBACK_SPARSE:
            backend_status["dense"] = "disabled"
        updated = replace(
            receipt,
            backend_status=backend_status,
            fallback_reason=status.value,
        )
        material = updated.to_dict()
        material.pop("retrieval_id", None)
        return replace(
            updated,
            retrieval_id=sha256_text(canonical_json(material))[:32],
        )

    def _engine_for_request(self) -> tuple[RetrievalEngine, RetrievalStatusCode | None]:
        if self._closed:
            raise BackendUnavailableError("guideline retriever is closed")
        if self._requested_mode == "sparse":
            return self._sparse_engine, None
        if not self.retrieval_config.dense.enabled:
            if not self.retrieval_config.engine.fallback_to_sparse:
                raise ContractError(
                    "dense retrieval is disabled and fallback_to_sparse is false"
                )
            return (
                self._sparse_engine,
                RetrievalStatusCode.DENSE_DISABLED_FALLBACK_SPARSE,
            )
        dense_engine = self._initialize_dense_engine()
        if dense_engine is None:
            return (
                self._sparse_engine,
                self._dense_initialization_status
                or RetrievalStatusCode.DENSE_INIT_UNAVAILABLE_FALLBACK_SPARSE,
            )
        return dense_engine, None

    def _record_runtime_state(
        self,
        *,
        initialization_fallback: RetrievalStatusCode | None,
    ) -> None:
        assert self.last_receipt is not None
        if initialization_fallback is not None:
            status = initialization_fallback
            effective = "sparse"
        elif self._requested_mode == "sparse":
            status = RetrievalStatusCode.SPARSE_OK
            effective = "sparse"
        elif self.last_receipt.backend_status.get("dense") == "used":
            status = (
                RetrievalStatusCode.DENSE_OK
                if self._requested_mode == "dense"
                else RetrievalStatusCode.HYBRID_OK
            )
            effective = self._requested_mode
        elif self.last_receipt.backend_status.get("dense") == "mismatch":
            status = RetrievalStatusCode.DENSE_QUERY_MISMATCH_FALLBACK_SPARSE
            effective = "sparse"
        else:
            status = RetrievalStatusCode.DENSE_QUERY_UNAVAILABLE_FALLBACK_SPARSE
            effective = "sparse"
        self.last_runtime_state = RetrievalRuntimeState(
            requested_mode=self._requested_mode,
            effective_mode=effective,
            status_code=status,
            dense_initialized=self._dense_engine is not None,
            fallback_reason=self.last_receipt.fallback_reason,
        )

    @staticmethod
    def _build_hit(
        *,
        chunk: dict[str, Any],
        document: RetrievalDocument,
        source: dict[str, Any],
        score: float,
        lexical_score: float,
        semantic_score: float,
    ) -> RetrievalHit:
        return RetrievalHit(
            score=score,
            lexical_score=lexical_score,
            semantic_score=semantic_score,
            citation=Citation(
                chunk_id=document.chunk_id,
                source_id=document.source_id,
                title=document.title,
                organization=document.organization,
                publication_year=int(chunk["publication_year"]),
                section=str(chunk["section"]),
                locator=document.locator,
                url=document.url,
                support_text=str(chunk["support_text"]),
            ),
            topics=tuple(chunk["topics"]),
            jurisdiction=document.jurisdiction,
            allowed_claim_scope=document.allowed_claim_scopes,
            treatment_details_allowed=bool(source.get("treatment_details_allowed", False)),
            claim_type=str(chunk.get("claim_type", "unspecified")),
            recommendation_strength=chunk.get("recommendation_strength"),
            certainty=chunk.get("certainty"),
            text=str(chunk["text"]),
        )

    @staticmethod
    def _hit_identity(hit: RetrievalHit) -> str:
        evidence = {
            "citation": hit.citation.model_dump(mode="json"),
            "topics": hit.topics,
            "jurisdiction": hit.jurisdiction,
            "allowed_claim_scope": hit.allowed_claim_scope,
            "treatment_details_allowed": hit.treatment_details_allowed,
            "claim_type": hit.claim_type,
            "recommendation_strength": hit.recommendation_strength,
            "certainty": hit.certainty,
            "text": hit.text,
        }
        return sha256_text(canonical_json(evidence))

    def attest_hits(
        self,
        hits: Iterable[RetrievalHit],
        *,
        required_claim_scopes: Iterable[str] | None = None,
        jurisdictions: Iterable[str] | None = None,
        topics: Iterable[str] | None = None,
    ) -> RetrievalAttestation:
        """Admit only immutable corpus hits that remain within the caller's claim scope.

        Retrieval results are data, never executable authority. This second boundary
        prevents a compromised adapter or in-process integration from smuggling a
        fabricated chunk, altered support text, or an out-of-scope citation into an
        Agent response.
        """

        required_scopes = set(required_claim_scopes or ())
        allowed_jurisdictions = set(jurisdictions or ())
        allowed_topics = set(topics or ())
        accepted: list[RetrievalHit] = []
        reasons: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            if not isinstance(hit, RetrievalHit):
                reasons.append("invalid_hit_type")
                continue
            chunk_id = hit.citation.chunk_id
            trusted_identity = self._trusted_hit_identities.get(chunk_id)
            if trusted_identity is None:
                reasons.append("unknown_chunk")
                continue
            if not all(
                math.isfinite(value) and value >= 0
                for value in (hit.score, hit.lexical_score, hit.semantic_score)
            ):
                reasons.append("invalid_score")
                continue
            if self._hit_identity(hit) != trusted_identity:
                reasons.append("evidence_identity_mismatch")
                continue
            if required_scopes and not required_scopes.intersection(hit.allowed_claim_scope):
                reasons.append("claim_scope_mismatch")
                continue
            if allowed_jurisdictions and hit.jurisdiction not in allowed_jurisdictions:
                reasons.append("jurisdiction_mismatch")
                continue
            if allowed_topics and not allowed_topics.intersection(hit.topics):
                reasons.append("topic_mismatch")
                continue
            if chunk_id in seen:
                reasons.append("duplicate_chunk")
                continue
            seen.add(chunk_id)
            accepted.append(hit)
        return RetrievalAttestation(
            hits=tuple(accepted),
            rejected_count=len(reasons),
            rejection_reasons=tuple(reasons),
        )

    def _explicit_source_constraint(self, query: str) -> tuple[set[str] | None, set[int]]:
        requested_titles = [
            _normalized_source_title(value) for value in _EXPLICIT_SOURCE_TITLE_RE.findall(query)
        ]
        requested_pages = {int(value) for value in _EXPLICIT_PAGE_RE.findall(query)}
        if not requested_titles:
            return None, requested_pages

        matched_source_ids: set[str] = set()
        normalized_sources = {
            source_id: _normalized_source_title(str(source["title"]))
            for source_id, source in self.sources.items()
        }
        for requested_title in requested_titles:
            title_matches = {
                source_id
                for source_id, source_title in normalized_sources.items()
                if requested_title == source_title
                or requested_title in source_title
                or source_title in requested_title
            }
            if not title_matches:
                # A user-requested source must not be substituted with a similarly
                # worded real source. Empty means fail closed after receipt creation.
                return set(), requested_pages
            matched_source_ids.update(title_matches)
        return matched_source_ids, requested_pages

    def retrieve(
        self,
        query: str,
        *,
        topics: Iterable[str] | None = None,
        jurisdictions: Iterable[str] | None = None,
        required_claim_scopes: Iterable[str] | None = None,
        required_source_ids: Iterable[str] | None = None,
        minimum_lexical_score: float = 0.0,
        minimum_semantic_score: float = 0.0,
        top_k: int = 4,
    ) -> list[RetrievalHit]:
        if not query.strip():
            return []
        if minimum_lexical_score < 0.0 or minimum_semantic_score < 0.0:
            raise ValueError("minimum retrieval scores must be non-negative")
        if not 1 <= top_k <= 8:
            raise ValueError("top_k must be between 1 and 8")
        source_ids = {item.strip() for item in (required_source_ids or ()) if item.strip()}
        if source_ids and not source_ids.issubset(self.sources):
            return []
        filters = RetrievalFilter(
            topics_any=tuple(sorted(set(topics or ()))),
            jurisdictions=tuple(sorted(set(jurisdictions or ()))),
            required_claim_scopes_any=tuple(sorted(set(required_claim_scopes or ()))),
            required_source_ids=tuple(sorted(source_ids)),
        )
        query_material = {
            "query_sha256": sha256_text(query.strip()),
            "filters": filters.to_dict(),
        }
        engine, initialization_fallback = self._engine_for_request()
        result = engine.retrieve(
            RetrievalQuery(
                query_id=f"agent-{sha256_text(canonical_json(query_material))[:24]}",
                text=query,
                top_k=min(100, max(top_k * 4, top_k)),
                filters=filters,
            )
        )
        self.last_receipt = result.receipt
        if initialization_fallback is not None:
            self.last_receipt = self._receipt_with_runtime_fallback(
                self.last_receipt,
                initialization_fallback,
            )
        self._record_runtime_state(initialization_fallback=initialization_fallback)
        explicit_source_ids, explicit_pages = self._explicit_source_constraint(query)
        hits: list[RetrievalHit] = []
        relevance_gate_enabled = minimum_lexical_score > 0 or minimum_semantic_score > 0
        for engine_hit in result.hits:
            lexical_score = float(engine_hit.backend_scores.get("bm25", 0.0))
            # Cosine similarity may be negative, while the long-standing
            # knowledge-layer RetrievalHit/attestation contract is non-negative.
            semantic_score = max(
                0.0,
                float(engine_hit.backend_scores.get("dense", 0.0)),
            )
            if relevance_gate_enabled and not (
                (minimum_lexical_score > 0 and lexical_score >= minimum_lexical_score)
                or (minimum_semantic_score > 0 and semantic_score >= minimum_semantic_score)
            ):
                continue
            document = engine_hit.document
            if source_ids and document.source_id not in source_ids:
                continue
            if explicit_source_ids is not None and document.source_id not in explicit_source_ids:
                continue
            if explicit_pages and not all(
                _locator_covers_page(document.locator, page) for page in explicit_pages
            ):
                continue
            chunk = self._chunks_by_id[document.chunk_id]
            source = self.sources[document.source_id]
            hits.append(
                self._build_hit(
                    chunk=chunk,
                    document=document,
                    source=source,
                    score=float(engine_hit.score),
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def retrieve_scoped(
        self,
        query: str,
        *,
        required_claim_scopes: Iterable[str],
        preferred_topics: Iterable[str] | None = None,
        ranking_terms: Iterable[str] | None = None,
        jurisdictions: Iterable[str] | None = None,
        required_source_ids: Iterable[str] | None = None,
        minimum_lexical_score: float = 0.0,
        minimum_semantic_score: float = 0.0,
        top_k: int = 4,
    ) -> list[RetrievalHit]:
        """Retrieve with a hard claim-scope boundary and soft topic preferences.

        Claim scope is authority, so it is applied by the retrieval engine and
        re-attested by the caller. Topics, population terms and product names
        affect ordering only. If no preferred-topic hit exists, callers still
        receive same-scope evidence and can return a precise PARTIAL result
        instead of silently crossing into another medical scope.
        """

        scopes = tuple(
            dict.fromkeys(item.strip() for item in required_claim_scopes if item.strip())
        )
        if not scopes:
            return []
        if not 1 <= top_k <= 8:
            raise ValueError("top_k must be between 1 and 8")
        topics = {item.casefold() for item in (preferred_topics or ()) if item.strip()}
        terms = tuple(
            dict.fromkeys(item.casefold().strip() for item in (ranking_terms or ()) if item.strip())
        )
        # Pull the bounded maximum so soft preferences can reorder evidence
        # without ever weakening the scope boundary.
        candidates = self.retrieve(
            query,
            jurisdictions=jurisdictions,
            required_claim_scopes=scopes,
            required_source_ids=required_source_ids,
            minimum_lexical_score=minimum_lexical_score,
            minimum_semantic_score=minimum_semantic_score,
            top_k=8,
        )

        def preference_key(hit: RetrievalHit) -> tuple[int, int, float, float, str]:
            topic_overlap = len(topics.intersection(item.casefold() for item in hit.topics))
            searchable = " ".join(
                (
                    hit.text,
                    hit.citation.support_text,
                    hit.citation.section,
                    hit.citation.title,
                    *hit.topics,
                )
            ).casefold()
            term_matches = sum(1 for term in terms if term in searchable)
            return (
                topic_overlap,
                term_matches,
                hit.score,
                hit.lexical_score + hit.semantic_score,
                hit.citation.chunk_id,
            )

        return sorted(candidates, key=preference_key, reverse=True)[:top_k]
