from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Literal

from .errors import ContractError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ReviewStatus = Literal["approved", "pending_medical_review", "rejected", "superseded"]
BackendStatus = Literal["used", "disabled", "unavailable", "mismatch", "not_queried"]
_REVIEW_STATUSES = {"approved", "pending_medical_review", "rejected", "superseded"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_values(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    cleaned = tuple(dict.fromkeys(item.strip() for item in values if item.strip()))
    if len(cleaned) != len(values):
        raise ContractError(f"{field_name} must contain unique non-empty values")
    return cleaned


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    """Immutable, content-addressed unit admitted to a retrieval index."""

    chunk_id: str
    source_id: str
    text: str
    content_sha256: str
    source_sha256: str
    title: str
    locator: str
    jurisdiction: str
    publication_date: str | None
    topics: tuple[str, ...]
    allowed_claim_scopes: tuple[str, ...]
    review_status: ReviewStatus
    retrievable: bool
    language: str = "zh-CN"
    organization: str = ""
    url: str = ""
    source_hash_kind: Literal["artifact_sha256", "manifest_record_sha256"] = "artifact_sha256"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "chunk_id",
            "source_id",
            "text",
            "title",
            "locator",
            "jurisdiction",
            "language",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ContractError(f"{field_name} must be non-empty")
        if not _SHA256_RE.fullmatch(self.content_sha256):
            raise ContractError("content_sha256 must be a lowercase SHA-256 digest")
        if sha256_text(self.text) != self.content_sha256:
            raise ContractError(f"content hash mismatch for chunk {self.chunk_id}")
        if not _SHA256_RE.fullmatch(self.source_sha256):
            raise ContractError("source_sha256 must be a lowercase SHA-256 digest")
        if self.source_hash_kind not in {"artifact_sha256", "manifest_record_sha256"}:
            raise ContractError("unsupported source_hash_kind")
        object.__setattr__(self, "topics", _clean_values(self.topics, "topics"))
        object.__setattr__(
            self,
            "allowed_claim_scopes",
            _clean_values(self.allowed_claim_scopes, "allowed_claim_scopes"),
        )
        if not self.allowed_claim_scopes:
            raise ContractError("allowed_claim_scopes cannot be empty")
        if self.review_status not in _REVIEW_STATUSES:
            raise ContractError("unsupported review_status")
        if self.publication_date is not None:
            try:
                date.fromisoformat(self.publication_date)
            except ValueError as exc:
                raise ContractError("publication_date must be ISO-8601 YYYY-MM-DD") from exc
        if self.schema_version != 1:
            raise ContractError("unsupported RetrievalDocument schema_version")

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.locator}\n{self.text}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalFilter:
    """Fail-closed metadata policy applied before scoring."""

    topics_any: tuple[str, ...] = ()
    jurisdictions: tuple[str, ...] = ()
    required_claim_scopes_any: tuple[str, ...] = ()
    published_on_or_after: str | None = None
    published_on_or_before: str | None = None
    allowed_review_statuses: tuple[ReviewStatus, ...] = ("approved",)
    retrievable_only: bool = True
    required_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_source_ids",
            _clean_values(self.required_source_ids, "required_source_ids"),
        )
        object.__setattr__(self, "topics_any", _clean_values(self.topics_any, "topics_any"))
        object.__setattr__(
            self,
            "jurisdictions",
            _clean_values(self.jurisdictions, "jurisdictions"),
        )
        object.__setattr__(
            self,
            "required_claim_scopes_any",
            _clean_values(self.required_claim_scopes_any, "required_claim_scopes_any"),
        )
        if not self.allowed_review_statuses:
            raise ContractError("allowed_review_statuses cannot be empty")
        if len(set(self.allowed_review_statuses)) != len(self.allowed_review_statuses) or not set(
            self.allowed_review_statuses
        ).issubset(_REVIEW_STATUSES):
            raise ContractError("allowed_review_statuses contains invalid or duplicate values")
        for bound in (self.published_on_or_after, self.published_on_or_before):
            if bound is not None:
                try:
                    date.fromisoformat(bound)
                except ValueError as exc:
                    raise ContractError("publication date filters must be ISO-8601") from exc
        if (
            self.published_on_or_after
            and self.published_on_or_before
            and self.published_on_or_after > self.published_on_or_before
        ):
            raise ContractError("publication date filter lower bound exceeds upper bound")

    def admits(self, document: RetrievalDocument) -> bool:
        if self.required_source_ids and document.source_id not in self.required_source_ids:
            return False
        if self.retrievable_only and not document.retrievable:
            return False
        if document.review_status not in self.allowed_review_statuses:
            return False
        if self.topics_any and not set(self.topics_any).intersection(document.topics):
            return False
        if self.jurisdictions and document.jurisdiction not in self.jurisdictions:
            return False
        if self.required_claim_scopes_any and not set(self.required_claim_scopes_any).intersection(
            document.allowed_claim_scopes
        ):
            return False
        if self.published_on_or_after or self.published_on_or_before:
            if document.publication_date is None:
                return False
            if (
                self.published_on_or_after
                and document.publication_date < self.published_on_or_after
            ):
                return False
            if (
                self.published_on_or_before
                and document.publication_date > self.published_on_or_before
            ):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        # Metadata lists have set semantics. Equivalent caller orderings must
        # produce the same request identity and the same backend policy.
        return {
            key: tuple(sorted(value)) if isinstance(value, (tuple, list)) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    query_id: str
    text: str
    top_k: int = 5
    filters: RetrievalFilter = field(default_factory=RetrievalFilter)

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.text.strip():
            raise ContractError("query_id and text must be non-empty")
        if not 1 <= self.top_k <= 100:
            raise ContractError("top_k must be between 1 and 100")
        object.__setattr__(self, "text", self.text.strip())

    @property
    def query_sha256(self) -> str:
        return sha256_text(self.text.strip())

    @property
    def request_sha256(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "query_id": self.query_id,
                    "query_sha256": self.query_sha256,
                    "top_k": self.top_k,
                    "filters": self.filters.to_dict(),
                }
            )
        )


@dataclass(frozen=True, slots=True)
class EmbeddingProvenance:
    provider: str
    model_id: str
    dimensions: int
    normalized: bool
    model_sha256: str | None = None
    revision: str | None = None
    # These fields deliberately change even the empty-prefix fingerprint relative
    # to legacy manifests, whose embedding preprocessing was not attested.
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model_id.strip():
            raise ContractError("embedding provider and model_id must be non-empty")
        if self.dimensions <= 0:
            raise ContractError("embedding dimensions must be positive")
        if self.model_sha256 is not None and not _SHA256_RE.fullmatch(self.model_sha256):
            raise ContractError("embedding model_sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.query_prefix, str) or not isinstance(self.document_prefix, str):
            raise ContractError("embedding prefixes must be strings")

    @property
    def fingerprint(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    provenance: EmbeddingProvenance

    def __post_init__(self) -> None:
        if not self.vectors:
            raise ContractError("embedding response cannot be empty")
        for vector in self.vectors:
            if len(vector) != self.provenance.dimensions:
                raise ContractError("embedding dimensions do not match declared provenance")
            if any(not math.isfinite(value) for value in vector):
                raise ContractError("embedding contains a non-finite value")


@dataclass(frozen=True, slots=True)
class RankedItem:
    chunk_id: str
    score: float
    backend: str
    rank: int


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    rank: int
    score: float
    document: RetrievalDocument
    backend_ranks: dict[str, int]
    backend_scores: dict[str, float]
    rerank_score: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Backend-neutral chunk returned by dense, sparse, or hybrid retrieval."""

    rank: int
    score: float
    document: RetrievalDocument
    backend_ranks: dict[str, int] = field(default_factory=dict)
    backend_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1 or not math.isfinite(self.score):
            raise ContractError("retrieved chunk rank/score is invalid")
        if any(rank < 1 for rank in self.backend_ranks.values()):
            raise ContractError("retrieved chunk backend rank is invalid")
        if any(not math.isfinite(score) for score in self.backend_scores.values()):
            raise ContractError("retrieved chunk backend score is invalid")

    @property
    def chunk_id(self) -> str:
        return self.document.chunk_id

    @property
    def source_id(self) -> str:
        return self.document.source_id

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the same governed metadata shape for every backend."""

        return {
            "chunk_id": self.document.chunk_id,
            "source_id": self.document.source_id,
            "content_sha256": self.document.content_sha256,
            "source_sha256": self.document.source_sha256,
            "title": self.document.title,
            "locator": self.document.locator,
            "jurisdiction": self.document.jurisdiction,
            "publication_date": self.document.publication_date,
            "topics": self.document.topics,
            "allowed_claim_scopes": self.document.allowed_claim_scopes,
            "review_status": self.document.review_status,
            "retrievable": self.document.retrievable,
            "language": self.document.language,
            "organization": self.document.organization,
            "url": self.document.url,
        }


# Public name used by the high-level retrievers.  Keeping the original name is
# backward compatible with the deployed GuidelineRetriever and smoke suite.
MetadataFilter = RetrievalFilter


@dataclass(frozen=True, slots=True)
class RetrievalReceipt:
    schema_version: int
    retrieval_id: str
    retrieval_version: str
    config_sha256: str
    query_id: str
    query_sha256: str
    corpus_generation_id: str
    corpus_manifest_sha256: str
    backend_status: dict[str, BackendStatus]
    embedding_fingerprint: str | None
    fallback_reason: str | None
    suppressed_duplicate_ids: tuple[str, ...]
    # Version 1 receipts remain readable, but these absent fields are unknown;
    # they must never be reconstructed from present-day defaults or results.
    top_k: int | None = None
    filters: dict[str, Any] | None = None
    request_sha256: str | None = None
    returned_chunks: tuple[dict[str, str], ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ContractError("unsupported RetrievalReceipt schema_version")
        object.__setattr__(self, "suppressed_duplicate_ids", tuple(self.suppressed_duplicate_ids))
        if self.schema_version == 1:
            if any(
                value is not None
                for value in (self.top_k, self.filters, self.request_sha256, self.returned_chunks)
            ):
                raise ContractError("legacy receipt cannot claim version 2 request/result identity")
            return
        if self.top_k is None or not 1 <= self.top_k <= 100:
            raise ContractError("receipt top_k must be between 1 and 100")
        if self.filters is None or self.returned_chunks is None:
            raise ContractError("version 2 receipt requires filters and returned_chunks")
        if set(self.filters) != set(RetrievalFilter().to_dict()):
            raise ContractError("version 2 receipt must record every metadata filter")
        policy = RetrievalFilter(**self.filters)
        object.__setattr__(self, "filters", policy.to_dict())
        expected_request_hash = sha256_text(
            canonical_json(
                {
                    "query_id": self.query_id,
                    "query_sha256": self.query_sha256,
                    "top_k": self.top_k,
                    "filters": self.filters,
                }
            )
        )
        if self.request_sha256 != expected_request_hash:
            raise ContractError("receipt request hash mismatch")
        identities = tuple(dict(chunk) for chunk in self.returned_chunks)
        if len(identities) > self.top_k:
            raise ContractError("receipt result count exceeds top_k")
        seen: set[str] = set()
        for chunk in identities:
            if set(chunk) != {"chunk_id", "source_id", "content_sha256", "source_sha256"}:
                raise ContractError("receipt chunk identity has invalid fields")
            if not chunk["chunk_id"] or not chunk["source_id"] or chunk["chunk_id"] in seen:
                raise ContractError("receipt chunk IDs must be non-empty and unique")
            if any(
                not _SHA256_RE.fullmatch(chunk[name])
                for name in ("content_sha256", "source_sha256")
            ):
                raise ContractError("receipt chunk identity requires SHA-256 hashes")
            seen.add(chunk["chunk_id"])
        object.__setattr__(self, "returned_chunks", identities)

    def to_dict(self) -> dict[str, Any]:
        material = asdict(self)
        if self.schema_version == 1:
            for key in ("top_k", "filters", "request_sha256", "returned_chunks"):
                material.pop(key)
        return material

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RetrievalReceipt:
        """Read historical v1 or current v2 without silently upgrading identity."""

        try:
            receipt = cls(**value)
            material = receipt.to_dict()
            retrieval_id = material.pop("retrieval_id")
            if sha256_text(canonical_json(material))[:32] != retrieval_id:
                raise ContractError("receipt identity hash mismatch")
            return receipt
        except ContractError:
            raise
        except (TypeError, ValueError, KeyError) as exc:
            raise ContractError("invalid retrieval receipt") from exc


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    hits: tuple[RetrievalHit, ...]
    receipt: RetrievalReceipt
