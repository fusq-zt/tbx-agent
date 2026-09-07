from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import (
    EmbeddingProvenance,
    RetrievalDocument,
    canonical_json,
    sha256_text,
)
from .errors import ContractError, StaleIndexError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def corpus_sha256(documents: Iterable[RetrievalDocument]) -> str:
    """Hash every field that can change index admission or embedded content."""

    ordered = tuple(sorted(documents, key=lambda item: item.chunk_id))
    if not ordered:
        raise ContractError("index corpus cannot be empty")
    if len({item.chunk_id for item in ordered}) != len(ordered):
        raise ContractError("index corpus contains duplicate chunk_id values")
    return sha256_text(
        canonical_json(
            [
                {
                    "chunk_id": item.chunk_id,
                    "content_sha256": item.content_sha256,
                    "source_sha256": item.source_sha256,
                    "source_hash_kind": item.source_hash_kind,
                    "title": item.title,
                    "locator": item.locator,
                    "jurisdiction": item.jurisdiction,
                    "publication_date": item.publication_date,
                    "topics": item.topics,
                    "allowed_claim_scopes": item.allowed_claim_scopes,
                    "review_status": item.review_status,
                    "retrievable": item.retrievable,
                    "language": item.language,
                }
                for item in ordered
            ]
        )
    )


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Portable identity contract for any dense vector index."""

    schema_version: int
    index_backend: str
    generation_id: str
    corpus_sha256: str
    embedding_fingerprint: str
    embedding_provenance: dict[str, object]
    dimensions: int
    chunk_count: int
    source_manifest_sha256: str | None = None
    chunks_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported index manifest schema_version")
        if not self.index_backend.strip() or not self.generation_id.strip():
            raise ContractError("index backend and generation_id must be non-empty")
        for name in ("corpus_sha256", "embedding_fingerprint"):
            if not _SHA256_RE.fullmatch(str(getattr(self, name))):
                raise ContractError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("source_manifest_sha256", "chunks_sha256"):
            value = getattr(self, name)
            if value is not None and not _SHA256_RE.fullmatch(value):
                raise ContractError(f"{name} must be a lowercase SHA-256 digest")
        if self.dimensions < 1 or self.chunk_count < 1:
            raise ContractError("index dimensions and chunk_count must be positive")

    @property
    def manifest_sha256(self) -> str:
        return sha256_text(canonical_json(asdict(self)))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_index_manifest(
    documents: Iterable[RetrievalDocument],
    embedding_provenance: EmbeddingProvenance,
    *,
    index_backend: str,
    source_manifest_sha256: str | None = None,
    chunks_sha256: str | None = None,
) -> IndexManifest:
    ordered = tuple(sorted(documents, key=lambda item: item.chunk_id))
    digest = corpus_sha256(ordered)
    identity = {
        "schema_version": 1,
        "index_backend": index_backend,
        "corpus_sha256": digest,
        "embedding_fingerprint": embedding_provenance.fingerprint,
    }
    return IndexManifest(
        schema_version=1,
        index_backend=index_backend,
        generation_id=sha256_text(canonical_json(identity))[:32],
        corpus_sha256=digest,
        embedding_fingerprint=embedding_provenance.fingerprint,
        embedding_provenance=asdict(embedding_provenance),
        dimensions=embedding_provenance.dimensions,
        chunk_count=len(ordered),
        source_manifest_sha256=source_manifest_sha256,
        chunks_sha256=chunks_sha256,
    )


def validate_index_manifest(
    manifest: IndexManifest,
    documents: Iterable[RetrievalDocument],
    embedding_provenance: EmbeddingProvenance,
    *,
    source_manifest_sha256: str | None = None,
    chunks_sha256: str | None = None,
) -> None:
    ordered = tuple(documents)
    mismatches: list[str] = []
    if manifest.corpus_sha256 != corpus_sha256(ordered):
        mismatches.append("corpus_sha256")
    if manifest.embedding_fingerprint != embedding_provenance.fingerprint:
        mismatches.append("embedding_fingerprint")
    if manifest.dimensions != embedding_provenance.dimensions:
        mismatches.append("dimensions")
    if manifest.chunk_count != len(ordered):
        mismatches.append("chunk_count")
    if (
        source_manifest_sha256 is not None
        and manifest.source_manifest_sha256 != source_manifest_sha256
    ):
        mismatches.append("source_manifest_sha256")
    if chunks_sha256 is not None and manifest.chunks_sha256 != chunks_sha256:
        mismatches.append("chunks_sha256")
    if mismatches:
        raise StaleIndexError("stale vector index: " + ", ".join(mismatches))


def write_index_manifest(path: Path, manifest: IndexManifest) -> None:
    """Atomically persist one canonical manifest."""

    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary.write(canonical_json(manifest.to_dict()))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_index_manifest(path: Path) -> IndexManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("manifest root is not an object")
        return IndexManifest(**value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ContractError(f"cannot load index manifest: {path}") from exc
