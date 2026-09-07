from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import sqlite3
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Protocol
from urllib.parse import quote, urlparse

from .contracts import (
    EmbeddingProvenance,
    RankedItem,
    RetrievalDocument,
    RetrievalFilter,
    canonical_json,
    sha256_text,
)
from .errors import (
    BackendUnavailableError,
    ContractError,
    GenerationMismatchError,
    OptionalDependencyError,
)
from .indexing import corpus_sha256

_COLLECTION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_QDRANT_LOCKS_GUARD = Lock()
_QDRANT_ROOT_LOCKS: dict[Path, Any] = {}


def _qdrant_root_lock(root: Path) -> Any:
    # Qdrant Local exclusively locks its database even for reads. All instances
    # in this process must share the same open/query/close critical section.
    with _QDRANT_LOCKS_GUARD:
        return _QDRANT_ROOT_LOCKS.setdefault(root, RLock())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VectorGenerationManifest:
    schema_version: int
    store_type: str
    store_version: str
    generation_id: str
    corpus_sha256: str
    embedding_fingerprint: str
    embedding_provenance: dict[str, Any]
    dimensions: int
    vector_count: int
    database_sha256: str

    @property
    def manifest_sha256(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


class VectorStore(Protocol):
    @property
    def manifest(self) -> VectorGenerationManifest: ...

    def search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilter,
        limit: int,
    ) -> tuple[RankedItem, ...]: ...


class SQLiteVecStore:
    """Exact, metadata-first sqlite-vec store for a small single-node corpus.

    Each immutable database and manifest are written before an atomic `current.json`
    pointer swap. Readers therefore observe the old or new complete generation.
    """

    STORE_SCHEMA_VERSION = 1

    def __init__(
        self,
        root: Path,
        *,
        maximum_exact_candidates: int = 50_000,
        verify_database_sha256: bool = True,
    ) -> None:
        if maximum_exact_candidates < 1:
            raise ContractError("maximum_exact_candidates must be positive")
        self.root = root.resolve()
        self.maximum_exact_candidates = maximum_exact_candidates
        self.verify_database_sha256 = verify_database_sha256
        self._manifest: VectorGenerationManifest | None = None

    @staticmethod
    def _load_extension(connection: sqlite3.Connection) -> Any:
        try:
            module = importlib.import_module("sqlite_vec")
        except ModuleNotFoundError as exc:
            raise OptionalDependencyError(
                "sqlite-vec is not installed; sparse retrieval remains available"
            ) from exc
        try:
            connection.enable_load_extension(True)
            module.load(connection)
        except (sqlite3.Error, OSError) as exc:
            raise BackendUnavailableError("sqlite-vec extension could not be loaded") from exc
        finally:
            connection.enable_load_extension(False)
        return module

    @staticmethod
    def _corpus_sha256(documents: Sequence[RetrievalDocument]) -> str:
        return corpus_sha256(documents)

    @property
    def manifest(self) -> VectorGenerationManifest:
        if self._manifest is None:
            self._manifest = self._read_current_manifest()
        return self._manifest

    def _read_current_manifest(self) -> VectorGenerationManifest:
        pointer_path = self.root / "current.json"
        if not pointer_path.is_file():
            raise BackendUnavailableError("sqlite-vec store has no promoted generation")
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            generation_id = str(pointer["generation_id"])
            expected_manifest_hash = str(pointer["manifest_sha256"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError("sqlite-vec current pointer is invalid") from exc
        manifest_path = self.root / "generations" / f"{generation_id}.manifest.json"
        database_path = self.root / "generations" / f"{generation_id}.sqlite3"
        if not manifest_path.is_file() or not database_path.is_file():
            raise BackendUnavailableError("sqlite-vec promoted generation is incomplete")
        raw = manifest_path.read_text(encoding="utf-8")
        if sha256_text(raw) != expected_manifest_hash:
            raise GenerationMismatchError("sqlite-vec manifest hash does not match current pointer")
        try:
            manifest = VectorGenerationManifest(**json.loads(raw))
        except (TypeError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError("sqlite-vec generation manifest is invalid") from exc
        if manifest.generation_id != generation_id:
            raise GenerationMismatchError("sqlite-vec manifest generation_id mismatch")
        if self.verify_database_sha256 and _file_sha256(database_path) != manifest.database_sha256:
            raise GenerationMismatchError("sqlite-vec database hash mismatch")
        return manifest

    def build_generation(
        self,
        documents: Iterable[RetrievalDocument],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_provenance: EmbeddingProvenance,
    ) -> VectorGenerationManifest:
        supplied_documents = tuple(documents)
        if not supplied_documents or len(supplied_documents) != len(vectors):
            raise ContractError("documents and vectors must have the same non-zero length")
        paired = sorted(
            zip(supplied_documents, vectors, strict=True),
            key=lambda item: item[0].chunk_id,
        )
        ordered = tuple(document for document, _ in paired)
        if len({item.chunk_id for item in ordered}) != len(ordered):
            raise ContractError("vector generation contains duplicate chunk_id values")
        converted = tuple(tuple(float(value) for value in vector) for _, vector in paired)
        if any(len(vector) != embedding_provenance.dimensions for vector in converted):
            raise ContractError("vector dimensions do not match embedding provenance")
        if any(not math.isfinite(value) for vector in converted for value in vector):
            raise ContractError("vectors must contain only finite values")
        corpus_sha256 = self._corpus_sha256(ordered)
        generation_id = sha256_text(
            canonical_json(
                {
                    "schema_version": self.STORE_SCHEMA_VERSION,
                    "corpus_sha256": corpus_sha256,
                    "embedding_fingerprint": embedding_provenance.fingerprint,
                }
            )
        )[:32]
        generation_dir = self.root / "generations"
        generation_dir.mkdir(parents=True, exist_ok=True)
        final_database = generation_dir / f"{generation_id}.sqlite3"
        final_manifest = generation_dir / f"{generation_id}.manifest.json"
        nonce = uuid.uuid4().hex
        temporary_database = generation_dir / f".{generation_id}.{nonce}.tmp.sqlite3"
        temporary_manifest = generation_dir / f".{generation_id}.{nonce}.tmp.manifest.json"
        temporary_pointer = self.root / f".current.{nonce}.tmp.json"
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(temporary_database)
            sqlite_vec = self._load_extension(connection)
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE chunk_metadata (
                    rowid INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE,
                    document_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    retrievable INTEGER NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    publication_date TEXT
                )
                """
            )
            vector_schema = (
                "CREATE VIRTUAL TABLE chunk_vectors USING "
                f"vec0(embedding float[{embedding_provenance.dimensions}])"
            )
            connection.execute(vector_schema)
            for rowid, (document, vector) in enumerate(
                zip(ordered, converted, strict=True), start=1
            ):
                connection.execute(
                    "INSERT INTO chunk_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        rowid,
                        document.chunk_id,
                        canonical_json(document.to_dict()),
                        document.review_status,
                        int(document.retrievable),
                        document.jurisdiction,
                        document.publication_date,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunk_vectors(rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector)),
                )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise BackendUnavailableError("new sqlite-vec generation failed integrity_check")
            store_version = str(connection.execute("SELECT vec_version()").fetchone()[0])
            connection.close()
            connection = None
            database_sha256 = _file_sha256(temporary_database)
            manifest = VectorGenerationManifest(
                schema_version=self.STORE_SCHEMA_VERSION,
                store_type="sqlite-vec",
                store_version=store_version,
                generation_id=generation_id,
                corpus_sha256=corpus_sha256,
                embedding_fingerprint=embedding_provenance.fingerprint,
                embedding_provenance=asdict(embedding_provenance),
                dimensions=embedding_provenance.dimensions,
                vector_count=len(ordered),
                database_sha256=database_sha256,
            )
            manifest_text = canonical_json(asdict(manifest))
            temporary_manifest.write_text(manifest_text, encoding="utf-8")
            if final_database.exists() or final_manifest.exists():
                if not final_database.is_file() or not final_manifest.is_file():
                    raise GenerationMismatchError("existing vector generation is incomplete")
                if (
                    _file_sha256(final_database) != database_sha256
                    or final_manifest.read_text(encoding="utf-8") != manifest_text
                ):
                    raise GenerationMismatchError("content-addressed vector generation collision")
                temporary_database.unlink()
                temporary_manifest.unlink()
            else:
                os.replace(temporary_database, final_database)
                os.replace(temporary_manifest, final_manifest)
            pointer_text = canonical_json(
                {
                    "schema_version": 1,
                    "generation_id": generation_id,
                    "manifest_sha256": sha256_text(manifest_text),
                }
            )
            temporary_pointer.write_text(pointer_text, encoding="utf-8")
            os.replace(temporary_pointer, self.root / "current.json")
            self._manifest = manifest
            return manifest
        finally:
            if connection is not None:
                connection.close()
            for temporary in (temporary_database, temporary_manifest, temporary_pointer):
                if temporary.exists():
                    temporary.unlink()

    def search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilter,
        limit: int,
    ) -> tuple[RankedItem, ...]:
        manifest = self.manifest
        values = tuple(float(value) for value in vector)
        if len(values) != manifest.dimensions or any(not math.isfinite(value) for value in values):
            raise ContractError("query vector is invalid for the promoted generation")
        if limit < 1:
            raise ContractError("vector search limit must be positive")
        database_path = self.root / "generations" / f"{manifest.generation_id}.sqlite3"
        connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
        try:
            sqlite_vec = self._load_extension(connection)
            rows = connection.execute(
                "SELECT rowid, document_json FROM chunk_metadata ORDER BY rowid"
            ).fetchall()
            admitted: list[tuple[int, RetrievalDocument]] = []
            for rowid, raw_document in rows:
                document = RetrievalDocument(**json.loads(raw_document))
                if filters.admits(document):
                    admitted.append((int(rowid), document))
            if len(admitted) > self.maximum_exact_candidates:
                raise BackendUnavailableError(
                    "sqlite-vec exact metadata-filtered candidate limit exceeded; use Qdrant local"
                )
            packed = sqlite_vec.serialize_float32(values)
            scored: list[tuple[str, float]] = []
            for rowid, document in admitted:
                distance = connection.execute(
                    "SELECT vec_distance_cosine(embedding, ?) FROM chunk_vectors WHERE rowid = ?",
                    (packed, rowid),
                ).fetchone()
                if distance is not None and distance[0] is not None:
                    scored.append((document.chunk_id, 1.0 - float(distance[0])))
            ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
            return tuple(
                RankedItem(chunk_id=chunk_id, score=score, backend="dense", rank=rank)
                for rank, (chunk_id, score) in enumerate(ranked, start=1)
            )
        except (sqlite3.Error, json.JSONDecodeError, TypeError) as exc:
            raise BackendUnavailableError("sqlite-vec query failed validation") from exc
        finally:
            connection.close()


@dataclass(frozen=True, slots=True)
class QdrantLocalConfig:
    root: Path
    collection: str = "tbx-guidelines"

    def __post_init__(self) -> None:
        if not _COLLECTION_RE.fullmatch(self.collection):
            raise ContractError("invalid Qdrant local collection name")


class QdrantLocalVectorStore:
    """Immutable Qdrant generations with serialized single-process access.

    Clients are operation-scoped, so queries never retain the Local database
    lock between calls. A store pins its first manifest; create a new store to
    read an externally promoted generation. Separate processes still require
    separate Local roots or a server deployment.
    """

    STORE_SCHEMA_VERSION = 1

    def __init__(self, config: QdrantLocalConfig) -> None:
        self.config = config
        self.root = config.root.resolve()
        self._manifest: VectorGenerationManifest | None = None
        self._operation_lock = _qdrant_root_lock(self.root)
        self._closed = False

    def close(self) -> None:
        """Wait for in-flight access, then reject subsequent operations."""

        with self._operation_lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendUnavailableError("Qdrant local store is closed")

    @staticmethod
    def _load_qdrant() -> tuple[Any, Any, str]:
        try:
            client_module = importlib.import_module("qdrant_client")
            models = importlib.import_module("qdrant_client.models")
            version = importlib.metadata.version("qdrant-client")
        except (ModuleNotFoundError, importlib.metadata.PackageNotFoundError) as exc:
            raise OptionalDependencyError(
                "qdrant-client is not installed; install the retrieval extra"
            ) from exc
        return client_module.QdrantClient, models, version

    @property
    def manifest(self) -> VectorGenerationManifest:
        with self._operation_lock:
            self._ensure_open()
            if self._manifest is None:
                self._manifest = self._read_current_manifest()
            return self._manifest

    def _read_current_manifest(self) -> VectorGenerationManifest:
        pointer_path = self.root / "current.json"
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            generation_id = str(pointer["generation_id"])
            expected_hash = str(pointer["manifest_sha256"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError("Qdrant local current pointer is invalid") from exc
        manifest_path = self.root / "generations" / f"{generation_id}.manifest.json"
        database_path = self.root / "generations" / generation_id
        if not manifest_path.is_file() or not database_path.is_dir():
            raise BackendUnavailableError("Qdrant local promoted generation is incomplete")
        try:
            raw = manifest_path.read_text(encoding="utf-8")
            if sha256_text(raw) != expected_hash:
                raise GenerationMismatchError("Qdrant local manifest pointer hash mismatch")
            manifest = VectorGenerationManifest(**json.loads(raw))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError("Qdrant local manifest is invalid") from exc
        if manifest.generation_id != generation_id or manifest.store_type != "qdrant-local":
            raise GenerationMismatchError("Qdrant local generation identity mismatch")
        return manifest

    @staticmethod
    def _filter(models: Any, filters: RetrievalFilter) -> Any:
        must: list[Any] = []
        if filters.required_source_ids:
            # All existing Local generations already contain the immutable
            # document payload; this path needs no payload/index migration.
            must.append(
                models.FieldCondition(
                    key="document.source_id",
                    match=models.MatchAny(any=list(filters.required_source_ids)),
                )
            )
        if filters.retrievable_only:
            must.append(
                models.FieldCondition(
                    key="retrievable",
                    match=models.MatchValue(value=True),
                )
            )
        must.append(
            models.FieldCondition(
                key="review_status",
                match=models.MatchAny(any=list(filters.allowed_review_statuses)),
            )
        )
        if filters.jurisdictions:
            must.append(
                models.FieldCondition(
                    key="jurisdiction",
                    match=models.MatchAny(any=list(filters.jurisdictions)),
                )
            )
        if filters.topics_any:
            must.append(
                models.FieldCondition(
                    key="topics",
                    match=models.MatchAny(any=list(filters.topics_any)),
                )
            )
        if filters.required_claim_scopes_any:
            must.append(
                models.FieldCondition(
                    key="allowed_claim_scopes",
                    match=models.MatchAny(any=list(filters.required_claim_scopes_any)),
                )
            )
        if filters.published_on_or_after or filters.published_on_or_before:
            date_range: dict[str, str] = {}
            if filters.published_on_or_after:
                date_range["gte"] = f"{filters.published_on_or_after}T00:00:00Z"
            if filters.published_on_or_before:
                date_range["lte"] = f"{filters.published_on_or_before}T23:59:59Z"
            must.append(
                models.FieldCondition(
                    key="publication_date",
                    datetime_range=models.DatetimeRange(**date_range),
                )
            )
        return models.Filter(must=must)

    def build_generation(
        self,
        documents: Iterable[RetrievalDocument],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_provenance: EmbeddingProvenance,
    ) -> VectorGenerationManifest:
        with self._operation_lock:
            self._ensure_open()
            return self._build_generation(
                documents, vectors, embedding_provenance=embedding_provenance
            )

    def _build_generation(
        self,
        documents: Iterable[RetrievalDocument],
        vectors: Sequence[Sequence[float]],
        *,
        embedding_provenance: EmbeddingProvenance,
    ) -> VectorGenerationManifest:
        supplied = tuple(documents)
        if not supplied or len(supplied) != len(vectors):
            raise ContractError("documents and vectors must have the same non-zero length")
        paired = sorted(zip(supplied, vectors, strict=True), key=lambda item: item[0].chunk_id)
        ordered = tuple(document for document, _ in paired)
        if len({item.chunk_id for item in ordered}) != len(ordered):
            raise ContractError("Qdrant generation contains duplicate chunk_id values")
        converted = tuple(
            tuple(float(value) for value in vector) for _, vector in paired
        )
        if any(len(vector) != embedding_provenance.dimensions for vector in converted):
            raise ContractError("Qdrant vector dimensions do not match embedding provenance")
        if any(not math.isfinite(value) for vector in converted for value in vector):
            raise ContractError("Qdrant vectors must contain only finite values")
        _client_class, _models, store_version = self._load_qdrant()
        corpus_digest = corpus_sha256(ordered)
        vectors_sha256 = sha256_text(canonical_json(converted))
        generation_id = sha256_text(
            canonical_json(
                {
                    "schema_version": self.STORE_SCHEMA_VERSION,
                    "corpus_sha256": corpus_digest,
                    "embedding_fingerprint": embedding_provenance.fingerprint,
                    "vectors_sha256": vectors_sha256,
                    "store_version": store_version,
                }
            )
        )[:32]
        manifest = VectorGenerationManifest(
            schema_version=1,
            store_type="qdrant-local",
            store_version=store_version,
            generation_id=generation_id,
            corpus_sha256=corpus_digest,
            embedding_fingerprint=embedding_provenance.fingerprint,
            embedding_provenance=asdict(embedding_provenance),
            dimensions=embedding_provenance.dimensions,
            vector_count=len(ordered),
            database_sha256=vectors_sha256,
        )
        generation_root = self.root / "generations"
        generation_root.mkdir(parents=True, exist_ok=True)
        final_database = generation_root / generation_id
        final_manifest = generation_root / f"{generation_id}.manifest.json"
        manifest_text = canonical_json(asdict(manifest))
        if final_database.exists() or final_manifest.exists():
            if not final_database.is_dir() or not final_manifest.is_file():
                raise GenerationMismatchError("existing Qdrant generation is incomplete")
            if final_manifest.read_text(encoding="utf-8") != manifest_text:
                raise GenerationMismatchError("Qdrant generation identity collision")
        else:
            nonce = uuid.uuid4().hex
            temporary_database = generation_root / f".{generation_id}.{nonce}.tmp"
            client: Any | None = None
            try:
                client_class, models, _version = self._load_qdrant()
                client = client_class(path=str(temporary_database))
                client.create_collection(
                    collection_name=self.config.collection,
                    vectors_config=models.VectorParams(
                        size=embedding_provenance.dimensions,
                        distance=models.Distance.COSINE,
                    ),
                )
                points = []
                for document, vector in zip(ordered, converted, strict=True):
                    payload = {
                        "chunk_id": document.chunk_id,
                        "review_status": document.review_status,
                        "retrievable": document.retrievable,
                        "jurisdiction": document.jurisdiction,
                        "publication_date": document.publication_date,
                        "topics": list(document.topics),
                        "allowed_claim_scopes": list(document.allowed_claim_scopes),
                        "document": document.to_dict(),
                    }
                    points.append(
                        models.PointStruct(
                            id=str(uuid.uuid5(uuid.NAMESPACE_URL, document.chunk_id)),
                            vector=list(vector),
                            payload=payload,
                        )
                    )
                for start in range(0, len(points), 256):
                    client.upsert(
                        collection_name=self.config.collection,
                        points=points[start : start + 256],
                        wait=True,
                    )
                close = getattr(client, "close", None)
                if callable(close):
                    close()
                client = None
                os.replace(temporary_database, final_database)
            finally:
                if client is not None:
                    close = getattr(client, "close", None)
                    if callable(close):
                        close()
                if temporary_database.exists():
                    shutil.rmtree(temporary_database)
            temporary_manifest = generation_root / (
                f".{generation_id}.{uuid.uuid4().hex}.tmp.manifest.json"
            )
            try:
                temporary_manifest.write_text(manifest_text, encoding="utf-8")
                os.replace(temporary_manifest, final_manifest)
            finally:
                temporary_manifest.unlink(missing_ok=True)
        pointer = canonical_json(
            {
                "schema_version": 1,
                "generation_id": generation_id,
                "manifest_sha256": sha256_text(manifest_text),
            }
        )
        temporary_pointer = self.root / f".current.{uuid.uuid4().hex}.tmp.json"
        temporary_pointer.write_text(pointer, encoding="utf-8")
        os.replace(temporary_pointer, self.root / "current.json")
        self._manifest = manifest
        return manifest

    def search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilter,
        limit: int,
    ) -> tuple[RankedItem, ...]:
        with self._operation_lock:
            self._ensure_open()
            return self._search(vector, filters=filters, limit=limit)

    def _search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilter,
        limit: int,
    ) -> tuple[RankedItem, ...]:
        manifest = self.manifest
        values = tuple(float(value) for value in vector)
        if len(values) != manifest.dimensions or any(not math.isfinite(value) for value in values):
            raise ContractError("query vector is invalid for Qdrant local generation")
        if limit < 1:
            raise ContractError("Qdrant local search limit must be positive")
        client_class, models, _version = self._load_qdrant()
        database = self.root / "generations" / manifest.generation_id
        client: Any | None = None
        try:
            client = client_class(path=str(database))
            response = client.query_points(
                collection_name=self.config.collection,
                query=list(values),
                query_filter=self._filter(models, filters),
                limit=min(manifest.vector_count, max(limit * 4, limit)),
                with_payload=True,
                with_vectors=False,
            )
            scored: list[tuple[str, float]] = []
            for point in response.points:
                payload = point.payload or {}
                document = RetrievalDocument(**payload["document"])
                if filters.admits(document):
                    scored.append((document.chunk_id, float(point.score)))
        except Exception as exc:  # noqa: BLE001 - optional client error boundary
            raise BackendUnavailableError("Qdrant local query failed validation") from exc
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        if any(not math.isfinite(score) for _, score in scored):
            raise BackendUnavailableError("Qdrant local returned a non-finite score")
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            RankedItem(chunk_id=chunk_id, score=score, backend="dense", rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        )


@dataclass(frozen=True, slots=True)
class QdrantServerConfig:
    base_url: str
    collection: str
    dimensions: int
    generation_id: str
    corpus_sha256: str
    embedding_fingerprint: str
    api_key_env: str | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ContractError("Qdrant base_url must target an http(s) server")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ContractError("Qdrant base_url cannot contain credentials, query, or fragment")
        if any(token in self.base_url.lower() for token in (":memory:", "file://", "path=")):
            raise ContractError("Qdrant local/in-memory mode is prohibited")
        if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ContractError("non-loopback Qdrant servers require HTTPS")
        if not _COLLECTION_RE.fullmatch(self.collection):
            raise ContractError("invalid Qdrant collection name")
        if self.dimensions < 1 or not 0 < self.timeout_seconds <= 120:
            raise ContractError("invalid Qdrant dimensions or timeout")


class QdrantServerVectorStore:
    """REST boundary for Qdrant server. It deliberately has no local-mode constructor."""

    def __init__(self, config: QdrantServerConfig) -> None:
        self.config = config

    @property
    def manifest(self) -> VectorGenerationManifest:
        return VectorGenerationManifest(
            schema_version=1,
            store_type="qdrant-server",
            store_version="server-managed",
            generation_id=self.config.generation_id,
            corpus_sha256=self.config.corpus_sha256,
            embedding_fingerprint=self.config.embedding_fingerprint,
            embedding_provenance={},
            dimensions=self.config.dimensions,
            vector_count=-1,
            database_sha256="server-managed",
        )

    @staticmethod
    def _filter_payload(filters: RetrievalFilter) -> dict[str, Any]:
        must: list[dict[str, Any]] = []
        if filters.required_source_ids:
            must.append(
                {"key": "source_id", "match": {"any": list(filters.required_source_ids)}}
            )
        if filters.retrievable_only:
            must.append({"key": "retrievable", "match": {"value": True}})
        must.append(
            {"key": "review_status", "match": {"any": list(filters.allowed_review_statuses)}}
        )
        if filters.jurisdictions:
            must.append({"key": "jurisdiction", "match": {"any": list(filters.jurisdictions)}})
        if filters.published_on_or_after or filters.published_on_or_before:
            date_range: dict[str, str] = {}
            if filters.published_on_or_after:
                date_range["gte"] = filters.published_on_or_after
            if filters.published_on_or_before:
                date_range["lte"] = filters.published_on_or_before
            if "gte" in date_range:
                date_range["gte"] = f"{date_range['gte']}T00:00:00Z"
            if "lte" in date_range:
                date_range["lte"] = f"{date_range['lte']}T23:59:59Z"
            must.append({"key": "publication_date", "datetime_range": date_range})
        if filters.topics_any:
            must.append(
                {
                    "should": [
                        {"key": "topics", "match": {"value": topic}} for topic in filters.topics_any
                    ]
                }
            )
        if filters.required_claim_scopes_any:
            must.append(
                {
                    "should": [
                        {"key": "allowed_claim_scopes", "match": {"value": scope}}
                        for scope in filters.required_claim_scopes_any
                    ]
                }
            )
        return {"must": must}

    def search(
        self,
        vector: Sequence[float],
        *,
        filters: RetrievalFilter,
        limit: int,
    ) -> tuple[RankedItem, ...]:
        values = [float(value) for value in vector]
        if len(values) != self.config.dimensions or any(
            not math.isfinite(value) for value in values
        ):
            raise ContractError("query vector is invalid for Qdrant collection")
        if limit < 1:
            raise ContractError("vector search limit must be positive")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise BackendUnavailableError(
                    "Qdrant API credential environment variable is unset: "
                    f"{self.config.api_key_env}"
                )
            headers["api-key"] = api_key
        endpoint = (
            f"{self.config.base_url.rstrip('/')}/collections/"
            f"{quote(self.config.collection, safe='')}/points/query"
        )
        request = urllib.request.Request(
            endpoint,
            data=canonical_json(
                {
                    "query": values,
                    "filter": self._filter_payload(filters),
                    "limit": limit,
                    "with_payload": ["chunk_id"],
                    "with_vector": False,
                }
            ).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:  # noqa: S310
                raw_response = response.read(8 * 1024 * 1024 + 1)
            if len(raw_response) > 8 * 1024 * 1024:
                raise BackendUnavailableError("Qdrant response exceeds 8 MiB")
            payload = json.loads(raw_response)
            raw_result = payload["result"]
            points = (
                raw_result.get("points", raw_result) if isinstance(raw_result, dict) else raw_result
            )
            scored = [(str(item["payload"]["chunk_id"]), float(item["score"])) for item in points]
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise BackendUnavailableError("Qdrant server query failed validation") from exc
        if any(not math.isfinite(score) for _, score in scored):
            raise BackendUnavailableError("Qdrant server returned a non-finite score")
        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))[:limit]
        return tuple(
            RankedItem(chunk_id=chunk_id, score=score, backend="dense", rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        )
