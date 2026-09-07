from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import RetrievalDocument, canonical_json, sha256_text
from .errors import ContractError


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    snapshot_id: str
    documents: tuple[RetrievalDocument, ...]
    sources: dict[str, dict[str, Any]]
    chunks: tuple[dict[str, Any], ...]
    source_manifest_sha256: str
    chunks_sha256: str


def reviewed_document_from_ingested_chunk(
    chunk: Any,
    *,
    allowed_claim_scopes: tuple[str, ...],
    review_status: Literal[
        "approved", "pending_medical_review", "rejected", "superseded"
    ] = "pending_medical_review",
    retrievable: bool = False,
) -> RetrievalDocument:
    """Bridge review-gated ingestion output into the unified retrieval contract.

    The defaults preserve the ingestion quarantine. Promotion therefore requires
    an explicit approved status, claim scopes, and ``retrievable=True`` from a
    separate review workflow.
    """

    def field(name: str) -> Any:
        if isinstance(chunk, dict):
            if name not in chunk:
                raise ContractError(f"ingested chunk lacks {name}")
            return chunk[name]
        try:
            return getattr(chunk, name)
        except AttributeError as exc:
            raise ContractError(f"ingested chunk lacks {name}") from exc

    text = str(field("text"))
    content_sha = str(field("content_sha256"))
    if sha256_text(text) != content_sha:
        raise ContractError("ingested chunk content hash mismatch")
    return RetrievalDocument(
        chunk_id=str(field("chunk_id")),
        source_id=str(field("source_id")),
        text=text,
        content_sha256=content_sha,
        source_sha256=str(field("source_sha256")),
        source_hash_kind="artifact_sha256",
        title=str(field("title")),
        locator=str(field("locator")),
        jurisdiction=str(field("jurisdiction")),
        publication_date=f"{int(field('publication_year')):04d}-01-01",
        topics=tuple(str(item) for item in field("topics")),
        allowed_claim_scopes=allowed_claim_scopes,
        review_status=review_status,
        retrievable=retrievable,
        language=str(field("language")),
        organization=str(field("organization")),
        url=str(field("url")),
    )


def load_curated_corpus(knowledge_dir: Path) -> CorpusSnapshot:
    """Load the reviewed knowledge snapshot into the unified chunk contract."""

    root = knowledge_dir.resolve()
    manifest_path = root / "source_manifest.json"
    chunks_path = root / "chunks.jsonl"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError("cannot load curated source manifest") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("sources"), list):
        raise ContractError("curated source manifest has an invalid schema")
    snapshot_id = str(manifest.get("snapshot_id", "")).strip()
    if not snapshot_id:
        raise ContractError("curated source manifest lacks snapshot_id")
    sources = {
        str(item["source_id"]): item
        for item in manifest["sources"]
        if isinstance(item, dict) and item.get("retrievable") is True
    }
    if not sources:
        raise ContractError("curated snapshot has no retrievable sources")

    raw_chunks: list[dict[str, Any]] = []
    documents: list[RetrievalDocument] = []
    seen: set[str] = set()
    try:
        lines = chunks_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError("cannot load curated chunks") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid curated chunk JSON at line {line_number}") from exc
        if not isinstance(chunk, dict):
            raise ContractError(f"curated chunk line {line_number} is not an object")
        source_id = str(chunk.get("source_id", ""))
        source = sources.get(source_id)
        if source is None:
            raise ContractError(f"chunk references non-retrievable source: {source_id}")
        chunk_id = str(chunk.get("chunk_id", ""))
        if not chunk_id or chunk_id in seen:
            raise ContractError("curated chunks contain an empty or duplicate chunk_id")
        seen.add(chunk_id)
        scopes = source.get("allowed_claim_scope")
        if not isinstance(scopes, list) or not scopes:
            raise ContractError(f"source lacks allowed_claim_scope: {source_id}")
        support_text = str(chunk.get("support_text", "")).strip()
        text = str(chunk.get("text", "")).strip()
        embedding_text = f"{text}\n{support_text}" if support_text else text
        source_file_sha = source.get("file_sha256")
        if isinstance(source_file_sha, str) and len(source_file_sha) == 64:
            source_sha = source_file_sha.casefold()
            source_hash_kind = "artifact_sha256"
        else:
            source_sha = sha256_text(canonical_json(source))
            source_hash_kind = "manifest_record_sha256"
        topics = tuple(dict.fromkeys([*source.get("topics", []), *chunk.get("topics", [])]))
        document = RetrievalDocument(
            chunk_id=chunk_id,
            source_id=source_id,
            text=embedding_text,
            content_sha256=sha256_text(embedding_text),
            source_sha256=source_sha,
            source_hash_kind=source_hash_kind,
            title=str(chunk["title"]),
            locator=str(chunk["locator"]),
            jurisdiction=str(chunk["jurisdiction"]),
            publication_date=f"{int(chunk['publication_year']):04d}-01-01",
            topics=topics,
            allowed_claim_scopes=tuple(str(item) for item in scopes),
            review_status="approved",
            retrievable=True,
            language=str(source.get("language", "zh-CN")),
            organization=str(chunk.get("organization", "")),
            url=str(chunk.get("url", "")),
        )
        raw_chunks.append(chunk)
        documents.append(document)
    if not documents:
        raise ContractError("curated snapshot contains no chunks")
    return CorpusSnapshot(
        snapshot_id=snapshot_id,
        documents=tuple(sorted(documents, key=lambda item: item.chunk_id)),
        sources=sources,
        chunks=tuple(raw_chunks),
        source_manifest_sha256=_file_sha256(manifest_path),
        chunks_sha256=_file_sha256(chunks_path),
    )
