from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from ..paths import resolve_portable_path
from .chunking import chunk_document
from .errors import ConfigurationError, IngestionError, IntegrityError
from .extractors import canonicalize_source
from .models import (
    ChunkingOptions,
    ExtractionOptions,
    IngestionConfig,
    IngestionResult,
    MutableReceipt,
    OcrAdapter,
    SourceReceipt,
    SourceSpec,
)

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,79}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_json(value: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown keys in {field}: {', '.join(unknown)}")


def _required_text(mapping: dict[str, Any], key: str, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field}.{key} must be a non-empty string")
    return value.strip()


def _local_path(raw: str, base: Path, field: str) -> Path:
    parsed = urlparse(raw)
    if parsed.scheme.lower() in {"http", "https", "ftp", "s3", "gs"} or raw.startswith("\\\\"):
        raise ConfigurationError(f"{field} must be a local path; network fetching is disabled")
    try:
        return resolve_portable_path(raw, base=base)
    except ValueError as exc:
        raise ConfigurationError(f"{field} contains an invalid runtime reference") from exc


def _parse_source(item: Any, base: Path, index: int) -> SourceSpec:
    field = f"sources[{index}]"
    value = _mapping(item, field)
    _reject_unknown(
        value,
        {
            "source_id",
            "input_path",
            "format",
            "title",
            "organization",
            "publication_year",
            "jurisdiction",
            "url",
            "topics",
            "language",
            "enabled",
            "expected_sha256",
        },
        field,
    )
    source_id = _required_text(value, "source_id", field)
    if not _SOURCE_ID_RE.fullmatch(source_id):
        raise ConfigurationError(
            f"{field}.source_id must be 3-80 lowercase ASCII letters, digits, "
            "dot, dash or underscore"
        )
    input_format = _required_text(value, "format", field).lower()
    if input_format == "md":
        input_format = "markdown"
    if input_format not in {"markdown", "html", "pdf"}:
        raise ConfigurationError(f"{field}.format must be markdown, html or pdf")
    raw_path = _required_text(value, "input_path", field)
    year = value.get("publication_year")
    if not isinstance(year, int) or not 1900 <= year <= 2100:
        raise ConfigurationError(f"{field}.publication_year must be an integer from 1900 to 2100")
    topics = value.get("topics", [])
    if not isinstance(topics, list) or not all(isinstance(item, str) and item for item in topics):
        raise ConfigurationError(f"{field}.topics must be a list of non-empty strings")
    expected = value.get("expected_sha256")
    if expected is not None and (
        not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected)
    ):
        raise ConfigurationError(f"{field}.expected_sha256 must be a 64-character hex digest")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigurationError(f"{field}.enabled must be boolean")
    language = value.get("language", "zh-CN")
    if not isinstance(language, str) or not language.strip():
        raise ConfigurationError(f"{field}.language must be a non-empty string")
    return SourceSpec(
        source_id=source_id,
        input_path=_local_path(raw_path, base, f"{field}.input_path"),
        input_format=input_format,  # type: ignore[arg-type]
        title=_required_text(value, "title", field),
        organization=_required_text(value, "organization", field),
        publication_year=year,
        jurisdiction=_required_text(value, "jurisdiction", field),
        url=_required_text(value, "url", field),
        topics=tuple(dict.fromkeys(topics)),
        language=language.strip(),
        enabled=enabled,
        expected_sha256=expected.lower() if expected else None,
    )


def load_config(path: Path) -> IngestionConfig:
    """Load a strict YAML config. Source contents are never fetched from URLs."""

    config_path = path.resolve()
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read ingestion config: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in ingestion config: {exc}") from exc
    root = _mapping(raw, "config")
    _reject_unknown(
        root,
        {
            "schema_version",
            "pipeline_version",
            "offline_only",
            "output_dir",
            "extraction",
            "chunking",
            "sources",
        },
        "config",
    )
    if root.get("schema_version") != 1:
        raise ConfigurationError("schema_version must equal 1")
    if root.get("offline_only", True) is not True:
        raise ConfigurationError("offline_only must be true; this pipeline never fetches URLs")
    base = config_path.parent
    output_dir = _local_path(_required_text(root, "output_dir", "config"), base, "output_dir")
    extraction_raw = _mapping(root.get("extraction", {}), "extraction")
    _reject_unknown(
        extraction_raw,
        {
            "pdf_engine",
            "pdf_layout_mode",
            "min_pdf_page_characters",
            "fail_on_suspected_scanned_page",
            "require_pdf_table_detection",
            "html_remove_selectors",
        },
        "extraction",
    )
    selectors = extraction_raw.get(
        "html_remove_selectors", list(ExtractionOptions().html_remove_selectors)
    )
    if not isinstance(selectors, list) or not all(isinstance(item, str) for item in selectors):
        raise ConfigurationError("extraction.html_remove_selectors must be a string list")
    extraction = ExtractionOptions(
        pdf_engine=extraction_raw.get("pdf_engine", "auto"),
        pdf_layout_mode=extraction_raw.get("pdf_layout_mode", "auto"),
        min_pdf_page_characters=extraction_raw.get("min_pdf_page_characters", 24),
        fail_on_suspected_scanned_page=extraction_raw.get("fail_on_suspected_scanned_page", True),
        require_pdf_table_detection=extraction_raw.get("require_pdf_table_detection", False),
        html_remove_selectors=tuple(selectors),
    )
    if extraction.pdf_engine not in {"auto", "pdfplumber", "pypdf"}:
        raise ConfigurationError("extraction.pdf_engine must be auto, pdfplumber or pypdf")
    if extraction.pdf_layout_mode not in {"auto", "single_column", "two_column"}:
        raise ConfigurationError(
            "extraction.pdf_layout_mode must be auto, single_column or two_column"
        )
    if not isinstance(extraction.min_pdf_page_characters, int) or not (
        1 <= extraction.min_pdf_page_characters <= 1000
    ):
        raise ConfigurationError("extraction.min_pdf_page_characters must be 1-1000")
    if not isinstance(extraction.fail_on_suspected_scanned_page, bool) or not isinstance(
        extraction.require_pdf_table_detection, bool
    ):
        raise ConfigurationError("PDF strictness settings must be boolean")
    if extraction.fail_on_suspected_scanned_page is not True:
        raise ConfigurationError(
            "extraction.fail_on_suspected_scanned_page must be true; partial scanned-page "
            "ingestion is not permitted"
        )
    chunking_raw = _mapping(root.get("chunking", {}), "chunking")
    _reject_unknown(
        chunking_raw,
        {"min_characters", "target_characters", "max_characters", "overlap_blocks"},
        "chunking",
    )
    chunking = ChunkingOptions(**chunking_raw)
    if not (
        isinstance(chunking.min_characters, int)
        and isinstance(chunking.target_characters, int)
        and isinstance(chunking.max_characters, int)
        and 1 <= chunking.min_characters <= chunking.target_characters <= chunking.max_characters
    ):
        raise ConfigurationError(
            "chunk limits must satisfy 1 <= min_characters <= target_characters <= max_characters"
        )
    if not isinstance(chunking.overlap_blocks, int) or not 0 <= chunking.overlap_blocks <= 4:
        raise ConfigurationError("chunking.overlap_blocks must be an integer from 0 to 4")
    sources_raw = root.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ConfigurationError("sources must be a non-empty list")
    sources = tuple(_parse_source(item, base, index) for index, item in enumerate(sources_raw))
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise ConfigurationError("source_id values must be unique")
    if not any(source.enabled for source in sources):
        raise ConfigurationError("at least one source must be enabled")
    pipeline_version = root.get("pipeline_version", "tbx-knowledge-ingestion-v1")
    if not isinstance(pipeline_version, str) or not pipeline_version.strip():
        raise ConfigurationError("pipeline_version must be a non-empty string")
    return IngestionConfig(
        config_path=config_path,
        output_dir=output_dir,
        sources=sources,
        extraction=extraction,
        chunking=chunking,
        offline_only=True,
        pipeline_version=pipeline_version.strip(),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _receipt_source_path(source: SourceSpec, config_path: Path) -> str:
    try:
        return source.input_path.relative_to(config_path.parent).as_posix()
    except ValueError:
        return source.input_path.name


def _failure_receipt(output_dir: Path, receipt: MutableReceipt, exc: Exception) -> Path:
    receipt.status = "failed"
    receipt.finished_at = _utc_now()
    receipt.error_type = type(exc).__name__
    receipt.error_message = str(exc)[:1000]
    path = output_dir / "receipts" / f"{receipt.run_id}.json"
    _atomic_write(path, _stable_json(receipt.to_dict(), pretty=True).encode("utf-8"))
    return path


def run_ingestion(
    config: IngestionConfig | Path,
    *,
    ocr_adapter: OcrAdapter | None = None,
) -> IngestionResult:
    """Build one immutable, review-gated snapshot and an auditable run receipt."""

    if isinstance(config, Path):
        config = load_config(config)
    if config.offline_only is not True:
        raise ConfigurationError("offline_only must be true")
    if any(str(source.input_path).startswith("\\\\") for source in config.sources):
        raise ConfigurationError("UNC network sources are not permitted")
    config_bytes = config.config_path.read_bytes()
    config_sha256 = _sha256_bytes(config_bytes)
    enabled = tuple(source for source in config.sources if source.enabled)
    if not enabled:
        raise ConfigurationError("at least one source must be enabled")
    fingerprint_parts = [config.pipeline_version, config_sha256]
    for source in enabled:
        if source.input_path.is_file():
            fingerprint_parts.extend(
                (source.source_id, _sha256_bytes(source.input_path.read_bytes()))
            )
        else:
            fingerprint_parts.extend((source.source_id, "missing"))
    build_id = hashlib.sha256("\x1f".join(fingerprint_parts).encode("utf-8")).hexdigest()[:24]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex[:12]}"
    receipt = MutableReceipt(
        run_id=run_id,
        build_id=build_id,
        pipeline_version=config.pipeline_version,
        config_sha256=config_sha256,
        started_at=_utc_now(),
        offline_only=True,
    )
    try:
        documents = [
            canonicalize_source(source, config.extraction, ocr_adapter=ocr_adapter)
            for source in enabled
        ]
        chunk_sets = [chunk_document(document, config.chunking) for document in documents]
        chunks = [chunk for chunk_set in chunk_sets for chunk in chunk_set]
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise IntegrityError("generated chunk_id collision across sources")
        chunks_bytes = ("\n".join(_stable_json(chunk.to_dict()) for chunk in chunks) + "\n").encode(
            "utf-8"
        )
        chunks_sha256 = _sha256_bytes(chunks_bytes)
        snapshot_payload = "\x1f".join(
            [
                config.pipeline_version,
                config_sha256,
                chunks_sha256,
                *(document.canonical_markdown_sha256 for document in documents),
                *(
                    _stable_json(
                        {
                            "extractor": document.extractor,
                            "extractor_version": document.extractor_version,
                            "warnings": document.warnings,
                            "ocr_used": document.ocr_used,
                            "ocr_pages": document.ocr_pages,
                        }
                    )
                    for document in documents
                ),
            ]
        )
        snapshot_id = hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()[:24]
        snapshot_dir = config.output_dir / "snapshots" / snapshot_id
        canonical_paths: list[Path] = []
        source_manifest: list[dict[str, Any]] = []
        for document, source_chunks in zip(documents, chunk_sets, strict=True):
            canonical_path = snapshot_dir / "canonical" / f"{document.source.source_id}.md"
            canonical_paths.append(canonical_path)
            relative = canonical_path.relative_to(config.output_dir).as_posix()
            source_receipt = SourceReceipt(
                source_id=document.source.source_id,
                input_path=_receipt_source_path(document.source, config.config_path),
                input_format=document.source.input_format,
                source_sha256=document.source_sha256,
                canonical_markdown_sha256=document.canonical_markdown_sha256,
                canonical_relative_path=relative,
                chunk_count=len(source_chunks),
                extractor=document.extractor,
                extractor_version=document.extractor_version,
                warnings=document.warnings,
                ocr_used=document.ocr_used,
                ocr_pages=document.ocr_pages,
            )
            receipt.source_receipts.append(source_receipt)
            source_manifest.append(
                {
                    **asdict(source_receipt),
                    "title": document.source.title,
                    "organization": document.source.organization,
                    "publication_year": document.source.publication_year,
                    "jurisdiction": document.source.jurisdiction,
                    "url": document.source.url,
                    "topics": document.source.topics,
                    "language": document.source.language,
                    "review_status": "pending_medical_review",
                    "retrievable": False,
                }
            )
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "build_id": build_id,
            "pipeline_version": config.pipeline_version,
            "config_sha256": config_sha256,
            "offline_only": True,
            "deployment_gate": {
                "eligible": False,
                "reason": (
                    "raw extraction requires medical review, claim-scope annotation and approval"
                ),
                "automatic_promotion": False,
            },
            "sources": source_manifest,
            "artifacts": {
                "chunks": "ingested_chunks.jsonl",
                "chunks_sha256": chunks_sha256,
                "chunk_count": len(chunks),
            },
        }
        manifest_bytes = _stable_json(manifest, pretty=True).encode("utf-8")
        manifest_path = snapshot_dir / "manifest.json"
        chunks_path = snapshot_dir / "ingested_chunks.jsonl"
        for document, canonical_path in zip(documents, canonical_paths, strict=True):
            _atomic_write(canonical_path, document.canonical_markdown.encode("utf-8"))
        _atomic_write(chunks_path, chunks_bytes)
        _atomic_write(manifest_path, manifest_bytes)
        receipt.status = "succeeded"
        receipt.finished_at = _utc_now()
        receipt.manifest_sha256 = _sha256_bytes(manifest_bytes)
        receipt.chunks_sha256 = chunks_sha256
        receipt_path = config.output_dir / "receipts" / f"{run_id}.json"
        _atomic_write(receipt_path, _stable_json(receipt.to_dict(), pretty=True).encode("utf-8"))
        return IngestionResult(
            run_id=run_id,
            build_id=build_id,
            output_dir=config.output_dir,
            manifest_path=manifest_path,
            chunks_path=chunks_path,
            receipt_path=receipt_path,
            canonical_paths=tuple(canonical_paths),
            chunk_count=len(chunks),
        )
    except Exception as exc:
        _failure_receipt(config.output_dir, receipt, exc)
        if isinstance(exc, IngestionError):
            raise
        raise IngestionError(f"unexpected ingestion failure: {exc}") from exc
