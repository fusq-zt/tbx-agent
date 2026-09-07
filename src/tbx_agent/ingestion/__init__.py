"""Offline, deterministic guideline ingestion for review-gated TBX knowledge builds."""

from .chunking import chunk_document
from .errors import (
    ConfigurationError,
    ExtractionError,
    IngestionError,
    IntegrityError,
    OptionalDependencyError,
    ScannedPdfError,
)
from .extractors import canonicalize_source
from .models import (
    CanonicalBlock,
    CanonicalDocument,
    ChunkingOptions,
    ChunkRecord,
    ExtractionOptions,
    IngestionConfig,
    IngestionResult,
    OcrAdapter,
    SourceSpec,
)
from .pipeline import load_config, run_ingestion

__all__ = [
    "CanonicalBlock",
    "CanonicalDocument",
    "ChunkRecord",
    "ChunkingOptions",
    "ConfigurationError",
    "ExtractionError",
    "ExtractionOptions",
    "IngestionConfig",
    "IngestionError",
    "IngestionResult",
    "IntegrityError",
    "OcrAdapter",
    "OptionalDependencyError",
    "ScannedPdfError",
    "SourceSpec",
    "canonicalize_source",
    "chunk_document",
    "load_config",
    "run_ingestion",
]
