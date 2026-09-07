from __future__ import annotations


class IngestionError(RuntimeError):
    """Base error for deterministic, offline knowledge ingestion."""


class ConfigurationError(IngestionError):
    """Raised when an ingestion configuration is unsafe or incomplete."""


class OptionalDependencyError(IngestionError):
    """Raised when a requested format needs an unavailable optional dependency."""


class ExtractionError(IngestionError):
    """Raised when source content cannot be extracted without guessing."""


class ScannedPdfError(ExtractionError):
    """Raised when a PDF page appears image-only and no explicit OCR adapter is configured."""


class IntegrityError(IngestionError):
    """Raised when deterministic output or source integrity validation fails."""
