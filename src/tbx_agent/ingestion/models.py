from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

BlockKind = Literal["heading", "paragraph", "list", "table", "code"]
InputFormat = Literal["markdown", "html", "pdf"]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    input_path: Path
    input_format: InputFormat
    title: str
    organization: str
    publication_year: int
    jurisdiction: str
    url: str
    topics: tuple[str, ...] = ()
    language: str = "zh-CN"
    enabled: bool = True
    expected_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    pdf_engine: Literal["auto", "pdfplumber", "pypdf"] = "auto"
    pdf_layout_mode: Literal["auto", "single_column", "two_column"] = "auto"
    min_pdf_page_characters: int = 24
    fail_on_suspected_scanned_page: bool = True
    require_pdf_table_detection: bool = False
    html_remove_selectors: tuple[str, ...] = (
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
    )


@dataclass(frozen=True, slots=True)
class ChunkingOptions:
    min_characters: int = 240
    target_characters: int = 700
    max_characters: int = 1100
    overlap_blocks: int = 1


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    config_path: Path
    output_dir: Path
    sources: tuple[SourceSpec, ...]
    extraction: ExtractionOptions = ExtractionOptions()
    chunking: ChunkingOptions = ChunkingOptions()
    offline_only: bool = True
    pipeline_version: str = "tbx-knowledge-ingestion-v1"


@dataclass(frozen=True, slots=True)
class CanonicalBlock:
    ordinal: int
    kind: BlockKind
    markdown: str
    heading_path: tuple[str, ...]
    page_start: int | None = None
    page_end: int | None = None
    heading_level: int | None = None


@dataclass(frozen=True, slots=True)
class CanonicalDocument:
    source: SourceSpec
    source_sha256: str
    canonical_markdown: str
    canonical_markdown_sha256: str
    blocks: tuple[CanonicalBlock, ...]
    extractor: str
    extractor_version: str | None
    warnings: tuple[str, ...] = ()
    ocr_used: bool = False
    ocr_pages: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    schema_version: int
    chunk_id: str
    source_id: str
    source_sha256: str
    canonical_markdown_sha256: str
    content_sha256: str
    title: str
    organization: str
    publication_year: int
    jurisdiction: str
    url: str
    language: str
    topics: tuple[str, ...]
    section_path: tuple[str, ...]
    block_types: tuple[BlockKind, ...]
    block_start: int
    block_end: int
    page_start: int | None
    page_end: int | None
    locator: str
    text: str
    review_status: Literal["pending_medical_review"] = "pending_medical_review"
    retrievable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceReceipt:
    source_id: str
    input_path: str
    input_format: InputFormat
    source_sha256: str
    canonical_markdown_sha256: str
    canonical_relative_path: str
    chunk_count: int
    extractor: str
    extractor_version: str | None
    warnings: tuple[str, ...]
    ocr_used: bool
    ocr_pages: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IngestionResult:
    run_id: str
    build_id: str
    output_dir: Path
    manifest_path: Path
    chunks_path: Path
    receipt_path: Path
    canonical_paths: tuple[Path, ...]
    chunk_count: int


class OcrAdapter(Protocol):
    """Explicit OCR boundary. Implementations must run locally and return only observed text."""

    @property
    def name(self) -> str: ...

    def extract_page(self, pdf_path: Path, page_number: int) -> str: ...


@dataclass(slots=True)
class MutableReceipt:
    run_id: str
    build_id: str
    pipeline_version: str
    config_sha256: str
    status: Literal["running", "succeeded", "failed"] = "running"
    started_at: str = ""
    finished_at: str | None = None
    offline_only: bool = True
    source_receipts: list[SourceReceipt] = field(default_factory=list)
    manifest_sha256: str | None = None
    chunks_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
