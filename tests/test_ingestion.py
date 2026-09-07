from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbx_agent.ingestion import (
    ChunkingOptions,
    ConfigurationError,
    ExtractionOptions,
    OptionalDependencyError,
    ScannedPdfError,
    SourceSpec,
    canonicalize_source,
    chunk_document,
    extractors,
    load_config,
    run_ingestion,
)
from tbx_agent.retrieval import (
    BM25Index,
    RetrievalFilter,
    reviewed_document_from_ingested_chunk,
)


def _source(path: Path, input_format: str = "markdown") -> SourceSpec:
    return SourceSpec(
        source_id="test_guideline_2026",
        input_path=path,
        input_format=input_format,  # type: ignore[arg-type]
        title="测试指南",
        organization="测试学会",
        publication_year=2026,
        jurisdiction="China",
        url="https://example.invalid/guideline",
        topics=("diagnosis", "screening"),
    )


def _write_config(tmp_path: Path, source_name: str = "guide.md") -> Path:
    config = tmp_path / "ingestion.yaml"
    config.write_text(
        f"""
schema_version: 1
pipeline_version: test-ingestion-v1
offline_only: true
output_dir: build
extraction:
  pdf_engine: auto
  min_pdf_page_characters: 12
  fail_on_suspected_scanned_page: true
  require_pdf_table_detection: false
chunking:
  min_characters: 30
  target_characters: 100
  max_characters: 180
  overlap_blocks: 1
sources:
  - source_id: test_guideline_2026
    input_path: {source_name}
    format: markdown
    title: 测试指南
    organization: 测试学会
    publication_year: 2026
    jurisdiction: China
    url: https://example.invalid/guideline
    topics: [diagnosis, screening]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def test_markdown_canonicalization_preserves_structure_and_page_metadata(tmp_path: Path):
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        """# 测试指南

<!-- tbx:page=3 -->

## 检查路径

这是第一段。仅用于测试结构。

- 痰标本检测
- 影像辅助检查

| 检查 | 用途 |
| --- | --- |
| 核酸 | 病原学依据 |
""",
        encoding="utf-8",
    )

    document = canonicalize_source(_source(source_path))

    assert document.extractor == "utf8-markdown"
    assert document.canonical_markdown.startswith("# 测试指南")
    assert {block.kind for block in document.blocks} >= {
        "heading",
        "paragraph",
        "list",
        "table",
    }
    table = next(block for block in document.blocks if block.kind == "table")
    assert table.page_start == 3
    assert table.heading_path == ("测试指南", "检查路径")
    assert "| 核酸 | 病原学依据 |" in table.markdown


def test_markdown_setext_heading_is_normalized_in_structural_blocks(tmp_path: Path):
    source_path = tmp_path / "setext.md"
    source_path.write_text(
        "测试指南\n========\n\n检查路径\n--------\n\n这是正文。\n",
        encoding="utf-8",
    )

    document = canonicalize_source(_source(source_path))
    headings = [block for block in document.blocks if block.kind == "heading"]

    assert [block.heading_level for block in headings] == [1, 2]
    assert headings[1].heading_path == ("测试指南", "检查路径")
    assert headings[1].markdown == "## 检查路径"


def test_section_first_chunking_is_bounded_review_gated_and_stable(tmp_path: Path):
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# 测试指南\n\n## 第一节\n\n"
        + "第一段用于解释检查路径。" * 12
        + "\n\n第二段用于解释转诊路径。" * 9
        + "\n\n## 第二节\n\n结尾内容。\n",
        encoding="utf-8",
    )
    document = canonicalize_source(_source(source_path))
    options = ChunkingOptions(
        min_characters=40,
        target_characters=120,
        max_characters=180,
        overlap_blocks=1,
    )

    first = chunk_document(document, options)
    second = chunk_document(document, options)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
    assert all(len(item.text) <= options.max_characters for item in first)
    assert all(item.retrievable is False for item in first)
    assert all(item.review_status == "pending_medical_review" for item in first)
    assert any(item.section_path[-1:] == ("第一节",) for item in first)
    assert any(item.section_path[-1:] == ("第二节",) for item in first)
    assert all(len(item.content_sha256) == 64 for item in first)
    assert all(
        "第一段" in item.text or "第二段" in item.text or "结尾内容" in item.text for item in first
    )
    assert all("## 第一节" in item.text for item in first if item.section_path[-1:] == ("第一节",))


def test_pipeline_rerun_is_content_addressed_and_emits_manifest_and_receipt(tmp_path: Path):
    source_path = tmp_path / "guide.md"
    source_path.write_text(
        "# 测试指南\n\n## 检查\n\n胸部影像只提供辅助信息。\n\n"
        "## 进一步检查\n\n需要结合病原学检查。\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path)

    first = run_ingestion(config_path)
    first_manifest = first.manifest_path.read_bytes()
    first_chunks = first.chunks_path.read_bytes()
    second = run_ingestion(config_path)

    assert second.run_id != first.run_id
    assert second.build_id == first.build_id
    assert second.receipt_path != first.receipt_path
    assert second.manifest_path == first.manifest_path
    assert second.manifest_path.read_bytes() == first_manifest
    assert second.chunks_path.read_bytes() == first_chunks
    manifest = json.loads(first_manifest)
    receipt = json.loads(second.receipt_path.read_text(encoding="utf-8"))
    chunks = [json.loads(line) for line in first_chunks.decode().splitlines()]
    assert manifest["offline_only"] is True
    assert manifest["deployment_gate"]["eligible"] is False
    assert manifest["deployment_gate"]["automatic_promotion"] is False
    assert receipt["status"] == "succeeded"
    assert receipt["build_id"] == first.build_id
    assert receipt["chunks_sha256"] == manifest["artifacts"]["chunks_sha256"]
    assert all(chunk["retrievable"] is False for chunk in chunks)
    assert all(Path(path).is_file() for path in second.canonical_paths)


def test_source_change_creates_new_snapshot_and_content_hash(tmp_path: Path):
    source_path = tmp_path / "guide.md"
    source_path.write_text("# 测试指南\n\n第一版内容用于测试。\n", encoding="utf-8")
    config_path = _write_config(tmp_path)
    first = run_ingestion(config_path)
    first_chunks = first.chunks_path.read_text(encoding="utf-8")

    source_path.write_text("# 测试指南\n\n第二版内容发生了变化。\n", encoding="utf-8")
    second = run_ingestion(config_path)

    assert second.run_id != first.run_id
    assert second.build_id != first.build_id
    assert second.manifest_path != first.manifest_path
    assert second.chunks_path.read_text(encoding="utf-8") != first_chunks


def test_ingested_chunk_stays_quarantined_until_explicit_review_promotion(tmp_path: Path):
    source_path = tmp_path / "guide.md"
    source_path.write_text("# 测试指南\n\n## 检查\n\n病原学检查证据。\n", encoding="utf-8")
    chunk = chunk_document(canonicalize_source(_source(source_path)))[0]

    pending = reviewed_document_from_ingested_chunk(
        chunk,
        allowed_claim_scopes=("diagnostic_support",),
    )
    approved = reviewed_document_from_ingested_chunk(
        chunk,
        allowed_claim_scopes=("diagnostic_support",),
        review_status="approved",
        retrievable=True,
    )

    assert pending.review_status == "pending_medical_review"
    assert pending.retrievable is False
    assert BM25Index((pending,)).search("病原学", filters=RetrievalFilter()) == ()
    assert [
        item.chunk_id
        for item in BM25Index((approved,)).search("病原学", filters=RetrievalFilter())
    ] == [approved.chunk_id]


def test_config_rejects_remote_input_before_any_fetch(tmp_path: Path):
    config = _write_config(tmp_path, "https://example.invalid/guide.pdf")

    with pytest.raises(ConfigurationError, match="network fetching is disabled"):
        load_config(config)


def test_html_missing_optional_dependency_has_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = tmp_path / "guide.html"
    source_path.write_text("<h1>测试指南</h1><p>正文</p>", encoding="utf-8")
    real_import = extractors.importlib.import_module

    def unavailable(name: str):
        if name in {"bs4", "markdownify"}:
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(extractors.importlib, "import_module", unavailable)

    with pytest.raises(OptionalDependencyError, match="beautifulsoup4 and markdownify"):
        canonicalize_source(_source(source_path, "html"))


class _ImageOnlyPage:
    images = [object()]

    @staticmethod
    def extract_text() -> str:
        return ""


class _FakeReader:
    is_encrypted = False
    pages = [_ImageOnlyPage()]

    def __init__(self, _path: Path):
        pass


def _fake_pdf_import(name: str):
    if name == "pypdf":
        return SimpleNamespace(PdfReader=_FakeReader)
    raise ModuleNotFoundError(name)


def test_scanned_pdf_fails_closed_without_explicit_ocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = tmp_path / "scan.pdf"
    source_path.write_bytes(b"%PDF-fake-image-only")
    monkeypatch.setattr(extractors.importlib, "import_module", _fake_pdf_import)

    with pytest.raises(ScannedPdfError, match="appears image-only"):
        canonicalize_source(
            _source(source_path, "pdf"),
            ExtractionOptions(pdf_engine="pypdf", min_pdf_page_characters=10),
        )


def test_scanned_pdf_uses_only_explicit_nonempty_ocr_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source_path = tmp_path / "scan.pdf"
    source_path.write_bytes(b"%PDF-fake-image-only")
    monkeypatch.setattr(extractors.importlib, "import_module", _fake_pdf_import)

    class LocalOcr:
        name = "test-local-ocr"

        @staticmethod
        def extract_page(_pdf_path: Path, page_number: int) -> str:
            assert page_number == 1
            return (
                "这是由显式本地OCR适配器提取并经过长度校验的可观察文本。"
                "第二句话用于确认整页存在足够的可验证文字，系统没有补写医学内容。"
            )

    document = canonicalize_source(
        _source(source_path, "pdf"),
        ExtractionOptions(pdf_engine="pypdf", min_pdf_page_characters=10),
        ocr_adapter=LocalOcr(),
    )

    assert document.ocr_used is True
    assert document.ocr_pages == (1,)
    assert "可观察文本" in document.canonical_markdown
    assert all("fabricat" not in block.markdown.lower() for block in document.blocks)


def test_two_column_layout_reads_complete_left_column_before_right_column():
    words = [
        {"text": "full-width-header", "x0": 40, "x1": 550, "top": 10, "bottom": 20},
    ]
    for index in range(6):
        top = 40 + index * 12
        words.extend(
            [
                {
                    "text": f"left-{index}",
                    "x0": 45,
                    "x1": 250,
                    "top": top,
                    "bottom": top + 8,
                },
                {
                    "text": f"right-{index}",
                    "x0": 330,
                    "x1": 545,
                    "top": top,
                    "bottom": top + 8,
                },
            ]
        )

    ordered, applied = extractors._column_ordered_lines(  # noqa: SLF001
        words, page_width=595, layout_mode="two_column"
    )
    text = [item[1] for item in ordered]

    assert applied is True
    assert text[0] == "full-width-header"
    assert text[1:7] == [f"left-{index}" for index in range(6)]
    assert text[7:] == [f"right-{index}" for index in range(6)]


def test_failed_run_still_writes_a_failure_receipt(tmp_path: Path):
    config_path = _write_config(tmp_path, "missing.md")
    loaded = load_config(config_path)

    with pytest.raises(Exception, match="does not exist"):
        run_ingestion(loaded)

    receipts = list((tmp_path / "build" / "receipts").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error_type"] == "ExtractionError"
    assert receipt["manifest_sha256"] is None
