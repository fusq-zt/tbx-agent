from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import ExtractionError, OptionalDependencyError, ScannedPdfError
from .models import (
    CanonicalBlock,
    CanonicalDocument,
    ExtractionOptions,
    OcrAdapter,
    SourceSpec,
)

_PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*tbx:page=(\d+)\s*-->\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_HEADING_RE = re.compile(r"^\s*(=+|-{3,})\s*$")
_LIST_RE = re.compile(r"^\s*(?:[-+*]|\d+[.)]|[（(]?[一二三四五六七八九十]+[）)、.])\s+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百0-9]+[章节篇部分]\s*\S+")
_NUMBERED_HEADING_RE = re.compile(
    r"^(?:[一二三四五六七八九十]+[、.]|[（(][一二三四五六七八九十]+[）)]|"
    r"\d+(?:\.\d+){0,3}[、.．]?\s+)\S+"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_markdown(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    blank_count = 0
    for line in lines:
        if line.strip():
            blank_count = 0
            normalized.append(line)
        else:
            blank_count += 1
            if blank_count <= 1:
                normalized.append("")
    return "\n".join(normalized).strip() + "\n"


def _ensure_title(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    first_index = next((index for index, line in enumerate(lines) if line.strip()), None)
    first_content = lines[first_index] if first_index is not None else ""
    match = _HEADING_RE.match(first_content)
    if match and match.group(2).strip() == title.strip():
        return markdown
    if (
        first_index is not None
        and first_index + 1 < len(lines)
        and first_content.strip() == title.strip()
        and _SETEXT_HEADING_RE.match(lines[first_index + 1])
    ):
        return markdown
    return f"# {title.strip()}\n\n{markdown.lstrip()}"


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and "|" in lines[index]
        and bool(_TABLE_SEPARATOR_RE.match(lines[index + 1]))
    )


def parse_canonical_markdown(markdown: str) -> tuple[CanonicalBlock, ...]:
    """Parse normalized Markdown into provenance-bearing structural blocks."""

    lines = _normalize_markdown(markdown).splitlines()
    blocks: list[CanonicalBlock] = []
    heading_stack: list[str] = []
    page: int | None = None
    index = 0

    def append_block(
        kind: str,
        content: Iterable[str],
        *,
        heading_level: int | None = None,
    ) -> None:
        nonlocal blocks
        text = "\n".join(content).strip()
        if not text:
            return
        blocks.append(
            CanonicalBlock(
                ordinal=len(blocks),
                kind=kind,  # type: ignore[arg-type]
                markdown=text,
                heading_path=tuple(heading_stack),
                page_start=page,
                page_end=page,
                heading_level=heading_level,
            )
        )

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        page_match = _PAGE_MARKER_RE.match(line)
        if page_match:
            page = int(page_match.group(1))
            index += 1
            continue
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_stack[:] = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("(untitled)")
            heading_stack.append(title)
            append_block("heading", [f"{'#' * level} {title}"], heading_level=level)
            index += 1
            continue
        if index + 1 < len(lines) and (setext_match := _SETEXT_HEADING_RE.match(lines[index + 1])):
            level = 1 if setext_match.group(1).startswith("=") else 2
            title = line.strip()
            heading_stack[:] = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("(untitled)")
            heading_stack.append(title)
            append_block("heading", [f"{'#' * level} {title}"], heading_level=level)
            index += 2
            continue
        if line.lstrip().startswith("```"):
            collected = [line]
            index += 1
            while index < len(lines):
                collected.append(lines[index])
                terminal = lines[index].lstrip().startswith("```")
                index += 1
                if terminal:
                    break
            append_block("code", collected)
            continue
        if _is_table_start(lines, index):
            collected = [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                collected.append(lines[index])
                index += 1
            append_block("table", collected)
            continue
        if _LIST_RE.match(line):
            collected = [line]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if not candidate.strip():
                    break
                if _LIST_RE.match(candidate) or candidate.startswith(("  ", "\t")):
                    collected.append(candidate)
                    index += 1
                    continue
                break
            append_block("list", collected)
            continue
        collected = [line]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if (
                not candidate.strip()
                or _PAGE_MARKER_RE.match(candidate)
                or _HEADING_RE.match(candidate)
                or _LIST_RE.match(candidate)
                or candidate.lstrip().startswith("```")
                or _is_table_start(lines, index)
            ):
                break
            collected.append(candidate)
            index += 1
        append_block("paragraph", collected)
    if not blocks:
        raise ExtractionError("canonical document contains no extractable blocks")
    return tuple(blocks)


def _plain_pdf_text_to_markdown(text: str, page_number: int) -> str:
    output = [f"<!-- tbx:page={page_number} -->"]
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            output.append("")
        elif _CHAPTER_RE.match(line):
            output.extend((f"## {line}", ""))
        elif _NUMBERED_HEADING_RE.match(line) and len(line) <= 80:
            level = 3 if re.match(r"^(?:[（(]|\d+\.\d+)", line) else 2
            output.extend((f"{'#' * level} {line}", ""))
        elif _LIST_RE.match(line):
            output.append(line if line.startswith(("- ", "* ", "+ ")) else f"- {line}")
        else:
            output.append(line)
    return "\n".join(output)


def _module_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _html_to_markdown(raw: bytes, options: ExtractionOptions) -> tuple[str, str, str | None]:
    try:
        bs4 = importlib.import_module("bs4")
        markdownify = importlib.import_module("markdownify")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "HTML ingestion requires optional packages beautifulsoup4 and markdownify; "
            "install them in the offline build environment"
        ) from exc
    soup = bs4.BeautifulSoup(raw, "html.parser")
    for selector in options.html_remove_selectors:
        for element in soup.select(selector):
            element.decompose()
    root = soup.body or soup
    markdown = markdownify.markdownify(
        str(root),
        heading_style="ATX",
        bullets="-",
        strip=["img"],
    )
    return markdown, "beautifulsoup4+markdownify", _module_version("markdownify")


def _pdf_page_has_images(page: Any) -> bool:
    try:
        return bool(page.images)
    except Exception:
        pass
    try:
        resources = page.get("/Resources") or {}
        xobjects = resources.get("/XObject") or {}
        return any(obj.get_object().get("/Subtype") == "/Image" for obj in xobjects.values())
    except Exception:
        return False


def _validated_ocr_text(
    adapter: OcrAdapter,
    path: Path,
    page_number: int,
    min_characters: int,
) -> str:
    text = adapter.extract_page(path, page_number)
    if not isinstance(text, str) or len(re.sub(r"\s+", "", text)) < min_characters:
        raise ScannedPdfError(
            f"OCR adapter {adapter.name!r} returned insufficient text for PDF page {page_number}; "
            "ingestion stopped instead of fabricating content"
        )
    return text


def _extract_pdf_with_pypdf(
    path: Path,
    options: ExtractionOptions,
    ocr_adapter: OcrAdapter | None,
) -> tuple[str, tuple[str, ...], tuple[int, ...], str | None]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "PDF ingestion requires optional package pypdf (or pdfplumber); install it in the "
            "offline build environment"
        ) from exc
    try:
        reader = pypdf.PdfReader(path)
    except Exception as exc:
        raise ExtractionError(f"unable to open PDF {path.name}: {exc}") from exc
    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ExtractionError("encrypted PDF cannot be read with an empty password") from exc
        if not unlocked:
            raise ExtractionError(
                "encrypted PDF requires a password; no password handling is enabled"
            )
    pages: list[str] = []
    warnings: list[str] = []
    ocr_pages: list[int] = []
    verified_characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            raise ExtractionError(
                f"PDF text extraction failed on page {page_number}: {exc}"
            ) from exc
        observed = len(re.sub(r"\s+", "", text))
        image_only = observed < options.min_pdf_page_characters and _pdf_page_has_images(page)
        if image_only:
            if ocr_adapter is None:
                raise ScannedPdfError(
                    f"PDF page {page_number} appears image-only; configure an explicit "
                    "local OCR adapter or supply a text-native document"
                )
            else:
                text = _validated_ocr_text(
                    ocr_adapter, path, page_number, options.min_pdf_page_characters
                )
                ocr_pages.append(page_number)
        elif observed < options.min_pdf_page_characters:
            warnings.append(f"page {page_number}: blank or very little extractable text")
        verified_characters += len(re.sub(r"\s+", "", text))
        pages.append(_plain_pdf_text_to_markdown(text, page_number))
    if not pages or verified_characters < 40:
        raise ExtractionError("PDF yielded insufficient verified text")
    return "\n\n".join(pages), tuple(warnings), tuple(ocr_pages), _module_version("pypdf")


def _table_to_markdown(rows: list[list[Any]]) -> str:
    normalized = [
        [str(cell or "").replace("|", "\\|").replace("\n", " ").strip() for cell in row]
        for row in rows
        if row
    ]
    if not normalized:
        return ""
    width = max(len(row) for row in normalized)
    normalized = [row + [""] * (width - len(row)) for row in normalized]
    header = normalized[0]
    body = normalized[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _outside_bboxes(word: dict[str, Any], bboxes: list[tuple[float, float, float, float]]) -> bool:
    center_x = (float(word["x0"]) + float(word["x1"])) / 2
    center_y = (float(word["top"]) + float(word["bottom"])) / 2
    return not any(
        x0 <= center_x <= x1 and top <= center_y <= bottom for x0, top, x1, bottom in bboxes
    )


def _group_words_into_lines(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if grouped and abs(float(grouped[-1]["top"]) - top) <= 3.0:
            grouped[-1]["words"].append(word)
        else:
            grouped.append({"top": top, "words": [word]})
    for line in grouped:
        line["words"].sort(key=lambda item: float(item["x0"]))
    return grouped


def _line_text(words: list[dict[str, Any]]) -> str:
    return " ".join(str(word["text"]) for word in words).strip()


def _column_ordered_lines(
    words: list[dict[str, Any]],
    *,
    page_width: float,
    layout_mode: str,
) -> tuple[list[tuple[float, str]], bool]:
    lines = _group_words_into_lines(words)
    if layout_mode == "single_column" or len(lines) < 4:
        return [(float(line["top"]), _line_text(line["words"])) for line in lines], False

    midpoint = page_width / 2
    minimum_gutter = max(14.0, page_width * 0.025)
    dual_indices: list[int] = []
    for index, line in enumerate(lines):
        left = [
            word for word in line["words"] if (float(word["x0"]) + float(word["x1"])) / 2 < midpoint
        ]
        right = [word for word in line["words"] if word not in left]
        if not left or not right:
            continue
        gutter = min(float(word["x0"]) for word in right) - max(float(word["x1"]) for word in left)
        if gutter >= minimum_gutter:
            dual_indices.append(index)

    minimum_dual_lines = 2 if layout_mode == "two_column" else 5
    if len(dual_indices) < minimum_dual_lines:
        return [(float(line["top"]), _line_text(line["words"])) for line in lines], False
    if layout_mode == "auto" and len(dual_indices) / len(lines) < 0.12:
        return [(float(line["top"]), _line_text(line["words"])) for line in lines], False

    vertical_steps = [
        float(lines[index + 1]["top"]) - float(lines[index]["top"])
        for index in range(len(lines) - 1)
        if float(lines[index + 1]["top"]) > float(lines[index]["top"])
    ]
    typical_step = sorted(vertical_steps)[len(vertical_steps) // 2] if vertical_steps else 10.0
    top_start = float(lines[min(dual_indices)]["top"]) - typical_step * 1.5
    top_end = float(lines[max(dual_indices)]["top"]) + typical_step * 1.5

    before: list[tuple[float, str]] = []
    left_column: list[tuple[float, str]] = []
    right_column: list[tuple[float, str]] = []
    after: list[tuple[float, str]] = []
    for line in lines:
        top = float(line["top"])
        if top < top_start:
            before.append((top, _line_text(line["words"])))
            continue
        if top > top_end:
            after.append((top, _line_text(line["words"])))
            continue
        left_words = [
            word for word in line["words"] if (float(word["x0"]) + float(word["x1"])) / 2 < midpoint
        ]
        right_words = [word for word in line["words"] if word not in left_words]
        if left_words:
            left_column.append((top, _line_text(left_words)))
        if right_words:
            right_column.append((top, _line_text(right_words)))

    # Synthetic offsets express reading order only; source locators retain the page.
    ordered = before
    ordered.extend((top_start + index * 0.001, text) for index, (_, text) in enumerate(left_column))
    right_offset = top_start + len(left_column) * 0.001 + 0.5
    ordered.extend(
        (right_offset + index * 0.001, text) for index, (_, text) in enumerate(right_column)
    )
    ordered.extend((top_end + 1 + index * 0.001, text) for index, (_, text) in enumerate(after))
    return ordered, True


def _pdfplumber_page_markdown(page: Any, page_number: int, *, layout_mode: str) -> tuple[str, bool]:
    tables = page.find_tables()
    bboxes = [tuple(table.bbox) for table in tables]
    words = [word for word in page.extract_words() if _outside_bboxes(word, bboxes)]
    lines, column_order_applied = _column_ordered_lines(
        words,
        page_width=float(page.width),
        layout_mode=layout_mode,
    )
    ordered: list[tuple[float, str]] = list(lines)
    for table in tables:
        markdown = _table_to_markdown(table.extract())
        if markdown:
            ordered.append((float(table.bbox[1]), markdown))
    content: list[str] = [f"<!-- tbx:page={page_number} -->"]
    for _, item in sorted(ordered, key=lambda pair: pair[0]):
        if item.startswith("|"):
            content.extend(("", item, ""))
        else:
            content.append(_plain_pdf_text_to_markdown(item, page_number).split("\n", 1)[-1])
    return "\n".join(content), column_order_applied


def _extract_pdf_with_pdfplumber(
    path: Path,
    options: ExtractionOptions,
    ocr_adapter: OcrAdapter | None,
) -> tuple[str, tuple[str, ...], tuple[int, ...], str | None]:
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "PDF table-preserving ingestion requires optional package pdfplumber"
        ) from exc
    pages: list[str] = []
    warnings: list[str] = []
    ocr_pages: list[int] = []
    verified_characters = 0
    try:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                observed = len(re.sub(r"\s+", "", text))
                image_only = observed < options.min_pdf_page_characters and bool(page.images)
                if image_only and ocr_adapter is None:
                    raise ScannedPdfError(
                        f"PDF page {page_number} appears image-only; configure an explicit "
                        "local OCR "
                        "adapter or supply a text-native document"
                    )
                if image_only and ocr_adapter is not None:
                    verified = _validated_ocr_text(
                        ocr_adapter, path, page_number, options.min_pdf_page_characters
                    )
                    verified_characters += len(re.sub(r"\s+", "", verified))
                    pages.append(_plain_pdf_text_to_markdown(verified, page_number))
                    ocr_pages.append(page_number)
                else:
                    verified_characters += observed
                    if observed < options.min_pdf_page_characters:
                        warnings.append(
                            f"page {page_number}: blank or very little extractable text"
                        )
                    page_markdown, column_order_applied = _pdfplumber_page_markdown(
                        page,
                        page_number,
                        layout_mode=options.pdf_layout_mode,
                    )
                    pages.append(page_markdown)
                    if column_order_applied:
                        warnings.append(
                            f"page {page_number}: two-column reading order applied; "
                            "visual review required"
                        )
    except (ScannedPdfError, ExtractionError):
        raise
    except Exception as exc:
        raise ExtractionError(f"pdfplumber failed to extract {path.name}: {exc}") from exc
    if not pages or verified_characters < 40:
        raise ExtractionError("PDF yielded insufficient verified text")
    return "\n\n".join(pages), tuple(warnings), tuple(ocr_pages), _module_version("pdfplumber")


def _pdf_to_markdown(
    path: Path,
    options: ExtractionOptions,
    ocr_adapter: OcrAdapter | None,
) -> tuple[str, str, str | None, tuple[str, ...], tuple[int, ...]]:
    if options.require_pdf_table_detection and options.pdf_engine == "pypdf":
        raise ExtractionError("pypdf cannot satisfy require_pdf_table_detection=true")
    engine = options.pdf_engine
    if engine in {"auto", "pdfplumber"}:
        try:
            markdown, warnings, ocr_pages, version = _extract_pdf_with_pdfplumber(
                path, options, ocr_adapter
            )
            return markdown, "pdfplumber", version, warnings, ocr_pages
        except OptionalDependencyError:
            if engine == "pdfplumber" or options.require_pdf_table_detection:
                raise
    markdown, warnings, ocr_pages, version = _extract_pdf_with_pypdf(path, options, ocr_adapter)
    fallback_warning = (
        "pdfplumber unavailable: table structure was not detected; review canonical Markdown",
    )
    return markdown, "pypdf", version, warnings + fallback_warning, ocr_pages


def canonicalize_source(
    source: SourceSpec,
    options: ExtractionOptions | None = None,
    *,
    ocr_adapter: OcrAdapter | None = None,
) -> CanonicalDocument:
    """Convert one local source to canonical Markdown without any network access."""

    options = options or ExtractionOptions()
    path = source.input_path.resolve()
    if not path.is_file():
        raise ExtractionError(f"local source does not exist or is not a file: {path}")
    raw = path.read_bytes()
    source_sha256 = _sha256_bytes(raw)
    if source.expected_sha256 and source.expected_sha256.lower() != source_sha256:
        raise ExtractionError(
            f"source SHA-256 mismatch for {source.source_id}: expected "
            f"{source.expected_sha256}, observed {source_sha256}"
        )
    warnings: tuple[str, ...] = ()
    ocr_pages: tuple[int, ...] = ()
    if source.input_format == "markdown":
        try:
            markdown = raw.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExtractionError("Markdown sources must be valid UTF-8") from exc
        extractor = "utf8-markdown"
        version = None
    elif source.input_format == "html":
        markdown, extractor, version = _html_to_markdown(raw, options)
    elif source.input_format == "pdf":
        markdown, extractor, version, warnings, ocr_pages = _pdf_to_markdown(
            path, options, ocr_adapter
        )
    else:  # pragma: no cover - validated by configuration loading
        raise ExtractionError(f"unsupported source format: {source.input_format}")
    canonical = _normalize_markdown(_ensure_title(markdown, source.title))
    blocks = parse_canonical_markdown(canonical)
    return CanonicalDocument(
        source=source,
        source_sha256=source_sha256,
        canonical_markdown=canonical,
        canonical_markdown_sha256=_sha256_bytes(canonical.encode("utf-8")),
        blocks=blocks,
        extractor=extractor,
        extractor_version=version,
        warnings=warnings,
        ocr_used=bool(ocr_pages),
        ocr_pages=ocr_pages,
    )
