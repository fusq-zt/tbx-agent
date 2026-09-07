from __future__ import annotations

import hashlib
import re
from dataclasses import replace

from .errors import ConfigurationError, IntegrityError
from .models import CanonicalBlock, CanonicalDocument, ChunkingOptions, ChunkRecord

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[。！？；.!?;])\s*")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _split_text(text: str, maximum: int) -> list[str]:
    sentences = [item.strip() for item in _SENTENCE_BOUNDARY_RE.split(text) if item.strip()]
    if not sentences:
        sentences = [text]
    output: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > maximum:
            if current:
                output.append(current)
                current = ""
            output.extend(
                sentence[index : index + maximum] for index in range(0, len(sentence), maximum)
            )
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= maximum:
            current = f"{current}\n{sentence}"
        else:
            output.append(current)
            current = sentence
    if current:
        output.append(current)
    return output


def _split_table(markdown: str, maximum: int) -> list[str]:
    rows = markdown.splitlines()
    if len(rows) < 2:
        return _split_text(markdown, maximum)
    prefix = rows[:2]
    header = "\n".join(prefix)
    if len(header) > maximum:
        raise IntegrityError("table header exceeds chunk max_characters and cannot be split safely")
    output: list[str] = []
    current = list(prefix)
    for row in rows[2:]:
        candidate = "\n".join([*current, row])
        if len(candidate) <= maximum:
            current.append(row)
        else:
            if len(current) == 2:
                raise IntegrityError("single table row exceeds chunk max_characters")
            output.append("\n".join(current))
            current = [*prefix, row]
    if len(current) > 2 or not output:
        output.append("\n".join(current))
    return output


def _split_block(block: CanonicalBlock, maximum: int) -> list[CanonicalBlock]:
    if len(block.markdown) <= maximum:
        return [block]
    if block.kind == "heading":
        raise IntegrityError("heading exceeds chunk max_characters and cannot be split safely")
    if block.kind == "table":
        fragments = _split_table(block.markdown, maximum)
    elif block.kind == "code":
        raise IntegrityError("code block exceeds chunk max_characters and cannot be split safely")
    elif block.kind == "list":
        fragments = []
        current: list[str] = []
        for line in block.markdown.splitlines():
            candidate = "\n".join([*current, line])
            if current and len(candidate) > maximum:
                fragments.append("\n".join(current))
                current = [line]
            elif len(line) > maximum:
                if current:
                    fragments.append("\n".join(current))
                    current = []
                fragments.extend(_split_text(line, maximum))
            else:
                current.append(line)
        if current:
            fragments.append("\n".join(current))
    else:
        fragments = _split_text(block.markdown, maximum)
    return [replace(block, markdown=fragment) for fragment in fragments]


def _validate_options(options: ChunkingOptions) -> None:
    if not (1 <= options.min_characters <= options.target_characters <= options.max_characters):
        raise ConfigurationError(
            "chunk limits must satisfy 1 <= min_characters <= target_characters <= max_characters"
        )
    if options.overlap_blocks < 0 or options.overlap_blocks > 4:
        raise ConfigurationError("overlap_blocks must be between 0 and 4")


def _render(blocks: list[CanonicalBlock]) -> str:
    return "\n\n".join(block.markdown.strip() for block in blocks if block.markdown.strip()).strip()


def _section_prefix(section_path: tuple[str, ...]) -> str:
    return "\n\n".join(
        f"{'#' * min(level, 6)} {title}" for level, title in enumerate(section_path, start=1)
    )


def _group_section(
    blocks: list[CanonicalBlock],
    section_path: tuple[str, ...],
    options: ChunkingOptions,
) -> list[list[CanonicalBlock]]:
    prefix = _section_prefix(section_path)
    prefix_cost = len(prefix) + 2 if prefix else 0
    available = options.max_characters - prefix_cost
    if available < 1:
        raise IntegrityError("section heading path leaves no space within max_characters")
    expanded = [fragment for block in blocks for fragment in _split_block(block, available)]
    groups: list[list[CanonicalBlock]] = []
    current: list[CanonicalBlock] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for block in expanded:
        candidate = [*current, block]
        candidate_length = prefix_cost + len(_render(candidate))
        current_length = prefix_cost + len(_render(current))
        should_flush = (current and candidate_length > options.max_characters) or (
            current
            and current_length >= options.target_characters
            and current_length >= options.min_characters
        )
        if should_flush:
            previous = list(current)
            flush()
            if options.overlap_blocks and previous:
                overlap = previous[-options.overlap_blocks :]
                if prefix_cost + len(_render([*overlap, block])) <= options.max_characters:
                    current.extend(overlap)
            current.append(block)
        else:
            current.append(block)
    flush()
    return groups


def _chunk_locator(
    section_path: tuple[str, ...],
    page_start: int | None,
    page_end: int | None,
    block_start: int,
    block_end: int,
) -> str:
    section = " > ".join(section_path) if section_path else "document root"
    if page_start is None:
        pages = "page n/a"
    elif page_start == page_end:
        pages = f"page {page_start}"
    else:
        pages = f"pages {page_start}-{page_end}"
    return f"{section}; {pages}; blocks {block_start}-{block_end}"


def chunk_document(
    document: CanonicalDocument,
    options: ChunkingOptions | None = None,
) -> tuple[ChunkRecord, ...]:
    """Chunk a canonical document at section/block boundaries with stable identifiers."""

    options = options or ChunkingOptions()
    _validate_options(options)
    sections: list[tuple[tuple[str, ...], list[CanonicalBlock]]] = []
    for block in document.blocks:
        if block.kind == "heading":
            continue
        if sections and sections[-1][0] == block.heading_path:
            sections[-1][1].append(block)
        else:
            sections.append((block.heading_path, [block]))
    groups = [
        (section_path, group)
        for section_path, blocks in sections
        for group in _group_section(blocks, section_path, options)
    ]
    records: list[ChunkRecord] = []
    identifier_counts: dict[str, int] = {}
    for section_path, group in groups:
        prefix = _section_prefix(section_path)
        payload = _render(group)
        text = f"{prefix}\n\n{payload}" if prefix else payload
        if not text:
            continue
        if len(text) > options.max_characters:
            raise IntegrityError(
                f"chunk exceeds max_characters after splitting: {len(text)} > "
                f"{options.max_characters}"
            )
        pages = [
            page
            for block in group
            for page in (block.page_start, block.page_end)
            if page is not None
        ]
        page_start = min(pages) if pages else None
        page_end = max(pages) if pages else None
        block_start = min(block.ordinal for block in group)
        block_end = max(block.ordinal for block in group)
        content_sha256 = _sha256_text(text)
        identity_payload = "\x1f".join(
            (document.source.source_id, " > ".join(section_path), content_sha256)
        )
        base = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:24]
        occurrence = identifier_counts.get(base, 0)
        identifier_counts[base] = occurrence + 1
        suffix = f"-{occurrence + 1}" if occurrence else ""
        records.append(
            ChunkRecord(
                schema_version=1,
                chunk_id=f"{document.source.source_id}:{base}{suffix}",
                source_id=document.source.source_id,
                source_sha256=document.source_sha256,
                canonical_markdown_sha256=document.canonical_markdown_sha256,
                content_sha256=content_sha256,
                title=document.source.title,
                organization=document.source.organization,
                publication_year=document.source.publication_year,
                jurisdiction=document.source.jurisdiction,
                url=document.source.url,
                language=document.source.language,
                topics=document.source.topics,
                section_path=section_path,
                block_types=tuple(
                    dict.fromkeys(
                        (("heading",) if section_path else ())
                        + tuple(block.kind for block in group)
                    )
                ),
                block_start=block_start,
                block_end=block_end,
                page_start=page_start,
                page_end=page_end,
                locator=_chunk_locator(section_path, page_start, page_end, block_start, block_end),
                text=text,
            )
        )
    if not records:
        raise IntegrityError(f"source {document.source.source_id} produced no chunks")
    return tuple(records)
