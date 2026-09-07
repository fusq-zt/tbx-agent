#!/usr/bin/env python3
"""Validate the Git upload candidate and/or a TBX-Agent source ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

from build_source_release import (
    ARCHIVE_PREFIX,
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_TOTAL_BYTES,
    FORMAT_VERSION,
    ZIP_TIMESTAMP,
    ReleasePolicyError,
    SourceFile,
    collect_source_files,
    normalize_relative_path,
    scan_content,
    sha256_file,
    source_tree_sha256,
    validate_release_path,
)


def _git_output(root: Path, *arguments: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise ReleasePolicyError(
            f"Git candidate inspection failed: {message or process.returncode}"
        )
    return process.stdout


def git_candidate_paths(root: Path) -> list[str]:
    root = root.resolve(strict=True)
    git_root = Path(
        _git_output(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve(strict=True)
    try:
        relative_root = root.relative_to(git_root)
    except ValueError as exc:
        raise ReleasePolicyError("source root is outside the detected Git worktree") from exc
    pathspec = relative_root.as_posix() if relative_root.parts else "."
    raw = _git_output(
        git_root,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        pathspec,
    )
    candidates: list[str] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        absolute = (git_root / encoded.decode("utf-8", errors="strict")).resolve()
        try:
            relative = absolute.relative_to(root).as_posix()
        except ValueError as exc:
            raise ReleasePolicyError(f"Git returned path outside source root: {absolute}") from exc
        candidates.append(normalize_relative_path(relative))
    folded: set[str] = set()
    for candidate in candidates:
        if candidate.casefold() in folded:
            raise ReleasePolicyError(f"case-insensitive duplicate Git candidate: {candidate}")
        folded.add(candidate.casefold())
    return sorted(candidates, key=str.casefold)


def check_git_candidate(root: Path, expected: list[SourceFile]) -> dict[str, Any]:
    candidate = git_candidate_paths(root)
    expected_paths = [item.path for item in expected]
    expected_set = set(expected_paths)
    candidate_set = set(candidate)
    extra = sorted(candidate_set - expected_set)
    missing = sorted(expected_set - candidate_set)
    if extra or missing:
        details = []
        if extra:
            details.append(f"unapproved Git files={extra[:20]}")
        if missing:
            details.append(f"release files ignored/missing from Git={missing[:20]}")
        raise ReleasePolicyError(
            "Git candidate does not match release allowlist: " + "; ".join(details)
        )
    # Re-validate each path rather than trusting a matching set derived from a
    # different subprocess boundary.
    for relative in candidate:
        validate_release_path(relative)
        path = root / relative
        if path.is_symlink():
            raise ReleasePolicyError(f"Git candidate contains a symlink: {relative}")
        if not path.is_file():
            raise ReleasePolicyError(f"Git candidate is not a regular file: {relative}")
        scan_content(relative, path.read_bytes())
    return {"file_count": len(candidate), "source_tree_sha256": source_tree_sha256(expected)}


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleasePolicyError(f"invalid release manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ReleasePolicyError("unsupported release manifest format")
    if payload.get("archive_prefix") != ARCHIVE_PREFIX:
        raise ReleasePolicyError("release manifest archive_prefix mismatch")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ReleasePolicyError("release manifest has no files")
    return payload


def check_zip(
    archive_path: Path,
    manifest_path: Path,
    *,
    max_file_bytes: int,
    max_total_bytes: int,
    expected: list[SourceFile] | None,
) -> dict[str, Any]:
    archive_path = archive_path.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = _load_manifest(manifest_path)
    if manifest.get("archive") != archive_path.name:
        raise ReleasePolicyError("manifest archive filename does not match selected ZIP")
    observed_archive_hash = sha256_file(archive_path)
    if manifest.get("archive_sha256") != observed_archive_hash:
        raise ReleasePolicyError("source ZIP SHA256 does not match manifest")

    manifest_rows = manifest["files"]
    manifest_by_path: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise ReleasePolicyError("release manifest file row is not an object")
        relative = validate_release_path(str(row.get("path", "")))
        size = row.get("size")
        digest = row.get("sha256")
        if not isinstance(size, int) or size < 0 or size > max_file_bytes:
            raise ReleasePolicyError(f"invalid manifest size for {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ReleasePolicyError(f"invalid manifest SHA256 for {relative}")
        if relative.casefold() in {name.casefold() for name in manifest_by_path}:
            raise ReleasePolicyError(f"duplicate manifest path: {relative}")
        manifest_by_path[relative] = row

    total = 0
    observed: list[SourceFile] = []
    seen_casefold: set[str] = set()
    with zipfile.ZipFile(archive_path, "r") as archive:
        if archive.testzip() is not None:
            raise ReleasePolicyError("source ZIP has a CRC failure")
        for info in archive.infolist():
            if info.flag_bits & 0x1:
                raise ReleasePolicyError(f"encrypted ZIP entry is forbidden: {info.filename}")
            if info.is_dir():
                raise ReleasePolicyError(
                    f"explicit directory ZIP entries are forbidden: {info.filename}"
                )
            name = normalize_relative_path(info.filename)
            prefix = f"{ARCHIVE_PREFIX}/"
            if not name.startswith(prefix):
                raise ReleasePolicyError(f"ZIP entry has wrong archive prefix: {name}")
            relative = validate_release_path(name[len(prefix) :])
            if relative.casefold() in seen_casefold:
                raise ReleasePolicyError(f"case-insensitive duplicate ZIP entry: {relative}")
            seen_casefold.add(relative.casefold())
            unix_mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                raise ReleasePolicyError(f"ZIP symlink entry is forbidden: {relative}")
            if info.date_time != ZIP_TIMESTAMP:
                raise ReleasePolicyError(f"non-deterministic ZIP timestamp: {relative}")
            if info.file_size > max_file_bytes:
                raise ReleasePolicyError(f"ZIP entry exceeds size limit: {relative}")
            total += info.file_size
            if total > max_total_bytes:
                raise ReleasePolicyError("ZIP exceeds total uncompressed size limit")
            data = archive.read(info)
            if len(data) != info.file_size:
                raise ReleasePolicyError(f"truncated ZIP entry: {relative}")
            scan_content(relative, data)
            digest = hashlib.sha256(data).hexdigest()
            row = manifest_by_path.get(relative)
            if row is None or row["size"] != len(data) or row["sha256"] != digest:
                raise ReleasePolicyError(f"ZIP entry does not match manifest: {relative}")
            observed.append(
                SourceFile(path=relative, source=archive_path, size=len(data), sha256=digest)
            )

    if set(manifest_by_path) != {item.path for item in observed}:
        raise ReleasePolicyError("manifest and ZIP file sets differ")
    observed.sort(key=lambda item: item.path.casefold())
    tree_hash = source_tree_sha256(observed)
    if manifest.get("source_tree_sha256") != tree_hash:
        raise ReleasePolicyError("source tree SHA256 does not match manifest")
    if manifest.get("file_count") != len(observed):
        raise ReleasePolicyError("manifest file_count mismatch")
    if manifest.get("total_uncompressed_bytes") != total:
        raise ReleasePolicyError("manifest total_uncompressed_bytes mismatch")
    if expected is not None:
        expected_rows = [(item.path, item.size, item.sha256) for item in expected]
        observed_rows = [(item.path, item.size, item.sha256) for item in observed]
        if expected_rows != observed_rows:
            raise ReleasePolicyError("source ZIP is stale relative to the Git/source candidate")
    return {
        "archive": str(archive_path),
        "archive_sha256": observed_archive_hash,
        "file_count": len(observed),
        "source_tree_sha256": tree_hash,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--zip", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--skip-git", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = args.root.expanduser().resolve(strict=True)
        expected = collect_source_files(
            root,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
        result: dict[str, Any] = {
            "status": "valid",
            "source": {
                "file_count": len(expected),
                "source_tree_sha256": source_tree_sha256(expected),
            },
        }
        if not args.skip_git:
            result["git_candidate"] = check_git_candidate(root, expected)
        if args.zip is not None:
            manifest = args.manifest or args.zip.with_suffix(".manifest.json")
            result["zip"] = check_zip(
                args.zip,
                manifest,
                max_file_bytes=args.max_file_bytes,
                max_total_bytes=args.max_total_bytes,
                expected=expected,
            )
        elif args.manifest is not None:
            raise ReleasePolicyError("--manifest requires --zip")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ReleasePolicyError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(
            json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
