#!/usr/bin/env python3
"""Build a deterministic, policy-checked TBX-Agent source archive.

The release is assembled from an explicit public-source allowlist. Runtime state,
datasets, model weights, experiment receipts and local credentials are never
copied merely because they happen to exist in the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FORMAT_VERSION = 1
ARCHIVE_PREFIX = "tbx-agent"
DEFAULT_MAX_FILE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 50 * 1024 * 1024
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

PUBLIC_ROOT_FILES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitattributes",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "Dockerfile",
        "LICENSE",
        "Makefile",
        "NOTICE",
        "README.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        "docker-compose.yml",
        "pyproject.toml",
    }
)
PUBLIC_DIRS = frozenset(
    {
        ".github",
        ".streamlit",
        "configs",
        "knowledge",
        "retrieval",
        "src",
        "tests",
        "ui",
    }
)

DEPLOYMENT_RELEASE_FILES = frozenset(
    {
        ".dockerignore",
        ".env.example",
        ".gitignore",
        "Dockerfile",
        "docker-compose.yml",
        "docs/deployment.md",
        "docs/deployment/README.md",
        "docs/deployment/docker.md",
        "docs/deployment/model_artifacts.md",
        "docs/deployment/medgemma_runtime.md",
        "docs/deployment/quickstart.md",
        "docs/deployment/qwen_runtime.md",
        "docs/deployment/inference_models.md",
        "docs/deployment/publishing.md",
        "docs/deployment/source_release.md",
        "docs/anatomy_spatial_evidence.md",
        "docs/medsam_refinement.md",
        "configs/app.yaml",
        "configs/llm_runtime.yaml",
        "configs/retrieval.yaml",
        "configs/model_sources.yaml",
        "configs/qwen_runtime_sources.yaml",
        "configs/rank03_runtime.json",
        "evaluation/fixtures/medical_dialogue_qa_v1.json",
        "scripts/bootstrap.ps1",
        "scripts/bootstrap.sh",
        "scripts/bootstrap_dfine.py",
        "scripts/bootstrap_models.py",
        "scripts/bootstrap_qwen.py",
        "scripts/build_llamacpp_linux.sh",
        "scripts/build_rag_index.py",
        "scripts/check_setup.py",
        "scripts/evaluate_agent_runtime.py",
        "scripts/evaluate_medical_dialogue_runtime.py",
        "scripts/evaluate_rag_retrieval.py",
        "scripts/smoke_medsam_refinement.py",
        "scripts/build_source_release.py",
        "scripts/check_release.py",
        "scripts/run_local.ps1",
        "scripts/run_local.sh",
        "scripts/setup_medgemma.py",
        "scripts/install_vision_bundle.py",
    }
)
PUBLIC_SCRIPT_FILES = frozenset(
    {
        "scripts/bootstrap.ps1",
        "scripts/bootstrap.sh",
        "scripts/bootstrap_dfine.py",
        "scripts/bootstrap_models.py",
        "scripts/bootstrap_qwen.py",
        "scripts/build_llamacpp_linux.sh",
        "scripts/build_rag_index.py",
        "scripts/check_setup.py",
        "scripts/evaluate_agent_runtime.py",
        "scripts/evaluate_medical_dialogue_runtime.py",
        "scripts/evaluate_rag_retrieval.py",
        "scripts/smoke_medsam_refinement.py",
        "scripts/build_source_release.py",
        "scripts/check_release.py",
        "scripts/run_local.ps1",
        "scripts/run_local.sh",
        "scripts/setup_medgemma.py",
        "scripts/test_openai_compatible_api.py",
        "scripts/install_vision_bundle.py",
    }
)
PUBLIC_EVALUATION_FILES = frozenset(
    {
        "evaluation/cases.jsonl",
        "evaluation/eval_config.json",
        "evaluation/fixtures/medical_dialogue_qa_v1.json",
        "evaluation/fixtures/narrator_precision_pair_v1.json",
        "evaluation/llamacpp_eval_config.json",
        "evaluation/llamacpp_eval_config_v1.json",
        "evaluation/ollama_eval_config.json",
        "evaluation/software_conformance_config_v1.json",
        "evaluation/system_bench_config.json",
        "evaluation/system_bench_config_v1_1.json",
        "evaluation/system_bench_config_v1_2.json",
        "evaluation/system_bench_config_v1_3.json",
        "evaluation/system_bench_config_v1_4.json",
        "evaluation/system_bench_config_v1_5.json",
        "evaluation/system_bench_config_v1_6.json",
        "evaluation/suites/system_v1/README.md",
        "evaluation/suites/system_v1/cases.jsonl",
        "evaluation/suites/system_v1/manifest.json",
        "evaluation/suites/system_v1_2/README.md",
        "evaluation/suites/system_v1_2/cases.jsonl",
        "evaluation/suites/system_v1_2/manifest.json",
        "evaluation/suites/system_v1_3/README.md",
        "evaluation/suites/system_v1_3/cases.jsonl",
        "evaluation/suites/system_v1_3/manifest.json",
        "evaluation/suites/system_v1_4/README.md",
        "evaluation/suites/system_v1_4/cases.jsonl",
        "evaluation/suites/system_v1_4/manifest.json",
        "evaluation/suites/system_v1_5/README.md",
        "evaluation/suites/system_v1_5/cases.jsonl",
        "evaluation/suites/system_v1_5/manifest.json",
        "evaluation/suites/system_v1_6/README.md",
        "evaluation/suites/system_v1_6/cases.jsonl",
        "evaluation/suites/system_v1_6/manifest.json",
        "evaluation/suites/trajectory_v1/README.md",
        "evaluation/suites/trajectory_v1/cases.jsonl",
        "evaluation/suites/trajectory_v1/manifest.json",
        "evaluation/suites/trajectory_v2/README.md",
        "evaluation/suites/trajectory_v2/cases.jsonl",
        "evaluation/suites/trajectory_v2/manifest.json",
        "evaluation/suites/trajectory_v3/README.md",
        "evaluation/suites/trajectory_v3/cases.jsonl",
        "evaluation/suites/trajectory_v3/manifest.json",
        "evaluation/suites/software_conformance_v1/README.md",
        "evaluation/suites/software_conformance_v1/cases.jsonl",
        "evaluation/suites/software_conformance_v1/manifest.json",
    }
)
PUBLIC_RETRIEVAL_EVAL_FILES = frozenset(
    {
        "evaluation/retrieval/smoke_v1/config.json",
        "evaluation/retrieval/smoke_v1/qrels.jsonl",
        "evaluation/retrieval/smoke_v1/queries.jsonl",
        "evaluation/retrieval/smoke_v2/config.json",
        "evaluation/retrieval/smoke_v2/qrels.jsonl",
        "evaluation/retrieval/smoke_v2/queries.jsonl",
        "evaluation/retrieval/smoke_v2/README.md",
        "evaluation/retrieval/smoke_v3/config.json",
        "evaluation/retrieval/smoke_v3/qrels.jsonl",
        "evaluation/retrieval/smoke_v3/queries.jsonl",
        "evaluation/retrieval/core_guideline_six_v2/config.json",
        "evaluation/retrieval/core_guideline_six_v2/qrels.jsonl",
        "evaluation/retrieval/core_guideline_six_v2/queries.jsonl",
        "evaluation/retrieval/smoke_v4/config.json",
        "evaluation/retrieval/smoke_v4/qrels.jsonl",
        "evaluation/retrieval/smoke_v4/queries.jsonl",
        "evaluation/retrieval/core_guideline_seven_v1/config.json",
        "evaluation/retrieval/core_guideline_seven_v1/qrels.jsonl",
        "evaluation/retrieval/core_guideline_seven_v1/queries.jsonl",
        "evaluation/retrieval/cdc_supplemental_v1/config.json",
        "evaluation/retrieval/cdc_supplemental_v1/qrels.jsonl",
        "evaluation/retrieval/cdc_supplemental_v1/queries.jsonl",
        "evaluation/retrieval/cdc_supplemental_v1/README.md",
        "evaluation/retrieval/smoke_v5/config.json",
        "evaluation/retrieval/smoke_v5/qrels.jsonl",
        "evaluation/retrieval/smoke_v5/queries.jsonl",
        "evaluation/retrieval/smoke_v5/README.md",
        "evaluation/retrieval/core_guideline_seven_v2/config.json",
        "evaluation/retrieval/core_guideline_seven_v2/qrels.jsonl",
        "evaluation/retrieval/core_guideline_seven_v2/queries.jsonl",
        "evaluation/retrieval/core_guideline_seven_v2/README.md",
        "evaluation/retrieval/cdc_supplemental_v2/config.json",
        "evaluation/retrieval/cdc_supplemental_v2/qrels.jsonl",
        "evaluation/retrieval/cdc_supplemental_v2/queries.jsonl",
        "evaluation/retrieval/cdc_supplemental_v2/README.md",
    }
)
INTERNAL_RESEARCH_FILES = frozenset(
    {
        # Frozen workstation/post-hoc studies bind old private checkpoints and
        # observations. They remain in the maintainer archive, not the public
        # inference-only project.
        "evaluation/suites/system_v1/deterministic_mock_candidate.json",
        "evaluation/tbx11k_protocol_a_validation_config.json",
        "src/tbx_agent/evaluation/external_rank03_archives.py",
        "src/tbx_agent/evaluation/shenzhen_classifier_half_split.py",
        "src/tbx_agent/evaluation/shenzhen_classifier_panel.py",
        "src/tbx_agent/evaluation/shenzhen_detector_sweep.py",
        "src/tbx_agent/evaluation/tbx11k_validation.py",
        "tests/test_external_rank03_archives.py",
        "tests/test_narrator_precision_pair.py",
        "tests/test_shenzhen_classifier_half_split.py",
        "tests/test_shenzhen_classifier_panel.py",
        "tests/test_shenzhen_detector_sweep.py",
        "tests/test_tbx11k_validation.py",
    }
)
PUBLIC_DOC_FILES = frozenset(
    {
        "docs/api.md",
        "docs/agent_trajectory_evaluation.md",
        "docs/anatomy_spatial_evidence.md",
        "docs/architecture.md",
        "docs/deployment.md",
        "docs/deployment/README.md",
        "docs/deployment/docker.md",
        "docs/deployment/model_artifacts.md",
        "docs/deployment/medgemma_runtime.md",
        "docs/deployment/quickstart.md",
        "docs/deployment/qwen_runtime.md",
        "docs/deployment/inference_models.md",
        "docs/deployment/publishing.md",
        "docs/deployment/source_release.md",
        "docs/evaluation.md",
        "docs/guideline_audit.md",
        "docs/knowledge_ingestion.md",
        "docs/medsam_refinement.md",
        "docs/production_security.md",
        "docs/retrieval.md",
        "docs/safety_case.md",
    }
)

DENIED_COMPONENTS = frozenset(
    {
        ".git",
        ".idea",
        ".models",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".run",
        ".runtime-smoke",
        ".tmp_ui_smoke",
        ".venv",
        ".vscode",
        "__pycache__",
        "artifacts",
        "build",
        "cache",
        "checkpoints",
        "data",
        "datasets",
        "dist",
        "env",
        "htmlcov",
        "logs",
        "models",
        "reports",
        "runtime",
        "secrets",
        "temp",
        "third_party",
        "training",
        "tmp",
        "uploads",
        "venv",
        "weights",
    }
)
UNIVERSAL_DENIED_COMPONENTS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        ".vscode",
        "__pycache__",
        "training",
        "env",
        "secrets",
        "venv",
    }
)
NESTED_DENIED_COMPONENTS = frozenset(
    {
        "artifacts",
        "build",
        "cache",
        "checkpoints",
        "data",
        "datasets",
        "dist",
        "logs",
        "models",
        "reports",
        "runtime",
        "temp",
        "tmp",
        "uploads",
        "weights",
    }
)
DENIED_SUFFIXES = frozenset(
    {
        ".7z",
        ".avi",
        ".ckpt",
        ".db",
        ".dcm",
        ".dicom",
        ".engine",
        ".gguf",
        ".jpeg",
        ".jpg",
        ".log",
        ".mov",
        ".mp4",
        ".onnx",
        ".pem",
        ".pid",
        ".png",
        ".pt",
        ".pth",
        ".pyc",
        ".pyd",
        ".rar",
        ".safetensors",
        ".sqlite",
        ".sqlite3",
        ".tgz",
        ".zip",
    }
)
DENIED_MULTI_SUFFIXES = (".tar.gz",)
ALLOWED_SUFFIXES = frozenset(
    {
        "",
        ".cff",
        ".css",
        ".csv",
        ".dockerignore",
        ".html",
        ".json",
        ".jsonl",
        ".md",
        ".pdf",
        ".ps1",
        ".py",
        ".sh",
        ".svg",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private-key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws-access-key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("openai-key", re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("huggingface-token", re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b")),
    ("github-token", re.compile(rb"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github-fine-grained-token", re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
)
SECRET_ASSIGNMENT = re.compile(
    rb"(?im)^\s*(?:export\s+)?"
    rb"(?:[A-Z0-9_]*(?:API_KEY|PASSWORD|PASSWD|CLIENT_SECRET|ACCESS_TOKEN|AUTH_TOKEN))"
    rb"\s*[:=]\s*['\"]?([^\s'\"#,]{12,})"
)
PLACEHOLDER_MARKERS = (
    b"${",
    b"<",
    b"changeme",
    b"example",
    b"placeholder",
    b"replace",
    b"your_",
)
WORKSTATION_PATH = re.compile(
    rb"(?i)(?:[A-Z]:[\\/]Users[\\/][^\\/\s'\"]+|/(?:home/[^/\s'\"]+|Users/[^/\s'\"]+|root)/)"
)

REQUIRED_RELEASE_FILES = (
    frozenset(
        {
            ".github/workflows/ci.yml",
            "README.md",
            "docs/evaluation.md",
            "evaluation/system_bench_config.json",
            "evaluation/system_bench_config_v1_1.json",
            "evaluation/system_bench_config_v1_2.json",
            "evaluation/system_bench_config_v1_3.json",
            "evaluation/system_bench_config_v1_4.json",
            "evaluation/system_bench_config_v1_5.json",
            "evaluation/system_bench_config_v1_6.json",
            "evaluation/software_conformance_config_v1.json",
            "evaluation/suites/system_v1/README.md",
            "evaluation/suites/system_v1/cases.jsonl",
            "evaluation/suites/system_v1/manifest.json",
            "evaluation/suites/system_v1_2/README.md",
            "evaluation/suites/system_v1_2/cases.jsonl",
            "evaluation/suites/system_v1_2/manifest.json",
            "evaluation/suites/system_v1_3/README.md",
            "evaluation/suites/system_v1_3/cases.jsonl",
            "evaluation/suites/system_v1_3/manifest.json",
            "evaluation/suites/system_v1_4/README.md",
            "evaluation/suites/system_v1_4/cases.jsonl",
            "evaluation/suites/system_v1_4/manifest.json",
            "evaluation/suites/system_v1_5/README.md",
            "evaluation/suites/system_v1_5/cases.jsonl",
            "evaluation/suites/system_v1_5/manifest.json",
            "evaluation/suites/system_v1_6/README.md",
            "evaluation/suites/system_v1_6/cases.jsonl",
            "evaluation/suites/system_v1_6/manifest.json",
            "evaluation/suites/trajectory_v1/README.md",
            "evaluation/suites/trajectory_v1/cases.jsonl",
            "evaluation/suites/trajectory_v1/manifest.json",
            "evaluation/suites/trajectory_v2/README.md",
            "evaluation/suites/trajectory_v2/cases.jsonl",
            "evaluation/suites/trajectory_v2/manifest.json",
            "evaluation/suites/trajectory_v3/README.md",
            "evaluation/suites/trajectory_v3/cases.jsonl",
            "evaluation/suites/trajectory_v3/manifest.json",
            "docs/agent_trajectory_evaluation.md",
            "evaluation/suites/software_conformance_v1/README.md",
            "evaluation/suites/software_conformance_v1/cases.jsonl",
            "evaluation/suites/software_conformance_v1/manifest.json",
            "pyproject.toml",
            "src/tbx_agent/__init__.py",
            "src/tbx_agent/evaluation/trajectory.py",
            "src/tbx_agent/evaluation/system_bench.py",
            "src/tbx_agent/evaluation/system_bench_ci.py",
            "src/tbx_agent/evaluation/software_conformance.py",
        }
    )
    | DEPLOYMENT_RELEASE_FILES
    | PUBLIC_RETRIEVAL_EVAL_FILES
)


class ReleasePolicyError(RuntimeError):
    """Raised when a candidate is unsafe or outside the public release policy."""


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: str
    source: Path
    size: int
    sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_relative_path(value: str | PurePosixPath) -> str:
    raw = str(value).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or not candidate.parts:
        raise ReleasePolicyError(f"unsafe release path: {value!s}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ReleasePolicyError(f"unsafe release path: {value!s}")
    return candidate.as_posix()


def is_public_path(relative: str) -> bool:
    relative = normalize_relative_path(relative)
    parts = PurePosixPath(relative).parts
    if "training" in {part.casefold() for part in parts}:
        return False
    if PurePosixPath(relative).name.casefold().startswith("train_"):
        return False
    if relative in INTERNAL_RESEARCH_FILES:
        return False
    # Raw guideline documents may have independent copyright/reuse terms. The
    # public source archive contains only the curated metadata/chunk snapshot
    # and instructions for operators to prepare their own source documents.
    if relative.startswith("knowledge/sources/"):
        return False
    if len(parts) == 1:
        return relative in PUBLIC_ROOT_FILES
    if parts[0] in PUBLIC_DIRS:
        return True
    if relative in PUBLIC_SCRIPT_FILES:
        return True
    if relative in PUBLIC_EVALUATION_FILES or relative in PUBLIC_RETRIEVAL_EVAL_FILES:
        return True
    if parts[0] == "docs":
        return relative in PUBLIC_DOC_FILES
    return False


def validate_release_path(relative: str) -> str:
    relative = normalize_relative_path(relative)
    path = PurePosixPath(relative)
    lowered_parts = {part.casefold() for part in path.parts}
    denied = sorted(
        lowered_parts.intersection(
            component.casefold() for component in UNIVERSAL_DENIED_COMPONENTS
        )
    )
    if path.parts[0].casefold() in {component.casefold() for component in DENIED_COMPONENTS}:
        denied.append(path.parts[0].casefold())
    nested_denied = lowered_parts.intersection(
        component.casefold() for component in NESTED_DENIED_COMPONENTS
    )
    # ``tbx_agent.artifacts`` is source code for the external model-artifact
    # manager. Runtime artifact directories are denied everywhere else.
    if relative.startswith("src/tbx_agent/artifacts/"):
        nested_denied.discard("artifacts")
    denied.extend(sorted(nested_denied))
    if denied:
        denied = sorted(set(denied))
        raise ReleasePolicyError(f"denied path component in {relative}: {', '.join(denied)}")
    lowered = relative.casefold()
    if any(lowered.endswith(suffix) for suffix in DENIED_MULTI_SUFFIXES):
        raise ReleasePolicyError(f"denied archive/model/data suffix: {relative}")
    suffix = path.suffix.casefold()
    if suffix in DENIED_SUFFIXES:
        raise ReleasePolicyError(f"denied generated/model/data suffix: {relative}")
    if suffix not in ALLOWED_SUFFIXES and path.name not in PUBLIC_ROOT_FILES:
        raise ReleasePolicyError(f"unreviewed source suffix: {relative}")
    if not is_public_path(relative):
        raise ReleasePolicyError(f"path is outside the public source allowlist: {relative}")
    return relative


def scan_content(relative: str, data: bytes) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(data):
            raise ReleasePolicyError(f"possible {label} in {relative}")
    for match in SECRET_ASSIGNMENT.finditer(data):
        value = match.group(1).lower()
        # Function/variable references in examples are not embedded credentials.
        code_reference = value[:-2] if value.endswith(b"()") else value
        if re.fullmatch(rb"[a-z_][a-z0-9_.]*", code_reference):
            continue
        if re.fullmatch(rb"[a-z_][a-z0-9_.]*\([a-z0-9_.,]*\)", value):
            continue
        if not any(marker in value for marker in PLACEHOLDER_MARKERS):
            raise ReleasePolicyError(f"possible hard-coded credential assignment in {relative}")
    if WORKSTATION_PATH.search(data):
        raise ReleasePolicyError(f"workstation-specific user-profile path in {relative}")


def _walk_candidate_paths(root: Path) -> Iterable[Path]:
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for directory in sorted(directories, key=str.casefold):
            candidate = current_path / directory
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                raise ReleasePolicyError(f"symlinks are not allowed in source releases: {relative}")
            relative_parts = candidate.relative_to(root).parts
            is_universally_denied = directory.casefold() in {
                item.casefold() for item in UNIVERSAL_DENIED_COMPONENTS
            }
            is_denied_root = len(relative_parts) == 1 and directory.casefold() in {
                item.casefold() for item in DENIED_COMPONENTS
            }
            if is_universally_denied or is_denied_root:
                continue
            retained.append(directory)
        directories[:] = retained
        for filename in sorted(filenames, key=str.casefold):
            yield current_path / filename


def collect_source_files(
    root: str | Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> list[SourceFile]:
    resolved_root = Path(root).expanduser().resolve(strict=True)
    if not resolved_root.is_dir():
        raise ReleasePolicyError(f"source root is not a directory: {resolved_root}")
    if max_file_bytes <= 0 or max_total_bytes <= 0:
        raise ReleasePolicyError("release byte limits must be positive")

    files: list[SourceFile] = []
    seen_casefold: set[str] = set()
    total_bytes = 0
    for source in _walk_candidate_paths(resolved_root):
        relative = source.relative_to(resolved_root).as_posix()
        if not is_public_path(relative):
            continue
        relative = validate_release_path(relative)
        if source.is_symlink():
            raise ReleasePolicyError(f"symlinks are not allowed in source releases: {relative}")
        source_stat = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ReleasePolicyError(f"non-regular file in source release: {relative}")
        if source_stat.st_size > max_file_bytes:
            raise ReleasePolicyError(
                f"source file exceeds {max_file_bytes} bytes: {relative} ({source_stat.st_size})"
            )
        folded = relative.casefold()
        if folded in seen_casefold:
            raise ReleasePolicyError(f"case-insensitive duplicate release path: {relative}")
        seen_casefold.add(folded)
        data = source.read_bytes()
        if len(data) != source_stat.st_size:
            raise ReleasePolicyError(f"file changed while building release: {relative}")
        scan_content(relative, data)
        total_bytes += len(data)
        if total_bytes > max_total_bytes:
            raise ReleasePolicyError(
                f"release exceeds total uncompressed limit of {max_total_bytes} bytes"
            )
        files.append(
            SourceFile(
                path=relative,
                source=source,
                size=len(data),
                sha256=_sha256_bytes(data),
            )
        )
    files.sort(key=lambda item: item.path.casefold())
    missing = sorted(REQUIRED_RELEASE_FILES.difference(item.path for item in files))
    if missing:
        raise ReleasePolicyError(f"required release files are missing: {', '.join(missing)}")
    return files


def source_tree_sha256(files: Iterable[SourceFile]) -> str:
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(item.size).encode("ascii"))
        digest.update(b"\0")
        digest.update(item.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_archive(files: list[SourceFile], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as archive:
            for item in files:
                data = item.source.read_bytes()
                if len(data) != item.size or _sha256_bytes(data) != item.sha256:
                    raise ReleasePolicyError(f"file changed while writing release: {item.path}")
                info = zipfile.ZipInfo(f"{ARCHIVE_PREFIX}/{item.path}", ZIP_TIMESTAMP)
                info.create_system = 3
                executable = item.path.endswith((".sh", ".py")) and item.path.startswith("scripts/")
                mode = 0o755 if executable else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                info.flag_bits |= 0x800
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_manifest(files: list[SourceFile], archive: Path, manifest: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_prefix": ARCHIVE_PREFIX,
        "source_tree_sha256": source_tree_sha256(files),
        "file_count": len(files),
        "total_uncompressed_bytes": sum(item.size for item in files),
        "files": [{"path": item.path, "size": item.size, "sha256": item.sha256} for item in files],
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, manifest)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_root / "dist/tbx-agent-source.zip")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "dist/tbx-agent-source.manifest.json",
    )
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    parser.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-files", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        files = collect_source_files(
            args.root,
            max_file_bytes=args.max_file_bytes,
            max_total_bytes=args.max_total_bytes,
        )
        summary = {
            "status": "valid",
            "dry_run": bool(args.dry_run),
            "file_count": len(files),
            "total_uncompressed_bytes": sum(item.size for item in files),
            "source_tree_sha256": source_tree_sha256(files),
        }
        if args.print_files:
            summary["files"] = [item.path for item in files]
        if not args.dry_run:
            output = args.output.expanduser().resolve()
            manifest = args.manifest.expanduser().resolve()
            build_archive(files, output)
            payload = write_manifest(files, output, manifest)
            summary.update(
                {
                    "archive": str(output),
                    "archive_sha256": payload["archive_sha256"],
                    "manifest": str(manifest),
                }
            )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, ReleasePolicyError, zipfile.BadZipFile) as exc:
        print(
            json.dumps({"status": "rejected", "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
