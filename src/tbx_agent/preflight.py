from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .artifacts import default_artifact_root
from .config import (
    PROJECT_ROOT,
    local_paths_equivalent,
    local_paths_overlap,
    resolve_model_path,
)
from .paths import default_runtime_root, resolve_portable_path
from .vision.rank03 import _sha256

_CONFIG_FILES = {
    "app": ("app.yaml", "yaml"),
    "fusion_policy": ("fusion_policy.json", "json"),
    "fusion_policy_argmax_v2": ("fusion_policy_argmax_v2.json", "json"),
    "fusion_policy_sens98_legacy": ("fusion_policy_sens98_legacy.json", "json"),
    "rank03_runtime": ("rank03_runtime.json", "json"),
    "safety_policy": ("safety_policy.json", "json"),
    "knowledge_ingestion": ("knowledge_ingestion.yaml", "yaml"),
    "retrieval": ("retrieval.yaml", "yaml"),
}
_REQUIRED_KNOWLEDGE_FILES = {
    "source_manifest.json": "json",
    "chunks.jsonl": "jsonl",
    "active_screening_questions.json": "json",
}
_LLAMACPP_EVAL_REQUIRED_METRICS = frozenset(
    {
        "route_accuracy",
        "emergency_recall",
        "citation_coverage_for_guideline_answers",
        "forbidden_output_rate",
        "narration_applied_rate",
        "narration_fallback_rate",
        "narration_safety_rejection_rate",
    }
)
_LLAMACPP_EVAL_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "evaluation_id",
        "hypothesis",
        "seed",
        "selection_use",
        "locked_or_hidden_test_used",
        "clinical_validation",
        "major_variables_changed",
        "narrator",
        "metrics",
    }
)
_OPENAI_COMPAT_CONTRACT_FILENAME = "openai_compat_test.yaml"
_OPENAI_COMPAT_KIND = "llamacpp_openai_python_sdk_compatibility_smoke"
_OPENAI_COMPAT_SDK_VERSION = "2.54.0"
_OPENAI_COMPAT_TESTED_ENDPOINTS = [
    "/v1/models",
    "/v1/chat/completions:sync",
    "/v1/chat/completions:sse",
]
_OPENAI_COMPAT_SOURCE_PATHS = {
    "tbx_agent/llm/openai_compat.py": Path("src/tbx_agent/llm/openai_compat.py"),
    "tbx_agent/llm/runtime_supervisor.py": Path("src/tbx_agent/llm/runtime_supervisor.py"),
    "scripts/test_openai_compatible_api.py": Path("scripts/test_openai_compatible_api.py"),
}
_RUNTIME_RELEASE_EVIDENCE_ARTIFACTS = (
    "evaluation/runtime_operations_ledger.jsonl",
    "evaluation/ledger.jsonl",
    "evaluation/fixtures/narrator_precision_pair_v1.json",
    "docs/cards/qwen35_4b_q4_runtime_card.md",
    "scripts/benchmark_llamacpp_runtime.ps1",
    "scripts/start_llamacpp.ps1",
    "scripts/stop_llamacpp.ps1",
    "scripts/test_llamacpp_recovery.ps1",
    "scripts/run_rank03_llamacpp_coexistence_smoke.py",
    "scripts/run_narrator_precision_pair.py",
    "scripts/record_llamacpp_operator_recovery.py",
)
_UPSTREAM_DFINE_BASE = Path("configs") / "dfine" / "dfine_hgnetv2_l_coco.yml"
_ACTIVE_FUSION_POLICY_ID = "rank03-user-trained-native-argmax-v2"
_ACTIVE_CLASSIFIER_POLICY_ID = "three_class_native_argmax_v1"
_CLASS_PROBABILITY_ORDER = ["healthy", "sick_non_tb", "tb"]
_ACTIVE_CLASS_ROUTES = {
    "healthy": "model_not_flagged",
    "sick_non_tb": "non_tb_abnormal",
    "tb": "model_flagged",
}
_ARGMAX_POLICY_ID = "rank03-agent-screening-demo-cls-argmax-det-advisory-v2"
_LEGACY_POLICY_ID = "rank03-agent-screening-demo-official-val-sens98-v1"
_ARGMAX_POLICY_SNAPSHOT: dict[str, Any] = {
    "schema_version": 2,
    "policy_id": _ARGMAX_POLICY_ID,
    "purpose": "research_demo_screening_only",
    "classifier_policy_id": "convnextT_three_class_argmax_v1",
    "classifier_rule": "native_three_class_argmax",
    "classifier_probability_order": _CLASS_PROBABILITY_ORDER,
    "tie_handling": "stable_native_argmax",
    "classifier_routes": {
        "healthy": "model_not_flagged",
        "sick_non_tb": "non_tb_abnormal",
        "tb": "model_flagged",
    },
    "classifier_sensitivity_on_selection_split": 0.975,
    "classifier_specificity_on_selection_split": 0.999375,
    "detector_role": "advisory_localization_only",
    "quality_warning": "retain_argmax_with_advisory",
    "technical_failure": "technical_failure",
    "selection_dataset": "frozen official TBX11K validation, 1800 images",
    "selection_split_sha256": ("2cadd49c24d4b2b7f3bc4a40a498aa74582acb8efbce50758b106626db5601aa"),
    "analysis_report": "../reports/rank03_sensitivity_specificity_official_val_20260828.md",
    "deployment_contract": "../reports/rank03_deployment_inference_contract_20260828.md",
    "hidden_or_locked_test_used": False,
    "clinical_validation": False,
    "warning": (
        "The native three-class argmax rule and detector advisory outputs were measured on "
        "the same internal validation split and require independent external validation "
        "before any real-world use."
    ),
}
_LEGACY_POLICY_SNAPSHOT: dict[str, Any] = {
    "schema_version": 1,
    "policy_id": _LEGACY_POLICY_ID,
    "purpose": "research_demo_screening_only",
    "classifier_rule": "p_tb_gte_threshold",
    "classifier_threshold": 0.027215289,
    "classifier_target_sensitivity_on_selection_split": 0.98,
    "classifier_specificity_on_selection_split": 0.986875,
    "detector_rule": "max_category_agnostic_score_gte_threshold",
    "detector_threshold": 0.069879934,
    "detector_target_sensitivity_on_selection_split": 0.98,
    "detector_specificity_on_selection_split": 0.98625,
    "agreement_flagged": "model_flagged",
    "agreement_not_flagged": "model_not_flagged",
    "disagreement": "pending_human_review",
    "technical_failure": "technical_failure",
    "selection_dataset": "frozen official TBX11K validation, 1800 images",
    "selection_split_sha256": ("2cadd49c24d4b2b7f3bc4a40a498aa74582acb8efbce50758b106626db5601aa"),
    "analysis_report": "../reports/rank03_sensitivity_specificity_official_val_20260828.md",
    "clinical_validation": False,
    "warning": (
        "Thresholds were selected and measured on the same internal validation split and "
        "require independent external calibration before any real-world use."
    ),
}


class _ReportBuilder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        status: str,
        message: str,
        *,
        path: Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "id": check_id,
            "status": status,
            "message": message,
        }
        if path is not None:
            item["path"] = str(path)
        if details is not None:
            item["details"] = details
        self.checks.append(item)

    def pass_(
        self,
        check_id: str,
        message: str,
        *,
        path: Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.add(check_id, "pass", message, path=path, details=details)

    def fail(
        self,
        check_id: str,
        message: str,
        *,
        path: Path | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.add(check_id, "fail", message, path=path, details=details)

    def skip(self, check_id: str, message: str) -> None:
        self.add(check_id, "skip", message)


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    return resolve_portable_path(value, base=base)


def _read_mapping(path: Path, kind: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = yaml.safe_load(text) if kind == "yaml" else json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("top-level value must be an object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number} top-level value must be an object")
        records.append(record)
    return records


def _config_path(
    key: str,
    *,
    config_dir: Path,
    environ: Mapping[str, str],
) -> Path:
    filename, _kind = _CONFIG_FILES[key]
    if key == "rank03_runtime":
        override = environ.get("TBX_AGENT_RANK03_RUNTIME_CONFIG")
        if override:
            return _resolve_path(override, base=config_dir.parent)
    if key == "retrieval":
        override = environ.get("TBX_AGENT_RETRIEVAL_CONFIG")
        if override:
            return _resolve_path(override, base=config_dir.parent)
    return config_dir / filename


def _load_configs(
    config_dir: Path,
    report: _ReportBuilder,
    *,
    environ: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for key, (_filename, kind) in _CONFIG_FILES.items():
        path = _config_path(key, config_dir=config_dir, environ=environ)
        exists_id = f"config.{key}.exists"
        parse_id = f"config.{key}.parse"
        if not path.is_file():
            report.fail(exists_id, "required configuration file is missing", path=path)
            report.skip(parse_id, "configuration parsing skipped because the file is missing")
            continue
        report.pass_(exists_id, "required configuration file exists", path=path)
        try:
            parsed[key] = _read_mapping(path, kind)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            report.fail(parse_id, f"configuration is not parseable: {exc}", path=path)
        else:
            report.pass_(parse_id, f"configuration parsed as {kind.upper()}", path=path)
    return parsed


def _load_knowledge(
    knowledge_dir: Path,
    report: _ReportBuilder,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    if knowledge_dir.is_dir():
        report.pass_(
            "knowledge.directory.exists",
            "knowledge directory exists",
            path=knowledge_dir,
        )
    else:
        report.fail(
            "knowledge.directory.exists",
            "knowledge directory is missing",
            path=knowledge_dir,
        )

    # This fixed allowlist is deliberate: preflight must never discover or read
    # dataset manifests merely because another JSON file is placed nearby.
    candidates: dict[str, str] = dict(_REQUIRED_KNOWLEDGE_FILES)

    parsed_json: dict[str, dict[str, Any]] = {}
    parsed_jsonl: dict[str, list[dict[str, Any]]] = {}
    for filename in sorted(candidates):
        kind = candidates[filename]
        path = knowledge_dir / filename
        exists_id = f"knowledge.{filename}.exists"
        parse_id = f"knowledge.{filename}.parse"
        if not path.is_file():
            report.fail(exists_id, "required knowledge asset is missing", path=path)
            report.skip(parse_id, "knowledge parsing skipped because the file is missing")
            continue
        report.pass_(exists_id, "knowledge asset exists", path=path)
        try:
            if kind == "jsonl":
                parsed_jsonl[filename] = _read_jsonl(path)
            else:
                parsed_json[filename] = _read_mapping(path, "json")
        except (OSError, UnicodeError, ValueError) as exc:
            report.fail(parse_id, f"knowledge asset is not parseable: {exc}", path=path)
        else:
            report.pass_(parse_id, f"knowledge asset parsed as {kind.upper()}", path=path)

    manifest = parsed_json.get("source_manifest.json")
    chunks = parsed_jsonl.get("chunks.jsonl")
    screening = parsed_json.get("active_screening_questions.json")
    _check_source_governance(manifest, report)
    _check_local_source_assets(manifest, knowledge_dir=knowledge_dir, report=report)
    _check_chunk_sources(manifest, chunks, report)
    _check_active_screening_references(manifest, chunks, screening, report)
    return manifest, chunks


def _check_source_governance(
    manifest: dict[str, Any] | None,
    report: _ReportBuilder,
) -> None:
    check_id = "knowledge.sources.governance_fields"
    if manifest is None:
        report.skip(check_id, "source governance validation skipped because manifest did not parse")
        return
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        report.fail(check_id, "source_manifest.sources must be an array")
        return
    required = {
        "source_id",
        "status",
        "retrievable",
        "full_text_reviewed",
        "allowed_claim_scope",
        "treatment_details_allowed",
    }
    invalid: list[dict[str, Any]] = []
    for index, item in enumerate(sources):
        missing = sorted(required - set(item)) if isinstance(item, dict) else sorted(required)
        wrong_types: list[str] = []
        if isinstance(item, dict):
            for field in ("retrievable", "full_text_reviewed", "treatment_details_allowed"):
                if not isinstance(item.get(field), bool):
                    wrong_types.append(field)
            if not isinstance(item.get("allowed_claim_scope"), list):
                wrong_types.append("allowed_claim_scope")
        if missing or wrong_types:
            invalid.append({"index": index, "missing": missing, "wrong_types": sorted(wrong_types)})
    if invalid:
        report.fail(
            check_id,
            "one or more source records lack explicit governance fields",
            details={"invalid_sources": invalid},
        )
    else:
        report.pass_(
            check_id,
            "every source has explicit retrieval, review, claim-scope, and treatment-detail fields",
            details={"source_count": len(sources)},
        )


def _check_local_source_assets(
    manifest: dict[str, Any] | None,
    *,
    knowledge_dir: Path,
    report: _ReportBuilder,
) -> None:
    check_id = "knowledge.sources.local_assets"
    if manifest is None:
        report.skip(check_id, "local source validation skipped because manifest did not parse")
        return
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        report.fail(check_id, "source_manifest.sources must be an array")
        return

    declared = [
        item
        for item in sources
        if isinstance(item, dict) and ("local_path" in item or "file_sha256" in item)
    ]
    failures: list[dict[str, Any]] = []
    verified: list[dict[str, str]] = []
    knowledge_root = knowledge_dir.resolve()
    for item in declared:
        source_id = item.get("source_id")
        local_path = item.get("local_path")
        expected = item.get("file_sha256")
        if not isinstance(source_id, str):
            failures.append({"source_id": source_id, "reason": "invalid_source_id"})
            continue
        if not isinstance(local_path, str) or not local_path.strip():
            failures.append({"source_id": source_id, "reason": "missing_local_path"})
            continue
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in expected)
        ):
            failures.append({"source_id": source_id, "reason": "invalid_file_sha256"})
            continue

        relative_path = Path(local_path)
        if relative_path.is_absolute():
            failures.append({"source_id": source_id, "reason": "local_path_must_be_relative"})
            continue
        resolved = (knowledge_root / relative_path).resolve()
        try:
            resolved.relative_to(knowledge_root)
        except ValueError:
            failures.append({"source_id": source_id, "reason": "local_path_outside_knowledge_dir"})
            continue
        if not resolved.is_file():
            failures.append(
                {"source_id": source_id, "reason": "local_file_missing", "path": str(resolved)}
            )
            continue
        actual = _sha256(resolved)
        if actual.lower() != expected.lower():
            failures.append(
                {
                    "source_id": source_id,
                    "reason": "sha256_mismatch",
                    "expected_sha256": expected.lower(),
                    "actual_sha256": actual.lower(),
                }
            )
            continue
        verified.append({"source_id": source_id, "path": str(resolved), "sha256": actual.lower()})

    if failures:
        report.fail(
            check_id,
            "one or more declared local guideline assets failed validation",
            details={"failures": failures, "verified": verified},
        )
    else:
        report.pass_(
            check_id,
            "all declared local guideline assets exist within the knowledge directory "
            "and match SHA256",
            details={"declared_count": len(declared), "verified": verified},
        )


def _check_chunk_sources(
    manifest: dict[str, Any] | None,
    chunks: list[dict[str, Any]] | None,
    report: _ReportBuilder,
) -> None:
    check_id = "knowledge.chunks.allowed_sources"
    if manifest is None or chunks is None:
        report.skip(check_id, "source validation skipped because a prerequisite did not parse")
        return
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        report.fail(check_id, "source_manifest.sources must be an array")
        return

    source_statuses: dict[str, str] = {}
    source_retrievable: dict[str, bool] = {}
    malformed_sources: list[int] = []
    duplicate_sources: set[str] = set()
    for index, item in enumerate(sources):
        if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
            malformed_sources.append(index)
            continue
        source_id = item["source_id"]
        if source_id in source_statuses:
            duplicate_sources.add(source_id)
        source_statuses[source_id] = str(item.get("status", ""))
        source_retrievable[source_id] = item.get("retrievable") is True
    if malformed_sources or duplicate_sources:
        report.fail(
            check_id,
            "source manifest contains malformed or duplicate source identities",
            details={
                "malformed_source_indexes": malformed_sources,
                "duplicate_source_ids": sorted(duplicate_sources),
            },
        )
        return

    allowed = {source_id for source_id, retrievable in source_retrievable.items() if retrievable}
    invalid_references: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        source_id = chunk.get("source_id")
        if not isinstance(source_id, str) or source_id not in allowed:
            invalid_references.append(
                {
                    "line": index + 1,
                    "chunk_id": chunk.get("chunk_id"),
                    "source_id": source_id,
                    "manifest_status": source_statuses.get(source_id)
                    if isinstance(source_id, str)
                    else None,
                    "retrievable": source_retrievable.get(source_id)
                    if isinstance(source_id, str)
                    else None,
                }
            )
    if not chunks:
        report.fail(check_id, "chunks.jsonl contains no retrievable chunks")
    elif invalid_references:
        report.fail(
            check_id,
            "one or more chunks reference a source that is not retrievable",
            details={"invalid_references": invalid_references},
        )
    else:
        report.pass_(
            check_id,
            "all chunks reference sources explicitly marked retrievable=true",
            details={"chunk_count": len(chunks), "allowed_source_count": len(allowed)},
        )


def _check_active_screening_references(
    manifest: dict[str, Any] | None,
    chunks: list[dict[str, Any]] | None,
    screening: dict[str, Any] | None,
    report: _ReportBuilder,
) -> None:
    check_id = "knowledge.active_screening.references"
    if manifest is None or chunks is None or screening is None:
        report.skip(
            check_id,
            "screening reference validation skipped because a prerequisite did not parse",
        )
        return
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list):
        report.fail(check_id, "source_manifest.sources must be an array")
        return

    source_ids = {
        item["source_id"]
        for item in manifest_sources
        if isinstance(item, dict) and isinstance(item.get("source_id"), str)
    }
    chunk_by_id: dict[str, dict[str, Any]] = {}
    duplicate_chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str):
            continue
        if chunk_id in chunk_by_id:
            duplicate_chunk_ids.add(chunk_id)
        chunk_by_id[chunk_id] = chunk

    errors: list[dict[str, Any]] = []
    declared_sources = screening.get("sources", {})
    if not isinstance(declared_sources, dict):
        errors.append({"reason": "screening_sources_not_object"})
        declared_sources = {}
    for source_id in declared_sources:
        if source_id not in source_ids:
            errors.append({"reason": "screening_source_not_in_manifest", "source_id": source_id})

    questions = screening.get("questions", [])
    if not isinstance(questions, list):
        errors.append({"reason": "questions_not_array"})
        questions = []
    for question in questions:
        if not isinstance(question, dict):
            errors.append({"reason": "question_not_object"})
            continue
        source_id = question.get("source_id")
        if source_id not in source_ids or source_id not in declared_sources:
            errors.append(
                {
                    "reason": "question_source_not_registered",
                    "question_id": question.get("question_id"),
                    "source_id": source_id,
                }
            )

    citations = screening.get("summary_citations", {})
    if not isinstance(citations, dict):
        errors.append({"reason": "summary_citations_not_object"})
        citations = {}
    citation_fields = (
        "source_id",
        "title",
        "organization",
        "publication_year",
        "section",
        "locator",
        "url",
        "support_text",
    )
    for citation_name, citation in citations.items():
        if not isinstance(citation, dict):
            errors.append({"reason": "summary_citation_not_object", "citation": citation_name})
            continue
        chunk_id = citation.get("chunk_id")
        chunk = chunk_by_id.get(chunk_id) if isinstance(chunk_id, str) else None
        if chunk is None:
            errors.append(
                {
                    "reason": "summary_citation_chunk_missing",
                    "citation": citation_name,
                    "chunk_id": chunk_id,
                }
            )
            continue
        mismatched = [field for field in citation_fields if citation.get(field) != chunk.get(field)]
        if mismatched:
            errors.append(
                {
                    "reason": "summary_citation_chunk_mismatch",
                    "citation": citation_name,
                    "chunk_id": chunk_id,
                    "fields": mismatched,
                }
            )

    if duplicate_chunk_ids:
        errors.append({"reason": "duplicate_chunk_ids", "chunk_ids": sorted(duplicate_chunk_ids)})
    if errors:
        report.fail(
            check_id,
            "active-screening sources or citations are not anchored to the governed snapshot",
            details={"errors": errors},
        )
    else:
        report.pass_(
            check_id,
            "active-screening questions and citations are anchored to registered sources "
            "and chunks",
            details={"question_count": len(questions), "citation_count": len(citations)},
        )


def _check_hidden_or_locked_declaration(
    configs: dict[str, dict[str, Any]],
    report: _ReportBuilder,
) -> None:
    check_id = "fusion.hidden_or_locked_test_unused"
    fusion = configs.get("fusion_policy")
    runtime = configs.get("rank03_runtime")
    if fusion is None or runtime is None:
        report.skip(check_id, "governance declaration skipped because configuration did not parse")
        return

    declarations: dict[str, Any] = {}
    if "hidden_or_locked_test_used" in fusion:
        declarations["fusion_policy.hidden_or_locked_test_used"] = fusion[
            "hidden_or_locked_test_used"
        ]
    validation_scope = runtime.get("validation_scope")
    if isinstance(validation_scope, dict) and "hidden_or_locked_test_used" in validation_scope:
        declarations["rank03_runtime.validation_scope.hidden_or_locked_test_used"] = (
            validation_scope["hidden_or_locked_test_used"]
        )

    if not declarations:
        report.fail(
            check_id,
            "hidden/locked-test use must be explicitly declared as false",
        )
    elif any(value is not False for value in declarations.values()):
        report.fail(
            check_id,
            "a hidden/locked-test declaration is not exactly false",
            details={"declarations": declarations},
        )
    else:
        report.pass_(
            check_id,
            "configuration explicitly declares that hidden/locked tests were not used",
            details={
                "declarations": declarations,
                "selection_dataset": fusion.get("selection_dataset"),
            },
        )


def _check_active_fusion_policy_contract(
    configs: dict[str, dict[str, Any]],
    report: _ReportBuilder,
) -> None:
    """Fail closed when the deployed image-routing contract drifts."""

    check_id = "fusion.active_policy_contract"
    fusion = configs.get("fusion_policy")
    runtime = configs.get("rank03_runtime")
    if fusion is None or runtime is None:
        report.fail(check_id, "active fusion policy or rank03 runtime did not parse")
        return

    classifier = runtime.get("classifier")
    runtime_order = classifier.get("probability_order") if isinstance(classifier, dict) else None
    expected = {
        "schema_version": 5,
        "policy_id": _ACTIVE_FUSION_POLICY_ID,
        "purpose": "research_screening_support_only",
        "classifier_policy_id": _ACTIVE_CLASSIFIER_POLICY_ID,
        "classifier_rule": "native_three_class_argmax",
        "classifier_probability_order": _CLASS_PROBABILITY_ORDER,
        "tie_handling": "stable_native_argmax",
        "classifier_routes": _ACTIVE_CLASS_ROUTES,
        "detector_role": "advisory_localization_only",
        "quality_warning": "retain_argmax_with_advisory",
        "technical_failure": "technical_failure",
        "selection_dataset": None,
        "selection_metrics": None,
        "heldout_metrics": None,
        "hidden_or_locked_test_used": False,
        "clinical_validation": False,
        "performance_claims_inherited": False,
        "warning": (
            "The repository contains inference code only, with no model weights or clinical "
            "performance claim. Install the independently distributed, hash-verified inference "
            "bundle and validate its runtime identity before use."
        ),
    }
    mismatches = {
        key: {"expected": value, "actual": fusion.get(key)}
        for key, value in expected.items()
        if fusion.get(key) != value
    }
    forbidden_fields = sorted(
        field
        for field in (
            "detector_threshold",
            "detector_rule",
            "classifier_threshold",
            "classifier_threshold_comparison",
            "agreement_flagged",
            "agreement_not_flagged",
            "disagreement",
            "selection_seed",
            "selection_split_sha256",
            "source_evaluation_id",
        )
        if field in fusion
    )
    if runtime_order != _CLASS_PROBABILITY_ORDER:
        mismatches["rank03_runtime.classifier.probability_order"] = {
            "expected": _CLASS_PROBABILITY_ORDER,
            "actual": runtime_order,
        }

    if mismatches or forbidden_fields:
        report.fail(
            check_id,
            "active native-argmax/advisory-detector policy contract drifted",
            details={
                "mismatches": mismatches,
                "forbidden_fields_present": forbidden_fields,
            },
        )
        return
    report.pass_(
        check_id,
        "active native three-class argmax and advisory detector policy contract is exact",
        details={
            "policy_id": fusion["policy_id"],
            "classifier_policy_id": fusion["classifier_policy_id"],
            "classifier_rule": fusion["classifier_rule"],
            "performance_claims_inherited": False,
            "clinical_validation": False,
            "probability_order": runtime_order,
        },
    )


def _check_retained_fusion_policy_snapshots(
    configs: dict[str, dict[str, Any]],
    report: _ReportBuilder,
) -> None:
    """Keep prior governed policies explicitly selectable for reproducibility."""

    check_id = "fusion.retained_policy_snapshots"
    argmax = configs.get("fusion_policy_argmax_v2")
    legacy = configs.get("fusion_policy_sens98_legacy")
    expected = {"argmax": _ARGMAX_POLICY_SNAPSHOT, "legacy": _LEGACY_POLICY_SNAPSHOT}
    actual = {"argmax": argmax, "legacy": legacy}
    mismatches: dict[str, dict[str, Any]] = {}
    for name, snapshot in expected.items():
        payload = actual[name]
        if payload == snapshot:
            continue
        keys = set(snapshot)
        if isinstance(payload, dict):
            keys.update(payload)
        changed_fields = sorted(
            field
            for field in keys
            if not isinstance(payload, dict) or payload.get(field) != snapshot.get(field)
        )
        mismatches[name] = {"changed_fields": changed_fields}
    if mismatches:
        report.fail(
            check_id,
            "a retained argmax or legacy policy snapshot drifted",
            details={"mismatches": mismatches},
        )
        return
    report.pass_(
        check_id,
        "argmax v2 and legacy sens98 policy snapshots remain available",
        details={
            "argmax_policy_id": _ARGMAX_POLICY_ID,
            "legacy_policy_id": _LEGACY_POLICY_ID,
        },
    )


def _asset_path(
    configured: Any,
    env_name: str,
    *,
    project_root: Path,
    environ: Mapping[str, str],
) -> Path:
    override = environ.get(env_name)
    value = override if override is not None else configured
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"missing path field and {env_name} override")
    return _resolve_path(value, base=project_root)


def _check_sha_asset(
    report: _ReportBuilder,
    *,
    check_id: str,
    configured_path: Any,
    env_name: str,
    expected_sha256: Any,
    project_root: Path,
    environ: Mapping[str, str],
) -> None:
    try:
        path = _asset_path(
            configured_path,
            env_name,
            project_root=project_root,
            environ=environ,
        )
    except (OSError, ValueError) as exc:
        report.fail(check_id, f"asset path is invalid: {exc}")
        return
    expected = str(expected_sha256).lower() if expected_sha256 is not None else ""
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        report.fail(check_id, "configured SHA256 is not a 64-character hex digest", path=path)
        return
    if not path.is_file():
        report.fail(check_id, "frozen rank03 asset is missing", path=path)
        return
    try:
        actual = _sha256(path)
    except OSError as exc:
        report.fail(check_id, f"frozen rank03 asset could not be hashed: {exc}", path=path)
        return
    if actual.lower() != expected:
        report.fail(
            check_id,
            "frozen rank03 asset SHA256 does not match",
            path=path,
            details={"expected_sha256": expected, "actual_sha256": actual.lower()},
        )
        return
    report.pass_(
        check_id,
        "frozen rank03 asset exists and its SHA256 matches",
        path=path,
        details={"sha256": actual.lower()},
    )


def _check_rank03_assets(
    runtime: dict[str, Any] | None,
    report: _ReportBuilder,
    *,
    project_root: Path,
    environ: Mapping[str, str],
) -> None:
    if runtime is None:
        report.fail("rank03.runtime.available", "rank03 runtime configuration did not parse")
        return
    if runtime.get("template") is True:
        report.fail(
            "rank03.runtime.registered",
            "checked-in rank03 config is a non-runnable template; install the inference bundle "
            "and set TBX_AGENT_RANK03_RUNTIME_CONFIG",
        )
        return
    report.pass_(
        "rank03.runtime.registered",
        "an external or operator-generated rank03 runtime contract is active",
    )
    classifier = runtime.get("classifier")
    detector = runtime.get("detector")
    if not isinstance(classifier, dict) or not isinstance(detector, dict):
        report.fail(
            "rank03.runtime.contract",
            "rank03 runtime must contain classifier and detector objects",
        )
        return
    report.pass_("rank03.runtime.contract", "rank03 asset contract is structurally available")

    asset_specs = (
        (
            "rank03.classifier.checkpoint_sha256",
            classifier.get("checkpoint_path"),
            "TBX_AGENT_CLASSIFIER_CHECKPOINT",
            classifier.get("checkpoint_sha256"),
        ),
        (
            "rank03.classifier.config_sha256",
            classifier.get("config_path"),
            "TBX_AGENT_CLASSIFIER_CONFIG",
            classifier.get("config_sha256"),
        ),
        (
            "rank03.detector.checkpoint_sha256",
            detector.get("checkpoint_path"),
            "TBX_AGENT_DETECTOR_CHECKPOINT",
            detector.get("checkpoint_sha256"),
        ),
        (
            "rank03.detector.resolved_config_sha256",
            detector.get("resolved_config_path"),
            "TBX_AGENT_DETECTOR_CONFIG",
            detector.get("resolved_config_sha256"),
        ),
    )
    for check_id, path_value, env_name, expected_sha256 in asset_specs:
        _check_sha_asset(
            report,
            check_id=check_id,
            configured_path=path_value,
            env_name=env_name,
            expected_sha256=expected_sha256,
            project_root=project_root,
            environ=environ,
        )

    try:
        dfine_root = _asset_path(
            detector.get("source_root"),
            "TBX_AGENT_DFINE_ROOT",
            project_root=project_root,
            environ=environ,
        )
    except (OSError, ValueError) as exc:
        report.fail("rank03.dfine.source_root", f"D-FINE source path is invalid: {exc}")
        dfine_root = None
    if dfine_root is not None:
        if dfine_root.is_dir():
            report.pass_(
                "rank03.dfine.source_root",
                "D-FINE source root exists",
                path=dfine_root,
            )
        else:
            report.fail(
                "rank03.dfine.source_root",
                "D-FINE source root is missing",
                path=dfine_root,
            )
        upstream_base = dfine_root / _UPSTREAM_DFINE_BASE
        if upstream_base.is_file():
            report.pass_(
                "rank03.dfine.upstream_base",
                "D-FINE upstream base configuration exists",
                path=upstream_base,
            )
        else:
            report.fail(
                "rank03.dfine.upstream_base",
                "D-FINE upstream base configuration is missing",
                path=upstream_base,
            )

    try:
        declared_base = _asset_path(
            detector.get("base_config_path"),
            "TBX_AGENT_DFINE_BASE_CONFIG",
            project_root=project_root,
            environ=environ,
        )
    except (OSError, ValueError) as exc:
        report.fail(
            "rank03.dfine.declared_base",
            f"declared D-FINE base path is invalid: {exc}",
        )
    else:
        if declared_base.is_file():
            report.pass_(
                "rank03.dfine.declared_base",
                "declared D-FINE base configuration exists",
                path=declared_base,
            )
        else:
            report.fail(
                "rank03.dfine.declared_base",
                "declared D-FINE base configuration is missing",
                path=declared_base,
            )


def _check_pipeline_configs(
    configs: dict[str, dict[str, Any]],
    report: _ReportBuilder,
    *,
    config_dir: Path,
    environ: Mapping[str, str],
) -> None:
    ingestion_path = config_dir / _CONFIG_FILES["knowledge_ingestion"][0]
    if "knowledge_ingestion" not in configs:
        report.skip(
            "config.knowledge_ingestion.contract",
            "strict ingestion validation skipped because configuration did not parse",
        )
    else:
        try:
            from .ingestion.pipeline import load_config as load_ingestion_config

            ingestion = load_ingestion_config(ingestion_path)
        except Exception as exc:
            report.fail(
                "config.knowledge_ingestion.contract",
                f"strict ingestion contract validation failed: {type(exc).__name__}: {exc}",
                path=ingestion_path,
            )
        else:
            source_failures: list[dict[str, Any]] = []
            for source in ingestion.sources:
                if not source.enabled:
                    continue
                if source.expected_sha256 is None:
                    source_failures.append(
                        {"source_id": source.source_id, "reason": "missing_expected_sha256"}
                    )
                elif not source.input_path.is_file():
                    source_failures.append(
                        {"source_id": source.source_id, "reason": "source_file_missing"}
                    )
                else:
                    actual = hashlib.sha256(source.input_path.read_bytes()).hexdigest()
                    if actual != source.expected_sha256:
                        source_failures.append(
                            {
                                "source_id": source.source_id,
                                "reason": "source_sha256_mismatch",
                                "expected_sha256": source.expected_sha256,
                                "actual_sha256": actual,
                            }
                        )
            details = {
                "pipeline_version": ingestion.pipeline_version,
                "offline_only": ingestion.offline_only,
                "enabled_source_count": sum(source.enabled for source in ingestion.sources),
                "source_failures": source_failures,
            }
            if source_failures:
                report.fail(
                    "config.knowledge_ingestion.contract",
                    "ingestion sources are not present and hash-pinned exactly",
                    path=ingestion_path,
                    details=details,
                )
            else:
                report.pass_(
                    "config.knowledge_ingestion.contract",
                    "ingestion configuration is strict, offline-only and source-hash pinned",
                    path=ingestion_path,
                    details=details,
                )

    retrieval_path = _config_path(
        "retrieval",
        config_dir=config_dir,
        environ=environ,
    )
    if "retrieval" not in configs:
        report.skip(
            "config.retrieval.contract",
            "strict retrieval validation skipped because configuration did not parse",
        )
    else:
        try:
            from .retrieval.config import load_retrieval_config

            retrieval = load_retrieval_config(retrieval_path)
        except Exception as exc:
            report.fail(
                "config.retrieval.contract",
                f"strict retrieval contract validation failed: {type(exc).__name__}: {exc}",
                path=retrieval_path,
            )
        else:
            report.pass_(
                "config.retrieval.contract",
                "retrieval configuration passed strict backend and model-pin validation",
                path=retrieval_path,
                details={
                    "retrieval_version": retrieval.engine.retrieval_version,
                    "dense_enabled": retrieval.dense.enabled,
                    "dense_model_sha256": retrieval.dense.model_sha256,
                    "reranker_enabled": retrieval.reranker.enabled,
                    "reranker_model_sha256": retrieval.reranker.model_sha256,
                    "vector_backend": retrieval.vector_store.backend,
                },
            )


def _check_system_evaluation_suite(
    project_root: Path,
    report: _ReportBuilder,
    *,
    validate_deterministic_candidate: bool = True,
) -> None:
    check_id = "evaluation.system_suite.integrity"
    config_path = project_root / "evaluation" / "system_bench_config.json"
    if not config_path.is_file():
        report.fail(check_id, "system evaluation configuration is missing", path=config_path)
        return
    try:
        from .evaluation.system_bench import load_config, load_suite

        config = load_config(config_path)
        manifest_path = (project_root / config.suite_manifest).resolve()
        manifest_path.relative_to(project_root.resolve())
        manifest, cases = load_suite(manifest_path)
    except Exception as exc:
        report.fail(
            check_id,
            f"system evaluation suite failed closed: {type(exc).__name__}: {exc}",
            path=config_path,
        )
        return
    report.pass_(
        check_id,
        "system evaluation config, manifest, case count, dimensions and case SHA256 match",
        path=manifest_path,
        details={
            "evaluation_id": config.evaluation_id,
            "suite_id": manifest.suite_id,
            "suite_version": manifest.suite_version,
            "case_count": len(cases),
            "cases_sha256": manifest.cases_sha256,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
        },
    )
    candidate_check_id = "evaluation.deterministic_candidate.integrity"
    if not validate_deterministic_candidate:
        report.skip(
            candidate_check_id,
            "deterministic mock candidate is outside the selected real-inference profile",
        )
        return
    candidate_path = manifest_path.parent / "deterministic_mock_candidate.json"
    if not candidate_path.is_file():
        report.skip(
            candidate_check_id,
            "deterministic mock candidate is optional for this deployment profile",
        )
        return
    try:
        from .evaluation.system_bench import _verify_deterministic_candidate_artifacts

        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict) or not candidate:
            raise ValueError("candidate must be a non-empty JSON object")
        _verify_deterministic_candidate_artifacts(project_root, candidate)
    except Exception as exc:
        report.fail(
            candidate_check_id,
            f"deterministic candidate snapshot failed closed: {type(exc).__name__}: {exc}",
            path=candidate_path,
        )
    else:
        report.pass_(
            candidate_check_id,
            "deterministic candidate source tree and small artifacts match SHA256 pins",
            path=candidate_path,
            details={
                "source_tree_sha256": candidate.get("source_tree_sha256"),
                "artifact_count": len(candidate.get("artifacts", {})),
            },
        )


def _effective_text(
    environ: Mapping[str, str],
    env_name: str,
    configured: Any,
    default: str,
) -> str:
    raw = environ.get(env_name, configured if configured is not None else default)
    return str(raw).strip().lower()


def _effective_bool(
    environ: Mapping[str, str],
    env_name: str,
    configured: Any,
) -> bool:
    raw = environ.get(env_name)
    if raw is None:
        return configured is True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _llamacpp_contract_mismatches(
    app_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mismatches: dict[str, dict[str, Any]] = {}
    for key, expected in runtime_contract.items():
        actual = app_contract.get(key)
        matches = (
            local_paths_equivalent(str(actual), str(expected))
            if key == "model_path"
            else str(actual) == str(expected)
        )
        if not matches:
            mismatches[key] = {"app": actual, "runtime_contract": expected}
    return mismatches


def _check_deployment_and_llm(
    configs: dict[str, dict[str, Any]],
    report: _ReportBuilder,
    *,
    config_dir: Path,
    environ: Mapping[str, str],
    vision_backend: str,
) -> tuple[str, str, Path]:
    app = configs.get("app", {})
    deployment = app.get("deployment", {})
    runtime = app.get("runtime", {})
    security = app.get("security", {})
    observability = app.get("observability", {})
    privacy = app.get("privacy", {})
    if not isinstance(deployment, dict):
        deployment = {}
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(security, dict):
        security = {}
    if not isinstance(observability, dict):
        observability = {}
    if not isinstance(privacy, dict):
        privacy = {}
    profile = _effective_text(
        environ,
        "TBX_AGENT_DEPLOYMENT_PROFILE",
        deployment.get("profile"),
        "research",
    )
    narrator = _effective_text(
        environ,
        "TBX_AGENT_NARRATOR_BACKEND",
        runtime.get("narrator_backend"),
        "none",
    )
    require_real_inference = _effective_bool(
        environ,
        "TBX_AGENT_REQUIRE_REAL_INFERENCE",
        runtime.get("require_real_inference"),
    )
    require_llm_inference = _effective_bool(
        environ,
        "TBX_AGENT_REQUIRE_LLM_INFERENCE",
        runtime.get("require_llm_inference"),
    )
    if profile not in {"research", "development", "production"}:
        report.fail(
            "deployment.profile.contract",
            f"unsupported deployment profile: {profile!r}",
        )
    elif profile == "production":
        trusted_proxy = _effective_bool(
            environ,
            "TBX_AGENT_TRUSTED_PROXY_AUTH_ENABLED",
            security.get("trusted_proxy_auth_enabled"),
        )
        trusted_proxy_secret = str(environ.get("TBX_AGENT_TRUSTED_PROXY_HMAC_SECRET", ""))
        failures = []
        if not require_real_inference:
            failures.append("require_real_inference_must_be_enabled")
        if not require_llm_inference:
            failures.append("require_llm_inference_must_be_enabled")
        if vision_backend != "rank03":
            failures.append("vision_backend_must_be_rank03")
        if narrator != "llama_cpp":
            failures.append("narrator_backend_must_be_llama_cpp")
        llama_allow_remote = _effective_bool(
            environ,
            "TBX_AGENT_LLAMA_CPP_ALLOW_REMOTE",
            runtime.get("llama_cpp_allow_remote"),
        )
        llama_base_url = _effective_text(
            environ,
            "LLAMA_CPP_BASE_URL",
            runtime.get("llama_cpp_base_url"),
            "http://127.0.0.1:11435",
        )
        if llama_allow_remote or urlparse(llama_base_url).hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            failures.append("llama_cpp_must_be_loopback_only")
        if not trusted_proxy:
            failures.append("trusted_proxy_auth_must_be_enabled")
        if runtime.get("retain_uploaded_image", True) is not False:
            failures.append("uploaded_image_retention_requires_governed_storage")
        privacy_defaults = {
            "allow_real_name": False,
            "allow_raw_phi_in_external_tracing": False,
            "allow_case_content_in_user_memory": False,
        }
        privacy_drift = sorted(
            field
            for field, required in privacy_defaults.items()
            if privacy.get(field) is not required
        )
        if privacy_drift:
            failures.append("production_privacy_defaults_missing_or_weakened")
        if len(trusted_proxy_secret.encode("utf-8")) < 32 or (
            trusted_proxy_secret.strip().lower()
            in {"change-me", "changeme", "secret", "password", "test"}
        ):
            failures.append("trusted_proxy_hmac_secret_missing_or_weak")
        try:
            replay_window = int(
                environ.get(
                    "TBX_AGENT_TRUSTED_PROXY_REPLAY_WINDOW_SECONDS",
                    security.get("trusted_proxy_replay_window_seconds", 60),
                )
            )
            max_upload = int(runtime.get("max_upload_bytes", 20 * 1024 * 1024))
            max_body = int(
                environ.get(
                    "TBX_AGENT_MAX_REQUEST_BODY_BYTES",
                    security.get("max_request_body_bytes", max_upload + 2 * 1024 * 1024),
                )
            )
            max_concurrent = int(
                environ.get(
                    "TBX_AGENT_MAX_CONCURRENT_REQUESTS",
                    security.get("max_concurrent_requests", 16),
                )
            )
            rate_per_minute = int(
                environ.get(
                    "TBX_AGENT_RATE_LIMIT_REQUESTS_PER_MINUTE",
                    security.get("rate_limit_requests_per_minute", 120),
                )
            )
            rate_burst = int(
                environ.get(
                    "TBX_AGENT_RATE_LIMIT_BURST",
                    security.get("rate_limit_burst", 30),
                )
            )
        except (TypeError, ValueError):
            failures.append("production_numeric_security_setting_invalid")
        else:
            if not 5 <= replay_window <= 300:
                failures.append("trusted_proxy_replay_window_out_of_range")
            if max_body <= max_upload:
                failures.append("request_body_limit_must_exceed_upload_limit")
            if max_concurrent < 1:
                failures.append("max_concurrent_requests_invalid")
            if rate_per_minute < 1 or rate_burst < 1:
                failures.append("rate_limit_invalid")
        metrics_enabled = _effective_bool(
            environ,
            "TBX_AGENT_METRICS_ENABLED",
            observability.get("metrics_enabled"),
        )
        metrics_loopback = _effective_bool(
            environ,
            "TBX_AGENT_METRICS_ALLOW_LOOPBACK",
            observability.get("metrics_allow_loopback", True),
        )
        metrics_token = str(environ.get("TBX_AGENT_METRICS_ADMIN_TOKEN", ""))
        if metrics_enabled and not metrics_loopback and len(metrics_token.encode("utf-8")) < 32:
            failures.append("metrics_admin_guard_missing")
        if failures:
            report.fail(
                "deployment.profile.contract",
                "production profile prerequisites are incomplete",
                details={"profile": profile, "failures": failures},
            )
        else:
            report.pass_(
                "deployment.profile.contract",
                "production profile selects rank03, direct llama.cpp and trusted identity",
                details={"profile": profile},
            )
    else:
        inference_failures = []
        if require_real_inference and vision_backend != "rank03":
            inference_failures.append("required_real_inference_backend_mismatch")
        if require_llm_inference and narrator != "llama_cpp":
            inference_failures.append("required_llm_inference_backend_mismatch")
        if inference_failures:
            report.fail(
                "deployment.profile.contract",
                "required inference flags conflict with the selected runtime backends",
                details={"profile": profile, "failures": inference_failures},
            )
        else:
            report.pass_(
                "deployment.profile.contract",
                "non-production profile and required inference contract are explicit",
                details={
                    "profile": profile,
                    "narrator_backend": narrator,
                    "require_real_inference": require_real_inference,
                    "require_llm_inference": require_llm_inference,
                },
            )

    runtime_config_path = _resolve_path(
        environ.get("TBX_AGENT_LLM_RUNTIME_CONFIG", config_dir / "llm_runtime.yaml"),
        base=config_dir,
    )
    if narrator != "llama_cpp":
        if runtime_config_path.is_file():
            try:
                from .llm.runtime_supervisor import load_runtime_config

                llm_config = load_runtime_config(runtime_config_path, environment=environ)
            except Exception as exc:
                report.fail(
                    "llm.runtime.contract",
                    f"present llama.cpp contract is invalid: {type(exc).__name__}: {exc}",
                    path=runtime_config_path,
                )
            else:
                report.pass_(
                    "llm.runtime.contract",
                    "llama.cpp launch contract parses but is not active in this profile",
                    path=runtime_config_path,
                    details={
                        "active": False,
                        "runtime_id": llm_config.runtime_id,
                        "runtime_config_sha256": llm_config.canonical_sha256(),
                    },
                )
        else:
            report.skip(
                "llm.runtime.contract",
                "llama.cpp is inactive and no runtime contract is present",
            )
        return profile, narrator, runtime_config_path

    if not runtime_config_path.is_file():
        report.fail(
            "llm.runtime.contract",
            "active llama.cpp narrator requires configs/llm_runtime.yaml",
            path=runtime_config_path,
        )
        return profile, narrator, runtime_config_path
    try:
        from .llm.runtime_supervisor import load_runtime_config, verify_runtime_assets

        llm_config = load_runtime_config(runtime_config_path, environment=environ)
        attestation = verify_runtime_assets(llm_config)
        app_contract = {
            "model_alias": runtime.get("llama_cpp_model_alias"),
            "model_path": str(
                resolve_model_path(
                    str(runtime.get("llama_cpp_model_path", "")),
                    "LLAMA_CPP_MODEL_PATH",
                    project_root=config_dir.parent,
                    environment=environ,
                )
            ),
            "model_sha256": runtime.get("llama_cpp_model_sha256"),
            "server_build": runtime.get("llama_cpp_server_build"),
        }
        expected_contract = {
            "model_alias": llm_config.model_alias,
            "model_path": str(llm_config.model_path),
            "model_sha256": llm_config.model_sha256,
            "server_build": llm_config.server_build,
        }
        mismatches = _llamacpp_contract_mismatches(app_contract, expected_contract)
        if mismatches:
            raise ValueError(f"app/runtime llama.cpp contract mismatch: {mismatches}")
    except Exception as exc:
        report.fail(
            "llm.runtime.contract",
            f"active llama.cpp assets failed integrity validation: {type(exc).__name__}: {exc}",
            path=runtime_config_path,
        )
    else:
        report.pass_(
            "llm.runtime.contract",
            "active llama.cpp model, binary and complete bundle match frozen SHA256 values",
            path=runtime_config_path,
            details=attestation,
        )
    return profile, narrator, runtime_config_path


def _check_llamacpp_evaluation_contract(
    *,
    project_root: Path,
    runtime_config_path: Path,
    narrator_backend: str,
    required: bool,
    report: _ReportBuilder,
) -> Path:
    """Validate the non-clinical llama.cpp evaluation plan and its runtime pin.

    The plan is optional when llama.cpp is inactive.  Once present it is treated as a
    governed artifact, and an active llama.cpp narrator cannot pass preflight without it.
    """

    evaluation_path = project_root / "evaluation" / "llamacpp_eval_config.json"
    active = narrator_backend == "llama_cpp"
    if not required:
        report.skip(
            "llm.evaluation.contract",
            "maintainer evaluation receipt checks are disabled in runtime preflight mode",
        )
        return evaluation_path
    if not evaluation_path.is_file():
        if active:
            report.fail(
                "llm.evaluation.contract",
                "active llama.cpp narrator requires a pinned non-clinical evaluation contract",
                path=evaluation_path,
                details={"active": True, "reason": "missing"},
            )
        else:
            report.skip(
                "llm.evaluation.contract",
                "llama.cpp evaluation contract is absent while the narrator is inactive",
            )
        return evaluation_path

    try:
        payload = _read_mapping(evaluation_path, "json")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        report.fail(
            "llm.evaluation.contract",
            f"llama.cpp evaluation contract is not parseable: {exc}",
            path=evaluation_path,
            details={"active": active},
        )
        return evaluation_path

    mismatches: dict[str, Any] = {}
    observed_fields = frozenset(payload)
    if observed_fields != _LLAMACPP_EVAL_TOP_LEVEL_FIELDS:
        mismatches["top_level_fields"] = {
            "missing": sorted(_LLAMACPP_EVAL_TOP_LEVEL_FIELDS - observed_fields),
            "unknown": sorted(observed_fields - _LLAMACPP_EVAL_TOP_LEVEL_FIELDS),
        }
    if payload.get("schema_version") != 1:
        mismatches["schema_version"] = {"expected": 1, "actual": payload.get("schema_version")}
    for field in (
        "selection_use",
        "locked_or_hidden_test_used",
        "clinical_validation",
    ):
        if payload.get(field) is not False:
            mismatches[field] = {"expected": False, "actual": payload.get(field)}
    for field in ("evaluation_id", "hypothesis"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            mismatches[field] = {"expected": "non-empty string", "actual": value}
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        mismatches["seed"] = {"expected": "integer", "actual": seed}

    variables = payload.get("major_variables_changed")
    if (
        not isinstance(variables, list)
        or not 1 <= len(variables) <= 2
        or any(not isinstance(value, str) or not value.strip() for value in variables)
        or len(set(variables)) != len(variables)
    ):
        mismatches["major_variables_changed"] = {
            "expected": "one or two unique non-empty strings",
            "actual": variables,
        }

    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or any(
        not isinstance(value, str) or not value.strip() for value in metrics
    ):
        mismatches["metrics"] = {
            "expected": sorted(_LLAMACPP_EVAL_REQUIRED_METRICS),
            "actual": metrics,
        }
    else:
        metric_set = set(metrics)
        missing_metrics = sorted(_LLAMACPP_EVAL_REQUIRED_METRICS - metric_set)
        if missing_metrics or len(metric_set) != len(metrics):
            mismatches["metrics"] = {
                "missing_required": missing_metrics,
                "duplicates_present": len(metric_set) != len(metrics),
            }

    narrator = payload.get("narrator")
    expected_narrator_fields = {
        "backend",
        "runtime_config",
        "runtime_config_sha256",
    }
    if not isinstance(narrator, dict):
        mismatches["narrator"] = {"expected": "object", "actual": narrator}
    else:
        narrator_fields = set(narrator)
        if narrator_fields != expected_narrator_fields:
            mismatches["narrator.fields"] = {
                "missing": sorted(expected_narrator_fields - narrator_fields),
                "unknown": sorted(narrator_fields - expected_narrator_fields),
            }
        if narrator.get("backend") != "llama_cpp":
            mismatches["narrator.backend"] = {
                "expected": "llama_cpp",
                "actual": narrator.get("backend"),
            }
        declared_runtime = narrator.get("runtime_config")
        if not isinstance(declared_runtime, str) or not declared_runtime.strip():
            mismatches["narrator.runtime_config"] = {
                "expected": str(runtime_config_path),
                "actual": declared_runtime,
            }
        else:
            resolved_runtime = _resolve_path(declared_runtime, base=project_root)
            if resolved_runtime != runtime_config_path.resolve():
                mismatches["narrator.runtime_config"] = {
                    "expected": str(runtime_config_path.resolve()),
                    "actual": str(resolved_runtime),
                }
        if not runtime_config_path.is_file():
            mismatches["narrator.runtime_config_sha256"] = {
                "expected": "existing pinned runtime configuration",
                "actual": "missing",
            }
        else:
            runtime_sha256 = hashlib.sha256(runtime_config_path.read_bytes()).hexdigest()
            if narrator.get("runtime_config_sha256") != runtime_sha256:
                mismatches["narrator.runtime_config_sha256"] = {
                    "expected": runtime_sha256,
                    "actual": narrator.get("runtime_config_sha256"),
                }

    if mismatches:
        report.fail(
            "llm.evaluation.contract",
            "llama.cpp evaluation contract failed closed",
            path=evaluation_path,
            details={"active": active, "mismatches": mismatches},
        )
    else:
        report.pass_(
            "llm.evaluation.contract",
            "llama.cpp evaluation contract is non-clinical and pinned to the runtime bytes",
            path=evaluation_path,
            details={
                "active": active,
                "evaluation_id": payload["evaluation_id"],
                "clinical_validation": False,
                "selection_use": False,
                "locked_or_hidden_test_used": False,
                "runtime_config_sha256": hashlib.sha256(
                    runtime_config_path.read_bytes()
                ).hexdigest(),
                "required_metrics": sorted(_LLAMACPP_EVAL_REQUIRED_METRICS),
            },
        )
    return evaluation_path


_MISSING = object()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nested_value(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _strictly_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if expected is None:
        return actual is None
    return actual == expected


def _expect_value(
    failures: list[dict[str, Any]],
    value: Mapping[str, Any],
    dotted_path: str,
    expected: Any,
    *,
    reason: str,
) -> None:
    if not _strictly_equal(_nested_value(value, dotted_path), expected):
        failures.append(
            {
                "reason": reason,
                "field": dotted_path,
                "expected": expected,
            }
        )


def _openai_source_binding(project_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    missing: list[str] = []
    file_digests: dict[str, str] = {}
    for receipt_name, relative_path in _OPENAI_COMPAT_SOURCE_PATHS.items():
        source_path = project_root / relative_path
        if not source_path.is_file():
            missing.append(relative_path.as_posix())
            continue
        file_digests[receipt_name] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if missing:
        return None, sorted(missing)
    source_tree_sha256 = hashlib.sha256(
        json.dumps(file_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return (
        {
            "algorithm": "sha256(canonical relative-path to file-sha256 map)",
            "files": file_digests,
            "source_tree_sha256": source_tree_sha256,
        },
        [],
    )


def _load_openai_receipt(
    path: Path,
    *,
    expected_sha256: Any,
    label: str,
    failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not _is_sha256(expected_sha256):
        failures.append({"reason": f"{label}_declared_sha256_invalid"})
    if not path.is_file():
        failures.append({"reason": f"{label}_missing", "path": str(path)})
        return None
    if path.stat().st_size > 1024 * 1024:
        failures.append({"reason": f"{label}_exceeds_size_limit", "path": str(path)})
        return None
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        failures.append(
            {
                "reason": f"{label}_sha256_mismatch",
                "path": str(path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            }
        )
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failures.append(
            {
                "reason": f"{label}_not_parseable",
                "error_type": type(exc).__name__,
            }
        )
        return None
    if not isinstance(receipt, dict):
        failures.append({"reason": f"{label}_not_object"})
        return None
    return receipt


def _validate_openai_receipt_common(
    receipt: dict[str, Any],
    *,
    label: str,
    runtime_config: Any,
    runtime_raw_sha256: str,
    runtime_canonical_sha256: str,
    contract_live_smoke: dict[str, Any],
    source_binding: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    expected_values = {
        "schema_version": 1,
        "kind": _OPENAI_COMPAT_KIND,
        "seed": contract_live_smoke.get("seed"),
        "split_hash": None,
        "selection_use": False,
        "model_selection": False,
        "threshold_selection": False,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "clinical_validation": False,
        "major_variables_changed": [],
        "source_revision": contract_live_smoke.get("source_revision"),
        "source_tree_sha256": contract_live_smoke.get("source_tree_sha256"),
        "full_configuration.runtime_id": runtime_config.runtime_id,
        "full_configuration.runtime_config_file_sha256": runtime_raw_sha256,
        "full_configuration.runtime_config_canonical_sha256": runtime_canonical_sha256,
        "full_configuration.server_build": runtime_config.server_build,
        "full_configuration.model_alias": runtime_config.model_alias,
        "full_configuration.model_sha256": runtime_config.model_sha256,
        "full_configuration.openai_base_url": (
            f"http://{runtime_config.host}:{runtime_config.port}/v1"
        ),
        "full_configuration.openai_sdk_version": _OPENAI_COMPAT_SDK_VERSION,
        "full_configuration.openai_sdk_max_retries": 0,
        "full_configuration.http_client_trust_env": False,
        "full_configuration.http_client_follow_redirects": False,
        "full_configuration.tested_endpoints": _OPENAI_COMPAT_TESTED_ENDPOINTS,
        "full_configuration.request_timeout_seconds": runtime_config.request_timeout_seconds,
        "full_configuration.temperature": 0,
        "full_configuration.max_output_tokens": 16,
        "full_configuration.stream_include_usage": True,
        "full_configuration.chat_template_enable_thinking": False,
        "full_configuration.fixture_kind": "fixed_synthetic_non_medical_protocol_prompt",
        "full_configuration.credential_source": "runtime_acl_key_file",
        "full_configuration.credential_persisted_in_receipt": False,
        "full_configuration.dataset_used": False,
        "full_configuration.seed": runtime_config.seed,
        "full_configuration.runtime_asset_hashes_reverified": False,
        "full_configuration.served_process_binary_attested": False,
        "source_binding": source_binding,
        "full_configuration.smoke_module_sha256": source_binding["files"][
            "tbx_agent/llm/openai_compat.py"
        ],
        "full_configuration.smoke_entrypoint_sha256": source_binding["files"][
            "scripts/test_openai_compatible_api.py"
        ],
    }
    for field, expected in expected_values.items():
        _expect_value(
            failures,
            receipt,
            field,
            expected,
            reason=f"{label}_field_mismatch",
        )
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("openai-compat-"):
        failures.append({"reason": f"{label}_run_id_invalid"})
    for field in ("full_configuration.prompt_sha256", "full_configuration.expected_content_sha256"):
        if not _is_sha256(_nested_value(receipt, field)):
            failures.append({"reason": f"{label}_field_invalid", "field": field})


def _validate_openai_success_receipt(
    receipt: dict[str, Any],
    *,
    runtime_config: Any,
    runtime_raw_sha256: str,
    runtime_canonical_sha256: str,
    contract_live_smoke: dict[str, Any],
    source_binding: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    _validate_openai_receipt_common(
        receipt,
        label="passed_result",
        runtime_config=runtime_config,
        runtime_raw_sha256=runtime_raw_sha256,
        runtime_canonical_sha256=runtime_canonical_sha256,
        contract_live_smoke=contract_live_smoke,
        source_binding=source_binding,
        failures=failures,
    )
    _expect_value(
        failures,
        receipt,
        "status",
        "passed_openai_compat_smoke",
        reason="passed_result_status_mismatch",
    )
    _expect_value(
        failures,
        receipt,
        "failure",
        None,
        reason="passed_result_failure_not_null",
    )
    for metric in (
        "models_list_success",
        "expected_alias_present",
        "chat_sync_success",
        "chat_sync_exact_content",
        "chat_stream_success",
        "chat_stream_exact_content",
    ):
        _expect_value(
            failures,
            receipt,
            f"metrics.{metric}",
            True,
            reason="passed_result_metric_not_true",
        )
    _expect_value(
        failures,
        receipt,
        "metrics.stream_usage_chunk_count",
        1,
        reason="passed_result_usage_chunk_count_mismatch",
    )
    event_sequence = _nested_value(receipt, "metrics.stream_event_sequence")
    if not isinstance(event_sequence, list) or event_sequence[-2:] != ["finish", "usage"]:
        failures.append({"reason": "passed_result_stream_terminal_sequence_invalid"})
    for field in (
        "metrics.chat_sync_total_tokens",
        "metrics.chat_stream_total_tokens",
        "metrics.stream_chunk_count",
    ):
        value = _nested_value(receipt, field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            failures.append({"reason": "passed_result_metric_invalid", "field": field})
    for arm in ("chat_sync", "chat_stream"):
        prompt_tokens = _nested_value(receipt, f"metrics.{arm}_prompt_tokens")
        completion_tokens = _nested_value(receipt, f"metrics.{arm}_completion_tokens")
        total_tokens = _nested_value(receipt, f"metrics.{arm}_total_tokens")
        token_values = (prompt_tokens, completion_tokens, total_tokens)
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in token_values)
            or prompt_tokens <= 0
            or completion_tokens <= 0
            or total_tokens != prompt_tokens + completion_tokens
        ):
            failures.append(
                {
                    "reason": "passed_result_usage_invalid",
                    "field": f"metrics.{arm}",
                }
            )
    total_milliseconds = _nested_value(receipt, "runtime.total_milliseconds")
    if (
        isinstance(total_milliseconds, bool)
        or not isinstance(total_milliseconds, (int, float))
        or total_milliseconds <= 0
    ):
        failures.append({"reason": "passed_result_runtime_invalid"})
    _expect_value(
        failures,
        receipt,
        "peak_vram_bytes",
        None,
        reason="passed_result_peak_vram_must_be_unavailable",
    )
    _expect_value(
        failures,
        receipt,
        "peak_vram_measurement",
        _nested_value(contract_live_smoke, "passed_result.peak_vram_measurement"),
        reason="passed_result_peak_vram_measurement_mismatch",
    )


def _validate_openai_failed_receipt(
    receipt: dict[str, Any],
    *,
    runtime_config: Any,
    runtime_raw_sha256: str,
    runtime_canonical_sha256: str,
    contract_live_smoke: dict[str, Any],
    source_binding: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    _validate_openai_receipt_common(
        receipt,
        label="retained_failed_result",
        runtime_config=runtime_config,
        runtime_raw_sha256=runtime_raw_sha256,
        runtime_canonical_sha256=runtime_canonical_sha256,
        contract_live_smoke=contract_live_smoke,
        source_binding=source_binding,
        failures=failures,
    )
    expected_failure = contract_live_smoke.get("retained_failed_result", {})
    _expect_value(
        failures,
        receipt,
        "status",
        "failed_retained",
        reason="retained_failed_result_status_mismatch",
    )
    _expect_value(
        failures,
        receipt,
        "failure.stage",
        expected_failure.get("failure_stage"),
        reason="retained_failed_result_stage_mismatch",
    )
    _expect_value(
        failures,
        receipt,
        "failure.error_type",
        expected_failure.get("failure_type"),
        reason="retained_failed_result_type_mismatch",
    )
    if not isinstance(receipt.get("failure"), dict):
        failures.append({"reason": "retained_failed_result_failure_missing"})
    failure_reason = _nested_value(receipt, "failure.reason")
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        failures.append({"reason": "retained_failed_result_reason_missing"})
    for metric in (
        "models_list_success",
        "expected_alias_present",
        "chat_sync_success",
        "chat_sync_exact_content",
        "chat_stream_success",
        "chat_stream_exact_content",
    ):
        _expect_value(
            failures,
            receipt,
            f"metrics.{metric}",
            False,
            reason="retained_failed_result_metric_not_false",
        )
    _expect_value(
        failures,
        receipt,
        "metrics.stream_usage_chunk_count",
        0,
        reason="retained_failed_result_usage_count_mismatch",
    )
    for field in (
        "runtime.models_list_milliseconds",
        "runtime.chat_sync_milliseconds",
        "runtime.chat_stream_milliseconds",
    ):
        _expect_value(
            failures,
            receipt,
            field,
            None,
            reason="retained_failed_result_runtime_stage_not_null",
        )


def _validate_openai_contract(
    contract: dict[str, Any],
    *,
    runtime_config: Any,
    runtime_config_path: Path,
    source_binding: dict[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    runtime_raw_sha256 = hashlib.sha256(runtime_config_path.read_bytes()).hexdigest()
    runtime_canonical_sha256 = runtime_config.canonical_sha256()
    origin = f"http://{runtime_config.host}:{runtime_config.port}"
    openai_base_url = f"{origin}/v1"
    required_values = {
        "schema_version": 1,
        "contract_id": "qwen35-4b-q4-k-m-llamacpp-openai-protocol-test-v1",
        "status": "protocol_test_only",
        "source_runtime_config": "configs/llm_runtime.yaml",
        "governance.selection_use": False,
        "governance.locked_or_hidden_test_used": False,
        "governance.clinical_validation": False,
        "governance.clinical_authority": False,
        "governance.agent_output": False,
        "governance.intended_use": (
            "loopback_raw_model_protocol_testing_with_bearer_protected_generation"
        ),
        "governance.prohibited_claims": [
            "tbx_agent_response",
            "diagnosis_or_exclusion",
            "treatment_recommendation",
            "clinical_validation",
        ],
        "transport.implementation": "native_llama_server",
        "transport.listen_origin": origin,
        "transport.api_prefix": "/v1",
        "transport.openai_base_url": openai_base_url,
        "transport.loopback_only": True,
        "transport.public_network_exposure_allowed": False,
        "transport.tls_terminated_here": False,
        "transport.remote_access.allowed_without_gateway": False,
        "transport.remote_access.gateway_required": True,
        "transport.remote_access.gateway_implemented": False,
        "transport.remote_access.gateway_tls_required": True,
        "transport.remote_access.gateway_authentication": "independent_bearer_token",
        "transport.remote_access.internal_llama_key_reuse_allowed": False,
        "authentication.models_endpoint.scheme": "none",
        "authentication.models_endpoint.native_public_endpoint": True,
        "authentication.models_endpoint.fake_bearer_observed_http_status": 200,
        "authentication.models_endpoint.must_not_be_used_as_authentication_probe": True,
        "authentication.models_endpoint.risk_mitigation": "loopback_only",
        "authentication.generation_endpoint.scheme": "bearer",
        "authentication.generation_endpoint.invalid_bearer_observed_http_status": 401,
        "authentication.generation_endpoint.credential_purpose": (
            "internal_llama_server_loopback_only"
        ),
        "authentication.key_value_in_config_allowed": False,
        "authentication.key_value_in_logs_allowed": False,
        "authentication.key_value_in_command_line_allowed": False,
        "authentication.key_value_in_source_control_allowed": False,
        "model.public_alias": runtime_config.model_alias,
        "model.upstream_alias": runtime_config.model_alias,
        "model.model_sha256": runtime_config.model_sha256,
        "model.engine": runtime_config.engine,
        "model.server_build": runtime_config.server_build,
        "model.modality": "text_only",
        "model.thinking": False,
        "model.clinical_authority": False,
        "protocol.dialect": "openai_chat_completions",
        "protocol.endpoints.models.method": "GET",
        "protocol.endpoints.models.path": "/v1/models",
        "protocol.endpoints.chat_completions.method": "POST",
        "protocol.endpoints.chat_completions.path": "/v1/chat/completions",
        "protocol.endpoints.chat_completions.synchronous_json": True,
        "protocol.endpoints.chat_completions.streaming_sse": True,
        "protocol.endpoints.chat_completions.terminal_sse_marker": "[DONE]",
        "protocol.supported_message_roles": ["system", "user", "assistant"],
        "protocol.message_content": "plain_text_only",
        "protocol.generation.temperature": 0,
        "protocol.generation.enable_thinking": False,
        "protocol.generation.maximum_input_tokens": runtime_config.max_input_tokens,
        "protocol.generation.maximum_output_tokens": runtime_config.max_output_tokens,
        "protocol.generation.parallel_slots": runtime_config.parallel_slots,
        "separation_boundary.raw_model_api_base_url": openai_base_url,
        "separation_boundary.tbx_agent_api_base_url": "http://127.0.0.1:8000",
        "separation_boundary.raw_output_enters_agent_memory": False,
        "separation_boundary.raw_output_enters_case_or_report": False,
        "separation_boundary.raw_output_passes_agent_safety_verifier": False,
        "live_smoke.command": (
            "python scripts/test_openai_compatible_api.py --runtime-config configs/llm_runtime.yaml"
        ),
        "live_smoke.evidence_status": (
            "passed_engineering_protocol_smoke_with_failed_attempt_retained"
        ),
        "live_smoke.publication_or_release_evidence": False,
        "live_smoke.clinical_validation": False,
        "live_smoke.seed": runtime_config.seed,
        "live_smoke.split_hash": None,
        "live_smoke.source_tree_sha256": source_binding["source_tree_sha256"],
        "live_smoke.runtime_config_file_sha256": runtime_raw_sha256,
        "live_smoke.runtime_config_canonical_sha256": runtime_canonical_sha256,
        "live_smoke.sdk.version": _OPENAI_COMPAT_SDK_VERSION,
        "live_smoke.sdk.max_retries": 0,
        "live_smoke.sdk.http_client_trust_env": False,
        "live_smoke.sdk.http_client_follow_redirects": False,
        "live_smoke.passed_result.status": "passed_openai_compat_smoke",
        "live_smoke.passed_result.models_list_success": True,
        "live_smoke.passed_result.chat_sync_success": True,
        "live_smoke.passed_result.chat_stream_success": True,
        "live_smoke.passed_result.stream_terminal_usage_observed": True,
        "live_smoke.passed_result.peak_vram_bytes": None,
        "live_smoke.retained_failed_result.status": "failed_retained",
        "live_smoke.retained_failed_result.failure_stage": "load_acl_key",
        "live_smoke.retained_failed_result.failure_type": "PermissionError",
    }
    for field, expected in required_values.items():
        _expect_value(
            failures,
            contract,
            field,
            expected,
            reason="contract_runtime_mismatch",
        )
    unsupported = _nested_value(contract, "protocol.unsupported_capabilities")
    required_unsupported = {
        "agent_tools",
        "function_calling",
        "image_or_multimodal_input",
        "embeddings",
        "openai_responses_api",
    }
    if not isinstance(unsupported, list) or not required_unsupported.issubset(set(unsupported)):
        failures.append({"reason": "contract_unsupported_capabilities_incomplete"})
    if runtime_config.host != "127.0.0.1":
        failures.append({"reason": "runtime_not_ipv4_loopback"})
    if runtime_config.enable_thinking is not False:
        failures.append({"reason": "runtime_thinking_not_disabled"})
    if runtime_config.load_mmproj is not False:
        failures.append({"reason": "runtime_not_text_only"})
    if runtime_config.clinical_authority is not False:
        failures.append({"reason": "runtime_clinical_authority_not_false"})
    source_revision = _nested_value(contract, "live_smoke.source_revision")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        failures.append({"reason": "contract_source_revision_invalid"})
    live_smoke = contract.get("live_smoke")
    return (
        live_smoke if isinstance(live_smoke, dict) else {},
        runtime_raw_sha256,
        runtime_canonical_sha256,
    )


def _check_openai_compat_protocol(
    *,
    project_root: Path,
    config_dir: Path,
    runtime_config_path: Path,
    deployment_profile: str,
    report: _ReportBuilder,
    passed_result_override: str | Path | None,
    failed_result_override: str | Path | None,
    required: bool,
) -> list[Path]:
    contract_path = config_dir / _OPENAI_COMPAT_CONTRACT_FILENAME
    if not contract_path.is_file():
        report.skip(
            "llm.openai_compat.protocol",
            "optional local OpenAI protocol contract is absent",
        )
        return []
    inventory_paths = [contract_path]
    if not required:
        report.skip(
            "llm.openai_compat.protocol",
            "retained protocol receipts are disabled in runtime preflight mode",
        )
        return inventory_paths
    try:
        contract = _read_mapping(contract_path, "yaml")
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        report.fail(
            "llm.openai_compat.protocol",
            "local OpenAI protocol contract is not parseable",
            path=contract_path,
            details={
                "profile": deployment_profile,
                "engineering_protocol_only": True,
                "clinical_validation": False,
                "agent_output": False,
                "release_authorized": False,
                "deployment_safety_decision": "unchanged_no_go",
                "failures": [
                    {
                        "reason": "contract_not_parseable",
                        "error_type": type(exc).__name__,
                    }
                ],
            },
        )
        return inventory_paths
    declared_runtime = contract.get("source_runtime_config")
    if isinstance(declared_runtime, str):
        declared_runtime_path = _resolve_path(declared_runtime, base=project_root)
    else:
        declared_runtime_path = runtime_config_path
    failures: list[dict[str, Any]] = []
    if declared_runtime_path != runtime_config_path.resolve():
        failures.append({"reason": "contract_runtime_path_mismatch"})
    if not runtime_config_path.is_file():
        failures.append({"reason": "runtime_config_missing"})
        runtime_config = None
    else:
        try:
            from .llm.runtime_supervisor import load_runtime_config

            runtime_config = load_runtime_config(runtime_config_path)
        except Exception as exc:
            failures.append(
                {
                    "reason": "runtime_config_invalid",
                    "error_type": type(exc).__name__,
                }
            )
            runtime_config = None
    source_binding, missing_sources = _openai_source_binding(project_root)
    if missing_sources:
        failures.append(
            {
                "reason": "openai_compat_source_missing",
                "missing": missing_sources,
            }
        )
    if runtime_config is None or source_binding is None:
        report.fail(
            "llm.openai_compat.protocol",
            "local OpenAI engineering protocol evidence failed closed",
            path=contract_path,
            details={
                "profile": deployment_profile,
                "engineering_protocol_only": True,
                "clinical_validation": False,
                "agent_output": False,
                "release_authorized": False,
                "deployment_safety_decision": "unchanged_no_go",
                "failures": failures,
            },
        )
        return inventory_paths

    live_smoke, runtime_raw_sha256, runtime_canonical_sha256 = _validate_openai_contract(
        contract,
        runtime_config=runtime_config,
        runtime_config_path=runtime_config_path,
        source_binding=source_binding,
        failures=failures,
    )
    passed_declaration = live_smoke.get("passed_result", {})
    failed_declaration = live_smoke.get("retained_failed_result", {})
    passed_declared_path = passed_declaration.get("path")
    failed_declared_path = failed_declaration.get("path")
    if passed_result_override is not None:
        passed_path = _resolve_path(passed_result_override, base=project_root)
    elif isinstance(passed_declared_path, str):
        passed_path = _resolve_path(passed_declared_path, base=project_root)
    else:
        passed_path = None
        failures.append({"reason": "passed_result_path_invalid"})
    if failed_result_override is not None:
        failed_path = _resolve_path(failed_result_override, base=project_root)
    elif isinstance(failed_declared_path, str):
        failed_path = _resolve_path(failed_declared_path, base=project_root)
    else:
        failed_path = None
        failures.append({"reason": "retained_failed_result_path_invalid"})
    if passed_path is not None:
        inventory_paths.append(passed_path)
    if failed_path is not None:
        inventory_paths.append(failed_path)
    if passed_path is not None and failed_path is not None and passed_path == failed_path:
        failures.append({"reason": "passed_and_failed_result_paths_not_distinct"})

    passed_receipt = (
        _load_openai_receipt(
            passed_path,
            expected_sha256=passed_declaration.get("sha256"),
            label="passed_result",
            failures=failures,
        )
        if passed_path is not None
        else None
    )
    failed_receipt = (
        _load_openai_receipt(
            failed_path,
            expected_sha256=failed_declaration.get("sha256"),
            label="retained_failed_result",
            failures=failures,
        )
        if failed_path is not None
        else None
    )
    if passed_receipt is not None:
        _validate_openai_success_receipt(
            passed_receipt,
            runtime_config=runtime_config,
            runtime_raw_sha256=runtime_raw_sha256,
            runtime_canonical_sha256=runtime_canonical_sha256,
            contract_live_smoke=live_smoke,
            source_binding=source_binding,
            failures=failures,
        )
    if failed_receipt is not None:
        _validate_openai_failed_receipt(
            failed_receipt,
            runtime_config=runtime_config,
            runtime_raw_sha256=runtime_raw_sha256,
            runtime_canonical_sha256=runtime_canonical_sha256,
            contract_live_smoke=live_smoke,
            source_binding=source_binding,
            failures=failures,
        )
    if passed_receipt is not None and failed_receipt is not None:
        try:
            passed_created = datetime.fromisoformat(str(passed_receipt["created_at"]))
            failed_created = datetime.fromisoformat(str(failed_receipt["created_at"]))
        except (KeyError, TypeError, ValueError):
            failures.append({"reason": "receipt_created_at_invalid"})
        else:
            if failed_created >= passed_created:
                failures.append({"reason": "retained_failure_not_before_success"})

    details = {
        "profile": deployment_profile,
        "engineering_protocol_only": True,
        "clinical_validation": False,
        "agent_output": False,
        "release_authorized": False,
        "deployment_safety_decision": "unchanged_no_go",
        "contract_id": contract.get("contract_id"),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "runtime_config_file_sha256": runtime_raw_sha256,
        "runtime_config_canonical_sha256": runtime_canonical_sha256,
        "model_alias": runtime_config.model_alias,
        "model_sha256": runtime_config.model_sha256,
        "server_build": runtime_config.server_build,
        "source_tree_sha256": source_binding["source_tree_sha256"],
        "sdk_version": _OPENAI_COMPAT_SDK_VERSION,
        "tested_endpoints": list(_OPENAI_COMPAT_TESTED_ENDPOINTS),
        "publication_or_release_evidence": False,
        "passed_result_path": str(passed_path) if passed_path is not None else None,
        "passed_result_sha256": passed_declaration.get("sha256"),
        "passed_run_id": passed_receipt.get("run_id") if passed_receipt else None,
        "retained_failed_result_path": str(failed_path) if failed_path is not None else None,
        "retained_failed_result_sha256": failed_declaration.get("sha256"),
        "retained_failed_run_id": failed_receipt.get("run_id") if failed_receipt else None,
        "failures": failures,
    }
    if failures:
        report.fail(
            "llm.openai_compat.protocol",
            "local OpenAI engineering protocol evidence failed closed",
            path=contract_path,
            details=details,
        )
    else:
        report.pass_(
            "llm.openai_compat.protocol",
            "local OpenAI SDK protocol and retained failure receipts are intact",
            path=contract_path,
            details=details,
        )
    return inventory_paths


def _runtime_evidence_data_root(project_root: Path, environ: Mapping[str, str]) -> Path:
    return _resolve_path(default_runtime_root(environ), base=project_root)


def _false_declaration_errors(
    record: dict[str, Any],
    full_configuration: dict[str, Any],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for field in fields:
        declarations = []
        if field in record:
            declarations.append({"scope": "record", "value": record[field]})
        if field in full_configuration:
            declarations.append({"scope": "full_configuration", "value": full_configuration[field]})
        if not declarations or any(item["value"] is not False for item in declarations):
            failures.append(
                {
                    "reason": "governance_declaration_not_explicitly_false",
                    "field": field,
                    "declarations": declarations,
                }
            )
    return failures


def _result_receipt_errors(
    record: dict[str, Any],
    *,
    data_root: Path,
    result_required: bool,
) -> list[dict[str, Any]]:
    result_path_value = record.get("result_path")
    result_sha256 = record.get("result_sha256")
    if result_path_value is None and result_sha256 is None and not result_required:
        return []
    if not isinstance(result_path_value, str) or not result_path_value.strip():
        return [{"reason": "result_path_missing"}]
    if (
        not isinstance(result_sha256, str)
        or len(result_sha256) != 64
        or any(character not in "0123456789abcdef" for character in result_sha256)
    ):
        return [{"reason": "result_sha256_invalid", "actual": result_sha256}]
    result_path = Path(result_path_value).expanduser().resolve()
    try:
        result_path.relative_to(data_root.resolve())
    except ValueError:
        return [
            {
                "reason": "result_path_outside_data_root",
                "result_path": str(result_path),
                "data_root": str(data_root.resolve()),
            }
        ]
    if not result_path.is_file():
        return [{"reason": "result_file_missing", "result_path": str(result_path)}]
    actual_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if actual_sha256 != result_sha256:
        return [
            {
                "reason": "result_sha256_mismatch",
                "result_path": str(result_path),
                "expected": result_sha256,
                "actual": actual_sha256,
            }
        ]
    return []


def _runtime_record_errors(
    record: dict[str, Any],
    *,
    index: int,
    runtime_raw_sha256: str,
    runtime_canonical_sha256: str,
    runtime_id: str,
    model_sha256: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    full = record.get("full_configuration")
    if not isinstance(full, dict):
        return [{"record": index, "reason": "full_configuration_missing"}]
    failures.extend(
        {"record": index, **failure}
        for failure in _false_declaration_errors(
            record,
            full,
            (
                "selection_use",
                "locked_or_hidden_test_used",
                "clinical_validation",
            ),
        )
    )
    for field in ("official_hidden_test_used", "quantization_selection_use"):
        if field in record and record[field] is not False:
            failures.append(
                {
                    "record": index,
                    "reason": "governance_declaration_not_explicitly_false",
                    "field": field,
                    "declarations": [{"scope": "record", "value": record[field]}],
                }
            )
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        failures.append({"record": index, "reason": "seed_missing_or_invalid"})
    source_revision = record.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or not source_revision.strip()
        or source_revision == "unknown"
    ):
        failures.append({"record": index, "reason": "source_revision_missing"})
    variables = full.get("major_variables_changed", record.get("major_variables_changed"))
    if not isinstance(variables, list) or len(variables) > 2:
        failures.append(
            {"record": index, "reason": "major_variables_exceed_two", "actual": variables}
        )

    runtime_hashes = {
        field: full[field]
        for field in (
            "runtime_config_file_sha256",
            "runtime_config_canonical_sha256",
            "runtime_config_sha256",
            "llama_runtime_file_sha256",
            "production_runtime_config_file_sha256",
            "production_runtime_config_canonical_sha256",
        )
        if field in full
    }
    allowed_runtime_hashes = {runtime_raw_sha256, runtime_canonical_sha256}
    if not runtime_hashes or not any(
        value in allowed_runtime_hashes for value in runtime_hashes.values()
    ):
        failures.append(
            {
                "record": index,
                "reason": "runtime_config_sha256_missing_or_mismatched",
                "actual": runtime_hashes,
            }
        )
    model_hashes = {
        field: full[field]
        for field in (
            "model_sha256",
            "llama_model_sha256",
            "q4_model_sha256",
        )
        if field in full
    }
    if model_sha256 not in model_hashes.values():
        failures.append(
            {
                "record": index,
                "reason": "model_sha256_missing_or_mismatched",
                "actual": model_hashes,
            }
        )
    production_q4 = record.get("production_q4")
    production_before = production_q4.get("before") if isinstance(production_q4, dict) else None
    observed_runtime_id = full.get("runtime_id") or full.get("production_runtime_id")
    if observed_runtime_id is None and isinstance(production_before, dict):
        observed_runtime_id = production_before.get("runtime_id")
    production_after_harness = record.get("production_q4_after_harness_termination")
    if observed_runtime_id is None and isinstance(production_after_harness, dict):
        observed_runtime_id = production_after_harness.get("runtime_id")
    if observed_runtime_id != runtime_id:
        failures.append(
            {
                "record": index,
                "reason": "runtime_id_missing_or_mismatched",
                "expected": runtime_id,
                "actual": observed_runtime_id,
            }
        )

    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        failures.append({"record": index, "reason": "metrics_missing"})
        metrics = {}
    peak_value = metrics.get("peak_vram_bytes", record.get("peak_vram_bytes"))
    peak_method = (
        metrics.get("peak_vram_measurement")
        or metrics.get("vram_measurement")
        or record.get("peak_vram_measurement")
    )
    vram_measurement = record.get("vram_measurement")
    if peak_method is None and isinstance(vram_measurement, dict):
        peak_method = vram_measurement.get("process_vram") or vram_measurement.get("interpretation")
    peak_is_numeric = (
        not isinstance(peak_value, bool)
        and isinstance(peak_value, (int, float))
        and peak_value >= 0
    )
    peak_is_explicitly_unavailable = (
        peak_value is None and isinstance(peak_method, str) and bool(peak_method.strip())
    )
    if not peak_is_numeric and not peak_is_explicitly_unavailable:
        failures.append(
            {
                "record": index,
                "reason": "peak_vram_not_measured_or_explicitly_unavailable",
                "value": peak_value,
                "method": peak_method,
            }
        )

    status = record.get("status")
    successful = isinstance(status, str) and status.startswith("passed")
    retained_failure = isinstance(status, str) and ("failed" in status or "regressed" in status)
    if not successful and not retained_failure:
        failures.append({"record": index, "reason": "runtime_status_invalid", "actual": status})
    if retained_failure and not isinstance(record.get("failure"), dict):
        failures.append({"record": index, "reason": "retained_failure_payload_missing"})
    failures.extend(
        {"record": index, **failure}
        for failure in _result_receipt_errors(
            record,
            data_root=data_root,
            result_required=successful,
        )
    )
    return failures


def _coexistence_record_errors(
    record: dict[str, Any],
    *,
    index: int,
    runtime_raw_sha256: str,
    runtime_id: str,
    model_sha256: str,
    data_root: Path,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    full = record.get("full_configuration")
    if not isinstance(full, dict):
        return [{"record": index, "reason": "full_configuration_missing"}]
    failures.extend(
        {"record": index, **failure}
        for failure in _false_declaration_errors(
            record,
            full,
            (
                "selection_use",
                "locked_or_hidden_test_used",
                "clinical_validation",
            ),
        )
    )
    if record.get("official_hidden_test_used") is not False:
        failures.append({"record": index, "reason": "official_hidden_test_not_explicitly_false"})
    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        failures.append({"record": index, "reason": "seed_missing_or_invalid"})
    source_revision = record.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or not source_revision.strip()
        or source_revision == "unknown"
    ):
        failures.append({"record": index, "reason": "source_revision_missing"})
    if full.get("llama_runtime_file_sha256") != runtime_raw_sha256:
        failures.append({"record": index, "reason": "runtime_config_sha256_mismatched"})
    if full.get("llama_model_sha256") != model_sha256:
        failures.append({"record": index, "reason": "model_sha256_mismatched"})
    metrics = record.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        failures.append({"record": index, "reason": "metrics_missing"})
    else:
        if metrics.get("oom_event_count") != 0:
            failures.append({"record": index, "reason": "oom_event_observed"})
        if metrics.get("llama_cpp_healthy_after_rank03") is not True:
            failures.append({"record": index, "reason": "llama_cpp_not_healthy_after_rank03"})
        if (
            not isinstance(metrics.get("rank03_inference_success_count"), int)
            or metrics["rank03_inference_success_count"] < 1
        ):
            failures.append({"record": index, "reason": "rank03_inference_success_missing"})
    peak_value = record.get("peak_vram_bytes")
    if isinstance(peak_value, bool) or not isinstance(peak_value, (int, float)) or peak_value < 0:
        failures.append({"record": index, "reason": "peak_vram_invalid"})
    if record.get("status") != "passed_operational_smoke":
        failures.append(
            {
                "record": index,
                "reason": "coexistence_status_not_passed",
                "actual": record.get("status"),
            }
        )
    if full.get("llama_runtime_id") != runtime_id:
        failures.append({"record": index, "reason": "runtime_id_mismatched"})
    failures.extend(
        {"record": index, **failure}
        for failure in _result_receipt_errors(
            record,
            data_root=data_root,
            result_required=True,
        )
    )
    return failures


def _check_runtime_release_evidence_semantics(
    *,
    project_root: Path,
    runtime_config_path: Path,
    data_root: Path,
    report: _ReportBuilder,
) -> None:
    runtime_ledger_path = project_root / "evaluation" / "runtime_operations_ledger.jsonl"
    evaluation_ledger_path = project_root / "evaluation" / "ledger.jsonl"
    try:
        from .llm.runtime_supervisor import load_runtime_config

        runtime_config = load_runtime_config(runtime_config_path)
        runtime_raw_sha256 = hashlib.sha256(runtime_config_path.read_bytes()).hexdigest()
        runtime_canonical_sha256 = runtime_config.canonical_sha256()
        runtime_records = _read_jsonl(runtime_ledger_path)
        evaluation_records = _read_jsonl(evaluation_ledger_path)
    except Exception as exc:
        report.fail(
            "release.runtime_evidence.semantic",
            f"runtime evidence could not be parsed: {type(exc).__name__}: {exc}",
        )
        return

    failures: list[dict[str, Any]] = []
    for index, record in enumerate(runtime_records, 1):
        failures.extend(
            _runtime_record_errors(
                record,
                index=index,
                runtime_raw_sha256=runtime_raw_sha256,
                runtime_canonical_sha256=runtime_canonical_sha256,
                runtime_id=runtime_config.runtime_id,
                model_sha256=runtime_config.model_sha256,
                data_root=data_root,
            )
        )
    precision_pair_kinds = {
        "qwen35_bf16_q4_constrained_narrator_paired_regression",
        "qwen35_bf16_q4_constrained_narrator_pair_harness_failure",
    }
    precision_pair_records = [
        (index, record)
        for index, record in enumerate(evaluation_records, 1)
        if record.get("kind") in precision_pair_kinds
    ]
    for index, record in precision_pair_records:
        failures.extend(
            _runtime_record_errors(
                record,
                index=index,
                runtime_raw_sha256=runtime_raw_sha256,
                runtime_canonical_sha256=runtime_canonical_sha256,
                runtime_id=runtime_config.runtime_id,
                model_sha256=runtime_config.model_sha256,
                data_root=data_root,
            )
        )

    restore_incident_present = any(
        record.get("status") == "failed_restore_requires_operator" for record in runtime_records
    ) or any(
        record.get("status") == "failed_restore_requires_operator"
        for _index, record in precision_pair_records
    )
    required_runtime_records = {
        "failed_preflight_retained": any(
            record.get("status") == "failed_preflight_retained" for record in runtime_records
        ),
        "runtime_benchmark_passed": any(
            record.get("kind") == "llamacpp_runtime_benchmark" and record.get("status") == "passed"
            for record in runtime_records
        ),
        "recovery_benchmark_passed": any(
            record.get("kind") == "llamacpp_recovery_benchmark" and record.get("status") == "passed"
            for record in runtime_records
        ),
        "operator_recovery_after_restore_incident": (
            not restore_incident_present
            or any(
                record.get("kind") == "llamacpp_operator_recovery_receipt"
                and isinstance(record.get("status"), str)
                and record["status"].startswith("passed")
                for record in runtime_records
            )
        ),
    }
    missing_runtime_records = sorted(
        key for key, present in required_runtime_records.items() if not present
    )
    if missing_runtime_records:
        failures.append(
            {
                "reason": "required_runtime_records_missing",
                "missing": missing_runtime_records,
            }
        )

    coexistence_records = [
        (index, record)
        for index, record in enumerate(evaluation_records, 1)
        if record.get("kind") == "rank03_llamacpp_co_resident_operational_smoke"
    ]
    if not coexistence_records:
        failures.append({"reason": "rank03_llamacpp_coexistence_record_missing"})
    for index, record in coexistence_records:
        failures.extend(
            _coexistence_record_errors(
                record,
                index=index,
                runtime_raw_sha256=runtime_raw_sha256,
                runtime_id=runtime_config.runtime_id,
                model_sha256=runtime_config.model_sha256,
                data_root=data_root,
            )
        )

    if failures:
        report.fail(
            "release.runtime_evidence.semantic",
            "runtime release evidence failed semantic or external receipt validation",
            details={
                "runtime_record_count": len(runtime_records),
                "coexistence_record_count": len(coexistence_records),
                "precision_pair_record_count": len(precision_pair_records),
                "failures": failures,
            },
        )
    else:
        report.pass_(
            "release.runtime_evidence.semantic",
            (
                "retained failure, runtime, recovery, precision-pair and rank03 "
                "coexistence receipts are intact"
            ),
            details={
                "runtime_record_count": len(runtime_records),
                "coexistence_record_count": len(coexistence_records),
                "precision_pair_record_count": len(precision_pair_records),
                "runtime_config_file_sha256": runtime_raw_sha256,
                "runtime_config_canonical_sha256": runtime_canonical_sha256,
                "model_sha256": runtime_config.model_sha256,
                "data_root": str(data_root.resolve()),
                "clinical_validation": False,
            },
        )


def _check_runtime_release_evidence(
    *,
    project_root: Path,
    runtime_config_path: Path,
    narrator_backend: str,
    environ: Mapping[str, str],
    required: bool,
    report: _ReportBuilder,
) -> list[Path]:
    """Require a complete, hash-addressed runtime evidence pack when llama.cpp is active."""

    paths = [project_root / relative for relative in _RUNTIME_RELEASE_EVIDENCE_ARTIFACTS]
    if not required:
        report.skip(
            "release.runtime_evidence.inventory",
            "maintainer runtime evidence is disabled in runtime preflight mode",
        )
        report.skip(
            "release.runtime_evidence.semantic",
            "maintainer runtime evidence semantics are disabled in runtime preflight mode",
        )
        return []
    present = [path for path in paths if path.is_file()]
    active = narrator_backend == "llama_cpp"
    if not active and not present:
        report.skip(
            "release.runtime_evidence.inventory",
            "runtime release evidence is absent while llama.cpp is inactive",
        )
        report.skip(
            "release.runtime_evidence.semantic",
            "runtime evidence semantics are inactive with the llama.cpp narrator",
        )
        return []

    missing = [path.relative_to(project_root).as_posix() for path in paths if not path.is_file()]
    empty = [
        path.relative_to(project_root).as_posix() for path in present if path.stat().st_size == 0
    ]
    records = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in present
        if path.stat().st_size > 0
    ]
    if missing or empty:
        report.fail(
            "release.runtime_evidence.inventory",
            "runtime release evidence pack is incomplete",
            details={
                "active": active,
                "missing": missing,
                "empty": empty,
                "artifacts": records,
            },
        )
        if active:
            report.skip(
                "release.runtime_evidence.semantic",
                "semantic validation skipped because the active evidence pack is incomplete",
            )
    else:
        report.pass_(
            "release.runtime_evidence.inventory",
            "runtime benchmark, recovery and rank03 coexistence evidence is present and hashed",
            details={"active": active, "artifacts": records},
        )
        if active:
            _check_runtime_release_evidence_semantics(
                project_root=project_root,
                runtime_config_path=runtime_config_path,
                data_root=_runtime_evidence_data_root(project_root, environ),
                report=report,
            )
        else:
            report.skip(
                "release.runtime_evidence.semantic",
                "runtime evidence is hashed but semantic release validation is inactive",
            )
    # A partial inactive pack is included so the final inventory cannot hide what was
    # present, while missing paths only become inventory requirements in active mode.
    return paths if active else present


def _small_artifact_inventory(paths: list[Path]) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": str(path.resolve()),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return records, fingerprint


def _source_revision(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _finalize_report(
    report: _ReportBuilder,
    *,
    backend: str,
    project_root: Path,
    config_dir: Path,
    knowledge_dir: Path,
    deployment_profile: str,
    narrator_backend: str,
    artifact_inventory: list[dict[str, Any]],
    artifact_inventory_sha256: str,
    wall_clock_ms: float,
) -> dict[str, Any]:
    counts = {
        status: sum(item["status"] == status for item in report.checks)
        for status in ("pass", "fail", "skip")
    }
    ok = counts["fail"] == 0
    created_at = datetime.now(UTC).isoformat()
    run_material = {
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "backend": backend,
        "created_at": created_at,
        "deployment_profile": deployment_profile,
        "narrator_backend": narrator_backend,
        "source_revision": _source_revision(project_root),
    }
    run_id = (
        "preflight-"
        + hashlib.sha256(
            json.dumps(run_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
    )
    return {
        "schema_version": 1,
        "preflight_id": "tbx-agent-read-only-preflight-v1",
        "run_id": run_id,
        "created_at": created_at,
        "source_revision": run_material["source_revision"],
        "ok": ok,
        "exit_code": 0 if ok else 2,
        "mode": {
            "vision_backend": backend,
            "real_inference": backend == "rank03",
            "inference_mode": (
                "frozen_rank03_real_inference"
                if backend == "rank03"
                else "synthetic_mock_non_real_inference"
                if backend == "mock"
                else "unsupported"
            ),
        },
        "guardrails": {
            "read_only": True,
            "weights_loaded": False,
            "dataset_or_test_manifests_read": False,
            "model_assets_hashed_only_when_rank03": True,
            "declared_local_guideline_assets_hashed": True,
            "small_runtime_contracts_hashed": True,
            "secrets_reported": False,
        },
        "deployment": {
            "profile": deployment_profile,
            "narrator_backend": narrator_backend,
            "technical_profile_contract_passed": deployment_profile == "production" and ok,
            "release_authorized": False,
        },
        "paths": {
            "project_root": str(project_root),
            "config_dir": str(config_dir),
            "knowledge_dir": str(knowledge_dir),
        },
        "summary": counts,
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": artifact_inventory_sha256,
        "runtime": {
            "wall_clock_ms": wall_clock_ms,
            "wall_clock_scope": "read_only_preflight_checks_only",
            "peak_vram_mib": None,
            "peak_vram_measured": False,
            "peak_vram_measurement_method": "unknown_not_measured",
        },
        "checks": report.checks,
    }


def run_preflight(
    *,
    project_root: str | Path | None = None,
    config_dir: str | Path | None = None,
    knowledge_dir: str | Path | None = None,
    vision_backend: str | None = None,
    environ: Mapping[str, str] | None = None,
    openai_compat_result: str | Path | None = None,
    openai_compat_failed_result: str | Path | None = None,
) -> dict[str, Any]:
    """Validate deploy-time assets without importing a model or reading dataset metadata."""

    started = time.perf_counter()
    env = os.environ if environ is None else environ
    root = _resolve_path(project_root or PROJECT_ROOT, base=Path.cwd())
    config_value = config_dir or env.get("TBX_AGENT_CONFIG_DIR") or root / "configs"
    knowledge_value = knowledge_dir or env.get("TBX_AGENT_KNOWLEDGE_DIR") or root / "knowledge"
    resolved_config_dir = _resolve_path(config_value, base=root)
    resolved_knowledge_dir = _resolve_path(knowledge_value, base=root)
    report = _ReportBuilder()

    configs = _load_configs(resolved_config_dir, report, environ=env)
    app_runtime = configs.get("app", {}).get("runtime", {})
    configured_backend = (
        app_runtime.get("vision_backend") if isinstance(app_runtime, dict) else None
    )
    selected_backend = (
        vision_backend or env.get("TBX_AGENT_VISION_BACKEND") or configured_backend or "mock"
    )
    backend = str(selected_backend).strip()
    _load_knowledge(resolved_knowledge_dir, report)
    _check_pipeline_configs(
        configs,
        report,
        config_dir=resolved_config_dir,
        environ=env,
    )
    _check_system_evaluation_suite(
        root,
        report,
        validate_deterministic_candidate=backend == "mock",
    )
    _check_hidden_or_locked_declaration(configs, report)
    _check_active_fusion_policy_contract(configs, report)
    _check_retained_fusion_policy_snapshots(configs, report)

    if backend == "mock":
        report.pass_(
            "vision.backend",
            "mock backend selected: outputs are synthetic and are not real model inference",
            details={"real_inference": False},
        )
    elif backend == "rank03":
        report.pass_(
            "vision.backend",
            "frozen rank03 backend selected; assets will be hashed but weights will not be loaded",
            details={"real_inference": True},
        )
        _check_rank03_assets(
            configs.get("rank03_runtime"),
            report,
            project_root=root,
            environ=env,
        )
    else:
        report.fail(
            "vision.backend",
            f"unsupported vision backend: {backend!r}",
            details={"supported_backends": ["mock", "rank03"]},
        )

    deployment_profile, narrator_backend, llm_runtime_config_path = _check_deployment_and_llm(
        configs,
        report,
        config_dir=resolved_config_dir,
        environ=env,
        vision_backend=backend,
    )
    deployment_config = configs.get("app", {}).get("deployment", {})
    configured_preflight_mode = (
        deployment_config.get("preflight_mode", "release")
        if isinstance(deployment_config, Mapping)
        else "release"
    )
    preflight_mode = str(
        env.get("TBX_AGENT_PREFLIGHT_MODE", configured_preflight_mode)
    ).strip().lower()
    if preflight_mode not in {"runtime", "release"}:
        report.fail(
            "preflight.mode",
            "TBX_AGENT_PREFLIGHT_MODE must be runtime or release",
            details={"actual": preflight_mode},
        )
        preflight_mode = "release"
    else:
        report.pass_(
            "preflight.mode",
            f"{preflight_mode} preflight checks selected",
            details={"release_evidence_required": preflight_mode == "release"},
        )
    release_evidence_required = preflight_mode == "release"

    model_artifact_root = default_artifact_root(env).expanduser().resolve(strict=False)
    data_root = default_runtime_root(env).expanduser().resolve(strict=False)
    case_artifact_value = env.get(
        "TBX_AGENT_CASE_ARTIFACT_ROOT",
        env.get("TBX_AGENT_ARTIFACT_ROOT", str(data_root / "cases")),
    )
    case_artifact_root = _resolve_path(case_artifact_value, base=root)
    if local_paths_overlap(case_artifact_root, model_artifact_root):
        report.fail(
            "storage.artifact_roots.disjoint",
            "mutable case artifacts must not overlap the immutable model cache",
            details={
                "case_artifact_root": str(case_artifact_root),
                "model_artifact_root": str(model_artifact_root),
            },
        )
    else:
        report.pass_(
            "storage.artifact_roots.disjoint",
            "mutable case artifacts and immutable model cache are disjoint",
            details={
                "case_artifact_root": str(case_artifact_root),
                "model_artifact_root": str(model_artifact_root),
            },
        )

    llamacpp_evaluation_path = _check_llamacpp_evaluation_contract(
        project_root=root,
        runtime_config_path=llm_runtime_config_path,
        narrator_backend=narrator_backend,
        required=release_evidence_required,
        report=report,
    )
    openai_compat_paths = _check_openai_compat_protocol(
        project_root=root,
        config_dir=resolved_config_dir,
        runtime_config_path=llm_runtime_config_path,
        deployment_profile=deployment_profile,
        report=report,
        passed_result_override=(openai_compat_result or env.get("TBX_AGENT_OPENAI_COMPAT_RESULT")),
        failed_result_override=(
            openai_compat_failed_result or env.get("TBX_AGENT_OPENAI_COMPAT_FAILED_RESULT")
        ),
        required=release_evidence_required,
    )
    runtime_release_evidence_paths = _check_runtime_release_evidence(
        project_root=root,
        runtime_config_path=llm_runtime_config_path,
        narrator_backend=narrator_backend,
        environ=env,
        required=release_evidence_required,
        report=report,
    )
    system_evaluation_paths = [
        root / "evaluation" / "system_bench_config.json",
        root / "evaluation" / "system_bench_config_v1_1.json",
        root / "evaluation" / "system_bench_config_v1_2.json",
        root / "evaluation" / "system_bench_config_v1_3.json",
        root / "evaluation" / "system_bench_config_v1_4.json",
        root / "evaluation" / "system_bench_config_v1_5.json",
        root / "evaluation" / "system_bench_config_v1_6.json",
        root / "evaluation" / "suites" / "system_v1" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1" / "cases.jsonl",
        root / "evaluation" / "suites" / "system_v1_2" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1_2" / "cases.jsonl",
        root / "evaluation" / "suites" / "system_v1_3" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1_3" / "cases.jsonl",
        root / "evaluation" / "suites" / "system_v1_4" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1_4" / "cases.jsonl",
        root / "evaluation" / "suites" / "system_v1_5" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1_5" / "cases.jsonl",
        root / "evaluation" / "suites" / "system_v1_6" / "manifest.json",
        root / "evaluation" / "suites" / "system_v1_6" / "cases.jsonl",
    ]
    inventory_paths = [
        root / "pyproject.toml",
        *sorted((root / "src" / "tbx_agent").rglob("*.py")),
        *(
            _config_path(key, config_dir=resolved_config_dir, environ=env)
            for key in _CONFIG_FILES
        ),
        *(resolved_knowledge_dir / filename for filename in _REQUIRED_KNOWLEDGE_FILES),
        *system_evaluation_paths,
    ]
    if narrator_backend == "llama_cpp" or llm_runtime_config_path.is_file():
        inventory_paths.append(llm_runtime_config_path)
    if narrator_backend == "llama_cpp" or llamacpp_evaluation_path.is_file():
        inventory_paths.append(llamacpp_evaluation_path)
    inventory_paths.extend(openai_compat_paths)
    inventory_paths.extend(runtime_release_evidence_paths)
    deterministic_candidate_path = (
        root / "evaluation" / "suites" / "system_v1_6" / "deterministic_mock_candidate.json"
    )
    if deterministic_candidate_path.is_file():
        inventory_paths.append(deterministic_candidate_path)
    missing_inventory_paths = sorted(
        str(path.resolve()) for path in inventory_paths if not path.is_file()
    )
    if missing_inventory_paths:
        report.fail(
            "artifact_inventory.complete",
            "one or more required small runtime artifacts are absent",
            details={"missing_paths": missing_inventory_paths},
        )
    else:
        report.pass_(
            "artifact_inventory.complete",
            "all required small runtime artifacts are present for hashing",
            details={"artifact_count": len(inventory_paths)},
        )
    artifact_inventory, artifact_inventory_sha256 = _small_artifact_inventory(inventory_paths)

    return _finalize_report(
        report,
        backend=backend,
        project_root=root,
        config_dir=resolved_config_dir,
        knowledge_dir=resolved_knowledge_dir,
        deployment_profile=deployment_profile,
        narrator_backend=narrator_backend,
        artifact_inventory=artifact_inventory,
        artifact_inventory_sha256=artifact_inventory_sha256,
        wall_clock_ms=(time.perf_counter() - started) * 1000,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only TBX-Agent startup preflight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_preflight()
    except Exception as exc:  # pragma: no cover - last-resort machine-readable boundary
        result = {
            "schema_version": 1,
            "preflight_id": "tbx-agent-read-only-preflight-v1",
            "ok": False,
            "exit_code": 2,
            "mode": {"vision_backend": "unknown", "real_inference": False},
            "guardrails": {
                "read_only": True,
                "weights_loaded": False,
                "dataset_or_test_manifests_read": False,
            },
            "summary": {"pass": 0, "fail": 1, "skip": 0},
            "checks": [
                {
                    "id": "preflight.internal_error",
                    "status": "fail",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return int(result["exit_code"])


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
