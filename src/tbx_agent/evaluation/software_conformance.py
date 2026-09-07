"""Versioned synthetic conformance checks for optional imaging software boundaries.

This runner deliberately does not load rank03, D-FINE, PSPNet, MedSAM, an LLM, or
any clinical dataset.  It executes production DTOs and deterministic boundary
functions with generated non-patient inputs.  Its output is software-regression
evidence only; it is not model-performance or clinical-validation evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..anatomy_runs import AnatomyRunRecord, AnatomyRunStatus, RefinementRunStatus
from ..config import PROJECT_ROOT
from ..paths import default_runtime_root
from ..vision.anatomy import (
    AnatomyEvidence,
    AnatomyMask,
    AnatomyQCStatus,
    LungFieldZone,
    LungSide,
    build_generation_key,
    build_spatial_summary,
    encode_binary_mask,
    evaluate_lung_masks,
    localize_bbox_to_lung_fields,
)
from ..vision.image_validator import ImageValidationError, validate_image
from ..vision.refinement import (
    ContourRefinementEvidence,
    DetectionContourEvidence,
    DetectionRefinementStatus,
    detector_box_digest,
)

SCHEMA_VERSION = 1
ADAPTER_ID = "tbx-synthetic-software-conformance-adapter"
ADAPTER_VERSION = "1.0.0"
ZERO_HASH = "0" * 64
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
STABLE_CASE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")

ContractName = Literal[
    "dicom_canonical_metadata_free",
    "dicom_unsafe_burned_in_rejected",
    "anatomy_spatial_localization",
    "anatomy_qc_failure_abstains",
    "refinement_prompt_accounting",
    "api_transparent_artifact_boundary",
    "ui_routing_boundary_fail_closed",
    "ui_refinement_failure_degrades_safely",
]

KNOWN_CONTRACTS: tuple[str, ...] = (
    "dicom_canonical_metadata_free",
    "dicom_unsafe_burned_in_rejected",
    "anatomy_spatial_localization",
    "anatomy_qc_failure_abstains",
    "refinement_prompt_accounting",
    "api_transparent_artifact_boundary",
    "ui_routing_boundary_fail_closed",
    "ui_refinement_failure_degrades_safely",
)

SOURCE_BINDING_FILES: tuple[str, ...] = (
    "pyproject.toml",
    "evaluation/software_conformance_config_v1.json",
    "evaluation/suites/software_conformance_v1/manifest.json",
    "evaluation/suites/software_conformance_v1/cases.jsonl",
    "src/tbx_agent/anatomy_runs.py",
    "src/tbx_agent/api/main.py",
    "src/tbx_agent/evaluation/software_conformance.py",
    "src/tbx_agent/vision/image_validator.py",
    "src/tbx_agent/vision/anatomy/localization.py",
    "src/tbx_agent/vision/anatomy/models.py",
    "src/tbx_agent/vision/anatomy/presentation.py",
    "src/tbx_agent/vision/anatomy/qc.py",
    "src/tbx_agent/vision/anatomy/rle.py",
    "src/tbx_agent/vision/refinement/models.py",
    "ui/anatomy_client.py",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ConformanceCase(StrictModel):
    schema_version: Literal[1]
    case_id: str = Field(min_length=5, max_length=160)
    contract: ContractName
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1000)
    synthetic: Literal[True]
    clinical_validation: Literal[False]
    required_checks: list[str] = Field(min_length=1)

    @field_validator("case_id")
    @classmethod
    def stable_case_id(cls, value: str) -> str:
        if not STABLE_CASE_ID.fullmatch(value):
            raise ValueError("case_id must be a stable lowercase dotted identifier")
        return value

    @field_validator("required_checks")
    @classmethod
    def unique_checks(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("required_checks must be unique")
        if any(not item.strip() for item in value):
            raise ValueError("required_checks cannot contain blanks")
        return value


class ConformanceManifest(StrictModel):
    schema_version: Literal[1]
    suite_id: str = Field(min_length=3, max_length=128)
    suite_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str
    purpose: str
    fixture_kind: Literal["synthetic_non_clinical"]
    clinical_validation: Literal[False]
    selection_use: Literal[False]
    locked_or_hidden_test_used: Literal[False]
    cases_file: str
    cases_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_case_count: int = Field(gt=0)
    required_contracts: list[ContractName]
    split_kind: Literal["none_synthetic_software_suite"]
    change_policy: str

    @field_validator("cases_file")
    @classmethod
    def safe_cases_file(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("cases_file must be a safe relative path")
        return value


class ConformanceConfig(StrictModel):
    schema_version: Literal[1]
    evaluation_id: str = Field(min_length=3, max_length=160)
    hypothesis: str = Field(min_length=10, max_length=2000)
    seed: int = Field(ge=0)
    suite_manifest: str
    selection_use: Literal[False]
    locked_or_hidden_test_used: Literal[False]
    major_variables_changed: list[str] = Field(max_length=2)
    required_case_pass_rate: Literal[1.0]

    @field_validator("suite_manifest")
    @classmethod
    def safe_manifest_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("suite_manifest must be a safe project-relative path")
        return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
        records.append(value)
    return records


def load_suite(
    project_root: Path,
    config_path: Path,
) -> tuple[ConformanceConfig, ConformanceManifest, list[ConformanceCase], Path]:
    root = project_root.resolve(strict=True)
    resolved_config = config_path.expanduser().resolve(strict=True)
    resolved_config.relative_to(root)
    config = ConformanceConfig.model_validate_json(resolved_config.read_text(encoding="utf-8"))
    manifest_path = (root / config.suite_manifest).resolve(strict=True)
    manifest_path.relative_to(root)
    manifest = ConformanceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    cases_path = (manifest_path.parent / manifest.cases_file).resolve(strict=True)
    cases_path.relative_to(manifest_path.parent)
    actual_hash = _sha256_file(cases_path)
    if actual_hash != manifest.cases_sha256:
        raise ValueError(
            "software conformance cases hash mismatch: "
            f"expected={manifest.cases_sha256}, observed={actual_hash}"
        )
    cases = [ConformanceCase.model_validate(record) for record in _load_jsonl(cases_path)]
    if len(cases) != manifest.expected_case_count:
        raise ValueError(
            "software conformance case count mismatch: "
            f"expected={manifest.expected_case_count}, observed={len(cases)}"
        )
    case_ids = [case.case_id for case in cases]
    duplicates = sorted(key for key, count in Counter(case_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"software conformance case IDs must be unique: {duplicates}")
    observed_contracts = [case.contract for case in cases]
    if len(observed_contracts) != len(set(observed_contracts)):
        raise ValueError("software conformance contracts must each have exactly one case")
    if set(manifest.required_contracts) != set(KNOWN_CONTRACTS):
        raise ValueError("manifest must require every code-owned software contract")
    if set(observed_contracts) != set(manifest.required_contracts):
        raise ValueError("suite cases do not exactly cover required_contracts")
    return config, manifest, cases, manifest_path


def _check(expected: Any, observed: Any, passed: bool) -> dict[str, Any]:
    return {"passed": bool(passed), "expected": expected, "observed": observed}


def _synthetic_dicom(*, burned_in: str = "NO") -> bytes:
    """Create a deterministic non-patient Part-10 DX fixture in memory."""

    try:
        import numpy as np
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
    except ImportError as exc:
        raise RuntimeError("software conformance requires the project 'dicom' extra") from exc

    sop_class_uid = "1.2.840.10008.5.1.4.1.1.1.1"
    sop_instance_uid = "1.2.826.0.1.3680043.10.999.2026083001"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.999.1"
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = "DX"
    dataset.BodyPartExamined = "CHEST"
    dataset.ViewPosition = "PA"
    dataset.BurnedInAnnotation = burned_in
    dataset.PatientName = "SYNTHETIC^NONPATIENT"
    dataset.PatientID = "MUST-NOT-PERSIST"
    dataset.Rows = 512
    dataset.Columns = 512
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.WindowCenter = 2048
    dataset.WindowWidth = 4096
    row = np.linspace(0, 4095, 512, dtype=np.dtype("<u2"))
    dataset.PixelData = np.tile(row, (512, 1)).tobytes()
    output = io.BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


def _paired_masks() -> tuple[list[list[bool]], list[list[bool]]]:
    width = height = 30
    left = [[False] * width for _ in range(height)]
    right = [[False] * width for _ in range(height)]
    for y in range(3, 27):
        for x in range(17, 26):
            left[y][x] = True
        for x in range(4, 13):
            right[y][x] = True
    return left, right


def _anatomy_evidence(*, qc_status: AnatomyQCStatus | None = None) -> AnatomyEvidence:
    left, right = _paired_masks()
    qc = evaluate_lung_masks(left, right, source_width=30, source_height=30)
    if qc_status is not None and qc_status != qc.status:
        qc = qc.model_copy(update={"status": qc_status, "codes": ["synthetic_forced_qc_fail"]})
    image_hash = _sha256_bytes(b"software-conformance-synthetic-image-v1")
    weight_hash = _sha256_bytes(b"software-conformance-synthetic-weight-identity-v1")
    generation_key = build_generation_key(
        image_sha256=image_hash,
        model_weight_sha256=weight_hash,
        preprocessing_id="synthetic-source-space-v1",
        policy_id="paired-lung-qc-v1",
        backend_id="synthetic-pspnet-output-contract",
    )
    return AnatomyEvidence(
        run_id="software-conformance-anatomy-run",
        case_id="software-conformance-case",
        image_sha256=image_hash,
        image_width=30,
        image_height=30,
        backend_id="synthetic-pspnet-output-contract",
        model_id="SYNTHETIC_ONLY__no_model_loaded",
        model_weight_sha256=weight_hash,
        model_state_dict_sha256=weight_hash,
        preprocessing_id="synthetic-source-space-v1",
        policy_id="paired-lung-qc-v1",
        generation_key=generation_key,
        masks=[
            AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
            AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
        ],
        qc=qc,
        runtime_ms=0,
    )


def _png_layer_contract(payload: bytes, *, size: tuple[int, int]) -> dict[str, Any]:
    signature = payload.startswith(PNG_SIGNATURE)
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        mode = image.mode
        observed_size = image.size
        alpha_extrema = image.getchannel("A").getextrema() if mode == "RGBA" else None
    return {
        "signature": signature,
        "mode": mode,
        "size": list(observed_size),
        "alpha_extrema": list(alpha_extrema) if alpha_extrema is not None else None,
        "passed": bool(
            signature
            and mode == "RGBA"
            and observed_size == size
            and alpha_extrema is not None
            and alpha_extrema[0] == 0
            and alpha_extrema[1] > 0
        ),
    }


def _case_dicom_canonical_metadata_free() -> dict[str, dict[str, Any]]:
    payload = _synthetic_dicom()
    first = validate_image(payload, max_bytes=len(payload))
    second = validate_image(payload, max_bytes=len(payload))
    no_phi = all(
        marker not in first.artifact_bytes
        for marker in (b"SYNTHETIC^NONPATIENT", b"MUST-NOT-PERSIST")
    )
    return {
        "source_format": _check("DICOM", first.source_format, first.source_format == "DICOM"),
        "transform_id": _check(
            "dicom-crdx-windowed-rgb-v1",
            first.input_transform_id,
            first.input_transform_id == "dicom-crdx-windowed-rgb-v1",
        ),
        "deterministic_derived_png": _check(
            "stable metadata-free PNG digest",
            {
                "png_signature": first.artifact_bytes.startswith(PNG_SIGNATURE),
                "same_sha256": first.sha256 == second.sha256,
                "sha256_matches_artifact": first.sha256 == _sha256_bytes(first.artifact_bytes),
            },
            first.artifact_bytes.startswith(PNG_SIGNATURE)
            and first.sha256 == second.sha256
            and first.sha256 == _sha256_bytes(first.artifact_bytes),
        ),
        "raw_phi_not_persisted": _check(True, no_phi, no_phi),
        "no_quality_warning": _check(
            "transport_valid",
            {"status": first.quality_status, "warnings": list(first.quality_warnings)},
            first.quality_status == "transport_valid" and not first.quality_warnings,
        ),
    }


def _case_dicom_unsafe_burned_in_rejected() -> dict[str, dict[str, Any]]:
    rejected = False
    error_scope = ""
    try:
        validate_image(_synthetic_dicom(burned_in="YES"), max_bytes=2 * 1024 * 1024)
    except ImageValidationError as exc:
        rejected = True
        error_scope = "burned_in_annotation_rejected" if "烧录标注" in str(exc) else "other"
    return {
        "burned_in_rejected": _check(
            "burned_in_annotation_rejected",
            error_scope if rejected else "accepted",
            rejected and error_scope == "burned_in_annotation_rejected",
        )
    }


def _case_anatomy_spatial_localization() -> dict[str, dict[str, Any]]:
    anatomy = _anatomy_evidence()
    boxes = [
        (18.0, 4.0, 24.0, 9.0),
        (5.0, 12.0, 11.0, 18.0),
        (13.0, 3.0, 17.0, 8.0),
    ]
    locations = [localize_bbox_to_lung_fields(box, anatomy=anatomy) for box in boxes]
    summary = build_spatial_summary(locations, anatomy_qc_status=anatomy.qc.status)
    first_assignment = locations[0].assignments[0]
    second_assignment = locations[1].assignments[0]
    expected_zones = (
        first_assignment.lung == LungSide.LEFT
        and first_assignment.primary_zone == LungFieldZone.UPPER
        and second_assignment.lung == LungSide.RIGHT
        and second_assignment.primary_zone == LungFieldZone.MIDDLE
    )
    scope_is_explicit = any(
        "二维" in text and "不代表肺叶" in text for text in summary.statements
    )
    return {
        "anatomy_qc_pass": _check(
            "pass",
            anatomy.qc.status.value,
            anatomy.qc.status.value == "pass",
        ),
        "location_coverage": _check(
            3,
            summary.candidate_count,
            summary.candidate_count == len(boxes) == len(locations),
        ),
        "expected_side_zone": _check(
            ["left_lung/upper_lung_field", "right_lung/middle_lung_field"],
            [
                f"{first_assignment.lung.value}/{first_assignment.primary_zone.value}",
                f"{second_assignment.lung.value}/{second_assignment.primary_zone.value}",
            ],
            expected_zones,
        ),
        "outside_lung_abstention": _check(
            "outside_lungs",
            locations[2].status,
            locations[2].status == "outside_lungs" and not locations[2].assignments,
        ),
        "spatial_scope": _check(
            "two-dimensional lung fields, not lobes or diagnosis",
            scope_is_explicit,
            scope_is_explicit,
        ),
        "routing_neutral": _check(
            "none",
            {"anatomy": anatomy.routing_effect, "summary": summary.routing_effect},
            anatomy.routing_effect == summary.routing_effect == "none"
            and not anatomy.clinical_validation
            and not summary.clinical_validation,
        ),
    }


def _case_anatomy_qc_failure_abstains() -> dict[str, dict[str, Any]]:
    anatomy = _anatomy_evidence(qc_status=AnatomyQCStatus.FAIL)
    location = localize_bbox_to_lung_fields(
        (18.0, 4.0, 24.0, 9.0),
        anatomy=anatomy,
    )
    summary = build_spatial_summary([location], anatomy_qc_status=anatomy.qc.status)
    no_side_claim = all(
        "左侧肺野" not in item and "右侧肺野" not in item
        for item in summary.statements
    )
    return {
        "qc_fail": _check(
            "fail",
            anatomy.qc.status.value,
            anatomy.qc.status == AnatomyQCStatus.FAIL,
        ),
        "spatial_claims_abstained": _check(
            "invalid_anatomy with no side/zone assignment",
            {
                "status": location.status,
                "assignment_count": len(location.assignments),
                "side_claim_absent": no_side_claim,
            },
            location.status == "invalid_anatomy"
            and not location.assignments
            and summary.invalid_anatomy_count == 1
            and no_side_claim,
        ),
        "routing_neutral": _check("none", summary.routing_effect, summary.routing_effect == "none"),
    }


def _refinement_evidence() -> ContourRefinementEvidence:
    anatomy = _anatomy_evidence()
    first_box = (18.0, 4.0, 24.0, 9.0)
    second_box = (13.0, 3.0, 17.0, 8.0)
    contour = [[False] * 30 for _ in range(30)]
    for y in range(4, 9):
        for x in range(18, 24):
            contour[y][x] = True
    encoded = encode_binary_mask(contour)
    boxes = [first_box, second_box]
    return ContourRefinementEvidence(
        case_id=anatomy.case_id,
        image_sha256=anatomy.image_sha256,
        image_width=30,
        image_height=30,
        backend_id="synthetic-medsam-output-contract",
        model_id="SYNTHETIC_ONLY__no_model_loaded",
        model_revision="1" * 40,
        model_weight_sha256="2" * 64,
        model_state_dict_sha256="3" * 64,
        model_config_sha256="4" * 64,
        preprocessor_config_sha256="5" * 64,
        preprocessing_id="synthetic-medsam-preprocess-v1",
        policy_id="synthetic-refinement-policy-v1",
        generation_key="6" * 64,
        anatomy_generation_key=anatomy.generation_key,
        anatomy_mask_digest="7" * 64,
        detector_box_digest=detector_box_digest(boxes),
        items=[
            DetectionContourEvidence(
                detection_index=0,
                bbox_xyxy=first_box,
                status=DetectionRefinementStatus.REFINED,
                mask=encoded,
                raw_mask_pixels=encoded.foreground_pixels,
                lung_constrained_pixels=encoded.foreground_pixels,
                mask_prompt_overlap_fraction=1.0,
                note="visualization_only_nonvalidated_contour",
            ),
            DetectionContourEvidence(
                detection_index=1,
                bbox_xyxy=second_box,
                status=DetectionRefinementStatus.CAPACITY_ABSTAINED,
                raw_mask_pixels=0,
                lung_constrained_pixels=0,
                note="refinement_capacity_limit",
            ),
        ],
        max_prompts_per_run=1,
        max_batch_size=1,
        capacity_abstained_count=1,
        runtime_ms=0,
    )


def _case_refinement_prompt_accounting() -> dict[str, dict[str, Any]]:
    from ..api.main import _render_refinement_contours_png

    evidence = _refinement_evidence()
    layer = _png_layer_contract(
        _render_refinement_contours_png(evidence),
        size=(evidence.image_width, evidence.image_height),
    )
    mutation_rejected = False
    try:
        ContourRefinementEvidence.model_validate(
            {**evidence.model_dump(mode="json"), "routing_effect": "changes_argmax"}
        )
    except ValidationError:
        mutation_rejected = True
    return {
        "prompt_accounting": _check(
            [0, 1],
            [item.detection_index for item in evidence.items],
            [item.detection_index for item in evidence.items] == [0, 1],
        ),
        "capacity_abstention_explicit": _check(
            1,
            {
                "count": evidence.capacity_abstained_count,
                "status": evidence.items[1].status.value,
                "mask_present": evidence.items[1].mask is not None,
            },
            evidence.capacity_abstained_count == 1
            and evidence.items[1].status == DetectionRefinementStatus.CAPACITY_ABSTAINED
            and evidence.items[1].mask is None,
        ),
        "contour_layer_png": _check(
            "transparent RGBA source-coordinate PNG",
            {key: value for key, value in layer.items() if key != "passed"},
            layer["passed"],
        ),
        "routing_neutral": _check(
            "none/clinical_validation=false",
            {
                "routing_effect": evidence.routing_effect,
                "clinical_validation": evidence.clinical_validation,
            },
            evidence.routing_effect == "none" and not evidence.clinical_validation,
        ),
        "routing_mutation_rejected": _check(True, mutation_rejected, mutation_rejected),
    }


def _case_api_transparent_artifact_boundary() -> dict[str, dict[str, Any]]:
    from ..api.main import _redact_internal_artifact_references, _render_anatomy_boundary_png

    anatomy = _anatomy_evidence()
    layer = _png_layer_contract(
        _render_anatomy_boundary_png(anatomy, {LungSide.LEFT, LungSide.RIGHT}),
        size=(anatomy.image_width, anatomy.image_height),
    )
    raw = {
        "case_id": anatomy.case_id,
        "image_artifact_ref": "server-private/top-level.png",
        "nested": {
            "image_artifact_ref": "server-private/nested.png",
            "backend_id": anatomy.backend_id,
        },
    }
    redacted = _redact_internal_artifact_references(raw)
    serialized = json.dumps(redacted, ensure_ascii=False)
    return {
        "nested_artifact_reference_removed": _check(
            True,
            "image_artifact_ref" not in serialized,
            "image_artifact_ref" not in serialized and "server-private" not in serialized,
        ),
        "safe_fields_retained": _check(
            anatomy.backend_id,
            redacted.get("nested", {}).get("backend_id"),
            redacted.get("nested", {}).get("backend_id") == anatomy.backend_id,
        ),
        "lung_layer_png": _check(
            "transparent RGBA source-coordinate PNG without source CXR",
            {key: value for key, value in layer.items() if key != "passed"},
            layer["passed"],
        ),
    }


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any] | None = None,
        *,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        ok: bool = True,
    ) -> None:
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.ok = ok

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


def _ui_module():
    root_text = str(PROJECT_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module("ui.anatomy_client")


def _case_ui_routing_boundary_fail_closed() -> dict[str, dict[str, Any]]:
    ui = _ui_module()
    calls: list[str] = []

    def request(method: str, url: str, **_kwargs):
        calls.append(f"{method} {url}")
        return _FakeResponse(
            {
                "run_id": "synthetic-ui-run",
                "case_id": "synthetic-ui-case",
                "status": "completed",
                "routing_effect": "changes_argmax",
                "clinical_validation": False,
            }
        )

    rejected = False
    try:
        ui.request_and_poll_anatomy(
            api_url="http://127.0.0.1:8000",
            case_id="synthetic-ui-case",
            owner_scope="synthetic-owner",
            user_id="synthetic-user",
            image_name="synthetic.png",
            image_bytes=PNG_SIGNATURE + b"synthetic-non-image-transport-fixture",
            image_mime_type="image/png",
            request_fn=request,
            sleep_fn=lambda _seconds: None,
        )
    except ui.AnatomyClientError:
        rejected = True
    return {
        "ui_rejects_routing_mutation": _check(True, rejected, rejected),
        "ui_no_followup_after_rejection": _check(
            1,
            len(calls),
            rejected and len(calls) == 1,
        ),
    }


def _case_ui_refinement_failure_degrades_safely() -> dict[str, dict[str, Any]]:
    ui = _ui_module()
    anatomy = _anatomy_evidence()
    location = localize_bbox_to_lung_fields((18.0, 4.0, 24.0, 9.0), anatomy=anatomy)
    summary = build_spatial_summary([location], anatomy_qc_status=anatomy.qc.status)
    run = AnatomyRunRecord(
        run_id=anatomy.run_id,
        case_id=anatomy.case_id,
        owner_scope="synthetic-owner",
        user_id="synthetic-user",
        image_sha256=anatomy.image_sha256,
        generation_key="8" * 64,
        anatomy_generation_key=anatomy.generation_key,
        backend_id=anatomy.backend_id,
        status=AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE,
        evidence=anatomy,
        detector_locations=[location],
        spatial_summary=summary,
        refinement_backend_id="synthetic-medsam-output-contract",
        refinement_generation_key="9" * 64,
        refinement_status=RefinementRunStatus.TECHNICAL_FAILURE,
        refinement_error_code="synthetic_runtime_failure",
    )
    run_payload = run.model_dump(mode="json")
    boundary = _render_synthetic_boundary_for_ui(anatomy)
    calls: list[str] = []
    identity_scopes: list[dict[str, str]] = []

    def request(method: str, url: str, **kwargs):
        calls.append(f"{method} {url}")
        identity_scopes.append(dict(kwargs.get("params") or {}))
        if method == "POST":
            return _FakeResponse(run_payload)
        if url.endswith("boundary.png"):
            return _FakeResponse(
                content=boundary,
                headers={
                    "Content-Type": "image/png",
                    "X-Anatomy-Routing-Effect": "none",
                },
            )
        raise AssertionError("failed refinement must not request a contour layer")

    result = ui.request_and_poll_anatomy(
        api_url="http://127.0.0.1:8000",
        case_id=anatomy.case_id,
        owner_scope="synthetic-owner",
        user_id="synthetic-user",
        image_name="synthetic.png",
        image_bytes=PNG_SIGNATURE + b"synthetic-non-image-transport-fixture",
        image_mime_type="image/png",
        request_fn=request,
        sleep_fn=lambda _seconds: None,
    )
    contour_calls = [item for item in calls if item.endswith("contours.png")]
    expected_identity = {"owner_scope": "synthetic-owner", "user_id": "synthetic-user"}
    identity_is_preserved = bool(identity_scopes) and all(
        all(scope.get(key) == value for key, value in expected_identity.items())
        for scope in identity_scopes
    )
    return {
        "ui_keeps_lung_layer_on_refinement_failure": _check(
            "completed_with_refinement_failure with lung boundary",
            {
                "status": result.status,
                "boundary_png": bool(result.boundary_png),
                "contours_png": bool(result.contours_png),
            },
            result.status == "completed_with_refinement_failure"
            and result.boundary_png == boundary
            and result.contours_png is None,
        ),
        "ui_does_not_fetch_failed_contours": _check(0, len(contour_calls), not contour_calls),
        "ui_identity_scope_forwarded": _check(
            expected_identity,
            identity_scopes,
            identity_is_preserved,
        ),
    }


def _render_synthetic_boundary_for_ui(anatomy: AnatomyEvidence) -> bytes:
    from ..api.main import _render_anatomy_boundary_png

    return _render_anatomy_boundary_png(anatomy, {LungSide.LEFT, LungSide.RIGHT})


CASE_RUNNERS: dict[str, Callable[[], dict[str, dict[str, Any]]]] = {
    "dicom_canonical_metadata_free": _case_dicom_canonical_metadata_free,
    "dicom_unsafe_burned_in_rejected": _case_dicom_unsafe_burned_in_rejected,
    "anatomy_spatial_localization": _case_anatomy_spatial_localization,
    "anatomy_qc_failure_abstains": _case_anatomy_qc_failure_abstains,
    "refinement_prompt_accounting": _case_refinement_prompt_accounting,
    "api_transparent_artifact_boundary": _case_api_transparent_artifact_boundary,
    "ui_routing_boundary_fail_closed": _case_ui_routing_boundary_fail_closed,
    "ui_refinement_failure_degrades_safely": _case_ui_refinement_failure_degrades_safely,
}


def _source_revision(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", str(project_root)],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def _candidate_source_binding(project_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in SOURCE_BINDING_FILES:
        source = (project_root / relative).resolve(strict=True)
        source.relative_to(project_root.resolve())
        records.append(
            {"path": relative, "size": source.stat().st_size, "sha256": _sha256_file(source)}
        )
    digest = _sha256_bytes(_canonical_json(records))
    return {"source_tree_sha256": digest, "files": records}


def _full_configuration(
    *,
    config: ConformanceConfig,
    manifest: ConformanceManifest,
    candidate_id: str,
    source_binding: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation": config.model_dump(mode="json"),
        "suite": manifest.model_dump(mode="json"),
        "execution": {
            "adapter_id": ADAPTER_ID,
            "adapter_version": ADAPTER_VERSION,
            "fixture_generation": "deterministic_code_owned_non_patient_v1",
            "network_access": False,
            "model_weights_loaded": False,
            "dataset_access": False,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "candidate": {"candidate_id": candidate_id, **source_binding},
    }


def _run_cases(cases: list[ConformanceCase]) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for case in cases:
        case_started = time.perf_counter()
        error: dict[str, str] | None = None
        try:
            checks = CASE_RUNNERS[case.contract]()
            expected_checks = set(case.required_checks)
            observed_checks = set(checks)
            if observed_checks != expected_checks:
                checks["suite_runner_contract"] = _check(
                    sorted(expected_checks),
                    sorted(observed_checks),
                    False,
                )
        except Exception as exc:  # noqa: BLE001 - retain every failed case as evidence
            checks = {
                "execution": _check(
                    "completed without exception",
                    f"{type(exc).__name__}: {str(exc)[:500]}",
                    False,
                )
            }
            error = {"type": type(exc).__name__, "message": str(exc)[:500]}
        records.append(
            {
                "case_id": case.case_id,
                "contract": case.contract,
                "title": case.title,
                "passed": bool(checks) and all(item["passed"] for item in checks.values()),
                "checks": checks,
                "latency_ms": (time.perf_counter() - case_started) * 1000,
                "error": error,
            }
        )
    return records, (time.perf_counter() - started) * 1000


def _metrics(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(per_case)
    passed = sum(bool(item["passed"]) for item in per_case)
    checks = [check for item in per_case for check in item["checks"].values()]
    passed_checks = sum(bool(check["passed"]) for check in checks)
    return {
        "case_count": total,
        "passed_case_count": passed,
        "failed_case_count": total - passed,
        "case_pass_rate": passed / total if total else 0.0,
        "check_count": len(checks),
        "passed_check_count": passed_checks,
        "check_pass_rate": passed_checks / len(checks) if checks else 0.0,
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _ledger_entry_hash(entry_without_hash: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(entry_without_hash))


def verify_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"valid": True, "event_count": 0, "head_event_hash": None}
    previous = ZERO_HASH
    count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank software-conformance ledger line at {path}:{line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid software-conformance ledger JSON at line {line_number}"
            ) from exc
        if not isinstance(payload, dict) or payload.get("event_type") != "software_conformance_run":
            raise ValueError(f"invalid software-conformance ledger event at line {line_number}")
        if payload.get("previous_event_hash") != previous:
            raise ValueError(f"software-conformance ledger chain break at line {line_number}")
        event_hash = payload.pop("event_hash", None)
        calculated = _ledger_entry_hash(payload)
        if event_hash != calculated:
            raise ValueError(f"software-conformance ledger hash mismatch at line {line_number}")
        previous = calculated
        count += 1
    return {"valid": True, "event_count": count, "head_event_hash": previous if count else None}


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write while persisting software-conformance ledger")
        offset += written


@contextlib.contextmanager
def _exclusive_ledger_lock(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"software-conformance ledger is locked: {lock_path}") from exc
    try:
        _write_all(
            descriptor,
            _canonical_json({"pid": os.getpid(), "created_at": _utc_now()}),
        )
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink()


def append_ledger(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_ledger_lock(path):
        audit = verify_ledger(path)
        entry_without_hash = {
            "schema_version": SCHEMA_VERSION,
            "event_type": "software_conformance_run",
            **payload,
            "previous_event_hash": audit["head_event_hash"] or ZERO_HASH,
        }
        entry = {
            **entry_without_hash,
            "event_hash": _ledger_entry_hash(entry_without_hash),
        }
        encoded = (json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return entry


def run_conformance(
    *,
    project_root: Path,
    config_path: Path,
    output_dir: Path,
    ledger_path: Path,
    candidate_id: str,
) -> tuple[dict[str, Any], Path]:
    root = project_root.expanduser().resolve(strict=True)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config, manifest, cases, manifest_path = load_suite(root, config_path)
    random.seed(config.seed)
    source_binding = _candidate_source_binding(root)
    source_revision = {
        **_source_revision(root),
        "candidate_source_sha256": source_binding["source_tree_sha256"],
    }
    full_config = _full_configuration(
        config=config,
        manifest=manifest,
        candidate_id=candidate_id,
        source_binding=source_binding,
    )
    config_sha256 = _sha256_bytes(_canonical_json(full_config))
    per_case, execution_ms = _run_cases(cases)
    metrics = _metrics(per_case)
    passed = metrics["case_pass_rate"] >= config.required_case_pass_rate
    runtime = {
        "wall_clock_ms": (time.perf_counter() - started) * 1000,
        "candidate_execution_wall_clock_ms": execution_ms,
        "peak_vram_mib": None,
        "peak_vram_measured": False,
        "peak_vram_reason": (
            "not_measured: this CPU-only synthetic software-contract suite does not load "
            "CUDA or model runtimes"
        ),
    }
    created_at = _utc_now()
    fingerprint = {
        "candidate_id": candidate_id,
        "created_at": created_at,
        "config_sha256": config_sha256,
        "split_hash": manifest.cases_sha256,
        "source_revision": source_revision,
        "per_case": per_case,
    }
    run_id = f"swconf-{_sha256_bytes(_canonical_json(fingerprint))[:16]}"
    report = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": config.evaluation_id,
        "run_id": run_id,
        "created_at": created_at,
        "status": "passed" if passed else "regressed_retained",
        "candidate_id": candidate_id,
        "adapter_id": ADAPTER_ID,
        "adapter_version": ADAPTER_VERSION,
        "hypothesis": config.hypothesis,
        "full_config": full_config,
        "config_sha256": config_sha256,
        "seed": config.seed,
        "split": {
            "kind": manifest.split_kind,
            "hash": manifest.cases_sha256,
            "note": "No train/validation/test split exists for generated software fixtures.",
        },
        "suite_id": manifest.suite_id,
        "suite_version": manifest.suite_version,
        "suite_manifest_sha256": _sha256_file(manifest_path),
        "source_revision": source_revision,
        "fixture_kind": manifest.fixture_kind,
        "selection_use": False,
        "locked_or_hidden_test_used": False,
        "clinical_validation": False,
        "metrics": metrics,
        "per_case": per_case,
        "release_gate": {
            "required_case_pass_rate": config.required_case_pass_rate,
            "passed": passed,
        },
        "runtime": runtime,
        "limitations": [
            "No rank03, D-FINE, PSPNet, MedSAM or LLM weights are loaded.",
            (
                "No model accuracy, segmentation quality, diagnostic performance or "
                "clinical safety is measured."
            ),
            (
                "DICOM checks cover the deliberately restricted uncompressed "
                "single-frame CR/DX contract only."
            ),
            (
                "API checks exercise code-owned redaction/rendering functions; full "
                "HTTP lifecycle remains covered by pytest integration tests."
            ),
        ],
    }
    report_path = destination / "report.json"
    _write_json_exclusive(report_path, report)
    append_ledger(
        ledger_path,
        {
            "created_at": created_at,
            "status": report["status"],
            "evaluation_id": config.evaluation_id,
            "candidate_id": candidate_id,
            "run_id": run_id,
            "report_path": str(report_path),
            "report_sha256": _sha256_file(report_path),
            "config_sha256": config_sha256,
            "hypothesis": config.hypothesis,
            "full_config": full_config,
            "seed": config.seed,
            "split_hash": manifest.cases_sha256,
            "source_revision": source_revision,
            "metrics": metrics,
            "runtime": runtime,
            "error": None,
        },
    )
    return report, report_path


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return default_runtime_root() / "evaluation_runs" / f"software-conformance-{timestamp}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run versioned synthetic DICOM/anatomy/refinement/UI software contracts "
            "without models, network access or clinical data."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "software_conformance_config_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--candidate-id", default="current-source-candidate")
    args = parser.parse_args(argv)
    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    ledger_path = (
        args.ledger.expanduser().resolve()
        if args.ledger is not None
        else output_dir.parent / "software_conformance_ledger.jsonl"
    )
    output_preexisted = output_dir.exists()
    started = time.perf_counter()
    try:
        report, report_path = run_conformance(
            project_root=PROJECT_ROOT,
            config_path=args.config,
            output_dir=output_dir,
            ledger_path=ledger_path,
            candidate_id=args.candidate_id,
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "run_id": report["run_id"],
                    "report": str(report_path),
                    "ledger": str(ledger_path),
                    "metrics": report["metrics"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if report["release_gate"]["passed"] else 2
    except Exception as exc:  # noqa: BLE001 - retain failed attempts in the external ledger
        provenance_error: str | None = None
        try:
            config, manifest, _cases, _manifest_path = load_suite(
                PROJECT_ROOT,
                args.config,
            )
            source_binding = _candidate_source_binding(PROJECT_ROOT)
            source_revision = {
                **_source_revision(PROJECT_ROOT),
                "candidate_source_sha256": source_binding["source_tree_sha256"],
            }
            full_config = _full_configuration(
                config=config,
                manifest=manifest,
                candidate_id=args.candidate_id,
                source_binding=source_binding,
            )
            evaluation_id = config.evaluation_id
            hypothesis = config.hypothesis
            config_sha256: str | None = _sha256_bytes(_canonical_json(full_config))
            seed: int | None = config.seed
            split_hash: str | None = manifest.cases_sha256
        except Exception as provenance_exc:  # noqa: BLE001 - record unavailable provenance
            provenance_error = f"{type(provenance_exc).__name__}: {str(provenance_exc)[:500]}"
            source_revision = _source_revision(PROJECT_ROOT)
            full_config = {
                "invocation": {
                    "config_path": str(args.config),
                    "candidate_id": args.candidate_id,
                    "output_dir": str(output_dir),
                    "ledger_path": str(ledger_path),
                },
                "provenance_error": provenance_error,
            }
            evaluation_id = "tbx-agent-software-conformance-v1"
            hypothesis = "Unavailable because versioned suite provenance failed validation."
            config_sha256 = _sha256_bytes(_canonical_json(full_config))
            seed = None
            split_hash = None
        failure_runtime = {
            "wall_clock_ms_until_failure": (time.perf_counter() - started) * 1000,
            "peak_vram_mib": None,
            "peak_vram_measured": False,
            "peak_vram_reason": (
                "not_measured: failure occurred in a CPU-only synthetic suite"
            ),
        }
        failure_payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at": _utc_now(),
            "status": "failed_retained",
            "evaluation_id": evaluation_id,
            "candidate_id": args.candidate_id,
            "hypothesis": hypothesis,
            "full_config": full_config,
            "config_sha256": config_sha256,
            "seed": seed,
            "split_hash": split_hash,
            "source_revision": source_revision,
            "metrics": {},
            "runtime": failure_runtime,
            "provenance_error": provenance_error,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc)[:1000],
            },
        }
        failure_path: Path | None = None
        try:
            if (
                output_dir.is_dir()
                and not output_preexisted
                and not isinstance(exc, FileExistsError)
            ):
                failure_path = output_dir / "failure.json"
                if not failure_path.exists():
                    _write_json_exclusive(failure_path, failure_payload)
            append_ledger(
                ledger_path,
                {
                    **failure_payload,
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_retained",
                    "run_id": None,
                    "report_path": str(failure_path) if failure_path is not None else None,
                    "report_sha256": (
                        _sha256_file(failure_path) if failure_path is not None else None
                    ),
                },
            )
        except Exception as ledger_exc:  # noqa: BLE001 - report fail-closed ledger errors too
            print(
                "software conformance failed and failure receipt could not be appended: "
                f"{type(ledger_exc).__name__}: {ledger_exc}",
                file=sys.stderr,
            )
        print(
            f"software conformance failed closed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
