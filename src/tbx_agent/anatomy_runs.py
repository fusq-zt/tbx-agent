"""Persistable lifecycle records for optional anatomy inference.

These records are intentionally separate from ``CaseRecord`` and
``VisionEvidence``.  Anatomy output is review/visualization evidence and is not
allowed to change the frozen rank03 routing result.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from .schemas import StrictModel, utc_now
from .vision.anatomy import (
    AnatomyEvidence,
    AnatomySpatialSummary,
    DetectionAnatomyLocation,
)
from .vision.refinement import ContourRefinementEvidence


class AnatomyRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_REFINEMENT_FAILURE = "completed_with_refinement_failure"
    TECHNICAL_FAILURE = "technical_failure"


class RefinementRunStatus(StrEnum):
    DISABLED = "disabled"
    PENDING = "pending"
    COMPLETED = "completed"
    TECHNICAL_FAILURE = "technical_failure"


def build_anatomy_pipeline_generation_key(
    *,
    anatomy_generation_key: str,
    detector_run_id: str | None,
    detector_checkpoint_sha256: str | None,
    detector_boxes: list[tuple[float, float, float, float]],
    localization_policy_id: str,
    localization_minimum_box_overlap_fraction: float,
    presentation_policy_id: str,
    refinement_generation_key: str | None = None,
) -> str:
    """Hash every identity that can change persisted anatomy pipeline output."""

    if re.fullmatch(r"[0-9a-f]{64}", anatomy_generation_key) is None:
        raise ValueError("anatomy generation key must be a complete SHA-256 digest")
    if refinement_generation_key is not None and re.fullmatch(
        r"[0-9a-f]{64}", refinement_generation_key
    ) is None:
        raise ValueError("refinement generation key must be a complete SHA-256 digest")
    if not localization_policy_id.strip() or not presentation_policy_id.strip():
        raise ValueError("anatomy pipeline policy identities must not be empty")
    if not 0.0 <= localization_minimum_box_overlap_fraction <= 1.0:
        raise ValueError("localization overlap fraction must lie in [0, 1]")
    payload = {
        "schema": "tbx-anatomy-pipeline-generation-v1",
        "anatomy_generation_key": anatomy_generation_key,
        "detector": {
            "run_id": detector_run_id,
            "checkpoint_sha256": detector_checkpoint_sha256,
            "boxes_xyxy": detector_boxes,
        },
        "localization": {
            "policy_id": localization_policy_id,
            "minimum_box_overlap_fraction": localization_minimum_box_overlap_fraction,
        },
        "presentation_policy_id": presentation_policy_id,
        "refinement_generation_key": refinement_generation_key,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class AnatomyRunRecord(StrictModel):
    run_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    owner_scope: str = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=128)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    anatomy_generation_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    backend_id: str = Field(min_length=1, max_length=256)
    status: AnatomyRunStatus = AnatomyRunStatus.PENDING
    evidence: AnatomyEvidence | None = None
    detector_locations: list[DetectionAnatomyLocation] = Field(default_factory=list)
    spatial_summary: AnatomySpatialSummary | None = None
    localization_policy_id: str = Field(
        default="detector-lung-field-localization-v1",
        min_length=1,
        max_length=256,
    )
    localization_minimum_box_overlap_fraction: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
    )
    presentation_policy_id: str = Field(
        default="detector-lung-field-presentation-v1",
        min_length=1,
        max_length=256,
    )
    refinement_backend_id: str | None = Field(default=None, min_length=1, max_length=256)
    refinement_generation_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    refinement_status: RefinementRunStatus = RefinementRunStatus.DISABLED
    refinement_evidence: ContourRefinementEvidence | None = None
    refinement_error_code: str | None = Field(default=None, min_length=1, max_length=128)
    error_code: str | None = Field(default=None, min_length=1, max_length=128)
    reused_existing_run: bool = False
    routing_effect: str = "none"
    clinical_validation: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    record_version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def lifecycle_is_consistent(self):
        if self.routing_effect != "none":
            raise ValueError("anatomy runs cannot affect rank03 routing")
        if self.clinical_validation:
            raise ValueError("anatomy evidence is not clinically validated")
        anatomy_completed = self.status in {
            AnatomyRunStatus.COMPLETED,
            AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE,
        }
        if anatomy_completed:
            if self.evidence is None or self.error_code is not None:
                raise ValueError("completed anatomy runs require evidence and no error")
            if (
                self.evidence.run_id != self.run_id
                or self.evidence.case_id != self.case_id
                or self.evidence.image_sha256 != self.image_sha256
                or self.evidence.generation_key
                != (self.anatomy_generation_key or self.generation_key)
                or self.evidence.backend_id != self.backend_id
            ):
                raise ValueError("anatomy evidence identity differs from its run")
            # ``None`` remains readable for completed records created before the
            # spatial-summary contract was introduced. New workers always persist
            # the summary; callers must not silently synthesize it for legacy rows.
            if (
                self.spatial_summary is not None
                and self.spatial_summary.candidate_count != len(self.detector_locations)
            ):
                raise ValueError("spatial summary does not cover every detector location")
            if (
                self.spatial_summary is not None
                and self.spatial_summary.policy_id != self.presentation_policy_id
            ):
                raise ValueError("spatial summary presentation policy differs from its run")
        elif (
            self.evidence is not None
            or self.detector_locations
            or self.spatial_summary is not None
        ):
            raise ValueError("non-completed anatomy runs cannot expose partial evidence")
        if self.refinement_status == RefinementRunStatus.DISABLED:
            if any(
                value is not None
                for value in (
                    self.refinement_backend_id,
                    self.refinement_generation_key,
                    self.refinement_evidence,
                    self.refinement_error_code,
                )
            ):
                raise ValueError("disabled contour refinement cannot expose runtime state")
        else:
            if self.refinement_backend_id is None or self.refinement_generation_key is None:
                raise ValueError("configured contour refinement requires immutable identity")
            if self.refinement_status == RefinementRunStatus.PENDING:
                if self.refinement_evidence is not None or self.refinement_error_code is not None:
                    raise ValueError("pending contour refinement cannot expose a result")
            elif self.refinement_status == RefinementRunStatus.COMPLETED:
                evidence = self.refinement_evidence
                if evidence is None or self.refinement_error_code is not None:
                    raise ValueError("completed contour refinement requires evidence and no error")
                if (
                    evidence.case_id != self.case_id
                    or evidence.image_sha256 != self.image_sha256
                    or evidence.backend_id != self.refinement_backend_id
                    or evidence.generation_key != self.refinement_generation_key
                    or evidence.routing_effect != "none"
                    or evidence.clinical_validation
                ):
                    raise ValueError("contour refinement identity differs from its anatomy run")
            elif (
                self.refinement_status == RefinementRunStatus.TECHNICAL_FAILURE
                and (
                    self.refinement_evidence is not None
                    or self.refinement_error_code is None
                )
            ):
                raise ValueError("failed contour refinement requires only a stable error code")
        if self.status == AnatomyRunStatus.COMPLETED and self.refinement_status not in {
            RefinementRunStatus.DISABLED,
            RefinementRunStatus.COMPLETED,
        }:
            raise ValueError("completed anatomy run cannot contain unfinished refinement")
        if self.status == AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE and (
            self.refinement_status != RefinementRunStatus.TECHNICAL_FAILURE
        ):
            raise ValueError("degraded anatomy completion requires a failed refinement branch")
        if self.status == AnatomyRunStatus.TECHNICAL_FAILURE:
            if self.error_code is None:
                raise ValueError("failed anatomy runs require a stable error code")
            if self.refinement_status == RefinementRunStatus.PENDING:
                raise ValueError("failed anatomy run cannot leave refinement pending")
        elif self.error_code is not None:
            raise ValueError("only failed anatomy runs may contain an error code")
        return self
