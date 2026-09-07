"""Public, routing-neutral DTOs for chest anatomy segmentation.

The models in this module deliberately contain no screening decision fields.
Anatomy segmentation is supporting evidence and must never alter rank03's
three-class argmax route.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AnatomyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LungSide(StrEnum):
    LEFT = "left_lung"
    RIGHT = "right_lung"


class LungFieldZone(StrEnum):
    """Two-dimensional lung fields, not anatomical lung lobes."""

    UPPER = "upper_lung_field"
    MIDDLE = "middle_lung_field"
    LOWER = "lower_lung_field"


class AnatomyQCStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class AnatomyRuntimeProbe(AnatomyModel):
    backend_id: str = Field(min_length=1, max_length=256)
    loaded: bool
    available: Literal["yes", "no", "unverified"]
    detail: str = Field(min_length=1, max_length=1_000)


class CompactRLE(AnatomyModel):
    """A compact row-major binary mask.

    ``counts_b64`` contains unsigned-varint run lengths, beginning with the
    background run, then URL-safe base64 encoded.  This avoids exposing a
    server-local artifact path in API DTOs.
    """

    codec: Literal["tbx-rle-v1"] = "tbx-rle-v1"
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    counts_b64: str = Field(min_length=1)
    foreground_pixels: int = Field(ge=0)
    mask_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def foreground_fits_canvas(self):
        if self.foreground_pixels > self.width * self.height:
            raise ValueError("foreground pixel count exceeds the mask canvas")
        return self


class AnatomyMask(AnatomyModel):
    structure: LungSide
    coordinate_space: Literal["source_image_pixels"] = "source_image_pixels"
    payload: CompactRLE


class AnatomyQCReport(AnatomyModel):
    status: AnatomyQCStatus
    codes: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("codes")
    @classmethod
    def codes_are_unique(cls, value: list[str]):
        if len(value) != len(set(value)):
            raise ValueError("QC codes must be unique")
        return value

    @field_validator("metrics")
    @classmethod
    def metrics_are_finite(cls, value: dict[str, float]):
        if any(not math.isfinite(item) for item in value.values()):
            raise ValueError("QC metrics must be finite")
        return value


class AnatomyEvidence(AnatomyModel):
    """Immutable anatomy evidence produced independently of rank03 routing."""

    run_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=256)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    backend_id: str = Field(min_length=1, max_length=256)
    model_id: str = Field(min_length=1, max_length=256)
    model_weight_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 of the exact checkpoint file loaded by the backend.",
    )
    model_state_dict_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Canonical tensor-level identity of the loaded model state.",
    )
    preprocessing_id: str = Field(min_length=1, max_length=256)
    policy_id: str = Field(min_length=1, max_length=256)
    generation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    masks: list[AnatomyMask]
    qc: AnatomyQCReport
    runtime_ms: int = Field(ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    routing_effect: Literal["none"] = "none"
    clinical_validation: Literal[False] = False

    @model_validator(mode="after")
    def complete_source_space_pair(self):
        by_side = {item.structure: item for item in self.masks}
        if set(by_side) != {LungSide.LEFT, LungSide.RIGHT} or len(self.masks) != 2:
            raise ValueError("anatomy evidence requires exactly one mask for each lung")
        for item in self.masks:
            if (item.payload.width, item.payload.height) != (
                self.image_width,
                self.image_height,
            ):
                raise ValueError("lung masks must use source-image dimensions")
        return self


class LungFieldAssignment(AnatomyModel):
    lung: LungSide
    primary_zone: LungFieldZone
    zone_fractions: dict[LungFieldZone, float]
    intersection_pixels: int = Field(gt=0)
    box_overlap_fraction: float = Field(ge=0.0, le=1.0)
    lung_overlap_fraction: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def zone_distribution_is_valid(self):
        expected = set(LungFieldZone)
        if set(self.zone_fractions) != expected:
            raise ValueError("zone fractions must cover upper, middle, and lower lung fields")
        total = sum(self.zone_fractions.values())
        if any(value < 0.0 or value > 1.0 for value in self.zone_fractions.values()):
            raise ValueError("zone fractions must lie in [0, 1]")
        if abs(total - 1.0) > 1e-6:
            raise ValueError("zone fractions must sum to one")
        return self


class DetectionAnatomyLocation(AnatomyModel):
    bbox_xyxy: tuple[float, float, float, float]
    status: Literal["localized", "outside_lungs", "invalid_anatomy"]
    assignments: list[LungFieldAssignment] = Field(default_factory=list)
    note: Literal[
        "two_dimensional_lung_field_not_lobe",
        "no_lung_mask_overlap",
        "anatomy_qc_failed",
    ]

    @model_validator(mode="after")
    def status_matches_assignments(self):
        if self.status == "localized" and not self.assignments:
            raise ValueError("localized boxes require at least one lung-field assignment")
        if self.status != "localized" and self.assignments:
            raise ValueError("non-localized boxes cannot contain assignments")
        return self


class AnatomySpatialSummary(AnatomyModel):
    """Code-owned, non-diagnostic summary of detector/mask relationships.

    The free-text statements are produced only by the deterministic formatter in
    :mod:`tbx_agent.vision.anatomy.presentation`.  They form a small allowlist that
    the UI may display and the language model may select from; the language model
    never receives masks and cannot invent an anatomic location.
    """

    policy_id: str = Field(min_length=1, max_length=256)
    anatomy_qc_status: AnatomyQCStatus
    candidate_count: int = Field(ge=0)
    localized_count: int = Field(ge=0)
    outside_lungs_count: int = Field(ge=0)
    invalid_anatomy_count: int = Field(ge=0)
    statements: list[str] = Field(default_factory=list, max_length=64)
    routing_effect: Literal["none"] = "none"
    clinical_validation: Literal[False] = False

    @model_validator(mode="after")
    def counts_cover_every_candidate(self):
        represented = (
            self.localized_count
            + self.outside_lungs_count
            + self.invalid_anatomy_count
        )
        if represented != self.candidate_count:
            raise ValueError("spatial-summary counts must cover every detector candidate")
        if not self.statements:
            raise ValueError("spatial summary requires at least one code-owned statement")
        return self
