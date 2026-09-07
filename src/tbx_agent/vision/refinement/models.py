"""Routing-neutral evidence models for optional detector-box contour refinement.

The contours in this module are visualization evidence.  They are not lesion
labels, do not change rank03's native three-class argmax route, and carry an
explicit ``clinical_validation=False`` marker at every public boundary.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..anatomy.models import CompactRLE


class RefinementModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RefinementRuntimeProbe(RefinementModel):
    backend_id: str = Field(min_length=1, max_length=256)
    loaded: bool
    available: Literal["yes", "no", "unverified"]
    detail: str = Field(min_length=1, max_length=1_000)


class DetectionRefinementStatus(StrEnum):
    REFINED = "refined"
    OUTSIDE_LUNGS = "outside_lungs"
    MASK_QC_FAILED = "mask_qc_failed"
    CAPACITY_ABSTAINED = "capacity_abstained"


class DetectionContourEvidence(RefinementModel):
    detection_index: int = Field(ge=0)
    bbox_xyxy: tuple[float, float, float, float]
    status: DetectionRefinementStatus
    mask: CompactRLE | None = None
    raw_mask_pixels: int = Field(ge=0)
    lung_constrained_pixels: int = Field(ge=0)
    mask_prompt_overlap_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    note: Literal[
        "visualization_only_nonvalidated_contour",
        "detection_box_has_no_lung_overlap",
        "refinement_mask_qc_failed",
        "refinement_capacity_limit",
    ]
    routing_effect: Literal["none"] = "none"
    clinical_validation: Literal[False] = False

    @model_validator(mode="after")
    def status_and_mask_are_consistent(self):
        if len(self.bbox_xyxy) != 4 or not all(math.isfinite(v) for v in self.bbox_xyxy):
            raise ValueError("refinement boxes must contain four finite xyxy coordinates")
        if self.status == DetectionRefinementStatus.REFINED:
            if self.mask is None or self.lung_constrained_pixels <= 0:
                raise ValueError("refined contours require a non-empty source-space mask")
            if self.note != "visualization_only_nonvalidated_contour":
                raise ValueError("refined contours require the non-validation scope note")
            if self.mask.foreground_pixels != self.lung_constrained_pixels:
                raise ValueError("refinement mask metadata is inconsistent")
            if self.mask_prompt_overlap_fraction is None:
                raise ValueError("refined contours require prompt-overlap metadata")
        elif self.mask is not None or self.lung_constrained_pixels != 0:
            raise ValueError("abstained contour refinements cannot expose a partial mask")
        expected_note = {
            DetectionRefinementStatus.REFINED: (
                "visualization_only_nonvalidated_contour"
            ),
            DetectionRefinementStatus.OUTSIDE_LUNGS: (
                "detection_box_has_no_lung_overlap"
            ),
            DetectionRefinementStatus.MASK_QC_FAILED: "refinement_mask_qc_failed",
            DetectionRefinementStatus.CAPACITY_ABSTAINED: "refinement_capacity_limit",
        }[self.status]
        if self.note != expected_note:
            raise ValueError("refinement status and scope note are inconsistent")
        return self


class ContourRefinementEvidence(RefinementModel):
    """One immutable batch produced from D-FINE box prompts and lung masks."""

    case_id: str = Field(min_length=1, max_length=256)
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    backend_id: str = Field(min_length=1, max_length=256)
    model_id: str = Field(min_length=1, max_length=256)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_weight_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_state_dict_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessor_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_id: str = Field(min_length=1, max_length=256)
    policy_id: str = Field(min_length=1, max_length=256)
    generation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    anatomy_generation_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    anatomy_mask_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector_box_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_source: Literal["dfine_xyxy_source_image_pixels"] = (
        "dfine_xyxy_source_image_pixels"
    )
    lung_constraint: Literal["pspnet_left_right_union"] = "pspnet_left_right_union"
    items: list[DetectionContourEvidence]
    max_prompts_per_run: int = Field(ge=1, le=300)
    max_batch_size: int = Field(ge=1, le=64)
    capacity_abstained_count: int = Field(ge=0)
    runtime_ms: int = Field(ge=0)
    routing_effect: Literal["none"] = "none"
    clinical_validation: Literal[False] = False
    interpretation_scope: Literal["visualization_only_nonvalidated_contour"] = (
        "visualization_only_nonvalidated_contour"
    )

    @model_validator(mode="after")
    def items_cover_each_prompt_once(self):
        indexes = [item.detection_index for item in self.items]
        if indexes != list(range(len(self.items))):
            raise ValueError("refinement items must preserve every detector prompt in order")
        for item in self.items:
            if item.mask is not None and (
                item.mask.width != self.image_width or item.mask.height != self.image_height
            ):
                raise ValueError("refinement masks must use source-image dimensions")
        observed_capacity = sum(
            item.status == DetectionRefinementStatus.CAPACITY_ABSTAINED
            for item in self.items
        )
        if self.capacity_abstained_count != observed_capacity:
            raise ValueError("refinement capacity count differs from item statuses")
        return self
