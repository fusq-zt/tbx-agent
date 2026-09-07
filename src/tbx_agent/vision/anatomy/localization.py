"""Map detector boxes to 2-D lung fields using source-coordinate masks."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass

from ...schemas import MAX_DETECTION_CANDIDATES
from .models import (
    AnatomyEvidence,
    AnatomyQCStatus,
    DetectionAnatomyLocation,
    LungFieldAssignment,
    LungFieldZone,
    LungSide,
)
from .rle import decode_binary_mask


@dataclass(frozen=True, slots=True)
class LungFieldLocalizationPolicy:
    policy_id: str = "detector-lung-field-localization-v1"
    minimum_box_overlap_fraction: float = 0.01


@dataclass(frozen=True, slots=True)
class _PreparedLungMask:
    """Query-ready mask state shared by every detector candidate in one run.

    A compact summed-area table makes both the whole-box and per-zone
    intersections constant-time.  The RLE is decoded exactly once per lung;
    detector fan-out therefore does not multiply decode, digest validation, or
    full-canvas scans.
    """

    lung: LungSide
    summed_area: array
    stride: int
    lung_top: int
    lung_bottom: int
    lung_pixels: int
    zone_row_ranges: dict[LungFieldZone, tuple[int, int] | None]

    def rectangle_sum(self, *, x1: int, y1: int, x2: int, y2: int) -> int:
        if x1 >= x2 or y1 >= y2:
            return 0
        top_left = y1 * self.stride + x1
        top_right = y1 * self.stride + x2
        bottom_left = y2 * self.stride + x1
        bottom_right = y2 * self.stride + x2
        return int(
            self.summed_area[bottom_right]
            - self.summed_area[top_right]
            - self.summed_area[bottom_left]
            + self.summed_area[top_left]
        )


def _row_zone(*, row: int, lung_top: int, lung_height: int) -> LungFieldZone:
    normalized_y = min(0.999999, max(0.0, (row + 0.5 - lung_top) / lung_height))
    return (
        LungFieldZone.UPPER
        if normalized_y < 1 / 3
        else LungFieldZone.MIDDLE
        if normalized_y < 2 / 3
        else LungFieldZone.LOWER
    )


def _prepare_lung_mask(*, lung: LungSide, mask: list[list[bool]]) -> _PreparedLungMask:
    height = len(mask)
    width = len(mask[0]) if mask else 0
    stride = width + 1
    # Unsigned 32-bit storage is sufficient for normal radiographs and is much
    # smaller than a Python-int matrix.  Keep the calculation correct for an
    # unusually large validated canvas by widening only when required.
    typecode = "I" if width * height <= 0xFFFFFFFF else "Q"
    summed_area = array(typecode, [0]) * ((height + 1) * stride)
    lung_top: int | None = None
    lung_bottom: int | None = None
    lung_pixels = 0

    for y, row in enumerate(mask, start=1):
        row_sum = 0
        previous_row_offset = (y - 1) * stride
        row_offset = y * stride
        for x, value in enumerate(row, start=1):
            row_sum += int(value)
            summed_area[row_offset + x] = summed_area[previous_row_offset + x] + row_sum
        if row_sum:
            source_y = y - 1
            if lung_top is None:
                lung_top = source_y
            lung_bottom = source_y + 1
            lung_pixels += row_sum

    if lung_top is None or lung_bottom is None:
        # AnatomyEvidence may legitimately carry empty masks only when QC has
        # already failed.  Non-failing evidence with an empty side previously
        # returned no assignment; retain that behavior with a zero-height range.
        lung_top = 0
        lung_bottom = 0

    zone_row_ranges: dict[LungFieldZone, tuple[int, int] | None] = {
        zone: None for zone in LungFieldZone
    }
    lung_height = max(1, lung_bottom - lung_top)
    for row in range(lung_top, lung_bottom):
        zone = _row_zone(row=row, lung_top=lung_top, lung_height=lung_height)
        current = zone_row_ranges[zone]
        zone_row_ranges[zone] = (row, row + 1) if current is None else (current[0], row + 1)

    return _PreparedLungMask(
        lung=lung,
        summed_area=summed_area,
        stride=stride,
        lung_top=lung_top,
        lung_bottom=lung_bottom,
        lung_pixels=lung_pixels,
        zone_row_ranges=zone_row_ranges,
    )


def _prepare_lung_masks(anatomy: AnatomyEvidence) -> dict[LungSide, _PreparedLungMask]:
    prepared: dict[LungSide, _PreparedLungMask] = {}
    # Preserve the evidence mask order (and thus malformed-RLE error order).
    # Each decoded matrix becomes unreachable immediately after its SAT is built,
    # which also avoids retaining two source-sized boolean matrices.
    for item in anatomy.masks:
        prepared[item.structure] = _prepare_lung_mask(
            lung=item.structure,
            mask=decode_binary_mask(item.payload),
        )
    return prepared


def _validate_bbox(
    bbox_xyxy: tuple[float, float, float, float], *, width: int, height: int
) -> tuple[int, int, int, int]:
    if len(bbox_xyxy) != 4 or not all(math.isfinite(value) for value in bbox_xyxy):
        raise ValueError("bbox coordinates must be finite xyxy values")
    x1, y1, x2, y2 = bbox_xyxy
    if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
        raise ValueError("bbox must have positive area within the source image")
    # Include every source pixel whose cell intersects the continuous box.
    return (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )


def _assignment(
    *,
    prepared: _PreparedLungMask,
    bbox: tuple[int, int, int, int],
    box_pixels: int,
) -> LungFieldAssignment | None:
    if prepared.lung_pixels == 0:
        return None
    zone_counts = {zone: 0 for zone in LungFieldZone}
    x1, y1, x2, y2 = bbox
    for zone, row_range in prepared.zone_row_ranges.items():
        if row_range is None:
            continue
        zone_y1 = max(y1, row_range[0])
        zone_y2 = min(y2, row_range[1])
        zone_counts[zone] = prepared.rectangle_sum(
            x1=x1,
            y1=zone_y1,
            x2=x2,
            y2=zone_y2,
        )
    intersection = sum(zone_counts.values())
    if intersection == 0:
        return None
    fractions = {zone: count / intersection for zone, count in zone_counts.items()}
    primary = max(
        LungFieldZone,
        key=lambda zone: (fractions[zone], -list(LungFieldZone).index(zone)),
    )
    return LungFieldAssignment(
        lung=prepared.lung,
        primary_zone=primary,
        zone_fractions=fractions,
        intersection_pixels=intersection,
        box_overlap_fraction=intersection / box_pixels,
        lung_overlap_fraction=intersection / prepared.lung_pixels,
    )


def _validated_policy(
    policy: LungFieldLocalizationPolicy | None,
) -> LungFieldLocalizationPolicy:
    resolved = policy or LungFieldLocalizationPolicy()
    if not 0.0 <= resolved.minimum_box_overlap_fraction <= 1.0:
        raise ValueError("minimum box-overlap fraction must lie in [0, 1]")
    return resolved


def _localize_prepared(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    anatomy: AnatomyEvidence,
    policy: LungFieldLocalizationPolicy,
    prepared_masks: dict[LungSide, _PreparedLungMask] | None,
) -> DetectionAnatomyLocation:
    box = _validate_bbox(
        bbox_xyxy,
        width=anatomy.image_width,
        height=anatomy.image_height,
    )
    if anatomy.qc.status == AnatomyQCStatus.FAIL:
        return DetectionAnatomyLocation(
            bbox_xyxy=bbox_xyxy,
            status="invalid_anatomy",
            note="anatomy_qc_failed",
        )
    if prepared_masks is None:  # pragma: no cover - guarded by both public entry points
        raise RuntimeError("prepared lung masks are required for non-failing anatomy")
    box_pixels = (box[2] - box[0]) * (box[3] - box[1])
    assignments = [
        candidate
        for side in LungSide
        if (
            candidate := _assignment(
                prepared=prepared_masks[side],
                bbox=box,
                box_pixels=box_pixels,
            )
        )
        is not None
        and candidate.box_overlap_fraction >= policy.minimum_box_overlap_fraction
    ]
    if not assignments:
        return DetectionAnatomyLocation(
            bbox_xyxy=bbox_xyxy,
            status="outside_lungs",
            note="no_lung_mask_overlap",
        )
    return DetectionAnatomyLocation(
        bbox_xyxy=bbox_xyxy,
        status="localized",
        assignments=assignments,
        note="two_dimensional_lung_field_not_lobe",
    )


def localize_bbox_to_lung_fields(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    anatomy: AnatomyEvidence,
    policy: LungFieldLocalizationPolicy | None = None,
) -> DetectionAnatomyLocation:
    """Locate a box in lung *fields*; this function never infers lung lobes."""

    resolved_policy = _validated_policy(policy)
    # Preserve the public API's fail-fast ordering: geometry was historically
    # rejected before either RLE was decoded.
    _validate_bbox(
        bbox_xyxy,
        width=anatomy.image_width,
        height=anatomy.image_height,
    )
    prepared_masks = (
        None if anatomy.qc.status == AnatomyQCStatus.FAIL else _prepare_lung_masks(anatomy)
    )
    return _localize_prepared(
        bbox_xyxy,
        anatomy=anatomy,
        policy=resolved_policy,
        prepared_masks=prepared_masks,
    )


def localize_detection_boxes(
    boxes: list[tuple[float, float, float, float]],
    *,
    anatomy: AnatomyEvidence,
    policy: LungFieldLocalizationPolicy | None = None,
) -> list[DetectionAnatomyLocation]:
    if len(boxes) > MAX_DETECTION_CANDIDATES:
        raise ValueError(
            f"detector candidate count exceeds the D-FINE limit of "
            f"{MAX_DETECTION_CANDIDATES}"
        )
    # The legacy list-comprehension implementation did no work for an empty
    # batch, including no policy validation or mask decoding.
    if not boxes:
        return []
    resolved_policy = _validated_policy(policy)
    # Match the single-box error order for the first item before shared setup.
    _validate_bbox(
        boxes[0],
        width=anatomy.image_width,
        height=anatomy.image_height,
    )
    prepared_masks = (
        None if anatomy.qc.status == AnatomyQCStatus.FAIL else _prepare_lung_masks(anatomy)
    )
    return [
        _localize_prepared(
            box,
            anatomy=anatomy,
            policy=resolved_policy,
            prepared_masks=prepared_masks,
        )
        for box in boxes
    ]
