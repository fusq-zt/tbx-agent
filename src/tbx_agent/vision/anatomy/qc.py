"""Deterministic quality-control gates for paired lung masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import AnatomyQCReport, AnatomyQCStatus
from .rle import _shape_and_flatten


@dataclass(frozen=True, slots=True)
class AnatomyQCPolicy:
    policy_id: str = "paired-lung-qc-v1"
    min_lung_fraction: float = 0.025
    max_lung_fraction: float = 0.55
    min_left_right_area_ratio: float = 0.35
    max_left_right_area_ratio: float = 2.85
    max_overlap_fraction_of_smaller_lung: float = 0.03
    max_border_fraction: float = 0.08
    expected_left_centroid_to_right_of_right: bool = True
    max_preprocessing_crop_fraction: float = 0.15

    def generation_parameters(self) -> dict[str, object]:
        return asdict(self)


def _centroid_x(values: list[bool], width: int) -> float:
    positions = [index % width for index, value in enumerate(values) if value]
    return sum(positions) / len(positions) if positions else -1.0


def _border_count(values: list[bool], width: int, height: int) -> int:
    indices: set[int] = set(range(width))
    indices.update(range((height - 1) * width, height * width))
    indices.update(row * width for row in range(height))
    indices.update(row * width + width - 1 for row in range(height))
    return sum(bool(values[index]) for index in indices)


def evaluate_lung_masks(
    left_mask: object,
    right_mask: object,
    *,
    policy: AnatomyQCPolicy | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
    preprocessing_crop_fraction: float = 0.0,
) -> AnatomyQCReport:
    policy = policy or AnatomyQCPolicy()
    left_width, left_height, left = _shape_and_flatten(left_mask)
    right_width, right_height, right = _shape_and_flatten(right_mask)
    if (left_width, left_height) != (right_width, right_height):
        return AnatomyQCReport(
            status=AnatomyQCStatus.FAIL,
            codes=["left_right_mask_dimensions_mismatch"],
            metrics={
                "left_width": float(left_width),
                "left_height": float(left_height),
                "right_width": float(right_width),
                "right_height": float(right_height),
            },
        )
    width_mismatch = source_width is not None and source_width != left_width
    height_mismatch = source_height is not None and source_height != left_height
    if width_mismatch or height_mismatch:
        expected_width = source_width if source_width is not None else left_width
        expected_height = source_height if source_height is not None else left_height
        code = (
            "mask_canvas_out_of_source_bounds"
            if left_width > expected_width or left_height > expected_height
            else "mask_source_dimensions_mismatch"
        )
        return AnatomyQCReport(
            status=AnatomyQCStatus.FAIL,
            codes=[code],
            metrics={
                "mask_width": float(left_width),
                "mask_height": float(left_height),
                "source_width": float(expected_width),
                "source_height": float(expected_height),
            },
        )
    if not 0.0 <= preprocessing_crop_fraction < 1.0:
        raise ValueError("preprocessing crop fraction must lie in [0, 1)")

    canvas = left_width * left_height
    left_area = sum(left)
    right_area = sum(right)
    smaller = min(left_area, right_area)
    overlap = sum(a and b for a, b in zip(left, right, strict=True))
    left_fraction = left_area / canvas
    right_fraction = right_area / canvas
    area_ratio = left_area / right_area if right_area else 0.0
    border = _border_count(left, left_width, left_height) + _border_count(
        right, right_width, right_height
    )
    total_area = left_area + right_area
    border_fraction = border / total_area if total_area else 0.0
    overlap_fraction = overlap / smaller if smaller else 0.0
    left_centroid_x = _centroid_x(left, left_width)
    right_centroid_x = _centroid_x(right, right_width)

    failures: list[str] = []
    warnings: list[str] = []
    if left_area == 0:
        failures.append("left_lung_empty")
    if right_area == 0:
        failures.append("right_lung_empty")
    if left_area and not policy.min_lung_fraction <= left_fraction <= policy.max_lung_fraction:
        failures.append("left_lung_area_out_of_range")
    if right_area and not policy.min_lung_fraction <= right_fraction <= policy.max_lung_fraction:
        failures.append("right_lung_area_out_of_range")
    if left_area and right_area and not (
        policy.min_left_right_area_ratio
        <= area_ratio
        <= policy.max_left_right_area_ratio
    ):
        failures.append("left_right_area_ratio_out_of_range")
    if smaller and overlap_fraction > policy.max_overlap_fraction_of_smaller_lung:
        failures.append("lung_masks_overlap_excessively")
    if total_area and border_fraction > policy.max_border_fraction:
        warnings.append("lung_mask_touches_image_border")
    if preprocessing_crop_fraction > policy.max_preprocessing_crop_fraction:
        warnings.append("preprocessing_crop_fraction_high")
    if (
        left_area
        and right_area
        and policy.expected_left_centroid_to_right_of_right
        and left_centroid_x <= right_centroid_x
    ):
        warnings.append("left_right_order_unexpected")

    codes = failures + warnings
    status = (
        AnatomyQCStatus.FAIL
        if failures
        else AnatomyQCStatus.WARNING
        if warnings
        else AnatomyQCStatus.PASS
    )
    return AnatomyQCReport(
        status=status,
        codes=codes,
        metrics={
            "left_lung_fraction": left_fraction,
            "right_lung_fraction": right_fraction,
            "left_right_area_ratio": area_ratio,
            "overlap_fraction_of_smaller_lung": overlap_fraction,
            "border_fraction": border_fraction,
            "left_centroid_x": left_centroid_x,
            "right_centroid_x": right_centroid_x,
            "preprocessing_crop_fraction": preprocessing_crop_fraction,
        },
    )
