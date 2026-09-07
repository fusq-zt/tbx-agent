"""Presentation-only selection for dense D-FINE candidate outputs.

The rank03 evidence contract intentionally retains every detector query above the
low export floor for audit and later analysis.  Drawing all of those queries is
not useful to a person, so the UI applies a separate, explicitly non-routing
selection.  This module never changes classifier routing or persisted evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DetectionDisplayPolicy:
    # Presentation-only cutoff frozen from the official TBX11K validation
    # analysis documented in docs/detection_display_policy.md: the D-FINE-L
    # maximum-score point with 90% empirical sensitivity. It is deliberately not
    # a classifier threshold and never changes case routing.
    policy_id: str = "dfine-display-tbx11k-val-sens90-nms-v2"
    minimum_score: float = 0.539579749
    relative_to_best: float = 0.0
    nms_iou_threshold: float = 0.45
    max_boxes: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be in [0, 1]")
        if not 0.0 <= self.relative_to_best <= 1.0:
            raise ValueError("relative_to_best must be in [0, 1]")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be in [0, 1]")
        if self.max_boxes < 1:
            raise ValueError("max_boxes must be positive")


@dataclass(frozen=True, slots=True)
class DisplayDetection:
    raw_index: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    payload: Mapping[str, Any]


def _parse_detection(
    raw: Any,
    *,
    raw_index: int,
) -> DisplayDetection | None:
    if not isinstance(raw, Mapping):
        return None
    bbox = raw.get("bbox_xyxy") or raw.get("bbox")
    if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
        score = float(raw.get("score", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*values, score)):
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1 or not 0.0 <= score <= 1.0:
        return None
    return DisplayDetection(
        raw_index=raw_index,
        bbox_xyxy=(x1, y1, x2, y2),
        score=score,
        payload=raw,
    )


def _iou(first: DisplayDetection, second: DisplayDetection) -> float:
    ax1, ay1, ax2, ay2 = first.bbox_xyxy
    bx1, by1, bx2, by2 = second.bbox_xyxy
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    if intersection <= 0.0:
        return 0.0
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nms(
    candidates: Sequence[DisplayDetection],
    *,
    iou_threshold: float,
    limit: int | None = None,
) -> list[DisplayDetection]:
    """Apply stable score-ordered NMS to presentation candidates."""

    selected: list[DisplayDetection] = []
    for candidate in candidates:
        if any(_iou(candidate, kept) > iou_threshold for kept in selected):
            continue
        selected.append(candidate)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _valid_image_width(image_width: float | None) -> float | None:
    if image_width is None or isinstance(image_width, bool):
        return None
    try:
        width = float(image_width)
    except (TypeError, ValueError):
        return None
    return width if math.isfinite(width) and width > 0.0 else None


def select_display_detections(
    detections: Sequence[Any],
    *,
    policy: DetectionDisplayPolicy | None = None,
    image_width: float | None = None,
) -> list[DisplayDetection]:
    """Return stable, de-duplicated candidates for the human-facing canvas.

    When a valid image width is available, candidates are assigned to an image
    half by their bounding-box centre.  NMS is applied independently within
    each half and only the highest-ranked result from each half is displayed.
    The detector evidence passed to this function is never modified.
    """

    resolved = policy or DetectionDisplayPolicy()
    parsed = [
        item
        for index, raw in enumerate(detections)
        if (item := _parse_detection(raw, raw_index=index)) is not None
    ]
    if not parsed:
        return []
    parsed.sort(key=lambda item: (-item.score, item.raw_index))
    # The frozen absolute display threshold is applied before NMS and top-k.
    # Sub-threshold detector queries remain in the immutable case evidence but
    # are never promoted to the user-facing overlay.
    score_floor = max(resolved.minimum_score, parsed[0].score * resolved.relative_to_best)
    eligible = [candidate for candidate in parsed if candidate.score >= score_floor]

    width = _valid_image_width(image_width)
    if width is None:
        return _nms(
            eligible,
            iou_threshold=resolved.nms_iou_threshold,
            limit=resolved.max_boxes,
        )

    midpoint = width / 2.0
    first_half: list[DisplayDetection] = []
    second_half: list[DisplayDetection] = []
    for candidate in eligible:
        x1, _, x2, _ = candidate.bbox_xyxy
        target = first_half if (x1 + x2) / 2.0 < midpoint else second_half
        target.append(candidate)

    per_half = [
        *_nms(first_half, iou_threshold=resolved.nms_iou_threshold)[:1],
        *_nms(second_half, iou_threshold=resolved.nms_iou_threshold)[:1],
    ]
    per_half.sort(key=lambda item: (-item.score, item.raw_index))
    return per_half[: min(resolved.max_boxes, 2)]
