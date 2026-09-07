"""Deterministic presentation of detector-to-lung-mask relationships.

This module intentionally does not identify radiographic signs, lobes, disease,
activity, infectiousness, or drug resistance.  It only verbalizes already
validated source-pixel geometry so the UI and the constrained narrator share the
same evidence contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    AnatomyQCStatus,
    AnatomySpatialSummary,
    DetectionAnatomyLocation,
    LungFieldZone,
    LungSide,
)

SPATIAL_PRESENTATION_POLICY_ID = "detector-lung-field-presentation-v1"
MAX_PRESENTED_CANDIDATES = 24

_SIDE_LABELS = {
    LungSide.LEFT: "左侧肺野",
    LungSide.RIGHT: "右侧肺野",
}
_ZONE_LABELS = {
    LungFieldZone.UPPER: "上肺野",
    LungFieldZone.MIDDLE: "中肺野",
    LungFieldZone.LOWER: "下肺野",
}
_CHAT_SIDE_LABELS = {
    LungSide.RIGHT: "右",
    LungSide.LEFT: "左",
}
_CHAT_SIDE_ORDER = {
    LungSide.RIGHT: 0,
    LungSide.LEFT: 1,
}
_CHAT_ZONE_ORDER = {
    LungFieldZone.UPPER: 0,
    LungFieldZone.MIDDLE: 1,
    LungFieldZone.LOWER: 2,
}


def _percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def _localized_statement(index: int, location: DetectionAnatomyLocation) -> str:
    assignments = sorted(
        location.assignments,
        key=lambda item: (
            list(LungSide).index(item.lung),
            list(LungFieldZone).index(item.primary_zone),
        ),
    )
    parts = [
        (
            f"{_SIDE_LABELS[item.lung]}的{_ZONE_LABELS[item.primary_zone]}二维投影"
            f"（框内肺野掩膜交叠 {_percentage(item.box_overlap_fraction)}，"
            f"占该侧肺野 {_percentage(item.lung_overlap_fraction)}）"
        )
        for item in assignments
    ]
    return (
        f"候选框 {index} 与" + "；".join(parts) + "相交。"
        "这里的上/中/下仅指二维肺野分区，不代表肺叶或病灶诊断。"
    )


def _join_chinese(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]}和{items[1]}"
    return "、".join(items[:-1]) + f"和{items[-1]}"


def build_chat_spatial_summary(
    locations: Sequence[DetectionAnatomyLocation],
    *,
    anatomy_qc_status: AnatomyQCStatus,
) -> str:
    """Return a short, grouped location sentence for the chat surface.

    The persisted :class:`AnatomySpatialSummary` and detector-location DTOs keep
    every per-box overlap metric for audit and the case-detail view.  This
    formatter deliberately exposes only the distinct side/zone relationships
    and a compact duplicate count so the main answer does not repeat one long
    paragraph per detector box.
    """

    if not locations:
        return "当前没有可进行肺野定位的候选区域。"

    grouped: dict[tuple[LungSide, LungFieldZone], int] = {}
    for location in locations:
        if location.status != "localized":
            continue
        # Count a detector candidate at most once in each side/zone even if a
        # malformed legacy payload contains a duplicated assignment.
        candidate_groups = {
            (assignment.lung, assignment.primary_zone)
            for assignment in location.assignments
        }
        for group in candidate_groups:
            grouped[group] = grouped.get(group, 0) + 1

    outside_count = sum(item.status == "outside_lungs" for item in locations)
    invalid_count = sum(item.status == "invalid_anatomy" for item in locations)

    if anatomy_qc_status == AnatomyQCStatus.FAIL or (invalid_count and not grouped):
        return "肺野分割质量未通过，暂时无法判断候选区域位置。"

    if grouped:
        ordered = sorted(
            grouped,
            key=lambda item: (
                _CHAT_SIDE_ORDER[item[0]],
                _CHAT_ZONE_ORDER[item[1]],
            ),
        )
        labels = []
        for side, zone in ordered:
            label = f"{_CHAT_SIDE_LABELS[side]}{_ZONE_LABELS[zone]}"
            count = grouped[(side, zone)]
            labels.append(f"{label}（{count} 个）" if count > 1 else label)
        summary = f"候选区域位于{_join_chinese(labels)}。"
    else:
        summary = "候选区域未定位到肺野。"

    if outside_count:
        summary += f"另有 {outside_count} 个候选区域未定位到肺野。"
    if anatomy_qc_status == AnatomyQCStatus.WARNING:
        summary += "肺野分割存在质量警告。"
    return summary


def build_spatial_summary(
    locations: Sequence[DetectionAnatomyLocation],
    *,
    anatomy_qc_status: AnatomyQCStatus,
    policy_id: str = SPATIAL_PRESENTATION_POLICY_ID,
) -> AnatomySpatialSummary:
    """Build the complete safe statement allowlist for one anatomy run."""

    localized = sum(item.status == "localized" for item in locations)
    outside = sum(item.status == "outside_lungs" for item in locations)
    invalid = sum(item.status == "invalid_anatomy" for item in locations)
    statements: list[str] = []

    if not locations:
        statements.append(
            "D-FINE 未返回可用于空间关系计算的候选框；这不能排除影像异常或肺结核。"
        )
    else:
        for index, location in enumerate(locations[:MAX_PRESENTED_CANDIDATES], start=1):
            if location.status == "localized":
                statements.append(_localized_statement(index, location))
            elif location.status == "outside_lungs":
                statements.append(
                    f"候选框 {index} 未达到肺野掩膜交叠门槛，标记为肺野外/未定位候选；"
                    "这不等同于肺外异常或病变判断。"
                )
            else:
                statements.append(
                    f"候选框 {index} 因肺野分割质量控制未通过而停止空间定位；"
                    "系统不输出左右侧或上/中/下肺野结论。"
                )
        omitted = len(locations) - MAX_PRESENTED_CANDIDATES
        if omitted > 0:
            statements.append(
                f"另有 {omitted} 个候选框保留在结构化证据中；受控文本层不逐一展开。"
            )

    if anatomy_qc_status == AnatomyQCStatus.WARNING:
        statements.append(
            "肺野分割质量控制存在警告，空间关系仅作为候选区域复核信息。"
        )
    elif anatomy_qc_status == AnatomyQCStatus.FAIL and not invalid:
        # This branch also makes corrupted legacy payloads fail visibly instead
        # of presenting apparently valid geometry.
        statements.append(
            "肺野分割质量控制未通过，已有空间关系不得作为可用定位证据。"
        )

    statements.append("肺野分割和候选定位不改变胸片三分类结果。")
    return AnatomySpatialSummary(
        policy_id=policy_id,
        anatomy_qc_status=anatomy_qc_status,
        candidate_count=len(locations),
        localized_count=localized,
        outside_lungs_count=outside,
        invalid_anatomy_count=invalid,
        statements=statements,
    )
