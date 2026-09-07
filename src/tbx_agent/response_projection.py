"""Pure projections of code-owned capabilities and authorized public case state.

This module performs no model calls, tool execution or persistence. Callers must
resolve case access and run the emergency guard before using its answer.
"""

from __future__ import annotations

import re
from typing import Any

from .capability_answer import TBX_CAPABILITY_ANSWER
from .task_spec import TaskGoal, parse_task_spec

_PROJECTION_GOALS = {TaskGoal.CAPABILITIES, TaskGoal.CASE_STATUS, TaskGoal.SOCIAL}
# Separate independent requests before applying the status parser, which
# intentionally suppresses classification/localization phrases in status-only
# questions. Otherwise "summarize completed analysis, then analyze this image"
# can incorrectly become a zero-tool answer.
_REQUEST_BOUNDARY = re.compile(r"[，,。！？!?；;\n]+|(?:并且|并|然后|同时|另外|顺便|还要|以及)")


def projection_task_goals(query: str) -> set[TaskGoal]:
    """Keep explicit compound tool goals when a query also requests a projection.

    This is a narrow supplement to TaskSpec, not a separate tool router. Plain
    tool and general-chat requests use the existing parser unchanged. Clauses
    that carry only introductory prose do not grant any additional action.
    """
    goals = set(parse_task_spec(query).task_goals)
    if not goals & {TaskGoal.CAPABILITIES, TaskGoal.CASE_STATUS}:
        return goals
    for clause in _REQUEST_BOUNDARY.split(query):
        if clause.strip():
            goals.update(set(parse_task_spec(clause).task_goals) - {TaskGoal.GENERAL_CHAT})
    return goals


def trusted_non_tool_answer(
    query: str,
    *,
    case_context: dict[str, Any],
) -> str | None:
    """Answer system metadata and cached-case summaries without model invention.

    These are not tools because they create no new evidence.  They are also not
    delegated to the small orchestration model: capability metadata is owned by
    this runtime, while a case summary must be a projection of the public case
    context rather than a paraphrase of the private Plan/ReAct payload.
    """

    goals = projection_task_goals(query)
    if goals - _PROJECTION_GOALS:
        return None
    if TaskGoal.CAPABILITIES in goals:
        if TaskGoal.CASE_STATUS in goals:
            return TBX_CAPABILITY_ANSWER + "\n\n" + _case_status_answer(query, case_context)
        return TBX_CAPABILITY_ANSWER
    if TaskGoal.CASE_STATUS not in goals:
        return None

    return _case_status_answer(query, case_context)


def _case_status_answer(query: str, case_context: dict[str, Any]) -> str:
    concise_summary = any(
        marker in query
        for marker in ("分析摘要", "辅助分析摘要", "已完成的分析", "已经完成的分析")
    )

    if not case_context.get("image_loaded"):
        return "当前没有已加载的胸片，也没有可汇总的影像分析结果。"

    completed: list[str] = []
    classification = case_context.get("classification") or {}
    if classification.get("status") == "completed":
        classification_label = {
            "healthy": "健康类",
            "sick_non_tb": "非结核异常类",
            "tb": "结核类",
        }.get(str(classification.get("result")))
        if classification_label is not None:
            completed.append(f"胸片分类模型更倾向于{classification_label}。")
        else:
            completed.append("胸片分类已完成，但本轮没有形成可展示的单一类别。")
    elif not concise_summary:
        classification_status = {
            "failed": "运行失败",
            "unavailable": "暂不可用",
        }.get(str(classification.get("status")), "未运行")
        completed.append(f"当前分类{classification_status}。")

    localization = case_context.get("localization") or {}
    anatomy = case_context.get("anatomy") or {}
    anatomy_summary = str(anatomy.get("summary") or "").strip()
    if anatomy.get("status") == "completed" and anatomy_summary:
        completed.append(anatomy_summary)
    elif localization.get("status") == "completed_no_detection":
        completed.append("候选区域定位已完成，未发现达到显示门槛的候选区域。")
    elif localization.get("status") == "completed":
        candidate_count = localization.get("candidate_count")
        regions = [
            str(item).strip()
            for item in localization.get("regions") or []
            if str(item).strip()
        ]
        if isinstance(candidate_count, int) and candidate_count > 0:
            location = f"，大致位于{'、'.join(regions)}" if regions else ""
            completed.append(f"定位显示 {candidate_count} 个候选区域{location}。")
        else:
            completed.append("候选区域定位已完成。")
    elif not concise_summary:
        localization_status = {
            "failed": "运行失败",
            "unsupported": "暂不可用",
            "stale": "结果已过期",
        }.get(str(localization.get("status")), "未运行")
        completed.append(f"候选区域定位{localization_status}。")

    if not completed:
        return "胸片已载入，目前还没有完成可汇总的分类、定位或肺野分析。"
    prefix = "AI 辅助分析摘要：" if concise_summary else ""
    return prefix + "".join(completed)

