"""State-only validation of model plans. No user text or keyword router belongs here."""

from __future__ import annotations

from typing import Any

from .plan_react import (
    AnswerFocus,
    EvidenceNeed,
    PlanStep,
    PlanStepStatus,
    TurnPlan,
    reconcile_plan_conditions,
)


def prepare_evidence_plan(plan: TurnPlan, case_context: dict[str, Any]) -> TurnPlan:
    """Reuse authorized evidence and add genuine tool dependencies, without intent vetoes."""
    plan, _ = reconcile_plan_conditions(plan, case_context=case_context)
    ready = {
        EvidenceNeed.CLASSIFICATION: case_context.get("classification", {}).get("status")
        == "completed",
        EvidenceNeed.LOCALIZATION: case_context.get("localization", {}).get("status")
        in {"completed", "completed_no_detection"},
        EvidenceNeed.LUNG_ANATOMY: case_context.get("anatomy", {}).get("status") == "completed",
    }
    steps = list(plan.steps)
    if plan.answer_focus == AnswerFocus.CLASSIFICATION_RATIONALE and not any(
        step.evidence_need == EvidenceNeed.CLASSIFICATION for step in steps
    ):
        steps = [step for step in steps if step.evidence_need != EvidenceNeed.NONE]
        steps.insert(0, PlanStep(
            id="p1", objective="读取分类依据", evidence_need=EvidenceNeed.CLASSIFICATION,
        ))
    steps = [
        step.model_copy(update={"status": PlanStepStatus.COMPLETED})
        if step.status == PlanStepStatus.PENDING and ready.get(step.evidence_need, False)
        else step
        for step in steps
    ]
    anatomy = next((step for step in steps if (
        step.evidence_need == EvidenceNeed.LUNG_ANATOMY
        and step.status == PlanStepStatus.PENDING
    )), None)
    if anatomy is not None and not ready[EvidenceNeed.LOCALIZATION] and not any(
        step.evidence_need == EvidenceNeed.LOCALIZATION for step in steps
    ):
        steps = [step for step in steps if step.evidence_need != EvidenceNeed.NONE]
        steps.insert(steps.index(anatomy), anatomy.model_copy(update={
            "objective": "获取肺野空间分析所需的候选区域",
            "evidence_need": EvidenceNeed.LOCALIZATION,
        }))
    return plan.model_copy(update={"steps": [
        step.model_copy(update={"id": f"p{index}"})
        for index, step in enumerate(steps, 1)
    ]})


def preserve_plan_obligations(before: TurnPlan, revised: TurnPlan) -> TurnPlan:
    """A failed tool cannot erase a user's pending objective or reopen an attempted one.

    Replanning may reorder the original evidence objectives. It cannot silently
    enlarge the tool scope or replace the turn's presentation intent.
    """
    original = {step.evidence_need: step for step in before.steps
                if step.evidence_need != EvidenceNeed.NONE}
    if not original:
        return revised
    needs = list(dict.fromkeys([
        *(step.evidence_need for step in revised.steps if step.evidence_need in original),
        *original,
    ]))
    return revised.model_copy(update={
        "goal": before.goal,
        "answer_focus": before.answer_focus,
        "steps": [original[need].model_copy(update={"id": f"p{index}"})
                  for index, need in enumerate(needs, 1)],
    })


def unfinished_evidence(plan: TurnPlan) -> list[EvidenceNeed]:
    """Both ready and prerequisite-blocked objectives count as unfinished."""
    return [step.evidence_need for step in plan.steps
            if step.evidence_need != EvidenceNeed.NONE and step.status == PlanStepStatus.PENDING]
