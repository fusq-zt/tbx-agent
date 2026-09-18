from tbx_agent.plan_execution import prepare_evidence_plan, preserve_plan_obligations
from tbx_agent.plan_react import EvidenceNeed, PlanDraft, build_turn_plan


def _plan(*needs):
    return build_turn_plan(plan_id="execution", draft=PlanDraft.model_validate({
        "goal": "保持本轮目标",
        "steps": [{"objective": need, "evidence_need": need} for need in needs],
    }))


def test_anatomy_dependency_preserves_condition_and_cached_classification():
    plan = _plan("lung_anatomy")
    plan.steps[0].condition = "classification_abnormal"
    prepared = prepare_evidence_plan(plan, {
        "classification": {"status": "completed", "result": "tb"},
        "localization": {"status": "not_requested"},
        "anatomy": {"status": "not_requested"},
    })
    assert [step.evidence_need for step in prepared.steps] == [
        EvidenceNeed.LOCALIZATION, EvidenceNeed.LUNG_ANATOMY,
    ]
    assert all(step.condition == "classification_abnormal" for step in prepared.steps)


def test_replanning_cannot_drop_obligations_or_reopen_failed_tools():
    before = _plan("classification", "localization", "tb_knowledge")
    before.steps[0].status = "failed"
    revised = _plan("none")
    revised.revision = 1
    preserved = preserve_plan_obligations(before, revised)
    assert [step.evidence_need for step in preserved.steps] == [
        EvidenceNeed.CLASSIFICATION, EvidenceNeed.LOCALIZATION, EvidenceNeed.TB_KNOWLEDGE,
    ]
    assert [step.status for step in preserved.steps] == ["failed", "pending", "pending"]
    assert preserved.revision == 1


def test_replan_cannot_add_unrequested_vision_tool_or_change_condition():
    before = _plan("tb_knowledge")
    revised = _plan("localization", "tb_knowledge")
    revised.steps[1].condition = "classification_abnormal"
    preserved = preserve_plan_obligations(before, revised)
    assert [step.evidence_need for step in preserved.steps] == [EvidenceNeed.TB_KNOWLEDGE]
    assert preserved.steps[0].condition == "always"


def test_repeated_public_wording_cannot_erase_distinct_evidence_tasks():
    plan = build_turn_plan(plan_id="same-wording", draft=PlanDraft.model_validate({
        "goal": "分析胸片",
        "steps": [
            {"objective": "分析胸片", "evidence_need": "classification"},
            {"objective": "分析胸片", "evidence_need": "localization"},
        ],
    }))
    assert [step.evidence_need for step in plan.steps] == [
        EvidenceNeed.CLASSIFICATION, EvidenceNeed.LOCALIZATION,
    ]
