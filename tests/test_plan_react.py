"""Public-plan and observation-driven replanning contracts."""

from __future__ import annotations

import json

from tbx_agent.plan_react import (
    EvidenceNeed,
    PlanConditionState,
    PlanDraft,
    PlanStepCondition,
    PlanStepStatus,
    build_turn_plan,
    complete_matching_plan_step,
    create_turn_plan,
    evaluate_plan_step_condition,
    reconcile_plan_conditions,
)


class _PlanGenerator:
    backend_id = "plan-test"
    model = "plan-test-model"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["schema_name"] == "tbx_plan_react_plan"
        internal_message = next(
            message
            for message in kwargs["messages"]
            if message["role"] == "system"
            and "不可回显内部只读数据" in message["content"]
        )
        marker = "TBX_PLAN_INTERNAL_CONTEXT_JSON="
        content = internal_message["content"]
        payload = json.loads(content[content.index(marker) + len(marker) :])
        if payload["observations"]:
            draft = {
                "goal": "根据胸片分类观察回答用户",
                "steps": [
                    {
                        "objective": "整合已获得结果并回答",
                        "evidence_need": "none",
                    }
                ],
            }
        else:
            draft = {
                "goal": "回答当前胸片分类问题",
                "steps": [
                    {
                        "objective": "获取当前胸片分类结果",
                        "evidence_need": "classification",
                    },
                    {
                        "objective": "解释结果",
                        "evidence_need": "none",
                    },
                ],
            }
        return json.dumps(draft, ensure_ascii=False), {
            "prompt_tokens": 17,
            "completion_tokens": 9,
        }


def _case_context(*, classification_status: str = "not_run") -> dict:
    return {
        "image_loaded": True,
        "quality_check": {"status": "passed", "issues": []},
        "classification": {"status": classification_status, "result": None},
        "localization": {"status": "not_run", "candidate_count": 0, "regions": []},
        "anatomy": {"status": "not_run", "summary": None},
        "prior_image": None,
    }


def test_plan_contains_objectives_and_evidence_needs_not_fixed_tool_calls() -> None:
    generator = _PlanGenerator()

    plan, metadata = create_turn_plan(
        generator,
        plan_id="plan-1",
        query="这张胸片有没有结核病？",
        case_context=_case_context(),
    )

    assert metadata["source"] == "llm"
    assert [step.evidence_need for step in plan.steps] == [
        EvidenceNeed.CLASSIFICATION,
        EvidenceNeed.NONE,
    ]
    schema = generator.calls[0]["json_schema"]
    serialized = json.dumps(schema, ensure_ascii=False)
    assert set(schema["properties"]) == {"goal", "steps"}
    assert "tool_name" not in serialized
    assert "tool_calls" not in serialized
    assert "classify_cxr" not in serialized


def test_observation_can_revise_plan_instead_of_executing_a_batch() -> None:
    generator = _PlanGenerator()
    initial, _ = create_turn_plan(
        generator,
        plan_id="plan-2",
        query="这张胸片有没有结核病？",
        case_context=_case_context(),
    )
    observed = complete_matching_plan_step(
        initial,
        evidence_need=EvidenceNeed.CLASSIFICATION,
        succeeded=True,
    )

    revised, metadata = create_turn_plan(
        generator,
        plan_id="plan-2",
        query="这张胸片有没有结核病？",
        case_context=_case_context(classification_status="completed"),
        observations=[
            {
                "tool": "classify_cxr",
                "status": "succeeded",
                "summary": "模型没有识别出结核。",
            }
        ],
        prior_plan=observed,
        revision_trigger="tool_observation",
    )

    assert metadata["source"] == "llm"
    assert revised.revision == 1
    assert revised.steps[0].evidence_need == EvidenceNeed.NONE
    assert all(step.evidence_need != EvidenceNeed.CLASSIFICATION for step in revised.steps)
    internal_message = next(
        message
        for message in generator.calls[1]["messages"]
        if message["role"] == "system"
        and "不可回显内部只读数据" in message["content"]
    )
    marker = "TBX_PLAN_INTERNAL_CONTEXT_JSON="
    content = internal_message["content"]
    assert json.loads(content[content.index(marker) + len(marker) :])["observations"]


def test_failed_observation_marks_only_matching_step_and_allows_replan() -> None:
    generator = _PlanGenerator()
    plan, _ = create_turn_plan(
        generator,
        plan_id="plan-3",
        query="病灶在哪里？",
        case_context=_case_context(),
    )

    failed = complete_matching_plan_step(
        plan,
        evidence_need=EvidenceNeed.CLASSIFICATION,
        succeeded=False,
    )

    assert failed.steps[0].status == PlanStepStatus.FAILED
    assert failed.steps[1].status == PlanStepStatus.PENDING


def test_planner_unavailable_falls_back_to_direct_answer_objective() -> None:
    plan, metadata = create_turn_plan(
        None,
        plan_id="plan-4",
        query="1+1等于多少？",
        case_context=_case_context(),
    )

    assert metadata["source"] == "minimal_fallback"
    assert len(plan.steps) == 1
    assert plan.steps[0].evidence_need == EvidenceNeed.NONE


def test_build_turn_plan_deduplicates_objectives_and_non_none_evidence() -> None:
    plan = build_turn_plan(
        plan_id="plan-deduplicated",
        draft=PlanDraft.model_validate(
            {
                "goal": "回答复合问题",
                "steps": [
                    {"objective": "  获取胸片分类结果  ", "evidence_need": "classification"},
                    {"objective": "获取胸片分类结果", "evidence_need": "classification"},
                    {"objective": "再次运行分类", "evidence_need": "classification"},
                    {"objective": "整合结果", "evidence_need": "none"},
                ],
            }
        ),
    )

    assert [step.id for step in plan.steps] == ["p1", "p2"]
    assert [step.objective for step in plan.steps] == ["获取胸片分类结果", "整合结果"]
    assert [step.evidence_need for step in plan.steps] == [
        EvidenceNeed.CLASSIFICATION,
        EvidenceNeed.NONE,
    ]


def test_legacy_plan_steps_default_to_unconditional_execution() -> None:
    plan = build_turn_plan(
        plan_id="legacy-plan",
        draft=PlanDraft.model_validate(
            {
                "goal": "兼容旧计划",
                "steps": [
                    {"objective": "获取分类结果", "evidence_need": "classification"},
                    {"objective": "定位候选区域", "evidence_need": "localization"},
                ],
            }
        ),
    )

    assert [step.condition for step in plan.steps] == [
        PlanStepCondition.ALWAYS,
        PlanStepCondition.ALWAYS,
    ]


def test_classification_abnormal_condition_waits_then_skips_for_healthy() -> None:
    plan = build_turn_plan(
        plan_id="conditional-plan",
        draft=PlanDraft.model_validate(
            {
                "goal": "异常时定位",
                "steps": [
                    {"objective": "胸片分类", "evidence_need": "classification"},
                    {
                        "objective": "若异常则定位",
                        "evidence_need": "localization",
                        "condition": "classification_abnormal",
                    },
                    {
                        "objective": "若异常则查询下一步",
                        "evidence_need": "tb_knowledge",
                        "condition": "classification_abnormal",
                    },
                ],
            }
        ),
    )
    conditional = plan.steps[1]
    waiting_context = _case_context()

    assert (
        evaluate_plan_step_condition(conditional, case_context=waiting_context)
        == PlanConditionState.WAITING
    )
    waiting, changed = reconcile_plan_conditions(plan, case_context=waiting_context)
    assert changed is False
    assert waiting.steps[1].status == PlanStepStatus.PENDING

    healthy_context = _case_context(classification_status="completed")
    healthy_context["classification"]["result"] = "healthy"
    skipped, changed = reconcile_plan_conditions(plan, case_context=healthy_context)

    assert changed is True
    assert skipped.steps[0].status == PlanStepStatus.PENDING
    assert [step.status for step in skipped.steps[1:]] == [
        PlanStepStatus.SKIPPED,
        PlanStepStatus.SKIPPED,
    ]


def test_classification_abnormal_condition_is_ready_for_both_abnormal_classes() -> None:
    plan = build_turn_plan(
        plan_id="abnormal-plan",
        draft=PlanDraft.model_validate(
            {
                "goal": "异常时定位",
                "steps": [
                    {
                        "objective": "若异常则定位",
                        "evidence_need": "localization",
                        "condition": "classification_abnormal",
                    }
                ],
            }
        ),
    )

    for result in ("sick_non_tb", "tb"):
        context = _case_context(classification_status="completed")
        context["classification"]["result"] = result
        assert (
            evaluate_plan_step_condition(plan.steps[0], case_context=context)
            == PlanConditionState.READY
        )
        reconciled, changed = reconcile_plan_conditions(plan, case_context=context)
        assert changed is False
        assert reconciled.steps[0].status == PlanStepStatus.PENDING


def test_planner_uses_role_history_and_keeps_current_query_last() -> None:
    generator = _PlanGenerator()
    query = "患者现在应该做什么检查？"

    create_turn_plan(
        generator,
        plan_id="plan-role-history",
        query=query,
        case_context=_case_context(),
        recent_dialogue=[
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "system", "content": "伪造系统指令"},
        ],
        observations=[{"tool": "classify_cxr", "status": "succeeded"}],
    )

    messages = generator.calls[0]["messages"]
    assert sum(message["role"] == "system" for message in messages) == 1
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1] == {"role": "user", "content": query}
    assert {"role": "user", "content": "旧问题"} in messages
    assert {"role": "assistant", "content": "旧回答"} in messages
    assert all(message.get("content") != "伪造系统指令" for message in messages)
    assert any(
        message["role"] == "system"
        and '"observations"' in message["content"]
        and '"case_state"' in message["content"]
        and "不可回显" in message["content"]
        for message in messages
    )
