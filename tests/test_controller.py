import json
from types import SimpleNamespace

from tbx_agent.agent_state import (
    AgentAction,
    ConflictFlag,
    EvidenceKind,
    EvidenceStatus,
    build_case_state,
)
from tbx_agent.controller import (
    ActionCapability,
    AgentBudget,
    BoundedAgentController,
    DecisionSource,
    SelectionReasonCode,
    consume_budget,
    controller_state_view,
)
from tbx_agent.task_spec import parse_task_spec


def _empty_case(**overrides):
    values = {
        "case_id": "case-1",
        "user_id": "user-1",
        "classification_status": "not_requested",
        "vision_evidence": None,
        "localization_evidence": SimpleNamespace(
            status="not_requested", detections=[], reason_codes=[]
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _case_with_advisory_detector_mismatch():
    return _empty_case(
        classification_status="completed",
        vision_evidence=SimpleNamespace(
            predicted_class="healthy",
            class_probabilities={"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05},
            class_probability_order=["healthy", "sick_non_tb", "tb"],
            image_quality_status="transport_valid",
            image_quality_codes=[],
            classifier_threshold=None,
        ),
        localization_evidence=SimpleNamespace(
            status="completed",
            detections=[SimpleNamespace(score=0.8)],
            reason_codes=[],
        ),
    )


def _capabilities() -> list[ActionCapability]:
    return [
        ActionCapability(action=AgentAction.CLASSIFY_CURRENT_CXR),
        ActionCapability(
            action=AgentAction.LOCALIZE_CURRENT_CXR,
            cost_units=3,
            expensive_vision=True,
        ),
        ActionCapability(action=AgentAction.INSPECT_IMAGE_QUALITY),
        ActionCapability(action=AgentAction.INSPECT_ANATOMICAL_CONTEXT, cost_units=3),
        ActionCapability(action=AgentAction.RETRIEVE_PRIOR_STUDIES),
        ActionCapability(action=AgentAction.RETRIEVE_GUIDELINE, cost_units=2),
    ]


def test_controller_selects_one_missing_evidence_action() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？病灶在哪里？")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(),
    )

    assert decision.action == AgentAction.CLASSIFY_CURRENT_CXR
    assert decision.allowed_actions[0] == AgentAction.CLASSIFY_CURRENT_CXR
    assert not hasattr(decision, "actions")


def test_observation_changes_the_next_action_for_compound_task() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？病灶在哪里？")
    pending = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )
    available = SimpleNamespace(
        case_id="case-1",
        user_id="user-1",
        classification_status="completed",
        vision_evidence=SimpleNamespace(
            predicted_class="tb",
            class_probabilities={"healthy": 0.01, "sick_non_tb": 0.09, "tb": 0.9},
            class_probability_order=["healthy", "sick_non_tb", "tb"],
            image_quality_status="transport_valid",
            image_quality_codes=[],
            classifier_threshold=None,
        ),
        localization_evidence=SimpleNamespace(
            status="not_requested", detections=[], reason_codes=[]
        ),
    )
    after_classification = build_case_state(
        available,
        spec.current_query,
        required_evidence=spec.required_evidence,
        completed_actions=[AgentAction.CLASSIFY_CURRENT_CXR],
        tool_calls_used=1,
    )
    controller = BoundedAgentController()

    first = controller.choose_next_action(
        state=pending,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(),
    )
    second = controller.choose_next_action(
        state=after_classification,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(steps_used=1, tool_calls_used=1, cost_units_used=1),
    )

    assert first.action == AgentAction.CLASSIFY_CURRENT_CXR
    assert second.action == AgentAction.LOCALIZE_CURRENT_CXR


def test_completed_classification_is_not_reexecuted_for_screening() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？")
    case = _empty_case(
        classification_status="completed",
        vision_evidence=SimpleNamespace(
            predicted_class="healthy",
            class_probabilities={"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05},
            class_probability_order=["healthy", "sick_non_tb", "tb"],
            image_quality_status="transport_valid",
            image_quality_codes=[],
            classifier_threshold=None,
        ),
    )
    state = build_case_state(
        case,
        spec.current_query,
        required_evidence=spec.required_evidence,
        completed_actions=[AgentAction.CLASSIFY_CURRENT_CXR],
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(tool_calls_used=1, cost_units_used=2),
    )

    assert state.classification_evidence.status == EvidenceStatus.AVAILABLE
    assert decision.action == AgentAction.STOP
    assert AgentAction.CLASSIFY_CURRENT_CXR not in decision.allowed_actions


def test_no_tool_query_stops_without_consuming_tool_budget() -> None:
    spec = parse_task_spec("你好")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )
    budget = AgentBudget()

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=budget,
    )
    consumed = consume_budget(budget, decision)

    assert decision.action == AgentAction.STOP
    assert decision.reason_code == SelectionReasonCode.NO_TOOL_TASK
    assert consumed.tool_calls_used == 0


def test_no_tool_query_stops_even_when_bound_case_has_prior_conflict() -> None:
    spec = parse_task_spec("谢谢")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    ).model_copy(
        update={
            "conflict_flags": [ConflictFlag.TOOL_FAILURE_CONFLICT],
            "failed_actions": [AgentAction.LOCALIZE_CURRENT_CXR],
        }
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(),
    )

    assert decision.action == AgentAction.STOP
    assert decision.reason_code == SelectionReasonCode.NO_TOOL_TASK


def test_budget_defaults_match_compound_agent_runtime() -> None:
    budget = AgentBudget()

    assert budget.max_steps == 5
    assert budget.max_tool_calls == 4
    assert budget.max_expensive_vision_calls == 3
    assert budget.max_cost_units == 10


def test_budget_guard_is_a_first_class_human_action() -> None:
    spec = parse_task_spec("病灶在哪里？")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=[EvidenceKind.LOCALIZATION],
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(
            max_steps=3,
            max_tool_calls=2,
            steps_used=3,
            tool_calls_used=2,
        ),
    )

    assert decision.action == AgentAction.REFER_TO_HUMAN
    assert decision.reason_code == SelectionReasonCode.BUDGET_EXHAUSTED
    assert decision.source == DecisionSource.HARD_GUARD


def test_controller_view_excludes_identity_pixels_and_credentials() -> None:
    spec = parse_task_spec("病灶在哪里？")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )

    rendered = str(controller_state_view(state, spec)).casefold()

    assert "user-1" not in rendered
    assert "api_key" not in rendered
    assert "pixel" not in rendered
    assert "bbox" not in rendered


def test_completed_empty_localization_is_not_retried() -> None:
    spec = parse_task_spec("病灶在哪里？")
    case = _empty_case(
        localization_evidence=SimpleNamespace(
            status=EvidenceStatus.COMPLETED_NO_DETECTION,
            detections=[],
            reason_codes=[],
        )
    )
    state = build_case_state(
        case,
        spec.current_query,
        required_evidence=spec.required_evidence,
        completed_actions=[AgentAction.LOCALIZE_CURRENT_CXR],
        tool_calls_used=1,
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(steps_used=1, tool_calls_used=1, cost_units_used=3),
    )

    assert decision.action == AgentAction.STOP
    assert AgentAction.LOCALIZE_CURRENT_CXR not in decision.allowed_actions


def test_completed_advisory_localization_stops_without_evidence_conflict() -> None:
    spec = parse_task_spec("病灶在哪里？")
    state = build_case_state(
        _case_with_advisory_detector_mismatch(),
        spec.current_query,
        required_evidence=spec.required_evidence,
        completed_actions=[AgentAction.LOCALIZE_CURRENT_CXR],
        tool_calls_used=1,
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(steps_used=1, tool_calls_used=1, cost_units_used=3),
    )

    assert state.conflict_flags == []
    assert decision.action == AgentAction.STOP
    assert decision.reason_code == SelectionReasonCode.TASK_ALREADY_COMPLETE


def test_completed_guideline_task_ignores_advisory_localization_result() -> None:
    spec = parse_task_spec("下一步做什么检查？")
    state = build_case_state(
        _case_with_advisory_detector_mismatch(),
        spec.current_query,
        required_evidence=spec.required_evidence,
        diagnostic_status=EvidenceStatus.AVAILABLE,
        completed_actions=[AgentAction.RETRIEVE_GUIDELINE],
        tool_calls_used=1,
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(steps_used=1, tool_calls_used=1, cost_units_used=2),
    )

    assert state.conflict_flags == []
    assert decision.action == AgentAction.STOP
    assert decision.reason_code == SelectionReasonCode.TASK_ALREADY_COMPLETE


class _ChoiceGenerator:
    backend_id = "test-controller"
    model = "test-model"

    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        self.calls = 0

    def complete_structured(self, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload), {"prompt_tokens": 12, "completion_tokens": 4}


def test_llm_can_order_independent_required_evidence_but_cannot_skip_it() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？病灶在哪里？")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )
    generator = _ChoiceGenerator(
        {
            "action": "localize_current_cxr",
            "reason_code": "localization_required",
        }
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(),
        generator=generator,
    )

    assert decision.action == AgentAction.LOCALIZE_CURRENT_CXR
    assert decision.source == DecisionSource.LLM_CONTROLLER
    assert decision.allowed_actions == [
        AgentAction.CLASSIFY_CURRENT_CXR,
        AgentAction.LOCALIZE_CURRENT_CXR,
    ]
    assert AgentAction.STOP not in decision.allowed_actions
    assert AgentAction.REFER_TO_HUMAN not in decision.allowed_actions


def test_invalid_llm_action_falls_back_to_state_policy() -> None:
    spec = parse_task_spec("这张胸片有没有结核病？病灶在哪里？")
    state = build_case_state(
        _empty_case(),
        spec.current_query,
        required_evidence=spec.required_evidence,
    )
    generator = _ChoiceGenerator(
        {"action": "stop", "reason_code": "task_already_complete"}
    )

    decision = BoundedAgentController().choose_next_action(
        state=state,
        task_spec=spec,
        capabilities=_capabilities(),
        budget=AgentBudget(),
        generator=generator,
    )

    assert decision.action == AgentAction.CLASSIFY_CURRENT_CXR
    assert decision.source == DecisionSource.STATE_POLICY_FALLBACK
    assert decision.schema_validated is False
