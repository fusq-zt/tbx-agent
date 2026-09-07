from __future__ import annotations

from types import SimpleNamespace

import pytest

from tbx_agent.agent_state import (
    AgentAction,
    CalibrationStatus,
    ConflictFlag,
    EvidenceGap,
    EvidenceKind,
    EvidenceStatus,
    ScreeningDisposition,
    build_case_state,
    detect_conflicts,
    detect_evidence_gaps,
    post_action_check,
)
from tbx_agent.schemas import (
    CaseRecord,
    ClassifierClass,
    DetectionEvidence,
    LocalizationEvidence,
    VisionEvidence,
)
from tbx_agent.vision.fusion import fuse_rank03


def _vision(
    *,
    probabilities: dict[str, float] | None = None,
    detections: list[DetectionEvidence] | None = None,
    detector_execution: str = "detector_execution:not_requested",
    quality_status: str = "transport_valid",
    quality_codes: list[str] | None = None,
) -> VisionEvidence:
    probabilities = probabilities or {"healthy": 0.01, "sick_non_tb": 0.09, "tb": 0.90}
    maximum = max(probabilities.values())
    winners = [name for name, score in probabilities.items() if score == maximum]
    predicted = None if len(winners) != 1 else ClassifierClass(winners[0])
    return VisionEvidence(
        run_id="vision-run-1",
        case_id="case-1",
        image_sha256="a" * 64,
        image_quality_status=quality_status,
        image_quality_codes=quality_codes or [],
        image_source_format="PNG",
        input_transform_id="raster-exif-transpose-rgb-v1",
        image_width=512,
        image_height=512,
        classifier_model_id="classifier-v1",
        classifier_checkpoint_sha256="b" * 64,
        class_probability_order=["healthy", "sick_non_tb", "tb"],
        class_probabilities=probabilities,
        classifier_decision_rule="native_three_class_argmax",
        predicted_class=predicted,
        classifier_argmax_tied=len(winners) != 1,
        classifier_threshold=None,
        classifier_flagged=predicted == ClassifierClass.TB,
        detector_model_id="detector-v1",
        detector_checkpoint_sha256="c" * 64,
        detector_decision_role="advisory_localization_only",
        detector_threshold=None,
        detections=detections or [],
        detector_flagged=None,
        preprocessing_version="test-v1",
        threshold_config_version="argmax-v1",
        runtime_ms=10,
        artifact_refs=[detector_execution],
    )


def _case(**vision_kwargs: object) -> CaseRecord:
    detections = list(vision_kwargs.pop("detections", []) or [])
    detector_execution = str(
        vision_kwargs.pop("detector_execution", "detector_execution:not_requested")
    )
    if detector_execution == "detector_execution:not_requested":
        localization = LocalizationEvidence()
    elif detector_execution in {
        "detector_execution:on_demand_completed",
        "detector_execution:completed",
    }:
        localization = LocalizationEvidence(
            status="completed" if detections else "completed_no_detection",
            run_id="localization-run-1",
            generation_key="d" * 64,
            case_id="case-1",
            image_sha256="a" * 64,
            detector_model_id="detector-v1",
            detector_checkpoint_sha256="c" * 64,
            preprocessing_version="test-v1",
            detections=detections,
            runtime_ms=5,
            attempt_count=1,
        )
    elif detector_execution == "detector_execution:failed":
        localization = LocalizationEvidence(
            status="failed",
            generation_key="d" * 64,
            case_id="case-1",
            image_sha256="a" * 64,
            detector_model_id="detector-v1",
            detector_checkpoint_sha256="c" * 64,
            preprocessing_version="test-v1",
            attempt_count=1,
            error_code="inference_failed",
        )
    else:
        raise ValueError(f"unsupported test detector state: {detector_execution}")
    vision = _vision(**vision_kwargs)
    fusion = fuse_rank03(
        vision,
        {
            "policy_id": "argmax-v1",
            "classifier_rule": "native_three_class_argmax",
            "detector_role": "advisory_localization_only",
            "quality_warning_action": "continue_with_warning",
        },
    )
    return CaseRecord(
        case_id="case-1",
        owner_scope="owner-1",
        user_id="user-1",
        image_artifact_ref="cases/case-1/source.png",
        image_sha256="a" * 64,
        image_width=512,
        image_height=512,
        consent_scope="local_screening",
        classification_status="completed",
        classification_generation_key="e" * 64,
        classification_attempt_count=1,
        vision_evidence=vision,
        fusion_decision=fusion,
        localization_evidence=localization,
    )


def _one_detection() -> list[DetectionEvidence]:
    return [DetectionEvidence(bbox_xyxy=(10.0, 20.0, 80.0, 100.0), score=0.8)]


def test_required_enums_expose_explicit_state_and_terminal_actions() -> None:
    assert {status.value for status in EvidenceStatus} == {
        "not_requested",
        "not_run",
        "available",
        "completed",
        "completed_no_detection",
        "failed",
        "unsupported",
        "evidence_gap",
        "stale",
    }
    assert AgentAction.STOP.value == "stop"
    assert AgentAction.REFER_TO_HUMAN.value == "refer_to_human"


def test_build_case_state_is_compatible_with_case_record_and_exposes_scores() -> None:
    state = build_case_state(_case(), "这张胸片有没有结核病？", "screening")

    assert state.case_id == "case-1"
    assert state.subject_id == "user-1"
    assert state.predicted_class == "tb"
    assert state.classification_evidence.status == EvidenceStatus.AVAILABLE
    assert state.top1_class == "tb"
    assert state.top1_score == pytest.approx(0.90)
    assert state.top2_class == "sick_non_tb"
    assert state.top2_score == pytest.approx(0.09)
    assert state.top1_top2_margin == pytest.approx(0.81)
    assert state.calibration_status == CalibrationStatus.UNAVAILABLE
    assert state.screening_disposition == ScreeningDisposition.SCREEN_POSITIVE


def test_localization_not_requested_is_not_completed_without_detection() -> None:
    pending = build_case_state(_case(), "病灶在哪？", "localization")
    completed = build_case_state(
        _case(detector_execution="detector_execution:on_demand_completed"),
        "病灶在哪？",
        "localization",
    )

    assert pending.localization_evidence.status == EvidenceStatus.NOT_REQUESTED
    assert pending.localization_evidence.item_count is None
    assert ConflictFlag.CLASSIFIER_POSITIVE_DETECTOR_NEGATIVE not in pending.conflict_flags
    assert completed.localization_evidence.status == EvidenceStatus.COMPLETED_NO_DETECTION
    assert completed.localization_evidence.item_count == 0
    assert ConflictFlag.CLASSIFIER_POSITIVE_DETECTOR_NEGATIVE not in completed.conflict_flags


def test_explicit_future_localization_evidence_is_duck_typed() -> None:
    current = _case()
    future_case = SimpleNamespace(
        case_id=current.case_id,
        user_id=current.user_id,
        vision_evidence=current.vision_evidence,
        localization_evidence=SimpleNamespace(
            status="completed", detections=[], reason_codes=[]
        ),
    )

    state = build_case_state(future_case, "病灶在哪？", "localization")

    assert state.localization_evidence.status == EvidenceStatus.COMPLETED_NO_DETECTION
    assert state.localization_evidence.item_count == 0


def test_detector_result_never_changes_classifier_predicted_class() -> None:
    current = _case(
        detections=_one_detection(),
        detector_execution="detector_execution:on_demand_completed",
    )
    contradictory_fusion = SimpleNamespace(predicted_class="healthy")
    duck_case = SimpleNamespace(
        case_id=current.case_id,
        user_id=current.user_id,
        vision_evidence=current.vision_evidence,
        fusion_decision=contradictory_fusion,
    )

    state = detect_conflicts(build_case_state(duck_case, "筛查结果是什么？", "screening"))

    assert state.predicted_class == "tb"
    assert state.screening_disposition == ScreeningDisposition.SCREEN_POSITIVE


def test_advisory_detector_positive_does_not_conflict_with_classifier() -> None:
    state = build_case_state(
        _case(
            probabilities={"healthy": 0.80, "sick_non_tb": 0.15, "tb": 0.05},
            detections=_one_detection(),
            detector_execution="detector_execution:on_demand_completed",
        ),
        "病灶在哪？",
        "localization",
    )

    assert state.predicted_class == "healthy"
    assert ConflictFlag.CLASSIFIER_NEGATIVE_DETECTOR_POSITIVE not in state.conflict_flags
    assert state.screening_disposition == ScreeningDisposition.SCREEN_NEGATIVE


def test_configured_low_margin_policy_records_uncertainty_without_fake_conflict() -> None:
    state = build_case_state(
        _case(probabilities={"healthy": 0.10, "sick_non_tb": 0.44, "tb": 0.46}),
        "筛查结果是什么？",
        "screening",
        classification_low_margin_threshold=0.05,
    )

    assert state.top1_top2_margin == pytest.approx(0.02)
    assert state.calibration_status == CalibrationStatus.UNAVAILABLE
    assert state.classifier_operating_threshold is None
    assert ConflictFlag.CLASSIFICATION_LOW_MARGIN not in state.conflict_flags
    assert "classification_low_margin" in state.uncertainty_flags
    assert ConflictFlag.CLASSIFICATION_NEAR_THRESHOLD not in state.conflict_flags


def test_post_action_check_stops_when_task_evidence_is_complete() -> None:
    initial = build_case_state(_case(), "这张胸片有没有结核病？", "screening")

    checked = post_action_check(initial)

    assert checked.task_complete is True
    assert checked.human_review_required is False
    assert checked.terminal_action == AgentAction.STOP
    assert checked.stop_reason == "task_evidence_complete"
    assert checked.localization_evidence.status == EvidenceStatus.NOT_REQUESTED


def test_successful_localization_completes_without_cross_model_human_review() -> None:
    initial = build_case_state(
        _case(detector_execution="detector_execution:on_demand_completed"),
        "病灶在哪？",
        "localization",
        tool_calls_used=1,
    )

    checked = post_action_check(initial)

    assert checked.predicted_class == "tb"
    assert checked.task_complete is True
    assert checked.human_review_required is False
    assert checked.terminal_action == AgentAction.STOP
    assert checked.screening_disposition == ScreeningDisposition.SCREEN_POSITIVE
    assert checked.human_review_reasons == []


def test_unrelated_failed_classification_does_not_block_diagnostic_guidance() -> None:
    case = SimpleNamespace(
        case_id="case-1",
        user_id="user-1",
        classification_status="failed",
        classification_error_code="previous_backend_failure",
        vision_evidence=None,
        localization_evidence=SimpleNamespace(
            status="not_requested", detections=[], reason_codes=[]
        ),
    )
    initial = build_case_state(
        case,
        "下一步做什么检查？",
        required_evidence=[EvidenceKind.DIAGNOSTIC],
        diagnostic_status=EvidenceStatus.AVAILABLE,
    )

    checked = post_action_check(initial)

    assert ConflictFlag.TOOL_FAILURE_CONFLICT not in checked.conflict_flags
    assert checked.evidence_gaps == []
    assert checked.task_complete is True
    assert checked.human_review_required is False
    assert checked.terminal_action == AgentAction.STOP


def test_exhausted_budget_with_missing_required_evidence_refers_to_human() -> None:
    initial = build_case_state(
        _case(),
        "病灶在哪？",
        "localization",
        tool_budget=1,
        tool_calls_used=1,
    )

    checked = post_action_check(initial)

    assert EvidenceGap.LOCALIZATION_NOT_RUN in checked.evidence_gaps
    assert EvidenceGap.TOOL_BUDGET_EXHAUSTED in checked.evidence_gaps
    assert checked.remaining_tool_calls == 0
    assert checked.human_review_required is True
    assert checked.terminal_action == AgentAction.REFER_TO_HUMAN


def test_prior_image_gap_prevents_longitudinal_claim_and_stops() -> None:
    initial = build_case_state(
        _case(),
        "和半年前相比恶化了吗？",
        "comparison",
        prior_status=EvidenceStatus.EVIDENCE_GAP,
    )

    with_gaps = detect_evidence_gaps(initial)
    checked = post_action_check(with_gaps)

    assert EvidenceGap.PRIOR_IMAGE_NOT_FOUND in checked.evidence_gaps
    assert checked.task_complete is False
    assert checked.human_review_required is False
    assert checked.terminal_action == AgentAction.STOP
    assert checked.stop_reason == "required_evidence_unavailable"


def test_available_prior_but_unsupported_longitudinal_model_stops_with_gap() -> None:
    initial = build_case_state(
        _case(),
        "比较当前胸片和 prior study",
        "comparison",
        prior_status=EvidenceStatus.AVAILABLE,
        longitudinal_status=EvidenceStatus.UNSUPPORTED,
    )

    checked = post_action_check(initial)

    assert EvidenceGap.PRIOR_STUDY_NOT_CHECKED not in checked.evidence_gaps
    assert EvidenceGap.LONGITUDINAL_MODEL_UNAVAILABLE in checked.evidence_gaps
    assert checked.human_review_required is False
    assert checked.terminal_action == AgentAction.STOP


def test_raw_task_text_does_not_replace_structured_evidence_requirements() -> None:
    state = build_case_state(
        _case(),
        "病灶在哪？",
        "screening",
        required_evidence=[EvidenceKind.CLASSIFICATION],
    )

    checked = post_action_check(state)

    assert checked.required_evidence == [EvidenceKind.CLASSIFICATION]
    assert EvidenceGap.LOCALIZATION_NOT_RUN not in checked.evidence_gaps
    assert checked.terminal_action == AgentAction.STOP


def test_quality_warning_needs_configured_policy_code_to_be_a_conflict() -> None:
    current = _case(quality_status="warning", quality_codes=["very_low_dynamic_range"])

    unconfigured = build_case_state(current, "筛查结果是什么？", "screening")
    configured = build_case_state(
        current,
        "筛查结果是什么？",
        "screening",
        quality_conflict_codes=["very_low_dynamic_range"],
    )

    assert ConflictFlag.TECHNICAL_QUALITY_CONFLICT not in unconfigured.conflict_flags
    assert ConflictFlag.TECHNICAL_QUALITY_CONFLICT in configured.conflict_flags


def test_failed_tool_is_a_conflict_and_human_review_action_is_first_class() -> None:
    initial = build_case_state(
        _case(),
        "病灶在哪？",
        "localization",
        completed_actions=[AgentAction.GET_CLASSIFICATION_EVIDENCE],
        failed_actions=[AgentAction.LOCALIZE_CURRENT_CXR],
        last_action=AgentAction.REFER_TO_HUMAN,
    )

    checked = post_action_check(initial)

    assert ConflictFlag.TOOL_FAILURE_CONFLICT in checked.conflict_flags
    assert checked.terminal_action == AgentAction.REFER_TO_HUMAN
    assert "controller_requested_human_review" in checked.human_review_reasons


def test_failed_required_localization_is_a_gap_and_refers_to_human() -> None:
    initial = build_case_state(
        _case(detector_execution="detector_execution:failed"),
        "病灶在哪？",
        "localization",
        tool_calls_used=1,
    )

    checked = post_action_check(initial)

    assert EvidenceGap.LOCALIZATION_FAILED in checked.evidence_gaps
    assert ConflictFlag.TOOL_FAILURE_CONFLICT in checked.conflict_flags
    assert checked.human_review_required is True
    assert checked.terminal_action == AgentAction.REFER_TO_HUMAN


def test_explicit_stop_with_recoverable_gap_is_terminal_but_not_complete() -> None:
    initial = build_case_state(
        _case(),
        "病灶在哪？",
        "localization",
        last_action=AgentAction.STOP,
    )

    checked = post_action_check(initial)

    assert checked.terminal_action == AgentAction.STOP
    assert checked.task_complete is False
    assert checked.human_review_required is False
    assert checked.screening_disposition == ScreeningDisposition.INSUFFICIENT_EVIDENCE
    assert checked.stop_reason == "controller_stopped_with_evidence_gap"


def test_post_action_functions_are_pure_and_do_not_mutate_input_state() -> None:
    initial = build_case_state(_case(), "病灶在哪？", "localization")
    snapshot = initial.model_dump(mode="json")

    _ = detect_conflicts(initial)
    _ = detect_evidence_gaps(initial)
    _ = post_action_check(initial)

    assert initial.model_dump(mode="json") == snapshot
