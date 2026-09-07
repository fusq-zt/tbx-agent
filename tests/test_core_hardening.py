from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from PIL import Image
from pydantic import ValidationError

from tbx_agent.config import Settings
from tbx_agent.orchestration import TBXAgentGraph
from tbx_agent.reports import generate_case_report
from tbx_agent.schemas import (
    AgentResponse,
    CaseRecord,
    Citation,
    ClassifierClass,
    DetectionEvidence,
    FusionDecision,
    ResponseKind,
    ReviewRecord,
    ReviewStatus,
    VisionEvidence,
    VisualResult,
)
from tbx_agent.screening import ScreeningEngine, ScreeningInputError, load_question_bank
from tbx_agent.service import AssessmentStateConflictError, TBXAgentService
from tbx_agent.storage import AccessDeniedError
from tbx_agent.tools import (
    ToolCallStatus,
    ToolDefinition,
    ToolInvocation,
    ToolRegistry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path, *, argmax: bool = False) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        openai_enabled=False,
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        fusion_policy_filename=(
            "fusion_policy_argmax_v2.json" if argmax else base.fusion_policy_filename
        ),
    )


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _invocation(tool_name: str = "test_tool") -> ToolInvocation:
    return ToolInvocation(
        tool_name=tool_name,
        message="sensitive message",
        thread_id="thread",
        user_id="user",
        owner_scope="tenant:one",
        request_id="request",
        trace_id="trace",
        routing_policy_id="router-v1",
    )


def _safe_response(invocation: ToolInvocation) -> AgentResponse:
    return AgentResponse(
        request_id=invocation.request_id,
        trace_id=invocation.trace_id,
        thread_id=invocation.thread_id,
        case_id=invocation.case_id,
        response_kind=ResponseKind.SAFE_ABSTENTION,
        summary="本系统不用于确诊或排除肺结核。",
        limitations=["本系统不用于确诊或排除肺结核。"],
    )


def _fallback(
    invocation: ToolInvocation,
    _status: ToolCallStatus,
    _error_code: str,
) -> AgentResponse:
    return _safe_response(invocation)


def test_tool_output_identity_and_kind_are_bound_to_static_contract() -> None:
    registry = ToolRegistry(max_steps=1)

    def wrong_thread(invocation: ToolInvocation) -> AgentResponse:
        return _safe_response(invocation).model_copy(update={"thread_id": "other"})

    registry.register(
        ToolDefinition(
            name="test_tool",
            audit_action="test_action",
            handler=wrong_thread,
            timeout_seconds=0.5,
        )
    )
    result = registry.execute(_invocation(), fallback_factory=_fallback)

    assert result.receipt.status == ToolCallStatus.FAILED
    assert result.receipt.error_code == "tool_output_thread_mismatch"
    assert result.receipt.fallback_used is True
    assert result.response.thread_id == "thread"
    assert result.receipt.permission == "public_information"
    assert result.receipt.medical_decision_authority is False
    registry.close()


def test_tool_timeout_holds_capacity_until_worker_really_exits() -> None:
    release = Event()
    started = Event()
    registry = ToolRegistry(max_steps=1, max_workers=1)

    def blocked(invocation: ToolInvocation) -> AgentResponse:
        started.set()
        release.wait(timeout=1.0)
        return _safe_response(invocation)

    registry.register(
        ToolDefinition(
            name="test_tool",
            audit_action="test_action",
            handler=blocked,
            timeout_seconds=0.005,
        )
    )
    timed_out = registry.execute(_invocation(), fallback_factory=_fallback)
    assert started.is_set()
    saturated = registry.execute(_invocation(), fallback_factory=_fallback)
    release.set()

    assert timed_out.receipt.status == ToolCallStatus.TIMED_OUT
    assert saturated.receipt.status == ToolCallStatus.SATURATED
    assert saturated.receipt.error_code == "tool_capacity_exhausted"
    registry.close()


def test_fast_sequential_tool_calls_do_not_false_saturate_capacity() -> None:
    registry = ToolRegistry(max_steps=1, max_workers=1)
    registry.register(
        ToolDefinition(
            name="test_tool",
            audit_action="test_action",
            handler=_safe_response,
            timeout_seconds=0.5,
        )
    )

    results = [registry.execute(_invocation(), fallback_factory=_fallback) for _ in range(32)]

    assert {result.receipt.status for result in results} == {ToolCallStatus.SUCCEEDED}
    registry.close()


def test_invalid_fallback_is_replaced_by_last_resort_abstention() -> None:
    registry = ToolRegistry(max_steps=1)

    def invalid_fallback(
        invocation: ToolInvocation,
        _status: ToolCallStatus,
        _error_code: str,
    ) -> AgentResponse:
        return _safe_response(invocation).model_copy(update={"thread_id": "forged"})

    result = registry.execute(_invocation(tool_name="unknown"), fallback_factory=invalid_fallback)

    assert result.receipt.status == ToolCallStatus.UNAVAILABLE
    assert result.receipt.output_contract_validated is False
    assert result.response.response_kind == ResponseKind.SAFE_ABSTENTION
    assert result.response.thread_id == "thread"
    registry.close()


def test_service_subject_binding_case_context_and_digest_only_memory(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    first, _ = service.assess_cxr(
        _png((10, 20, 30)),
        user_id="user-a",
        owner_scope="tenant:one",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    second, _ = service.assess_cxr(
        _png((30, 20, 10)),
        user_id="user-a",
        owner_scope="tenant:one",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    with pytest.raises(AccessDeniedError, match="subject binding"):
        service.respond(
            message="解释这张胸片结果",
            thread_id="foreign-case",
            user_id="user-b",
            owner_scope="tenant:one",
            case_id=first.case_id,
        )
    with pytest.raises(AccessDeniedError, match="subject binding"):
        service.create_report(
            case_id=first.case_id,
            owner_scope="tenant:one",
            actor_id="user-b",
        )
    with pytest.raises(AccessDeniedError, match="subject binding"):
        service.start_active_screening(
            thread_id="foreign-screen",
            user_id="user-b",
            owner_scope="tenant:one",
            case_id=first.case_id,
            consent=True,
        )

    service.respond(
        message="解释这张胸片结果",
        thread_id="one-case-per-thread",
        user_id="user-a",
        owner_scope="tenant:one",
        case_id=first.case_id,
    )
    with pytest.raises(AccessDeniedError, match="different case"):
        service.respond(
            message="解释这张胸片结果",
            thread_id="one-case-per-thread",
            user_id="user-a",
            owner_scope="tenant:one",
            case_id=second.case_id,
        )

    message = "痰NAAT是什么检查？这是不应进入线程记忆的原文"
    result = service.respond_with_tool(
        selected_tool="search_tb_guidance",
        message=message,
        thread_id="digest-memory",
        user_id="user-a",
        owner_scope="tenant:one",
    )
    state = service.store.get_or_create_thread("digest-memory", "user-a", "tenant:one")
    serialized = state.model_dump_json()
    assert message not in serialized
    assert all(not hasattr(event, "content") for event in state.recent_messages)
    assert all(len(event.content_sha256) == 64 for event in state.recent_messages)
    expected_response_hash = hashlib.sha256(
        json.dumps(
            result.response.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert result.receipt.response_sha256 == expected_response_hash
    assert result.receipt.tool_name == "search_tb_knowledge"


def test_assessment_hash_identity_is_isolated_per_subject_within_tenant(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    payload = _png((90, 80, 70))
    service.assess_cxr(
        payload,
        user_id="user-a",
        owner_scope="tenant:one",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    second, _ = service.assess_cxr(
        payload,
        user_id="user-b",
        owner_scope="tenant:one",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    assert second.user_id == "user-b"
    assert (
        second.case_id
        != service.store.find_case_by_hash("tenant:one", "user-a", second.image_sha256).case_id
    )


def test_existing_assessment_requires_explicit_generation_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    payload = _png((12, 34, 56))
    original, _ = service.assess_cxr(
        payload,
        user_id="user-a",
        owner_scope="tenant:one",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    artifacts_before = {
        path.relative_to(service.settings.case_artifact_root)
        for path in service.settings.case_artifact_root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(service, "_can_reuse", lambda _case, **_kwargs: False)

    with pytest.raises(AssessmentStateConflictError, match="generation migration"):
        service.assess_cxr(
            payload,
            user_id="user-a",
            owner_scope="tenant:one",
            consent_to_process=True,
            attested_chest_radiograph=True,
        )

    assert (
        service.store.find_case_by_hash("tenant:one", "user-a", original.image_sha256).case_id
        == original.case_id
    )
    assert {
        path.relative_to(service.settings.case_artifact_root)
        for path in service.settings.case_artifact_root.rglob("*")
        if path.is_file()
    } == artifacts_before
    assert service.store.audit_count("assessment_rerun_requires_generation_migration") == 1


def test_review_requires_explicit_matching_subject_context(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = CaseRecord(
        case_id="review-case",
        owner_scope="tenant:one",
        user_id="patient-a",
        image_artifact_ref="not_retained",
        image_sha256="d" * 64,
        image_width=512,
        image_height=512,
        consent_scope="screening",
        review_id="review-1",
        review_status=ReviewStatus.PENDING,
    )
    review = ReviewRecord(
        review_id="review-1",
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        trigger_reasons=["manual_review_test"],
    )
    service.store.save_case_with_review(case, review)

    with pytest.raises(AccessDeniedError, match="subject binding"):
        service.complete_review(
            review_id=review.review_id,
            owner_scope=case.owner_scope,
            reviewer_id="reviewer-1",
            subject_user_id="patient-b",
            expected_version=1,
            decision="indeterminate",
            note=None,
        )
    assert service.store.get_review(review.review_id, case.owner_scope).status == "pending"

    completed = service.complete_review(
        review_id=review.review_id,
        owner_scope=case.owner_scope,
        reviewer_id="reviewer-1",
        subject_user_id="patient-a",
        expected_version=1,
        decision="indeterminate",
        note="人工复核",
    )
    assert completed.status == "completed"


def test_agent_boundary_rejects_all_internal_execution_state_without_checkpoint(
    tmp_path: Path,
) -> None:
    graph = TBXAgentGraph(TBXAgentService(_settings(tmp_path)))
    request = {
        "message": "痰NAAT是什么检查？",
        "thread_id": "strict-graph",
        "user_id": "private-user-7a91",
        "owner_scope": "tenant:private-9b31",
        "selected_tool": "search_tb_guidance",
    }
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        graph.invoke_with_receipt(request)
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        graph.invoke_with_receipt({**request, "untrusted_payload": "value"})
    graph.close()
    graph.close()
    checkpoint_files = list(tmp_path.glob("langgraph_checkpoints_v2.sqlite3*"))
    assert checkpoint_files == []


def test_question_bank_rejects_unknown_fields_and_recomputes_summary(
    tmp_path: Path,
) -> None:
    source = PROJECT_ROOT / "knowledge" / "active_screening_questions.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["questions"][0]["question_text_typo"] = "silently ignored before hardening"
    modified = tmp_path / "questions.json"
    modified.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ScreeningInputError, match="invalid question"):
        load_question_bank(modified)

    engine = ScreeningEngine(source)
    session = engine.start_session(
        thread_id="thread",
        user_id="user",
        owner_scope="tenant:one",
        consent=True,
    )
    session = engine.submit_answer(
        session,
        ["严重呼吸困难"],
        question_id="emergency_red_flags",
    )
    assert session.result is not None
    session.result.next_steps = ["已经确诊肺结核。"]
    response = engine.build_response(session, request_id="request", trace_id="trace")
    assert "已经确诊" not in response.model_dump_json()
    assert response.urgency == "emergency"


def test_repeated_screening_start_resumes_one_active_session(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    first, _ = service.start_active_screening(
        thread_id="screen-thread",
        user_id="user",
        owner_scope="tenant:one",
        consent=True,
    )
    second, response = service.start_active_screening(
        thread_id="screen-thread",
        user_id="user",
        owner_scope="tenant:one",
        consent=True,
    )
    assert second.session_id == first.session_id
    assert response.next_question is not None
    assert service.store.audit_count("active_screening_resumed") == 1


def test_explicit_consent_withdrawal_cancels_and_clears_active_session(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    active, _ = service.start_active_screening(
        thread_id="withdraw-thread",
        user_id="user",
        owner_scope="tenant:one",
        consent=True,
    )
    active, _ = service.answer_active_screening(
        session_id=active.session_id,
        user_id="user",
        owner_scope="tenant:one",
        question_id="emergency_red_flags",
        answer=["以上均无"],
    )
    assert active.answers

    cancelled, response = service.start_active_screening(
        thread_id="withdraw-thread",
        user_id="user",
        owner_scope="tenant:one",
        consent=False,
    )

    assert cancelled.session_id == active.session_id
    assert cancelled.status == "cancelled"
    assert cancelled.consent is False
    assert cancelled.answers == {}
    assert response.response_kind == ResponseKind.ACTIVE_SCREENING_SUMMARY
    thread = service.store.get_or_create_thread("withdraw-thread", "user", "tenant:one")
    assert thread.active_screening_session_id is None
    assert service.store.audit_count("active_screening_cancelled_on_consent_withdrawal") == 1
    with pytest.raises(AccessDeniedError, match="not the thread's active session"):
        service.answer_active_screening(
            session_id=cancelled.session_id,
            user_id="user",
            owner_scope="tenant:one",
            question_id="age_group",
            answer="15～64岁",
        )


def test_report_is_unique_atomic_hashed_and_escapes_reviewer_markdown(
    tmp_path: Path,
) -> None:
    case = CaseRecord(
        case_id="case-safe",
        owner_scope="tenant:one",
        user_id="user",
        image_artifact_ref="not_retained",
        image_sha256="a" * 64,
        image_width=512,
        image_height=512,
        consent_scope="screening",
    )
    review = ReviewRecord(
        review_id="review-safe",
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        trigger_reasons=["test"],
        reviewer_note="normal\n# forged diagnosis\n[click](javascript:alert(1))",
    )
    citation = Citation(
        chunk_id="chunk-1",
        source_id="source-1",
        title="指南",
        organization="机构",
        publication_year=2026,
        section="章节",
        locator="第1页",
        url="https://example.test/guide",
        support_text="支持文本",
    )
    kwargs = {
        "case": case,
        "review": review,
        "citations": [citation, citation],
        "artifact_root": tmp_path,
        "knowledge_snapshot_id": "snapshot",
        "knowledge_manifest_sha256": "b" * 64,
        "knowledge_chunks_sha256": "c" * 64,
    }
    first = generate_case_report(**kwargs)
    second = generate_case_report(**kwargs)
    markdown = first.markdown_path.read_text(encoding="utf-8")

    assert first.report_id != second.report_id
    assert "\n# forged diagnosis" not in markdown
    assert "\\# forged diagnosis" in markdown
    assert first.markdown_sha256 == hashlib.sha256(first.markdown_path.read_bytes()).hexdigest()
    assert first.json_sha256 == hashlib.sha256(first.json_path.read_bytes()).hexdigest()
    report_payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    assert len(report_payload["citations"]) == 1

    unsafe = case.model_copy(update={"case_id": "../escape"})
    with pytest.raises(ValueError, match="unsafe"):
        generate_case_report(**{**kwargs, "case": unsafe})


def test_evidence_and_fusion_schemas_fail_closed_on_impossible_state() -> None:
    with pytest.raises(ValidationError, match="finite"):
        DetectionEvidence(bbox_xyxy=(0.0, 0.0, float("nan"), 2.0), score=0.5)
    with pytest.raises(ValidationError, match="clinical validation"):
        FusionDecision(
            policy_id="policy",
            visual_result=VisualResult.MODEL_NOT_FLAGGED,
            review_required=False,
            classifier_decision_rule="native_three_class_argmax",
            predicted_class=ClassifierClass.HEALTHY,
            classifier_flagged=False,
            detector_decision_role="advisory_localization_only",
            detector_flagged=None,
            clinical_validation=True,
        )
    with pytest.raises(ValidationError, match="pending-review"):
        FusionDecision(
            policy_id="policy",
            visual_result=VisualResult.PENDING_HUMAN_REVIEW,
            review_required=False,
            classifier_decision_rule="native_three_class_argmax",
            predicted_class=ClassifierClass.SICK_NON_TB,
            classifier_flagged=False,
            detector_decision_role="advisory_localization_only",
            detector_flagged=None,
        )


def test_vision_evidence_rejects_boxes_outside_source_image() -> None:
    with pytest.raises(ValidationError, match="within the source image"):
        VisionEvidence(
            run_id="run",
            case_id="case",
            image_sha256="a" * 64,
            image_quality_status="ok",
            image_width=100,
            image_height=100,
            classifier_model_id="classifier",
            classifier_checkpoint_sha256="b" * 64,
            class_probability_order=[item.value for item in ClassifierClass],
            class_probabilities={"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1},
            classifier_decision_rule="native_three_class_argmax",
            predicted_class=ClassifierClass.HEALTHY,
            classifier_argmax_tied=False,
            classifier_threshold=None,
            classifier_flagged=False,
            detector_model_id="detector",
            detector_checkpoint_sha256="c" * 64,
            detector_decision_role="advisory_localization_only",
            detector_threshold=None,
            detections=[DetectionEvidence(bbox_xyxy=(1.0, 1.0, 101.0, 50.0), score=0.5)],
            detector_flagged=None,
            preprocessing_version="preprocess",
            threshold_config_version="policy",
            runtime_ms=1,
        )
