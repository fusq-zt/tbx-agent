from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.schemas import (
    ClassifierClass,
    DetectionEvidence,
    GuidelineAnswerStatus,
    ResponseKind,
    ReviewOrigin,
    ReviewStatus,
    Urgency,
    VisionEvidence,
    VisualResult,
)
from tbx_agent.service import TBXAgentService
from tbx_agent.vision.base import VisionBackendError
from tbx_agent.vision.rank03 import Rank03Backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path) -> Settings:
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
    )


def _argmax_settings(tmp_path: Path) -> Settings:
    return replace(_settings(tmp_path), fusion_policy_filename="fusion_policy_argmax_v2.json")


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=(40, 60, 80)).save(buffer, format="PNG")
    return buffer.getvalue()


def _quality_ok_png() -> bytes:
    buffer = io.BytesIO()
    Image.linear_gradient("L").resize((512, 512)).convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _upload_and_classify(
    service: TBXAgentService,
    payload: bytes,
    *,
    user_id: str,
    owner_scope: str,
):
    case, _ = service.assess_cxr(
        payload,
        user_id=user_id,
        owner_scope=owner_scope,
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    return service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope=owner_scope,
        user_id=user_id,
        payload=payload,
    )


def test_rank03_wraps_unexpected_runtime_failures_as_backend_errors() -> None:
    backend = object.__new__(Rank03Backend)
    backend._lock = threading.Semaphore(1)

    def fail(_image):
        raise RuntimeError("simulated CUDA OOM")

    backend._classifier_infer = fail

    with pytest.raises(VisionBackendError, match="模型加载或前向推理失败"):
        backend.infer(case_id="case-runtime-failure", image=None)


class _StaticArgmaxBackend:
    backend_id = "test-native-argmax"

    def __init__(
        self,
        probabilities: dict[str, float],
        detections: list[DetectionEvidence] | None = None,
    ):
        self.probabilities = probabilities
        self.detections = list(detections or [])

    def infer(self, *, case_id, image):
        maximum = max(self.probabilities.values())
        winners = [
            item.value
            for item in ClassifierClass
            if self.probabilities[item.value] == maximum
        ]
        tied = len(winners) != 1
        predicted = ClassifierClass(winners[0])
        return VisionEvidence(
            run_id="static-argmax-run",
            case_id=case_id,
            image_sha256=image.sha256,
            image_quality_status=image.quality_status,
            image_quality_codes=list(image.quality_warnings),
            image_source_format=image.source_format,
            input_transform_id=image.input_transform_id,
            image_width=image.width,
            image_height=image.height,
            classifier_model_id="MOCK_ONLY__static_argmax",
            classifier_checkpoint_sha256="0" * 64,
            class_probability_order=[item.value for item in ClassifierClass],
            class_probabilities=self.probabilities,
            classifier_decision_rule="native_three_class_argmax",
            predicted_class=predicted,
            classifier_argmax_tied=tied,
            classifier_threshold=None,
            classifier_flagged=predicted == ClassifierClass.TB,
            detector_model_id="MOCK_ONLY__advisory_detector",
            detector_checkpoint_sha256="1" * 64,
            detector_decision_role="advisory_localization_only",
            detector_threshold=None,
            detections=self.detections,
            detector_flagged=None,
            preprocessing_version="test-native-argmax-v1",
            threshold_config_version=("rank03-agent-screening-demo-cls-argmax-det-advisory-v2"),
            runtime_ms=1,
        )

    def localize(self, *, case_id, image):
        del case_id, image
        return list(self.detections)


def test_assessment_reuses_only_exact_scope_and_integrity(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    kwargs = {
        "user_id": "u1",
        "owner_scope": "tenant:u1",
        "consent_to_process": True,
        "attested_chest_radiograph": True,
    }
    first_case, first_response = service.assess_cxr(_png(), **kwargs)
    second_case, second_response = service.assess_cxr(_png(), **kwargs)

    assert first_case.case_id == second_case.case_id
    assert first_response.reused_existing_assessment is False
    assert second_response.reused_existing_assessment is True
    assert first_case.classification_status == "not_requested"
    assert first_case.vision_evidence is None
    assert first_case.fusion_decision is None
    assert service.vision.call_count == 0
    assert "尚未运行分类" in first_response.summary

    other_case, _ = service.assess_cxr(
        _png(),
        **{**kwargs, "user_id": "u2", "owner_scope": "tenant:u2"},
    )
    assert other_case.case_id != first_case.case_id


def test_concurrent_first_upload_builds_one_case_without_inference(tmp_path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    original_infer = service.vision.infer
    infer_calls = 0
    infer_guard = threading.Lock()
    second_infer_entered = threading.Event()
    start_gate = threading.Barrier(3)

    def observed_infer(*, case_id, image):
        nonlocal infer_calls
        with infer_guard:
            infer_calls += 1
            call_number = infer_calls
        if call_number == 1:
            # A missing subject/image lock lets the peer enter inference and set
            # this event. The correct path keeps it behind the first generation.
            second_infer_entered.wait(timeout=0.25)
        else:
            second_infer_entered.set()
        return original_infer(case_id=case_id, image=image)

    service.vision.infer = observed_infer

    def assess_once():
        start_gate.wait(timeout=2)
        return service.assess_cxr(
            _png(),
            user_id="concurrent-user",
            owner_scope="tenant:concurrent",
            consent_to_process=True,
            attested_chest_radiograph=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(assess_once) for _ in range(2)]
        start_gate.wait(timeout=2)
        results = [future.result(timeout=5) for future in futures]

    assert infer_calls == 0
    assert {case.case_id for case, _response in results} == {results[0][0].case_id}
    assert sorted(response.reused_existing_assessment for _case, response in results) == [
        False,
        True,
    ]

    case = results[0][0]
    first_case, first_response = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope="tenant:concurrent",
        user_id="concurrent-user",
    )
    second_case, second_response = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope="tenant:concurrent",
        user_id="concurrent-user",
    )
    assert infer_calls == 1
    assert first_case.classification_status == "completed"
    assert second_case.vision_evidence == first_case.vision_evidence
    assert first_response.reused_existing_assessment is False
    assert second_response.reused_existing_assessment is True


def test_classification_failure_and_unavailable_are_not_empty_results(tmp_path) -> None:
    class FailingBackend:
        backend_id = "failing-classifier"

        def infer(self, *, case_id, image):
            del case_id, image
            raise VisionBackendError("synthetic classifier failure")

        def localize(self, *, case_id, image):
            raise AssertionError("classification must not call localization")

    failed_service = TBXAgentService(
        _argmax_settings(tmp_path / "failed"),
        vision_backend=FailingBackend(),
    )
    failed_case, _ = failed_service.assess_cxr(
        _quality_ok_png(),
        user_id="failure-user",
        owner_scope="tenant:failure",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    failed_case, _ = failed_service.classify_cxr_case(
        case_id=failed_case.case_id,
        owner_scope="tenant:failure",
        user_id="failure-user",
    )
    assert failed_case.classification_status == "failed"
    assert failed_case.classification_error_code == "classification_backend_failed"
    assert failed_case.vision_evidence is None
    assert failed_case.localization_evidence.status == "not_requested"

    unavailable_service = TBXAgentService(
        replace(_argmax_settings(tmp_path / "unavailable"), retain_uploaded_image=False)
    )
    unavailable_case, _ = unavailable_service.assess_cxr(
        _quality_ok_png(),
        user_id="unavailable-user",
        owner_scope="tenant:unavailable",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    unavailable_case, _ = unavailable_service.classify_cxr_case(
        case_id=unavailable_case.case_id,
        owner_scope="tenant:unavailable",
        user_id="unavailable-user",
    )
    assert unavailable_case.classification_status == "unavailable"
    assert unavailable_case.vision_evidence is None
    assert unavailable_service.vision.call_count == 0


def test_case_images_and_reports_never_use_immutable_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "immutable-model-cache"
    case_root = tmp_path / "mutable-case-artifacts"
    monkeypatch.setenv("TBX_ARTIFACT_ROOT", str(model_root))
    service = TBXAgentService(replace(_settings(tmp_path), artifact_root=case_root))

    case, _ = service.assess_cxr(
        _png(),
        user_id="case-root-user",
        owner_scope="tenant:case-root",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    report = service.create_report(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        actor_id="case-root-user",
    )

    assert Path(case.image_artifact_ref).is_relative_to(case_root.resolve())
    assert Path(report["markdown_path"]).is_relative_to(case_root.resolve())
    assert Path(report["json_path"]).is_relative_to(case_root.resolve())
    assert not model_root.exists()


def test_diagnostic_and_emergency_agent_paths(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    diagnostic = service.respond(
        message="胸片异常后应该做什么检查？",
        thread_id="thread-1",
        user_id="u1",
        owner_scope="tenant:u1",
    )
    emergency = service.respond(
        message="我现在大量咯血并且严重呼吸困难",
        thread_id="thread-1",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert diagnostic.response_kind == ResponseKind.NEXT_TEST_INFORMATION
    assert diagnostic.citations
    assert emergency.urgency == Urgency.EMERGENCY


def test_pregnancy_testing_synonyms_use_direct_reviewed_evaluation_path(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    queries = (
        "孕妇怀疑肺结核时检查有什么不同？",
        "孕妇怀疑肺结核时应该做什么检查",
    )

    responses = [
        service.respond(
            message=query,
            thread_id=f"thread-pregnancy-{index}",
            user_id="u1",
            owner_scope="tenant:u1",
        )
        for index, query in enumerate(queries)
    ]

    for response in responses:
        assert response.answer_status == GuidelineAnswerStatus.ANSWERED
        assert response.guideline_scope == "special_population"
        assert response.guideline_subtopic == "special_population_testing"
        assert "结核病医学评估" in response.summary
        assert "胸部X线" in response.summary
        assert "痰等标本" in response.summary
        assert response.evidence_gap is None
        assert response.claims
        assert response.retrieved_evidence
        assert response.citations
        assert response.citations[0].chunk_id == "cdc25_pregnancy_tb_evaluation"


def test_child_initial_testing_uses_only_directly_conditional_paediatric_evidence(
    tmp_path,
):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message="15岁以下儿童一般优先做什么检查？",
        thread_id="thread-child-initial-testing",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.guideline_scope == "special_population"
    assert response.guideline_subtopic == "special_population_testing"
    chunk_ids = [item.chunk_id for item in response.citations]
    assert chunk_ids == [
        "as26_symptomatic_under15_initial",
        "who25_child_concurrent_samples",
        "as26_symptomatic_under15_no_sputum",
    ]
    rendered = response.model_dump_json()
    assert "可疑症状" in rendered
    assert "症状或筛查阳性" in rendered
    assert "无痰或难以获得合格痰标本" in rendered
    assert "15岁及以上" not in rendered
    assert "HIV感染的成人" not in rendered


def test_special_population_without_direct_clause_does_not_borrow_generic_adult_text(
    tmp_path,
):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message="免疫抑制人群怀疑肺结核时应该做什么检查？",
        thread_id="thread-immunosuppressed-evidence-gate",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.guideline_scope == "special_population"
    assert response.guideline_subtopic == "special_population_testing"
    assert response.claims == []
    assert response.citations == []
    assert response.retrieved_evidence == []
    assert "主动筛查场景条款未外推为诊断规则" in (response.evidence_gap or "")


def test_stop_drug_question_returns_review_boundary_instead_of_retrieval_abstention(tmp_path):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message="服药后不舒服，我是否现在就自行停掉所有药？",
        thread_id="thread-stop-drug",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    rendered = "\n".join([response.summary, *response.treatment_education, *response.limitations])
    assert response.response_kind == ResponseKind.TREATMENT_EDUCATION
    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert response.claims == []
    assert "不要自行" in rendered
    assert "治疗机构" in rendered


def test_short_followup_keeps_diagnostic_intent_without_storing_raw_text(tmp_path):
    service = TBXAgentService(_settings(tmp_path))

    first = service.respond(
        message="诊断",
        thread_id="thread-contextual-followup",
        user_id="u1",
        owner_scope="tenant:u1",
    )
    second = service.respond(
        message="给出",
        thread_id="thread-contextual-followup",
        user_id="u1",
        owner_scope="tenant:u1",
    )
    state = service.store.get_or_create_thread("thread-contextual-followup", "u1", "tenant:u1")

    assert first.response_kind == ResponseKind.NEXT_TEST_INFORMATION
    assert first.citations
    assert second.response_kind == ResponseKind.NEXT_TEST_INFORMATION
    assert second.citations
    assert state.active_intent == "search_tb_knowledge"
    assert "诊断" not in state.model_dump_json()
    assert "给出" not in state.model_dump_json()


def test_general_treatment_question_uses_reviewed_who_education(tmp_path):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message="可以怎么治疗",
        thread_id="thread-treatment-path",
        user_id="u1",
        owner_scope="tenant:u1",
    )
    assert response.response_kind == ResponseKind.TREATMENT_EDUCATION
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.guideline_scope == "treatment_education"
    assert response.guideline_subtopic == "treatment_principles"
    assert response.citations
    assert response.retrieved_evidence
    assert response.claims
    assert all(
        citation.source_id == "who_tb_treatment_module4_2025" for citation in response.citations
    )
    assert "药物敏感性" in "\n".join(response.treatment_education)


@pytest.mark.parametrize(
    "message",
    [
        "肺结核必须住院吗？",
        "确诊后能在门诊或社区治疗吗？",
        "耐药肺结核必须住院治疗吗？",
    ],
)
def test_care_setting_uses_only_reviewed_handbook_clauses(tmp_path, message):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message=message,
        thread_id="thread-care-setting",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert response.response_kind == ResponseKind.TREATMENT_EDUCATION
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.guideline_scope == "treatment_education"
    assert response.guideline_subtopic == "care_setting"
    assert [citation.chunk_id for citation in response.citations] == [
        "who25_care_setting_ambulatory_majority",
        "who25_care_setting_inpatient_indications",
    ]
    assert all(
        citation.source_id == "who_tb_treatment_handbook_module4_2025"
        for citation in response.citations
    )
    rendered = "\n".join(response.treatment_education)
    assert "多数结核病患者" in rendered
    assert "可能需要住院" in rendered
    assert "安全监测" in rendered
    assert "必须住院" not in rendered
    assert "所有药物敏感性肺结核" not in rendered


@pytest.mark.parametrize(
    ("message", "expected_chunks"),
    [
        (
            "Xpert和痰培养分别有什么作用？",
            ["cdc25_xpert_role", "cdc25_culture_role"],
        ),
        (
            "痰培养和痰涂片有什么区别？",
            ["cdc25_culture_role", "cdc25_smear_role"],
        ),
        (
            "胸片异常后应该做什么检查？",
            ["who25_initial_lc_anaat"],
        ),
        (
            "儿童咳不出痰时怎么办？",
            ["as26_symptomatic_under15_no_sputum"],
        ),
        (
            "孕妇怀疑肺结核时应该做什么检查？",
            ["cdc25_pregnancy_tb_evaluation"],
        ),
        (
            "HIV感染的成人有结核症状时做什么检查？",
            ["who25_hiv_adult_concurrent"],
        ),
        (
            "结核感染检测阳性说明什么？",
            ["cdc25_infection_test_positive"],
        ),
        (
            "肺结核患者什么时候可以认为没有传染性了？",
            ["cdc25_infectiousness_followup"],
        ),
        (
            "吃抗结核药后视力变模糊怎么办？",
            ["cdc25_blurred_vision_serious"],
        ),
    ],
)
def test_entity_and_applicability_selection_excludes_unasked_evidence(
    tmp_path,
    message,
    expected_chunks,
):
    service = TBXAgentService(_settings(tmp_path))

    result = service.respond_with_controller(
        message=message,
        thread_id="thread-entity-applicability",
        user_id="u1",
        owner_scope="tenant:u1",
        generator=None,
    )
    response = result.response

    actual_chunks = [citation.chunk_id for citation in response.citations]
    service.tool_registry.close()
    service.store.close()
    assert actual_chunks == expected_chunks, response.model_dump_json(indent=2)


def test_drug_resistant_treatment_question_does_not_borrow_ds_tb_regimen(tmp_path):
    service = TBXAgentService(_settings(tmp_path))

    response = service.respond(
        message="耐药肺结核怎么治疗？",
        thread_id="thread-resistant-treatment",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert response.response_kind == ResponseKind.TREATMENT_EDUCATION
    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert response.retrieved_evidence == []
    assert response.claims == []
    assert "不能把药物敏感性方案套用于耐药情形" in (response.evidence_gap or "")


def test_guideline_tool_quarantines_tampered_diagnostic_retrieval_content(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    trusted = service.retriever.retrieve_scoped(
        "肺结核 快速分子检测 NAAT 病原学",
        required_claim_scopes={
            "initial_diagnostic_testing",
            "rapid_drug_resistance_testing",
            "test_limitations",
        },
        preferred_topics={"rapid_diagnostics", "naat"},
        top_k=1,
    )[0]
    canary = "SECRET_DIAGNOSTIC_CANARY_Z9"
    tampered = replace(
        trusted,
        citation=trusted.citation.model_copy(update={"support_text": canary}),
    )
    service.retriever.retrieve_scoped = lambda *_args, **_kwargs: [tampered]

    response = service.respond(
        message="痰NAAT是什么检查？",
        thread_id="thread-diagnostic-tampered",
        user_id="u1",
        owner_scope="tenant:u1",
    )

    assert response.response_kind == ResponseKind.SAFE_ABSTENTION
    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.citations == []
    assert "完整性" in response.summary
    assert canary not in response.model_dump_json()


def test_visual_response_exposes_training_class_without_diagnostic_rewrite(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.1, "sick_non_tb": 0.8, "tb": 0.1}),
    )
    case, response = _upload_and_classify(
        service,
        _quality_ok_png(),
        user_id="u1",
        owner_scope="tenant:sick-class",
    )

    assert case.fusion_decision.predicted_class == ClassifierClass.SICK_NON_TB
    assert response.predicted_class == ClassifierClass.SICK_NON_TB
    assert response.visual_result == VisualResult.NON_TB_ABNORMAL
    assert response.summary.endswith("模型识别为非结核异常。")
    assert case.review_id is None
    assert case.review_status == ReviewStatus.NOT_REQUIRED
    assert service.store.list_pending_reviews(case.owner_scope) == []
    assert response.limitations == ["这是辅助筛查结果，不用于确诊或排除肺结核。"]


def test_interactive_uncertain_result_stays_out_of_review_queue(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.45, "sick_non_tb": 0.45, "tb": 0.1}),
    )

    case, response = _upload_and_classify(
        service,
        _quality_ok_png(),
        user_id="u1",
        owner_scope="tenant:interactive-tie",
    )

    assert response.visual_result == VisualResult.MODEL_NOT_FLAGGED
    assert response.predicted_class == ClassifierClass.HEALTHY
    assert "classification_exact_argmax_tie" in case.uncertainty_flags
    assert response.review_status == ReviewStatus.NOT_REQUIRED
    assert case.review_id is None
    assert case.review_status == ReviewStatus.NOT_REQUIRED
    assert service.store.list_pending_reviews(case.owner_scope) == []


def test_batch_enrollment_reuses_interactive_case_and_is_idempotent(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.1, "sick_non_tb": 0.8, "tb": 0.1}),
    )
    kwargs = {
        "user_id": "u1",
        "owner_scope": "tenant:batch-reuse",
        "consent_to_process": True,
        "attested_chest_radiograph": True,
    }
    interactive_case, _ = service.assess_cxr(_quality_ok_png(), **kwargs)

    first_case, first_response, first_review = service.assess_cxr_for_batch(
        _quality_ok_png(),
        batch_id="batch-001",
        batch_item_id="item-001",
        **kwargs,
    )
    second_case, second_response, second_review = service.assess_cxr_for_batch(
        _quality_ok_png(),
        batch_id="batch-001",
        batch_item_id="item-001",
        **kwargs,
    )

    assert first_case.case_id == interactive_case.case_id == second_case.case_id
    assert first_review is not None and second_review is not None
    assert first_review.review_id == second_review.review_id
    assert first_review.origin == ReviewOrigin.BATCH_SCREENING
    assert first_review.batch_id == "batch-001"
    assert first_review.batch_item_id == "item-001"
    assert first_response.review_status == ReviewStatus.PENDING
    assert second_response.review_status == ReviewStatus.PENDING
    pending = service.store.list_pending_reviews(
        first_case.owner_scope,
        origin=ReviewOrigin.BATCH_SCREENING,
    )
    assert [item.review_id for item in pending] == [first_review.review_id]


def test_batch_healthy_direct_route_does_not_enter_review_queue(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1}),
    )

    case, response, review = service.assess_cxr_for_batch(
        _quality_ok_png(),
        batch_id="batch-healthy",
        batch_item_id="item-healthy",
        user_id="u1",
        owner_scope="tenant:batch-healthy",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    assert response.visual_result == VisualResult.MODEL_NOT_FLAGGED
    assert review is None
    assert case.review_id is None
    assert case.review_status == ReviewStatus.NOT_REQUIRED
    assert (
        service.store.list_pending_reviews(
            case.owner_scope,
            origin=ReviewOrigin.BATCH_SCREENING,
        )
        == []
    )


@pytest.mark.parametrize(
    ("probabilities", "predicted_class", "visual_result", "expected_summary"),
    [
        (
            {"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1},
            ClassifierClass.HEALTHY,
            VisualResult.MODEL_NOT_FLAGGED,
            "模型更倾向于健康类。",
        ),
        (
            {"healthy": 0.1, "sick_non_tb": 0.1, "tb": 0.8},
            ClassifierClass.TB,
            VisualResult.MODEL_FLAGGED,
            "模型识别为结核类，建议进一步检查。",
        ),
    ],
)
def test_native_argmax_visual_primary_summary_is_concise(
    tmp_path, probabilities, predicted_class, visual_result, expected_summary
):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend(probabilities),
    )

    case, response = _upload_and_classify(
        service,
        _quality_ok_png(),
        user_id="u1",
        owner_scope=f"tenant:{predicted_class.value}",
    )

    assert case.fusion_decision.predicted_class == predicted_class
    assert response.predicted_class == predicted_class
    assert response.visual_result == visual_result
    assert response.summary.endswith(expected_summary)
    assert any("辅助筛查" in item and "不用于确诊" in item for item in response.limitations)


def test_case_followups_reuse_argmax_and_detector_evidence_instead_of_generic_advice(
    tmp_path,
):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend(
            {"healthy": 0.05, "sick_non_tb": 0.14, "tb": 0.81},
            detections=[
                DetectionEvidence(
                    bbox_xyxy=(48.0, 42.0, 178.0, 164.0),
                    score=0.93,
                )
            ],
        ),
    )
    case, _ = _upload_and_classify(
        service,
        _quality_ok_png(),
        user_id="u1",
        owner_scope="tenant:case-followups",
    )

    location = service.respond(
        message="病灶在哪？",
        thread_id="thread-case-followups",
        user_id="u1",
        owner_scope="tenant:case-followups",
        case_id=case.case_id,
    )
    rationale = service.respond(
        message="为什么认为是TB？",
        thread_id="thread-case-followups",
        user_id="u1",
        owner_scope="tenant:case-followups",
        case_id=case.case_id,
    )

    location_text = "\n".join([location.summary, *location.visual_evidence_notes])
    assert location.response_kind == ResponseKind.VISUAL_SCREENING_RESULT
    assert "检测到 1 个候选区域" in location.summary
    assert "图像左侧上部" in location_text
    assert "%" not in location_text
    assert "定位分数" not in location_text
    assert "不参与三分类" not in location_text
    assert location.diagnostic_information == []
    assert location.next_step_information == []
    assert location.citations == []

    rationale_text = "\n".join([rationale.summary, *rationale.visual_evidence_notes])
    assert rationale.response_kind == ResponseKind.VISUAL_SCREENING_RESULT
    assert rationale.summary == "因为胸片分类模型在三个训练类别中将结核类判为最高类别。"
    assert rationale.visual_evidence_notes == []
    assert "%" not in rationale_text
    assert "相对得分" not in rationale_text
    assert "相对分数" not in rationale_text
    assert "D-FINE" not in rationale_text
    assert "不参与" not in rationale_text
    assert rationale.diagnostic_information == []
    assert rationale.next_step_information == []
    assert rationale.citations == []


def test_localization_is_lazy_and_cached_per_case(tmp_path):
    service = TBXAgentService(_argmax_settings(tmp_path))
    case, _ = service.assess_cxr(
        _quality_ok_png(),
        user_id="lazy-user",
        owner_scope="tenant:lazy",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    assert service.vision.call_count == 0
    assert service.vision.localization_call_count == 0
    assert case.vision_evidence is None

    case, _ = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope="tenant:lazy",
        user_id="lazy-user",
    )
    assert service.vision.call_count == 1
    assert case.vision_evidence is not None
    assert case.vision_evidence.detections == []

    service.respond(
        message="为什么认为是TB？",
        thread_id="lazy-thread",
        user_id="lazy-user",
        owner_scope="tenant:lazy",
        case_id=case.case_id,
    )
    assert service.vision.localization_call_count == 0

    first = service.respond(
        message="病灶在哪？",
        thread_id="lazy-thread",
        user_id="lazy-user",
        owner_scope="tenant:lazy",
        case_id=case.case_id,
    )
    second = service.respond(
        message="候选框在哪里？",
        thread_id="lazy-thread",
        user_id="lazy-user",
        owner_scope="tenant:lazy",
        case_id=case.case_id,
    )

    assert service.vision.localization_call_count == 1
    assert "候选" in first.summary
    assert "候选" in second.summary
    assert "%" not in "\n".join(
        [
            first.summary,
            *first.visual_evidence_notes,
            second.summary,
            *second.visual_evidence_notes,
        ]
    )
    stored = service.store.get_case(case.case_id, "tenant:lazy")
    assert stored.localization_evidence.status in {"completed", "completed_no_detection"}
    assert "detector_execution:not_requested" in stored.vision_evidence.artifact_refs


def test_comparison_without_prior_reports_unavailable_longitudinal_capability(tmp_path):
    service = TBXAgentService(_argmax_settings(tmp_path))
    case, _ = service.assess_cxr(
        _quality_ok_png(),
        user_id="compare-user",
        owner_scope="tenant:compare",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    response = service.respond(
        message="和半年前相比恶化了吗？",
        thread_id="compare-thread",
        user_id="compare-user",
        owner_scope="tenant:compare",
        case_id=case.case_id,
    )

    assert response.response_kind == ResponseKind.SAFE_ABSTENTION
    assert "没有接入可用于比较的既往胸片" in response.summary
    assert "没有执行前后片比较" in response.summary
    assert "无法判断" in response.summary
    assert "请同时上传" not in response.summary
    assert "结核训练类" not in response.summary
    assert response.visual_result is None
    assert response.predicted_class is None
    assert response.visual_evidence_notes == []
    assert response.next_step_information == []
    assert response.diagnostic_information == []
    assert response.citations == []
    assert service.vision.localization_call_count == 0


def test_quality_feedback_reads_qc_instead_of_repeating_classification(tmp_path):
    service = TBXAgentService(_argmax_settings(tmp_path))
    case, _ = service.assess_cxr(
        _quality_ok_png(),
        user_id="quality-user",
        owner_scope="tenant:quality",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )

    response = service.respond(
        message="图像质量差",
        thread_id="quality-thread",
        user_id="quality-user",
        owner_scope="tenant:quality",
        case_id=case.case_id,
    )

    assert response.response_kind == ResponseKind.CASE_EXPLANATION
    assert "基础输入可用性检查" in response.summary
    assert "已覆盖" in response.summary
    assert "未覆盖" in response.summary
    assert "摆位" in response.summary
    assert "如果你认为" not in response.summary
    assert "结核训练类" not in response.summary
    assert response.visual_result is None
    assert response.predicted_class is None
    assert response.visual_evidence_notes == []
    assert response.next_step_information == []
    assert response.citations == []
    assert service.vision.localization_call_count == 0


def test_visual_response_uses_stable_class_order_when_classifier_argmax_is_tied(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.45, "sick_non_tb": 0.45, "tb": 0.1}),
    )
    case, response = _upload_and_classify(
        service,
        _png(),
        user_id="u1",
        owner_scope="tenant:tied-class",
    )

    assert case.fusion_decision.predicted_class == ClassifierClass.HEALTHY
    assert response.predicted_class == ClassifierClass.HEALTHY
    assert response.visual_result == VisualResult.MODEL_NOT_FLAGGED
    assert response.summary.endswith("模型更倾向于健康类。")
    assert "classification_exact_argmax_tie" in case.uncertainty_flags


def test_active_screening_consent_and_red_flag_stop(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    session, response = service.start_active_screening(
        thread_id="thread-screen",
        user_id="u1",
        owner_scope="tenant:u1",
        consent=True,
    )
    assert response.next_question is not None
    assert response.next_question.question_id == "emergency_red_flags"

    session, response = service.answer_active_screening(
        session_id=session.session_id,
        user_id="u1",
        owner_scope="tenant:u1",
        question_id="emergency_red_flags",
        answer=["严重呼吸困难"],
    )
    assert session.status == "complete"
    assert response.urgency == Urgency.EMERGENCY


def test_active_screening_combines_the_bound_cxr_result(tmp_path):
    service = TBXAgentService(
        _argmax_settings(tmp_path),
        vision_backend=_StaticArgmaxBackend({"healthy": 0.1, "sick_non_tb": 0.2, "tb": 0.7}),
    )
    case, _ = _upload_and_classify(
        service,
        _quality_ok_png(),
        user_id="u1",
        owner_scope="tenant:u1",
    )

    session, response = service.start_active_screening(
        thread_id="thread-screen-with-cxr",
        user_id="u1",
        owner_scope="tenant:u1",
        consent=True,
        case_id=case.case_id,
    )
    assert response.case_id == case.case_id
    assert response.visual_result == VisualResult.MODEL_FLAGGED
    assert response.predicted_class == ClassifierClass.TB
    assert response.visual_evidence_notes == ["当前胸片模型分流为结核样本训练类别。"]
    assert response.summary.startswith("已载入当前胸片模型结果")

    while session.status == "collecting":
        question = service.screening.current_question(session)
        assert question is not None
        answer = ["以上均无"] if question.question_id == "emergency_red_flags" else "跳过"
        session, response = service.answer_active_screening(
            session_id=session.session_id,
            user_id="u1",
            owner_scope="tenant:u1",
            question_id=question.question_id,
            answer=answer,
        )

    assert response.response_kind == ResponseKind.ACTIVE_SCREENING_SUMMARY
    assert response.urgency == Urgency.PROMPT_EVALUATION
    assert response.visual_result == VisualResult.MODEL_FLAGGED
    assert response.next_step_information[0].startswith("当前胸片模型结果需要进一步评估")


def test_report_is_created_under_runtime_artifacts(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    case, _ = _upload_and_classify(
        service,
        _png(),
        user_id="u1",
        owner_scope="tenant:u1",
    )
    report = service.create_report(case_id=case.case_id, owner_scope="tenant:u1", actor_id="u1")

    assert Path(report["markdown_path"]).is_file()
    assert Path(report["json_path"]).is_file()
    assert str(tmp_path) in report["markdown_path"]
    report_text = Path(report["markdown_path"]).read_text(encoding="utf-8")
    report_payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    assert "原生 argmax 训练类（冻结策略分流证据、非诊断）" in report_text
    assert "native_three_class_argmax" in report_text
    assert "冻结 p_tb 筛查阈值" not in report_text
    assert "p_tb 模型分数" not in report_text
    assert "advisory_localization_only" in report_text
    public_vision = report_payload["case"]["vision_evidence"]
    assert "class_probabilities" not in public_vision
    assert "class_probability_order" not in public_vision
    assert "top1_score" not in public_vision
    assert "top2_score" not in public_vision
    assert "top1_top2_margin" not in public_vision
    assert "classifier_threshold" not in public_vision
