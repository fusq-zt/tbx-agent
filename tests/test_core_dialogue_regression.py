from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.schemas import (
    ClassifierClass,
    GuidelineAnswerStatus,
    VisualResult,
)
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import GuidelineScope
from tbx_agent.vision import MockRank03Backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicTBBackend(MockRank03Backend):
    """Real controller contract with deterministic vision evidence for regression."""

    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        return evidence.model_copy(
            update={
                "class_probabilities": {
                    "healthy": 0.000004,
                    "sick_non_tb": 0.000025,
                    "tb": 0.999971,
                },
                "predicted_class": ClassifierClass.TB,
                "classifier_argmax_tied": False,
                "classifier_flagged": True,
                "top1_score": 0.999971,
                "top2_score": 0.000025,
                "top1_top2_margin": 0.999946,
                "detections": [],
                "detector_flagged": None,
            }
        )


class _OverauthorizingTaskGenerator:
    backend_id = "malicious-test-generator"
    model = "test-model"

    def complete_structured(self, **kwargs):
        if kwargs["schema_name"] == "tbx_react_decision":
            # The model is not allowed to choose another case or supply tool
            # arguments. This intentionally violates that boundary, rather
            # than relying on the retired plan-based tool allowlist.
            return json.dumps({
                "action": "tool", "tool": "localize_cxr",
                "case_id": "foreign-case-selected-by-model",
            }), {"prompt_tokens": 12, "completion_tokens": 8}
        if kwargs["schema_name"] == "tbx_agent_tool_selection":
            return (
                json.dumps(
                    {"tool": "localize_cxr", "direct_answer": None},
                    ensure_ascii=False,
                ),
                {"prompt_tokens": 12, "completion_tokens": 8},
            )
        return "{}", {"prompt_tokens": 6, "completion_tokens": 2}


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
    )


def _usable_png() -> bytes:
    output = io.BytesIO()
    Image.linear_gradient("L").resize((512, 512)).convert("RGB").save(
        output, format="PNG"
    )
    return output.getvalue()


def _turn(
    service: TBXAgentService,
    message: str,
    *,
    thread_id: str,
    case_id: str | None = None,
):
    return service.respond_with_controller(
        message=message,
        thread_id=thread_id,
        user_id="regression-user",
        owner_scope="tenant:regression",
        case_id=case_id,
    )


def _assert_grounded(result) -> None:
    response = result.response
    evidence = {item.chunk_id: item.text for item in response.retrieved_evidence}
    citation_ids = {item.chunk_id for item in response.citations}
    for claim in response.claims:
        assert set(claim.chunk_ids) <= citation_ids
        assert all(chunk_id in evidence for chunk_id in claim.chunk_ids)
        assert any(claim.text in evidence[chunk_id] for chunk_id in claim.chunk_ids)


def _assert_langgraph_execution(result, *, used_tools: list[str]) -> None:
    plan = result.execution_plan
    assert plan["framework"] == "langgraph"
    assert plan["tool_names"] == used_tools
    node_trace = plan["graph_node_trace"]
    if plan["plan_metadata"]["rule_fallback_used"]:
        assert (node_trace[:3] == ["load_context", "plan", "decide"]
                or node_trace[:4] == ["load_context", "decide", "plan", "decide"])
    else:
        assert node_trace[:2] == ["load_context", "decide"]
    assert node_trace[-1] == "finalize"
    assert plan["hidden_reasoning_persisted"] is False


def test_exact_ten_question_dialogue_contract(tmp_path: Path) -> None:
    """Lock the ten failures reported from the real UI as one E2E contract.

    This exercises the production controller, tool registry, reviewed JSONL
    corpus and BM25 retriever.  Only the image backend is deterministic so the
    assertions do not depend on a model weight file in CI.
    """

    settings = _settings(tmp_path)
    backend = _DeterministicTBBackend(
        settings.fusion_policy(), settings.rank03_config()
    )
    service = TBXAgentService(settings, vision_backend=backend)

    guideline_cases = [
        (
            "哪些人属于 TB 高风险人群？",
            GuidelineScope.SCREENING,
            "risk_groups",
            GuidelineAnswerStatus.ANSWERED,
            {"as26_high_risk"},
        ),
        (
            "哪些人建议主动筛查？",
            GuidelineScope.SCREENING,
            "active_screening_population",
            GuidelineAnswerStatus.ANSWERED,
            {"as26_key_groups", "as26_key_group_path"},
        ),
        (
            "Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
            GuidelineScope.DIAGNOSTIC_TESTING,
            "rapid_molecular_diagnostics",
            GuidelineAnswerStatus.PARTIAL,
            {"who25_initial_lc_anaat"},
        ),
        (
            "肺结核一般怎么治疗？",
            GuidelineScope.TREATMENT_EDUCATION,
            "treatment_principles",
            GuidelineAnswerStatus.ANSWERED,
            {"who25_ds_selection_factors"},
        ),
        (
            "标准疗程大概是什么？",
            GuidelineScope.TREATMENT_EDUCATION,
            "standard_regimen_duration",
            GuidelineAnswerStatus.ANSWERED,
            {"who25_ds_duration_options"},
        ),
        (
            "怀疑肺结核时是否需要佩戴口罩？",
            GuidelineScope.INFECTION_CONTROL,
            "respiratory_protection",
            GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE,
            set(),
        ),
    ]

    for index, (query, scope, subtopic, status, required_citations) in enumerate(
        guideline_cases
    ):
        result = _turn(service, query, thread_id=f"guideline-{index}")
        response = result.response
        _assert_langgraph_execution(result, used_tools=["search_tb_knowledge"])
        assert result.receipt is not None
        assert result.receipt.tool_name == "search_tb_knowledge"
        assert result.receipt.model_tool_name == "search_tb_knowledge"
        assert result.receipt.resolved_guideline_scope == scope
        assert result.receipt.resolved_guideline_subtopic == subtopic
        assert response.answer_status == status
        citation_ids = {item.chunk_id for item in response.citations}
        assert required_citations <= citation_ids
        assert not {
            "classify_cxr",
            "localize_cxr",
            "analyze_lung_anatomy",
        }.intersection(result.execution_plan["tool_names"])
        _assert_grounded(result)

    high_risk = _turn(
        service,
        "哪些人属于 TB 高风险人群？",
        thread_id="high-risk-wording",
    ).response
    assert "HIV感染者" in high_risk.summary
    assert "完整证据链" not in high_risk.summary
    assert len(high_risk.claims) <= 2

    active_screening = _turn(
        service,
        "哪些人建议主动筛查？",
        thread_id="active-screening-wording",
    ).response
    assert "肺结核主动筛查优先对象包括" in active_screening.summary
    assert "65岁及以上" in active_screening.summary
    assert "15岁及以上老年人" not in active_screening.summary
    assert len(active_screening.claims) <= 2

    xpert = _turn(
        service,
        "Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
        thread_id="xpert-wording",
    ).response
    assert "通用快速分子检测/NAAT" in (xpert.evidence_gap or "")
    assert "Xpert MTB/RIF" in (xpert.evidence_gap or "")
    assert "Xpert Ultra" in (xpert.evidence_gap or "")
    assert {claim.chunk_ids[0] for claim in xpert.claims} == {
        "who25_initial_lc_anaat",
    }

    treatment = _turn(
        service,
        "肺结核一般怎么治疗？",
        thread_id="grounded-treatment-principles",
    ).response
    assert treatment.answer_status == GuidelineAnswerStatus.ANSWERED
    assert "医疗团队" in "\n".join(treatment.treatment_education)
    assert "2HRZE" not in "\n".join(treatment.treatment_education)
    assert treatment.citations

    duration = _turn(
        service,
        "标准疗程大概是什么？",
        thread_id="grounded-treatment-duration",
    ).response
    assert duration.answer_status == GuidelineAnswerStatus.ANSWERED
    duration_text = "\n".join(duration.treatment_education)
    assert "6个月标准疗程" in duration_text
    assert "毫克" not in duration_text
    assert duration.citations

    respiratory = _turn(
        service,
        "怀疑肺结核时是否需要佩戴口罩？",
        thread_id="no-answer-respiratory-protection",
    ).response
    assert respiratory.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert respiratory.citations == []
    assert "快速分子检测" not in respiratory.summary

    case = service.assess_cxr(
        _usable_png(),
        user_id="regression-user",
        owner_scope="tenant:regression",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]
    image_thread = "image-dialogue"

    classification = _turn(
        service,
        "这张胸片有没有结核病？",
        thread_id=image_thread,
        case_id=case.case_id,
    )
    _assert_langgraph_execution(classification, used_tools=["classify_cxr"])
    assert classification.receipt is not None
    assert classification.receipt.tool_name == "classify_current_cxr"
    assert classification.receipt.model_tool_name == "classify_cxr"
    assert classification.response.predicted_class == ClassifierClass.TB
    assert classification.response.visual_result == VisualResult.MODEL_FLAGGED
    assert backend.call_count == 1
    assert backend.localization_call_count == 0
    rationale = _turn(
        service,
        "为什么认为是TB？",
        thread_id=image_thread,
        case_id=case.case_id,
    )
    rationale_text = "\n".join(
        [rationale.response.summary, *rationale.response.visual_evidence_notes]
    )
    _assert_langgraph_execution(rationale, used_tools=[])
    assert "结核类" in rationale.response.summary
    assert "分类模型" in rationale.response.summary
    assert rationale.response.visual_evidence_notes == []
    assert "%" not in rationale_text
    assert "相对得分" not in rationale_text
    assert "相对分数" not in rationale_text
    assert "D-FINE" not in rationale_text
    assert "99.997" not in rationale_text
    assert backend.localization_call_count == 0

    comparison = _turn(
        service,
        "和半年前相比恶化了吗？",
        thread_id=image_thread,
        case_id=case.case_id,
    )
    _assert_langgraph_execution(comparison, used_tools=[])
    assert "没有" in comparison.response.summary
    assert "既往胸片" in comparison.response.summary
    assert "比较" in comparison.response.summary
    assert "无法" in comparison.response.summary
    assert "前后片对比已完成" not in comparison.response.summary
    assert "结核类" not in comparison.response.summary
    assert backend.localization_call_count == 0

    quality = _turn(
        service,
        "图像质量有问题吗",
        thread_id=image_thread,
        case_id=case.case_id,
    )
    _assert_langgraph_execution(quality, used_tools=[])
    assert "基础" in quality.response.summary
    assert "输入可用性检查" in quality.response.summary
    assert "未发现" in quality.response.summary
    assert "文件解码" in quality.response.summary
    assert "灰度动态范围" in quality.response.summary
    assert "如果你认为" not in quality.response.summary
    assert "结核类" not in quality.response.summary
    assert backend.call_count == 1
    assert backend.localization_call_count == 0


@pytest.mark.parametrize(
    ("query", "scope", "subtopic", "population", "product_terms"),
    (
        (
            "哪些人属于 TB 高风险人群？",
            GuidelineScope.SCREENING,
            "risk_groups",
            ["tb_high_risk_population"],
            [],
        ),
        (
            "Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
            GuidelineScope.DIAGNOSTIC_TESTING,
            "rapid_molecular_diagnostics",
            [],
            ["Xpert MTB/RIF", "Xpert Ultra"],
        ),
        (
            "标准疗程大概是什么？",
            GuidelineScope.TREATMENT_EDUCATION,
            "standard_regimen_duration",
            [],
            [],
        ),
    ),
)
def test_terse_guideline_followup_keeps_the_last_successful_query_dimensions(
    tmp_path: Path,
    query: str,
    scope: GuidelineScope,
    subtopic: str,
    population: list[str],
    product_terms: list[str],
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    thread_id = "continuation-" + subtopic

    first = _turn(service, query, thread_id=thread_id)
    _assert_langgraph_execution(first, used_tools=["search_tb_knowledge"])
    continuation = _turn(service, "展开", thread_id=thread_id)

    _assert_langgraph_execution(continuation, used_tools=["search_tb_knowledge"])
    assert continuation.receipt is not None
    assert continuation.receipt.resolved_guideline_scope == scope
    assert continuation.receipt.resolved_guideline_subtopic == subtopic
    assert continuation.receipt.resolved_population == population
    assert continuation.receipt.resolved_product_terms == product_terms
    stored = service.store.get_or_create_thread(
        thread_id,
        "regression-user",
        "tenant:regression",
    )
    assert stored.recent_guideline_task is not None
    assert stored.recent_guideline_task.scope == scope.value
    assert stored.recent_guideline_task.subtopic == subtopic


def test_llm_cannot_supply_a_foreign_case_for_localization(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _DeterministicTBBackend(
        settings.fusion_policy(), settings.rank03_config()
    )
    service = TBXAgentService(settings, vision_backend=backend)
    case = service.assess_cxr(
        _usable_png(),
        user_id="regression-user",
        owner_scope="tenant:regression",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]

    result = service.respond_with_controller(
        message="这张胸片有没有结核病？",
        thread_id="overauthorization-regression",
        user_id="regression-user",
        owner_scope="tenant:regression",
        case_id=case.case_id,
        generator=_OverauthorizingTaskGenerator(),
    )

    _assert_langgraph_execution(result, used_tools=["classify_cxr"])
    assert result.receipt is not None
    assert result.receipt.tool_name == "classify_current_cxr"
    assert result.receipt.model_tool_name == "classify_cxr"
    assert result.receipt.selection_source == "plan_evidence_fallback"
    assert result.execution_plan["plan_revisions"] == []
    # The new decision contract rejects model-authored case IDs before they
    # can become a pending invocation. Rule recovery remains bound to the
    # actual authorized case and preserves its successful evidence.
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True
    assert result.execution_plan["graph_node_trace"][:4] == [
        "load_context", "decide", "plan", "decide",
    ]
    assert result.trace.terminal.reason_code == "react_answered_with_evidence_fallback"
    assert any(step["status"] == "selector_failed_plan_enforced"
               for step in result.execution_plan["react_steps"])
    assert all(
        item.receipt.model_tool_name != "localize_cxr"
        for item in result.tool_results
    )
    assert result.response.predicted_class == ClassifierClass.TB
    assert backend.call_count == 1
    assert backend.localization_call_count == 0
