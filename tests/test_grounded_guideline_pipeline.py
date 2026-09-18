from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from tbx_agent.config import Settings
from tbx_agent.narrator import (
    NarrationRejectedError,
    SafeGuidelineNarration,
    _approved_payload,
    _exact_grounded_narration_schema,
    validate_grounded_narration,
)
from tbx_agent.schemas import (
    GuidelineAnswerStatus,
    NarrationStatus,
)
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import GuidelineScenarioTag, GuidelineScope
from tbx_agent.tools.contracts import ToolInvocation, ToolName

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def service(tmp_path: Path):
    settings = replace(
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
    instance = TBXAgentService(settings)
    yield instance
    instance.tool_registry.close()
    instance.store.close()


def _invoke(
    service: TBXAgentService,
    *,
    message: str,
    scope: GuidelineScope,
    subtopic: str,
    product_terms: list[str] | None = None,
    scenario_tags: list[GuidelineScenarioTag] | None = None,
):
    invocation = ToolInvocation(
        tool_name=ToolName.RETRIEVE_GUIDELINE.value,
        message=message,
        thread_id="grounded-thread",
        user_id="grounded-user",
        owner_scope="tenant:grounded",
        request_id="grounded-request",
        trace_id="grounded-trace",
        routing_policy_id="grounded-test-policy",
        guideline_scope=scope,
        subtopic=subtopic,
        product_terms=product_terms or [],
        scenario_tags=scenario_tags or [],
        max_steps=service.tool_registry.max_steps,
    )
    return service._tool_retrieve_guideline(invocation)  # noqa: SLF001


@pytest.mark.parametrize(
    ("message", "subtopic", "required_chunk"),
    [
        ("哪些人属于 TB 高风险人群？", "risk_groups", "as26_high_risk"),
        ("哪些人建议主动筛查？", "active_screening_population", "as26_key_groups"),
    ],
)
def test_screening_questions_are_answered_from_real_reviewed_chunks(
    service: TBXAgentService,
    message: str,
    subtopic: str,
    required_chunk: str,
):
    response = _invoke(
        service,
        message=message,
        scope=GuidelineScope.SCREENING,
        subtopic=subtopic,
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.guideline_scope == "screening"
    assert required_chunk in {item.chunk_id for item in response.retrieved_evidence}
    assert response.claims
    evidence = {item.chunk_id: item.text for item in response.retrieved_evidence}
    assert all(
        any(claim.text in evidence[chunk_id] for chunk_id in claim.chunk_ids)
        for claim in response.claims
    )
    assert response.narration_status == NarrationStatus.NOT_CONFIGURED
    assert response.narrator_generation_invoked is False


def test_shared_utensil_question_returns_direct_scoped_transmission_evidence(
    service: TBXAgentService,
) -> None:
    response = _invoke(
        service,
        message="共用餐具会传播肺结核吗？",
        scope=GuidelineScope.INFECTION_CONTROL,
        subtopic="shared_utensil_transmission",
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.summary.startswith("通常不会通过共用餐具传播")
    assert [item.chunk_id for item in response.retrieved_evidence] == [
        "cdc24_shared_utensils_not_transmission"
    ]
    assert [item.chunk_id for item in response.citations] == [
        "cdc24_shared_utensils_not_transmission"
    ]
    assert "空气传播" in response.summary
    assert "而不是餐具本身" in response.summary
    assert "不存在共享空气" in response.citations[0].support_text


def test_shared_utensil_question_is_resolved_inside_the_public_search_tool(
    service: TBXAgentService,
) -> None:
    invocation = ToolInvocation(
        tool_name=ToolName.RETRIEVE_GUIDELINE.value,
        message="共用餐具会传播肺结核吗？",
        thread_id="utensil-thread",
        user_id="utensil-user",
        owner_scope="tenant:utensil",
        request_id="utensil-request",
        trace_id="utensil-trace",
        routing_policy_id="utensil-test-policy",
        max_steps=service.tool_registry.max_steps,
    )

    response = service._tool_search_tb_guidance(invocation)  # noqa: SLF001

    assert invocation.guideline_scope == GuidelineScope.INFECTION_CONTROL
    assert invocation.subtopic == "shared_utensil_transmission"
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.summary.startswith("通常不会通过共用餐具传播")
    assert [item.chunk_id for item in response.retrieved_evidence] == [
        "cdc24_shared_utensils_not_transmission"
    ]


def test_xpert_and_ultra_return_partial_generic_naat_evidence(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
        scope=GuidelineScope.DIAGNOSTIC_TESTING,
        subtopic="rapid_molecular_diagnostics",
        product_terms=["Xpert MTB/RIF", "Xpert Ultra"],
    )

    assert response.answer_status == GuidelineAnswerStatus.PARTIAL
    assert response.claims
    assert "通用快速分子检测/NAAT" in (response.evidence_gap or "")
    assert "Xpert MTB/RIF" in (response.evidence_gap or "")
    assert "Xpert Ultra" in (response.evidence_gap or "")
    assert all("xpert" not in evidence.text.casefold() for evidence in response.retrieved_evidence)


def test_structured_test_tag_overrides_negated_raw_test_name(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="不是问培养，是想问Xpert阴性能否排除肺结核？",
        scope=GuidelineScope.DIAGNOSTIC_TESTING,
        subtopic="negative_test_interpretation",
        product_terms=["Xpert MTB/RIF"],
        scenario_tags=[GuidelineScenarioTag.TEST_NAAT],
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert [item.chunk_id for item in response.retrieved_evidence] == [
        "cdc25_xpert_role"
    ]
    assert "培养阴性" not in response.model_dump_json()


def test_query_first_guidance_tool_resolves_negated_entity_inside_retrieval(
    service: TBXAgentService,
):
    class SemanticTaskGenerator:
        backend_id = "semantic-task-test"
        model = "semantic-task-test-model"

        def __init__(self):
            self.schemas = []

        def complete_structured(self, **kwargs):
            schema_name = kwargs["schema_name"]
            self.schemas.append(schema_name)
            assert schema_name == "tbx_react_decision"
            prefix = "TBX_INTERNAL_CONTEXT_JSON="
            system = kwargs["messages"][0]["content"]
            prompt = json.loads(system.split(prefix, maxsplit=1)[1])
            payload = (
                {"action": "answer", "answer_focus": "general",
                 "evidence": ["tb_knowledge"], "answer": None}
                if prompt["observations"]
                else {"action": "tool", "tool": "search_tb_knowledge"}
            )
            return json.dumps(payload, ensure_ascii=False), {
                "prompt_tokens": 20,
                "completion_tokens": 10,
            }

    generator = SemanticTaskGenerator()
    result = service.respond_with_controller(
        message="不是问培养，是想问Xpert阴性能否排除肺结核？",
        thread_id="semantic-tag-integration",
        user_id="grounded-user",
        owner_scope="tenant:grounded",
        generator=generator,
    )

    assert generator.schemas == [
        "tbx_react_decision",
        "tbx_react_decision",
    ]
    assert result.execution_plan["source"] == "plan_react"
    assert result.execution_plan["plan_metadata"]["planning_used"] is False
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert result.trace.task_spec.guideline_scope is None
    assert result.trace.task_spec.subtopic is None
    assert result.trace.task_spec.product_terms == []
    assert result.trace.task_spec.scenario_tags == []
    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    receipt = result.tool_results[0].receipt
    assert receipt.resolved_guideline_scope == "diagnostic_testing"
    assert receipt.resolved_guideline_subtopic == "negative_test_interpretation"
    assert receipt.resolved_product_terms == ["Xpert MTB/RIF"]
    assert receipt.resolved_scenario_tags == ["test_naat"]
    assert [item.chunk_id for item in result.response.citations] == [
        "cdc25_xpert_role"
    ]
    assert "培养阴性" not in result.response.model_dump_json()


def test_missing_tag_specific_clause_returns_gap_instead_of_other_test_evidence(
    service: TBXAgentService,
):
    culture_hit = next(
        hit
        for hit in service.retriever.retrieve_scoped(
            "结核分枝杆菌培养阴性",
            required_claim_scopes={"test_limitations", "culture_interpretation"},
            required_source_ids={"cdc_tb_clinical_lab_2025"},
            top_k=8,
        )
        if hit.citation.chunk_id == "cdc25_culture_negative_limit"
    )
    service.retriever.retrieve_scoped = lambda *_args, **_kwargs: [culture_hit]

    response = _invoke(
        service,
        message="Xpert阴性能否排除肺结核？",
        scope=GuidelineScope.DIAGNOSTIC_TESTING,
        subtopic="negative_test_interpretation",
        scenario_tags=[GuidelineScenarioTag.TEST_NAAT],
    )

    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.claims == []
    assert response.retrieved_evidence == []
    assert response.citations == []
    assert "未使用同主题的其他条款代答" in (response.evidence_gap or "")


@pytest.mark.parametrize(
    ("message", "scope", "subtopic", "gap_fragment"),
    [
        (
            "怀疑肺结核时是否需要佩戴口罩？",
            GuidelineScope.INFECTION_CONTROL,
            "respiratory_protection",
            "呼吸防护或佩戴口罩",
        ),
    ],
)
def test_missing_scopes_abstain_without_borrowing_diagnostic_evidence(
    service: TBXAgentService,
    message: str,
    scope: GuidelineScope,
    subtopic: str,
    gap_fragment: str,
):
    response = _invoke(
        service,
        message=message,
        scope=scope,
        subtopic=subtopic,
    )

    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert gap_fragment in (response.evidence_gap or "")
    assert response.claims == []
    assert response.retrieved_evidence == []
    assert response.citations == []


@pytest.mark.parametrize(
    ("message", "subtopic", "required_chunk", "expected_fragment"),
    [
        (
            "肺结核一般怎么治疗？",
            "treatment_principles",
            "who25_ds_selection_factors",
            "医疗团队",
        ),
        (
            "标准疗程大概是什么？",
            "standard_regimen_duration",
            "who25_ds_duration_options",
            "6个月标准疗程",
        ),
    ],
)
def test_drug_susceptible_treatment_education_uses_reviewed_who_sections(
    service: TBXAgentService,
    message: str,
    subtopic: str,
    required_chunk: str,
    expected_fragment: str,
):
    response = _invoke(
        service,
        message=message,
        scope=GuidelineScope.TREATMENT_EDUCATION,
        subtopic=subtopic,
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.guideline_scope == "treatment_education"
    assert required_chunk in {item.chunk_id for item in response.retrieved_evidence}
    assert expected_fragment in "\n".join(response.treatment_education)
    assert response.citations
    assert all(
        citation.source_id == "who_tb_treatment_module4_2025" for citation in response.citations
    )
    assert all(citation.url.startswith("https://tbksp.who.int/") for citation in response.citations)
    rendered = "\n".join(response.treatment_education)
    assert "毫克" not in rendered
    assert " mg" not in rendered.casefold()


def test_care_setting_is_grounded_in_the_separate_who_handbook(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="肺结核是否都必须住院，哪些情况可以转门诊或社区照护？",
        scope=GuidelineScope.TREATMENT_EDUCATION,
        subtopic="care_setting",
        scenario_tags=[
            GuidelineScenarioTag.CARE_UNIVERSAL_HOSPITALIZATION,
            GuidelineScenarioTag.CARE_INPATIENT_INDICATIONS,
            GuidelineScenarioTag.CARE_AMBULATORY_TRANSITION,
        ],
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert [item.chunk_id for item in response.retrieved_evidence] == [
        "who25_care_setting_ambulatory_majority",
        "who25_care_setting_inpatient_indications",
        "who25_care_setting_early_ambulatory_transition",
    ]
    assert all(
        citation.source_id == "who_tb_treatment_handbook_module4_2025"
        for citation in response.citations
    )
    rendered = "\n".join(response.treatment_education)
    assert "多数结核病患者" in rendered
    assert "医学安全" in rendered
    assert "6个月" not in rendered
    assert "毫克" not in rendered


@pytest.mark.parametrize(
    "message",
    [
        "耐药肺结核怎么治疗？",
        "耐药后如何治疗？",
        "MDR-TB的疗程是什么？",
        "RR-TB应该用什么方案？",
        "How is drug-resistant TB treated?",
    ],
)
def test_drug_resistant_treatment_questions_fail_closed_before_retrieval(
    service: TBXAgentService,
    message: str,
):
    response = _invoke(
        service,
        message=message,
        scope=GuidelineScope.TREATMENT_EDUCATION,
        subtopic="treatment_principles",
    )

    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.claims == []
    assert response.retrieved_evidence == []
    assert response.citations == []
    assert "只包含WHO药物敏感性肺结核" in (response.evidence_gap or "")
    assert "不能把药物敏感性方案套用于耐药情形" in (response.evidence_gap or "")


def test_personalized_dose_request_does_not_reuse_general_ds_tb_evidence(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="我体重62公斤，请给我每天每种抗结核药的剂量和疗程。",
        scope=GuidelineScope.TREATMENT_EDUCATION,
        subtopic="standard_regimen_duration",
    )

    assert response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert response.claims == []
    assert response.retrieved_evidence == []
    assert response.citations == []
    assert "个体治疗方案或剂量请求" in (response.evidence_gap or "")


def test_narrator_receives_query_and_full_evidence_but_cannot_add_claims(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="请整合主动筛查高风险人群和重点人群。",
        scope=GuidelineScope.SCREENING,
        subtopic="risk_groups",
    )
    payload = _approved_payload(response)
    assert payload["query"] == "请整合主动筛查高风险人群和重点人群。"
    assert payload["retrieved_evidence"][0]["text"]
    assert payload["retrieved_evidence"][0]["chunk_id"]

    synthesized_summary = (
        "建议优先关注两类人群：一类是HIV感染者、肺结核患者密切接触者及免疫抑制"
        "相关人群；另一类是糖尿病患者、65岁及以上老年人和学校等人员密集机构人群。"
    )
    selected_chunk_ids = list(
        dict.fromkeys(chunk_id for claim in response.claims for chunk_id in claim.chunk_ids)
    )
    selected = SafeGuidelineNarration(
        answer_status=GuidelineAnswerStatus.ANSWERED,
        summary=synthesized_summary,
        summary_chunk_ids=selected_chunk_ids,
    )
    rendered = validate_grounded_narration(response, selected)
    assert rendered.summary == synthesized_summary
    assert rendered.claims == response.claims
    assert rendered.diagnostic_information == [claim.text for claim in response.claims]
    assert rendered.next_step_information == []
    assert rendered.treatment_education == []

    schema = _exact_grounded_narration_schema(response)
    assert set(schema["properties"]) == {
        "answer_status",
        "summary",
        "summary_chunk_ids",
    }
    assert schema["required"] == [
        "answer_status",
        "summary_chunk_ids",
        "summary",
    ]
    assert "claims" not in schema["properties"]
    assert schema["properties"]["summary"]["maxLength"] == 360
    with pytest.raises(ValidationError, match="claims"):
        SafeGuidelineNarration.model_validate(
            {
                "answer_status": GuidelineAnswerStatus.ANSWERED,
                "summary": synthesized_summary,
                "summary_chunk_ids": selected_chunk_ids,
                "claims": [claim.model_dump(mode="json") for claim in response.claims],
            }
        )

    upgraded = selected.model_copy(update={"answer_status": GuidelineAnswerStatus.PARTIAL})
    with pytest.raises(NarrationRejectedError, match="coverage status"):
        validate_grounded_narration(response, upgraded)

    repeated = selected.model_copy(
        update={"summary": f"{synthesized_summary}{synthesized_summary}"}
    )
    with pytest.raises(NarrationRejectedError, match="repeated a complete sentence"):
        validate_grounded_narration(response, repeated)

    pasted = selected.model_copy(
        update={"summary": "".join(claim.text for claim in response.claims)}
    )
    integrated = validate_grounded_narration(response, pasted)
    assert integrated.summary.startswith("高风险人群主要包括")
    assert "此外" in integrated.summary
    assert all(claim.text not in integrated.summary for claim in response.claims)


def test_narrator_cannot_hide_a_retrieved_claim_from_the_synthesis(
    service: TBXAgentService,
):
    response = _invoke(
        service,
        message="请整合主动筛查高风险人群和重点人群。",
        scope=GuidelineScope.SCREENING,
        subtopic="risk_groups",
    )
    assert len(response.claims) >= 2
    response = response.model_copy(
        update={
            "diagnostic_information": [response.claims[0].text],
            "next_step_information": [response.claims[1].text],
            "treatment_education": [
                response.claims[0].text,
                response.claims[1].text,
            ],
        }
    )
    with pytest.raises(NarrationRejectedError, match="source ids do not match"):
        validate_grounded_narration(
            response,
            SafeGuidelineNarration(
                answer_status=GuidelineAnswerStatus.ANSWERED,
                summary=response.claims[1].text,
                summary_chunk_ids=response.claims[1].chunk_ids,
            ),
        )
