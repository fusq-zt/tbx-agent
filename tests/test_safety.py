from __future__ import annotations

import pytest

from tbx_agent.capability_answer import TBX_CAPABILITY_ANSWER
from tbx_agent.safety import (
    SafetyVerifier,
    SafetyViolationError,
    assess_red_flags,
    is_medication_change_request,
)
from tbx_agent.schemas import (
    AgentResponse,
    Citation,
    ClassifierClass,
    ResponseKind,
    ReviewStatus,
    Urgency,
    VisualResult,
)

POLICY = {
    "policy_id": "tbx-agent-safety-v1",
    "forbid_disease_confirmation_from_agent": True,
    "forbid_personalized_regimen": True,
    "require_citation_for_guideline_claims": True,
}


def _response(summary: str, **updates) -> AgentResponse:
    response_kind = updates.pop("response_kind", ResponseKind.SAFE_ABSTENTION)
    return AgentResponse(
        request_id="request",
        trace_id="trace",
        thread_id="thread",
        response_kind=response_kind,
        summary=summary,
        **updates,
    )


def test_scope_notice_is_allowed():
    verifier = SafetyVerifier(POLICY)
    response = _response("本系统不能确诊或排除肺结核。")
    assert verifier.verify(response) is response


def test_case_bound_active_screening_may_carry_xray_routing_evidence():
    response = _response(
        "请继续回答；本系统不能确诊或排除肺结核。",
        response_kind=ResponseKind.ACTIVE_SCREENING_QUESTION,
        case_id="case-screening",
        visual_result=VisualResult.MODEL_FLAGGED,
        predicted_class=ClassifierClass.TB,
        review_status=ReviewStatus.NOT_REQUIRED,
        visual_evidence_notes=["当前胸片模型分流为结核样本训练类别。"],
    )

    assert SafetyVerifier(POLICY).verify(response) is response


def test_active_screening_rejects_xray_fields_without_case_provenance():
    response = _response(
        "请继续回答；本系统不能确诊或排除肺结核。",
        response_kind=ResponseKind.ACTIVE_SCREENING_QUESTION,
        visual_result=VisualResult.MODEL_FLAGGED,
    )

    with pytest.raises(SafetyViolationError, match="case provenance"):
        SafetyVerifier(POLICY).verify(response)


@pytest.mark.parametrize(
    "safe_boundary",
    [
        "本系统不能确诊肺结核，也不能排除肺结核。",
        "该检查不用于确诊肺结核，目前无法完全排除肺结核。",
        "现有证据尚未确诊肺结核，也尚不能排除肺结核。",
    ],
)
def test_negated_diagnostic_claims_are_not_misclassified(safe_boundary: str):
    response = _response(
        safe_boundary,
        limitations=["本系统不用于确诊或排除肺结核。"],
    )
    assert SafetyVerifier(POLICY).verify(response) is response


def test_missing_scope_notice_is_blocked():
    with pytest.raises(SafetyViolationError, match="scope notice"):
        SafetyVerifier(POLICY).verify(_response("这里只是一个普通回答。"))


@pytest.mark.parametrize(
    "unsafe",
    ["你已确诊肺结核。", "现在可以排除肺结核。", "肯定不是肺结核。"],
)
def test_diagnosis_or_exclusion_claim_is_blocked(unsafe: str):
    with pytest.raises(SafetyViolationError):
        SafetyVerifier(POLICY).verify(_response(unsafe))


def test_personal_dose_is_blocked():
    with pytest.raises(SafetyViolationError):
        SafetyVerifier(POLICY).verify(_response("建议每天600mg。"))


def test_general_nutrition_amount_does_not_require_a_tb_scope_notice():
    response = _response(
        "通常建议把游离糖控制在每天25 g以内。",
        response_kind=ResponseKind.GENERAL_ANSWER,
    )

    assert SafetyVerifier(POLICY).verify(response) is response


def test_general_lifestyle_advice_is_not_mistaken_for_starting_medication():
    response = _response(
        "建议开始规律锻炼。",
        response_kind=ResponseKind.GENERAL_ANSWER,
    )

    assert SafetyVerifier(POLICY).verify(response) is response


@pytest.mark.parametrize(
    "unsafe",
    [
        "异烟肼每天300 mg。",
        "建议开始服用异烟肼。",
    ],
)
def test_general_answer_cannot_smuggle_medication_instructions(unsafe: str):
    with pytest.raises(SafetyViolationError):
        SafetyVerifier(POLICY).verify(
            _response(unsafe, response_kind=ResponseKind.GENERAL_ANSWER)
        )


@pytest.mark.parametrize("claim", [
    "我已经调用工具删除了全部记录。", "正在执行删除所有病例的指令。",
    "即将运行工具修改病例。",
])
def test_general_answer_cannot_claim_unreceipted_tool_execution(claim):
    with pytest.raises(SafetyViolationError, match="no tool receipt"):
        SafetyVerifier(POLICY).verify(
            _response(
                claim,
                response_kind=ResponseKind.GENERAL_ANSWER,
            )
        )


@pytest.mark.parametrize("claim", [
    "左上肺存在一个约2厘米的结节。",
    "双肺未见明显异常。",
    "这张胸片显示右侧积液。",
    "候选区域位于左肺上部。",
    "发现一个直径为8毫米的空洞。",
    "The left upper lung contains a 2 cm nodule.",
    "Your chest X-ray shows an opacity.",
    "No pleural effusion is identified.",
])
def test_general_chat_cannot_assert_patient_image_findings(claim):
    with pytest.raises(SafetyViolationError, match="ungrounded image findings"):
        SafetyVerifier(POLICY).verify(
            _response(claim, response_kind=ResponseKind.GENERAL_ANSWER)
        )


@pytest.mark.parametrize("text", [
    "25", "不客气。", "肺野是胸片上的二维区域，不等同于解剖学肺叶。",
    "肺结节是一种影像学描述。", "当前无法判断双肺是否正常。",
    "不能据此确定这张胸片显示什么。", "如果左肺存在结节，需要结合正式影像报告。",
    "I cannot determine whether the left lung is normal.",
    "A lung nodule is an imaging finding.",
])
def test_general_explanations_do_not_claim_image_evidence(text):
    assert SafetyVerifier(POLICY).verify(
        _response(text, response_kind=ResponseKind.GENERAL_ANSWER)
    ).summary == text


def test_capability_statement_is_always_tbx_specific():
    response = _response(
        "我是一个通用 AI，可以写代码、画图和回答任何问题。",
        response_kind=ResponseKind.CAPABILITY_STATEMENT,
    )

    verified = SafetyVerifier(POLICY).verify(response)

    assert verified.summary == TBX_CAPABILITY_ANSWER
    assert "胸片" in verified.summary
    assert "受审核指南" in verified.summary
    assert "写代码" not in verified.summary


@pytest.mark.parametrize(
    ("unsafe", "expected"),
    [
        ("候选区域位于右肺上叶。", "候选区域位于右上肺野。"),
        ("候选区域位于左下叶。", "候选区域位于左下肺野。"),
        ("The candidate is in the right upper lobe.", "right upper lung field"),
    ],
)
def test_visual_summary_cannot_convert_two_dimensional_lung_field_to_lobe(
    unsafe: str,
    expected: str,
):
    response = _response(
        unsafe,
        response_kind=ResponseKind.VISUAL_SCREENING_RESULT,
        case_id="case-1",
        visual_result=VisualResult.MODEL_FLAGGED,
        predicted_class=ClassifierClass.TB,
        review_status=ReviewStatus.NOT_REQUIRED,
        visual_evidence_notes=["候选区域位于右上肺野。"],
        limitations=["本系统不用于确诊或排除肺结核。"],
    )

    verified = SafetyVerifier(POLICY).verify(response)

    assert expected in verified.summary
    assert "上叶" not in verified.summary
    assert "下叶" not in verified.summary
    assert "upper lobe" not in verified.summary.casefold()


def test_confirmation_and_exclusion_controls_are_independent():
    confirmation_allowed = dict(POLICY, forbid_disease_confirmation_from_agent=False)
    response = _response(
        "已经确诊肺结核。",
        limitations=["本系统不用于确诊或排除肺结核。"],
    )
    assert SafetyVerifier(confirmation_allowed).verify(response) is response

    exclusion_allowed = dict(POLICY, forbid_disease_exclusion_from_agent=False)
    response = _response(
        "已经排除肺结核。",
        limitations=["本系统不用于确诊或排除肺结核。"],
    )
    assert SafetyVerifier(exclusion_allowed).verify(response) is response


def test_scope_notice_must_state_the_non_diagnostic_boundary():
    with pytest.raises(SafetyViolationError, match="scope notice"):
        SafetyVerifier(POLICY).verify(_response("还需要进一步确诊。"))


def test_duplicate_citations_are_blocked():
    citation = Citation(
        chunk_id="chunk-1",
        source_id="source-1",
        title="指南",
        organization="机构",
        publication_year=2026,
        section="章节",
        locator="页1",
        url="https://example.test/guide",
        support_text="支持文本",
    )
    response = _response(
        "本系统不能确诊或排除肺结核。",
        citations=[citation, citation],
    )

    with pytest.raises(SafetyViolationError, match="duplicate guideline citations"):
        SafetyVerifier(POLICY).verify(response)


def test_emergency_rule_is_separate_local_safety_layer():
    result = assess_red_flags("我现在大量咯血而且喘不上气")
    assert result.urgency == Urgency.EMERGENCY
    assert "large_or_persistent_hemoptysis" in result.matched
    assert "severe_breathlessness" in result.matched


@pytest.mark.parametrize(
    ("message", "expected_labels"),
    [
        (
            "突然剧烈胸痛，刚才还晕过去了。",
            {"severe_chest_pain", "confusion_or_fainting"},
        ),
        (
            "我咳出很多血并呼吸困难，耐药治疗方案怎么调整？",
            {"large_or_persistent_hemoptysis"},
        ),
    ],
)
def test_emergency_rule_understands_common_chinese_phrasing(
    message: str, expected_labels: set[str]
):
    result = assess_red_flags(message)

    assert result.urgency == Urgency.EMERGENCY
    assert expected_labels <= set(result.matched)


@pytest.mark.parametrize(
    "message",
    [
        "我是否现在就自行停掉所有药？",
        "不舒服的话是不是应该停药？",
        "我能不能自己减量？",
    ],
)
def test_medication_change_request_is_detected(message: str):
    assert is_medication_change_request(message)


def test_medication_education_without_a_change_request_is_not_misclassified():
    assert not is_medication_change_request("抗结核治疗为什么需要规律随访？")
