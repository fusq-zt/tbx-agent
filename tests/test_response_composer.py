from tbx_agent.response_composer import merge_agent_responses
from tbx_agent.schemas import AgentResponse, ResponseKind


def _response(summary: str, kind: ResponseKind, **updates) -> AgentResponse:
    return AgentResponse(
        request_id="request-1",
        trace_id="trace-1",
        thread_id="thread-1",
        case_id="case-1",
        response_kind=kind,
        summary=summary,
        **updates,
    )


def test_compound_composer_uses_only_executed_response_content() -> None:
    classification = _response(
        "三分类模型输出为结核类。",
        ResponseKind.VISUAL_SCREENING_RESULT,
        visual_evidence_notes=["分类结果来自已完成的胸片分类。"],
    )
    localization = _response(
        "检测到 1 个候选区域。",
        ResponseKind.VISUAL_SCREENING_RESULT,
        visual_evidence_notes=["候选区域位于图像右侧上部。"],
    )
    guidance = _response(
        "下一步检查信息。",
        ResponseKind.NEXT_TEST_INFORMATION,
        next_step_information=["到定点医疗机构完成病原学检查。"],
    )

    merged = merge_agent_responses([classification, localization, guidance])

    assert merged.response_kind == ResponseKind.VISUAL_SCREENING_RESULT
    assert merged.summary == (
        "三分类模型输出为结核类。\n\n"
        "检测到 1 个候选区域。\n\n下一步检查信息。"
    )
    assert merged.visual_evidence_notes == [
        "分类结果来自已完成的胸片分类。",
        "候选区域位于图像右侧上部。",
    ]
    assert merged.next_step_information == ["到定点医疗机构完成病原学检查。"]
