"""A selected state projection must not replace independent evidence or failures."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_plan_react_runtime import _upload
from test_react_first_runtime import Decisions, run, tool, wire_tasks
from test_react_first_runtime import services as services

from tbx_agent.capability_answer import TBX_CAPABILITY_ANSWER
from tbx_agent.react_runtime import ReactFirstDomain
from tbx_agent.safety import SafetyVerifier
from tbx_agent.schemas import ResponseKind
from tbx_agent.tools.contracts import ToolCallStatus


def _assert_knowledge_bundle_preserved(result):
    source = next(item.response for item in result.tool_results
                  if item.receipt.model_tool_name == "search_tb_knowledge")
    assert source.claims and source.citations and source.retrieved_evidence
    for field in (
        "claims", "citations", "retrieved_evidence", "source_query", "guideline_scope",
        "guideline_subtopic", "answer_status", "evidence_gap", "diagnostic_information",
        "next_step_information", "treatment_education",
    ):
        assert getattr(result.response, field) == getattr(source, field), field
    assert source.summary in result.response.summary


@pytest.mark.parametrize(("task", "query", "projection_text"), [
    ("explain_app_capabilities", "介绍你的功能，并告诉我结核通常怎么传播？", "我是 TBX-Agent"),
    ("summarize_completed_work", "汇总当前分析进度，并告诉我结核通常怎么传播？", "分类未运行"),
    ("explain_image_quality", "说明上传文件质量，并告诉我结核通常怎么传播？", "基础输入"),
    ("compare_with_prior_image", "与既往胸片比较，并告诉我结核通常怎么传播？", "没有接入"),
])
def test_state_projection_preserves_successful_knowledge_bundle(
    services, task, query, projection_text,
):
    service = services()
    case = _upload(service)
    intent = wire_tasks(task, "search_tb_knowledge")
    result = run(service, Decisions(intent, intent, intent), query, case)
    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert projection_text in result.response.summary
    assert result.response.response_kind != ResponseKind.CAPABILITY_STATEMENT
    _assert_knowledge_bundle_preserved(result)


def test_capability_projection_keeps_visual_results_and_knowledge_together(services):
    service = services()
    case = _upload(service)
    intent = wire_tasks("explain_app_capabilities", "classify_image",
                        "show_detection_boxes", "search_tb_knowledge")
    result = run(service, Decisions(intent, intent, intent, intent, intent),
                 "介绍功能、分类并标出候选区域，再说明下一步一般需要做什么检查。", case)
    assert result.execution_plan["tool_names"] == [
        "classify_cxr", "localize_cxr", "search_tb_knowledge",
    ]
    assert TBX_CAPABILITY_ANSWER in result.response.summary
    assert "模型证据" in result.response.summary and "指南建议" in result.response.summary
    assert "结核类" in result.response.summary and "候选区域" in result.response.summary
    assert result.response.predicted_class is not None
    assert result.response.visual_evidence_notes
    _assert_knowledge_bundle_preserved(result)


def test_capability_projection_preserves_selected_cached_detection(services):
    service = services()
    case = _upload(service)
    run(service, Decisions(tool("localize_cxr"), wire_tasks("show_detection_boxes")),
        "标出候选区域", case)
    calls = service.vision.localization_call_count
    intent = wire_tasks("explain_app_capabilities", "show_detection_boxes")
    result = run(service, Decisions(intent, intent), "介绍功能，然后显示刚才的检测结果。", case)
    assert result.execution_plan["tool_names"] == []
    assert result.execution_plan["cached_evidence"] == ["localization"]
    assert service.vision.localization_call_count == calls
    assert TBX_CAPABILITY_ANSWER in result.response.summary
    assert "检测到 1 个候选区域" in result.response.summary
    assert result.response.visual_evidence_notes


def test_capability_and_status_projection_does_not_normalize_away_status(services):
    service = services()
    case = _upload(service)
    intent = wire_tasks("explain_app_capabilities", "summarize_completed_work")
    result = run(service, Decisions(intent, intent), "介绍功能，以及哪些分析已经完成。", case)
    assert result.execution_plan["tool_names"] == []
    assert TBX_CAPABILITY_ANSWER in result.response.summary
    assert "分类未运行" in result.response.summary and "肺野分割未运行" in result.response.summary
    assert result.response.response_kind == ResponseKind.CASE_EXPLANATION


@pytest.mark.parametrize(("failed_tool", "failure_text"), [
    ("search_tb_knowledge", "指南检索本轮未完成"),
    ("classify_cxr", "胸片分类本轮未完成"),
])
def test_projection_preserves_failure_notice_without_claiming_completion(failed_tool, failure_text):
    # Exercise composition after an explicit focus decision, independently of
    # the graph's earlier stop on a failed observation with no remaining work.
    service = SimpleNamespace(safety=SafetyVerifier({"policy_id": "projection-test"}))
    context = SimpleNamespace(service=service, thread_id="projection")
    state = {
        "request_id": "projection", "trace_id": "projection", "query": "介绍功能并执行分析",
        "case_context": {"image_loaded": False}, "answer_focus": "capabilities",
        "tool_results": [SimpleNamespace(receipt=SimpleNamespace(
            step_id="s1", plan_id="projection", status=ToolCallStatus.FAILED,
            model_tool_name=failed_tool,
        ))],
    }
    response, _ = ReactFirstDomain()._compose(state, context)
    assert TBX_CAPABILITY_ANSWER in response.summary
    assert failure_text in response.summary
    assert response.response_kind == ResponseKind.CASE_EXPLANATION
    assert response.claims == [] and response.citations == []
    assert "已完成" not in response.summary


def test_projection_cannot_hide_missing_required_evidence():
    service = SimpleNamespace(safety=SafetyVerifier({"policy_id": "projection-test"}))
    context = SimpleNamespace(service=service, thread_id="projection")
    state = {
        "request_id": "projection", "trace_id": "projection", "query": "介绍功能并分析胸片",
        "case_context": {"image_loaded": False}, "answer_focus": "capabilities",
        "terminal_reason": "required_evidence_tool_unavailable",
    }
    response, _ = ReactFirstDomain()._compose(state, context)
    assert TBX_CAPABILITY_ANSWER in response.summary
    assert "请先上传胸片" in response.summary
    assert response.response_kind == ResponseKind.CASE_EXPLANATION
    assert response.claims == [] and response.citations == []
