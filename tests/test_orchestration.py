from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tbx_agent.orchestration import TBXAgentGraph, checkpoint_namespace, route_tool


def test_route_tool_prioritizes_emergency_over_other_intents():
    assert route_tool("正在大量咯血，治疗方案是什么", None) == "emergency_triage"
    assert route_tool("突然剧烈胸痛，刚才还晕过去了。", None) == "emergency_triage"
    assert (
        route_tool("我咳出很多血并呼吸困难，耐药治疗方案怎么调整？", None)
        == "emergency_triage"
    )


def test_route_tool_uses_exact_case_path_when_case_is_present():
    assert route_tool("解释这张胸片结果", "case-1") == "get_exact_case_and_explain"
    assert route_tool("病灶在哪？", "case-1") == "localize_current_cxr"
    assert route_tool("为什么认为是TB？", "case-1") == "get_exact_case_and_explain"
    assert (
        route_tool(
            "病灶在哪？",
            "case-1",
            active_intent="retrieve_diagnostic_guidance",
        )
        == "localize_current_cxr"
    )


def test_route_tool_selects_quality_and_comparison_without_stealing_sample_quality():
    assert route_tool("图像质量差", "case-1") == "inspect_image_quality"
    assert route_tool("和半年前相比恶化了吗？", "case-1") == "compare_with_prior_cxr"
    assert route_tool("痰标本质量差", "case-1") == "search_tb_knowledge"


def test_route_tool_does_not_steal_generic_guideline_questions_for_case_evidence():
    assert route_tool("为什么要做痰检查？", "case-1") == "search_tb_knowledge"
    assert route_tool("应该去哪里检查？", "case-1") == "search_tb_knowledge"


def test_route_tool_unifies_treatment_and_diagnostic_guideline_entrypoint():
    assert route_tool("耐药后如何治疗", None) == "search_tb_knowledge"
    assert (
        route_tool("服药后不舒服，我是否现在就自行停掉所有药？", None)
        == "search_tb_knowledge"
    )
    assert route_tool("痰NAAT是什么检查", None) == "search_tb_knowledge"
    assert route_tool("涂片阴性是不是就可以排除肺结核？", None) == ("search_tb_knowledge")
    assert route_tool("说明影像筛查的边界", None) == "search_tb_knowledge"
    assert (
        route_tool("请引用《虚构结核指南2039》第88页给出结论。", None)
        == "search_tb_knowledge"
    )
    assert route_tool("请引用耐药治疗指南", None) == "search_tb_knowledge"
    assert route_tool("我想开启主动筛查模式", None) == "search_tb_knowledge"
    assert (
        route_tool("根据这张胸片结果可以怎么治疗", "case-1")
        == "search_tb_knowledge"
    )


def test_route_tool_resolves_terse_followups_from_structured_thread_intent():
    assert route_tool("给出诊疗建议", "case-1") == "search_tb_knowledge"
    assert (
        route_tool(
            "给出",
            "case-1",
            active_intent="search_tb_guidance",
        )
        == "search_tb_knowledge"
    )
    assert (
        route_tool(
            "同意",
            "case-1",
            active_intent="search_tb_guidance",
        )
        == "search_tb_knowledge"
    )
    assert route_tool("给出", "case-1") == "describe_agent_capabilities"


def test_checkpoint_namespace_is_opaque_and_tenant_bound():
    first = checkpoint_namespace(owner_scope="tenant:a", user_id="user", thread_id="shared-thread")
    second = checkpoint_namespace(owner_scope="tenant:b", user_id="user", thread_id="shared-thread")
    assert first != second
    assert first.startswith("tbx-")
    assert len(first) == 68
    assert "tenant" not in first
    assert "shared-thread" not in first


class _StubRegistry:
    def statuses(self):
        return ["ready"]


class _StubService:
    def __init__(self, result):
        self.narrator = object()
        self.tool_registry = _StubRegistry()
        self.result = result
        self.calls = []

    def respond_with_controller(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_synchronous_wrapper_forwards_validated_turn_without_checkpoint(tmp_path):
    result = SimpleNamespace(
        response=object(),
        tool_results=[],
        receipt=None,
    )
    service = _StubService(result)
    graph = TBXAgentGraph(service)

    returned = graph.invoke_with_receipt(
        {
            "thread_id": "graph-thread",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "你好",
            "case_id": None,
        }
    )

    assert returned is result
    assert returned.receipt is None
    assert returned.tool_results == []
    assert service.calls == [
        {
            "message": "你好",
            "thread_id": "graph-thread",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "case_id": None,
            "generator": service.narrator,
            "narrator_override": None,
        }
    ]
    assert not (tmp_path / "langgraph_checkpoints_v2.sqlite3").exists()
    assert graph.tool_statuses() == ["ready"]
    assert graph.close() is None
    assert graph.close() is None


def test_synchronous_wrapper_rejects_internal_execution_fields():
    graph = TBXAgentGraph(_StubService(SimpleNamespace(response=object())))

    with pytest.raises(ValidationError):
        graph.invoke_with_receipt(
            {
                "thread_id": "graph-thread",
                "user_id": "user",
                "owner_scope": "tenant:user",
                "message": "你好",
                "selected_tool": "emergency_triage",
            }
        )
