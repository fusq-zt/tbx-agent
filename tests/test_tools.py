from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.orchestration import TBXAgentGraph
from tbx_agent.schemas import (
    AgentResponse,
    GuidelineAnswerStatus,
    GuidelineTaskMemory,
    ResponseKind,
)
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import GuidelineScope
from tbx_agent.tools import (
    ToolCallStatus,
    ToolDefinition,
    ToolInvocation,
    ToolName,
    ToolRegistry,
)
from tbx_agent.tools.contracts import ToolOutcome

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


def _invocation(
    *,
    tool_name: str = "test_tool",
    step_index: int = 0,
    max_steps: int = 1,
) -> ToolInvocation:
    return ToolInvocation(
        tool_name=tool_name,
        message="private patient message",
        thread_id="thread",
        user_id="user",
        owner_scope="tenant:user",
        request_id="request",
        trace_id="trace",
        routing_policy_id="test-router",
        step_index=step_index,
        max_steps=max_steps,
    )


def _response(invocation: ToolInvocation, summary: str = "safe result") -> AgentResponse:
    return AgentResponse(
        request_id=invocation.request_id,
        trace_id=invocation.trace_id,
        thread_id=invocation.thread_id,
        response_kind=ResponseKind.SAFE_ABSTENTION,
        summary=summary,
        limitations=["test limitation"],
    )


def _fallback(
    invocation: ToolInvocation,
    status: ToolCallStatus,
    error_code: str,
) -> AgentResponse:
    return _response(invocation, f"fallback:{status.value}:{error_code}")


def test_registry_returns_typed_receipt_without_raw_message():
    registry = ToolRegistry(max_steps=1)
    registry.register(
        ToolDefinition(
            name="test_tool",
            audit_action="test_action",
            handler=_response,
            timeout_seconds=0.5,
        )
    )

    result = registry.execute(_invocation(), fallback_factory=_fallback)

    assert result.receipt.status == ToolCallStatus.SUCCEEDED
    assert result.receipt.fallback_used is False
    assert result.receipt.request_id == "request"
    assert result.audit_action == "test_action"
    assert "private patient message" not in result.receipt.model_dump_json()
    assert len(result.receipt.input_sha256) == 64
    status = registry.statuses()[0]
    assert status.availability == "ready"
    assert status.deterministic_router is True
    assert status.visual_policy_mutation_allowed is False
    assert status.medical_route_mutation_allowed is False
    registry.close()


def test_insufficient_guideline_status_is_an_evidence_gap_for_any_response_kind():
    registry = ToolRegistry(max_steps=1)

    def gap_response(invocation: ToolInvocation) -> AgentResponse:
        return AgentResponse(
            request_id=invocation.request_id,
            trace_id=invocation.trace_id,
            thread_id=invocation.thread_id,
            response_kind=ResponseKind.TREATMENT_EDUCATION,
            summary="当前受审核知识库没有标准疗程的可引用条款。",
            answer_status=GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE,
            evidence_gap="当前受审核知识库没有标准疗程的可引用条款。",
        )

    registry.register(
        ToolDefinition(
            name="gap_tool",
            audit_action="gap_action",
            handler=gap_response,
            timeout_seconds=0.5,
            allowed_response_kinds=frozenset(
                {ResponseKind.SAFE_ABSTENTION, ResponseKind.TREATMENT_EDUCATION}
            ),
        )
    )

    result = registry.execute(
        _invocation(tool_name="gap_tool"),
        fallback_factory=_fallback,
    )

    assert result.receipt.status == ToolCallStatus.SUCCEEDED
    assert result.receipt.outcome == ToolOutcome.EVIDENCE_GAP
    registry.close()


def test_guidance_invocation_accepts_query_only_and_preserves_legacy_dimensions():
    query_only = _invocation(tool_name=ToolName.SEARCH_TB_GUIDANCE.value)

    assert query_only.guideline_scope is None
    assert query_only.subtopic is None
    assert query_only.population == []
    assert query_only.product_terms == []
    assert query_only.scenario_tags == []

    invocation = ToolInvocation(
        tool_name=ToolName.SEARCH_TB_GUIDANCE.value,
        message="Xpert MTB/RIF、Xpert Ultra 在什么情况下使用？",
        thread_id="thread",
        user_id="user",
        owner_scope="tenant:user",
        request_id="request",
        trace_id="trace",
        routing_policy_id="test-router",
        guideline_scope=GuidelineScope.DIAGNOSTIC_TESTING,
        subtopic="rapid_molecular_diagnostics",
        population=[],
        product_terms=["Xpert MTB/RIF", "Xpert Ultra"],
    )

    assert invocation.guideline_scope == GuidelineScope.DIAGNOSTIC_TESTING
    assert invocation.subtopic == "rapid_molecular_diagnostics"
    assert invocation.product_terms == ["Xpert MTB/RIF", "Xpert Ultra"]

    with pytest.raises(ValidationError):
        ToolInvocation(
            tool_name=ToolName.CLASSIFY_CURRENT_CXR.value,
            message="胸片分类",
            thread_id="thread",
            user_id="user",
            owner_scope="tenant:user",
            request_id="request",
            trace_id="trace",
            routing_policy_id="test-router",
            guideline_scope=GuidelineScope.SCREENING,
        )


@pytest.mark.parametrize(
    (
        "query",
        "scope",
        "subtopic",
        "population",
        "products",
        "scenario_tags",
    ),
    [
        (
            "孕妇怀疑肺结核时应该做什么检查？",
            "special_population",
            "special_population_testing",
            ["pregnant_people"],
            [],
            [],
        ),
        (
            "痰片没查到菌是不是就能排除结核？",
            "diagnostic_testing",
            "negative_test_interpretation",
            [],
            [],
            ["test_smear"],
        ),
        (
            "WHO 是否规定所有肺结核患者都必须住院？",
            "treatment_education",
            "care_setting",
            [],
            [],
            ["care_universal_hospitalization"],
        ),
        (
            "不是问培养，是想问 Xpert 阴性能否排除肺结核？",
            "diagnostic_testing",
            "negative_test_interpretation",
            [],
            ["Xpert MTB/RIF"],
            ["test_naat"],
        ),
    ],
)
def test_query_only_guidance_receipt_records_internal_resolution(
    tmp_path,
    query,
    scope,
    subtopic,
    population,
    products,
    scenario_tags,
):
    service = TBXAgentService(_settings(tmp_path))

    result = service.respond_with_tool(
        selected_tool=ToolName.SEARCH_TB_GUIDANCE,
        message=query,
        thread_id=f"query-first-{scope}-{subtopic}",
        user_id="user",
        owner_scope="tenant:user",
    )

    receipt = result.receipt
    assert receipt.tool_name == "search_tb_knowledge"
    assert receipt.status == ToolCallStatus.SUCCEEDED
    assert receipt.resolved_guideline_scope == scope
    assert receipt.resolved_guideline_subtopic == subtopic
    assert receipt.resolved_population == population
    assert receipt.resolved_product_terms == products
    assert receipt.resolved_scenario_tags == scenario_tags


def test_query_only_search_uses_trusted_thread_memory_only_for_elliptical_followup(
    tmp_path,
):
    service = TBXAgentService(_settings(tmp_path))
    thread = service.store.get_or_create_thread(
        "trusted-guidance-memory",
        "user",
        "tenant:user",
    )
    thread.recent_guideline_task = GuidelineTaskMemory(
        scope="special_population",
        subtopic="special_population_testing",
        population=["pregnant_people"],
    )
    service.store.save_thread(thread)

    followup = service.respond_with_tool(
        selected_tool=ToolName.SEARCH_TB_KNOWLEDGE,
        message="具体该怎么做？",
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        owner_scope=thread.owner_scope,
    )

    assert followup.receipt.resolved_guideline_scope == "special_population"
    assert followup.receipt.resolved_guideline_subtopic == "special_population_testing"
    assert followup.receipt.resolved_population == ["pregnant_people"]

    explicit_switch = service.respond_with_tool(
        selected_tool=ToolName.SEARCH_TB_KNOWLEDGE,
        message="耐药结核和普通结核治疗一样吗？",
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        owner_scope=thread.owner_scope,
    )

    assert explicit_switch.receipt.resolved_guideline_scope == "treatment_education"
    assert explicit_switch.receipt.resolved_guideline_subtopic == (
        "drug_resistant_treatment_comparison"
    )
    assert explicit_switch.receipt.resolved_population == []


def test_registry_timeout_failure_unknown_and_step_limit_fail_closed():
    registry = ToolRegistry(max_steps=1)

    def slow_handler(invocation: ToolInvocation) -> AgentResponse:
        time.sleep(0.05)
        return _response(invocation)

    registry.register(
        ToolDefinition(
            name="slow",
            audit_action="slow_action",
            handler=slow_handler,
            timeout_seconds=0.005,
        )
    )

    timed_out = registry.execute(_invocation(tool_name="slow"), fallback_factory=_fallback)
    unavailable = registry.execute(_invocation(tool_name="unknown"), fallback_factory=_fallback)
    step_limited = registry.execute(
        _invocation(tool_name="slow", step_index=1), fallback_factory=_fallback
    )

    assert timed_out.receipt.status == ToolCallStatus.TIMED_OUT
    assert unavailable.receipt.status == ToolCallStatus.UNAVAILABLE
    assert step_limited.receipt.status == ToolCallStatus.STEP_LIMIT_EXCEEDED
    for result in (timed_out, unavailable, step_limited):
        assert result.receipt.fallback_used is True
        assert result.response.response_kind == ResponseKind.SAFE_ABSTENTION
        assert result.audit_action == "tool_execution_degraded"
    slow_status = registry.statuses()[0]
    assert slow_status.availability == "degraded"
    assert slow_status.last_call_status == ToolCallStatus.STEP_LIMIT_EXCEEDED
    assert slow_status.consecutive_failures == 1
    assert slow_status.detail_code == "tool_timeout"
    registry.close()


def test_agent_boundary_rejects_injected_tool_name(tmp_path):
    settings = _settings(tmp_path)
    service = TBXAgentService(settings)
    graph = TBXAgentGraph(service)

    with pytest.raises(ValidationError):
        graph.invoke_with_receipt(
            {
                "thread_id": "graph-locked-route",
                "user_id": "user",
                "owner_scope": "tenant:user",
                "message": "痰NAAT是什么检查？",
                "case_id": None,
                "selected_tool": ToolName.LOCALIZE_CURRENT_CXR.value,
            }
        )


def test_service_timeout_degrades_before_state_commit(tmp_path):
    service = TBXAgentService(_settings(tmp_path), tool_timeout_seconds=0.005)

    class _SlowRetriever:
        def retrieve(self, *_args, **_kwargs):
            time.sleep(0.05)
            return []

    service.retriever = _SlowRetriever()
    result = service.respond_with_tool(
        selected_tool=ToolName.RETRIEVE_GUIDELINE,
        message="痰NAAT是什么检查？",
        thread_id="timeout-thread",
        user_id="user",
        owner_scope="tenant:user",
    )

    assert result.receipt.status == ToolCallStatus.TIMED_OUT
    assert result.receipt.error_code == "tool_timeout"
    assert result.response.response_kind == ResponseKind.SAFE_ABSTENTION
    assert "没有生成医学建议" in result.response.summary
    state = service.store.get_or_create_thread("timeout-thread", "user", "tenant:user")
    assert state.tool_call_counts == {"tool_execution_degraded": 1}


def test_agent_api_adds_receipt_without_changing_response_fields(tmp_path):
    client = TestClient(create_app(TBXAgentService(_settings(tmp_path))))
    response = client.post(
        "/v1/agent/respond",
        json={
            "thread_id": "api-receipt",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "痰NAAT是什么检查？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]
    assert payload["citations"]
    assert payload["execution_receipt"]["tool_name"] == (
        ToolName.RETRIEVE_GUIDELINE.value
    )
    assert payload["execution_receipt"]["status"] == ToolCallStatus.SUCCEEDED.value
