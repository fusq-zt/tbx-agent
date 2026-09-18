from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from tbx_agent.langgraph_runtime import (
    LANGGRAPH_NODE_NAMES,
    GraphRoute,
    PlanReActGraphContext,
    PlanReActRuntimeOps,
    compiled_graph,
    invoke_plan_react_graph,
    plan_react_provenance,
    workflow,
)
from tbx_agent.llm.tool_calling import (
    EmptyToolArguments,
    HighLevelToolCall,
    HighLevelToolName,
    HighLevelToolSelection,
    ToolSelectionMode,
)
from tbx_agent.orchestration import TBXAgentGraph


def _tool_selection(name: HighLevelToolName) -> HighLevelToolSelection:
    return HighLevelToolSelection(
        tool_call=HighLevelToolCall(name=name, arguments=EmptyToolArguments()),
        mode=ToolSelectionMode.JSON_SCHEMA_FALLBACK,
    )


@dataclass
class _FakeDomain:
    calls: list[str] = field(default_factory=list)
    decisions: int = 0
    observations: int = 0

    def load_context(self, state, context):
        self.calls.append("load_context")
        assert context.thread_id == "thread-1"
        return {"context_loaded": True, "next_node": GraphRoute.DECIDE}

    def plan(self, state, context):
        self.calls.append("plan")
        return {"plan_metadata": {"source": "test"}, "next_node": GraphRoute.DECIDE}

    def decide(self, state, context):
        self.calls.append("decide")
        self.decisions += 1
        if self.decisions == 1:
            return {
                "pending_selection": _tool_selection(HighLevelToolName.CLASSIFY_CXR),
                "next_node": GraphRoute.EXECUTE_TOOL,
            }
        return {
            "pending_selection": HighLevelToolSelection(
                direct_answer="已根据新的分类证据回答。",
                mode=ToolSelectionMode.JSON_SCHEMA_FALLBACK,
            ),
            "final_direct_answer": "已根据新的分类证据回答。",
            "terminal_reason": "react_answered",
            "next_node": GraphRoute.FINALIZE,
        }

    def execute_tool(self, state, context):
        self.calls.append("execute_tool")
        assert state["pending_selection"].tool_call.name == HighLevelToolName.CLASSIFY_CXR
        return {"pending_tool_result": {"status": "succeeded"}}

    def observe(self, state, context):
        self.calls.append("observe")
        self.observations += 1
        return {
            "observations": [{"tool": "classify_cxr", "status": "succeeded"}],
            "pending_tool_result": None,
            "next_node": GraphRoute.DECIDE,
        }

    def replan(self, state, context):
        self.calls.append("replan")
        return {"next_node": GraphRoute.DECIDE}

    def finalize(self, state, context):
        self.calls.append("finalize")
        return {
            "result": {"summary": state["final_direct_answer"]},
            "terminal_reason": state["terminal_reason"],
        }

    def ops(self) -> PlanReActRuntimeOps:
        return PlanReActRuntimeOps(
            load_context=self.load_context,
            plan=self.plan,
            decide=self.decide,
            execute_tool=self.execute_tool,
            observe=self.observe,
            replan=self.replan,
            finalize=self.finalize,
        )


def _context(domain: _FakeDomain) -> PlanReActGraphContext:
    return PlanReActGraphContext(
        ops=domain.ops(),
        service=object(),
        thread_id="thread-1",
        user_id="user-1",
        owner_scope="tenant:user-1",
        case_id="case-1",
        generator=None,
        narrator_override=None,
    )


def test_public_workflow_is_real_state_graph_with_required_nodes():
    assert workflow.__class__.__name__ == "StateGraph"
    assert compiled_graph.__class__.__name__ == "CompiledStateGraph"
    graph = compiled_graph.get_graph()
    assert {
        "load_context",
        "plan",
        "decide",
        "execute_tool",
        "observe",
        "replan",
        "finalize",
    }.issubset(graph.nodes)
    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert {
        ("load_context", "decide"),
        ("load_context", "plan"),
        ("decide", "plan"),
        ("decide", "decide"),
    }.issubset(edges)


def test_public_provenance_describes_production_graph_and_tool_boundary():
    provenance = plan_react_provenance()

    assert provenance["framework"] == "langgraph"
    assert provenance["policy_id"] == "tbx-react-first-v4"
    assert provenance["strategy"] == "react_first_optional_plan"
    assert provenance["graph_nodes"] == list(LANGGRAPH_NODE_NAMES)
    assert provenance["model_visible_tools"] == [
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    ]
    assert provenance["tool_selection_priority"] == [
        "structured_react_decision",
    ]
    assert provenance["tool_calling"] == {
        "preferred": "strict_json_schema_decision",
        "fallback": "rule_plan_after_model_error",
    }
    assert provenance["durable_langgraph_checkpointer"] is False
    assert provenance["business_state_authority"] == "SQLiteStore"
    assert provenance["replans_after_each_observation"] is False
    assert provenance["decides_after_each_observation"] is True
    assert provenance["replan_policy"] == {
        "trigger": "explicit_failure_or_missing_obligation",
        "max_plan_revisions": 2,
    }
    assert provenance["trusted_projection_before_model_planning"] is False
    assert provenance["planning_policy"] == "model_requested_complex_tasks_only"


def test_compiled_graph_runs_tool_observation_then_fresh_decision():
    domain = _FakeDomain()

    result = invoke_plan_react_graph(
        query="这张胸片有没有结核？",
        context=_context(domain),
        max_react_iterations=4,
    )

    assert result == {"summary": "已根据新的分类证据回答。"}
    assert domain.calls == [
        "load_context",
        "decide",
        "execute_tool",
        "observe",
        "decide",
        "finalize",
    ]
    assert domain.observations == 1


def test_graph_replans_conditionally_before_deciding_again():
    domain = _FakeDomain()

    def first_decision_requests_replan(state, context):
        domain.calls.append("decide")
        domain.decisions += 1
        if domain.decisions == 1:
            return {
                "replan_trigger": "invalid_action",
                "next_node": GraphRoute.REPLAN,
            }
        return {
            "pending_selection": HighLevelToolSelection(
                direct_answer="已恢复。",
                mode=ToolSelectionMode.JSON_SCHEMA_FALLBACK,
            ),
            "final_direct_answer": "已恢复。",
            "terminal_reason": "react_answered",
            "next_node": GraphRoute.FINALIZE,
        }

    ops = domain.ops()
    context = _context(domain)
    context = PlanReActGraphContext(
        ops=PlanReActRuntimeOps(
            load_context=ops.load_context,
            plan=ops.plan,
            decide=first_decision_requests_replan,
            execute_tool=ops.execute_tool,
            observe=ops.observe,
            replan=ops.replan,
            finalize=ops.finalize,
        ),
        service=context.service,
        thread_id=context.thread_id,
        user_id=context.user_id,
        owner_scope=context.owner_scope,
        case_id=context.case_id,
        generator=context.generator,
        narrator_override=context.narrator_override,
    )

    result = invoke_plan_react_graph(
        query="请恢复",
        context=context,
        max_react_iterations=4,
    )

    assert result == {"summary": "已恢复。"}
    assert domain.calls == [
        "load_context",
        "decide",
        "replan",
        "decide",
        "finalize",
    ]


def test_graph_rejects_execute_route_without_one_validated_tool_call():
    domain = _FakeDomain()
    ops = domain.ops()

    def invalid_decision(state, context):
        return {
            "pending_selection": HighLevelToolSelection(
                direct_answer="错误路由",
                mode=ToolSelectionMode.JSON_SCHEMA_FALLBACK,
            ),
            "next_node": GraphRoute.EXECUTE_TOOL,
        }

    context = _context(domain)
    invalid_context = PlanReActGraphContext(
        ops=PlanReActRuntimeOps(
            load_context=ops.load_context,
            plan=ops.plan,
            decide=invalid_decision,
            execute_tool=ops.execute_tool,
            observe=ops.observe,
            replan=ops.replan,
            finalize=ops.finalize,
        ),
        service=context.service,
        thread_id=context.thread_id,
        user_id=context.user_id,
        owner_scope=context.owner_scope,
        case_id=context.case_id,
        generator=context.generator,
        narrator_override=context.narrator_override,
    )

    with pytest.raises(ValueError, match="exactly one tool call"):
        invoke_plan_react_graph(
            query="invalid",
            context=invalid_context,
            max_react_iterations=2,
        )


def test_graph_rejects_unmodelled_state_fields_from_domain_callbacks():
    domain = _FakeDomain()
    ops = domain.ops()
    context = _context(domain)
    bad_context = PlanReActGraphContext(
        ops=PlanReActRuntimeOps(
            load_context=lambda state, ctx: {"chain_of_thought": "must not persist"},
            plan=ops.plan,
            decide=ops.decide,
            execute_tool=ops.execute_tool,
            observe=ops.observe,
            replan=ops.replan,
            finalize=ops.finalize,
        ),
        service=context.service,
        thread_id=context.thread_id,
        user_id=context.user_id,
        owner_scope=context.owner_scope,
        case_id=context.case_id,
        generator=context.generator,
        narrator_override=context.narrator_override,
    )

    with pytest.raises(ValueError, match="unknown state fields"):
        invoke_plan_react_graph(
            query="invalid",
            context=bad_context,
            max_react_iterations=2,
        )


def test_public_tbx_graph_boundary_invokes_production_compiled_graph(monkeypatch):
    expected = SimpleNamespace(response=object(), tool_results=[], receipt=None)
    calls = []

    class _Locks:
        @contextmanager
        def hold(self, key):
            yield

    class _Registry:
        @staticmethod
        def statuses():
            return []

    class _Service:
        narrator = None
        settings = SimpleNamespace(max_agent_steps=3)
        _state_locks = _Locks()
        tool_registry = _Registry()

        def respond_with_controller(self, **kwargs):
            from tbx_agent.agent_runtime import run_agent_turn

            return run_agent_turn(self, **kwargs)

    def spy_invoke(input_state, *, context, config):
        calls.append((input_state, context, config))
        return {"result": expected, "terminal_reason": "test"}

    monkeypatch.setattr(compiled_graph, "invoke", spy_invoke)
    graph = TBXAgentGraph(_Service())

    result = graph.invoke_with_receipt(
        {
            "thread_id": "thread-spy",
            "user_id": "user-spy",
            "owner_scope": "tenant:user-spy",
            "message": "你好",
            "case_id": None,
        }
    )

    assert result is expected
    assert len(calls) == 1
    assert calls[0][0]["query"] == "你好"
    assert calls[0][1].service is graph.service
    assert calls[0][2]["recursion_limit"] >= 20
