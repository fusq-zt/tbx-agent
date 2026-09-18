"""LangGraph execution topology for the TBX Plan + ReAct runtime.

This module owns the orchestration state machine.  Domain operations stay in
``agent_runtime`` so they can continue to use TBX's audited ``ToolRegistry``
and ``ToolReceipt`` contracts, but iteration and control flow are executed by
an actual :class:`langgraph.graph.StateGraph`.

Only public plan objectives, selected actions and bounded observations enter
the graph state.  There is deliberately no field for chain-of-thought or other
private reasoning text.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .llm.tool_calling import (
    HighLevelToolName,
    HighLevelToolSelection,
)
from .plan_react import (
    PLAN_REACT_POLICY_ID,
    PlanRevisionRecord,
    ReActStepRecord,
    TurnPlan,
)

LANGGRAPH_NODE_NAMES = (
    "load_context",
    "plan",
    "decide",
    "execute_tool",
    "observe",
    "replan",
    "finalize",
)


def plan_react_provenance() -> dict[str, object]:
    """Return the public contract of the production orchestration path.

    This intentionally describes only facts that callers can verify without
    exposing prompts, provider responses, graph state or private reasoning.
    ``controller_policy_id`` remains a compatibility alias for clients that
    consumed the original manifest; it now resolves to the active Plan + ReAct
    policy rather than the legacy deterministic controller.
    """

    return {
        "framework": "langgraph",
        "policy_id": PLAN_REACT_POLICY_ID,
        "controller_policy_id": PLAN_REACT_POLICY_ID,
        "strategy": "react_first_optional_plan",
        "graph_nodes": list(LANGGRAPH_NODE_NAMES),
        "model_visible_tools": [item.value for item in HighLevelToolName],
        "tool_selection_priority": [
            "structured_react_decision",
        ],
        "tool_calling": {
            "preferred": "strict_json_schema_decision",
            "fallback": "rule_plan_after_model_error",
        },
        "action_granularity": "one_tool_per_react_decision",
        # Retain the old key with its literal meaning for manifest consumers.
        # A fresh decision is not a model call to regenerate the whole plan.
        "replans_after_each_observation": False,
        "decides_after_each_observation": True,
        "replan_policy": {
            "trigger": "explicit_failure_or_missing_obligation",
            "max_plan_revisions": 2,
        },
        "trusted_projection_before_model_planning": False,
        "intent_authority": "model_first_rule_fallback",
        "planning_policy": "model_requested_complex_tasks_only",
        "bounded": True,
        "hidden_reasoning_persisted": False,
        "durable_langgraph_checkpointer": False,
        "business_state_authority": "SQLiteStore",
    }


class GraphRoute(StrEnum):
    """Explicit next-node choices accepted by the compiled workflow."""

    PLAN = "plan"
    DECIDE = "decide"
    EXECUTE_TOOL = "execute_tool"
    REPLAN = "replan"
    FINALIZE = "finalize"


class PlanReActGraphInput(TypedDict):
    """Minimal typed input accepted by the graph."""

    query: str
    max_react_iterations: int


class PlanReActGraphOutput(TypedDict, total=False):
    """Stable graph output; ``result`` is the public ``AgentTurnResult``."""

    result: Any
    terminal_reason: str
    visited_nodes: list[str]


class PlanReActGraphState(PlanReActGraphInput, total=False):
    """Typed, bounded state shared by all Plan + ReAct graph nodes."""

    context_loaded: bool
    react_first: bool
    rule_fallback: bool
    pending_plan_tasks: list[Any] | None
    decision_feedback: list[dict[str, Any]]
    answer_evidence: list[Any]
    answer_focus: str
    resolved_response: Any
    resolved_cached_evidence: list[str]
    decision_usage: list[dict[str, Any]]
    request_id: str
    trace_id: str
    run_id: str
    effective_case_id: str | None
    recent_dialogue: list[dict[str, str]]
    case_context: dict[str, Any]
    plan: TurnPlan
    initial_plan: TurnPlan
    plan_metadata: dict[str, Any]
    plan_revisions: list[PlanRevisionRecord]
    react_steps: list[ReActStepRecord]
    observations: list[dict[str, Any]]
    tool_results: list[Any]
    attempted_tools: set[Any]
    budget: Any
    pending_selection: HighLevelToolSelection | None
    pending_invocation: Any
    pending_public_tool: Any
    pending_cost_units: int
    pending_expensive: bool
    pending_recovered: bool
    pending_tool_result: Any
    replan_trigger: str | None
    replan_reason_code: str | None
    final_direct_answer: str | None
    final_direct_usage: tuple[int | None, int | None]
    direct_response_override: Any
    terminal_reason: str
    reflection_triggered: bool
    next_node: GraphRoute
    react_iteration: int
    visited_nodes: list[str]
    result: Any


NodeUpdate = Mapping[str, Any]
NodeHandler = Callable[[PlanReActGraphState, "PlanReActGraphContext"], NodeUpdate]


@dataclass(frozen=True, slots=True)
class PlanReActRuntimeOps:
    """Run-scoped domain callbacks consumed by the reusable StateGraph.

    The callbacks never choose edges directly outside the bounded
    :class:`GraphRoute` contract.  This keeps the graph topology inspectable
    while allowing the TBX domain layer to retain its existing safety, budget,
    registry and persistence logic.
    """

    load_context: NodeHandler
    plan: NodeHandler
    decide: NodeHandler
    execute_tool: NodeHandler
    observe: NodeHandler
    replan: NodeHandler
    finalize: NodeHandler


@dataclass(frozen=True, slots=True)
class PlanReActGraphContext:
    """Immutable run dependencies kept outside the persistable graph state."""

    ops: PlanReActRuntimeOps
    service: Any
    thread_id: str
    user_id: str
    owner_scope: str
    case_id: str | None
    generator: Any | None
    narrator_override: Any | None


_STATE_KEYS = frozenset(PlanReActGraphState.__annotations__)


def _record_visit(state: PlanReActGraphState, node: str) -> list[str]:
    return [*state.get("visited_nodes", []), node]


def _invoke_domain_node(
    node: str,
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    handler = getattr(runtime.context.ops, node)
    update = dict(handler(state, runtime.context))
    unknown = set(update) - _STATE_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"LangGraph node {node!r} returned unknown state fields: {names}")
    update["visited_nodes"] = _record_visit(state, node)
    return update


def _load_context(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    update = _invoke_domain_node("load_context", state, runtime)
    update.setdefault("context_loaded", True)
    update.setdefault("next_node", GraphRoute.DECIDE)
    update.setdefault("react_iteration", 0)
    return update


def _plan(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    update = _invoke_domain_node("plan", state, runtime)
    update.setdefault("next_node", GraphRoute.DECIDE)
    return update


def _decide(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    iteration = state.get("react_iteration", 0)
    maximum = state.get("max_react_iterations", 1)
    if maximum < 1:
        raise ValueError("max_react_iterations must be at least 1")
    if iteration >= maximum:
        return {
            "terminal_reason": "react_iteration_limit_reached",
            "next_node": GraphRoute.FINALIZE,
            "visited_nodes": _record_visit(state, "decide"),
        }

    update = _invoke_domain_node("decide", state, runtime)
    update["react_iteration"] = iteration + 1
    selection = update.get("pending_selection")
    route = GraphRoute(update.get("next_node", GraphRoute.FINALIZE))

    if route == GraphRoute.EXECUTE_TOOL:
        if not isinstance(selection, HighLevelToolSelection):
            raise TypeError("execute_tool route requires one validated HighLevelToolSelection")
        if selection.tool_call is None or selection.direct_answer is not None:
            raise ValueError("a ReAct iteration may execute exactly one tool call")
    elif isinstance(selection, HighLevelToolSelection) and selection.tool_call is not None:
        raise ValueError("a selected tool call must route to execute_tool")
    update["next_node"] = route
    return update


def _execute_tool(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    selection = state.get("pending_selection")
    if not isinstance(selection, HighLevelToolSelection) or selection.tool_call is None:
        raise ValueError("execute_tool requires the current iteration's single tool call")
    update = _invoke_domain_node("execute_tool", state, runtime)
    # Observation is a separate node so a raw tool result is never mistaken
    # for a final answer or silently skipped by an edge callback.
    update["next_node"] = GraphRoute.DECIDE
    return update


def _observe(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    update = _invoke_domain_node("observe", state, runtime)
    update.setdefault("next_node", GraphRoute.DECIDE)
    update["pending_selection"] = None
    return update


def _replan(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    update = _invoke_domain_node("replan", state, runtime)
    update.setdefault("next_node", GraphRoute.DECIDE)
    update["pending_selection"] = None
    return update


def _finalize(
    state: PlanReActGraphState,
    runtime: Runtime[PlanReActGraphContext],
) -> dict[str, Any]:
    return _invoke_domain_node("finalize", state, runtime)


def _validated_route(
    state: PlanReActGraphState,
    *,
    allowed: frozenset[GraphRoute],
    default: GraphRoute,
) -> str:
    route = GraphRoute(state.get("next_node", default))
    if route not in allowed:
        allowed_text = ", ".join(sorted(item.value for item in allowed))
        raise ValueError(f"invalid graph route {route.value!r}; expected one of {allowed_text}")
    return route.value


def _after_load(state: PlanReActGraphState) -> str:
    return _validated_route(
        state,
        allowed=frozenset({GraphRoute.PLAN, GraphRoute.DECIDE, GraphRoute.FINALIZE}),
        default=GraphRoute.DECIDE,
    )


def _after_plan(state: PlanReActGraphState) -> str:
    return _validated_route(
        state,
        allowed=frozenset({GraphRoute.DECIDE, GraphRoute.FINALIZE}),
        default=GraphRoute.DECIDE,
    )


def _after_decide(state: PlanReActGraphState) -> str:
    return _validated_route(
        state,
        allowed=frozenset(
            {GraphRoute.EXECUTE_TOOL, GraphRoute.PLAN, GraphRoute.DECIDE,
             GraphRoute.REPLAN, GraphRoute.FINALIZE}
        ),
        default=GraphRoute.FINALIZE,
    )


def _after_observe(state: PlanReActGraphState) -> str:
    return _validated_route(
        state,
        allowed=frozenset({GraphRoute.DECIDE, GraphRoute.REPLAN, GraphRoute.FINALIZE}),
        default=GraphRoute.DECIDE,
    )


def _after_replan(state: PlanReActGraphState) -> str:
    return _validated_route(
        state,
        allowed=frozenset({GraphRoute.DECIDE, GraphRoute.FINALIZE}),
        default=GraphRoute.DECIDE,
    )


def build_plan_react_workflow() -> StateGraph:
    """Build the public, inspectable Plan + ReAct StateGraph topology."""

    builder = StateGraph(
        state_schema=PlanReActGraphState,
        context_schema=PlanReActGraphContext,
        input_schema=PlanReActGraphInput,
        output_schema=PlanReActGraphOutput,
    )
    builder.add_node(LANGGRAPH_NODE_NAMES[0], _load_context)
    builder.add_node(LANGGRAPH_NODE_NAMES[1], _plan)
    builder.add_node(LANGGRAPH_NODE_NAMES[2], _decide)
    builder.add_node(LANGGRAPH_NODE_NAMES[3], _execute_tool)
    builder.add_node(LANGGRAPH_NODE_NAMES[4], _observe)
    builder.add_node(LANGGRAPH_NODE_NAMES[5], _replan)
    builder.add_node(LANGGRAPH_NODE_NAMES[6], _finalize)

    builder.add_edge(START, "load_context")
    builder.add_conditional_edges(
        "load_context",
        _after_load,
        {GraphRoute.PLAN.value: "plan", GraphRoute.DECIDE.value: "decide",
         GraphRoute.FINALIZE.value: "finalize"},
    )
    builder.add_conditional_edges(
        "plan",
        _after_plan,
        {GraphRoute.DECIDE.value: "decide", GraphRoute.FINALIZE.value: "finalize"},
    )
    builder.add_conditional_edges(
        "decide",
        _after_decide,
        {
            GraphRoute.PLAN.value: "plan",
            GraphRoute.DECIDE.value: "decide",
            GraphRoute.EXECUTE_TOOL.value: "execute_tool",
            GraphRoute.REPLAN.value: "replan",
            GraphRoute.FINALIZE.value: "finalize",
        },
    )
    builder.add_edge("execute_tool", "observe")
    builder.add_conditional_edges(
        "observe",
        _after_observe,
        {
            GraphRoute.DECIDE.value: "decide",
            GraphRoute.REPLAN.value: "replan",
            GraphRoute.FINALIZE.value: "finalize",
        },
    )
    builder.add_conditional_edges(
        "replan",
        _after_replan,
        {GraphRoute.DECIDE.value: "decide", GraphRoute.FINALIZE.value: "finalize"},
    )
    builder.add_edge("finalize", END)
    return builder


# Export both the builder and the production compiled graph so topology tests
# can assert real LangGraph nodes/edges without invoking a model or tool.
workflow = build_plan_react_workflow()
compiled_graph = workflow.compile(name="tbx-plan-react")


def invoke_plan_react_graph(
    *,
    query: str,
    context: PlanReActGraphContext,
    max_react_iterations: int,
) -> Any:
    """Invoke the production compiled StateGraph and return AgentTurnResult."""

    if max_react_iterations < 1:
        raise ValueError("max_react_iterations must be at least 1")
    output = compiled_graph.invoke(
        {
            "query": query,
            "max_react_iterations": max_react_iterations,
        },
        context=context,
        config={"recursion_limit": max(20, max_react_iterations * 5 + 10)},
    )
    if "result" not in output:
        raise RuntimeError("LangGraph finalized without an AgentTurnResult")
    return output["result"]


__all__ = [
    "GraphRoute",
    "LANGGRAPH_NODE_NAMES",
    "PlanReActGraphContext",
    "PlanReActGraphInput",
    "PlanReActGraphOutput",
    "PlanReActGraphState",
    "PlanReActRuntimeOps",
    "build_plan_react_workflow",
    "compiled_graph",
    "invoke_plan_react_graph",
    "plan_react_provenance",
    "workflow",
]
