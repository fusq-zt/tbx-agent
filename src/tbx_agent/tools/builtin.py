from __future__ import annotations

from collections.abc import Mapping

from ..schemas import ResponseKind
from .contracts import ToolCostTier, ToolInvocation, ToolName, ToolPermission
from .registry import ToolDefinition, ToolHandler, ToolHealthCheck, ToolRegistry

_AUDIT_ACTIONS: dict[ToolName, str] = {
    ToolName.EMERGENCY_TRIAGE: "emergency_escalation",
    ToolName.GET_EXACT_CASE_AND_EXPLAIN: "case_explanation",
    ToolName.CLASSIFY_CURRENT_CXR: "case_classification",
    ToolName.LOCALIZE_CURRENT_CXR: "case_localization",
    ToolName.INSPECT_ANATOMICAL_CONTEXT: "anatomical_context",
    ToolName.INSPECT_IMAGE_QUALITY: "image_quality_inspection",
    ToolName.COMPARE_WITH_PRIOR_CXR: "longitudinal_comparison",
    ToolName.SEARCH_TB_GUIDANCE: "guideline_retrieval",
    ToolName.DESCRIBE_AGENT_CAPABILITIES: "capability_response",
}

_TOOL_PERMISSIONS: dict[ToolName, ToolPermission] = {
    ToolName.EMERGENCY_TRIAGE: ToolPermission.PUBLIC_INFORMATION,
    ToolName.GET_EXACT_CASE_AND_EXPLAIN: ToolPermission.CASE_READ,
    ToolName.CLASSIFY_CURRENT_CXR: ToolPermission.CASE_COMPUTE,
    ToolName.LOCALIZE_CURRENT_CXR: ToolPermission.CASE_COMPUTE,
    ToolName.INSPECT_ANATOMICAL_CONTEXT: ToolPermission.CASE_COMPUTE,
    ToolName.INSPECT_IMAGE_QUALITY: ToolPermission.CASE_READ,
    ToolName.COMPARE_WITH_PRIOR_CXR: ToolPermission.CASE_READ,
    ToolName.SEARCH_TB_GUIDANCE: ToolPermission.PUBLIC_INFORMATION,
    ToolName.DESCRIBE_AGENT_CAPABILITIES: ToolPermission.PUBLIC_INFORMATION,
}

_ALLOWED_RESPONSE_KINDS: dict[ToolName, frozenset[ResponseKind]] = {
    ToolName.EMERGENCY_TRIAGE: frozenset({ResponseKind.SAFE_ABSTENTION}),
    ToolName.GET_EXACT_CASE_AND_EXPLAIN: frozenset(
        {ResponseKind.VISUAL_SCREENING_RESULT, ResponseKind.SAFE_ABSTENTION}
    ),
    ToolName.CLASSIFY_CURRENT_CXR: frozenset(
        {ResponseKind.VISUAL_SCREENING_RESULT, ResponseKind.SAFE_ABSTENTION}
    ),
    ToolName.LOCALIZE_CURRENT_CXR: frozenset(
        {
            ResponseKind.VISUAL_SCREENING_RESULT,
            ResponseKind.LOCALIZATION_RESULT,
            ResponseKind.SAFE_ABSTENTION,
        }
    ),
    ToolName.INSPECT_ANATOMICAL_CONTEXT: frozenset(
        {ResponseKind.CASE_EXPLANATION, ResponseKind.SAFE_ABSTENTION}
    ),
    ToolName.INSPECT_IMAGE_QUALITY: frozenset(
        {ResponseKind.CASE_EXPLANATION, ResponseKind.SAFE_ABSTENTION}
    ),
    ToolName.COMPARE_WITH_PRIOR_CXR: frozenset({ResponseKind.SAFE_ABSTENTION}),
    ToolName.SEARCH_TB_GUIDANCE: frozenset(
        {
            ResponseKind.DIAGNOSTIC_INFORMATION,
            ResponseKind.NEXT_TEST_INFORMATION,
            ResponseKind.TREATMENT_EDUCATION,
            ResponseKind.SAFE_ABSTENTION,
        }
    ),
    ToolName.DESCRIBE_AGENT_CAPABILITIES: frozenset({ResponseKind.SAFE_ABSTENTION}),
}

_TOOL_COSTS: dict[ToolName, tuple[ToolCostTier, int, bool]] = {
    ToolName.EMERGENCY_TRIAGE: (ToolCostTier.LOW, 0, False),
    ToolName.GET_EXACT_CASE_AND_EXPLAIN: (ToolCostTier.LOW, 1, False),
    ToolName.CLASSIFY_CURRENT_CXR: (ToolCostTier.MEDIUM, 2, True),
    ToolName.LOCALIZE_CURRENT_CXR: (ToolCostTier.MEDIUM, 3, True),
    ToolName.INSPECT_ANATOMICAL_CONTEXT: (ToolCostTier.HIGH, 3, True),
    ToolName.INSPECT_IMAGE_QUALITY: (ToolCostTier.LOW, 1, False),
    ToolName.COMPARE_WITH_PRIOR_CXR: (ToolCostTier.LOW, 1, False),
    ToolName.SEARCH_TB_GUIDANCE: (ToolCostTier.MEDIUM, 2, False),
    ToolName.DESCRIBE_AGENT_CAPABILITIES: (ToolCostTier.LOW, 1, False),
}


def build_builtin_registry(
    handlers: Mapping[ToolName, ToolHandler],
    *,
    timeout_seconds: float,
    max_steps: int,
    health_checks: Mapping[ToolName, ToolHealthCheck] | None = None,
) -> ToolRegistry:
    """Build the fixed medical-tool allowlist used by the deterministic router."""

    missing = set(ToolName) - set(handlers)
    extra = set(handlers) - set(ToolName)
    if missing or extra:
        raise ValueError(
            f"builtin handler set mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    resolved_health_checks = dict(health_checks or {})
    unknown_health_checks = set(resolved_health_checks) - set(ToolName)
    if unknown_health_checks:
        raise ValueError(
            f"builtin health-check set contains unknown tools: {sorted(unknown_health_checks)}"
        )
    registry = ToolRegistry(max_steps=max_steps)
    for tool_name in ToolName:
        handler = handlers[tool_name]

        def bound_handler(
            invocation: ToolInvocation,
            *,
            _handler: ToolHandler = handler,
        ):
            return _handler(invocation)

        registry.register(
            ToolDefinition(
                name=tool_name.value,
                audit_action=_AUDIT_ACTIONS[tool_name],
                handler=bound_handler,
                timeout_seconds=timeout_seconds,
                permission=_TOOL_PERMISSIONS[tool_name],
                allowed_response_kinds=_ALLOWED_RESPONSE_KINDS[tool_name],
                requires_case=tool_name
                in {
                    ToolName.GET_EXACT_CASE_AND_EXPLAIN,
                    ToolName.CLASSIFY_CURRENT_CXR,
                    ToolName.LOCALIZE_CURRENT_CXR,
                    ToolName.INSPECT_ANATOMICAL_CONTEXT,
                    ToolName.INSPECT_IMAGE_QUALITY,
                    ToolName.COMPARE_WITH_PRIOR_CXR,
                },
                state_mutation_allowed=tool_name
                in {
                    ToolName.CLASSIFY_CURRENT_CXR,
                    ToolName.LOCALIZE_CURRENT_CXR,
                    ToolName.INSPECT_ANATOMICAL_CONTEXT,
                },
                cost_tier=_TOOL_COSTS[tool_name][0],
                cost_units=_TOOL_COSTS[tool_name][1],
                expensive_vision=_TOOL_COSTS[tool_name][2],
                health_check=resolved_health_checks.get(tool_name),
            )
        )
    return registry
