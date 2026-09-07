from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..schemas import AgentResponse, ResponseKind
from ..task_spec import GuidelineScenarioTag, GuidelineScope

TOOL_CONTRACT_VERSION = "tbx-tool-contract-v7"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _StrictToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ToolCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    SATURATED = "saturated"
    REJECTED = "rejected"
    STEP_LIMIT_EXCEEDED = "step_limit_exceeded"


class ToolOutcome(StrEnum):
    ANSWER_READY = "answer_ready"
    CASE_EVIDENCE_READY = "case_evidence_ready"
    EVIDENCE_GAP = "evidence_gap"
    EMERGENCY_HANDOFF = "emergency_handoff"
    CAPABILITY_ONLY = "capability_only"
    EXECUTION_DEGRADED = "execution_degraded"


class ToolErrorCategory(StrEnum):
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    DEPENDENCY = "dependency"
    CONTRACT = "contract"
    CAPACITY = "capacity"
    POLICY = "policy"


class ToolName(StrEnum):
    EMERGENCY_TRIAGE = "emergency_triage"
    GET_EXACT_CASE_AND_EXPLAIN = "get_exact_case_and_explain"
    CLASSIFY_CURRENT_CXR = "classify_current_cxr"
    LOCALIZE_CURRENT_CXR = "localize_current_cxr"
    INSPECT_ANATOMICAL_CONTEXT = "inspect_anatomical_context"
    INSPECT_IMAGE_QUALITY = "inspect_image_quality"
    COMPARE_WITH_PRIOR_CXR = "compare_with_prior_cxr"
    SEARCH_TB_KNOWLEDGE = "search_tb_knowledge"
    SEARCH_TB_GUIDANCE = "search_tb_knowledge"
    # Source compatibility for Python callers while enum iteration and new
    # receipts expose only the query-first public tool name.
    RETRIEVE_GUIDELINE = "search_tb_knowledge"
    DESCRIBE_AGENT_CAPABILITIES = "describe_agent_capabilities"

    @classmethod
    def _missing_(cls, value: object):
        if value in {"retrieve_guideline", "search_tb_guidance"}:
            return cls.SEARCH_TB_KNOWLEDGE
        return None


class ToolAvailability(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ToolPermission(StrEnum):
    """Narrow data capability granted to a deterministic tool handler."""

    PUBLIC_INFORMATION = "public_information"
    CASE_READ = "case_read"
    CASE_COMPUTE = "case_compute"


class ToolCostTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolUnavailableError(RuntimeError):
    """A configured tool cannot run because its dependency is unavailable."""


class ToolInvocation(_StrictToolModel):
    """Internal invocation contract. It is never populated from LLM output."""

    tool_name: str = Field(min_length=1, max_length=128)
    # Public Plan/ReAct capability that proposed this internal handler.  Older
    # direct API calls legitimately leave it unset.
    model_tool_name: str | None = Field(default=None, min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=256)
    user_id: str = Field(min_length=1, max_length=128)
    owner_scope: str = Field(min_length=1, max_length=256)
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    routing_policy_id: str = Field(min_length=1, max_length=128)
    case_id: str | None = Field(default=None, min_length=1, max_length=256)
    guideline_scope: GuidelineScope | None = None
    subtopic: str | None = Field(default=None, min_length=1, max_length=128)
    population: list[str] = Field(default_factory=list, max_length=16)
    product_terms: list[str] = Field(default_factory=list, max_length=16)
    scenario_tags: list[GuidelineScenarioTag] = Field(default_factory=list, max_length=16)
    step_index: int = Field(default=0, ge=0)
    max_steps: int = Field(default=1, ge=1, le=8)
    safety_policy_id: str = Field(default="tbx-agent-safety-v1", min_length=1, max_length=128)
    plan_id: str | None = Field(default=None, min_length=1, max_length=128)
    step_id: str | None = Field(default=None, min_length=1, max_length=32)
    selection_source: str = Field(default="rule_fallback", max_length=64)
    attempt: int = Field(default=1, ge=1, le=2)
    max_attempts: int = Field(default=1, ge=1, le=2)
    idempotency_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator("tool_name", mode="before")
    @classmethod
    def _canonicalize_legacy_tool_name(cls, value: object) -> object:
        if value in {"retrieve_guideline", "search_tb_guidance"}:
            return ToolName.SEARCH_TB_KNOWLEDGE.value
        return value

    @model_validator(mode="after")
    def _validate_guideline_dimensions(self) -> ToolInvocation:
        is_guideline = self.tool_name == ToolName.SEARCH_TB_GUIDANCE.value
        # Query-first calls deliberately omit every retrieval dimension.  The
        # same contract still accepts dimensions for direct legacy adapters;
        # the public search handler resolves and records its own dimensions.
        if is_guideline and self.guideline_scope is None and any(
            (
                self.subtopic is not None,
                bool(self.population),
                bool(self.product_terms),
                bool(self.scenario_tags),
            )
        ):
            raise ValueError("guideline dimensions require guideline_scope")
        if not is_guideline and any(
            (
                self.guideline_scope is not None,
                self.subtopic is not None,
                bool(self.population),
                bool(self.product_terms),
                bool(self.scenario_tags),
            )
        ):
            raise ValueError("guideline retrieval dimensions require retrieve_guideline")
        return self


class ToolReceipt(_StrictToolModel):
    """Non-sensitive execution provenance suitable for the audit log."""

    call_id: str
    request_id: str
    trace_id: str
    tool_name: str
    model_tool_name: str | None = Field(default=None, min_length=1, max_length=128)
    routing_policy_id: str
    status: ToolCallStatus
    started_at: datetime = Field(default_factory=_utc_now)
    finished_at: datetime = Field(default_factory=_utc_now)
    runtime_ms: int = Field(ge=0)
    timeout_ms: int = Field(gt=0)
    step_index: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    plan_id: str | None = None
    step_id: str | None = None
    selection_source: str = "rule_fallback"
    attempt: int = Field(default=1, ge=1, le=2)
    max_attempts: int = Field(default=1, ge=1, le=2)
    idempotency_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str | None = None
    response_kind: ResponseKind
    citation_count: int = Field(ge=0)
    fallback_used: bool = False
    error_code: str | None = None
    error_type: str | None = None
    error_category: ToolErrorCategory | None = None
    retryable: bool = False
    outcome: ToolOutcome
    tool_contract_version: str = TOOL_CONTRACT_VERSION
    permission: ToolPermission
    requires_case: bool
    output_contract_validated: bool = True
    deterministic_router: bool = True
    state_mutation_allowed: bool = False
    cost_tier: ToolCostTier = ToolCostTier.LOW
    cost_units: int = Field(default=1, ge=0, le=10)
    expensive_vision: bool = False
    observation_code: str | None = Field(default=None, max_length=128)
    resolved_guideline_scope: GuidelineScope | None = None
    resolved_guideline_subtopic: str | None = Field(default=None, max_length=128)
    resolved_population: list[str] = Field(default_factory=list, max_length=16)
    resolved_product_terms: list[str] = Field(default_factory=list, max_length=16)
    resolved_scenario_tags: list[GuidelineScenarioTag] = Field(
        default_factory=list,
        max_length=16,
    )
    medical_decision_authority: bool = False
    visual_policy_mutation_allowed: bool = False
    medical_route_mutation_allowed: bool = False


class ToolResult(_StrictToolModel):
    response: AgentResponse
    receipt: ToolReceipt
    audit_action: str = Field(min_length=1, max_length=128)


class ToolStatus(_StrictToolModel):
    name: str
    availability: ToolAvailability
    timeout_ms: int = Field(gt=0)
    max_steps: int = Field(ge=1)
    permission: ToolPermission
    requires_case: bool
    allowed_response_kinds: list[ResponseKind]
    tool_contract_version: str = TOOL_CONTRACT_VERSION
    deterministic_router: bool = True
    state_mutation_allowed: bool = False
    cost_tier: ToolCostTier = ToolCostTier.LOW
    cost_units: int = Field(default=1, ge=0, le=10)
    expensive_vision: bool = False
    medical_decision_authority: bool = False
    visual_policy_mutation_allowed: bool = False
    medical_route_mutation_allowed: bool = False
    detail_code: str | None = None
    last_call_status: ToolCallStatus | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    last_runtime_ms: int | None = Field(default=None, ge=0)
