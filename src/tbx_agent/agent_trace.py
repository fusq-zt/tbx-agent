"""Public audit models for bounded Agent decisions and state transitions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .agent_state import AgentAction
from .controller import AgentBudget, ControllerDecision
from .task_spec import TaskSpec


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StateTransition(_StrictModel):
    step_index: int = Field(ge=0)
    action: AgentAction
    state_before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_name: str | None = Field(default=None, max_length=128)
    tool_status: str | None = Field(default=None, max_length=64)
    observation_code: str | None = Field(default=None, max_length=128)
    new_conflict_flags: list[str] = Field(default_factory=list, max_length=32)
    new_evidence_gaps: list[str] = Field(default_factory=list, max_length=32)
    predicted_class_unchanged: bool = True


class TerminalRecord(_StrictModel):
    action: AgentAction
    reason_code: str = Field(min_length=1, max_length=128)
    human_review_required: bool = False
    human_review_reason_codes: list[str] = Field(default_factory=list, max_length=32)


class AgentRunTrace(_StrictModel):
    trace_version: str = "tbx-agent-trace-v2"
    controller_policy_id: str
    task_spec: TaskSpec
    decisions: list[ControllerDecision] = Field(default_factory=list, max_length=8)
    state_transitions: list[StateTransition] = Field(default_factory=list, max_length=8)
    terminal: TerminalRecord
    budget: AgentBudget
    hidden_reasoning_persisted: bool = False

