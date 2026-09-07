"""Bounded, state-aware, one-action controller.

The controller never receives pixels, credentials, full guideline text, or raw
conversation history.  It sees a public projection of TaskSpec and CaseState,
then selects one enum-valued action.  Deterministic guards validate that action;
they do not run a second intent router.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_state import (
    AgentAction,
    CaseState,
    ConflictFlag,
    EvidenceKind,
    EvidenceStatus,
)
from .task_spec import TaskGoal, TaskSpec

CONTROLLER_POLICY_ID = "tbx-state-aware-controller-v1"


def controller_provenance() -> dict[str, object]:
    """Return public orchestration metadata without prompts or hidden state."""

    return {
        "controller_policy_id": CONTROLLER_POLICY_ID,
        "strategy": "task_spec_case_state_observe_replan",
        "action_granularity": "one_tool_per_decision",
        "typed_action_allowlist": True,
        "replans_after_each_observation": True,
        "bounded": True,
        "free_form_reflection": False,
        "hidden_reasoning_persisted": False,
    }


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DecisionSource(StrEnum):
    LLM_CONTROLLER = "llm_controller"
    STATE_POLICY_FALLBACK = "state_policy_fallback"
    HARD_GUARD = "hard_guard"
    POST_ACTION_CHECK = "post_action_check"


class SelectionReasonCode(StrEnum):
    CLASSIFICATION_REQUIRED = "classification_required"
    LOCALIZATION_REQUIRED = "localization_required"
    ANATOMY_REQUIRED = "anatomy_required"
    QUALITY_EVIDENCE_REQUIRED = "quality_evidence_required"
    PRIOR_LOOKUP_REQUIRED = "prior_lookup_required"
    DIAGNOSTIC_GUIDELINE_REQUIRED = "diagnostic_guideline_required"
    TREATMENT_GUIDELINE_REQUIRED = "treatment_guideline_required"
    TASK_ALREADY_COMPLETE = "task_already_complete"
    NO_TOOL_TASK = "no_tool_task"
    EVIDENCE_CONFLICT = "evidence_conflict"
    TECHNICAL_FAILURE = "technical_failure"
    REQUIRED_CAPABILITY_UNAVAILABLE = "required_capability_unavailable"
    PRIOR_EVIDENCE_UNAVAILABLE = "prior_evidence_unavailable"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOOL_UNAVAILABLE = "tool_unavailable"
    ACTION_ALREADY_ATTEMPTED = "action_already_attempted"
    CONTROLLER_OUTPUT_REJECTED = "controller_output_rejected"
    EMERGENCY_GUARD = "emergency_guard"
    NO_SAFE_ACTION = "no_safe_action"


class AgentBudget(_StrictModel):
    max_steps: int = Field(default=5, ge=1, le=8)
    max_tool_calls: int = Field(default=4, ge=0, le=8)
    max_expensive_vision_calls: int = Field(default=3, ge=0, le=4)
    max_cost_units: int = Field(default=10, ge=0, le=30)
    steps_used: int = Field(default=0, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    expensive_vision_calls_used: int = Field(default=0, ge=0)
    cost_units_used: int = Field(default=0, ge=0)

    @property
    def remaining_steps(self) -> int:
        return max(0, self.max_steps - self.steps_used)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls_used)

    @property
    def remaining_expensive_vision_calls(self) -> int:
        return max(
            0,
            self.max_expensive_vision_calls - self.expensive_vision_calls_used,
        )

    @property
    def remaining_cost_units(self) -> int:
        return max(0, self.max_cost_units - self.cost_units_used)


class ActionCapability(_StrictModel):
    action: AgentAction
    available: bool = True
    cost_units: int = Field(default=1, ge=0, le=10)
    expensive_vision: bool = False
    detail_code: str | None = Field(default=None, max_length=128)


class ControllerDraft(_StrictModel):
    action: AgentAction
    reason_code: SelectionReasonCode


class ControllerDecision(_StrictModel):
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = CONTROLLER_POLICY_ID
    step_index: int = Field(ge=0)
    action: AgentAction
    reason_code: SelectionReasonCode
    source: DecisionSource
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    allowed_actions: list[AgentAction]
    cost_units: int = Field(default=0, ge=0, le=10)
    expensive_vision: bool = False
    schema_validated: bool = True
    controller_backend: str | None = None
    controller_model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=1)
    completion_tokens: int | None = Field(default=None, ge=1)
    hidden_reasoning_persisted: bool = False


class StructuredControllerGenerator(Protocol):
    backend_id: str
    model: str

    def complete_structured(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
        seed: int,
    ) -> tuple[str, dict[str, int]]: ...


_SATISFIED = {
    EvidenceStatus.AVAILABLE,
    EvidenceStatus.COMPLETED,
    EvidenceStatus.COMPLETED_NO_DETECTION,
}


def _sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def controller_state_view(state: CaseState, task_spec: TaskSpec) -> dict[str, Any]:
    """Return the only CaseState projection an LLM controller may inspect."""

    return {
        "task": {
            "goals": [goal.value for goal in task_spec.task_goals],
            "required_evidence": [item.value for item in task_spec.required_evidence],
            "guideline_scope": (
                task_spec.guideline_scope.value
                if task_spec.guideline_scope is not None
                else None
            ),
            "subtopic": task_spec.subtopic,
            "population": task_spec.population,
            "product_terms": task_spec.product_terms,
            "scenario_tags": [item.value for item in task_spec.scenario_tags],
            "goal_evidence": [
                item.model_dump(mode="json") for item in task_spec.goal_evidence
            ],
            "completion_criteria": task_spec.completion_criteria,
            "forbidden_claims": task_spec.forbidden_claims,
        },
        "case_available": state.case_id is not None,
        "evidence": {
            "classification": state.classification_evidence.status.value,
            "quality": state.quality_evidence.status.value,
            "localization": state.localization_evidence.status.value,
            "localization_item_count": state.localization_evidence.item_count,
            "anatomy": state.anatomical_evidence.status.value,
            "diagnostic_guideline": state.diagnostic_guideline_evidence.status.value,
            "treatment_guideline": state.treatment_guideline_evidence.status.value,
            "prior": state.prior_evidence.status.value,
            "longitudinal": state.longitudinal_evidence.status.value,
        },
        "classification": {
            "predicted_class": state.predicted_class,
            "top1_class": state.top1_class,
            "top1_score": state.top1_score,
            "top2_class": state.top2_class,
            "top2_score": state.top2_score,
            "top1_top2_margin": state.top1_top2_margin,
            "calibration_status": state.calibration_status.value,
        },
        "conflict_flags": [item.value for item in state.conflict_flags],
        "evidence_gaps": [item.value for item in state.evidence_gaps],
        "completed_actions": [item.value for item in state.completed_actions],
        "failed_actions": [item.value for item in state.failed_actions],
        "screening_disposition": state.screening_disposition.value,
    }


def _slot_for_requirement(state: CaseState, kind: EvidenceKind):
    return {
        EvidenceKind.CLASSIFICATION: state.classification_evidence,
        EvidenceKind.LOCALIZATION: state.localization_evidence,
        EvidenceKind.QUALITY: state.quality_evidence,
        EvidenceKind.DIAGNOSTIC: state.diagnostic_guideline_evidence,
        EvidenceKind.TREATMENT: state.treatment_guideline_evidence,
        EvidenceKind.PRIOR: state.prior_evidence,
        EvidenceKind.LONGITUDINAL: state.longitudinal_evidence,
        EvidenceKind.ANATOMY: state.anatomical_evidence,
    }[kind]


def _required_next_action(state: CaseState, task_spec: TaskSpec) -> ControllerDraft:
    if task_spec.no_tool_only:
        return ControllerDraft(
            action=AgentAction.STOP,
            reason_code=SelectionReasonCode.NO_TOOL_TASK,
        )

    deferred_terminal: ControllerDraft | None = None

    def defer_terminal(candidate: ControllerDraft) -> None:
        nonlocal deferred_terminal
        if (
            deferred_terminal is None
            or candidate.action == AgentAction.REFER_TO_HUMAN
            and deferred_terminal.action != AgentAction.REFER_TO_HUMAN
        ):
            deferred_terminal = candidate

    action_by_evidence = {
        EvidenceKind.CLASSIFICATION: AgentAction.CLASSIFY_CURRENT_CXR,
        EvidenceKind.LOCALIZATION: AgentAction.LOCALIZE_CURRENT_CXR,
        EvidenceKind.ANATOMY: AgentAction.INSPECT_ANATOMICAL_CONTEXT,
        EvidenceKind.QUALITY: AgentAction.INSPECT_IMAGE_QUALITY,
        EvidenceKind.PRIOR: AgentAction.RETRIEVE_PRIOR_STUDIES,
        EvidenceKind.LONGITUDINAL: AgentAction.RETRIEVE_PRIOR_STUDIES,
        EvidenceKind.DIAGNOSTIC: AgentAction.RETRIEVE_GUIDELINE,
        EvidenceKind.TREATMENT: AgentAction.RETRIEVE_GUIDELINE,
    }
    for kind in task_spec.required_evidence:
        slot = _slot_for_requirement(state, kind)
        action = action_by_evidence[kind]
        if slot.status in _SATISFIED and action in state.completed_actions:
            continue
        # Persisted evidence is a cache, not an observation from this turn.
        # Invoke the selected high-level tool once so its handler can reuse the
        # cache while still producing a real receipt for the agent trajectory.
        if action in state.failed_actions:
            defer_terminal(
                ControllerDraft(
                    action=AgentAction.REFER_TO_HUMAN,
                    reason_code=SelectionReasonCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                )
            )
            continue
        if kind == EvidenceKind.PRIOR and slot.status == EvidenceStatus.EVIDENCE_GAP:
            defer_terminal(
                ControllerDraft(
                    action=AgentAction.STOP,
                    reason_code=SelectionReasonCode.PRIOR_EVIDENCE_UNAVAILABLE,
                )
            )
            continue
        if kind == EvidenceKind.LONGITUDINAL and slot.status in {
            EvidenceStatus.EVIDENCE_GAP,
            EvidenceStatus.UNSUPPORTED,
        }:
            defer_terminal(
                ControllerDraft(
                    action=AgentAction.STOP,
                    reason_code=SelectionReasonCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                )
            )
            continue
        if kind in {EvidenceKind.DIAGNOSTIC, EvidenceKind.TREATMENT} and (
            slot.status == EvidenceStatus.EVIDENCE_GAP
        ):
            defer_terminal(
                ControllerDraft(
                    action=AgentAction.STOP,
                    reason_code=SelectionReasonCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                )
            )
            continue
        if slot.status in {EvidenceStatus.FAILED, EvidenceStatus.UNSUPPORTED}:
            defer_terminal(
                ControllerDraft(
                    action=AgentAction.REFER_TO_HUMAN,
                    reason_code=SelectionReasonCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                )
            )
            continue
        if kind == EvidenceKind.CLASSIFICATION:
            return ControllerDraft(
                action=AgentAction.CLASSIFY_CURRENT_CXR,
                reason_code=SelectionReasonCode.CLASSIFICATION_REQUIRED,
            )
        if kind == EvidenceKind.LOCALIZATION:
            return ControllerDraft(
                action=AgentAction.LOCALIZE_CURRENT_CXR,
                reason_code=SelectionReasonCode.LOCALIZATION_REQUIRED,
            )
        if kind == EvidenceKind.ANATOMY:
            if (
                TaskGoal.ANATOMICAL_CONTEXT in task_spec.task_goals
                and state.localization_evidence.status not in _SATISFIED
            ):
                defer_terminal(
                    ControllerDraft(
                        action=(
                            AgentAction.REFER_TO_HUMAN
                            if state.localization_evidence.status
                            in {EvidenceStatus.FAILED, EvidenceStatus.UNSUPPORTED}
                            else AgentAction.STOP
                        ),
                        reason_code=SelectionReasonCode.REQUIRED_CAPABILITY_UNAVAILABLE,
                    )
                )
                continue
            return ControllerDraft(
                action=AgentAction.INSPECT_ANATOMICAL_CONTEXT,
                reason_code=SelectionReasonCode.ANATOMY_REQUIRED,
            )
        if kind == EvidenceKind.QUALITY:
            return ControllerDraft(
                action=AgentAction.INSPECT_IMAGE_QUALITY,
                reason_code=SelectionReasonCode.QUALITY_EVIDENCE_REQUIRED,
            )
        if kind == EvidenceKind.PRIOR:
            return ControllerDraft(
                action=AgentAction.RETRIEVE_PRIOR_STUDIES,
                reason_code=SelectionReasonCode.PRIOR_LOOKUP_REQUIRED,
            )
        if kind == EvidenceKind.LONGITUDINAL:
            return ControllerDraft(
                action=AgentAction.STOP,
                reason_code=SelectionReasonCode.PRIOR_EVIDENCE_UNAVAILABLE,
            )
        if kind == EvidenceKind.DIAGNOSTIC:
            return ControllerDraft(
                action=AgentAction.RETRIEVE_GUIDELINE,
                reason_code=SelectionReasonCode.DIAGNOSTIC_GUIDELINE_REQUIRED,
            )
        if kind == EvidenceKind.TREATMENT:
            return ControllerDraft(
                action=AgentAction.RETRIEVE_GUIDELINE,
                reason_code=SelectionReasonCode.TREATMENT_GUIDELINE_REQUIRED,
            )

    return deferred_terminal or ControllerDraft(
        action=AgentAction.STOP,
        reason_code=SelectionReasonCode.TASK_ALREADY_COMPLETE,
    )


def _independent_required_actions(
    state: CaseState,
    task_spec: TaskSpec,
    preferred: ControllerDraft,
) -> list[ControllerDraft]:
    """Return currently executable evidence actions for LLM ordering.

    TaskSpec defines what must be obtained; this list only lets the controller
    choose the order of independent missing evidence.  Dependencies (for
    example anatomy after localization) and failed/terminal evidence remain
    deterministic guards.
    """

    if preferred.action in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN}:
        return [preferred]
    reason_by_kind = {
        EvidenceKind.CLASSIFICATION: (
            AgentAction.CLASSIFY_CURRENT_CXR,
            SelectionReasonCode.CLASSIFICATION_REQUIRED,
        ),
        EvidenceKind.LOCALIZATION: (
            AgentAction.LOCALIZE_CURRENT_CXR,
            SelectionReasonCode.LOCALIZATION_REQUIRED,
        ),
        EvidenceKind.ANATOMY: (
            AgentAction.INSPECT_ANATOMICAL_CONTEXT,
            SelectionReasonCode.ANATOMY_REQUIRED,
        ),
        EvidenceKind.QUALITY: (
            AgentAction.INSPECT_IMAGE_QUALITY,
            SelectionReasonCode.QUALITY_EVIDENCE_REQUIRED,
        ),
        EvidenceKind.PRIOR: (
            AgentAction.RETRIEVE_PRIOR_STUDIES,
            SelectionReasonCode.PRIOR_LOOKUP_REQUIRED,
        ),
        EvidenceKind.DIAGNOSTIC: (
            AgentAction.RETRIEVE_GUIDELINE,
            SelectionReasonCode.DIAGNOSTIC_GUIDELINE_REQUIRED,
        ),
        EvidenceKind.TREATMENT: (
            AgentAction.RETRIEVE_GUIDELINE,
            SelectionReasonCode.TREATMENT_GUIDELINE_REQUIRED,
        ),
    }
    drafts: list[ControllerDraft] = [preferred]
    for kind in task_spec.required_evidence:
        mapping = reason_by_kind.get(kind)
        if mapping is None:
            continue
        slot = _slot_for_requirement(state, kind)
        if slot.status in _SATISFIED or slot.status in {
            EvidenceStatus.FAILED,
            EvidenceStatus.UNSUPPORTED,
            EvidenceStatus.EVIDENCE_GAP,
        }:
            continue
        if (
            kind == EvidenceKind.ANATOMY
            and TaskGoal.ANATOMICAL_CONTEXT in task_spec.task_goals
            and state.localization_evidence.status not in _SATISFIED
        ):
            continue
        drafts.append(ControllerDraft(action=mapping[0], reason_code=mapping[1]))
    unique: dict[AgentAction, ControllerDraft] = {}
    for draft in drafts:
        unique.setdefault(draft.action, draft)
    return list(unique.values())


def _hard_guard(state: CaseState, budget: AgentBudget) -> ControllerDraft | None:
    technical_conflicts = {
        ConflictFlag.TECHNICAL_QUALITY_CONFLICT,
        ConflictFlag.TOOL_FAILURE_CONFLICT,
    }
    if any(flag in technical_conflicts for flag in state.conflict_flags):
        return ControllerDraft(
            action=AgentAction.REFER_TO_HUMAN,
            reason_code=SelectionReasonCode.TECHNICAL_FAILURE,
        )
    if state.failed_actions:
        return ControllerDraft(
            action=AgentAction.REFER_TO_HUMAN,
            reason_code=SelectionReasonCode.TECHNICAL_FAILURE,
        )
    # Non-technical uncertainty is an observation for answer synthesis, not a
    # reason to send an interactive case to the batch review workbench.
    return None


def _allowed_actions(
    candidates: list[ControllerDraft],
    capabilities: dict[AgentAction, ActionCapability],
    budget: AgentBudget,
) -> list[AgentAction]:
    if candidates and candidates[0].action in {
        AgentAction.STOP,
        AgentAction.REFER_TO_HUMAN,
    }:
        return [candidates[0].action]
    allowed: list[AgentAction] = []
    for candidate in candidates:
        capability = capabilities.get(candidate.action)
        if capability is None or not capability.available:
            continue
        if capability.cost_units > budget.remaining_cost_units:
            continue
        if capability.expensive_vision and budget.remaining_expensive_vision_calls == 0:
            continue
        allowed.append(candidate.action)
    return list(dict.fromkeys(allowed))


class BoundedAgentController:
    """Choose and validate one next action from structured runtime state."""

    def choose_next_action(
        self,
        *,
        state: CaseState,
        task_spec: TaskSpec,
        capabilities: list[ActionCapability],
        budget: AgentBudget,
        generator: StructuredControllerGenerator | None = None,
    ) -> ControllerDecision:
        capability_map = {item.action: item for item in capabilities}
        # Social/capability/status turns are intentionally tool-free even when
        # the bound case contains a conflict from an earlier medical task.
        required_next = _required_next_action(state, task_spec)
        hard_guard = None if task_spec.no_tool_only else _hard_guard(state, budget)
        # A non-technical cross-evidence conflict must not prevent the Agent
        # from collecting another explicitly requested, independent evidence
        # source.  It is handled after those bounded actions are complete.
        conflict_safe_actions = {
            AgentAction.RETRIEVE_GUIDELINE,
            AgentAction.INSPECT_ANATOMICAL_CONTEXT,
            AgentAction.INSPECT_IMAGE_QUALITY,
            AgentAction.RETRIEVE_PRIOR_STUDIES,
        }
        if (
            hard_guard is not None
            and hard_guard.reason_code == SelectionReasonCode.EVIDENCE_CONFLICT
            and required_next.action in conflict_safe_actions
        ):
            hard_guard = None
        # A failed tool blocks that evidence branch, not independent goals from
        # the same user turn.  Keep the failure in state/receipts, but allow a
        # different required action to run before the terminal handoff.  Input
        # quality conflicts remain fail-closed because they can invalidate all
        # image-derived branches.
        if (
            hard_guard is not None
            and hard_guard.reason_code == SelectionReasonCode.TECHNICAL_FAILURE
            and state.failed_actions
            and required_next.action
            not in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN, *state.failed_actions}
            and ConflictFlag.TECHNICAL_QUALITY_CONFLICT not in state.conflict_flags
        ):
            hard_guard = None
        preferred = hard_guard or required_next
        if (
            hard_guard is None
            and preferred.action not in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN}
            and (budget.remaining_steps == 0 or budget.remaining_tool_calls == 0)
        ):
            hard_guard = ControllerDraft(
                action=AgentAction.REFER_TO_HUMAN,
                reason_code=SelectionReasonCode.BUDGET_EXHAUSTED,
            )
            preferred = hard_guard
        candidates = _independent_required_actions(state, task_spec, preferred)
        candidate_by_action = {item.action: item for item in candidates}
        allowed = _allowed_actions(candidates, capability_map, budget)
        source = (
            DecisionSource.HARD_GUARD
            if hard_guard is not None
            else DecisionSource.STATE_POLICY_FALLBACK
        )
        draft = preferred
        usage: dict[str, int] = {}
        schema_validated = True

        can_generate = generator is not None and callable(
            getattr(generator, "complete_structured", None)
        )
        if (
            hard_guard is None
            and preferred.action not in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN}
            and can_generate
            and len(allowed) > 1
        ):
            view = controller_state_view(state, task_spec)
            prompt = json.dumps(
                {
                    "state": view,
                    "budget": budget.model_dump(mode="json"),
                    "allowed_actions": [item.value for item in allowed],
                    "instruction": "Choose exactly one next action; do not add future steps.",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            try:
                content, usage = generator.complete_structured(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a bounded action selector. Use only structured state; "
                                "return one allowlisted action and one reason code as JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    json_schema=ControllerDraft.model_json_schema(),
                    schema_name="tbx_next_action",
                    max_tokens=64,
                    seed=20260831,
                )
                candidate = ControllerDraft.model_validate_json(content)
                if candidate.action not in allowed:
                    raise ValueError("controller chose an action outside the guarded allowlist")
                expected = candidate_by_action[candidate.action]
                if candidate.reason_code != expected.reason_code:
                    raise ValueError(
                        "controller reason does not match the selected evidence action"
                    )
                draft = expected
                source = DecisionSource.LLM_CONTROLLER
            except (ValidationError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
                # Invalid model output cannot broaden authority. The state policy
                # remains the auditable, bounded fallback for this one decision.
                draft = preferred
                source = DecisionSource.STATE_POLICY_FALLBACK
                schema_validated = False
            except Exception:
                draft = preferred
                source = DecisionSource.STATE_POLICY_FALLBACK
                schema_validated = False

        if draft.action not in allowed:
            draft = ControllerDraft(
                action=AgentAction.STOP,
                reason_code=(
                    SelectionReasonCode.TOOL_UNAVAILABLE
                    if preferred.action not in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN}
                    else preferred.reason_code
                ),
            )
            source = DecisionSource.HARD_GUARD

        capability = capability_map.get(draft.action)
        state_view = controller_state_view(state, task_spec)
        return ControllerDecision(
            step_index=budget.steps_used,
            action=draft.action,
            reason_code=draft.reason_code,
            source=source,
            state_sha256=_sha256(state_view),
            task_spec_sha256=_sha256(task_spec.model_dump(mode="json")),
            allowed_actions=allowed,
            cost_units=capability.cost_units if capability is not None else 0,
            expensive_vision=(capability.expensive_vision if capability is not None else False),
            schema_validated=schema_validated,
            controller_backend=(getattr(generator, "backend_id", None) if generator else None),
            controller_model=(getattr(generator, "model", None) if generator else None),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


def consume_budget(budget: AgentBudget, decision: ControllerDecision) -> AgentBudget:
    """Return the budget after one decision and, if applicable, one tool call."""

    terminal = decision.action in {AgentAction.STOP, AgentAction.REFER_TO_HUMAN}
    return budget.model_copy(
        update={
            "steps_used": budget.steps_used + 1,
            "tool_calls_used": budget.tool_calls_used + (0 if terminal else 1),
            "expensive_vision_calls_used": (
                budget.expensive_vision_calls_used
                + (1 if decision.expensive_vision and not terminal else 0)
            ),
            "cost_units_used": budget.cost_units_used + (0 if terminal else decision.cost_units),
        }
    )
