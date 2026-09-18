"""One structured ReAct decision; planning is optional internal control.

The model never supplies executable arguments or case identifiers. Validated
decisions remain proposals: the runtime owns permissions, dependencies, budgets,
evidence availability and final response verification.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .llm.tool_calling import HighLevelToolName, sanitize_model_answer
from .plan_react import AnswerFocus, EvidenceNeed
from .semantic_intent import INTENT_PROMPT, IntentItem, SemanticTask, TurnIntent


class Action(StrEnum):
    TOOL = "tool"
    ANSWER = "answer"
    PLAN = "plan"


DecisionFocus = AnswerFocus | Literal["lung_lobe_limit", "screening_limit"]


class _DecisionDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action
    tool: HighLevelToolName | None = None
    tasks: list[IntentItem] = Field(default_factory=list, max_length=4)
    answer_focus: DecisionFocus = AnswerFocus.GENERAL
    evidence: list[EvidenceNeed] = Field(default_factory=list, max_length=4)
    answer: str | None = Field(default=None, min_length=1, max_length=8_192)

    @model_validator(mode="after")
    def _validate_action(self) -> _DecisionDraft:
        if self.answer is not None:
            self.answer = self.answer.strip()
            if not self.answer:
                raise ValueError("answer must not be blank")
        if EvidenceNeed.NONE in self.evidence:
            raise ValueError("evidence must name an actual evidence class")
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("evidence must not contain duplicates")

        if self.action == Action.TOOL:
            if self.tool is None:
                raise ValueError("tool action requires one domain tool")
            if self.tasks or self.evidence or self.answer is not None:
                raise ValueError("tool action cannot contain tasks, evidence or an answer")
            if self.answer_focus != AnswerFocus.GENERAL:
                raise ValueError("tool action cannot select an answer focus")
        elif self.action == Action.ANSWER:
            if self.tool is not None or self.tasks:
                raise ValueError("answer action cannot contain a tool or tasks")
            if (
                self.answer_focus == AnswerFocus.GENERAL
                and not self.evidence
                and self.answer is None
            ):
                raise ValueError("ordinary chat requires a nonempty answer")
        else:
            if self.tool is not None or self.evidence or self.answer is not None:
                raise ValueError("plan action cannot contain a tool, evidence or an answer")
            if self.answer_focus != AnswerFocus.GENERAL:
                raise ValueError("plan action cannot select an answer focus")
            if not self.tasks or (
                len(self.tasks) < 2
                and not any(task.when == "classification_abnormal" for task in self.tasks)
            ):
                raise ValueError("plan requires multiple tasks or a conditional task")
            if len({task.task for task in self.tasks}) != len(self.tasks):
                raise ValueError("plan tasks must not repeat the same task")
        return self


class Decision(_DecisionDraft):
    """Validated decision and bounded provider usage, without private reasoning."""

    prompt_tokens: int | None = Field(default=None, ge=1)
    completion_tokens: int | None = Field(default=None, ge=1)


DECISION_PROMPT = (
    INTENT_PROMPT.replace(
        "Return only JSON tasks. Do not answer, generate a verbose plan, or select tasks merely",
        "Return JSON tasks and optional answer. "
        "Do not generate a verbose plan or select tasks merely",
    ).replace(
        "No tool arguments, identifiers, private reasoning or extra fields.",
        "No tool arguments, identifiers or private reasoning. "
        "For general_chat, include a short answer "
        "in the user's language. Other tasks need no answer prose.",
    )
    + """
Additional tasks:
explain_lung_lobe_limits: whether a 2-D lung field establishes an anatomical lung LOBE.
A question WHERE IN THE LUNG FIELDS still means locate_within_lungs, not this boundary.
explain_screening_limits: whether the model result/no boxes can confirm or exclude disease.
These explain existing system limits and require no new tool.
Your tasks are the next action proposal. One task executes directly (or reads its cache);
multiple/conditional tasks request a lightweight public plan, never a batch tool call.
Only current requested work is selected. Ignore previous goals already completed.
"""
)


class _SemanticDecision(TurnIntent):
    answer: str | None = Field(default=None, min_length=1, max_length=8192)


def decision_schema(*, allowed_tools, allow_plan=True) -> dict[str, Any]:
    # The semantic wire contract is deliberately independent of which tools are
    # currently runnable. Missing images/cached results cannot alter user intent.
    allowed = tuple(HighLevelToolName(tool) for tool in allowed_tools)
    if len(set(allowed)) != len(allowed):
        raise ValueError("allowed_tools must not contain duplicates")
    return _SemanticDecision.model_json_schema()


def _decode_semantic(content, *, allowed, allow_plan, case_context):
    import json

    raw = json.loads(content)
    if not isinstance(raw, dict) or "action" in raw:
        return _decode_wire(content)
    wire = _SemanticDecision.model_validate(raw)
    if allow_plan and (len(wire.tasks) > 1 or any(t.when != "always" for t in wire.tasks)):
        return _DecisionDraft(action=Action.PLAN, tasks=wire.tasks)
    operations = {
        SemanticTask.CLASSIFY: (HighLevelToolName.CLASSIFY_CXR, EvidenceNeed.CLASSIFICATION),
        SemanticTask.BOXES: (HighLevelToolName.LOCALIZE_CXR, EvidenceNeed.LOCALIZATION),
        SemanticTask.ANATOMY: (HighLevelToolName.ANALYZE_LUNG_ANATOMY, EvidenceNeed.LUNG_ANATOMY),
        SemanticTask.KNOWLEDGE: (HighLevelToolName.SEARCH_TB_KNOWLEDGE, EvidenceNeed.TB_KNOWLEDGE),
    }
    focuses = {
        SemanticTask.RATIONALE: AnswerFocus.CLASSIFICATION_RATIONALE,
        SemanticTask.STATUS: AnswerFocus.CASE_STATUS,
        SemanticTask.CAPABILITIES: AnswerFocus.CAPABILITIES,
        SemanticTask.QUALITY: AnswerFocus.IMAGE_QUALITY,
        SemanticTask.COMPARISON: AnswerFocus.PRIOR_COMPARISON,
        SemanticTask.LOBE_LIMIT: AnswerFocus.LUNG_LOBE_LIMIT,
        SemanticTask.SCREENING_LIMIT: AnswerFocus.SCREENING_LIMIT,
        SemanticTask.CHAT: AnswerFocus.GENERAL,
    }
    evidence = []
    focus = AnswerFocus.GENERAL
    tasks = []
    for item in wire.tasks:
        if (
            item.when == "classification_abnormal"
            and case_context.get("classification", {}).get("result") == "healthy"
        ):
            continue
        tasks.append(item.task)
        if item.task in operations:
            tool, need = operations[item.task]
            if tool in allowed:
                return _DecisionDraft(action=Action.TOOL, tool=tool)
            if need not in evidence:
                evidence.append(need)
        else:
            focus = focuses[item.task]
    if SemanticTask.STATUS in tasks and SemanticTask.CAPABILITIES in tasks:
        focus = AnswerFocus.CAPABILITIES_AND_STATUS
    return _DecisionDraft(
        action=Action.ANSWER, answer_focus=focus, evidence=evidence, answer=wire.answer
    )


def _decode_wire(content: str) -> _DecisionDraft:
    """Typed adapters share validation with the semantic production protocol."""
    return _DecisionDraft.model_validate_json(content)


def select_decision(
    generator: Any,
    *,
    messages: list[dict[str, str]],
    trusted_query: str,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...],
    max_tokens: int = 512,
    seed: int = 20260901,
    allow_plan: bool = True,
    case_context: dict[str, Any] | None = None,
) -> Decision:
    """Make one structured call; failures are handled by the runtime's fallback.

    ``messages`` must contain the decision instructions and trusted context. This
    adapter neither adds an intent pass nor invokes native tool calling. The current
    trusted question is used only for answer sanitization, never as a model argument.
    """

    if not trusted_query.strip():
        raise ValueError("trusted_query must not be blank")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    allowed = tuple(HighLevelToolName(tool) for tool in allowed_tools)
    schema = decision_schema(allowed_tools=allowed, allow_plan=allow_plan)
    complete = getattr(generator, "complete_structured", None)
    if not callable(complete):
        raise RuntimeError("structured decision API is unavailable")
    content, usage = complete(
        messages=messages,
        json_schema=schema,
        schema_name="tbx_react_decision",
        max_tokens=max_tokens,
        seed=seed,
    )
    draft = _decode_semantic(
        content, allowed=allowed, allow_plan=allow_plan, case_context=case_context or {}
    )
    if draft.action == Action.PLAN and not allow_plan:
        raise ValueError("planning is disabled for this decision")
    if draft.action == Action.TOOL and draft.tool not in allowed:
        raise ValueError("selected tool is not allowed for this decision")
    payload = draft.model_dump()
    if draft.answer is not None:
        # The model already selected status semantically. Do not let the legacy
        # keyword adapter veto that focus; private-state/reasoning checks remain.
        status_focus = draft.answer_focus in {
            AnswerFocus.CASE_STATUS,
            AnswerFocus.CAPABILITIES_AND_STATUS,
        }
        payload["answer"] = sanitize_model_answer(
            draft.answer, trusted_query=None if status_focus else trusted_query
        )
    if isinstance(usage, dict):
        for field in ("prompt_tokens", "completion_tokens"):
            value = usage.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                payload[field] = value
    return Decision.model_validate(payload)


__all__ = [
    "Action",
    "DECISION_PROMPT",
    "Decision",
    "DecisionFocus",
    "decision_schema",
    "select_decision",
]
