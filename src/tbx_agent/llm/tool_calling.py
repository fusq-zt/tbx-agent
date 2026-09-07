"""Provider-neutral contracts for one high-level Agent tool decision.

The model may request at most one of four public capabilities.  It never owns
case identifiers or other executable arguments: the runtime binds those from
trusted state.  ``search_tb_knowledge`` is the sole exception at the wire
boundary, where a native provider must echo the current question exactly; the
echo is validated and then replaced by the trusted copy.

This module deliberately contains no execution loop. A validated selection is
still permission-, state-, argument- and budget-checked by the LangGraph
runtime before any audited internal handler can run.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class HighLevelToolName(StrEnum):
    """The complete model-visible TBX-Agent tool catalog."""

    CLASSIFY_CXR = "classify_cxr"
    LOCALIZE_CXR = "localize_cxr"
    ANALYZE_LUNG_ANATOMY = "analyze_lung_anatomy"
    SEARCH_TB_KNOWLEDGE = "search_tb_knowledge"


class ToolSelectionMode(StrEnum):
    """Non-sensitive provenance for the provider selection path."""

    NATIVE_TOOL_CALL = "native_tool_call"
    JSON_SCHEMA_FALLBACK = "json_schema_fallback"
    PLAN_EVIDENCE_FALLBACK = "plan_evidence_fallback"


class NativeToolFailureCode(StrEnum):
    """Bounded failure codes safe to retain in an audit trace."""

    API_UNAVAILABLE = "native_api_unavailable"
    NO_TOOLS_ALLOWED = "native_no_tools_allowed"
    CACHED_UNAVAILABLE = "native_cached_unavailable"
    PROVIDER_UNSUPPORTED = "native_provider_unsupported"
    REQUEST_FAILED = "native_request_failed"
    NO_SELECTION = "native_no_selection"
    MULTIPLE_CALLS = "native_multiple_calls"
    INVALID_CALL_SHAPE = "native_invalid_call_shape"
    UNKNOWN_TOOL = "native_unknown_tool"
    TOOL_NOT_ALLOWED = "native_tool_not_allowed"
    INVALID_ARGUMENTS = "native_invalid_arguments"
    QUERY_MISMATCH = "native_query_mismatch"
    AMBIGUOUS_RESPONSE = "native_ambiguous_response"


class NativeToolCallError(RuntimeError):
    """A native selection failure without provider text or credentials."""

    def __init__(
        self,
        reason_code: NativeToolFailureCode,
        *,
        cache_unavailable: bool = False,
    ) -> None:
        super().__init__(reason_code.value)
        self.reason_code = reason_code
        self.cache_unavailable = cache_unavailable


class ModelAnswerRejectedError(ValueError):
    """A model answer exposed non-user-facing reasoning or runtime state."""


_FINAL_ANSWER_MARKER = re.compile(
    r"(?:^|\n)[ \t]*(?:#{1,6}[ \t]*)?(?:\*{0,2})"
    r"(?:最终答案|最后答案|答案|final[ \t]+answer|answer)"
    r"(?:\*{0,2})[ \t]*[:：][ \t]*(?:\*{0,2})",
    re.IGNORECASE,
)
_CLOSED_REASONING_BLOCK = re.compile(
    r"<(?:think|analysis|reasoning)>.*?</(?:think|analysis|reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
_REASONING_PREFIX = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*{0,2})?"
    r"(?:思考|分析|推理|思维过程|推理过程|thought|analysis|reasoning)"
    r"(?:\*{0,2})?(?:[ \t]*[:：]|[ \t]+(?:用户|user|the[ \t]+user|we[ \t]+need|我需要))",
    re.IGNORECASE | re.MULTILINE,
)
_REASONING_TAG = re.compile(r"</?(?:think|analysis|reasoning)>", re.IGNORECASE)
_INTERNAL_RUNTIME_MARKER = re.compile(
    r"(?:TBX_(?:PLAN_)?INTERNAL_CONTEXT_JSON|allowed_tools_this_step|"
    r"case_state|observations|recent_dialogue|pending_selection|"
    r"\"(?:image_loaded|quality_check|classification|localization|anatomy)\"[ \t]*:)",
    re.IGNORECASE,
)
_CASE_STATE_LINE = re.compile(
    r"^[ \t]*(?:[-*][ \t]*)?(?:当前)?病例(?:状态)?[ \t]*[:：].*"
    r"(?:分类|定位|肺野|图像).*(?:未运行|已完成|未请求|不可用|失败|"
    r"not[_ -]?run|completed|unavailable|failed)",
    re.IGNORECASE,
)


def _explicit_case_status_question(query: str) -> bool:
    normalized = "".join(query.casefold().split())
    return any(
        marker in normalized
        for marker in (
            "当前病例",
            "病例状态",
            "分类和定位运行了吗",
            "分类运行了吗",
            "定位运行了吗",
            "当前分类状态",
            "当前定位状态",
            "case_status",
        )
    )


def sanitize_model_answer(text: str, *, trusted_query: str | None = None) -> str:
    """Project an untrusted model completion onto user-facing answer text.

    Complete reasoning blocks and text before an explicit final-answer marker
    are discarded so a useful answer can survive.  Ambiguous reasoning or raw
    runtime state inside the remaining answer is rejected rather than exposed.
    Case-status prose is admitted only when the user explicitly asked for that
    state; machine-shaped context is never user-facing.
    """

    rendered = text.strip()
    if not rendered:
        raise ModelAnswerRejectedError("model answer is blank")
    rendered = _CLOSED_REASONING_BLOCK.sub("", rendered).strip()
    final_markers = list(_FINAL_ANSWER_MARKER.finditer(rendered))
    if final_markers:
        rendered = rendered[final_markers[-1].end() :].strip()
    if not rendered:
        raise ModelAnswerRejectedError("model answer has no final answer")
    if _REASONING_PREFIX.search(rendered) or _REASONING_TAG.search(rendered):
        raise ModelAnswerRejectedError("model answer contains a reasoning prefix")
    if _INTERNAL_RUNTIME_MARKER.search(rendered):
        raise ModelAnswerRejectedError("model answer contains runtime state")

    if trusted_query is not None and not _explicit_case_status_question(trusted_query):
        lines = [
            line
            for line in rendered.splitlines()
            if not _CASE_STATE_LINE.search(line)
        ]
        rendered = "\n".join(lines).strip()
        if not rendered:
            raise ModelAnswerRejectedError("model answer contains only unrelated case state")
    return rendered


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyToolArguments(_StrictModel):
    """Arguments for case-bound tools; all values come from runtime state."""


class SearchTBKnowledgeArguments(_StrictModel):
    """Trusted copy of the current question passed to the knowledge tool."""

    query: str = Field(min_length=1, max_length=8_192)


class HighLevelToolCall(_StrictModel):
    """Validated, runtime-bound internal tool call."""

    name: HighLevelToolName
    arguments: EmptyToolArguments | SearchTBKnowledgeArguments

    @model_validator(mode="after")
    def _arguments_match_tool(self) -> HighLevelToolCall:
        if self.name == HighLevelToolName.SEARCH_TB_KNOWLEDGE:
            if not isinstance(self.arguments, SearchTBKnowledgeArguments):
                raise ValueError("search_tb_knowledge requires the trusted query")
        elif not isinstance(self.arguments, EmptyToolArguments):
            raise ValueError("case-bound tools do not accept model arguments")
        return self


class HighLevelToolSelection(_StrictModel):
    """Exactly one tool request or one direct response for the current step."""

    tool_call: HighLevelToolCall | None = None
    direct_answer: str | None = Field(default=None, min_length=1, max_length=8_192)
    mode: ToolSelectionMode
    native_failure_reason: NativeToolFailureCode | None = None
    prompt_tokens: int | None = Field(default=None, ge=1)
    completion_tokens: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> HighLevelToolSelection:
        if (self.tool_call is None) == (self.direct_answer is None):
            raise ValueError("selection requires exactly one tool call or direct answer")
        if self.direct_answer is not None:
            answer = self.direct_answer.strip()
            if not answer:
                raise ValueError("direct answer must not be blank")
            self.direct_answer = answer
        if (
            self.mode == ToolSelectionMode.NATIVE_TOOL_CALL
            and self.native_failure_reason is not None
        ):
            raise ValueError("successful native selection cannot carry a failure reason")
        return self


class StructuredToolSelectionDraft(_StrictModel):
    """Minimal JSON-grammar fallback emitted by small local models.

    The model emits only a nullable tool name.  Arguments are intentionally not
    part of this schema and are bound below from the trusted runtime query.
    Both fields are required in the JSON document so omission cannot be
    confused with a direct-response decision.
    """

    tool: HighLevelToolName | None
    direct_answer: str | None = Field(min_length=1, max_length=8_192)

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> StructuredToolSelectionDraft:
        if (self.tool is None) == (self.direct_answer is None):
            raise ValueError("draft requires exactly one tool or direct answer")
        if self.direct_answer is not None:
            answer = self.direct_answer.strip()
            if not answer:
                raise ValueError("direct answer must not be blank")
            self.direct_answer = answer
        return self


class NativeSelectionAttempt(_StrictModel):
    """Result of the optional native path before JSON fallback."""

    selection: HighLevelToolSelection | None = None
    failure_reason: NativeToolFailureCode | None = None

    @model_validator(mode="after")
    def _one_result(self) -> NativeSelectionAttempt:
        if (self.selection is None) == (self.failure_reason is None):
            raise ValueError("native attempt requires success or one failure code")
        return self


def _normalize_allowed_tools(
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None,
) -> tuple[HighLevelToolName, ...]:
    if allowed_tools is None:
        values = tuple(HighLevelToolName)
    else:
        try:
            values = tuple(HighLevelToolName(item) for item in allowed_tools)
        except ValueError:
            raise ValueError("allowed_tools contains an unknown tool") from None
    if len(values) != len(set(values)):
        raise ValueError("allowed_tools must not contain duplicates")
    return values


def structured_tool_selection_schema(
    *,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> dict[str, Any]:
    """Return the strict fallback grammar shared by all providers."""

    allowed = _normalize_allowed_tools(allowed_tools)
    schema = StructuredToolSelectionDraft.model_json_schema()
    if allowed:
        definition = schema.get("$defs", {}).get("HighLevelToolName")
        if isinstance(definition, dict):
            definition["enum"] = [item.value for item in allowed]
    else:
        schema["properties"]["tool"] = {"type": "null"}
        schema.pop("$defs", None)
    return schema


def native_tool_definitions(
    *,
    trusted_query: str,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible definitions without runtime-owned identifiers."""

    if not trusted_query:
        raise ValueError("trusted_query must not be empty")
    allowed = set(_normalize_allowed_tools(allowed_tools))
    empty_parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    descriptions = {
        HighLevelToolName.CLASSIFY_CXR: "Classify the current chest X-ray.",
        HighLevelToolName.LOCALIZE_CXR: "Localize candidate findings on the current chest X-ray.",
        HighLevelToolName.ANALYZE_LUNG_ANATOMY: (
            "Analyze lung-field anatomy for the current chest X-ray."
        ),
    }
    definitions = [
        {
            "type": "function",
            "function": {
                "name": name.value,
                "description": description,
                "strict": True,
                "parameters": dict(empty_parameters),
            },
        }
        for name, description in descriptions.items()
        if name in allowed
    ]
    if HighLevelToolName.SEARCH_TB_KNOWLEDGE in allowed:
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": HighLevelToolName.SEARCH_TB_KNOWLEDGE.value,
                    "description": (
                        "Search reviewed tuberculosis knowledge for the user's question."
                    ),
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "const": trusted_query,
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    return definitions


def _trusted_arguments(
    name: HighLevelToolName,
    *,
    trusted_query: str,
) -> EmptyToolArguments | SearchTBKnowledgeArguments:
    if name == HighLevelToolName.SEARCH_TB_KNOWLEDGE:
        return SearchTBKnowledgeArguments(query=trusted_query)
    return EmptyToolArguments()


def selection_from_structured_json(
    content: str,
    *,
    trusted_query: str,
    native_failure_reason: NativeToolFailureCode | None,
    usage: Any = None,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> HighLevelToolSelection:
    """Validate JSON fallback and bind all executable arguments locally."""

    draft = StructuredToolSelectionDraft.model_validate_json(content)
    direct_answer = (
        sanitize_model_answer(draft.direct_answer, trusted_query=trusted_query)
        if draft.direct_answer is not None
        else None
    )
    allowed = set(_normalize_allowed_tools(allowed_tools))
    if draft.tool is not None and draft.tool not in allowed:
        raise ValueError("structured selection requested a tool outside the current allowlist")
    call = (
        HighLevelToolCall(
            name=draft.tool,
            arguments=_trusted_arguments(draft.tool, trusted_query=trusted_query),
        )
        if draft.tool is not None
        else None
    )
    return HighLevelToolSelection(
        tool_call=call,
        direct_answer=direct_answer,
        mode=ToolSelectionMode.JSON_SCHEMA_FALLBACK,
        native_failure_reason=native_failure_reason,
        **_bounded_usage(usage),
    )


def _bounded_usage(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens <= 0
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
    ):
        return {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _provider_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _decode_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > 16_384:
        raise NativeToolCallError(NativeToolFailureCode.INVALID_ARGUMENTS)
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        raise NativeToolCallError(NativeToolFailureCode.INVALID_ARGUMENTS) from None
    if not isinstance(decoded, dict):
        raise NativeToolCallError(NativeToolFailureCode.INVALID_ARGUMENTS)
    return decoded


def selection_from_native_response(
    *,
    tool_calls: Any,
    content: Any,
    trusted_query: str,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> HighLevelToolSelection:
    """Convert an untrusted OpenAI-compatible response to the internal call."""

    calls = [] if tool_calls is None else tool_calls
    if not isinstance(calls, list):
        try:
            calls = list(calls)
        except TypeError:
            raise NativeToolCallError(NativeToolFailureCode.INVALID_CALL_SHAPE) from None
    text = content if isinstance(content, str) else ""
    if not calls:
        if not text.strip():
            raise NativeToolCallError(
                NativeToolFailureCode.NO_SELECTION,
                cache_unavailable=True,
            )
        return HighLevelToolSelection(
            direct_answer=sanitize_model_answer(text, trusted_query=trusted_query),
            mode=ToolSelectionMode.NATIVE_TOOL_CALL,
        )
    if len(calls) != 1:
        raise NativeToolCallError(NativeToolFailureCode.MULTIPLE_CALLS)
    if text.strip():
        raise NativeToolCallError(NativeToolFailureCode.AMBIGUOUS_RESPONSE)

    raw_call = calls[0]
    function = _provider_value(raw_call, "function")
    if function is None:
        function = raw_call
    raw_name = _provider_value(function, "name")
    raw_arguments = _provider_value(function, "arguments")
    if not isinstance(raw_name, str):
        raise NativeToolCallError(NativeToolFailureCode.INVALID_CALL_SHAPE)
    try:
        name = HighLevelToolName(raw_name)
    except ValueError:
        raise NativeToolCallError(NativeToolFailureCode.UNKNOWN_TOOL) from None
    if name not in set(_normalize_allowed_tools(allowed_tools)):
        raise NativeToolCallError(NativeToolFailureCode.TOOL_NOT_ALLOWED)
    decoded = _decode_arguments(raw_arguments)

    if name == HighLevelToolName.SEARCH_TB_KNOWLEDGE:
        try:
            proposed = SearchTBKnowledgeArguments.model_validate(decoded)
        except ValidationError:
            raise NativeToolCallError(NativeToolFailureCode.INVALID_ARGUMENTS) from None
        if proposed.query != trusted_query:
            raise NativeToolCallError(NativeToolFailureCode.QUERY_MISMATCH)
    else:
        try:
            EmptyToolArguments.model_validate(decoded)
        except ValidationError:
            raise NativeToolCallError(NativeToolFailureCode.INVALID_ARGUMENTS) from None

    return HighLevelToolSelection(
        tool_call=HighLevelToolCall(
            name=name,
            arguments=_trusted_arguments(name, trusted_query=trusted_query),
        ),
        mode=ToolSelectionMode.NATIVE_TOOL_CALL,
    )


def attempt_native_tool_selection(
    generator: Any,
    *,
    messages: list[dict[str, str]],
    trusted_query: str,
    max_tokens: int,
    seed: int,
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> NativeSelectionAttempt:
    """Try one native provider call and return only bounded failure metadata."""

    allowed = _normalize_allowed_tools(allowed_tools)
    if not allowed:
        return NativeSelectionAttempt(failure_reason=NativeToolFailureCode.NO_TOOLS_ALLOWED)
    complete = getattr(generator, "complete_tool_calls", None)
    if not callable(complete):
        return NativeSelectionAttempt(failure_reason=NativeToolFailureCode.API_UNAVAILABLE)
    try:
        tool_calls, content, usage = complete(
            messages=messages,
            tools=native_tool_definitions(
                trusted_query=trusted_query,
                allowed_tools=allowed,
            ),
            tool_choice="auto",
            max_tokens=max_tokens,
            seed=seed,
        )
        selection = selection_from_native_response(
            tool_calls=tool_calls,
            content=content,
            trusted_query=trusted_query,
            allowed_tools=allowed,
        ).model_copy(update=_bounded_usage(usage))
    except NativeToolCallError as exc:
        mark = getattr(generator, "mark_native_tool_calling_unavailable", None)
        if exc.cache_unavailable and callable(mark):
            mark(exc.reason_code)
        return NativeSelectionAttempt(failure_reason=exc.reason_code)
    except Exception:
        # Provider exceptions can contain URLs, headers, or response bodies.
        # Do not persist or interpolate them, and do not permanently cache a
        # network/timeout failure as lack of protocol support.
        return NativeSelectionAttempt(failure_reason=NativeToolFailureCode.REQUEST_FAILED)
    return NativeSelectionAttempt(selection=selection)


def select_react_action(
    generator: Any,
    *,
    messages: list[dict[str, str]],
    trusted_query: str,
    max_tokens: int = 512,
    seed: int = 20260901,
    schema_name: str = "tbx_agent_tool_selection",
    allowed_tools: list[HighLevelToolName] | tuple[HighLevelToolName, ...] | None = None,
) -> HighLevelToolSelection:
    """Select one ReAct action using native calls with strict JSON fallback.

    Provider selection is only an action proposal.  The caller must still pass
    ``selection.tool_call`` through controller state, permission and budget
    guards.  A direct answer is likewise subject to the normal evidence and
    response-safety checks before it can be shown to the user.
    """

    if not trusted_query:
        raise ValueError("trusted_query must not be empty")
    native = attempt_native_tool_selection(
        generator,
        messages=messages,
        trusted_query=trusted_query,
        max_tokens=max_tokens,
        seed=seed,
        allowed_tools=allowed_tools,
    )
    if native.selection is not None:
        return native.selection

    complete = getattr(generator, "complete_structured", None)
    if not callable(complete):
        raise RuntimeError("structured tool selection API is unavailable")
    content, usage = complete(
        messages=messages,
        json_schema=structured_tool_selection_schema(allowed_tools=allowed_tools),
        schema_name=schema_name,
        max_tokens=max_tokens,
        seed=seed,
    )
    return selection_from_structured_json(
        content,
        trusted_query=trusted_query,
        native_failure_reason=native.failure_reason,
        usage=usage,
        allowed_tools=allowed_tools,
    )


__all__ = [
    "EmptyToolArguments",
    "HighLevelToolCall",
    "HighLevelToolName",
    "HighLevelToolSelection",
    "ModelAnswerRejectedError",
    "NativeSelectionAttempt",
    "NativeToolCallError",
    "NativeToolFailureCode",
    "SearchTBKnowledgeArguments",
    "StructuredToolSelectionDraft",
    "ToolSelectionMode",
    "attempt_native_tool_selection",
    "native_tool_definitions",
    "selection_from_native_response",
    "selection_from_structured_json",
    "select_react_action",
    "sanitize_model_answer",
    "structured_tool_selection_schema",
]
