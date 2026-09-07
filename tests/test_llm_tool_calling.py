from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tbx_agent.llm.tool_calling import (
    HighLevelToolName,
    NativeToolFailureCode,
    ToolSelectionMode,
    native_tool_definitions,
    select_react_action,
)

QUERY = "孕妇怀疑肺结核时应该做什么检查？"
MESSAGES = [
    {"role": "system", "content": "Choose one action or answer directly."},
    {"role": "user", "content": QUERY},
]


class _StructuredOnlyGenerator:
    backend_id = "test"
    model = "test-model"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.payload, ensure_ascii=False), {
            "prompt_tokens": 11,
            "completion_tokens": 5,
        }


class _NativeGenerator(_StructuredOnlyGenerator):
    def __init__(
        self,
        *,
        tool_calls: list[object],
        content: str | None,
        fallback: dict[str, object] | None = None,
    ) -> None:
        super().__init__(fallback or {"tool": None, "direct_answer": "fallback"})
        self.tool_calls = tool_calls
        self.content = content
        self.native_calls: list[dict[str, object]] = []
        self.unavailable: list[NativeToolFailureCode] = []

    def complete_tool_calls(self, **kwargs):
        self.native_calls.append(kwargs)
        return self.tool_calls, self.content, {
            "prompt_tokens": 7,
            "completion_tokens": 3,
        }

    def mark_native_tool_calling_unavailable(self, reason_code):
        self.unavailable.append(reason_code)


def _tool_call(name: str, arguments: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        )
    )


def test_native_catalog_contains_only_four_evidence_tools() -> None:
    definitions = native_tool_definitions(trusted_query=QUERY)

    assert [item["function"]["name"] for item in definitions] == [
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    ]
    search = definitions[-1]["function"]["parameters"]
    assert search["properties"]["query"]["const"] == QUERY
    serialized = json.dumps(definitions, ensure_ascii=False)
    assert "case_id" not in serialized
    for removed_pseudo_tool in (
        "general_chat",
        "social",
        "capabilities",
        "case_status",
        "inspect_image_quality",
        "compare_with_prior",
    ):
        assert removed_pseudo_tool not in serialized


def test_native_and_json_catalogs_are_restricted_to_current_allowed_subset() -> None:
    generator = _NativeGenerator(
        tool_calls=[_tool_call("localize_cxr", {})],
        content=None,
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
        allowed_tools=[HighLevelToolName.LOCALIZE_CXR],
    )

    assert selection.tool_call is not None
    assert selection.tool_call.name == HighLevelToolName.LOCALIZE_CXR
    assert [
        item["function"]["name"] for item in generator.native_calls[0]["tools"]
    ] == ["localize_cxr"]


def test_disallowed_native_tool_is_rejected_before_json_fallback() -> None:
    generator = _NativeGenerator(
        tool_calls=[_tool_call("classify_cxr", {})],
        content=None,
        fallback={"tool": "localize_cxr", "direct_answer": None},
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
        allowed_tools=[HighLevelToolName.LOCALIZE_CXR],
    )

    assert selection.mode == ToolSelectionMode.JSON_SCHEMA_FALLBACK
    assert selection.native_failure_reason == NativeToolFailureCode.TOOL_NOT_ALLOWED
    assert selection.tool_call is not None
    assert selection.tool_call.name == HighLevelToolName.LOCALIZE_CXR
    schema = generator.calls[0]["json_schema"]
    assert schema["$defs"]["HighLevelToolName"]["enum"] == ["localize_cxr"]


def test_disallowed_structured_tool_is_rejected_even_if_provider_ignores_schema() -> None:
    generator = _StructuredOnlyGenerator(
        {"tool": "classify_cxr", "direct_answer": None}
    )

    with pytest.raises(ValueError, match="outside the current allowlist"):
        select_react_action(
            generator,
            messages=MESSAGES,
            trusted_query=QUERY,
            allowed_tools=[HighLevelToolName.LOCALIZE_CXR],
        )


def test_empty_allowed_subset_skips_native_and_permits_only_direct_answer() -> None:
    generator = _NativeGenerator(tool_calls=[], content="native should not run")

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
        allowed_tools=[],
    )

    assert generator.native_calls == []
    assert selection.direct_answer == "fallback"
    assert selection.native_failure_reason == NativeToolFailureCode.NO_TOOLS_ALLOWED
    schema = generator.calls[0]["json_schema"]
    assert schema["properties"]["tool"] == {"type": "null"}
    assert "$defs" not in schema


def test_native_tool_call_is_validated_and_runtime_binds_arguments() -> None:
    generator = _NativeGenerator(
        tool_calls=[_tool_call("search_tb_knowledge", {"query": QUERY})],
        content=None,
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
    )

    assert selection.mode == ToolSelectionMode.NATIVE_TOOL_CALL
    assert selection.tool_call is not None
    assert selection.tool_call.name == HighLevelToolName.SEARCH_TB_KNOWLEDGE
    assert selection.tool_call.arguments.model_dump() == {"query": QUERY}
    assert selection.direct_answer is None
    assert selection.prompt_tokens == 7
    assert selection.completion_tokens == 3
    assert generator.calls == []
    assert generator.native_calls[0]["tool_choice"] == "auto"


def test_native_direct_answer_is_a_zero_action_react_step() -> None:
    generator = _NativeGenerator(tool_calls=[], content="2")

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query="1+1=?",
    )

    assert selection.mode == ToolSelectionMode.NATIVE_TOOL_CALL
    assert selection.tool_call is None
    assert selection.direct_answer == "2"
    assert generator.calls == []


@pytest.mark.parametrize(
    "content",
    (
        "思考 用户希望知道一道算术题。\n最终答案：2",
        "Analysis: The user wants a concise calculation.\nFinal Answer: 2",
        (
            "<think>用户希望得到结果；当前病例：分类未运行；定位未运行。</think>"
            "\n答案：2"
        ),
    ),
)
def test_native_direct_answer_keeps_only_explicit_final_answer(content: str) -> None:
    generator = _NativeGenerator(tool_calls=[], content=content)

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query="1+1=?",
    )

    assert selection.mode == ToolSelectionMode.NATIVE_TOOL_CALL
    assert selection.direct_answer == "2"
    assert "思考" not in selection.direct_answer
    assert "Analysis" not in selection.direct_answer
    assert "当前病例" not in selection.direct_answer


def test_native_runtime_state_leak_falls_back_to_clean_structured_answer() -> None:
    generator = _NativeGenerator(
        tool_calls=[],
        content='{"case_state":{"classification":{"status":"not_run"}}}',
        fallback={"tool": None, "direct_answer": "2"},
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query="1+1=?",
    )

    assert selection.mode == ToolSelectionMode.JSON_SCHEMA_FALLBACK
    assert selection.native_failure_reason == NativeToolFailureCode.REQUEST_FAILED
    assert selection.direct_answer == "2"


@pytest.mark.parametrize(
    "answer",
    (
        "思考 用户希望知道答案，但我还需要分析。",
        "<think>The user wants a concise answer but the reasoning block is unclosed.",
        'case_state={"image_loaded":true,"classification":{"status":"not_run"}}',
        "当前病例：分类未运行；定位未运行。",
    ),
)
def test_structured_direct_answer_rejects_unrecoverable_reasoning_or_state(
    answer: str,
) -> None:
    generator = _StructuredOnlyGenerator(
        {"tool": None, "direct_answer": answer}
    )

    with pytest.raises(ValueError):
        select_react_action(
            generator,
            messages=MESSAGES,
            trusted_query="1+1=?",
        )


def test_explicit_case_status_question_may_return_concise_case_status() -> None:
    answer = "当前病例：分类未运行；定位未运行。"
    generator = _StructuredOnlyGenerator(
        {"tool": None, "direct_answer": answer}
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query="当前病例的分类和定位运行了吗？",
    )

    assert selection.direct_answer == answer


def test_invalid_native_query_safely_falls_back_to_schema() -> None:
    generator = _NativeGenerator(
        tool_calls=[
            _tool_call(
                "search_tb_knowledge",
                {"query": "different or model-injected query"},
            )
        ],
        content=None,
        fallback={"tool": "search_tb_knowledge", "direct_answer": None},
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
    )

    assert selection.mode == ToolSelectionMode.JSON_SCHEMA_FALLBACK
    assert selection.native_failure_reason == NativeToolFailureCode.QUERY_MISMATCH
    assert selection.tool_call is not None
    assert selection.tool_call.arguments.model_dump() == {"query": QUERY}
    assert len(generator.calls) == 1
    assert generator.unavailable == []


def test_multiple_native_tool_calls_are_rejected_then_fall_back_to_one() -> None:
    generator = _NativeGenerator(
        tool_calls=[
            _tool_call("classify_cxr", {}),
            _tool_call("localize_cxr", {}),
        ],
        content=None,
        fallback={"tool": "classify_cxr", "direct_answer": None},
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query="分析并定位这张胸片",
    )

    assert selection.mode == ToolSelectionMode.JSON_SCHEMA_FALLBACK
    assert selection.native_failure_reason == NativeToolFailureCode.MULTIPLE_CALLS
    assert selection.tool_call is not None
    assert selection.tool_call.name == HighLevelToolName.CLASSIFY_CXR
    assert len(generator.calls) == 1


def test_structured_fallback_has_no_model_argument_surface() -> None:
    generator = _StructuredOnlyGenerator(
        {"tool": "classify_cxr", "direct_answer": None}
    )

    selection = select_react_action(
        generator,
        messages=MESSAGES,
        trusted_query=QUERY,
    )

    assert selection.mode == ToolSelectionMode.JSON_SCHEMA_FALLBACK
    assert selection.native_failure_reason == NativeToolFailureCode.API_UNAVAILABLE
    assert selection.tool_call is not None
    assert selection.tool_call.name == HighLevelToolName.CLASSIFY_CXR
    assert selection.tool_call.arguments.model_dump() == {}
    schema = generator.calls[0]["json_schema"]
    assert set(schema["properties"]) == {"tool", "direct_answer"}
    assert set(schema["required"]) == {"tool", "direct_answer"}
    assert "arguments" not in json.dumps(schema)


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "classify_cxr", "direct_answer": "also answer"},
        {"tool": None, "direct_answer": None},
        {"tool": "run_shell", "direct_answer": None},
        {"tool": "classify_cxr", "direct_answer": None, "case_id": "model-owned"},
    ],
)
def test_structured_fallback_rejects_ambiguous_or_untrusted_payloads(
    payload: dict[str, object],
) -> None:
    generator = _StructuredOnlyGenerator(payload)

    with pytest.raises((ValidationError, ValueError)):
        select_react_action(
            generator,
            messages=MESSAGES,
            trusted_query=QUERY,
        )
