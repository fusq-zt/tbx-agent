"""Wire contracts for the single-call ReAct decision protocol."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tbx_agent.llm.tool_calling import HighLevelToolName
from tbx_agent.plan_react import EvidenceNeed
from tbx_agent.react_decision import Action, Decision, decision_schema, select_decision


class Generator:
    def __init__(self, payload, *, usage=None):
        self.payload = payload
        self.usage = usage or {"prompt_tokens": 13, "completion_tokens": 5}
        self.calls = []

    def complete_structured(self, **kwargs):
        self.calls.append(kwargs)
        return json.dumps(self.payload), self.usage

    def complete_tool_calls(self, **kwargs):
        raise AssertionError("the structured protocol must not make a native call")


def select(generator, *, allowed=(), allow_plan=True, case_context=None):
    return select_decision(
        generator, messages=[{"role": "user", "content": "当前问题"}],
        trusted_query="当前问题", allowed_tools=list(allowed), allow_plan=allow_plan,
        case_context=case_context,
    )


@pytest.mark.parametrize("payload", [
    {"action": "tool", "tool": "classify_cxr"},
    {"action": "answer", "answer": "25"},
    {"action": "answer", "answer_focus": "case_status"},
    {"action": "answer", "answer_focus": "lung_lobe_limit"},
    {"action": "answer", "answer_focus": "screening_limit"},
    {"action": "answer", "evidence": ["lung_anatomy"]},
    {"action": "plan", "tasks": [
        {"task": "classify_image", "when": "always"},
        {"task": "show_detection_boxes", "when": "classification_abnormal"},
    ]},
    {"action": "plan", "tasks": [
        {"task": "show_detection_boxes", "when": "classification_abnormal"},
    ]},
])
def test_valid_decision_variants(payload):
    assert Decision.model_validate(payload).action == payload["action"]


@pytest.mark.parametrize("payload", [
    {"action": "answer"},
    {"action": "answer", "answer": " "},
    {"action": "tool", "tool": "shell"},
    {"action": "tool", "tool": "classify_cxr", "case_id": "other-case"},
    {"action": "tool", "tool": "search_tb_knowledge", "query": "override"},
    {"action": "tool", "tool": "classify_cxr", "answer": "完成"},
    {"action": "answer", "answer": "完成", "tool": "classify_cxr"},
    {"action": "answer", "answer": "完成", "evidence": ["none"]},
    {"action": "answer", "answer": "完成", "evidence": ["imaginary"]},
    {"action": "answer", "answer": "完成", "evidence": ["classification"] * 2},
    {"action": "plan", "tasks": [{"task": "classify_image", "when": "always"}]},
    {"action": "plan", "tasks": [{"task": "classify_image", "when": "always"}] * 2},
    {"action": "plan", "tasks": [
        {"task": "show_detection_boxes", "when": "classification_abnormal"},
    ], "answer": "完成"},
    {"action": "answer", "answer": "完成", "reasoning": "private reasoning"},
])
def test_invalid_or_ambiguous_actions_are_rejected(payload):
    with pytest.raises(ValidationError):
        Decision.model_validate(payload)


def test_one_structured_call_and_runtime_owned_usage():
    generator = Generator({"action": "answer", "answer": "25"})
    decision = select(generator)
    assert decision.action == Action.ANSWER
    assert (decision.prompt_tokens, decision.completion_tokens) == (13, 5)
    assert len(generator.calls) == 1
    request = generator.calls[0]
    assert request["schema_name"] == "tbx_react_decision"
    assert '"prompt_tokens"' not in json.dumps(request["json_schema"])
    assert request["messages"][-1]["content"] == "当前问题"


@pytest.mark.parametrize("payload", [
    {"action": "tool", "tool": "classify_cxr"},
    {"action": "plan", "tasks": [
        {"task": "show_detection_boxes", "when": "classification_abnormal"},
    ]},
    {"action": "answer", "answer": "25", "prompt_tokens": 999},
])
def test_provider_cannot_bypass_schema_constraints(payload):
    with pytest.raises(ValueError):
        select(Generator(payload), allow_plan=False)


def test_semantic_schema_separates_requested_tasks_from_executable_tools():
    schema = decision_schema(allowed_tools=[HighLevelToolName.LOCALIZE_CXR], allow_plan=False)
    assert set(schema["properties"]) == {"tasks", "answer"}
    assert schema["additionalProperties"] is False
    assert "classify_image" in schema["$defs"]["SemanticTask"]["enum"]
    assert "HighLevelToolName" not in schema["$defs"]
    assert schema == decision_schema(allowed_tools=[], allow_plan=False)


@pytest.mark.parametrize("text", [
    "TBX_INTERNAL_CONTEXT_JSON={}", "<think>private reasoning", "思考过程：继续分析",
])
def test_private_state_or_unclosed_reasoning_cannot_become_an_answer(text):
    with pytest.raises(ValueError):
        select(Generator({"action": "answer", "answer": text}))


def test_usage_rejects_boolean_and_noninteger_counts():
    decision = select(Generator({"action": "answer", "answer": "25"}, usage={
        "prompt_tokens": True, "completion_tokens": "5",
    }))
    assert decision.prompt_tokens is None and decision.completion_tokens is None


def semantic(*tasks, conditional=(), answer=None):
    payload = {"tasks": [
        {"task": task, "when": "classification_abnormal" if task in conditional else "always"}
        for task in tasks
    ]}
    if answer is not None:
        payload["answer"] = answer
    return payload


@pytest.mark.parametrize(("task", "tool"), [
    ("classify_image", HighLevelToolName.CLASSIFY_CXR),
    ("show_detection_boxes", HighLevelToolName.LOCALIZE_CXR),
    ("locate_within_lungs", HighLevelToolName.ANALYZE_LUNG_ANATOMY),
    ("search_tb_knowledge", HighLevelToolName.SEARCH_TB_KNOWLEDGE),
])
def test_semantic_single_task_dispatches_one_available_tool_without_planning(task, tool):
    generator = Generator(semantic(task))
    decision = select(generator, allowed=[tool])
    assert decision.action == Action.TOOL and decision.tool == tool
    assert decision.tasks == [] and decision.answer is None
    assert len(generator.calls) == 1


def test_semantic_cached_result_becomes_selected_evidence_without_execution():
    generator = Generator(semantic("show_detection_boxes"))
    decision = select(generator, case_context={"localization": {"status": "completed"}})
    assert decision.action == Action.ANSWER and decision.tool is None
    assert decision.evidence == [EvidenceNeed.LOCALIZATION]
    assert len(generator.calls) == 1


@pytest.mark.parametrize("payload", [
    semantic("classify_image", "show_detection_boxes"),
    semantic("show_detection_boxes", conditional=["show_detection_boxes"]),
])
def test_semantic_compound_or_conditional_task_requests_internal_plan(payload):
    generator = Generator(payload)
    decision = select(generator, allowed=list(HighLevelToolName))
    assert decision.action == Action.PLAN and decision.tool is None
    assert [item.model_dump(mode="json") for item in decision.tasks] == payload["tasks"]
    assert len(generator.calls) == 1


def test_semantic_after_plan_advances_past_completed_classification():
    generator = Generator(semantic("classify_image", "show_detection_boxes"))
    decision = select(generator, allowed=[HighLevelToolName.LOCALIZE_CXR], allow_plan=False,
                      case_context={"classification": {"status": "completed", "result": "tb"}})
    assert decision.action == Action.TOOL and decision.tool == HighLevelToolName.LOCALIZE_CXR
    assert len(generator.calls) == 1


def test_semantic_healthy_state_skips_conditional_localization_even_if_tool_available():
    generator = Generator(semantic("classify_image", "show_detection_boxes",
                                   conditional=["show_detection_boxes"]))
    decision = select(generator, allowed=[HighLevelToolName.LOCALIZE_CXR], allow_plan=False,
                      case_context={"classification": {"status": "completed", "result": "healthy"}})
    assert decision.action == Action.ANSWER and decision.tool is None
    assert decision.evidence == [EvidenceNeed.CLASSIFICATION]
    assert len(generator.calls) == 1


def test_semantic_missing_tool_preserves_evidence_request_without_execution_authority():
    generator = Generator(semantic("classify_image"))
    decision = select(generator, allowed=[], case_context={"image_loaded": False})
    assert decision.action == Action.ANSWER and decision.tool is None
    assert decision.evidence == [EvidenceNeed.CLASSIFICATION]
    assert decision.answer is None
    assert len(generator.calls) == 1


def test_semantic_chat_includes_answer_in_same_model_call():
    generator = Generator(semantic("general_chat", answer="25"))
    decision = select(generator, allowed=list(HighLevelToolName))
    assert decision.action == Action.ANSWER and decision.answer == "25"
    assert decision.evidence == [] and decision.tool is None
    assert len(generator.calls) == 1


@pytest.mark.parametrize("payload", [
    {**semantic("classify_image"), "case_id": "another-user-case"},
    {**semantic("search_tb_knowledge"), "query": "model override"},
    semantic("execute_shell"),
    semantic("general_chat"),
])
def test_semantic_wire_rejects_extra_authority_unknown_tasks_or_answerless_chat(payload):
    with pytest.raises(ValueError):
        select(Generator(payload), allowed=list(HighLevelToolName))
