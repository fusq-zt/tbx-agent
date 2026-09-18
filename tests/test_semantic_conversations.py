"""Execution contracts with explicit model intents; live understanding is evaluated separately."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import replace

import pytest
from test_plan_react_runtime import (
    _run,
    _settings,
    _upload,
)

from tbx_agent.evaluation.conversation_fixtures import ConversationAnatomy, ConversationVision
from tbx_agent.service import TBXAgentService


def _plan(*tasks, conditional=()):
    return {"action": "plan", "tasks": [{"task": task, "when": (
        "classification_abnormal" if task in conditional else "always"
    )} for task in tasks]}


def _answer(*evidence, focus="general", text=None):
    return {"action": "answer", "answer_focus": focus,
            "evidence": list(evidence), "answer": text}


def _tool(name):
    return {"action": "tool", "tool": name}


class _ScriptedReActGenerator:
    """Script next-action decisions, never use schema failure to reach rule fallback."""

    backend_id = "scripted-react-v4"
    model = "scripted-react-v4"
    model_digest = None

    def __init__(self, actions):
        self.actions = deque(actions)
        self.requests = []

    def complete_structured(self, **kwargs):
        self.requests.append(kwargs)
        assert kwargs["schema_name"] == "tbx_react_decision"
        assert self.actions, "unexpected extra decision"
        return json.dumps(self.actions.popleft(), ensure_ascii=False), {
            "prompt_tokens": 17, "completion_tokens": 7,
        }


def _assert_model_decisions(generator, *results):
    assert not generator.actions
    assert all(request["schema_name"] == "tbx_react_decision" for request in generator.requests)
    assert all(not result.execution_plan["plan_metadata"]["rule_fallback_used"]
               for result in results)


def _service(tmp_path, *, healthy=False, fail=False):
    settings = replace(_settings(tmp_path), anatomy_backend="xrv_pspnet",
                       contour_refinement_backend="none")
    anatomy = ConversationAnatomy(fail=fail)
    vision = ConversationVision(settings.fusion_policy(), settings.rank03_config(), healthy=healthy)
    return TBXAgentService(settings, vision_backend=vision, anatomy_backend=anatomy)


def test_cached_boxes_cannot_satisfy_new_lung_field_request(tmp_path):
    service = _service(tmp_path)
    case = _upload(service)
    generator = _ScriptedReActGenerator([
        _tool("localize_cxr"), _answer("localization"),
        # A premature answer referring to missing lung evidence must acquire it.
        _answer("lung_anatomy", text="候选区域在图像右侧中部。"),
        _answer("lung_anatomy"), _answer("lung_anatomy"),
    ])
    initial = _run(service, generator, query="圈出可疑区域",
                   case_id=case.case_id, thread_id="space")
    assert initial.execution_plan["tool_names"] == ["localize_cxr"]
    located = _run(service, generator, query="位于肺野哪里",
                   case_id=case.case_id, thread_id="space")
    assert located.execution_plan["tool_names"] == ["analyze_lung_anatomy"]
    assert "左中肺野" in located.response.summary
    assert located.execution_plan["plan_metadata"]["planning_used"] is False
    again = _run(service, generator, query="再说一遍在哪个肺野", case_id=case.case_id,
                 thread_id="space")
    assert again.execution_plan["tool_names"] == []
    assert "左中肺野" in again.response.summary
    assert service.anatomy.call_count == 1
    assert service.vision.localization_call_count == 1
    _assert_model_decisions(generator, initial, located, again)
    service.close()


def test_conditional_request_and_explicit_followup_keep_distinct_scope(tmp_path):
    service = _service(tmp_path, healthy=True)
    case = _upload(service)
    generator = _ScriptedReActGenerator([
        _plan("classify_image", "show_detection_boxes", conditional=("show_detection_boxes",)),
        _answer("classification", text="已完成。"), _answer("classification"),
        _tool("localize_cxr"), _answer("localization"),
    ])
    screened = _run(service, generator, query="正常就不画框", case_id=case.case_id,
                    thread_id="conditional")
    assert screened.execution_plan["tool_names"] == ["classify_cxr"]
    located = _run(service, generator, query="现在无论类别都画框", case_id=case.case_id,
                  thread_id="conditional")
    assert located.execution_plan["tool_names"] == ["localize_cxr"]
    assert screened.execution_plan["plan_metadata"]["planning_used"] is True
    assert located.execution_plan["plan_metadata"]["planning_used"] is False
    _assert_model_decisions(generator, screened, located)
    service.close()


def test_current_status_intent_does_not_run_missing_anatomy(tmp_path):
    service = _service(tmp_path)
    case = _upload(service)
    generator = _ScriptedReActGenerator([_answer(focus="case_status")])
    result = _run(service, generator, query="分割做了吗？只告诉我进度", case_id=case.case_id,
                  thread_id="status")
    assert result.execution_plan["tool_names"] == []
    assert service.anatomy.call_count == 0
    assert "肺野分割未运行" in result.response.summary
    _assert_model_decisions(generator, result)
    service.close()


@pytest.mark.parametrize(("focus", "query", "required", "forbidden"), [
    ("prior_comparison", "与我以前那张相比有什么变化？", "没有接入", "新的病变"),
    ("image_quality", "这张图片清晰吗，质量合格吗？", "基础输入", "质量合格"),
])
def test_state_answers_are_not_left_to_free_generation(
    tmp_path, focus, query, required, forbidden,
):
    service = _service(tmp_path)
    case = _upload(service)
    generator = _ScriptedReActGenerator([_answer(focus=focus)])
    result = _run(service, generator, query=query, case_id=case.case_id, thread_id="metadata")
    assert result.execution_plan["tool_names"] == []
    assert required in result.response.summary
    assert forbidden not in result.response.summary
    assert result.response.narrator_generation_invoked is False
    _assert_model_decisions(generator, result)
    service.close()
