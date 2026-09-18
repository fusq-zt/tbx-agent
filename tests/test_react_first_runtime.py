"""Public-runtime execution contracts with synthetic geometry and scripted decisions.

These tests cover orchestration, not real-model language understanding or medical
accuracy; the live dialogue harness evaluates the former separately.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import replace

import pytest
from test_plan_react_runtime import _run, _settings, _upload

from tbx_agent.evaluation.conversation_fixtures import ConversationAnatomy, ConversationVision
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import TaskGoal, parse_task_spec


class Decisions:
    backend_id = "scripted-react-first"
    model = "scripted-decisions"
    model_digest = None

    def __init__(self, *actions):
        self.actions = deque(actions)
        self.requests = []

    def complete_structured(self, **kwargs):
        self.requests.append(kwargs)
        if kwargs["schema_name"] != "tbx_react_decision":
            raise RuntimeError("legacy model protocol unavailable in this fixture")
        if not self.actions:
            raise AssertionError("unexpected extra decision call")
        payload = self.actions.popleft()
        return (payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)), {
            "prompt_tokens": 19, "completion_tokens": 7,
        }


def tool(name):
    return {"action": "tool", "tool": name}


def answer(*evidence, focus="general", text=None):
    result = {"action": "answer", "evidence": list(evidence), "answer_focus": focus}
    if text is not None:
        result["answer"] = text
    return result


def plan(*tasks, conditional=()):
    return {"action": "plan", "tasks": [
        {"task": task, "when": "classification_abnormal" if task in conditional else "always"}
        for task in tasks
    ]}


def context(request):
    content = request["messages"][0]["content"]
    return json.loads(content.split("TBX_INTERNAL_CONTEXT_JSON=", 1)[1])


@pytest.fixture
def services(tmp_path):
    created = []

    def create(*, healthy=False, empty=False, fail=False):
        settings = replace(_settings(tmp_path / str(len(created))), anatomy_backend="xrv_pspnet",
                           contour_refinement_backend="none")
        vision = ConversationVision(settings.fusion_policy(), settings.rank03_config(),
                                    healthy=healthy, empty=empty)
        service = TBXAgentService(settings, vision_backend=vision,
                                  anatomy_backend=ConversationAnatomy(fail=fail))
        created.append(service)
        return service

    yield create
    for service in created:
        service.close()


def run(service, generator, query, case=None, thread="conversation", *, allow_fallback=False):
    result = _run(service, generator, query=query, case_id=case.case_id if case else None,
                  thread_id=thread)
    if not allow_fallback:
        assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert not generator.actions, "the runtime did not reach the scripted terminal decision"
    return result


def test_simple_chat_uses_one_decision_and_never_plans(services):
    service = services()
    generator = Decisions(answer(text="25"))
    result = run(service, generator, "12加13呢")
    assert result.response.summary == "25"
    assert result.execution_plan["tool_names"] == []
    assert result.execution_plan["plan_metadata"]["planning_used"] is False
    assert [request["schema_name"] for request in generator.requests] == ["tbx_react_decision"]
    assert context(generator.requests[0])["commitments"] is None


def test_classification_and_cached_status_do_not_force_planning_or_rerun(services):
    service = services()
    case = _upload(service)
    first = run(service, Decisions(tool("classify_cxr"), answer("classification")),
                "筛查一下", case)
    before = service.vision.call_count
    status = run(service, Decisions(answer(focus="case_status")), "分类跑过了吗，先别重跑", case)
    assert first.execution_plan["tool_names"] == ["classify_cxr"]
    assert first.execution_plan["plan_metadata"]["planning_used"] is False
    assert status.execution_plan["tool_names"] == []
    assert service.vision.call_count == before
    assert "结核类" in status.response.summary
    assert "肺野分割未运行" in status.response.summary


def test_cached_boxes_then_lung_followup_runs_anatomy_once(services):
    service = services()
    case = _upload(service)
    run(service, Decisions(tool("localize_cxr"), answer("localization")), "圈出来", case)
    generator = Decisions(tool("analyze_lung_anatomy"), answer("lung_anatomy"))
    localized = run(service, generator, "位于肺野哪里", case)
    assert "analyze_lung_anatomy" in context(generator.requests[0])["allowed_tools_this_step"]
    assert localized.execution_plan["tool_names"] == ["analyze_lung_anatomy"]
    assert "左中肺野" in localized.response.summary
    again = run(service, Decisions(answer("lung_anatomy")), "再说一次属于哪个肺区", case)
    assert again.execution_plan["tool_names"] == []
    assert "左中肺野" in again.response.summary
    assert service.anatomy.call_count == 1
    assert service.vision.localization_call_count == 1


def test_healthy_conditional_plan_skips_boxes_then_explicit_new_turn_can_draw(services):
    service = services(healthy=True)
    case = _upload(service)
    generator = Decisions(
        plan("classify_image", "show_detection_boxes", conditional=("show_detection_boxes",)),
        tool("classify_cxr"), answer("classification"),
    )
    result = run(service, generator, "先分类，只有异常才画框", case)
    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert result.execution_plan["plan_metadata"]["planning_used"] is True
    assert "localize_cxr" not in context(generator.requests[-1])["allowed_tools_this_step"]
    assert service.vision.localization_call_count == 0
    followup = run(service, Decisions(tool("localize_cxr"), answer("localization")),
                   "现在不管类别都标记候选位置", case)
    assert followup.execution_plan["tool_names"] == ["localize_cxr"]


def test_premature_answers_cannot_drop_explicit_plan_obligations(services):
    service = services()
    case = _upload(service)
    generator = Decisions(
        plan("classify_image", "show_detection_boxes"),
        answer(text="已全部完成。"), answer(text="已全部完成。"),
        answer("classification", "localization"),
    )
    result = run(service, generator, "分类并标记位置", case)
    assert result.execution_plan["tool_names"] == ["classify_cxr", "localize_cxr"]
    assert "候选区域" in result.response.summary
    assert "已全部完成" not in result.response.summary
    assert result.execution_plan["unfinished_evidence"] == []


def test_plan_omission_does_not_hide_state_valid_anatomy_tool(services):
    service = services()
    case = _upload(service)
    generator = Decisions(
        plan("classify_image", "show_detection_boxes"), tool("classify_cxr"),
        tool("localize_cxr"), tool("analyze_lung_anatomy"), answer("lung_anatomy"),
    )
    result = run(service, generator, "分类、标框，并说明候选区位于哪个肺野", case)
    assert "analyze_lung_anatomy" in context(generator.requests[3])["allowed_tools_this_step"]
    assert result.execution_plan["tool_names"] == [
        "classify_cxr", "localize_cxr", "analyze_lung_anatomy",
    ]
    assert "左中肺野" in result.response.summary


def test_model_cannot_force_a_condition_blocked_tool(services):
    service = services(healthy=True)
    case = _upload(service)
    result = run(service, Decisions(
        plan("classify_image", "show_detection_boxes", conditional=("show_detection_boxes",)),
        tool("classify_cxr"), tool("localize_cxr"),
    ), "仅异常时定位", case)
    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert service.vision.localization_call_count == 0


def test_lobe_followup_explains_boundary_instead_of_repeating_field_only(services):
    service = services()
    case = _upload(service)
    run(service, Decisions(tool("localize_cxr"), tool("analyze_lung_anatomy"),
                           answer("lung_anatomy")), "位于哪个肺野", case)
    result = run(service, Decisions(answer("lung_anatomy", focus="lung_lobe_limit")),
                 "能明确是哪个肺叶吗", case)
    assert result.execution_plan["tool_names"] == []
    assert "不能确定解剖学肺叶" in result.response.summary
    assert "左中肺野" in result.response.summary


def test_no_boxes_does_not_turn_exclusion_question_into_classification(services):
    service = services(empty=True)
    case = _upload(service)
    run(service, Decisions(tool("localize_cxr"), answer("localization")), "标可疑区域", case)
    result = run(service, Decisions(answer("localization", focus="screening_limit")),
                 "那能完全排除结核吗", case)
    assert result.execution_plan["tool_names"] == []
    assert "不能仅据此确诊或排除结核" in result.response.summary
    assert "未检出候选框也不等于没有结核" in result.response.summary


def test_failed_anatomy_preserves_partial_evidence_and_status_never_retries(services):
    service = services(fail=True)
    case = _upload(service)
    generator = Decisions(tool("localize_cxr"), tool("analyze_lung_anatomy"))
    failed = run(service, generator, "标出可疑区域并说明肺野位置", case)
    attempts = service.anatomy.call_count
    status = run(service, Decisions(answer(focus="case_status")), "刚才肺野分析成功了吗", case)
    assert "候选区域" in failed.response.summary
    assert "未完成" in failed.response.summary and "无法可靠说明" in failed.response.summary
    assert "左中肺野" not in failed.response.summary
    assert len(generator.requests) == 2
    assert status.execution_plan["tool_names"] == []
    assert service.anatomy.call_count == attempts
    assert "肺野分割运行失败" in status.response.summary


def test_missing_image_cannot_produce_claimed_classification(services):
    service = services()
    result = run(service, Decisions(answer("classification", text="分类结果正常。")),
                 "帮我筛查这张片子")
    assert result.execution_plan["tool_names"] == []
    assert "上传胸片" in result.response.summary
    assert "分类结果正常" not in result.response.summary


@pytest.mark.parametrize(("focus", "query", "expected", "forbidden"), [
    ("image_quality", "图像质量可以吗", "基础输入", "质量完全合格"),
    ("prior_comparison", "比上次有没有好转", "没有接入", "病灶明显缩小"),
])
def test_state_projection_overrides_fabricated_quality_and_comparison(
    services, focus, query, expected, forbidden,
):
    service = services()
    case = _upload(service)
    result = run(service, Decisions(answer(focus=focus, text=forbidden)), query, case)
    assert result.execution_plan["tool_names"] == []
    assert expected in result.response.summary
    assert forbidden not in result.response.summary


@pytest.mark.parametrize("query", [
    "帮我筛查一下这张胸片，看看有没有结核可疑。", "做胸片分类",
])
def test_bad_decision_json_uses_recorded_rule_fallback(services, query):
    service = services()
    case = _upload(service)
    generator = Decisions("not json")
    result = run(service, generator, query, case, allow_fallback=True)
    assert generator.requests[0]["schema_name"] == "tbx_react_decision"
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True
    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert "结核类" in result.response.summary


@pytest.mark.parametrize(("query", "classify"), [
    ("做胸片分类", True),
    ("请进行胸片分类", True),
    ("帮我分类筛查一下这张胸片", True),
    ("不要做胸片分类", False),
    ("不做胸片分类", False),
    ("不进行胸片分类", False),
    ("先别进行胸片分类", False),
    ("不用对这张胸片做分类筛查", False),
    ("不运行分类，只介绍做胸片分类的流程", False),
    ("做胸片分类了吗？", False),
    ("胸片分类跑过了吗？先别重跑", False),
    ("进行胸片分类是否完成？", False),
    ("做胸片分类，不用标位置", True),
])
def test_short_classification_fallback_respects_negation_and_status(query, classify):
    assert (TaskGoal.SCREEN_CLASSIFICATION in parse_task_spec(query).task_goals) is classify


def wire_tasks(*tasks):
    return {"tasks": [{"task": task, "when": "always"} for task in tasks]}


def test_explicit_global_prohibition_blocks_mistaken_semantic_anatomy_request(services):
    service = services()
    case = _upload(service)
    run(service, Decisions(wire_tasks("show_detection_boxes"), wire_tasks("show_detection_boxes")),
        "先标记候选区域", case)
    before = (service.vision.call_count, service.vision.localization_call_count)
    generator = Decisions(wire_tasks("locate_within_lungs"))
    result = run(service, generator, "现在不要运行。只想了解肺野分割的状态。", case)
    assert result.execution_plan["tool_names"] == []
    assert service.anatomy.call_count == 0
    assert (service.vision.call_count, service.vision.localization_call_count) == before
    assert context(generator.requests[0])["allowed_tools_this_step"] == []
    assert "未运行" in result.response.summary


def test_scoped_prohibitions_leave_requested_localization_available(services):
    service = services()
    case = _upload(service)
    generator = Decisions(wire_tasks("show_detection_boxes"), wire_tasks("show_detection_boxes"))
    result = run(service, generator,
                 "不要运行分类，也不要运行肺野分割；只标出候选区域。", case)
    assert result.execution_plan["tool_names"] == ["localize_cxr"]
    assert service.vision.localization_call_count == 1
    assert service.vision.call_count == 0 and service.anatomy.call_count == 0
    allowed = context(generator.requests[0])["allowed_tools_this_step"]
    assert "localize_cxr" in allowed
    assert "classify_cxr" not in allowed and "analyze_lung_anatomy" not in allowed


def test_bad_json_rule_fallback_cannot_bypass_explicit_global_prohibition(services):
    service = services()
    case = _upload(service)
    result = run(service, Decisions("not json"),
                 "帮我筛查一下这张胸片，看看有没有结核可疑；现在不要运行。", case,
                 allow_fallback=True)
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True
    assert result.execution_plan["tool_names"] == []
    assert service.vision.call_count == 0
    assert service.vision.localization_call_count == 0
    assert service.anatomy.call_count == 0
    assert "不执行" in result.response.summary
