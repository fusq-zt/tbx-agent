from __future__ import annotations

import io
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.capability_answer import TBX_CAPABILITY_ANSWER
from tbx_agent.config import Settings
from tbx_agent.schemas import ClassifierClass, ResponseKind
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import TaskGoal, parse_task_spec
from tbx_agent.vision import MockRank03Backend
from tbx_agent.vision.anatomy import (
    AnatomyBackendUnavailable,
    AnatomyRuntimeProbe,
    build_generation_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        max_agent_steps=8,
        max_tool_calls=4,
        max_expensive_vision_calls=3,
        agent_tool_cost_budget=12,
    )


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), color=(48, 68, 88)).save(output, format="PNG")
    return output.getvalue()


def _upload(service: TBXAgentService):
    return service.assess_cxr(
        _png(),
        user_id="react-user",
        owner_scope="tenant:react",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]


def _plan(*needs: str, answer_focus: str = "general") -> dict:
    return {
        "goal": "完成用户当前请求",
        "answer_focus": answer_focus,
        "steps": [
            {
                "objective": f"获取或整合 {need} 证据",
                "evidence_need": need,
            }
            for need in needs
        ],
    }


def _conditional_screening_plan() -> dict:
    return {
        "goal": "先分类，并仅在异常时定位和检索下一步检查",
        "steps": [
            {
                "objective": "胸片分类",
                "evidence_need": "classification",
                "condition": "always",
            },
            {
                "objective": "若分类异常，定位主要候选区域",
                "evidence_need": "localization",
                "condition": "classification_abnormal",
            },
            {
                "objective": "若分类异常，检索下一步检查",
                "evidence_need": "tb_knowledge",
                "condition": "classification_abnormal",
            },
        ],
    }


def _tool(name: str) -> dict:
    return {"tool": name, "direct_answer": None}


def _answer(
    text: str, *, evidence: list[str] | None = None, answer_focus: str | None = None,
) -> dict:
    if evidence is not None or answer_focus is not None:
        return {"action": "answer", "answer": text, "evidence": evidence or [],
                "answer_focus": answer_focus or "general"}
    return {"tool": None, "direct_answer": text}


def _internal_context(request: dict) -> dict:
    """Read the private ReAct state without assuming it is the user message."""

    prefix = "TBX_INTERNAL_CONTEXT_JSON="
    matches = [
        message["content"]
        for message in request["messages"]
        if message.get("role") == "system" and prefix in message.get("content", "")
    ]
    assert len(matches) == 1
    encoded = matches[0]
    return json.loads(encoded[encoded.index(prefix) + len(prefix) :])


class _ScriptedPlanReactGenerator:
    """Emit v4 decisions from the existing evidence-oriented test fixtures.

    Single-purpose fixtures go straight to a tool/answer. Only a compound or
    conditional fixture emits a plan action. ``plans`` is retained as a fixture
    input for tests importing this helper; it is never an extra planner call.
    """

    backend_id = "scripted-react-first"
    model = "scripted-react-first-model"
    model_digest = None

    def __init__(
        self,
        *,
        plans: list[dict],
        actions: list[dict],
        general_answers: list[str] | None = None,
    ) -> None:
        self.plans = deque(plans)
        self.actions = deque(actions)
        self.general_answers = deque(general_answers or [])
        self.plan_requests: list[dict] = []
        self.action_requests: list[dict] = []
        self.general_requests: list[dict] = []
        self.decision_requests: list[dict] = []
        self._started = False
        self._fixture = plans[0] if plans else None

    def _decision_payload(self, request: dict) -> dict:
        self.decision_requests.append(request)
        if not self._started:
            self._started = True
            if self._fixture is None:
                raise RuntimeError("synthetic decision provider failure")
            if "steps" not in self._fixture:
                return self._fixture  # Explicit malformed-response outage test.
            steps = [step for step in self._fixture["steps"]
                     if step["evidence_need"] != "none"]
            if len(steps) > 1 or any(
                step.get("condition") == "classification_abnormal" for step in steps
            ):
                self.plan_requests.append(request)
                task_names = {
                    "classification": "classify_image",
                    "localization": "show_detection_boxes",
                    "lung_anatomy": "locate_within_lungs",
                    "tb_knowledge": "search_tb_knowledge",
                }
                return {"action": "plan", "tasks": [
                    {"task": task_names[step["evidence_need"]],
                     "when": step.get("condition", "always")}
                    for step in steps
                ]}
        self.action_requests.append(request)
        if not self.actions:
            raise AssertionError("unexpected extra ReAct decision")
        action = self.actions.popleft()
        if "action" in action:
            return action
        if action.get("tool"):
            return {"action": "tool", "tool": action["tool"]}
        context = _internal_context(request)
        classification = context["case_state"].get("classification", {})
        evidence = [
            step["evidence_need"] for step in self._fixture["steps"]
            if step["evidence_need"] != "none"
            and not (step.get("condition") == "classification_abnormal"
                     and classification.get("result") == "healthy")
        ]
        return {
            "action": "answer",
            "answer_focus": self._fixture.get("answer_focus", "general"),
            "evidence": evidence,
            "answer": action["direct_answer"],
        }

    def complete_structured(self, **kwargs):
        schema_name = kwargs["schema_name"]
        if schema_name == "tbx_react_decision":
            payload = self._decision_payload(kwargs)
        elif schema_name == "tbx_agent_tool_selection":
            self.action_requests.append(kwargs)
            if not self.actions:
                raise AssertionError("unexpected extra ReAct step")
            payload = self.actions.popleft()
        elif schema_name == "tbx_general_chat_answer":
            self.general_requests.append(kwargs)
            if not self.general_answers:
                raise AssertionError("unexpected general-answer recovery call")
            payload = {"answer": self.general_answers.popleft()}
        else:
            raise AssertionError(f"unexpected schema: {schema_name}")
        return json.dumps(payload, ensure_ascii=False), {
            "prompt_tokens": 17,
            "completion_tokens": 7,
        }


class _FixedClassMockBackend(MockRank03Backend):
    predicted_class: ClassifierClass

    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        probabilities = {
            ClassifierClass.HEALTHY: {
                "healthy": 0.90,
                "sick_non_tb": 0.06,
                "tb": 0.04,
            },
            ClassifierClass.SICK_NON_TB: {
                "healthy": 0.06,
                "sick_non_tb": 0.90,
                "tb": 0.04,
            },
        }[self.predicted_class]
        ranked = sorted(probabilities.values(), reverse=True)
        return evidence.model_copy(
            update={
                "class_probabilities": probabilities,
                "top1_score": ranked[0],
                "top2_score": ranked[1],
                "top1_top2_margin": ranked[0] - ranked[1],
                "predicted_class": self.predicted_class,
                "classifier_argmax_tied": False,
                "classifier_flagged": self.predicted_class == ClassifierClass.TB,
            }
        )


class _HealthyMockBackend(_FixedClassMockBackend):
    predicted_class = ClassifierClass.HEALTHY


class _NonTBAbnormalMockBackend(_FixedClassMockBackend):
    predicted_class = ClassifierClass.SICK_NON_TB


class _FailingAnatomyBackend:
    backend_id = "failing-anatomy-runtime-test"
    loaded = True

    @staticmethod
    def probe_runtime(*, load: bool = False) -> AnatomyRuntimeProbe:
        return AnatomyRuntimeProbe(
            backend_id="failing-anatomy-runtime-test",
            loaded=True,
            available="yes",
            detail="test backend fails only when invoked",
        )

    def generation_key_for(self, image_sha256: str) -> str:
        return build_generation_key(
            image_sha256=image_sha256,
            model_weight_sha256="a" * 64,
            preprocessing_id="failing-anatomy-test-v1",
            policy_id="paired-lung-qc-v1",
            backend_id=self.backend_id,
            parameters={"failure_mode": "backend_unavailable"},
        )

    def infer(self, *, case_id, image):
        raise AnatomyBackendUnavailable("synthetic anatomy worker failure")


def _run(
    service: TBXAgentService,
    generator: _ScriptedPlanReactGenerator,
    *,
    query: str,
    case_id: str | None,
    thread_id: str,
):
    return service.respond_with_controller(
        message=query,
        thread_id=thread_id,
        user_id="react-user",
        owner_scope="tenant:react",
        case_id=case_id,
        generator=generator,
    )


@pytest.mark.parametrize(
    ("query", "answer"),
    (
        ("1+1等于多少？", "2"),
        ("人每天应该摄入多少糖分？", "一般健康建议需要结合年龄和健康状况。"),
    ),
)
def test_general_questions_are_direct_answers_without_tools(
    tmp_path: Path,
    query: str,
    answer: str,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none")],
        actions=[_answer(answer)],
    )

    result = _run(
        service,
        generator,
        query=query,
        case_id=None,
        thread_id=f"general-{len(query)}",
    )

    assert result.trace.task_spec.task_goals == [TaskGoal.GENERAL_CHAT]
    assert result.execution_plan["tool_names"] == []
    assert result.tool_results == []
    assert result.response.summary == answer
    assert generator.plan_requests == []
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False


@pytest.mark.parametrize(
    ("query", "scope", "subtopic", "population", "scenario_tags"),
    (
        (
            "结核病会传染吗？",
            "infection_control",
            "infection_control",
            [],
            [],
        ),
        (
            "孕妇怀疑肺结核时应该做什么检查？",
            "special_population",
            "special_population_testing",
            ["pregnant_people"],
            [],
        ),
        (
            "痰片没查到菌是不是就能排除结核？",
            "diagnostic_testing",
            "negative_test_interpretation",
            [],
            ["test_smear"],
        ),
        (
            "WHO 是否规定所有肺结核患者都必须住院？",
            "treatment_education",
            "care_setting",
            [],
            ["care_universal_hospitalization"],
        ),
    ),
)
def test_tb_questions_use_one_query_only_knowledge_action(
    tmp_path: Path,
    query: str,
    scope: str,
    subtopic: str,
    population: list[str],
    scenario_tags: list[str],
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("tb_knowledge", "none")],
        actions=[
            _tool("search_tb_knowledge"),
            _answer("已整合检索观察回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=query,
        case_id=None,
        thread_id=f"knowledge-{subtopic}",
    )

    assert result.trace.task_spec.task_goals == [TaskGoal.SEARCH_TB_KNOWLEDGE]
    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert len(result.tool_results) == 1
    receipt = result.tool_results[0].receipt
    assert receipt.tool_name == "search_tb_knowledge"
    assert receipt.model_tool_name == "search_tb_knowledge"
    assert receipt.resolved_guideline_scope == scope
    assert receipt.resolved_guideline_subtopic == subtopic
    assert receipt.resolved_population == population
    assert receipt.resolved_scenario_tags == scenario_tags
    assert generator.plan_requests == []
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False


def test_cached_classification_why_is_answered_without_another_tool(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    first_generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "none")],
        actions=[
            _tool("classify_cxr"),
            _answer("模型已完成胸片分类。"),
        ],
    )
    first = _run(
        service,
        first_generator,
        query="这张胸片有没有结核病？",
        case_id=case.case_id,
        thread_id="cached-classification",
    )
    classifier_calls = service.vision.call_count

    followup_generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", answer_focus="classification_rationale")],
        actions=[_answer("因为胸片分类模型将结果归入当前训练类别。")],
    )
    followup = _run(
        service,
        followup_generator,
        query="为什么这样分类？",
        case_id=case.case_id,
        thread_id="cached-classification",
    )

    assert first.execution_plan["tool_names"] == ["classify_cxr"]
    assert followup.execution_plan["tool_names"] == []
    assert followup.tool_results == []
    assert followup.receipt is None
    assert service.vision.call_count == classifier_calls == 1
    action_payload = _internal_context(followup_generator.action_requests[0])
    assert action_payload["case_state"]["classification"]["status"] == "completed"
    current_message = followup_generator.action_requests[0]["messages"][-1]
    assert current_message["role"] == "user"
    assert json.dumps("为什么这样分类？", ensure_ascii=False) in current_message["content"]
    assert followup_generator.plan_requests == []
    assert followup.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    action_messages = followup_generator.action_requests[0]["messages"]
    assert sum(item["role"] == "system" for item in action_messages) == 1
    assert all(
        current["role"] != following["role"]
        for current, following in zip(action_messages, action_messages[1:], strict=False)
    )
    assert "%" not in followup.response.summary


def test_cached_localization_is_answered_without_detector_rerun(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    first = _ScriptedPlanReactGenerator(
        plans=[_plan("localization", "none")],
        actions=[_tool("localize_cxr"), _answer("定位完成。")],
    )
    initial = _run(
        service,
        first,
        query="候选区域在哪里？",
        case_id=case.case_id,
        thread_id="cached-localization",
    )
    detector_calls = service.vision.localization_call_count
    followup_generator = _ScriptedPlanReactGenerator(
        plans=[_plan("localization")],
        actions=[_answer("候选区域仍显示在当前胸片的上部区域。")],
    )
    followup = _run(
        service,
        followup_generator,
        query="刚才的候选区域在哪？",
        case_id=case.case_id,
        thread_id="cached-localization",
    )

    assert initial.execution_plan["tool_names"] == ["localize_cxr"]
    assert followup.execution_plan["tool_names"] == []
    assert followup.tool_results == []
    assert service.vision.localization_call_count == detector_calls == 1
    payload = _internal_context(followup_generator.action_requests[0])
    assert payload["case_state"]["localization"]["status"] in {
        "completed",
        "completed_no_detection",
    }


@pytest.mark.parametrize("query", [
    "把这张片子里模型认为可疑的区域标出来",
    "把这张片子里模型认为可疑的区域标出来。你听不懂吗",
    "那你把刚才提到的地方圈给我看看",
    "Please draw boxes around the suspicious areas in this image.",
])
def test_model_localization_followup_never_passes_through_keyword_router(
    tmp_path: Path, monkeypatch, query: str,
) -> None:
    import tbx_agent.agent_runtime as runtime

    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    _run(service, _ScriptedPlanReactGenerator(
        plans=[_plan("classification")],
        actions=[_tool("classify_cxr"), _answer("分类完成。")],
    ), query="帮我筛查一下这张胸片，看看有没有结核可疑。",
        case_id=case.case_id, thread_id="natural-localization")

    def forbidden(*args, **kwargs):
        raise AssertionError("successful model plan reached the keyword intent router")

    monkeypatch.setattr(runtime, "parse_task_spec", forbidden)
    monkeypatch.setattr(runtime, "projection_task_goals", forbidden)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("localization")],
        # Even an unhelpful ReAct answer cannot discharge a localization obligation.
        actions=[_answer("模型识别为结核类，建议进一步检查。"), _answer("定位完成。")],
    )
    result = _run(service, generator, query=query,
                  case_id=case.case_id, thread_id="natural-localization")
    assert result.execution_plan["tool_names"] == ["localize_cxr"]
    assert result.execution_plan["plan_metadata"]["intent_authority"] == "model"
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert result.execution_plan["unfinished_evidence"] == []
    assert service.vision.call_count == 1
    assert service.vision.localization_call_count == 1
    # Availability is determined by actual state, not an earlier intent gate.
    assert set(_internal_context(generator.action_requests[0])["allowed_tools_this_step"]) == {
        "localize_cxr", "search_tb_knowledge",
    }


@pytest.mark.parametrize("failure", ["invalid", "provider", "absent"])
def test_original_localization_request_has_explicit_rule_outage_fallback(
    tmp_path: Path, failure: str,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = None if failure == "absent" else _ScriptedPlanReactGenerator(
        plans=[{"unexpected": True}] if failure == "invalid" else [],
        actions=[_tool("localize_cxr"), _answer("定位完成。")],
    )
    result = _run(service, generator,
                  query="把这张片子里模型认为可疑的区域标出来",
                  case_id=case.case_id, thread_id="fallback-localization")
    assert result.execution_plan["tool_names"] == ["localize_cxr"]
    assert result.execution_plan["plan_metadata"]["intent_authority"] == "rule_fallback"
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True


def test_budget_exhaustion_cannot_hide_unfinished_localization(tmp_path: Path) -> None:
    service = TBXAgentService(replace(_settings(tmp_path), max_tool_calls=1))
    case = _upload(service)
    result = _run(service, _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "localization")],
        actions=[_tool("classify_cxr"), _tool("localize_cxr")],
    ), query="先筛查，再标出可疑区域", case_id=case.case_id, thread_id="budget-partial")
    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert result.execution_plan["unfinished_evidence"] == ["localization"]
    assert "本轮尚未完成：候选区域标注" in result.response.summary


def test_upload_quality_question_is_zero_tool(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none")],
        actions=[_answer("当前上传文件已通过基础输入可用性检查。")],
    )

    result = _run(
        service,
        generator,
        query="图像质量有问题吗？",
        case_id=case.case_id,
        thread_id="quality-zero-tool",
    )

    assert result.execution_plan["tool_names"] == []
    assert result.tool_results == []
    assert service.vision.call_count == 0
    assert service.vision.localization_call_count == 0
    payload = _internal_context(generator.action_requests[0])
    assert payload["case_state"]["quality_check"]["status"] in {"passed", "warning"}


def test_compound_request_calls_one_tool_per_observed_react_step(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "localization", "tb_knowledge", "none")],
        actions=[
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已根据本轮观察完成回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query="判断这张胸片，标出候选位置，并说明下一步检查。",
        case_id=case.case_id,
        thread_id="compound-observe-replan",
    )

    assert result.execution_plan["source"] == "plan_react"
    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.tool_name for item in result.tool_results] == [
        "classify_current_cxr",
        "localize_current_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.model_tool_name for item in result.tool_results] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    react_steps = result.execution_plan["react_steps"]
    assert [step["outcome"] for step in react_steps] == [
        "tool_call",
        "tool_call",
        "tool_call",
        "answer",
    ]
    assert all(
        (step["tool_name"] is None) != (step["outcome"] == "tool_call")
        for step in react_steps
    )
    observation_counts = [
        len(_internal_context(call)["observations"])
        for call in generator.action_requests
    ]
    assert observation_counts == [0, 1, 2, 3]
    assert service.vision.call_count == 1
    assert service.vision.localization_call_count == 1
    assert result.response.citations


def test_conditional_screening_skips_downstream_tools_after_healthy_observation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _HealthyMockBackend(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_conditional_screening_plan()],
        actions=[
            _tool("classify_cxr"),
            _answer("模型更倾向于健康类，因此没有运行候选区域定位。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=(
            "判断这张胸片；如果分类异常，请标出候选区域，"
            "并说明胸片筛查异常后下一步做什么检查。"
        ),
        case_id=case.case_id,
        thread_id="conditional-healthy",
    )

    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert backend.call_count == 1
    assert backend.localization_call_count == 0
    final_steps = result.execution_plan["final_plan"]["steps"]
    assert [step["status"] for step in final_steps] == [
        "completed",
        "skipped",
        "skipped",
    ]
    assert [step["condition"] for step in final_steps] == [
        "always",
        "classification_abnormal",
        "classification_abnormal",
    ]
    assert len(generator.action_requests) == 2
    after_classification = _internal_context(generator.action_requests[1])
    assert after_classification["case_state"]["classification"]["result"] == "healthy"
    assert [step["status"] for step in after_classification["commitments"]["steps"]] == [
        "completed",
        "skipped",
        "skipped",
    ]


def test_conditional_screening_runs_downstream_tools_after_abnormal_observation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _NonTBAbnormalMockBackend(
        settings.fusion_policy(),
        settings.rank03_config(),
    )
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_conditional_screening_plan()],
        actions=[
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已整合异常分类、候选区域和下一步检查依据。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=(
            "判断这张胸片；如果分类异常，请标出候选区域，"
            "并说明胸片筛查异常后下一步做什么检查。"
        ),
        case_id=case.case_id,
        thread_id="conditional-abnormal",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert backend.call_count == 1
    assert backend.localization_call_count == 1
    assert [step["status"] for step in result.execution_plan["final_plan"]["steps"]] == [
        "completed",
        "completed",
        "completed",
    ]


def test_outage_rule_fallback_recovers_explicit_abnormal_branch(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _HealthyMockBackend(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    # Only a failed/malformed model decision permits keyword-based recovery.
    generator = _ScriptedPlanReactGenerator(
        plans=[{"invalid_plan": True}],
        actions=[
            _tool("classify_cxr"),
            _answer("模型更倾向于健康类。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=(
            "这张胸片更倾向于哪一类？如果异常，请标出主要候选区域，"
            "然后告诉我这种筛查异常一般下一步需要做什么。"
        ),
        case_id=case.case_id,
        thread_id="conditional-compatibility",
    )

    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert result.execution_plan["plan_metadata"][
        "explicit_abnormal_condition_applied"
    ] is True
    initial_steps = result.execution_plan["initial_plan"]["steps"]
    assert [step["condition"] for step in initial_steps] == [
        "always",
        "classification_abnormal",
        "classification_abnormal",
    ]
    assert [step["objective"] for step in initial_steps] == [
        "胸片分类",
        "若分类异常，定位候选区域",
        "若分类异常，查询下一步检查",
    ]
    assert [step["status"] for step in result.execution_plan["final_plan"]["steps"]] == [
        "completed",
        "skipped",
        "skipped",
    ]


def test_required_evidence_is_executed_when_model_only_answers_directly(
    tmp_path: Path,
) -> None:
    """A model answer cannot satisfy planned classification or retrieval evidence."""

    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "tb_knowledge", "none")],
        actions=[
            _answer("痰涂片阴性后基本可以排除肺结核。"),
            _answer("不需要再查指南，可以直接排除。"),
            _answer("已结合真实模型与指南证据回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query="这个患者胸片模型提示 TB，但是痰涂片阴性，是不是基本可以排除了？",
        case_id=case.case_id,
        thread_id="direct-answer-cannot-bypass-evidence",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.selection_source for item in result.tool_results] == [
        "plan_evidence_fallback",
        "plan_evidence_fallback",
    ]
    assert service.vision.call_count == 1
    assert result.execution_plan["plan_revisions"] == []
    steps = result.execution_plan["react_steps"]
    # Reflection is a bounded missing-observation recovery in the same loop,
    # not a second planner or an extra synthetic tool step.
    assert [step["status"] for step in steps] == ["succeeded", "succeeded", "completed"]
    feedback = _internal_context(generator.action_requests[-1])["feedback"]
    assert [item["code"] for item in feedback] == [
        "answer_missing_observation", "answer_missing_observation",
    ]
    assert [
        step["selection_mode"]
        for step in steps
        if step["outcome"] == "tool_call"
    ] == ["plan_evidence_fallback", "plan_evidence_fallback"]


def test_internal_context_echo_is_rejected_and_never_returned(tmp_path: Path) -> None:
    leaked = json.dumps(
        {
            "allowed_tools_this_step": ["search_tb_knowledge"],
            "case_state": {"image_loaded": True},
            "observations": [],
        },
        ensure_ascii=False,
    )
    service = TBXAgentService(_settings(tmp_path))
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none", answer_focus="capabilities")],
        actions=[_answer(leaked)],
        general_answers=["我可以回答一般问题，并在需要时调用胸片或指南工具。"],
    )

    result = _run(
        service,
        generator,
        query="你能做什么？",
        case_id=None,
        thread_id="reject-internal-context-echo",
    )

    assert result.response.summary == TBX_CAPABILITY_ANSWER
    assert result.response.summary != leaked
    assert all(
        marker not in result.response.summary
        for marker in ("allowed_tools_this_step", "case_state", "observations")
    )
    assert result.trace.terminal.reason_code == "trusted_non_tool_answer"
    assert result.execution_plan["react_steps"][0]["status"] == "completed"
    assert result.execution_plan["react_steps"][0]["observation_code"] is None
    assert len(generator.general_requests) == 0
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True


def test_changed_question_cannot_reuse_exact_prior_answer(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    old_answer = "第一轮专属回答。"
    first_generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none")],
        actions=[_answer(old_answer)],
    )
    first = _run(
        service,
        first_generator,
        query="1+1 等于多少？",
        case_id=None,
        thread_id="stale-answer-guard",
    )
    assert first.response.summary == old_answer

    second_generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none")],
        actions=[
            _answer(old_answer),
            _answer("以下是当前应用的能力。", answer_focus="capabilities"),
        ],
    )
    second = _run(
        service,
        second_generator,
        query="请介绍一下你的工具能力。",
        case_id=None,
        thread_id="stale-answer-guard",
    )

    assert second.response.summary == TBX_CAPABILITY_ANSWER
    assert second.response.summary != first.response.summary
    assert len(second_generator.action_requests) == 2
    feedback = _internal_context(second_generator.action_requests[1])["feedback"]
    assert feedback[-1]["code"] == "stale_dialogue_answer"
    assert second.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert second_generator.plan_requests == []
    assert second_generator.general_requests == []


def test_direct_generic_capability_answer_is_normalized_to_tbx_agent(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("none", answer_focus="capabilities")],
        actions=[_answer("我是普通 AI 助手，可以处理各种任务。")],
    )

    result = _run(
        service,
        generator,
        query="你会干什么？",
        case_id=None,
        thread_id="tbx-capability-normalization",
    )

    assert result.response.summary == TBX_CAPABILITY_ANSWER
    assert result.response.response_kind.value == "capability_statement"
    assert "普通 AI" not in result.response.summary


def test_compound_plan_order_is_enforced_when_model_declines_every_tool(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "localization", "tb_knowledge", "none")],
        actions=[
            _answer("先直接给结论。"),
            _answer("不需要定位。"),
            _answer("不需要检索。"),
            _answer("已根据全部观察完成回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query="判断这张胸片，说明候选位置，再结合指南说明下一步检查。",
        case_id=case.case_id,
        thread_id="compound-plan-order-guard",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.selection_source for item in result.tool_results] == [
        "plan_evidence_fallback",
        "plan_evidence_fallback",
        "plan_evidence_fallback",
    ]
    assert [
        step["tool_name"]
        for step in result.execution_plan["react_steps"]
        if step["outcome"] == "tool_call"
    ] == ["classify_cxr", "localize_cxr", "search_tb_knowledge"]
    assert [
        len(_internal_context(call)["observations"])
        for call in generator.action_requests
    ] == [0, 1, 2, 3]
    assert service.vision.call_count == 1
    assert service.vision.localization_call_count == 1


def test_natural_screening_compound_keeps_all_model_planned_evidence(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _NonTBAbnormalMockBackend(
        settings.fusion_policy(),
        settings.rank03_config(),
    )
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification", "localization", "tb_knowledge")],
        actions=[
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已分别整合模型结果、候选区域和下一步检查依据。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=(
            "这张片是体检发现的，患者目前没有明显症状。"
            "你先告诉我模型更倾向于健康、非结核异常还是 TB；"
            "如果异常，请标出主要候选区域，然后告诉我这种筛查异常一般下一步需要做什么。"
        ),
        case_id=case.case_id,
        thread_id="natural-screening-compound",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert [
        step["evidence_need"]
        for step in result.execution_plan["initial_plan"]["steps"]
    ] == ["classification", "localization", "tb_knowledge"]
    assert result.tool_results[-1].receipt.resolved_guideline_scope == (
        "diagnostic_testing"
    )
    assert result.tool_results[-1].receipt.resolved_guideline_subtopic == (
        "diagnostic_pathway"
    )
    assert "after_abnormal_cxr" in (
        result.tool_results[-1].receipt.resolved_scenario_tags
    )


def test_compatible_model_planned_classification_survives_parser_miss(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    query = "针对当前上传的片，请按模型给出一个倾向。"
    assert TaskGoal.SCREEN_CLASSIFICATION not in parse_task_spec(query).task_goals
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification")],
        actions=[
            _tool("classify_cxr"),
            _answer("已根据当前胸片模型结果回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query=query,
        case_id=case.case_id,
        thread_id="compatible-planned-classification",
    )

    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert service.vision.call_count == 1


def test_react_can_add_requested_evidence_omitted_from_optional_plan(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("localization", "tb_knowledge")],
        actions=[
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已根据本轮知识检索结果回答。"),
        ],
    )

    result = _run(
        service,
        generator,
        query="帮我筛查这张胸片，标出可疑区域，再说明下一步做什么检查。",
        case_id=case.case_id,
        thread_id="dose-plan-tool-contract",
    )

    # An optional plan is a commitment ledger, not the tool allowlist.
    assert result.execution_plan["tool_names"] == [
        "classify_cxr", "localize_cxr", "search_tb_knowledge",
    ]
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert result.trace.terminal.reason_code == "react_answered"
    assert service.vision.call_count == 1


def test_no_image_state_rejects_model_requested_image_tool(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    generator = _ScriptedPlanReactGenerator(
        plans=[_plan("classification")],
        actions=[_tool("classify_cxr")],
    )

    result = _run(
        service,
        generator,
        query="请分析这张胸片。",
        case_id=None,
        thread_id="no-image-guard",
    )

    assert result.execution_plan["tool_names"] == []
    assert result.tool_results == []
    assert result.trace.terminal.reason_code == "required_evidence_tool_unavailable"
    assert service.vision.call_count == 0
    assert "上传胸片" in result.response.summary
    assert "已分析" not in result.response.summary
    assert "模型识别" not in result.response.summary
    assert len(generator.action_requests) == 1
    allowed = _internal_context(generator.action_requests[0])["allowed_tools_this_step"]
    assert "classify_cxr" not in allowed
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is True


class _FailingClassifier(MockRank03Backend):
    def infer(self, *, case_id, image):
        self.call_count += 1
        raise RuntimeError("synthetic classifier failure")


def test_failed_tool_observation_returns_to_react_for_independent_knowledge_tool(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _FailingClassifier(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[
            _plan("classification", "tb_knowledge", "none"),
        ],
        actions=[
            _tool("classify_cxr"),
            _tool("search_tb_knowledge"),
            _answer("分类失败，但已根据指南说明下一步检查。", evidence=["tb_knowledge"]),
        ],
    )

    result = _run(
        service,
        generator,
        query="判断这张胸片；即使分类失败，也说明下一步检查。",
        case_id=case.case_id,
        thread_id="tool-failure-replan",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.status.value for item in result.tool_results] == [
        "failed",
        "succeeded",
    ]
    assert result.execution_plan["plan_revisions"] == []
    assert len(generator.plan_requests) == 1
    assert len(_internal_context(generator.action_requests[1])["observations"]) == 1
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert result.execution_plan["react_steps"][0]["status"] == "failed"
    assert result.execution_plan["react_steps"][1]["tool_name"] == (
        "search_tb_knowledge"
    )
    assert backend.call_count == 1
    assert result.response.citations


def test_anatomy_failure_keeps_successful_visual_and_guideline_evidence_partial(
    tmp_path: Path,
) -> None:
    """One optional worker failure must not turn a compound turn into abstention."""

    settings = replace(
        _settings(tmp_path),
        anatomy_backend="xrv_pspnet",
        anatomy_required=False,
        anatomy_max_workers=1,
    )
    service = TBXAgentService(
        settings,
        anatomy_backend=_FailingAnatomyBackend(),
    )
    case = _upload(service)
    generator = _ScriptedPlanReactGenerator(
        plans=[
            _plan(
                "classification",
                "localization",
                "lung_anatomy",
                "tb_knowledge",
            ),
        ],
        actions=[
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("analyze_lung_anatomy"),
            _tool("search_tb_knowledge"),
            _answer("已整合当前仍然可用的模型证据和指南依据。", evidence=[
                "classification", "localization", "tb_knowledge",
            ]),
        ],
    )

    result = _run(
        service,
        generator,
        query=(
            "先判断这张胸片的模型分类，标出候选区域，并说明候选区在"
            "左肺还是右肺、上中下哪个肺野区域；"
            "再结合指南说明筛查异常后下一步做什么检查。"
        ),
        case_id=case.case_id,
        thread_id="anatomy-failure-partial-success",
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    ]
    assert [item.receipt.status.value for item in result.tool_results] == [
        "succeeded",
        "succeeded",
        "unavailable",
        "succeeded",
    ]
    assert result.response.response_kind != ResponseKind.SAFE_ABSTENTION
    assert result.trace.terminal.reason_code == "react_answered_with_partial_evidence"
    assert result.execution_plan["finalization_recovery"]["partial_evidence"] is True
    assert result.execution_plan["plan_metadata"]["rule_fallback_used"] is False
    assert result.response.citations
    # Preserve successful facts, sourced guidance and the explicit failure;
    # the presentation need not retain the v3 section titles.
    summary = result.response.summary
    assert "模型" in summary
    assert "候选" in summary
    assert "肺野" in summary
    assert "NAAT" in summary
    assert any(
        marker in summary
        for marker in ("不可用", "未完成", "未能完成")
    )
    assert "本次所需工具未能在受控执行边界内完成" not in summary
    assert "因此没有生成医学建议" not in summary
