from __future__ import annotations

import io
import json
from collections import deque
from dataclasses import replace
from pathlib import Path

import requests
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.schemas import ClassifierClass
from tbx_agent.service import TBXAgentService
from tbx_agent.vision import MockRank03Backend
from tests.test_streamlit_ui import RequestsStub, _run_app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TB0050_QUERY = (
    "这张片是体检发现的，患者目前没有明显症状。"
    "你先告诉我模型更倾向于健康、非结核异常还是 TB；"
    "如果异常，请标出主要候选区域，然后告诉我这种筛查异常一般下一步需要做什么。"
)


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
        user_id="tb0050-user",
        owner_scope="tenant:tb0050",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]


def _explicit_condition_plan() -> dict:
    """Test execution of explicit model conditions; semantic model accuracy is tested live."""

    return {
        "action": "plan",
        "tasks": [
            {"task": "classify_image", "when": "always"},
            {"task": "show_detection_boxes", "when": "classification_abnormal"},
            {"task": "search_tb_knowledge", "when": "classification_abnormal"},
        ],
    }


def _tool(name: str) -> dict:
    return {"action": "tool", "tool": name}


def _answer(text: str, *evidence: str) -> dict:
    return {"action": "answer", "answer_focus": "general", "evidence": list(evidence),
            "answer": text}


class _Scripted4BGenerator:
    backend_id = "scripted-4b"
    model = "scripted-4b-model"
    model_digest = None

    def __init__(self, actions: list[dict]) -> None:
        self.actions = deque([_explicit_condition_plan(), *actions])
        self.action_requests: list[dict] = []

    def complete_structured(self, **kwargs):
        if kwargs["schema_name"] == "tbx_react_decision":
            self.action_requests.append(kwargs)
            if not self.actions:
                raise AssertionError("unexpected ReAct step")
            payload = self.actions.popleft()
        else:
            raise AssertionError(f"unexpected schema: {kwargs['schema_name']}")
        return json.dumps(payload, ensure_ascii=False), {
            "prompt_tokens": 17,
            "completion_tokens": 7,
        }


class _FixedClassBackend(MockRank03Backend):
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


class _HealthyBackend(_FixedClassBackend):
    predicted_class = ClassifierClass.HEALTHY


class _AbnormalBackend(_FixedClassBackend):
    predicted_class = ClassifierClass.SICK_NON_TB


def _run(
    service: TBXAgentService,
    case_id: str,
    generator: _Scripted4BGenerator,
    *,
    thread_id: str,
):
    return service.respond_with_controller(
        message=TB0050_QUERY,
        thread_id=thread_id,
        user_id="tb0050-user",
        owner_scope="tenant:tb0050",
        case_id=case_id,
        generator=generator,
    )


def _assert_public_conditional_plan(
    result,
    *,
    cached_classification: bool = False,
) -> None:
    steps = result.execution_plan["initial_plan"]["steps"]
    assert [step["evidence_need"] for step in steps] == [
        "classification",
        "localization",
        "tb_knowledge",
    ]
    assert [step["condition"] for step in steps] == [
        "always",
        "classification_abnormal",
        "classification_abnormal",
    ]
    assert [step["objective"] for step in steps] == [
        (
            "读取已有胸片分类结果"
            if cached_classification
            else "胸片分类"
        ),
        "若分类异常，定位候选区域",
        "若分类异常，查询下一步检查",
    ]
    assert result.execution_plan["plan_metadata"][
        "rule_fallback_used"
    ] is False
    assert result.execution_plan["plan_metadata"]["planning_used"] is True


def _public_response_text(response) -> str:
    return "\n".join(
        [
            response.summary,
            *response.visual_evidence_notes,
            *response.diagnostic_information,
            *response.next_step_information,
            *response.treatment_education,
            *response.limitations,
            *(claim.text for claim in response.claims),
        ]
    )


def test_tb0050_abnormal_runs_classify_then_localize_then_search(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    backend = _AbnormalBackend(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _Scripted4BGenerator(
        [
            _tool("classify_cxr"),
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已整合模型结果、候选区域和下一步检查。",
                    "classification", "localization", "tb_knowledge"),
        ]
    )

    result = _run(service, case.case_id, generator, thread_id="tb0050-abnormal")

    _assert_public_conditional_plan(result)
    assert not generator.actions
    assert {request["schema_name"] for request in generator.action_requests} == {
        "tbx_react_decision",
    }
    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert [item.receipt.status.value for item in result.tool_results] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert backend.call_count == 1
    assert backend.localization_call_count == 1
    assert result.response.citations
    assert result.tool_results[-1].receipt.resolved_guideline_scope == (
        "diagnostic_testing"
    )
    assert result.tool_results[-1].receipt.resolved_guideline_subtopic == (
        "diagnostic_pathway"
    )
    assert "after_abnormal_cxr" in (
        result.tool_results[-1].receipt.resolved_scenario_tags
    )

    public_text = _public_response_text(result.response)
    assert "模型证据" in public_text
    assert "指南建议" in public_text
    assert "非结核异常" in public_text
    assert "候选" in public_text
    for leak in (
        "%",
        "class_probabilities",
        "allowed_tools_this_step",
        "case_state",
        "observations",
        "TBX_INTERNAL_CONTEXT_JSON",
    ):
        assert leak not in public_text


def test_tb0050_healthy_runs_only_classifier_and_skips_abnormal_branch(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _HealthyBackend(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    generator = _Scripted4BGenerator(
        [
            _tool("classify_cxr"),
            _answer("模型更倾向于健康类，本轮无需定位候选区域。", "classification"),
        ]
    )

    result = _run(service, case.case_id, generator, thread_id="tb0050-healthy")

    _assert_public_conditional_plan(result)
    assert not generator.actions
    assert result.execution_plan["tool_names"] == ["classify_cxr"]
    assert [step["status"] for step in result.execution_plan["final_plan"]["steps"]] == [
        "completed",
        "skipped",
        "skipped",
    ]
    assert backend.call_count == 1
    assert backend.localization_call_count == 0
    assert "健康" in _public_response_text(result.response)


def test_tb0050_reuses_cached_abnormal_classification_without_rerunning_model(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    backend = _AbnormalBackend(settings.fusion_policy(), settings.rank03_config())
    service = TBXAgentService(settings, vision_backend=backend)
    case = _upload(service)
    service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope="tenant:tb0050",
        user_id="tb0050-user",
        payload=_png(),
    )
    assert backend.call_count == 1
    generator = _Scripted4BGenerator(
        [
            _tool("localize_cxr"),
            _tool("search_tb_knowledge"),
            _answer("已复用分类并整合定位和检查建议。",
                    "classification", "localization", "tb_knowledge"),
        ]
    )

    result = _run(service, case.case_id, generator, thread_id="tb0050-cached")

    _assert_public_conditional_plan(result, cached_classification=True)
    assert not generator.actions
    assert result.execution_plan["initial_plan"]["steps"][0]["status"] == "completed"
    assert result.execution_plan["tool_names"] == [
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert result.execution_plan["cached_evidence"] == ["classification"]
    assert backend.call_count == 1
    assert backend.localization_call_count == 1


def test_tb0050_condition_is_visible_in_public_ui_plan(monkeypatch) -> None:
    stub = RequestsStub()
    monkeypatch.setattr(requests, "request", stub.request)
    monkeypatch.setattr(requests, "get", stub.get)
    stub.agent_execution_receipts = []
    stub.agent_execution_plan = {
        "source": "plan_react",
        "initial_plan": {
            "steps": [
                {"id": "p1", "objective": "胸片分类", "status": "pending"},
                {
                    "id": "p2",
                    "objective": "若分类异常，定位候选区域",
                    "condition": "classification_abnormal",
                    "status": "pending",
                },
                {
                    "id": "p3",
                    "objective": "若分类异常，查询下一步检查",
                    "condition": "classification_abnormal",
                    "status": "pending",
                },
            ]
        },
    }
    stub.agent_response_overrides = {"summary": "已完成本轮回答。"}

    app = _run_app()
    app.chat_input[0].set_value(TB0050_QUERY).run()

    assert not app.exception
    plan_markup = "\n".join(
        item.value for item in app.markdown if "tbx-turn-plan" in item.value
    )
    assert "胸片分类" in plan_markup
    assert "若分类异常，定位候选区域" in plan_markup
    assert "若分类异常，查询下一步检查" in plan_markup
