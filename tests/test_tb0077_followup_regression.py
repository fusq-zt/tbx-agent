from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.schemas import ClassifierClass, DetectionEvidence
from tbx_agent.service import TBXAgentService
from tbx_agent.vision import MockRank03Backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THREAD_ID = "tb0077-followup-regression"
USER_ID = "tb0077-user"
OWNER_SCOPE = "tenant:tb0077-user"


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
        anatomy_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
    )


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), color=(48, 68, 88)).save(output, format="PNG")
    return output.getvalue()


class _Tb0077VisionBackend(MockRank03Backend):
    """Fixed synthetic evidence matching the structure of the reported case."""

    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        return evidence.model_copy(
            update={
                "class_probabilities": {
                    "healthy": 0.01,
                    "sick_non_tb": 0.04,
                    "tb": 0.95,
                },
                "predicted_class": ClassifierClass.TB,
                "classifier_argmax_tied": False,
                "classifier_flagged": True,
                "top1_score": 0.95,
                "top2_score": 0.04,
                "top1_top2_margin": 0.91,
                "detections": [],
                "detector_flagged": None,
            }
        )

    def localize(self, *, case_id, image):
        self.localization_call_count += 1
        return [
            DetectionEvidence(
                bbox_xyxy=(60.0, 70.0, 205.0, 245.0),
                score=0.88,
            ),
            DetectionEvidence(
                bbox_xyxy=(295.0, 75.0, 440.0, 235.0),
                score=0.47,
            ),
        ]


class _AdversarialDirectAnswerGenerator:
    """Small-model double that tries to leak its private reasoning."""

    backend_id = "test-medgemma"
    model = "test-medgemma-4b"
    model_digest = None

    def __init__(self) -> None:
        self.schema_calls: list[str] = []

    def complete_structured(self, **kwargs):
        schema_name = kwargs["schema_name"]
        self.schema_calls.append(schema_name)
        usage = {"prompt_tokens": 23, "completion_tokens": 11}
        if schema_name == "tbx_plan_react_plan":
            return (
                json.dumps(
                    {
                        "goal": "回答用户当前问题",
                        "steps": [
                            {
                                "objective": "直接回答",
                                "evidence_need": "none",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                usage,
            )
        if schema_name == "tbx_agent_tool_selection":
            return (
                json.dumps(
                    {
                        "tool": None,
                        "direct_answer": (
                            "思考 用户希望我读取 case_state 和 plan。\n"
                            "case_state={\"classification\":{\"status\":\"completed\"}}\n"
                            "最终答案：是的，肺结核通常可以治愈，但需要完成规范治疗和随访。"
                        ),
                    },
                    ensure_ascii=False,
                ),
                usage,
            )
        if schema_name == "tbx_general_chat_answer":
            return (
                json.dumps(
                    {
                        "answer": (
                            "最终答案：是的，肺结核通常可以治愈，但需要完成规范治疗和随访。"
                        )
                    },
                    ensure_ascii=False,
                ),
                usage,
            )
        raise AssertionError(f"unexpected schema: {schema_name}")


@pytest.fixture()
def completed_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = _settings(tmp_path)
    backend = _Tb0077VisionBackend(
        settings.fusion_policy(),
        settings.rank03_config(),
    )
    service = TBXAgentService(settings, vision_backend=backend)
    payload = _png()
    case = service.assess_cxr(
        payload,
        user_id=USER_ID,
        owner_scope=OWNER_SCOPE,
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]
    case, _ = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope=OWNER_SCOPE,
        user_id=USER_ID,
        payload=payload,
    )
    service.respond_with_tool(
        selected_tool="localize_current_cxr",
        message="定位候选区域",
        thread_id=THREAD_ID,
        user_id=USER_ID,
        owner_scope=OWNER_SCOPE,
        case_id=case.case_id,
    )

    # This suite tests dialogue projection, not the segmentation worker.  Give
    # the controller the exact already-completed public anatomy observation
    # from tb0077 while leaving production persistence and model code untouched.
    monkeypatch.setattr(
        service.store,
        "find_latest_completed_anatomy_run",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "_anatomy_run_response",
        lambda *args, **kwargs: SimpleNamespace(
            summary="候选区域位于右上肺野和左上肺野。"
        ),
    )
    yield service, case.case_id
    service.tool_registry.close()
    service.store.close()


def _turn(
    service: TBXAgentService,
    case_id: str,
    message: str,
    *,
    generator=None,
):
    return service.respond_with_controller(
        message=message,
        thread_id=THREAD_ID,
        user_id=USER_ID,
        owner_scope=OWNER_SCOPE,
        case_id=case_id,
        generator=generator,
    )


def test_tb0077_capability_answer_is_product_specific(completed_case) -> None:
    service, case_id = completed_case
    generator = _AdversarialDirectAnswerGenerator()

    result = _turn(service, case_id, "你会干什么", generator=generator)

    assert "TBX-Agent" in result.response.summary
    assert "胸片" in result.response.summary
    assert "候选" in result.response.summary
    assert "肺野" in result.response.summary
    assert "指南" in result.response.summary
    assert "通用 AI" not in result.response.summary
    assert "各种任务" not in result.response.summary
    assert result.execution_plan["tool_names"] == []


def test_tb0077_case_summary_is_a_public_evidence_projection(completed_case) -> None:
    service, case_id = completed_case
    generator = _AdversarialDirectAnswerGenerator()

    result = _turn(
        service,
        case_id,
        "我是医生，帮我把目前已经完成的分析整理成一个简短的 AI 辅助分析摘要",
        generator=generator,
    )
    answer = result.response.summary

    assert answer.startswith("AI 辅助分析摘要：")
    assert "胸片分类模型更倾向于结核类" in answer
    assert "右上肺野和左上肺野" in answer
    assert "诊断为" not in answer
    assert "上叶" not in answer
    assert "中叶" not in answer
    assert "下叶" not in answer
    assert all(
        marker not in answer.casefold()
        for marker in (
            "思考",
            "thought",
            "case_state",
            "allowed_tools_this_step",
            "observations",
            "plan",
        )
    )
    assert result.execution_plan["tool_names"] == []


def test_tb0077_shared_utensil_question_answers_the_actual_question(
    completed_case,
) -> None:
    service, case_id = completed_case

    result = _turn(service, case_id, "共用餐具会传播肺结核吗？")
    answer = result.response.summary

    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert answer.startswith("通常不会通过共用餐具传播")
    assert "空气传播" in answer
    assert "而不是餐具本身" in answer
    assert "模型识别为" not in answer


def test_tb0077_treatment_followup_never_exposes_reasoning_or_runtime_state(
    completed_case,
) -> None:
    service, case_id = completed_case
    first = _turn(service, case_id, "肺结核可以治好吗？肺结核一般需要治疗多久？")
    assert first.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert first.response.summary

    generator = _AdversarialDirectAnswerGenerator()
    followup = _turn(service, case_id, "所以可以治好吗？", generator=generator)
    answer = followup.response.summary

    assert "可以治愈" in answer
    assert all(
        marker not in answer.casefold()
        for marker in (
            "思考",
            "thought",
            "reasoning",
            "case_state",
            "allowed_tools_this_step",
            "observations",
            "plan",
        )
    )
    assert "目标尚未完成" not in answer
    assert "无法确定病例是否可治愈" not in answer
