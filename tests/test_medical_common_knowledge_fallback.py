from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tbx_agent.config import Settings
from tbx_agent.narrator import (
    MEDICAL_COMMON_KNOWLEDGE_POLICY_ID,
    NarrationRejectedError,
    complete_medical_common_knowledge,
    select_medical_common_knowledge_card,
)
from tbx_agent.schemas import GuidelineAnswerStatus, NarrationStatus
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import GuidelineScope, TaskGoal, parse_task_spec

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
    )


class _StructuredFallbackGenerator:
    backend_id = "test-provider"
    model = "test-model"
    model_digest = None

    def __init__(
        self,
        answer_override: str | None = None,
        *,
        fail_fallback: bool = False,
    ) -> None:
        self.answer_override = answer_override
        self.fail_fallback = fail_fallback
        self.schemas: list[str] = []
        self.requests: list[dict] = []

    def complete_structured(self, **kwargs):
        schema_name = kwargs["schema_name"]
        self.schemas.append(schema_name)
        self.requests.append(kwargs)
        if schema_name == "tbx_plan_react_plan":
            return (
                json.dumps(
                    {
                        "goal": "回答当前结核病相关问题",
                        "steps": [
                            {
                                "objective": "检索并整合适用的结核病知识",
                                "evidence_need": "tb_knowledge",
                            },
                            {
                                "objective": "根据观察形成回答",
                                "evidence_need": "none",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                {"prompt_tokens": 17, "completion_tokens": 4},
            )
        if schema_name == "tbx_agent_tool_selection":
            prompt = json.loads(kwargs["messages"][-1]["content"])
            if prompt["observations"]:
                selection = {
                    "tool": None,
                    "direct_answer": "已根据检索观察完成回答。",
                }
            else:
                selection = {
                    "tool": "search_tb_knowledge",
                    "direct_answer": None,
                }
            return (
                json.dumps(selection, ensure_ascii=False),
                {"prompt_tokens": 17, "completion_tokens": 4},
            )
        if schema_name == "tbx_medical_common_knowledge_answer":
            if self.fail_fallback:
                raise RuntimeError("synthetic provider failure")
            question = json.loads(kwargs["messages"][-1]["content"])["question"]
            default_answer = next(
                (
                    answer
                    for marker, answer in (
                        (
                            "怎么清洁消毒",
                            "家庭环境以开窗通风和常规清洁为主，重点清洁经常接触的表面。",
                        ),
                        (
                            "平时要注意",
                            "应尽快评估，并保持通风、注意咳嗽礼仪、减少密闭空间近距离接触。",
                        ),
                        ("口罩", "怀疑有传染性时，就医和近距离接触他人可佩戴贴合良好的口罩。"),
                        ("耐药", "两者不一样；耐药结核需要结合耐药检测结果由专科团队评估。"),
                        ("家庭成员", "共同居住者应联系医疗机构接受接触者评估。"),
                        ("传染", "肺结核可经空气传播，但并非所有结核病患者都具有传染性。"),
                    )
                    if marker in question
                ),
                "这是不带指南引用的通用医学信息。",
            )
            return (
                json.dumps(
                    {
                        "card_id": None,
                        "answer": (
                            self.answer_override
                            if self.answer_override is not None
                            else default_answer
                        ),
                        "basis": "model_common_knowledge",
                        "individualized_diagnosis": False,
                        "medication_or_regimen_advice": False,
                        "emergency_triage": False,
                        "claims_guideline_evidence": False,
                    },
                    ensure_ascii=False,
                ),
                {"prompt_tokens": 31, "completion_tokens": 13},
            )
        raise AssertionError(f"unexpected schema: {schema_name}")


def _turn(
    service: TBXAgentService,
    generator: _StructuredFallbackGenerator | None,
    query: str,
):
    return service.respond_with_controller(
        message=query,
        thread_id="common-knowledge-thread",
        user_id="user",
        owner_scope="tenant:user",
        generator=generator,
    )


def _stub_empty_rag(service: TBXAgentService, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep common-knowledge fallback tests independent of corpus growth."""

    monkeypatch.setattr(
        service.retriever,
        "retrieve_scoped",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        service.retriever,
        "retrieve",
        lambda *_args, **_kwargs: [],
    )


def test_family_contact_question_is_an_infection_control_goal() -> None:
    spec = parse_task_spec("家里有人得肺结核，其他家庭成员怎么办？")

    assert spec.task_goals == [TaskGoal.GUIDELINE_INFECTION_CONTROL]
    assert spec.guideline_scope == GuidelineScope.INFECTION_CONTROL
    assert spec.subtopic == "contact_evaluation"
    assert spec.population == ["close_contacts"]


@pytest.mark.parametrize(
    ("query", "answer_fragment"),
    (
        (
            "结核病会传染吗？",
            "肺结核可经空气传播",
        ),
        (
            "怀疑肺结核时需要戴口罩吗？",
            "可佩戴贴合良好的医用口罩",
        ),
        (
            "怀疑有传染性结核时平时要注意什么？",
            "保持通风、注意咳嗽礼仪",
        ),
        (
            "耐药结核和普通结核治疗一样吗？",
            "两者不一样",
        ),
    ),
)
def test_empty_rag_can_use_one_labelled_common_knowledge_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    answer_fragment: str,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)
    generator = _StructuredFallbackGenerator()

    result = _turn(service, generator, query)

    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert result.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.response.citations == []
    assert result.response.claims == []
    assert result.response.retrieved_evidence == []
    assert "本轮未检索到可引用指南依据" in result.response.summary
    assert "以上为通用医学信息" in result.response.summary
    assert answer_fragment in result.response.summary
    assert result.response.summary.endswith(
        "注：本轮未检索到可引用指南依据，以上为通用医学信息。"
    )
    assert result.response.narrator_policy_id == MEDICAL_COMMON_KNOWLEDGE_POLICY_ID
    assert result.response.narration_status == NarrationStatus.APPLIED
    assert result.response.narrator_generation_invoked is True
    assert generator.schemas.count("tbx_medical_common_knowledge_answer") == 1

    fallback_request = next(
        item
        for item in generator.requests
        if item["schema_name"] == "tbx_medical_common_knowledge_answer"
    )
    assert [item["role"] for item in fallback_request["messages"]] == [
        "system",
        "user",
    ]
    payload = json.loads(fallback_request["messages"][1]["content"])
    assert set(payload) == {
        "question",
        "allowed_scope",
        "allowed_subtopic",
        "population",
    }
    assert fallback_request["max_tokens"] == 768
    schema = fallback_request["json_schema"]
    assert "const" not in schema["properties"]["answer"]
    assert schema["properties"]["basis"]["const"] == "model_common_knowledge"


def test_empty_contact_evaluation_uses_model_common_knowledge_not_a_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)
    generator = _StructuredFallbackGenerator()

    result = _turn(
        service,
        generator,
        "家里有人得肺结核，其他家庭成员怎么办？",
    )

    assert result.trace.task_spec.subtopic is None
    assert result.tool_results[0].receipt.resolved_guideline_subtopic == (
        "contact_evaluation"
    )
    assert result.tool_results[0].receipt.resolved_population == ["close_contacts"]
    assert result.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.response.citations == []
    assert result.response.claims == []
    assert result.response.retrieved_evidence == []
    assert "共同居住者应联系医疗机构" in result.response.summary
    assert result.response.narration_status == NarrationStatus.APPLIED
    assert generator.schemas.count("tbx_medical_common_knowledge_answer") == 1


def test_precautions_question_does_not_receive_the_transmission_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)

    result = _turn(
        service,
        _StructuredFallbackGenerator(),
        "怀疑有传染性结核时平时要注意什么？",
    )

    assert result.response.summary.startswith("应尽快评估")
    assert "保持通风" in result.response.summary
    assert not result.response.summary.startswith("会。")
    assert result.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.response.claims == []
    assert result.response.citations == []


def test_respiratory_protection_uses_reviewed_source_control_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)
    generator = _StructuredFallbackGenerator(
        "外出时戴好口罩，保持口罩清洁并勤洗手，家里人都按同样方法防护。"
    )

    result = _turn(service, generator, "怀疑有传染性肺结核，在家需要戴口罩吗？")

    assert result.response.summary.startswith("怀疑有传染性肺结核时")
    assert "患者与家人同处" in result.response.summary
    assert "医用口罩" in result.response.summary
    assert "源头控制" in result.response.summary
    assert "家中保持通风" in result.response.summary
    assert "家属是否需要额外呼吸防护" in result.response.summary
    assert "保持口罩清洁" not in result.response.summary
    assert "勤洗手" not in result.response.summary
    assert result.response.summary.endswith(
        "注：本轮未检索到可引用指南依据，以上为通用医学信息。"
    )
    assert result.response.narration_status == NarrationStatus.APPLIED
    assert result.response.citations == []
    assert result.response.claims == []
    assert generator.schemas.count("tbx_medical_common_knowledge_answer") == 1


def test_full_sentence_negation_is_not_overridden_by_keyword_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)

    result = _turn(
        service,
        _StructuredFallbackGenerator(),
        "怀疑传染性肺结核，我不是问口罩，而是家里怎么清洁消毒？",
    )

    assert result.trace.task_spec.subtopic is None
    assert result.tool_results[0].receipt.resolved_guideline_subtopic == (
        "infection_control_precautions"
    )
    assert "开窗通风和常规清洁" in result.response.summary
    assert "佩戴贴合良好的口罩" not in result.response.summary
    assert result.response.narration_status == NarrationStatus.APPLIED
    assert result.response.citations == []
    assert result.response.claims == []


def test_drug_resistant_comparison_answers_boundary_without_regimen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)

    result = _turn(
        service,
        _StructuredFallbackGenerator(),
        "耐药结核和普通结核治疗一样吗？",
    )

    assert result.response.summary.startswith("两者不一样")
    assert "耐药检测结果" in result.response.summary
    assert "剂量" not in result.response.summary
    assert "疗程" not in result.response.summary
    assert result.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.response.claims == []
    assert result.response.citations == []
    assert result.response.narrator_policy_id == MEDICAL_COMMON_KNOWLEDGE_POLICY_ID


def test_unavailable_model_uses_the_same_reviewed_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)
    generator = _StructuredFallbackGenerator(fail_fallback=True)

    result = _turn(service, generator, "结核病会传染吗？")

    assert result.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.response.citations == []
    assert result.response.claims == []
    assert result.response.narrator_generation_invoked is True
    assert result.response.narration_status == NarrationStatus.FALLBACK_ERROR
    assert result.response.narrator_prompt_tokens is None
    assert result.response.narrator_completion_tokens is None
    assert "会。具有传染性的肺结核患者" in result.response.summary
    assert "以上为通用医学信息" in result.response.summary
    assert generator.schemas.count("tbx_medical_common_knowledge_answer") == 1


def test_prohibited_model_text_is_discarded_for_the_fallback_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)
    generator = _StructuredFallbackGenerator("根据指南建议，应使用利福平治疗。")

    result = _turn(service, generator, "结核病会传染吗？")

    selected = select_medical_common_knowledge_card(
        query="结核病会传染吗？",
        guideline_scope="infection_control",
        guideline_subtopic="infection_control",
    )
    assert result.response.summary.startswith(selected.answer)
    assert "利福平" not in result.response.summary
    assert result.response.narration_status == NarrationStatus.FALLBACK_ERROR
    assert result.response.narrator_generation_invoked is True
    assert result.response.citations == []
    assert result.response.claims == []


def test_missing_model_still_returns_the_reviewed_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    _stub_empty_rag(service, monkeypatch)

    result = _turn(service, None, "结核病会传染吗？")

    assert result.response.summary.startswith(
        "会。具有传染性的肺结核患者"
    )
    assert result.response.narration_status == NarrationStatus.FALLBACK_ERROR
    assert result.response.narrator_generation_invoked is False
    assert result.response.narrator_backend is None
    assert result.response.narrator_model is None
    assert result.response.citations == []
    assert result.response.claims == []


def test_common_knowledge_validator_rejects_prohibited_medical_output() -> None:
    generator = _StructuredFallbackGenerator("根据指南建议，应使用利福平治疗。")

    with pytest.raises(NarrationRejectedError, match="protected boundary"):
        complete_medical_common_knowledge(
            generator,
            query="结核病会传染吗？",
            guideline_scope="infection_control",
            guideline_subtopic="infection_control",
        )


@pytest.mark.parametrize(
    ("scope", "subtopic", "query"),
    (
        ("treatment_education", "treatment_principles", "耐药结核怎么治疗？"),
        ("diagnostic_testing", "diagnostic_pathway", "我能排除结核吗？"),
    ),
)
def test_common_knowledge_fallback_cannot_cross_medical_scopes(
    scope: str,
    subtopic: str,
    query: str,
) -> None:
    generator = _StructuredFallbackGenerator("不应被调用。")

    with pytest.raises(NarrationRejectedError, match="out of scope"):
        complete_medical_common_knowledge(
            generator,
            query=query,
            guideline_scope=scope,
            guideline_subtopic=subtopic,
        )

    assert generator.schemas == []


def test_drug_resistant_comparison_card_rejects_regimen_details() -> None:
    generator = _StructuredFallbackGenerator()

    with pytest.raises(NarrationRejectedError, match="out of scope"):
        complete_medical_common_knowledge(
            generator,
            query="耐药结核和普通结核治疗一样吗？具体疗程和剂量是什么？",
            guideline_scope="treatment_education",
            guideline_subtopic="drug_resistant_treatment_comparison",
        )

    assert generator.schemas == []


def test_population_can_select_the_household_contact_card() -> None:
    card = select_medical_common_knowledge_card(
        query="接下来要做什么？",
        guideline_scope="infection_control",
        guideline_subtopic="infection_control",
        population=["close_contacts"],
    )

    assert card.card_id == "tb_household_contacts_general_zh_v1"
    assert card.answer.startswith("家庭成员应联系当地结核病防治机构或医疗机构")


def test_shared_utensil_card_answers_the_named_route_before_generic_transmission() -> None:
    card = select_medical_common_knowledge_card(
        query="共用餐具会传播肺结核吗？",
        guideline_scope="infection_control",
        guideline_subtopic="shared_utensil_transmission",
    )

    assert card.card_id == "tb_shared_utensil_transmission_zh_v1"
    assert card.answer.startswith("通常不会通过共用餐具传播")
    assert "共同呼吸空气" in card.answer
    assert "而不是餐具本身" in card.answer


@pytest.mark.parametrize(
    "query",
    (
        "我是不是传染性肺结核？",
        "耐药结核会传染吗？",
        "我咯血并且呼吸困难，结核会传染吗？",
    ),
)
def test_infection_scope_still_rejects_diagnosis_resistance_and_acute_queries(
    query: str,
) -> None:
    generator = _StructuredFallbackGenerator("不应被调用。")

    with pytest.raises(NarrationRejectedError, match="out of scope"):
        complete_medical_common_knowledge(
            generator,
            query=query,
            guideline_scope="infection_control",
            guideline_subtopic="infection_control",
        )

    assert generator.schemas == []
