from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.narrator import _approved_payload
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_TOOLS = {
    "classify_cxr",
    "localize_cxr",
    "analyze_lung_anatomy",
    "search_tb_knowledge",
}


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


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), color=(48, 68, 88)).save(output, format="PNG")
    return output.getvalue()


def _upload(service: TBXAgentService):
    return service.assess_cxr(
        _png(),
        user_id="stress-user",
        owner_scope="tenant:stress",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]


def _turn(
    service: TBXAgentService,
    *,
    thread_id: str,
    query: str,
    case_id: str | None = None,
    generator=None,
):
    return service.respond_with_controller(
        message=query,
        thread_id=thread_id,
        user_id="stress-user",
        owner_scope="tenant:stress",
        case_id=case_id,
        generator=generator,
    )


def _assert_langgraph(result, *, tools: list[str]) -> None:
    plan = result.execution_plan
    assert plan["framework"] == "langgraph"
    assert plan["tool_names"] == tools
    assert set(tools) <= PUBLIC_TOOLS
    trace = plan["graph_node_trace"]
    assert trace[:3] == ["load_context", "plan", "decide"]
    assert trace[-1] == "finalize"
    assert plan["hidden_reasoning_persisted"] is False
    assert result.trace.hidden_reasoning_persisted is False
    if tools:
        assert trace.count("execute_tool") == len(tools)
        assert trace.count("observe") == len(tools)


def _audit_grounded_narrator_payload(result, *, query: str) -> None:
    response = result.response
    if response.answer_status is None:
        return
    payload = _approved_payload(response)
    assert payload["query"] == query
    assert payload["guideline_scope"] == response.guideline_scope
    assert payload["guideline_subtopic"] == response.guideline_subtopic
    assert payload["required_answer_status"] == response.answer_status.value
    assert payload["allowed_claims"] == [
        item.model_dump(mode="json") for item in response.claims
    ]
    assert payload["retrieved_evidence"] == [
        item.model_dump(mode="json") for item in response.retrieved_evidence
    ]
    evidence_ids = {item.chunk_id for item in response.retrieved_evidence}
    assert all(set(claim.chunk_ids) <= evidence_ids for claim in response.claims)


def _assert_knowledge_turn(
    result,
    *,
    query: str,
    scope: str,
    subtopic: str,
    population: list[str],
) -> None:
    _assert_langgraph(result, tools=["search_tb_knowledge"])
    receipt = result.receipt
    assert receipt is not None
    assert receipt.tool_name == "search_tb_knowledge"
    assert receipt.model_tool_name == "search_tb_knowledge"
    assert receipt.resolved_guideline_scope is not None
    assert receipt.resolved_guideline_scope.value == scope
    assert receipt.resolved_guideline_subtopic == subtopic
    assert receipt.resolved_population == population
    assert "通用问答模型" not in result.response.summary
    _audit_grounded_narrator_payload(result, query=query)


# Query understanding belongs inside search_tb_knowledge. These chains validate
# the tool's resolved dimensions and stored continuation state, not an obsolete
# top-level intent classifier.
NO_CASE_CHAINS = (
    (
        "pregnancy-next-step",
        (
            (
                "我今年32岁，怀孕8周，最近咳嗽严重，该怎么判断自己有没有肺结核",
                "special_population",
                "special_population_testing",
                ["pregnant_people"],
            ),
            (
                "具体该怎么做",
                "special_population",
                "special_population_testing",
                ["pregnant_people"],
            ),
        ),
    ),
    (
        "pregnancy-to-child",
        (
            (
                "孕妇怀疑肺结核时应该做什么检查？",
                "special_population",
                "special_population_testing",
                ["pregnant_people"],
            ),
            (
                "那儿童呢？",
                "special_population",
                "special_population_testing",
                ["children"],
            ),
        ),
    ),
    (
        "xpert-negative",
        (
            (
                "Xpert MTB/RIF 在什么情况下使用？",
                "diagnostic_testing",
                "rapid_molecular_diagnostics",
                [],
            ),
            (
                "如果阴性呢？",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
        ),
    ),
    (
        "smear-negative-repeat",
        (
            (
                "痰涂片阴性是否排除肺结核？",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
            (
                "如果阴性呢",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
        ),
    ),
    (
        "risk-group-to-child",
        (
            (
                "哪些人属于 TB 高风险人群？",
                "screening",
                "risk_groups",
                ["tb_high_risk_population"],
            ),
            ("那儿童呢？", "screening", "risk_groups", ["children"]),
        ),
    ),
    (
        "mask-specific-action",
        (
            (
                "怀疑肺结核时需要戴口罩吗？",
                "infection_control",
                "respiratory_protection",
                [],
            ),
            (
                "具体怎么做",
                "infection_control",
                "respiratory_protection",
                [],
            ),
        ),
    ),
    (
        "adult-symptom-no-image",
        (
            (
                "我咳嗽两周，该怎么判断有没有肺结核",
                "diagnostic_testing",
                "diagnostic_pathway",
                [],
            ),
            (
                "具体该怎么做",
                "diagnostic_testing",
                "diagnostic_pathway",
                [],
            ),
        ),
    ),
    (
        "explicit-topic-replacement",
        (
            (
                "哪些人属于 TB 高风险人群？",
                "screening",
                "risk_groups",
                ["tb_high_risk_population"],
            ),
            (
                "Xpert Ultra 在什么情况下使用？",
                "diagnostic_testing",
                "rapid_molecular_diagnostics",
                [],
            ),
            (
                "如果阴性呢？",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
        ),
    ),
    (
        "duplicate-guideline-question",
        (
            (
                "痰片没查到菌是不是就能排除结核？",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
            (
                "痰片没查到菌是不是就能排除结核？",
                "diagnostic_testing",
                "negative_test_interpretation",
                [],
            ),
        ),
    ),
)


@pytest.mark.parametrize(("chain_name", "turns"), NO_CASE_CHAINS, ids=lambda item: item)
def test_no_case_multiturn_context_contracts(
    tmp_path: Path,
    chain_name: str,
    turns: tuple,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    thread_id = f"stress-{chain_name}"

    for query, scope, subtopic, population in turns:
        result = _turn(service, thread_id=thread_id, query=query)
        _assert_knowledge_turn(
            result,
            query=query,
            scope=scope,
            subtopic=subtopic,
            population=population,
        )

        thread = service.store.get_or_create_thread(
            thread_id,
            "stress-user",
            "tenant:stress",
        )
        assert thread.active_intent == "search_tb_knowledge"
        assert thread.recent_guideline_task is not None
        assert thread.recent_guideline_task.scope == scope
        assert thread.recent_guideline_task.subtopic == subtopic
        assert thread.recent_guideline_task.population == population


def test_explicit_general_topic_switch_disarms_stale_guideline_context(tmp_path: Path) -> None:
    service = TBXAgentService(_settings(tmp_path))
    thread_id = "stress-general-topic-switch"
    first = _turn(
        service,
        thread_id=thread_id,
        query="哪些人属于 TB 高风险人群？",
    )
    arithmetic = _turn(service, thread_id=thread_id, query="1+1 = ？")
    terse = _turn(service, thread_id=thread_id, query="具体该怎么做")

    _assert_langgraph(first, tools=["search_tb_knowledge"])
    _assert_langgraph(arithmetic, tools=[])
    _assert_langgraph(terse, tools=[])
    for result in (arithmetic, terse):
        assert result.receipt is None
        assert result.trace.terminal.reason_code in {
            "react_answered",
            "direct_answer_recovered",
        }
        assert result.response.citations == []
        assert "模型" in result.response.summary
        assert any(
            marker in result.response.summary
            for marker in ("不可用", "未连接", "未启用")
        )
        assert "HIV" not in result.response.summary
    thread = service.store.get_or_create_thread(
        thread_id,
        "stress-user",
        "tenant:stress",
    )
    assert thread.active_intent is None
    assert thread.recent_guideline_task is None


def test_no_image_localization_failure_does_not_poison_later_symptom_query(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    thread_id = "stress-no-image-switch"
    no_image = _turn(service, thread_id=thread_id, query="病灶在哪里？")
    symptom_query = "我咳嗽两周，该怎么判断有没有肺结核"
    symptom = _turn(service, thread_id=thread_id, query=symptom_query)

    _assert_langgraph(no_image, tools=[])
    assert no_image.receipt is None
    assert "上传" in no_image.response.summary
    _assert_knowledge_turn(
        symptom,
        query=symptom_query,
        scope="diagnostic_testing",
        subtopic="diagnostic_pathway",
        population=[],
    )
    assert "上传" not in symptom.response.summary


def test_case_classification_why_and_next_action_are_distinct_contexts(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    thread_id = "stress-case-why-action"
    classification = _turn(
        service,
        thread_id=thread_id,
        query="这张胸片有没有结核病？",
        case_id=case.case_id,
    )
    why = _turn(
        service,
        thread_id=thread_id,
        query="为什么？",
        case_id=case.case_id,
    )
    next_query = "接下来应该做什么检查？"
    action = _turn(
        service,
        thread_id=thread_id,
        query=next_query,
        case_id=case.case_id,
    )

    _assert_langgraph(classification, tools=["classify_cxr"])
    assert classification.receipt is not None
    assert classification.receipt.tool_name == "classify_current_cxr"
    assert classification.receipt.model_tool_name == "classify_cxr"
    _assert_langgraph(why, tools=[])
    assert why.receipt is None
    why_text = "\n".join([why.response.summary, *why.response.visual_evidence_notes])
    assert "分类模型" in why_text
    assert "%" not in why_text
    assert "D-FINE" not in why_text
    _assert_knowledge_turn(
        action,
        query=next_query,
        scope="diagnostic_testing",
        subtopic="diagnostic_pathway",
        population=[],
    )
    assert action.response.predicted_class is None
    assert action.response.visual_result is None
    assert "胸片分类模型将" not in action.response.summary


def test_case_bound_guideline_question_never_reuses_visual_classification(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)
    thread_id = "stress-case-guideline"
    _turn(
        service,
        thread_id=thread_id,
        query="这张胸片有没有结核病？",
        case_id=case.case_id,
    )
    classifier_calls = service.vision.call_count
    turns = (
        (
            "痰涂片阴性是否排除肺结核？",
            "negative_test_interpretation",
        ),
        ("如果阴性呢？", "negative_test_interpretation"),
    )

    for query, subtopic in turns:
        result = _turn(
            service,
            thread_id=thread_id,
            query=query,
            case_id=case.case_id,
        )
        _assert_knowledge_turn(
            result,
            query=query,
            scope="diagnostic_testing",
            subtopic=subtopic,
            population=[],
        )
        assert result.response.predicted_class is None
        assert result.response.visual_result is None
        assert "模型识别为" not in result.response.summary
    assert service.vision.call_count == classifier_calls


class _WrongToolGenerator:
    backend_id = "wrong-tool-test"
    model = "wrong-tool-test-model"

    def complete_structured(self, **kwargs):
        if kwargs["schema_name"] == "tbx_plan_react_plan":
            return (
                json.dumps(
                    {
                        "goal": "回答结核检查问题",
                        "steps": [
                            {
                                "objective": "查找结核知识依据",
                                "evidence_need": "tb_knowledge",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                {"prompt_tokens": 12, "completion_tokens": 5},
            )
        assert kwargs["schema_name"] == "tbx_agent_tool_selection"
        return (
            json.dumps(
                {"tool": "classify_cxr", "direct_answer": None},
                ensure_ascii=False,
            ),
            {"prompt_tokens": 12, "completion_tokens": 5},
        )


class _FailingReActGenerator:
    backend_id = "failure-test"
    model = "failure-test-model"

    def complete_structured(self, **kwargs):
        raise RuntimeError("synthetic provider failure")


@pytest.mark.parametrize(
    "generator",
    (_WrongToolGenerator(), _FailingReActGenerator()),
    ids=("wrong-tool", "provider-failure"),
)
def test_evidence_boundary_recovers_smear_guidance_when_model_cannot_select_tool(
    tmp_path: Path,
    generator,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    result = _turn(
        service,
        thread_id=f"stress-guard-{generator.backend_id}",
        query="痰涂片阴性是否排除肺结核？",
        generator=generator,
    )

    _assert_knowledge_turn(
        result,
        query="痰涂片阴性是否排除肺结核？",
        scope="diagnostic_testing",
        subtopic="negative_test_interpretation",
        population=[],
    )
    assert result.receipt is not None
    assert result.receipt.status.value == "succeeded"
    assert result.receipt.selection_source == "plan_evidence_fallback"
    assert result.trace.terminal.reason_code == "react_answered_with_evidence_fallback"
    assert result.execution_plan["finalization_recovery"]["react_status"] == (
        "model_failed_authoritative_evidence_preserved"
    )
    assert result.execution_plan["plan_revisions"] == []
    assert result.response.citations
    assert result.response.claims
    assert "涂片阴性不能排除肺结核" in result.response.summary
    assert "本轮没有获得" not in result.response.summary
    assert not {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
    }.intersection(result.execution_plan["tool_names"])
    recovery_steps = [
        step
        for step in result.execution_plan["react_steps"]
        if step["selection_mode"] == "plan_evidence_fallback"
    ]
    assert recovery_steps
    assert recovery_steps[0]["tool_name"] == "search_tb_knowledge"
