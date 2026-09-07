from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.schemas import GuidelineAnswerStatus
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import GuidelineScenarioTag

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
        anatomy_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        max_agent_steps=8,
        max_tool_calls=6,
        max_expensive_vision_calls=4,
        agent_tool_cost_budget=20,
    )


def _png() -> bytes:
    output = io.BytesIO()
    Image.linear_gradient("L").resize((512, 512)).convert("RGB").save(
        output,
        format="PNG",
    )
    return output.getvalue()


def _upload(service: TBXAgentService):
    return service.assess_cxr(
        _png(),
        user_id="failure-user",
        owner_scope="tenant:failure",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]


def _turn(
    service: TBXAgentService,
    query: str,
    *,
    thread_id: str,
    case_id: str | None = None,
):
    return service.respond_with_controller(
        message=query,
        thread_id=thread_id,
        user_id="failure-user",
        owner_scope="tenant:failure",
        case_id=case_id,
    )


def test_unclassified_upload_plus_model_tb_and_negative_smear_uses_both_evidence_paths(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)

    result = _turn(
        service,
        "这个患者胸片模型提示 TB，但是痰涂片阴性，是不是基本可以排除了？",
        thread_id="visual-plus-smear",
        case_id=case.case_id,
    )

    assert result.execution_plan["tool_names"] == [
        "classify_cxr",
        "search_tb_knowledge",
    ]
    assert [
        item["evidence_need"]
        for item in result.execution_plan["initial_plan"]["steps"]
    ] == ["classification", "tb_knowledge"]
    assert [
        item["objective"] for item in result.execution_plan["initial_plan"]["steps"]
    ] == ["胸片分类", "指南证据检索"]
    assert [item.receipt.tool_name for item in result.tool_results] == [
        "classify_current_cxr",
        "search_tb_knowledge",
    ]
    assert service.vision.call_count == 1
    assert result.tool_results[1].receipt.resolved_guideline_subtopic == (
        "negative_test_interpretation"
    )
    assert GuidelineScenarioTag.TEST_SMEAR in (
        result.tool_results[1].receipt.resolved_scenario_tags
    )
    assert result.response.citations
    assert "涂片阴性不能排除" in result.response.summary


def test_explicit_topic_switches_each_run_fresh_knowledge_retrieval(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    expected = (
        (
            "下一步检查？",
            "diagnostic_testing",
            "diagnostic_pathway",
            [],
            [],
            GuidelineAnswerStatus.ANSWERED,
        ),
        (
            "儿童难咳痰怎么办？",
            "special_population",
            "special_population_testing",
            ["children"],
            [GuidelineScenarioTag.NO_SPUTUM],
            GuidelineAnswerStatus.ANSWERED,
        ),
        (
            "怀疑有传染性结核需要戴口罩吗？",
            "infection_control",
            "respiratory_protection",
            [],
            [],
            GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE,
        ),
        (
            "利福平剂量是多少？",
            "treatment_education",
            "medication_dose",
            [],
            [],
            GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE,
        ),
    )
    summaries: list[str] = []

    for query, scope, subtopic, population, scenario_tags, answer_status in expected:
        result = _turn(service, query, thread_id="topic-switches")

        assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
        assert result.execution_plan["initial_plan"]["steps"] == [
                {
                    "id": "p1",
                    "objective": "指南证据检索",
                    "evidence_need": "tb_knowledge",
                    "condition": "always",
                    "status": "pending",
                }
        ]
        assert len(result.tool_results) == 1
        receipt = result.receipt
        assert receipt is not None
        assert receipt.resolved_guideline_scope == scope
        assert receipt.resolved_guideline_subtopic == subtopic
        assert receipt.resolved_population == population
        assert receipt.resolved_scenario_tags == scenario_tags
        assert result.response.source_query == query
        assert result.response.answer_status == answer_status
        assert result.response.summary not in summaries
        summaries.append(result.response.summary)

    dose = _turn(
        service,
        "利福平剂量是多少？",
        thread_id="dose-independent-repeat",
    )
    assert dose.response.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE
    assert dose.response.citations == []
    assert "剂量" in (dose.response.evidence_gap or "")


def test_compound_request_records_unavailable_anatomy_then_continues_retrieval(
    tmp_path: Path,
) -> None:
    service = TBXAgentService(_settings(tmp_path))
    case = _upload(service)

    result = _turn(
        service,
        "分析这张胸片是否提示结核，定位病灶并说明它位于哪个肺区，再告诉我下一步做什么检查。",
        thread_id="compound-with-unavailable-anatomy",
        case_id=case.case_id,
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
    assert [step["tool_name"] for step in result.execution_plan["react_steps"]] == [
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
        None,
    ]
    assert result.execution_plan["graph_node_trace"].count("execute_tool") == 4
    anatomy_result = result.tool_results[2]
    assert anatomy_result.receipt.error_code
    assert "未能" in anatomy_result.response.summary
    assert result.trace.terminal.reason_code == "react_answered_with_partial_evidence"
    assert "肺野分区分析本轮未完成" in result.response.summary
    assert "因此没有生成医学建议" not in result.response.summary
    assert result.response.citations
    assert service.vision.call_count == 1
    assert service.vision.localization_call_count == 1
