from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tbx_agent.config import Settings
from tbx_agent.retrieval.query_understanding import understand_guidance_query
from tbx_agent.service import TBXAgentService
from tbx_agent.task_spec import parse_task_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "evaluation" / "fixtures" / "medical_dialogue_qa_v1.json"
QA_FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = QA_FIXTURE["cases"]
REGRESSION_CASES = CASES


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
        openai_enabled=False,
    )


@pytest.fixture(scope="module")
def qa_service(tmp_path_factory: pytest.TempPathFactory):
    service = TBXAgentService(_settings(tmp_path_factory.mktemp("medical-qa")))
    try:
        yield service
    finally:
        service.tool_registry.close()
        service.store.close()


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return value.value


def _run_case(service: TBXAgentService, case: dict[str, Any]):
    return service.respond_with_controller(
        message=case["question"],
        thread_id=f"medical-qa-{case['case_id']}",
        user_id="offline-medical-qa",
        owner_scope="evaluation:offline-medical-qa",
        generator=None,
    )


def _answer_contract_text(response: Any) -> str:
    """Combine synthesized prose and grounded claim text for content checks."""

    return "\n".join([response.summary, *(claim.text for claim in response.claims)])


def test_medical_dialogue_qa_fixture_is_complete_and_bounded() -> None:
    assert QA_FIXTURE["schema_version"] == 1
    assert QA_FIXTURE["clinical_validation"] is False
    assert 25 <= len(CASES) <= 40
    assert len({case["case_id"] for case in CASES}) == len(CASES)
    assert {case["category"] for case in CASES} == {
        "传播与防护",
        "筛查人群",
        "痰涂片与分子检测",
        "特殊人群",
        "住院与照护场景",
        "治疗原则与安全",
        "一般常识与证据缺失",
    }
    assert REGRESSION_CASES
    assert all(case["mode"] == "regression" for case in CASES)

    expected_keys = {
        "task_goals",
        "scope",
        "subtopic",
        "population",
        "scenario_tags",
        "tool_names",
        "answer_status",
        "answer_contains_all",
        "answer_semantic_groups",
        "forbidden_contains",
        "required_chunk_ids",
    }
    for case in CASES:
        assert set(case) == {"case_id", "category", "question", "mode", "expected"}
        assert case["mode"] == "regression"
        assert case["question"].strip()
        assert set(case["expected"]) == expected_keys
        assert case["expected"]["task_goals"]
        assert case["expected"]["answer_contains_all"]
        assert case["expected"]["answer_semantic_groups"]
        assert case["expected"]["forbidden_contains"]


@pytest.mark.parametrize(
    "case",
    REGRESSION_CASES,
    ids=[case["case_id"] for case in REGRESSION_CASES],
)
def test_regression_matrix_task_routing(case: dict[str, Any]) -> None:
    expected = case["expected"]
    profile = understand_guidance_query(case["question"])

    if expected["tool_names"] == ["search_tb_knowledge"]:
        assert expected["task_goals"] == ["search_tb_knowledge"]
        assert profile is not None
        assert _enum_value(profile.scope) == expected["scope"]
        assert profile.subtopic == expected["subtopic"]
        assert list(profile.population) == expected["population"]
        assert [item.value for item in profile.scenario_tags] == expected["scenario_tags"]
    else:
        spec = parse_task_spec(case["question"])
        assert profile is None
        assert [goal.value for goal in spec.task_goals] == expected["task_goals"]
        assert spec.guideline_scope is None
        assert spec.subtopic is None
        assert spec.population == []


@pytest.mark.parametrize(
    "case",
    REGRESSION_CASES,
    ids=[case["case_id"] for case in REGRESSION_CASES],
)
def test_regression_matrix_offline_response(
    qa_service: TBXAgentService,
    case: dict[str, Any],
) -> None:
    expected = case["expected"]
    result = _run_case(qa_service, case)
    response = result.response
    answer_text = _answer_contract_text(response)
    actual_chunks = {item.chunk_id for item in response.retrieved_evidence}
    receipt = result.tool_results[-1].receipt if result.tool_results else None
    actual = {
        "tool_names": result.execution_plan.get("tool_names", []),
        "answer_status": _enum_value(response.answer_status),
        "scope": (
            _enum_value(receipt.resolved_guideline_scope) if receipt is not None else None
        ),
        "subtopic": (
            receipt.resolved_guideline_subtopic if receipt is not None else None
        ),
        "population": (
            list(receipt.resolved_population) if receipt is not None else []
        ),
        "scenario_tags": (
            [item.value for item in receipt.resolved_scenario_tags]
            if receipt is not None
            else []
        ),
    }
    mismatches: dict[str, Any] = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in actual
        if actual[key] != expected[key]
    }
    if case["case_id"] == "general-sugar-intake":
        # This matrix deliberately runs without a language model.  General
        # knowledge is a zero-tool path and therefore reports model
        # unavailability rather than pretending a static medical answer.
        missing_fragments = [] if "语言模型未连接" in answer_text else ["语言模型未连接"]
    else:
        missing_fragments = [
            fragment
            for fragment in expected["answer_contains_all"]
            if fragment not in answer_text
        ]
    forbidden_fragments = [
        fragment for fragment in expected["forbidden_contains"] if fragment in answer_text
    ]
    missing_chunks = sorted(set(expected["required_chunk_ids"]).difference(actual_chunks))
    if missing_fragments:
        mismatches["missing_fragments"] = missing_fragments
    if forbidden_fragments:
        mismatches["forbidden_fragments"] = forbidden_fragments
    if missing_chunks:
        mismatches["missing_chunks"] = missing_chunks
    assert not mismatches, json.dumps(
        {
            "mismatches": mismatches,
            "actual_chunks": sorted(actual_chunks),
            "answer_text": answer_text,
        },
        ensure_ascii=False,
        indent=2,
    )
