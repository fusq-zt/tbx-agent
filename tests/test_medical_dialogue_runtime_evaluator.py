from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_medical_dialogue_runtime.py"
FIXTURE = PROJECT_ROOT / "evaluation" / "fixtures" / "medical_dialogue_qa_v1.json"


def _load_module():
    module_name = "tbx_medical_dialogue_runtime_evaluator_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _expected(*, answer: str = "依据内容", chunk_ids: list[str] | None = None) -> dict:
    return {
        "task_goals": ["search_tb_knowledge"],
        "scope": "diagnostic_testing",
        "subtopic": "diagnostic_pathway",
        "population": [],
        "scenario_tags": [],
        "tool_names": ["search_tb_knowledge"],
        "answer_status": "ANSWERED",
        "answer_contains_all": [answer],
        "answer_semantic_groups": [
            {"id": "grounded-answer", "any_of": [answer, "等价表述"]}
        ],
        "forbidden_contains": ["禁止内容"],
        "required_chunk_ids": chunk_ids or ["chunk-1"],
    }


def _payload(
    *,
    answer: str = "这里是依据内容。",
    chunk_ids: list[str] | None = None,
) -> dict:
    ids = chunk_ids or ["chunk-1"]
    return {
        "summary": answer,
        "answer_status": "ANSWERED",
        "narration_status": "applied",
        "narrator_backend": "llama_cpp",
        "narrator_model": "tbx-medgemma-1.5-4b-it-q4-k-m",
        "narrator_generation_invoked": True,
        "guideline_scope": "diagnostic_testing",
        "guideline_subtopic": "diagnostic_pathway",
        "retrieved_evidence": [
            {
                "chunk_id": chunk_id,
                "metadata": {"allowed_claim_scope": ["diagnostic_testing"]},
            }
            for chunk_id in ids
        ],
        "execution_receipts": [
            {
                "tool_name": "search_tb_knowledge",
                "model_tool_name": "search_tb_knowledge",
                "status": "succeeded",
                "attempt": 1,
                "step_index": 0,
                "plan_id": "plan-1",
                "step_id": "s1",
                "fallback_used": False,
                "error_code": None,
                "resolved_guideline_scope": "diagnostic_testing",
                "resolved_guideline_subtopic": "diagnostic_pathway",
                "resolved_population": [],
                "resolved_scenario_tags": [],
            }
        ],
        "execution_plan": {
            "source": "plan_react",
            "hidden_reasoning_persisted": False,
            "initial_plan": {
                "plan_id": "plan-1",
                "revision": 0,
                "goal": "回答用户的结核病检查问题",
                "steps": [
                    {
                        "id": "p1",
                        "objective": "检索适用的结核病检查证据",
                        "evidence_need": "tb_knowledge",
                        "status": "pending",
                    },
                    {
                        "id": "p2",
                        "objective": "整合观察并回答",
                        "evidence_need": "none",
                        "status": "pending",
                    },
                ],
            },
            "plan_revisions": [],
            "react_steps": [
                {
                    "step_index": 0,
                    "plan_revision": 0,
                    "outcome": "tool_call",
                    "tool_name": "search_tb_knowledge",
                    "selection_mode": "native_tool_call",
                    "status": "succeeded",
                    "observation_code": "guidance_search_succeeded",
                    "recovery": False,
                },
                {
                    "step_index": 1,
                    "plan_revision": 0,
                    "outcome": "answer",
                    "tool_name": None,
                    "selection_mode": "native_direct_answer",
                    "status": "completed",
                    "observation_code": None,
                    "recovery": False,
                },
            ],
            "tool_names": ["search_tb_knowledge"],
        },
        "agent_trace": {
            "task_spec": {
                "task_goals": ["search_tb_knowledge"],
            }
        },
    }


def _case(case_id: str, question: str) -> dict:
    return {
        "case_id": case_id,
        "category": "测试",
        "question": question,
        "mode": "regression",
        "expected": _expected(),
    }


def test_checked_in_fixture_loads_with_stable_hash_and_selection() -> None:
    module = _load_module()
    fixture, digest = module.load_fixture(FIXTURE)

    assert fixture["suite_id"] == "tbx-medical-dialogue-qa-v1"
    assert len(digest) == 64
    selected = module.select_cases(
        fixture["cases"],
        ["diagnosis-smear-negative-formal", "general-sugar-intake"],
    )
    assert [case["case_id"] for case in selected] == [
        "diagnosis-smear-negative-formal",
        "general-sugar-intake",
    ]
    with pytest.raises(ValueError, match="unknown fixture case IDs"):
        module.select_cases(fixture["cases"], ["does-not-exist"])


def test_extract_and_judge_observation_passes_complete_contract() -> None:
    module = _load_module()
    observation = module.extract_observation(_payload())
    checks, failures = module.judge_observation(_expected(), observation)

    assert all(checks.values())
    assert failures == []
    assert observation == {
        "route": {
            "task_goals": ["search_tb_knowledge"],
            "scope": "diagnostic_testing",
            "subtopic": "diagnostic_pathway",
            "population": [],
            "scenario_tags": [],
            "guidance_resolution_source": "tool_receipt",
        },
        "case_id": None,
        "controller_steps": [],
        "observations": [],
        "terminal": None,
        "execution_tool_history": ["search_tb_knowledge"],
        "plan_react": {
            "source": "plan_react",
            "hidden_reasoning_persisted": False,
            "initial_plan": {
                "plan_id": "plan-1",
                "revision": 0,
                "goal": "回答用户的结核病检查问题",
                "step_count": 2,
                "steps_valid": True,
            },
            "plan_revisions": [],
            "react_steps": [
                {
                    "step_index": 0,
                    "plan_revision": 0,
                    "outcome": "tool_call",
                    "tool_name": "search_tb_knowledge",
                    "selection_mode": "native_tool_call",
                    "status": "succeeded",
                    "observation_code": "guidance_search_succeeded",
                    "recovery": False,
                },
                {
                    "step_index": 1,
                    "plan_revision": 0,
                    "outcome": "answer",
                    "tool_name": None,
                    "selection_mode": "native_direct_answer",
                    "status": "completed",
                    "observation_code": None,
                    "recovery": False,
                },
            ],
        },
        "tool_names": ["search_tb_knowledge"],
        "tools": [
            {
                "name": "search_tb_knowledge",
                "internal_name": "search_tb_knowledge",
                "status": "succeeded",
                "attempt": 1,
                "step_index": 0,
                "plan_id": "plan-1",
                "step_id": "s1",
                "case_id": None,
                "requires_case": False,
                "fallback_used": False,
                "error_code": None,
                "resolved_scope": "diagnostic_testing",
                "resolved_subtopic": "diagnostic_pathway",
                "resolved_population": [],
                "resolved_scenario_tags": [],
            }
        ],
        "answer_status": "ANSWERED",
        "chunk_ids": ["chunk-1"],
        "evidence_items": [
            {
                "chunk_id": "chunk-1",
                "allowed_claim_scope": ["diagnostic_testing"],
            }
        ],
        "narration_status": "applied",
        "narrator_backend": "llama_cpp",
        "narrator_model": "tbx-medgemma-1.5-4b-it-q4-k-m",
        "narrator_generation_invoked": True,
        "answer": "这里是依据内容。",
    }


def test_judge_reports_missing_required_forbidden_and_evidence() -> None:
    module = _load_module()
    observation = module.extract_observation(
        _payload(answer="出现禁止内容。", chunk_ids=["other-chunk"])
    )
    checks, failures = module.judge_observation(_expected(), observation)

    assert checks["answer_semantics"] is False
    assert checks["forbidden_fragments"] is False
    assert checks["required_chunk_ids"] is False
    assert {item["check"] for item in failures} == {
        "answer_semantics",
        "forbidden_fragments",
        "required_chunk_ids",
        "guideline_evidence_applicability",
    }


def test_judge_accepts_an_approved_semantic_alternative_not_exact_template_text() -> None:
    module = _load_module()
    expected = _expected(answer="痰涂片阴性不能排除肺结核")
    expected["answer_semantic_groups"] = [
        {
            "id": "negative-smear-does-not-rule-out",
            "any_of": [
                "痰涂片阴性不能排除肺结核",
                "涂片没有查到菌也不代表没有肺结核",
            ],
        }
    ]
    observation = module.extract_observation(
        _payload(answer="涂片没有查到菌也不代表没有肺结核。")
    )

    checks, failures = module.judge_observation(expected, observation)

    assert checks["answer_semantics"] is True
    assert failures == []


def test_judge_matches_compositional_semantic_anchor_sets() -> None:
    module = _load_module()
    expected = _expected()
    expected["answer_semantic_groups"] = [
        {
            "id": "negative-smear-does-not-rule-out",
            "all_of": [
                ["涂片阴性", "涂片没有查到菌"],
                ["不能排除", "不代表没有"],
                ["肺结核", "结核病"],
            ],
        }
    ]
    observation = module.extract_observation(
        _payload(answer="虽然涂片没有查到菌，但这不代表没有肺结核。")
    )

    checks, failures = module.judge_observation(expected, observation)

    assert checks["answer_semantics"] is True
    assert failures == []


def test_judge_requires_every_semantic_group_and_reports_stable_group_id() -> None:
    module = _load_module()
    expected = _expected()
    expected["answer_semantic_groups"] = [
        {"id": "not-an-exclusion-test", "any_of": ["不能排除肺结核"]},
        {"id": "needs-clinical-context", "any_of": ["需要结合临床判断", "需综合评估"]},
    ]
    observation = module.extract_observation(_payload(answer="这个结果不能排除肺结核。"))

    checks, failures = module.judge_observation(expected, observation)

    assert checks["answer_semantics"] is False
    assert failures == [
        {
            "check": "answer_semantics",
            "missing": [
                {
                    "id": "needs-clinical-context",
                    "missing_alternative_sets": [["需要结合临床判断", "需综合评估"]],
                }
            ],
        }
    ]


def test_load_fixture_rejects_malformed_or_duplicate_semantic_groups(tmp_path: Path) -> None:
    module = _load_module()
    fixture = {
        "schema_version": 1,
        "clinical_validation": False,
        "cases": [_case("semantic-contract", "测试语义契约")],
    }
    fixture["cases"][0]["expected"]["answer_semantic_groups"] = [
        {"id": "same", "any_of": ["答案"]},
        {"id": "same", "any_of": ["其他答案"]},
    ]
    path = tmp_path / "bad-fixture.json"
    path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate semantic group IDs"):
        module.load_fixture(path)


def test_run_evaluation_uses_fresh_threads_and_continues_after_runtime_error() -> None:
    module = _load_module()
    cases = [_case("case-one", "第一问"), _case("case-two", "第二问")]
    requests: list[dict] = []

    def fake_post(endpoint: str, payload: dict, timeout_seconds: float) -> dict:
        assert endpoint == "http://127.0.0.1:8000/v1/agent/respond"
        assert timeout_seconds == 12
        requests.append(payload)
        if payload["message"] == "第一问":
            raise module.RuntimeRequestError("HTTP 503 from Agent API")
        return _payload()

    report = module.run_evaluation(
        {"suite_id": "test-suite"},
        fixture_sha256="a" * 64,
        cases=cases,
        base_url="http://127.0.0.1:8000/",
        provider="local_medgemma",
        timeout_seconds=12,
        post=fake_post,
        run_id="fixedrun",
    )

    assert requests[0]["thread_id"] != requests[1]["thread_id"]
    assert all(request["llm_provider"] == "local_medgemma" for request in requests)
    assert all("api_key" not in request for request in requests)
    summary = dict(report["summary"])
    quality = summary.pop("agent_quality_metrics")
    assert summary == {
        "total": 2,
        "passed": 1,
        "failed": 0,
        "runtime_errors": 1,
        "pass_rate": 0.5,
        "by_fixture_mode": {
            "regression": {"total": 2, "passed": 1, "failed": 0, "runtime_error": 1}
        },
        "narration_status_counts": {"applied": 1},
        "tool_status_counts": {"succeeded": 1},
        "failed_check_counts": {},
    }
    assert quality["high_level_tool_selection_accuracy"] == {
        "correct": 1,
        "total": 1,
        "rate": 1.0,
    }
    assert quality["plan_quality"] == {"correct": 1, "total": 1, "rate": 1.0}
    assert quality["react_step_cardinality"]["contract_violations"] == 0
    assert quality["guideline_resolution_accuracy"]["rate"] == 1.0
    assert quality["guideline_evidence_applicability"]["rate"] == 1.0
    assert report["cases"][0]["status"] == "runtime_error"
    assert report["cases"][1]["status"] == "passed"
    assert "api_key" not in json.dumps(report)


def test_post_json_turns_http_failure_into_explicit_sanitized_error() -> None:
    module = _load_module()

    def failing_opener(*_args, **_kwargs):
        raise HTTPError(
            "http://127.0.0.1:8000/v1/agent/respond",
            500,
            "internal error",
            {},
            None,
        )

    with pytest.raises(module.RuntimeRequestError, match="HTTP 500") as error:
        module.post_json(
            "http://127.0.0.1:8000/v1/agent/respond",
            {"message": "test"},
            1,
            opener=failing_opener,
        )
    assert "internal error" not in str(error.value)


def test_post_json_accepts_a_json_object_and_enforces_response_contract() -> None:
    module = _load_module()

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def getcode(self):
            return self.status

        def read(self, _limit: int) -> bytes:
            return b'{"summary":"ok"}'

    decoded = module.post_json(
        "http://127.0.0.1:8000/v1/agent/respond",
        {"message": "test"},
        1,
        opener=lambda *_args, **_kwargs: FakeResponse(),
    )
    assert decoded == {"summary": "ok"}
    with pytest.raises(module.ResponseContractError, match="agent_trace"):
        module.extract_observation(decoded)


@pytest.mark.parametrize(
    "base_url",
    (
        "127.0.0.1:8000",
        "ftp://127.0.0.1",
        "http://user:secret@127.0.0.1:8000",
        "http://127.0.0.1:8000?api_key=secret",
    ),
)
def test_base_url_rejects_invalid_or_secret_bearing_values(base_url: str) -> None:
    module = _load_module()
    with pytest.raises(ValueError):
        module.normalize_base_url(base_url)


def test_cli_help_exposes_live_runtime_controls() -> None:
    module = _load_module()
    help_text = module._parser().format_help()
    for option in (
        "--base-url",
        "--provider",
        "--fixture",
        "--case-id",
        "--timeout-seconds",
        "--output",
    ):
        assert option in help_text


def test_write_report_refuses_silent_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    output = tmp_path / "report.json"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        module._write_report(output, {"result": "new"})
    assert output.read_text(encoding="utf-8") == "existing"

    new_output = tmp_path / "new-report.json"
    module._write_report(new_output, {"result": "new"})
    assert json.loads(new_output.read_text(encoding="utf-8")) == {"result": "new"}
