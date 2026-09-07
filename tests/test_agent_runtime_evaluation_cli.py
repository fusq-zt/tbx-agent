from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_agent_runtime.py"


def _load_runtime_evaluator_module():
    module_name = "tbx_agent_runtime_evaluator_test_module"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_runtime_evaluation_help_exposes_offline_and_provenance_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--suite",
        "--case-id",
        "--runtime-mode",
        "--seed",
        "--source-revision",
        "--work-dir",
        "--output",
        "--markdown-output",
        "--force",
    ):
        assert option in result.stdout
    assert "never downloads" in result.stdout


def test_scenario_hash_binds_message_setup_turns_and_fault_injection() -> None:
    module = _load_runtime_evaluator_module()
    case_id = "traj.v3.cached-rationale.001"
    cases = [SimpleNamespace(case_id=case_id)]
    original = module.SCENARIOS[case_id]
    baseline = module._scenario_set_sha256(
        cases,
        scenarios={case_id: original},
    )

    variants = (
        replace(original, message=f"{original.message}（变更）"),
        replace(original, setup_messages=(*original.setup_messages, "追加前置轮次")),
        replace(original, saturate_first_call=not original.saturate_first_call),
        replace(original, unsupported_tool="synthetic_fault_tool"),
    )
    assert len(baseline) == 64
    assert all(
        module._scenario_set_sha256(cases, scenarios={case_id: variant}) != baseline
        for variant in variants
    )


def test_runtime_cost_accounting_distinguishes_local_assumption_from_remote_unknown() -> None:
    module = _load_runtime_evaluator_module()
    local = SimpleNamespace(
        narrator_backend="llama_cpp",
        llama_cpp_base_url="http://127.0.0.1:11435",
        llama_cpp_allow_remote=False,
    )
    remote = SimpleNamespace(
        narrator_backend="openai",
        llama_cpp_base_url="http://127.0.0.1:11435",
        llama_cpp_allow_remote=False,
    )

    assert module._runtime_cost_accounting(local, mode="configured") == {
        "basis": "marginal_cost_assumed_zero",
        "estimated_cost_usd": 0.0,
        "scope": "loopback_llama_cpp_marginal_api_cost_only",
    }
    assert module._runtime_cost_accounting(remote, mode="configured") == {
        "basis": "not_measured",
        "estimated_cost_usd": None,
        "scope": "remote_or_unclassified_backend_cost_not_measured",
    }


def test_mock_cli_runs_full_real_controller_suite_and_writes_cards(tmp_path: Path) -> None:
    output = tmp_path / "agent-runtime.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--runtime-mode",
            "mock",
            "--seed",
            "20260831",
            "--source-revision",
            "runtime-cli-test-revision",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "completed"
    assert summary["case_count"] == 7
    assert summary["passed"] is True
    assert output.is_file()
    assert output.with_suffix(".md").is_file()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "tbx-agent-runtime-evaluation-v1"
    assert report["evaluation_kind"] == "actual_bounded_agent_runtime"
    assert report["clinical_validation"] is False
    assert report["synthetic"] is True
    assert report["selection_use"] is False
    assert report["locked_or_hidden_test_used"] is False
    assert report["passed"] is True
    assert report["source"] == {
        "revision": "runtime-cli-test-revision",
        "dirty": None,
    }
    assert report["suite"]["selected_case_count"] == 7
    assert report["suite"]["full_suite_case_count"] == 7
    assert report["suite"]["suite_id"] == "tbx_agent_trajectory_runtime_v3"
    assert report["suite"]["suite_version"] == "3.2.0"
    assert len(report["suite"]["split_hash"]) == 64
    assert len(report["suite"]["suite_manifest_sha256"]) == 64
    assert len(report["suite"]["scenario_set_sha256"]) == 64
    assert report["suite"]["scenario_set_sha256"] == (
        report["suite"]["full_scenario_set_sha256"]
    )
    assert report["runtime"]["seed"] == 20260831
    assert report["runtime"]["duration_seconds"] >= 0
    assert report["runtime"]["peak_vram_mb"] == 0.0
    assert report["runtime"]["peak_vram_measured"] is True
    configuration = report["runtime"]["configuration"]
    assert configuration["runtime_mode"] == "mock"
    assert configuration["vision_backend"] == "mock"
    assert configuration["anatomy_backend"] == "none"
    assert configuration["narrator_backend"] == "none"
    assert configuration["scenario_set_sha256"] == report["suite"][
        "scenario_set_sha256"
    ]
    assert configuration["cost_accounting"] == {
        "basis": "marginal_cost_assumed_zero",
        "estimated_cost_usd": 0.0,
        "scope": "deterministic_mock_runtime_no_billed_model_calls",
    }
    assert configuration["model_downloads_allowed"] is False

    assert set(report["scorecard"]) == {
        "observation_coverage",
        "tool_selection_exact_rate",
        "tool_selection_acceptable_rate",
        "tool_contract_validity_rate",
        "step_completion_rate",
        "recovery_rate",
        "context_provenance_rate",
        "secret_non_leak_rate",
        "evidence_faithfulness_rate",
        "execution_claim_consistency_rate",
        "cost_accounting_coverage",
        "latency_p50_ms",
        "latency_p95_ms",
    }
    assert report["scorecard"]["evidence_faithfulness_rate"] is None
    assert all(
        value == 1.0
        for key, value in report["scorecard"].items()
        if "latency" not in key and key != "evidence_faithfulness_rate"
    )
    evidence_counts = report["metric_fractions"][
        "final_evidence_faithfulness_rate"
    ]
    assert evidence_counts == {"numerator": 0, "denominator": 0, "value": None}
    assert report["trajectory_evaluation"]["metrics"][
        "cost_measurement_coverage"
    ] == 0.0
    assert report["trajectory_evaluation"]["metrics"][
        "cost_accounting_coverage"
    ] == 1.0
    assert report["trajectory_evaluation"]["metrics"][
        "assumed_zero_marginal_cost_cases"
    ] == 7
    assert report["trajectory_evaluation"]["clinical_validation"] is False
    assert report["trajectory_evaluation"]["source_revision"] == (
        "runtime-cli-test-revision"
    )
    assert len(report["cases"]) == len(report["observations"]) == 7
    assert all(card["execution_plan_source"] == "plan_react" for card in report["cases"])
    assert all(card["trace_version"] == "tbx-agent-trace-v2" for card in report["cases"])
    assert all(
        card["synthetic_plan_or_receipt_injected"] is False
        for card in report["cases"]
    )
    assert {
        call["contract_version"]
        for card in report["cases"]
        for call in card["tool_calls"]
    } == {"tbx-tool-contract-v7"}
    assert all(
        card["graph_node_trace"][0:3] == ["load_context", "plan", "decide"]
        and card["graph_node_trace"][-1] == "finalize"
        for card in report["cases"]
    )

    recovery = next(
        card
        for card in report["cases"]
        if card["case_id"] == "traj.v3.recovery.saturated.001"
    )
    assert [call["status"] for call in recovery["tool_calls"]] == [
        "saturated",
        "succeeded",
    ]
    assert [call["attempt"] for call in recovery["tool_calls"]] == [1, 2]
    assert recovery["tool_calls"][1]["selection_source"] == "react_recovery"
    assert recovery["recoveries"][0]["reason_code"] == "saturated_retry_succeeded"
    assert recovery["reflections"] == []
    assert recovery["graph_node_trace"].count("execute_tool") == 1
    assert recovery["graph_node_trace"].count("observe") == 1
    assert recovery["runtime_findings"] == [
        "recovery:saturated_retry_succeeded"
    ]

    compound = next(
        card
        for card in report["cases"]
        if card["case_id"] == "traj.v3.compound-replan.001"
    )
    assert [item["tool_name"] for item in compound["tool_calls"]] == [
        "classify_cxr",
        "localize_cxr",
        "search_tb_knowledge",
    ]
    assert compound["graph_node_trace"].count("execute_tool") == 3
    assert compound["graph_node_trace"].count("observe") == 3
    assert compound["terminal"]["action"] == "stop"
    assert compound["terminal"]["reason_code"] == "react_answered"
    assert report["failure_class_counts"] == {}

    cached = next(
        card
        for card in report["cases"]
        if card["case_id"] == "traj.v3.cached-rationale.001"
    )
    assert cached["setup_runs"][0]["tool_calls"][0]["tool_name"] == "classify_cxr"
    assert cached["tool_calls"] == []

    prior = next(
        card for card in report["cases"] if card["case_id"] == "traj.v3.prior-gap.001"
    )
    assert prior["terminal"]["action"] == "stop"
    assert prior["terminal"]["reason_code"] == "react_answered"
    assert prior["tool_calls"] == []
    assert report["runtime_finding_counts"][
        "recovery:saturated_retry_succeeded"
    ] == 1
    assert "terminal_gap:prior_evidence_unavailable" not in report[
        "runtime_finding_counts"
    ]

    rendered = output.read_text(encoding="utf-8")
    for private_value in (
        "SECRET_RUNTIME_CANARY_42E9",
        "agent-runtime-evaluator",
        "tenant:agent-runtime-evaluation",
        "痰NAAT是什么检查？",
        "current_query",
        "hidden_reasoning_persisted",
    ):
        assert private_value not in rendered
    for internal_tool_name in (
        "classify_current_cxr",
        "localize_current_cxr",
        "inspect_anatomical_context",
        "retrieve_guideline",
        "search_tb_guidance",
    ):
        assert internal_tool_name not in rendered

    markdown = output.with_suffix(".md").read_text(encoding="utf-8")
    assert "TBX-Agent 真实运行时评测卡" in markdown
    assert "clinical_validation = false" in markdown
    assert "tool_selection_acceptable_rate" in markdown
    assert "Scenario hash" in markdown
    assert "指标覆盖" in markdown
    assert "状态转换与失败明细" in markdown
    assert "失败分类汇总" in markdown
    assert "真实运行时发现" in markdown


def test_v3_suite_encodes_v7_public_tools_without_legacy_guideline_tools() -> None:
    suite_root = PROJECT_ROOT / "evaluation" / "suites" / "trajectory_v3"
    manifest = json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))
    cases = [
        json.loads(line)
        for line in (suite_root / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert manifest["suite_version"] == "3.2.0"
    assert manifest["expected_case_count"] == len(cases) == 7
    assert {case["case_id"] for case in cases} == {
        "traj.v3.classify-only.001",
        "traj.v3.localize-only.001",
        "traj.v3.compound-replan.001",
        "traj.v3.cached-rationale.001",
        "traj.v3.prior-gap.001",
        "traj.v3.recovery.saturated.001",
        "traj.v3.injection.reject.001",
    }
    assert all(
        case["expected"]["require_checkpoint_provenance"] is False for case in cases
    )
    assert all(case["expected"]["reflection_should_run"] is False for case in cases)
    rendered = json.dumps(cases, ensure_ascii=False)
    assert "get_exact_case_and_explain" not in rendered
    assert "describe_agent_capabilities" not in rendered
    assert "retrieve_diagnostic_guidance" not in rendered
    assert "retrieve_treatment_education" not in rendered
    assert "search_tb_knowledge" in rendered
    assert "planner_context" not in rendered
