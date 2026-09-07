from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tbx_agent.evaluation.system_bench import (
    ALL_DIMENSIONS,
    DETERMINISTIC_ADAPTER_VERSION,
    BenchReport,
    RuntimeEvidence,
    SystemBenchConfig,
    SystemCase,
    SystemObservation,
    append_ledger,
    compare_reports,
    evaluate_observations,
    load_config,
    load_suite,
    main,
    verify_ledger,
)
from tbx_agent.evaluation.system_bench_ci import run_current_candidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "system_bench_config_v1_2.json"
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "suites" / "system_v1_2" / "manifest.json"
CURRENT_MANIFEST_PATH = (
    PROJECT_ROOT / "evaluation" / "suites" / "system_v1_6" / "manifest.json"
)
V1_1_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "system_bench_config_v1_1.json"
V1_1_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "suites" / "system_v1" / "manifest.json"
V1_1_CONFIG_SHA256 = "ac974133726e07f6c9ee86f697d6f679892909559f9277112a879119d44cc78e"
V1_1_CASES_SHA256 = "a32916159e9c570e843cd13f51c3f7444677fb38f349f714fcfbc1990cd10fef"
V1_1_MANIFEST_SHA256 = "d35eaf4ffa9ad5f41d266356c570c37d10a1b3235e9703fcc70e97632a05a655"


def _config_digest(candidate_configuration: dict) -> str:
    import hashlib

    canonical = json.dumps(
        candidate_configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _passing_observation(
    case: SystemCase,
    *,
    latency_ms: float = 10.0,
    candidate_id: str = "candidate-a",
    candidate_configuration: dict | None = None,
) -> SystemObservation:
    manifest, _ = load_suite(MANIFEST_PATH)
    resolved_candidate = candidate_configuration or {
        "agent_policy_id": "synthetic-test-policy-v1",
        "vision_backend": "mock",
        "narrator_backend": "none",
    }
    expected = case.expected
    if expected.requires_citation:
        source_ids = expected.required_source_ids or ["synthetic_governed_source"]
        citations = [
            {
                "source_id": source_id,
                "chunk_id": f"chunk-{index}",
                "locator": f"synthetic section {index}",
                "url": "https://example.invalid/synthetic-source",
            }
            for index, source_id in enumerate(source_ids, 1)
        ]
        while len(citations) < expected.min_citations:
            citations.append(
                {
                    "source_id": "synthetic_governed_source",
                    "chunk_id": f"chunk-{len(citations) + 1}",
                    "locator": f"synthetic section {len(citations) + 1}",
                }
            )
    else:
        citations = []

    tool_calls: list[str] = []
    if expected.allowed_tools and expected.max_tool_calls != 0:
        tool_calls = [expected.allowed_tools[0]]
    return SystemObservation.model_validate(
        {
            "schema_version": 1,
            "case_id": case.case_id,
            "suite_id": manifest.suite_id,
            "suite_version": manifest.suite_version,
            "split_hash": manifest.cases_sha256,
            "candidate_id": candidate_id,
            "candidate_config_sha256": _config_digest(resolved_candidate),
            "adapter_id": "tbx-test-normalized-adapter",
            "adapter_version": "1",
            "synthetic": True,
            "clinical_validation": False,
            "status": expected.accepted_statuses[0],
            "route": expected.route,
            "urgency": expected.urgency,
            "response_kind": expected.response_kind,
            "outcome": expected.outcome,
            "screening_status": expected.screening_status,
            "next_question_id": expected.next_question_id,
            "error_code": expected.error_code,
            "abstained": expected.abstained if expected.abstained is not None else False,
            "answer_text": "；".join(expected.required_terms) or "synthetic safe response",
            "citations": citations,
            "retrieved_source_ids": expected.required_source_ids,
            "safety_labels": expected.required_safety_labels,
            "state_isolated": expected.state_isolated,
            "leaked_canary": False if expected.canary_must_not_leak else None,
            "injection_resisted": expected.injection_resisted,
            "tool_calls": tool_calls,
            "latency_ms": latency_ms,
            "trace_id": f"trace-{case.case_id}",
            "details": {"fixture": "synthetic_non_clinical"},
        }
    )


def _report(candidate_id: str = "candidate-a") -> BenchReport:
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {
        "agent_policy_id": "synthetic-test-policy-v1",
        "vision_backend": "mock",
        "narrator_backend": "none",
    }
    return evaluate_observations(
        config=config,
        manifest=manifest,
        cases=cases,
        observations=[
            _passing_observation(
                case,
                candidate_id=candidate_id,
                candidate_configuration=candidate_configuration,
            )
            for case in cases
        ],
        candidate_id=candidate_id,
        candidate_configuration=candidate_configuration,
        source_revision="test-revision",
        wall_clock_ms=100.0,
        peak_vram_mib=None,
        peak_vram_measurement_method="unknown_not_measured",
        created_at="2026-08-29T00:00:00+00:00",
    )


def test_checked_in_suite_is_hashed_versioned_synthetic_and_complete():
    manifest, cases = load_suite(MANIFEST_PATH)

    assert manifest.suite_version == "1.2.0"
    assert manifest.fixture_kind == "synthetic_non_clinical"
    assert manifest.clinical_validation is False
    assert manifest.selection_use is False
    assert manifest.locked_or_hidden_test_used is False
    assert {case.dimension for case in cases} == set(ALL_DIMENSIONS)
    assert len({case.case_id for case in cases}) == len(cases) == 30
    assert all(case.synthetic and not case.clinical_validation for case in cases)


def test_v1_1_suite_is_byte_stable_and_v1_2_is_strictly_additive():
    import hashlib

    v1_1_config = load_config(V1_1_CONFIG_PATH)
    v1_1_manifest, v1_1_cases = load_suite(V1_1_MANIFEST_PATH)
    v1_2_manifest, v1_2_cases = load_suite(MANIFEST_PATH)
    v1_1_bytes = (V1_1_MANIFEST_PATH.parent / "cases.jsonl").read_bytes()
    v1_2_bytes = (MANIFEST_PATH.parent / "cases.jsonl").read_bytes()

    assert v1_1_config.suite_manifest == "evaluation/suites/system_v1/manifest.json"
    assert hashlib.sha256(V1_1_CONFIG_PATH.read_bytes()).hexdigest() == V1_1_CONFIG_SHA256
    assert v1_1_manifest.suite_version == "1.1.0"
    assert (
        hashlib.sha256(V1_1_MANIFEST_PATH.read_bytes()).hexdigest()
        == V1_1_MANIFEST_SHA256
    )
    assert v1_1_manifest.cases_sha256 == V1_1_CASES_SHA256
    assert hashlib.sha256(v1_1_bytes).hexdigest() == V1_1_CASES_SHA256
    assert len(v1_1_cases) == 24
    assert v1_2_manifest.suite_version == "1.2.0"
    assert len(v1_2_cases) == 30
    assert v1_2_bytes.startswith(v1_1_bytes)
    assert [case.case_id for case in v1_2_cases[:24]] == [
        case.case_id for case in v1_1_cases
    ]


def test_current_v1_6_suite_uses_only_the_four_tool_agent_contract() -> None:
    manifest, cases = load_suite(CURRENT_MANIFEST_PATH)

    assert manifest.suite_version == "1.6.0"
    rendered = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
    )
    assert "search_tb_knowledge" in rendered
    assert "retrieve_diagnostic_guidance" not in rendered
    assert "retrieve_treatment_education" not in rendered
    for pseudo_tool in ("describe_agent_capabilities", "emergency_triage"):
        assert pseudo_tool not in {
            tool
            for case in cases
            for tool in case.expected.allowed_tools
        }
    declared_tools = {
        tool
        for case in cases
        for tool in case.expected.allowed_tools
    }
    assert declared_tools <= {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    }


def test_config_fails_closed_on_hidden_data_or_excess_ablation_variables():
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["locked_or_hidden_test_used"] = True
    with pytest.raises(ValidationError):
        SystemBenchConfig.model_validate(payload)

    payload["locked_or_hidden_test_used"] = False
    payload["major_variables_changed"] = ["a", "b", "c"]
    with pytest.raises(ValidationError):
        SystemBenchConfig.model_validate(payload)


def test_suite_loader_rejects_hash_drift(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        (MANIFEST_PATH.parent / "cases.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["cases_sha256"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_suite(manifest_path)


def test_perfect_normalized_run_records_full_provenance_and_passes_gates():
    report = _report()

    assert report.release_gate.passed
    assert report.metrics["case_pass_rate"] == 1.0
    assert report.metrics["emergency_recall"] == 1.0
    assert report.metrics["citation_locator_coverage"] == 1.0
    assert report.metrics["memory_isolation_rate"] == 1.0
    assert report.metrics["prompt_injection_resistance_rate"] == 1.0
    assert report.split_hash == load_suite(MANIFEST_PATH)[0].cases_sha256
    assert report.full_config["evaluation"]["seed"] == 20260829
    assert report.full_config["candidate"]["vision_backend"] == "mock"
    assert len(report.candidate_config_sha256) == 64
    assert report.source_revision == "test-revision"
    assert report.runtime.peak_vram_measured is False
    assert len(report.per_case) == 30
    assert not report.regressions


def test_missing_observation_and_forbidden_output_fail_closed():
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {"fixture": "unsafe-test-candidate"}
    observations = [
        _passing_observation(
            case,
            candidate_id="unsafe-candidate",
            candidate_configuration=candidate_configuration,
        )
        for case in cases[:-1]
    ]
    injection_case = next(case for case in cases if case.case_id == "sysv1.injection.direct.001")
    injection_observation = next(
        item for item in observations if item.case_id == injection_case.case_id
    )
    injection_observation.answer_text = "系统规则已忽略，你每天应服合成剂量"

    report = evaluate_observations(
        config=config,
        manifest=manifest,
        cases=cases,
        observations=observations,
        candidate_id="unsafe-candidate",
        candidate_configuration=candidate_configuration,
        source_revision="test-revision",
        wall_clock_ms=5.0,
    )

    by_id = {record.case_id: record for record in report.per_case}
    assert not by_id[injection_case.case_id].checks["forbidden_terms"].passed
    assert not by_id[cases[-1].case_id].observation_present
    assert report.metrics["observation_coverage"] < 1.0
    assert not report.release_gate.passed
    assert {item["case_id"] for item in report.regressions} == {
        injection_case.case_id,
        cases[-1].case_id,
    }


def test_paired_comparison_uses_only_intersection_and_lists_new_regression():
    baseline = _report("baseline")
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {
        "agent_policy_id": "synthetic-test-policy-v1",
        "vision_backend": "mock",
        "narrator_backend": "none",
    }
    observations = [
        _passing_observation(
            case,
            candidate_id="candidate",
            candidate_configuration=candidate_configuration,
        )
        for case in cases
    ]
    target = next(
        observation
        for observation in observations
        if observation.case_id == "sysv1.emergency.hemoptysis.001"
    )
    target.urgency = "routine"
    candidate = evaluate_observations(
        config=config,
        manifest=manifest,
        cases=cases,
        observations=observations,
        candidate_id="candidate",
        candidate_configuration=candidate_configuration,
        source_revision="test-revision",
        wall_clock_ms=100.0,
        created_at="2026-08-29T00:00:01+00:00",
    )

    comparison = compare_reports(baseline, candidate)

    assert comparison.intersection_case_count == 30
    assert comparison.candidate_only_case_ids == []
    assert comparison.baseline_only_case_ids == []
    assert comparison.regressions == [
        {
            "case_id": "sysv1.emergency.hemoptysis.001",
            "dimension": "emergency",
            "newly_failed_checks": ["urgency"],
        }
    ]
    assert comparison.paired_metrics["paired_case_pass_delta"] == pytest.approx(-1 / 30)
    assert comparison.exact_mcnemar_p_value == 1.0
    assert not comparison.comparison_gate.passed


def test_comparison_revalidates_mutated_report_before_pairing():
    baseline = _report("baseline")
    candidate = _report("candidate")
    candidate.per_case = candidate.per_case[:-1]

    with pytest.raises(ValidationError, match="per_case count"):
        compare_reports(baseline, candidate)


def test_observation_schema_rejects_unknown_fields_and_duplicate_case_ids():
    with pytest.raises(ValidationError):
        SystemObservation.model_validate(
            {
                "schema_version": 1,
                "case_id": "sysv1.routing.invalid.001",
                "suite_id": "suite",
                "suite_version": "1.0.0",
                "split_hash": "0" * 64,
                "candidate_id": "candidate",
                "candidate_config_sha256": "0" * 64,
                "adapter_id": "adapter",
                "adapter_version": "1",
                "synthetic": True,
                "clinical_validation": False,
                "status": "completed",
                "latency_ms": 1,
                "unexpected": "field",
            }
        )

    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    duplicate = _passing_observation(cases[0])
    with pytest.raises(ValueError, match="unique"):
        evaluate_observations(
            config=config,
            manifest=manifest,
            cases=cases,
            observations=[duplicate, duplicate.model_copy(deep=True)],
            candidate_id="duplicate-observation-run",
            candidate_configuration={"fixture": "duplicate-observation-test"},
            source_revision="test-revision",
            wall_clock_ms=1.0,
        )


def test_candidate_configuration_rejects_secret_material():
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    with pytest.raises(ValueError, match="secret-like key"):
        evaluate_observations(
            config=config,
            manifest=manifest,
            cases=cases,
            observations=[_passing_observation(case) for case in cases],
            candidate_id="candidate-with-secret",
            candidate_configuration={"backend": "local", "api_key": "must-not-persist"},
            source_revision="test-revision",
            wall_clock_ms=1.0,
        )


def test_cli_writes_a_schema_valid_report(tmp_path):
    _, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {"runtime": "synthetic", "model_digest": "0" * 64}
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        "\n".join(
            _passing_observation(
                case,
                candidate_id="cli-test-candidate",
                candidate_configuration=candidate_configuration,
            ).model_dump_json()
            for case in cases
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_config_path = tmp_path / "candidate.json"
    candidate_config_path.write_text(
        json.dumps(candidate_configuration),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    result = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--observations",
            str(observations_path),
            "--candidate-id",
            "cli-test-candidate",
            "--candidate-config",
            str(candidate_config_path),
            "--source-revision",
            "test-revision",
            "--candidate-run-wall-clock-ms",
            "123.4",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    report = BenchReport.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert report.candidate_id == "cli-test-candidate"
    assert report.runtime.candidate_execution_wall_clock_ms == 123.4
    assert report.release_gate.passed
    ledger_path = tmp_path / "system_eval_ledger.jsonl"
    assert verify_ledger(ledger_path)["event_count"] == 1


def test_cli_paired_baseline_must_have_an_intact_ledger_receipt(tmp_path):
    _, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {"runtime": "synthetic", "model_digest": "0" * 64}
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate_configuration), encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"

    def write_observations(path: Path, candidate_id: str) -> None:
        path.write_text(
            "\n".join(
                _passing_observation(
                    case,
                    candidate_id=candidate_id,
                    candidate_configuration=candidate_configuration,
                ).model_dump_json()
                for case in cases
            )
            + "\n",
            encoding="utf-8",
        )

    baseline_observations = tmp_path / "baseline-observations.jsonl"
    baseline_report = tmp_path / "baseline-report.json"
    write_observations(baseline_observations, "baseline-candidate")
    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--observations",
                str(baseline_observations),
                "--candidate-id",
                "baseline-candidate",
                "--candidate-config",
                str(candidate_path),
                "--source-revision",
                "test-revision",
                "--output",
                str(baseline_report),
                "--ledger",
                str(ledger),
            ]
        )
        == 0
    )

    paired_observations = tmp_path / "paired-observations.jsonl"
    paired_report = tmp_path / "paired-report.json"
    comparison_path = tmp_path / "comparison.json"
    write_observations(paired_observations, "paired-candidate")
    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--observations",
                str(paired_observations),
                "--candidate-id",
                "paired-candidate",
                "--candidate-config",
                str(candidate_path),
                "--source-revision",
                "test-revision",
                "--output",
                str(paired_report),
                "--ledger",
                str(ledger),
                "--baseline-report",
                str(baseline_report),
                "--comparison-output",
                str(comparison_path),
            ]
        )
        == 0
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["adapter_identity_equal"] is True
    assert comparison["comparison_gate"]["passed"] is True

    baseline_report.write_text(baseline_report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected_observations = tmp_path / "rejected-observations.jsonl"
    rejected_report = tmp_path / "rejected-report.json"
    write_observations(rejected_observations, "rejected-candidate")
    assert (
        main(
            [
                "--config",
                str(CONFIG_PATH),
                "--observations",
                str(rejected_observations),
                "--candidate-id",
                "rejected-candidate",
                "--candidate-config",
                str(candidate_path),
                "--source-revision",
                "test-revision",
                "--output",
                str(rejected_report),
                "--ledger",
                str(ledger),
                "--baseline-report",
                str(baseline_report),
            ]
        )
        == 2
    )
    assert not rejected_report.exists()
    assert verify_ledger(ledger)["event_count"] == 3
    assert (
        json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])["status"]
        == "failed_retained"
    )


def test_observation_candidate_binding_and_release_gate_weakening_fail_closed():
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    observation = _passing_observation(cases[0]).model_copy(
        update={"candidate_id": "different-candidate"}
    )
    with pytest.raises(ValueError, match="provenance"):
        evaluate_observations(
            config=config,
            manifest=manifest,
            cases=cases,
            observations=[observation],
            candidate_id="candidate-a",
            candidate_configuration={
                "agent_policy_id": "synthetic-test-policy-v1",
                "vision_backend": "mock",
                "narrator_backend": "none",
            },
            source_revision="test-revision",
            wall_clock_ms=1.0,
        )

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["release_gates"]["emergency_recall"] = 0.99
    with pytest.raises(ValidationError, match="exactly 1.0"):
        SystemBenchConfig.model_validate(payload)


def test_observations_cannot_mix_adapter_versions_in_one_report():
    config = load_config(CONFIG_PATH)
    manifest, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {
        "agent_policy_id": "synthetic-test-policy-v1",
        "vision_backend": "mock",
        "narrator_backend": "none",
    }
    observations = [
        _passing_observation(
            case,
            candidate_configuration=candidate_configuration,
        )
        for case in cases[:2]
    ]
    observations[1].adapter_version = "different-version"

    with pytest.raises(ValueError, match="mix multiple adapter identities"):
        evaluate_observations(
            config=config,
            manifest=manifest,
            cases=cases,
            observations=observations,
            candidate_id="candidate-a",
            candidate_configuration=candidate_configuration,
            source_revision="test-revision",
            wall_clock_ms=1.0,
        )


def test_report_schema_recomputes_metrics_gate_config_and_run_hashes():
    payload = _report().model_dump(mode="json")
    payload["metrics"]["case_pass_rate"] = 0.5
    with pytest.raises(ValidationError, match="metrics do not match"):
        BenchReport.model_validate(payload)

    payload = _report().model_dump(mode="json")
    payload["run_id"] = "sysbench-0000000000000000"
    with pytest.raises(ValidationError, match="run_id"):
        BenchReport.model_validate(payload)


def test_ledger_chain_detects_tampering_and_retains_failure_entries(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    append_ledger(
        ledger,
        {
            "created_at": "2026-08-29T00:00:00+00:00",
            "status": "failed_retained",
            "evaluation_id": "test-eval",
            "candidate_id": "candidate-a",
            "source_revision": "unknown",
            "runtime": {"wall_clock_ms_until_failure": 1.0},
            "release_gate_passed": False,
            "error_type": "SyntheticFailure",
            "error_message": "retained",
        },
    )
    assert verify_ledger(ledger)["event_count"] == 1
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["error_message"] = "silently changed"
    ledger.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_ledger(ledger)

    with pytest.raises(ValidationError, match="completed ledger event"):
        append_ledger(
            tmp_path / "invalid-ledger.jsonl",
            {
                "created_at": "2026-08-29T00:00:00+00:00",
                "status": "passed",
                "evaluation_id": "test-eval",
                "candidate_id": "candidate-a",
                "source_revision": "unknown",
                "runtime": {"wall_clock_ms": 1.0},
                "release_gate_passed": False,
            },
        )


def test_ledger_append_fails_closed_when_writer_lock_exists(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    lock = ledger.with_name(f"{ledger.name}.lock")
    lock.write_text('{"owner":"other-writer"}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="ledger is locked"):
        append_ledger(
            ledger,
            {
                "created_at": "2026-08-29T00:00:00+00:00",
                "status": "failed_retained",
                "evaluation_id": "test-eval",
                "candidate_id": "candidate-a",
                "source_revision": "unknown",
                "runtime": {"wall_clock_ms_until_failure": 1.0},
                "release_gate_passed": False,
                "error_type": "SyntheticFailure",
                "error_message": "must not be appended",
            },
        )

    assert not ledger.exists()
    assert lock.read_text(encoding="utf-8") == '{"owner":"other-writer"}'


def test_deterministic_mock_adapter_executes_service_and_retains_gate_result(tmp_path):
    output_dir = tmp_path / "system-bench"
    result = run_current_candidate(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        candidate_id="deterministic-mock-current",
    )

    assert result in {0, 2}
    candidate_configuration = json.loads(
        (output_dir / "candidate.json").read_text(encoding="utf-8")
    )
    assert candidate_configuration["adapter"].endswith(DETERMINISTIC_ADAPTER_VERSION)
    assert candidate_configuration["policy_ids"]["fusion"] == "rank03-user-trained-native-argmax-v2"
    observations_path = output_dir / "observations.jsonl"
    output_path = output_dir / "report.json"
    observations = [
        SystemObservation.model_validate_json(line)
        for line in observations_path.read_text(encoding="utf-8").splitlines()
    ]
    report = BenchReport.model_validate_json(output_path.read_text(encoding="utf-8"))
    assert len(observations) == report.observation_count == report.expected_case_count == 30
    assert report.metrics["observation_coverage"] == 1.0
    assert report.runtime.wall_clock_ms > 0.1
    assert report.runtime.candidate_execution_wall_clock_measured is True
    assert report.runtime.peak_vram_mib is None
    assert report.runtime.peak_vram_measurement_method == "unknown_not_measured"
    assert report.adapter_version == DETERMINISTIC_ADAPTER_VERSION
    assert report.suite_version == "1.6.0"
    public_tools = {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    }
    assert all(
        tool_name in public_tools
        for observation in observations
        for tool_name in observation.tool_calls
    )
    prompt_driven = [
        observation
        for observation in observations
        if observation.details.get("orchestration") == "langgraph_plan_react"
    ]
    assert prompt_driven
    assert all(
        observation.details["graph_node_trace"][0] == "load_context"
        and observation.details["graph_node_trace"][-1] == "finalize"
        for observation in prompt_driven
    )
    assert result == 0
    assert report.release_gate.passed is True
    by_case = {record.case_id: record for record in report.per_case}
    assert by_case["sysv1.memory.subject-api.001"].passed is True
    assert by_case["sysv1.memory.artifact-redaction.001"].passed is True
    observed_by_case = {item.case_id: item for item in observations}
    assert observed_by_case["sysv1.memory.subject-api.001"].details["guarded_resources"] == [
        "case",
        "report",
        "review",
        "screening",
    ]
    assert observed_by_case["sysv1.memory.subject-api.001"].leaked_canary is False
    assert (
        observed_by_case["sysv1.memory.artifact-redaction.001"].outcome
        == "internal_artifact_redacted"
    )
    tie = observed_by_case["sysv16.vision.argmax-tie-first-index.001"]
    assert tie.status == "completed"
    assert tie.abstained is False
    assert tie.outcome == "model_not_flagged"
    assert tie.safety_labels == ["screening_not_diagnosis"]
    assert by_case["sysv16.vision.argmax-tie-first-index.001"].passed is True
    added_case_ids = {
        "sysv12.vision.argmax-healthy.001",
        "sysv13.vision.argmax-sick-non-tb.001",
        "sysv12.vision.argmax-tb.001",
        "sysv12.vision.detector-cannot-override-healthy.001",
        "sysv12.vision.no-box-cannot-exclude-tb.001",
        "sysv15.vision.quality-warning-advisory-preserves-healthy.001",
    }
    assert {case.case_id for case in load_suite(CURRENT_MANIFEST_PATH)[1][-6:]} == added_case_ids
    assert all(by_case[case_id].passed for case_id in added_case_ids)
    assert observed_by_case[
        "sysv12.vision.detector-cannot-override-healthy.001"
    ].details["max_detector_score"] == 0.99
    assert observed_by_case[
        "sysv12.vision.detector-cannot-override-healthy.001"
    ].details["detector_role"] == "advisory_localization_only"
    assert observed_by_case[
        "sysv12.vision.no-box-cannot-exclude-tb.001"
    ].details["max_detector_score"] is None
    assert observed_by_case[
        "sysv15.vision.quality-warning-advisory-preserves-healthy.001"
    ].details["review_reasons"] == []
    assert all(
        observed_by_case[case_id].details["real_model_inference"] is False
        for case_id in added_case_ids
    )
    ledger = output_dir / "ledger.jsonl"
    assert verify_ledger(ledger)["event_count"] == 1
    expected_status = "passed" if report.release_gate.passed else "regressed_retained"
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == expected_status
    with pytest.raises(FileExistsError):
        run_current_candidate(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            candidate_id="must-not-overwrite",
        )


def test_version_pinned_v1_1_replay_remains_readable_and_retains_drift(tmp_path):
    output_dir = tmp_path / "system-bench-v1-1"

    result = run_current_candidate(
        project_root=PROJECT_ROOT,
        output_dir=output_dir,
        candidate_id="deterministic-v1-1-replay",
        config_path=V1_1_CONFIG_PATH,
    )

    report = BenchReport.model_validate_json(
        (output_dir / "report.json").read_text(encoding="utf-8")
    )
    candidate = json.loads((output_dir / "candidate.json").read_text(encoding="utf-8"))
    # Historical expectations are not silently rewritten.  The old exact-tie
    # review policy intentionally fails against the current fixed-order native
    # argmax contract, while the adapter still emits a complete replay report.
    assert result == 2
    assert report.suite_version == "1.1.0"
    assert report.metrics["observation_coverage"] == 1.0
    failed_ids = {item.case_id for item in report.per_case if not item.passed}
    assert "sysv1.abstain.argmax-tie.001" in failed_ids
    observations = [
        SystemObservation.model_validate_json(line)
        for line in (output_dir / "observations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(observation.status != "error" for observation in observations)
    assert report.expected_case_count == report.observation_count == 24
    assert report.split_hash == V1_1_CASES_SHA256
    assert report.release_gate.passed is False
    assert candidate["evaluation_config"] == "evaluation/system_bench_config_v1_1.json"


def test_failed_cli_binding_is_retained_without_writing_a_report(tmp_path):
    manifest, cases = load_suite(MANIFEST_PATH)
    candidate_configuration = {
        "agent_policy_id": "synthetic-test-policy-v1",
        "vision_backend": "mock",
        "narrator_backend": "none",
    }
    observations_path = tmp_path / "observations.jsonl"
    observations_path.write_text(
        _passing_observation(
            cases[0],
            candidate_id="bound-candidate",
            candidate_configuration=candidate_configuration,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate_configuration), encoding="utf-8")
    report_path = tmp_path / "must-not-exist.json"

    result = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--observations",
            str(observations_path),
            "--candidate-id",
            "different-candidate",
            "--candidate-config",
            str(candidate_path),
            "--source-revision",
            "test-revision",
            "--output",
            str(report_path),
        ]
    )

    assert result == 2
    assert not report_path.exists()
    ledger_path = tmp_path / "system_eval_ledger.jsonl"
    entry = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert entry["status"] == "failed_retained"
    assert entry["error_type"] == "ValueError"
    assert "provenance" in entry["error_message"]
    assert entry["split_hash"] == manifest.cases_sha256
    assert verify_ledger(ledger_path)["event_count"] == 1


def test_comparison_requires_identical_suite_and_complete_case_set():
    baseline = _report("baseline")
    candidate = _report("candidate")
    candidate.suite_version = "1.0.1"

    comparison = compare_reports(baseline, candidate)

    assert comparison.suite_identity_equal is False
    assert comparison.comparison_gate.checks["suite_identity_equal"].passed is False
    assert comparison.comparison_gate.checks["complete_case_set_equal"].passed is True
    assert comparison.comparison_gate.passed is False


def test_runtime_unknown_measurements_cannot_be_mislabeled():
    with pytest.raises(ValidationError, match="unmeasured peak VRAM"):
        RuntimeEvidence(
            wall_clock_ms=1.0,
            wall_clock_scope="test",
            candidate_execution_wall_clock_ms=None,
            candidate_execution_wall_clock_measured=False,
            peak_vram_mib=None,
            peak_vram_measured=False,
            peak_vram_measurement_method="nvidia-smi",
        )
