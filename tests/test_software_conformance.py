from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tbx_agent.evaluation import software_conformance as swconf
from tbx_agent.evaluation.software_conformance import (
    KNOWN_CONTRACTS,
    load_suite,
    main,
    run_conformance,
    verify_ledger,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "software_conformance_config_v1.json"
V12_MANIFEST = PROJECT_ROOT / "evaluation" / "suites" / "system_v1_2" / "manifest.json"


def test_versioned_suite_is_hash_bound_and_does_not_repurpose_system_v12() -> None:
    config, manifest, cases, _manifest_path = load_suite(PROJECT_ROOT, CONFIG_PATH)

    assert config.selection_use is False
    assert config.locked_or_hidden_test_used is False
    assert config.major_variables_changed == ["software_conformance_contract"]
    assert manifest.split_kind == "none_synthetic_software_suite"
    assert manifest.clinical_validation is False
    assert manifest.selection_use is False
    assert manifest.locked_or_hidden_test_used is False
    assert len(cases) == 8
    assert {case.contract for case in cases} == set(KNOWN_CONTRACTS)

    v12 = json.loads(V12_MANIFEST.read_text(encoding="utf-8"))
    assert v12["suite_version"] == "1.2.0"
    assert v12["cases_sha256"] == (
        "feaa80e49bebf3d0e7fa9ac7b7cfaf1f8b2e3c04bf7332163483aeef10a28c74"
    )
    assert v12["expected_case_count"] == 30


def test_runner_records_complete_nonclinical_provenance_and_passes(tmp_path: Path) -> None:
    output_dir = tmp_path / "run-1"
    ledger = tmp_path / "software_conformance_ledger.jsonl"

    report, report_path = run_conformance(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        ledger_path=ledger,
        candidate_id="pytest-source-candidate",
    )

    assert report_path == output_dir / "report.json"
    assert report_path.is_file()
    assert report["status"] == "passed"
    assert report["release_gate"] == {"required_case_pass_rate": 1.0, "passed": True}
    assert report["metrics"]["case_count"] == 8
    assert report["metrics"]["passed_case_count"] == 8
    assert report["metrics"]["failed_case_count"] == 0
    assert report["metrics"]["case_pass_rate"] == 1.0
    assert report["seed"] == 20260830
    assert report["split"]["kind"] == "none_synthetic_software_suite"
    assert len(report["split"]["hash"]) == 64
    assert report["selection_use"] is False
    assert report["locked_or_hidden_test_used"] is False
    assert report["clinical_validation"] is False
    assert report["full_config"]["execution"]["network_access"] is False
    assert report["full_config"]["execution"]["model_weights_loaded"] is False
    assert report["full_config"]["execution"]["dataset_access"] is False
    assert report["source_revision"]["candidate_source_sha256"] == (
        report["full_config"]["candidate"]["source_tree_sha256"]
    )
    assert report["runtime"]["candidate_execution_wall_clock_ms"] >= 0
    assert report["runtime"]["peak_vram_mib"] is None
    assert report["runtime"]["peak_vram_measured"] is False
    assert "not_measured" in report["runtime"]["peak_vram_reason"]
    assert {item["contract"] for item in report["per_case"]} == set(KNOWN_CONTRACTS)
    assert all(item["passed"] for item in report["per_case"])

    audit = verify_ledger(ledger)
    assert audit["valid"] is True
    assert audit["event_count"] == 1
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["status"] == "passed"
    assert event["hypothesis"] == report["hypothesis"]
    assert event["full_config"] == report["full_config"]
    assert event["seed"] == report["seed"]
    assert event["split_hash"] == report["split"]["hash"]
    assert event["source_revision"] == report["source_revision"]
    assert event["metrics"] == report["metrics"]
    assert event["runtime"] == report["runtime"]


def test_existing_output_is_not_modified_and_failed_attempt_is_retained(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "immutable-run"
    ledger = tmp_path / "software_conformance_ledger.jsonl"
    report, report_path = run_conformance(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        ledger_path=ledger,
        candidate_id="pytest-source-candidate",
    )
    assert report["status"] == "passed"
    original_report = report_path.read_bytes()
    original_digest = hashlib.sha256(original_report).hexdigest()

    exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--output-dir",
            str(output_dir),
            "--ledger",
            str(ledger),
            "--candidate-id",
            "pytest-source-candidate",
        ]
    )

    assert exit_code == 2
    assert report_path.read_bytes() == original_report
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == original_digest
    assert not (output_dir / "failure.json").exists()
    assert verify_ledger(ledger)["event_count"] == 2
    events = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert [item["status"] for item in events] == ["passed", "failed_retained"]
    failed = events[-1]
    assert failed["hypothesis"]
    assert failed["full_config"]
    assert failed["seed"] == 20260830
    assert len(failed["split_hash"]) == 64
    assert failed["source_revision"]["candidate_source_sha256"]
    assert failed["metrics"] == {}
    assert failed["runtime"]["peak_vram_mib"] is None
    assert "not_measured" in failed["runtime"]["peak_vram_reason"]
    assert failed["error"]["type"] == "FileExistsError"


def test_contract_regression_report_and_ledger_are_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = "dicom_canonical_metadata_free"
    original_runner = swconf.CASE_RUNNERS[contract]

    def regressed_contract():
        checks = original_runner()
        checks["source_format"] = {
            **checks["source_format"],
            "passed": False,
            "observed": "synthetic_regression_injected_by_test",
        }
        return checks

    monkeypatch.setitem(swconf.CASE_RUNNERS, contract, regressed_contract)
    output_dir = tmp_path / "regressed-run"
    ledger = tmp_path / "software_conformance_ledger.jsonl"

    report, report_path = run_conformance(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        ledger_path=ledger,
        candidate_id="pytest-regressed-candidate",
    )

    assert report_path.is_file()
    assert report["status"] == "regressed_retained"
    assert report["release_gate"]["passed"] is False
    assert report["metrics"]["failed_case_count"] == 1
    event = json.loads(ledger.read_text(encoding="utf-8"))
    assert event["status"] == "regressed_retained"
    assert event["report_path"] == str(report_path)
    assert event["metrics"] == report["metrics"]
    assert verify_ledger(ledger)["event_count"] == 1


def test_ledger_tamper_is_detected(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    ledger = tmp_path / "software_conformance_ledger.jsonl"
    run_conformance(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        ledger_path=ledger,
        candidate_id="pytest-source-candidate",
    )
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["candidate_id"] = "tampered-candidate"
    ledger.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ledger hash mismatch"):
        verify_ledger(ledger)
