from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.evaluation.shenzhen import (
    LEGACY_SHENZHEN_EVALUATION_ID,
    LEGACY_SHENZHEN_FUSION_POLICY_FILENAME,
    LEGACY_SHENZHEN_FUSION_POLICY_ID,
    ExternalEvaluationContractError,
    _canonical_split_manifest,
    _fusion_policy_binding,
    _read_zip_entry_bounded,
    _sha256_bytes,
    _sha256_file,
    run_shenzhen_evaluation,
)
from tbx_agent.schemas import DetectionEvidence, VisionEvidence
from tbx_agent.vision.base import VisionBackendError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_POLICY_ID = "rank03-agent-screening-demo-cls-argmax-det-advisory-v2"


def _png(red: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (512, 512), color=(red, 20, 30)).save(stream, format="PNG")
    return stream.getvalue()


def _build_archive(path: Path, cases: list[tuple[str, int]]) -> bytes:
    metadata_stream = io.StringIO(newline="")
    writer = csv.DictWriter(metadata_stream, fieldnames=["study_id", "sex", "age", "findings"])
    writer.writeheader()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for study_id, red in cases:
            label = int(study_id.rsplit("_", 1)[1].split(".", 1)[0])
            archive.writestr(f"images/images/{study_id}", _png(red))
            writer.writerow(
                {
                    "study_id": study_id,
                    "sex": "",
                    "age": "",
                    "findings": "normal" if label == 0 else "PTB",
                }
            )
        metadata = metadata_stream.getvalue().encode("utf-8")
        archive.writestr("shenzhen_metadata.csv", metadata)
    return metadata


def _write_config(
    path: Path,
    *,
    archive_path: Path,
    metadata: bytes,
    cases: list[tuple[str, int]],
) -> Path:
    with zipfile.ZipFile(archive_path, "r") as archive:
        samples = []
        for study_id, _ in cases:
            info = archive.getinfo(f"images/images/{study_id}")
            label = int(study_id.rsplit("_", 1)[1].split(".", 1)[0])
            samples.append(
                {
                    "study_id": study_id,
                    "label": label,
                    "entry_name": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "uncompressed_size": info.file_size,
                }
            )
    split_hash, _ = _canonical_split_manifest(samples)
    label_counts = {
        "0": sum(study_id.endswith("_0.png") for study_id, _ in cases),
        "1": sum(study_id.endswith("_1.png") for study_id, _ in cases),
    }
    config = {
        "schema_version": 1,
        "evaluation_id": "small-shenzhen-contract-test",
        "expected_fusion_policy_id": ACTIVE_POLICY_ID,
        "hypothesis": "The injected test backend exercises the external runner contract.",
        "seed": 1234,
        "seed_use": "Confidence intervals only; no sampling or shuffle.",
        "selection_use": False,
        "threshold_selection": False,
        "calibration": False,
        "external_test_only": True,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "major_variables_changed": ["evaluation_dataset"],
        "dataset": {
            "dataset_id": "small-test",
            "expected_archive_bytes": archive_path.stat().st_size,
            "expected_archive_sha256": _sha256_file(archive_path),
            "metadata_entry": "shenzhen_metadata.csv",
            "expected_metadata_sha256": _sha256_bytes(metadata),
            "image_entry_prefix": "images/images/",
            "study_id_pattern": r"^CHNCXR_\d{4}_(?P<label>[01])\.png$",
            "negative_finding": "normal",
            "expected_image_count": len(cases),
            "expected_label_counts": label_counts,
            "expected_split_manifest_sha256": split_hash,
            "bbox_ground_truth_available": False,
        },
        "confidence_intervals": {
            "method": "stratified_bootstrap_percentile_95",
            "replicates": 25,
        },
        "metrics": ["classifier_image_level", "final_operational_triage"],
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    return replace(
        base,
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path / "runtime",
        db_path=tmp_path / "unused.sqlite3",
        artifact_root=tmp_path / "unused-artifacts",
        vision_backend="mock",
        openai_enabled=False,
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        max_upload_bytes=1024 * 1024,
        retain_uploaded_image=True,
    )


class PixelBackend:
    backend_id = "INJECTED_TEST_BACKEND"

    def __init__(self, *, fail_all: bool = False, tie_all: bool = False):
        self.fail_all = fail_all
        self.tie_all = tie_all
        self.call_count = 0

    def infer(self, *, case_id: str, image) -> VisionEvidence:
        self.call_count += 1
        red = image.image.getpixel((0, 0))[0]
        if self.fail_all or red >= 230:
            raise VisionBackendError("synthetic backend failure")
        classifier_flagged = red >= 80 and not self.tie_all
        detector_flagged = red >= 160
        probabilities = (
            {"healthy": 0.45, "sick_non_tb": 0.10, "tb": 0.45}
            if self.tie_all
            else {"healthy": 0.05, "sick_non_tb": 0.05, "tb": 0.9}
            if classifier_flagged
            else {"healthy": 0.94, "sick_non_tb": 0.05, "tb": 0.01}
        )
        detections = (
            [
                DetectionEvidence(
                    bbox_xyxy=(10.0, 10.0, 100.0, 100.0),
                    score=0.9,
                    label="tb_lesion_candidate",
                )
            ]
            if detector_flagged
            else []
        )
        return VisionEvidence(
            run_id=f"test-{self.call_count}",
            case_id=case_id,
            image_sha256=image.sha256,
            image_quality_status=image.quality_status,
            image_width=image.width,
            image_height=image.height,
            classifier_model_id="INJECTED_TEST_CLASSIFIER",
            classifier_checkpoint_sha256="b" * 64,
            class_probability_order=["healthy", "sick_non_tb", "tb"],
            class_probabilities=probabilities,
            classifier_decision_rule="native_three_class_argmax",
            predicted_class=(None if self.tie_all else "tb" if classifier_flagged else "healthy"),
            classifier_argmax_tied=self.tie_all,
            classifier_threshold=None,
            classifier_flagged=classifier_flagged,
            detector_model_id="INJECTED_TEST_DETECTOR",
            detector_checkpoint_sha256="c" * 64,
            detector_decision_role="advisory_localization_only",
            detector_threshold=None,
            detections=detections,
            detector_flagged=None,
            preprocessing_version="injected-test-v1",
            threshold_config_version="injected-test-policy-v1",
            runtime_ms=1,
            artifact_refs=["synthetic_test_output"],
        )


def test_external_runner_streams_zip_through_service_and_retains_partial_failures(
    tmp_path: Path,
) -> None:
    cases = [
        ("CHNCXR_0001_0.png", 10),
        ("CHNCXR_0002_0.png", 100),
        ("CHNCXR_0003_1.png", 180),
        ("CHNCXR_0004_1.png", 240),
    ]
    archive_path = tmp_path / "small-shenzhen.zip"
    metadata = _build_archive(archive_path, cases)
    config_path = _write_config(
        tmp_path / "config.json",
        archive_path=archive_path,
        metadata=metadata,
        cases=cases,
    )
    backend = PixelBackend()
    ledger_path = tmp_path / "ledger.jsonl"

    result_path = run_shenzhen_evaluation(
        _settings(tmp_path),
        archive_path=archive_path,
        output_root=tmp_path / "runs",
        config_path=config_path,
        _vision_backend=backend,
        _ledger_path=ledger_path,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in Path(result["records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert backend.call_count == 4
    assert result["status"] == "completed_with_technical_failures_retained"
    assert result["record_count"] == 4
    assert Path(result["database_path"]).is_file()
    assert not list(tmp_path.rglob("CHNCXR_*.png"))
    assert all(record["duplicate_payload_of"] is None for record in records)
    assert records[0]["classifier"]["tb_probability"] == pytest.approx(0.01)
    assert records[0]["classifier"]["decision_rule"] == "native_three_class_argmax"
    assert records[0]["classifier"]["predicted_class"] == "healthy"
    assert records[0]["classifier"]["threshold"] is None
    assert records[2]["detector"]["max_detector_score"] == pytest.approx(0.9)
    assert records[2]["detector"]["decision_role"] == "advisory_localization_only"
    assert records[2]["detector"]["threshold"] is None
    assert records[2]["detector"]["flagged"] is None
    assert records[3]["technical_failure"]["occurred"] is True
    assert all(record["service"]["response_contract"]["contract_pass"] for record in records[:3])
    assert records[0]["service"]["response_contract"]["narrator_backend"] is None
    assert records[0]["fusion"]["decision_contract"]["contract_pass"] is True
    service_runtime = result["full_configuration"]["service_runtime"]
    assert service_runtime["fusion_policy_filename"] == "fusion_policy_argmax_v2.json"
    assert service_runtime["expected_fusion_policy_id"] == ACTIVE_POLICY_ID
    assert service_runtime["fusion_policy"]["policy_id"] == ACTIVE_POLICY_ID
    assert result["metrics"]["classifier_image_level"]["roc_auc"]["value"] is not None
    assert result["metrics"]["detector_derived_image_level"]["roc_auc"]["value"] is not None
    assert result["metrics"]["detector_derived_image_level"]["evaluable_count"] == 0
    assert (
        result["metrics"]["detector_derived_image_level"]["status"]
        == "not_computed_detector_advisory_only"
    )
    assert (
        result["metrics"]["object_detection_localization"]["status"]
        == "not_computed_no_bbox_ground_truth"
    )
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert ledger[0]["status"] == "completed_with_technical_failures_retained"


def test_all_service_technical_failures_are_failed_and_retained(tmp_path: Path) -> None:
    cases = [
        ("CHNCXR_0001_0.png", 10),
        ("CHNCXR_0002_1.png", 180),
    ]
    archive_path = tmp_path / "all-fail.zip"
    metadata = _build_archive(archive_path, cases)
    config_path = _write_config(
        tmp_path / "all-fail-config.json",
        archive_path=archive_path,
        metadata=metadata,
        cases=cases,
    )
    ledger_path = tmp_path / "all-fail-ledger.jsonl"

    with pytest.raises(RuntimeError, match="retained"):
        run_shenzhen_evaluation(
            _settings(tmp_path),
            archive_path=archive_path,
            output_root=tmp_path / "all-fail-runs",
            config_path=config_path,
            _vision_backend=PixelBackend(fail_all=True),
            _ledger_path=ledger_path,
        )

    result_path = next((tmp_path / "all-fail-runs").glob("*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed_retained"
    assert result["failure"]["error_type"] == "ExternalEvaluationContractError"
    assert result["metrics"]["classifier_image_level"]["evaluable_count"] == 0
    ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert ledger[0]["status"] == "failed_retained"


def test_native_argmax_tie_is_metadata_and_stable_route_not_review(
    tmp_path: Path,
) -> None:
    cases = [
        ("CHNCXR_0001_0.png", 10),
        ("CHNCXR_0002_1.png", 180),
    ]
    archive_path = tmp_path / "tie.zip"
    metadata = _build_archive(archive_path, cases)
    config_path = _write_config(
        tmp_path / "tie-config.json",
        archive_path=archive_path,
        metadata=metadata,
        cases=cases,
    )

    result_path = run_shenzhen_evaluation(
        _settings(tmp_path),
        archive_path=archive_path,
        output_root=tmp_path / "tie-runs",
        config_path=config_path,
        _vision_backend=PixelBackend(tie_all=True),
        _ledger_path=tmp_path / "tie-ledger.jsonl",
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in Path(result["records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    tie_rate = result["metrics"]["workflow_rates"]["classifier_argmax_tie_abstention_rate"]
    assert result["status"] == "completed_observational"
    assert all(record["technical_failure"]["occurred"] is False for record in records)
    assert all(record["classifier"]["predicted_class"] is None for record in records)
    assert all(record["classifier"]["argmax_tied"] is True for record in records)
    assert all(record["fusion"]["visual_result"] == "model_not_flagged" for record in records)
    assert result["metrics"]["classifier_image_level"]["evaluable_count"] == 0
    assert tie_rate["count"] == 2
    assert tie_rate["value"] == pytest.approx(1.0)
    assert result["metrics"]["workflow_rates"]["human_review_rate"]["value"] == 0.0
    assert (
        result["metrics"]["workflow_rates"]["definitive_output_coverage_rate"]["value"]
        == 1.0
    )


def test_prior_outcomes_force_explicit_post_hoc_status(tmp_path: Path) -> None:
    cases = [("CHNCXR_0001_0.png", 10), ("CHNCXR_0002_1.png", 180)]
    archive_path = tmp_path / "post-hoc.zip"
    metadata = _build_archive(archive_path, cases)
    config_path = _write_config(
        tmp_path / "post-hoc-config.json",
        archive_path=archive_path,
        metadata=metadata,
        cases=cases,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        analysis_status="post_hoc_descriptive_reanalysis_only",
        prior_external_outcomes_observed=True,
        independent_external_validation=False,
    )
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    result_path = run_shenzhen_evaluation(
        _settings(tmp_path),
        archive_path=archive_path,
        output_root=tmp_path / "post-hoc-runs",
        config_path=config_path,
        _vision_backend=PixelBackend(),
        _ledger_path=tmp_path / "post-hoc-ledger.jsonl",
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "completed_post_hoc_observational"
    assert result["full_configuration"]["independent_external_validation"] is False


def test_legacy_evaluation_identity_is_bound_to_the_immutable_legacy_policy() -> None:
    assert _fusion_policy_binding({"evaluation_id": LEGACY_SHENZHEN_EVALUATION_ID}) == (
        LEGACY_SHENZHEN_FUSION_POLICY_FILENAME,
        LEGACY_SHENZHEN_FUSION_POLICY_ID,
    )


def test_policy_identity_mismatch_fails_before_inference_and_is_retained(
    tmp_path: Path,
) -> None:
    cases = [("CHNCXR_0001_0.png", 10)]
    archive_path = tmp_path / "policy-mismatch.zip"
    metadata = _build_archive(archive_path, cases)
    config_path = _write_config(
        tmp_path / "policy-mismatch-config.json",
        archive_path=archive_path,
        metadata=metadata,
        cases=cases,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["expected_fusion_policy_id"] = "unexpected-policy-id"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    backend = PixelBackend()
    ledger_path = tmp_path / "policy-mismatch-ledger.jsonl"

    with pytest.raises(RuntimeError, match="retained"):
        run_shenzhen_evaluation(
            _settings(tmp_path),
            archive_path=archive_path,
            output_root=tmp_path / "policy-mismatch-runs",
            config_path=config_path,
            _vision_backend=backend,
            _ledger_path=ledger_path,
        )

    result_path = next((tmp_path / "policy-mismatch-runs").glob("*/result.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert backend.call_count == 0
    assert result["status"] == "failed_retained"
    assert result["failure"]["error_type"] == "ExternalEvaluationContractError"
    assert "fusion policy identity mismatch" in result["failure"]["error"]
    assert result["record_count"] == 0


def test_bounded_zip_reader_rejects_oversize_without_extracting(tmp_path: Path) -> None:
    archive_path = tmp_path / "bounded.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("images/images/CHNCXR_0001_0.png", b"x" * 32)
    with zipfile.ZipFile(archive_path, "r") as archive:
        info = archive.getinfo("images/images/CHNCXR_0001_0.png")
        with pytest.raises(ExternalEvaluationContractError, match="bounded read limit"):
            _read_zip_entry_bounded(archive, info, max_bytes=16)
    assert not (tmp_path / "images").exists()
