from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from tbx_agent import preflight


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_preflight_tree(
    tmp_path: Path,
    *,
    backend: str,
) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "project"
    config_dir = root / "configs"
    knowledge_dir = root / "knowledge"
    config_dir.mkdir(parents=True)
    knowledge_dir.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'synthetic-preflight-fixture'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    package_dir = root / "src" / "tbx_agent"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")

    (config_dir / "app.yaml").write_text(
        f"runtime:\n  vision_backend: {backend}\n",
        encoding="utf-8",
    )
    _write_json(
        config_dir / "fusion_policy.json",
        {
            "schema_version": 5,
            "policy_id": "rank03-user-trained-native-argmax-v2",
            "purpose": "research_screening_support_only",
            "classifier_policy_id": "three_class_native_argmax_v1",
            "classifier_rule": "native_three_class_argmax",
            "classifier_probability_order": ["healthy", "sick_non_tb", "tb"],
            "tie_handling": "stable_native_argmax",
            "classifier_routes": {
                "healthy": "model_not_flagged",
                "sick_non_tb": "non_tb_abnormal",
                "tb": "model_flagged",
            },
            "detector_role": "advisory_localization_only",
            "quality_warning": "retain_argmax_with_advisory",
            "technical_failure": "technical_failure",
            "selection_dataset": None,
            "selection_metrics": None,
            "heldout_metrics": None,
            "hidden_or_locked_test_used": False,
            "clinical_validation": False,
            "performance_claims_inherited": False,
            "warning": (
                "The repository contains inference code only, with no model weights or clinical "
                "performance claim. Install the independently distributed, hash-verified inference "
                "bundle and validate its runtime identity before use."
            ),
        },
    )
    _write_json(
        config_dir / "fusion_policy_argmax_v2.json",
        preflight._ARGMAX_POLICY_SNAPSHOT,
    )
    _write_json(
        config_dir / "fusion_policy_sens98_legacy.json",
        preflight._LEGACY_POLICY_SNAPSHOT,
    )
    _write_json(config_dir / "safety_policy.json", {"schema_version": 1})

    assets = root / "assets"
    assets.mkdir()
    asset_paths = {
        "classifier_checkpoint": assets / "classifier.pt",
        "classifier_config": assets / "classifier-config.json",
        "detector_checkpoint": assets / "detector.pt",
        "detector_config": assets / "detector-resolved.json",
    }
    for name, path in asset_paths.items():
        path.write_bytes(f"small-test-asset:{name}".encode())

    dfine_root = root / "third_party" / "D-FINE"
    upstream_base = dfine_root / "configs" / "dfine" / "dfine_hgnetv2_l_coco.yml"
    upstream_base.parent.mkdir(parents=True)
    upstream_base.write_text("model: test\n", encoding="utf-8")
    declared_base = root / "reports" / "dfine_l_coco_e80.yml"
    declared_base.parent.mkdir()
    declared_base.write_text("includes: test\n", encoding="utf-8")
    asset_paths["dfine_root"] = dfine_root
    asset_paths["upstream_base"] = upstream_base
    asset_paths["declared_base"] = declared_base

    _write_json(
        config_dir / "rank03_runtime.json",
        {
            "schema_version": 1,
            "classifier": {
                "probability_order": ["healthy", "sick_non_tb", "tb"],
                "checkpoint_path": str(asset_paths["classifier_checkpoint"].relative_to(root)),
                "checkpoint_sha256": _sha256(asset_paths["classifier_checkpoint"]),
                "config_path": str(asset_paths["classifier_config"].relative_to(root)),
                "config_sha256": _sha256(asset_paths["classifier_config"]),
            },
            "detector": {
                "checkpoint_path": str(asset_paths["detector_checkpoint"].relative_to(root)),
                "checkpoint_sha256": _sha256(asset_paths["detector_checkpoint"]),
                "resolved_config_path": str(asset_paths["detector_config"].relative_to(root)),
                "resolved_config_sha256": _sha256(asset_paths["detector_config"]),
                "source_root": str(dfine_root.relative_to(root)),
                "base_config_path": str(declared_base.relative_to(root)),
            },
            "validation_scope": {"hidden_or_locked_test_used": False},
        },
    )

    _write_json(
        knowledge_dir / "source_manifest.json",
        {
            "schema_version": 1,
            "snapshot_id": "test-snapshot",
            "sources": [
                {
                    "source_id": "allowed",
                    "status": "included",
                    "retrievable": True,
                    "full_text_reviewed": True,
                    "allowed_claim_scope": ["test"],
                    "treatment_details_allowed": False,
                },
                {
                    "source_id": "limited",
                    "status": "included_limited",
                    "retrievable": True,
                    "full_text_reviewed": True,
                    "allowed_claim_scope": ["test"],
                    "treatment_details_allowed": False,
                },
                {
                    "source_id": "excluded",
                    "status": "excluded",
                    "retrievable": False,
                    "full_text_reviewed": False,
                    "allowed_claim_scope": [],
                    "treatment_details_allowed": False,
                },
            ],
        },
    )
    (knowledge_dir / "chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "source_id": "allowed",
                "text": "isolated guideline content",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        knowledge_dir / "active_screening_questions.json",
        {"schema_version": 1, "questions": []},
    )
    chunks_digest = _sha256(knowledge_dir / "chunks.jsonl")
    (config_dir / "knowledge_ingestion.yaml").write_text(
        "\n".join(
            [
                "schema_version: 1",
                "pipeline_version: test-ingestion-v1",
                "offline_only: true",
                f"output_dir: {(root / 'runtime' / 'ingestion').as_posix()}",
                "extraction:",
                "  fail_on_suspected_scanned_page: true",
                "chunking:",
                "  min_characters: 10",
                "  target_characters: 20",
                "  max_characters: 40",
                "  overlap_blocks: 0",
                "sources:",
                "  - source_id: allowed_test_source",
                "    input_path: ../knowledge/chunks.jsonl",
                "    format: markdown",
                "    title: Synthetic source",
                "    organization: Test",
                "    publication_year: 2026",
                "    jurisdiction: Local",
                "    url: urn:test:source",
                f"    expected_sha256: {chunks_digest}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    source_retrieval_config = Path(__file__).resolve().parents[1] / "configs" / "retrieval.yaml"
    (config_dir / "retrieval.yaml").write_bytes(source_retrieval_config.read_bytes())

    source_eval_root = Path(__file__).resolve().parents[1] / "evaluation"
    target_eval_root = root / "evaluation"
    for config_name in (
        "system_bench_config.json",
        "system_bench_config_v1_1.json",
        "system_bench_config_v1_2.json",
        "system_bench_config_v1_3.json",
        "system_bench_config_v1_4.json",
        "system_bench_config_v1_5.json",
        "system_bench_config_v1_6.json",
    ):
        (target_eval_root / config_name).parent.mkdir(parents=True, exist_ok=True)
        (target_eval_root / config_name).write_bytes(
            (source_eval_root / config_name).read_bytes()
        )
    for suite_name in (
        "system_v1",
        "system_v1_2",
        "system_v1_3",
        "system_v1_4",
        "system_v1_5",
        "system_v1_6",
    ):
        source_suite_root = source_eval_root / "suites" / suite_name
        target_suite_root = target_eval_root / "suites" / suite_name
        target_suite_root.mkdir(parents=True)
        for filename in ("manifest.json", "cases.jsonl"):
            (target_suite_root / filename).write_bytes(
                (source_suite_root / filename).read_bytes()
            )
    return root, asset_paths


def _checks(result: dict) -> dict[str, dict]:
    return {item["id"]: item for item in result["checks"]}


def test_mock_preflight_is_explicitly_non_real_and_never_hashes_model_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")

    def unexpected_hash(_path: Path) -> str:
        raise AssertionError("mock preflight must not touch model artifacts")

    monkeypatch.setattr(preflight, "_sha256", unexpected_hash)
    result = preflight.run_preflight(project_root=root, environ={})
    checks = _checks(result)

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["mode"] == {
        "vision_backend": "mock",
        "real_inference": False,
        "inference_mode": "synthetic_mock_non_real_inference",
    }
    assert result["guardrails"]["weights_loaded"] is False
    assert result["guardrails"]["dataset_or_test_manifests_read"] is False
    assert result["runtime"]["wall_clock_ms"] >= 0
    assert result["runtime"]["peak_vram_mib"] is None
    assert result["runtime"]["peak_vram_measurement_method"] == "unknown_not_measured"
    assert checks["artifact_inventory.complete"]["status"] == "pass"
    assert checks["vision.backend"]["details"] == {"real_inference": False}
    assert checks["knowledge.chunks.allowed_sources"]["status"] == "pass"
    assert checks["fusion.hidden_or_locked_test_unused"]["status"] == "pass"
    assert checks["storage.artifact_roots.disjoint"]["status"] == "pass"
    assert not any(check_id.startswith("rank03.classifier") for check_id in checks)


def test_preflight_fails_closed_when_case_storage_is_inside_model_cache(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    model_root = tmp_path / "shared-artifacts"

    result = preflight.run_preflight(
        project_root=root,
        environ={
            "TBX_ARTIFACT_ROOT": str(model_root),
            "TBX_AGENT_CASE_ARTIFACT_ROOT": str(model_root / "cases"),
            "TBX_AGENT_PREFLIGHT_MODE": "runtime",
        },
    )
    check = _checks(result)["storage.artifact_roots.disjoint"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["case_artifact_root"] == str((model_root / "cases").resolve())
    assert check["details"]["model_artifact_root"] == str(model_root.resolve())


def test_preflight_inventory_fails_closed_when_packaging_manifest_is_missing(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    (root / "pyproject.toml").unlink()

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["artifact_inventory.complete"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["missing_paths"] == [str((root / "pyproject.toml").resolve())]


def test_preflight_ignores_unlisted_dataset_or_test_manifests(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    decoy = root / "knowledge" / "locked_local_test_manifest.json"
    decoy.write_text("this is intentionally not JSON and must not be read", encoding="utf-8")

    result = preflight.run_preflight(project_root=root, environ={})

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["guardrails"]["dataset_or_test_manifests_read"] is False
    assert not any(str(decoy) == item.get("path") for item in result["checks"])


def test_rank03_preflight_hashes_exactly_four_assets_and_checks_dfine_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, assets = _build_preflight_tree(tmp_path, backend="rank03")
    hashed: list[Path] = []
    real_sha256 = preflight._sha256

    def recording_hash(path: Path) -> str:
        hashed.append(path)
        return real_sha256(path)

    monkeypatch.setattr(preflight, "_sha256", recording_hash)
    result = preflight.run_preflight(project_root=root, environ={})
    checks = _checks(result)

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["mode"]["real_inference"] is True
    assert set(hashed) == {
        assets["classifier_checkpoint"],
        assets["classifier_config"],
        assets["detector_checkpoint"],
        assets["detector_config"],
    }
    assert len(hashed) == 4
    for check_id in (
        "rank03.classifier.checkpoint_sha256",
        "rank03.classifier.config_sha256",
        "rank03.detector.checkpoint_sha256",
        "rank03.detector.resolved_config_sha256",
        "rank03.dfine.source_root",
        "rank03.dfine.upstream_base",
        "rank03.dfine.declared_base",
    ):
        assert checks[check_id]["status"] == "pass"


def test_rank03_sha_mismatch_is_machine_readable_failure(tmp_path: Path) -> None:
    root, assets = _build_preflight_tree(tmp_path, backend="rank03")
    assets["detector_checkpoint"].write_bytes(b"tampered-after-contract-was-frozen")

    result = preflight.run_preflight(project_root=root, environ={})
    failed = _checks(result)["rank03.detector.checkpoint_sha256"]

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert failed["status"] == "fail"
    assert failed["message"] == "frozen rank03 asset SHA256 does not match"
    assert failed["details"]["expected_sha256"] != failed["details"]["actual_sha256"]


def test_preflight_uses_external_rank03_runtime_contract(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="rank03")
    checked_in = root / "configs" / "rank03_runtime.json"
    external = tmp_path / "runtime" / "rank03_runtime.json"
    external.parent.mkdir()
    external.write_bytes(checked_in.read_bytes())
    checked_in.unlink()

    result = preflight.run_preflight(
        project_root=root,
        environ={"TBX_AGENT_RANK03_RUNTIME_CONFIG": str(external)},
    )
    checks = _checks(result)

    assert checks["config.rank03_runtime.exists"]["status"] == "pass"
    assert checks["config.rank03_runtime.exists"]["path"] == str(external.resolve())
    assert checks["config.rank03_runtime.parse"]["status"] == "pass"


@pytest.mark.parametrize(
    ("target", "field", "value", "expected_detail"),
    [
        ("fusion", "classifier_rule", "p_tb_gte_threshold", "classifier_rule"),
        (
            "fusion",
            "classifier_probability_order",
            ["tb", "sick_non_tb", "healthy"],
            "classifier_probability_order",
        ),
        ("fusion", "classifier_threshold", 0.027215289, "classifier_threshold"),
        ("fusion", "detector_role", "legacy_vote", "detector_role"),
        ("fusion", "tie_handling", "abstain_to_human_review", "tie_handling"),
        (
            "fusion",
            "quality_warning",
            "pending_human_review",
            "quality_warning",
        ),
        (
            "runtime",
            "probability_order",
            ["tb", "sick_non_tb", "healthy"],
            "rank03_runtime.classifier.probability_order",
        ),
    ],
)
def test_preflight_fails_closed_on_active_argmax_policy_drift(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    expected_detail: str,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="rank03")
    if target == "fusion":
        path = root / "configs" / "fusion_policy.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = value
    else:
        path = root / "configs" / "rank03_runtime.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["classifier"][field] = value
    _write_json(path, payload)

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["fusion.active_policy_contract"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    details = check["details"]
    assert (
        expected_detail in details["mismatches"]
        or expected_detail in details["forbidden_fields_present"]
    )


def test_preflight_records_no_inherited_performance_and_retained_policies(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")

    result = preflight.run_preflight(project_root=root, environ={})
    checks = _checks(result)

    assert result["ok"] is True
    assert checks["fusion.active_policy_contract"]["details"] == {
        "policy_id": "rank03-user-trained-native-argmax-v2",
        "classifier_policy_id": "three_class_native_argmax_v1",
        "classifier_rule": "native_three_class_argmax",
        "performance_claims_inherited": False,
        "clinical_validation": False,
        "probability_order": ["healthy", "sick_non_tb", "tb"],
    }
    assert checks["fusion.retained_policy_snapshots"]["status"] == "pass"


def test_preflight_accepts_checked_in_active_and_retained_policy_contracts(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    source_configs = Path(__file__).resolve().parents[1] / "configs"
    for filename in (
        "fusion_policy.json",
        "fusion_policy_argmax_v2.json",
        "fusion_policy_sens98_legacy.json",
    ):
        (root / "configs" / filename).write_bytes((source_configs / filename).read_bytes())

    checks = _checks(preflight.run_preflight(project_root=root, environ={}))

    # Read the shipped files independently of preflight's expected snapshots,
    # so tests cannot pass merely because two synthetic fixtures share drift.
    assert checks["fusion.active_policy_contract"]["status"] == "pass"
    assert checks["fusion.retained_policy_snapshots"]["status"] == "pass"


def test_preflight_rejects_injected_performance_claim(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    path = root / "configs" / "fusion_policy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["heldout_metrics"] = {"sensitivity": 0.9}
    _write_json(path, payload)

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["fusion.active_policy_contract"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["mismatches"]["heldout_metrics"]["actual"] == {
        "sensitivity": 0.9
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("warning", "silently weakened"),
        ("tie_handling", "abstain_to_human_review"),
        ("quality_warning", "pending_human_review"),
    ],
)
def test_preflight_rejects_retained_policy_snapshot_drift(
    tmp_path: Path, field: str, value: str
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    path = root / "configs" / "fusion_policy_argmax_v2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(path, payload)

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["fusion.retained_policy_snapshots"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    assert check["details"]["mismatches"] == {"argmax": {"changed_fields": [field]}}


def test_chunks_may_only_reference_retrievable_manifest_sources(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    chunks_path = root / "knowledge" / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps({"chunk_id": "forbidden", "source_id": "excluded"}) + "\n",
        encoding="utf-8",
    )

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["knowledge.chunks.allowed_sources"]

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert check["status"] == "fail"
    assert check["details"]["invalid_references"] == [
        {
            "line": 1,
            "chunk_id": "forbidden",
            "source_id": "excluded",
            "manifest_status": "excluded",
            "retrievable": False,
        }
    ]


def test_declared_local_guideline_asset_is_confined_and_sha_pinned(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    knowledge_dir = root / "knowledge"
    source_path = knowledge_dir / "sources" / "guideline.pdf"
    source_path.parent.mkdir()
    source_path.write_bytes(b"small-reviewed-guideline")
    manifest_path = knowledge_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0].update(
        {
            "local_path": "sources/guideline.pdf",
            "file_sha256": _sha256(source_path),
        }
    )
    _write_json(manifest_path, manifest)

    valid = preflight.run_preflight(project_root=root, environ={})
    valid_check = _checks(valid)["knowledge.sources.local_assets"]
    assert valid["ok"] is True
    assert valid_check["status"] == "pass"
    assert valid_check["details"]["declared_count"] == 1

    source_path.write_bytes(b"tampered-guideline")
    invalid = preflight.run_preflight(project_root=root, environ={})
    invalid_check = _checks(invalid)["knowledge.sources.local_assets"]
    assert invalid["ok"] is False
    assert invalid_check["status"] == "fail"
    assert invalid_check["details"]["failures"][0]["reason"] == "sha256_mismatch"


def test_active_screening_references_must_resolve_to_manifest_and_chunks(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    questions_path = root / "knowledge" / "active_screening_questions.json"
    _write_json(
        questions_path,
        {
            "schema_version": 1,
            "sources": {"unregistered": {}},
            "questions": [
                {
                    "question_id": "bad-source",
                    "source_id": "unregistered",
                }
            ],
            "summary_citations": {
                "bad-chunk": {
                    "chunk_id": "missing",
                    "source_id": "allowed",
                }
            },
        },
    )

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["knowledge.active_screening.references"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    reasons = {item["reason"] for item in check["details"]["errors"]}
    assert reasons == {
        "question_source_not_registered",
        "screening_source_not_in_manifest",
        "summary_citation_chunk_missing",
    }


def test_invalid_knowledge_jsonl_and_missing_config_are_reported_without_crashing(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    (root / "knowledge" / "chunks.jsonl").write_text("{not-json}\n", encoding="utf-8")
    (root / "configs" / "safety_policy.json").unlink()

    result = preflight.run_preflight(project_root=root, environ={})
    checks = _checks(result)

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert checks["config.safety_policy.exists"]["status"] == "fail"
    assert checks["config.safety_policy.parse"]["status"] == "skip"
    assert checks["knowledge.chunks.jsonl.parse"]["status"] == "fail"
    assert "line 1" in checks["knowledge.chunks.jsonl.parse"]["message"]
    assert checks["knowledge.chunks.allowed_sources"]["status"] == "skip"


def test_hidden_or_locked_test_use_must_be_explicitly_false(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    runtime_path = root / "configs" / "rank03_runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["validation_scope"]["hidden_or_locked_test_used"] = True
    _write_json(runtime_path, runtime)

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["fusion.hidden_or_locked_test_unused"]

    assert result["ok"] is False
    assert result["exit_code"] == 2
    assert check["status"] == "fail"
    assert check["details"]["declarations"] == {
        "fusion_policy.hidden_or_locked_test_used": False,
        "rank03_runtime.validation_scope.hidden_or_locked_test_used": True,
    }


def test_cli_prints_one_json_document_and_returns_only_zero_or_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    monkeypatch.setattr(preflight, "PROJECT_ROOT", root)
    monkeypatch.delenv("TBX_AGENT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_KNOWLEDGE_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_VISION_BACKEND", raising=False)

    exit_code = preflight.main([])
    output = capsys.readouterr()
    payload = json.loads(output.out)

    assert exit_code == 0
    assert exit_code in {0, 2}
    assert output.err == ""
    assert payload["exit_code"] == exit_code
    assert payload["ok"] is True


def test_cli_preflight_receipt_is_exclusive_and_schema_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    monkeypatch.setattr(preflight, "PROJECT_ROOT", root)
    monkeypatch.delenv("TBX_AGENT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_KNOWLEDGE_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_VISION_BACKEND", raising=False)
    output_path = tmp_path / "receipts" / "preflight.json"

    assert preflight.main(["--output", str(output_path)]) == 0
    capsys.readouterr()
    receipt = json.loads(output_path.read_text(encoding="utf-8"))
    assert receipt["run_id"].startswith("preflight-")
    assert receipt["artifact_inventory_sha256"]
    assert receipt["runtime"]["peak_vram_measurement_method"] == "unknown_not_measured"
    assert receipt["deployment"]["release_authorized"] is False

    with pytest.raises(FileExistsError):
        preflight.main(["--output", str(output_path)])


def test_console_entry_main_reads_process_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    output_path = tmp_path / "console-receipt.json"
    monkeypatch.setattr(preflight, "PROJECT_ROOT", root)
    monkeypatch.setattr(sys, "argv", ["tbx-agent-preflight", "--output", str(output_path)])
    monkeypatch.delenv("TBX_AGENT_CONFIG_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_KNOWLEDGE_DIR", raising=False)
    monkeypatch.delenv("TBX_AGENT_VISION_BACKEND", raising=False)

    assert preflight.main() == 0
    capsys.readouterr()
    assert json.loads(output_path.read_text(encoding="utf-8"))["ok"] is True


def test_console_entry_help_reads_process_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["tbx-agent-preflight", "--help"])

    with pytest.raises(SystemExit) as raised:
        preflight.main()

    output = capsys.readouterr()
    assert raised.value.code == 0
    assert "--output" in output.out
    assert output.err == ""


def test_llamacpp_contract_path_comparison_is_windows_case_portable_and_fail_closed() -> None:
    runtime_contract = {
        "model_alias": "tbx-qwen",
        "model_path": r"Z:\synthetic-fixture\models\qwen.gguf",
        "model_sha256": "a" * 64,
        "server_build": "b10517",
    }
    app_contract = {
        **runtime_contract,
        "model_path": r"z:\SYNTHETIC-FIXTURE\MODELS\QWEN.GGUF",
    }

    assert preflight._llamacpp_contract_mismatches(app_contract, runtime_contract) == {}

    app_contract["model_path"] = r"Z:\synthetic-fixture\models\other.gguf"
    path_drift = preflight._llamacpp_contract_mismatches(app_contract, runtime_contract)
    assert set(path_drift) == {"model_path"}

    app_contract = {**runtime_contract, "model_alias": "TBX-QWEN"}
    alias_drift = preflight._llamacpp_contract_mismatches(app_contract, runtime_contract)
    assert set(alias_drift) == {"model_alias"}


def test_extended_ingestion_retrieval_and_suite_integrity_fail_closed(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    (root / "knowledge" / "chunks.jsonl").write_text(
        '{"chunk_id":"chunk-1","source_id":"allowed","text":"tampered"}\n',
        encoding="utf-8",
    )
    retrieval_path = root / "configs" / "retrieval.yaml"
    retrieval = retrieval_path.read_text(encoding="utf-8").replace(
        "enabled: false", "enabled: true", 1
    )
    retrieval_path.write_text(retrieval, encoding="utf-8")
    cases_path = root / "evaluation" / "suites" / "system_v1_6" / "cases.jsonl"
    cases_path.write_text(cases_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = preflight.run_preflight(project_root=root, environ={})
    checks = _checks(result)

    assert result["ok"] is False
    assert checks["config.knowledge_ingestion.contract"]["status"] == "fail"
    assert checks["config.retrieval.contract"]["status"] == "fail"
    assert checks["evaluation.system_suite.integrity"]["status"] == "fail"


def test_preflight_rejects_a_stale_deterministic_candidate_snapshot(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    candidate_path = (
        root / "evaluation" / "suites" / "system_v1_6" / "deterministic_mock_candidate.json"
    )
    _write_json(
        candidate_path,
        {
            "source_tree_sha256": "0" * 64,
            "artifacts": {},
        },
    )

    result = preflight.run_preflight(project_root=root, environ={})
    check = _checks(result)["evaluation.deterministic_candidate.integrity"]

    assert result["ok"] is False
    assert check["status"] == "fail"
    assert "artifact set mismatch" in check["message"]


def test_production_profile_never_inherits_mock_or_unauthenticated_defaults(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")

    result = preflight.run_preflight(
        project_root=root,
        environ={"TBX_AGENT_DEPLOYMENT_PROFILE": "production"},
    )
    check = _checks(result)["deployment.profile.contract"]

    assert result["ok"] is False
    assert result["deployment"]["technical_profile_contract_passed"] is False
    assert result["deployment"]["release_authorized"] is False
    assert check["status"] == "fail"
    assert set(check["details"]["failures"]) == {
        "vision_backend_must_be_rank03",
        "narrator_backend_must_be_llama_cpp",
        "require_real_inference_must_be_enabled",
        "require_llm_inference_must_be_enabled",
        "trusted_proxy_auth_must_be_enabled",
        "trusted_proxy_hmac_secret_missing_or_weak",
        "uploaded_image_retention_requires_governed_storage",
        "production_privacy_defaults_missing_or_weakened",
    }


def test_production_preflight_verifies_complete_llamacpp_bundle_and_secret_boundary(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="rank03")
    runtime_root = root / "runtime" / "llama"
    runtime_root.mkdir(parents=True)
    binary = runtime_root / "llama-server.exe"
    model = runtime_root / "qwen.gguf"
    bundle = runtime_root / "bundle.json"
    key_file = runtime_root / "api.keys"
    binary.write_bytes(b"synthetic-llama-server")
    model.write_bytes(b"synthetic-q4-k-m")
    key_file.write_text("k" * 48 + "\n", encoding="utf-8")
    _write_json(
        bundle,
        {
            "files": [
                {
                    "relative_path": binary.name,
                    "size_bytes": binary.stat().st_size,
                    "sha256": _sha256(binary),
                }
            ]
        },
    )
    _write_json(
        root / "configs" / "llm_runtime.yaml",
        {
            "schema_version": 1,
            "runtime_id": "synthetic-qwen-llamacpp",
            "engine": "llama.cpp",
            "server_build": "b10517",
            "release_archive_sha256": "a" * 64,
            "binary_path": str(binary),
            "binary_sha256": _sha256(binary),
            "bundle_manifest_path": str(bundle),
            "bundle_manifest_sha256": _sha256(bundle),
            "api_key_file": str(key_file),
            "model_id": "Qwen/Qwen3.5-4B",
            "model_alias": "tbx-qwen",
            "model_path": str(model),
            "model_sha256": _sha256(model),
            "quantization": "Q4_K_M",
            "allowed_roles": ["narrator", "evidence_composer"],
            "clinical_authority": False,
            "load_mmproj": False,
        },
    )
    _write_json(
        root / "evaluation" / "llamacpp_eval_config.json",
        {
            "schema_version": 1,
            "evaluation_id": "synthetic-llamacpp-evaluation-v1",
            "hypothesis": ("The synthetic narrator preserves every deterministic safety contract."),
            "seed": 20260829,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "major_variables_changed": ["narrator_backend"],
            "narrator": {
                "backend": "llama_cpp",
                "runtime_config": "configs/llm_runtime.yaml",
                "runtime_config_sha256": _sha256(root / "configs" / "llm_runtime.yaml"),
            },
            "metrics": sorted(preflight._LLAMACPP_EVAL_REQUIRED_METRICS),
        },
    )
    for relative in preflight._RUNTIME_RELEASE_EVIDENCE_ARTIFACTS:
        evidence_path = root / relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            f"synthetic non-clinical runtime evidence: {relative}\n",
            encoding="utf-8",
        )
    from tbx_agent.llm.runtime_supervisor import load_runtime_config

    runtime_config_path = root / "configs" / "llm_runtime.yaml"
    runtime_config = load_runtime_config(runtime_config_path)
    runtime_raw_sha256 = _sha256(runtime_config_path)
    runtime_canonical_sha256 = runtime_config.canonical_sha256()
    model_sha256 = _sha256(model)
    data_root = root / "runtime" / "evidence"
    data_root.mkdir(parents=True)
    benchmark_result = data_root / "benchmark-result.json"
    recovery_result = data_root / "recovery-result.json"
    coexistence_result = data_root / "coexistence-result.json"
    pair_harness_result = data_root / "pair-harness-failure.json"
    _write_json(benchmark_result, {"status": "synthetic-benchmark-pass"})
    _write_json(recovery_result, {"status": "synthetic-recovery-pass"})
    _write_json(coexistence_result, {"status": "synthetic-coexistence-pass"})
    _write_json(pair_harness_result, {"status": "synthetic-pair-harness-failure"})
    source_revision = "a" * 40
    common_runtime_configuration = {
        "runtime_id": runtime_config.runtime_id,
        "runtime_config_file_sha256": runtime_raw_sha256,
        "runtime_config_canonical_sha256": runtime_canonical_sha256,
        "model_sha256": model_sha256,
        "selection_use": False,
        "locked_or_hidden_test_used": False,
        "clinical_validation": False,
        "major_variables_changed": [],
    }
    runtime_records = [
        {
            "schema_version": 1,
            "kind": "llamacpp_runtime_benchmark_preflight",
            "run_id": "synthetic-failed-preflight",
            "status": "failed_preflight_retained",
            "full_configuration": common_runtime_configuration,
            "seed": 20260829,
            "source_revision": source_revision,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "metrics": {
                "runtime_seconds": 1.0,
                "peak_vram_bytes": None,
                "peak_vram_measurement": "not captured because preflight failed",
            },
            "failure": {"type": "synthetic_retained_failure"},
        },
        {
            "schema_version": 1,
            "kind": "llamacpp_runtime_benchmark",
            "run_id": "synthetic-runtime-pass",
            "status": "passed",
            "full_configuration": common_runtime_configuration,
            "seed": 20260829,
            "source_revision": source_revision,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "metrics": {
                "short_wall_ms_p95": 10.0,
                "peak_vram_bytes": None,
                "peak_vram_measurement": "unavailable under synthetic WDDM",
            },
            "result_path": str(benchmark_result),
            "result_sha256": _sha256(benchmark_result),
        },
        {
            "schema_version": 1,
            "kind": "llamacpp_recovery_benchmark",
            "run_id": "synthetic-recovery-pass",
            "status": "passed",
            "full_configuration": common_runtime_configuration,
            "seed": 20260829,
            "source_revision": source_revision,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "metrics": {
                "cold_start_to_health_ms": 20.0,
                "peak_vram_bytes": None,
                "peak_vram_measurement": "unavailable under synthetic WDDM",
            },
            "result_path": str(recovery_result),
            "result_sha256": _sha256(recovery_result),
        },
    ]
    runtime_ledger_path = root / "evaluation" / "runtime_operations_ledger.jsonl"
    runtime_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in runtime_records) + "\n",
        encoding="utf-8",
    )
    coexistence_record = {
        "schema_version": 1,
        "kind": "rank03_llamacpp_co_resident_operational_smoke",
        "run_id": "synthetic-coexistence-pass",
        "status": "passed_operational_smoke",
        "full_configuration": {
            "llama_runtime_id": runtime_config.runtime_id,
            "llama_runtime_file_sha256": runtime_raw_sha256,
            "llama_model_sha256": model_sha256,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
        },
        "seed": 20260829,
        "source_revision": source_revision,
        "selection_use": False,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "clinical_validation": False,
        "metrics": {
            "oom_event_count": 0,
            "llama_cpp_healthy_after_rank03": True,
            "rank03_inference_success_count": 1,
        },
        "peak_vram_bytes": 1024,
        "result_path": str(coexistence_result),
        "result_sha256": _sha256(coexistence_result),
    }
    pair_harness_record = {
        "schema_version": 1,
        "kind": "qwen35_bf16_q4_constrained_narrator_pair_harness_failure",
        "run_id": "synthetic-pair-harness-failure",
        "status": "failed_harness_pipe_inheritance_retained",
        "full_configuration": {
            "runtime_config_file_sha256": runtime_raw_sha256,
            "runtime_config_canonical_sha256": runtime_canonical_sha256,
            "q4_model_sha256": model_sha256,
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "major_variables_changed": ["gguf_weight_precision"],
        },
        "production_q4_after_harness_termination": {
            "runtime_id": runtime_config.runtime_id,
        },
        "seed": 20260829,
        "source_revision": source_revision,
        "selection_use": False,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "clinical_validation": False,
        "quantization_selection_use": False,
        "metrics": {
            "pair_arm_launch_count": 0,
            "peak_vram_bytes": None,
            "peak_vram_measurement": "unavailable under synthetic WDDM",
        },
        "failure": {"type": "synthetic_pair_harness_failure"},
        "result_path": str(pair_harness_result),
        "result_sha256": _sha256(pair_harness_result),
    }
    evaluation_ledger_path = root / "evaluation" / "ledger.jsonl"
    evaluation_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in (coexistence_record, pair_harness_record))
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root / "configs" / "app.yaml",
        {
            "deployment": {"profile": "production"},
            "runtime": {
                "vision_backend": "rank03",
                "narrator_backend": "llama_cpp",
                "require_real_inference": True,
                "require_llm_inference": True,
                "llama_cpp_model_alias": "tbx-qwen",
                "llama_cpp_model_path": str(model),
                "llama_cpp_model_sha256": _sha256(model),
                "llama_cpp_server_build": "b10517",
                "max_upload_bytes": 1024,
                "retain_uploaded_image": False,
            },
            "privacy": {
                "allow_real_name": False,
                "allow_raw_phi_in_external_tracing": False,
                "allow_case_content_in_user_memory": False,
            },
            "security": {
                "trusted_proxy_auth_enabled": True,
                "trusted_proxy_replay_window_seconds": 60,
                "max_request_body_bytes": 2048,
                "max_concurrent_requests": 1,
                "rate_limit_requests_per_minute": 1,
                "rate_limit_burst": 1,
            },
            "observability": {
                "metrics_enabled": False,
                "metrics_allow_loopback": True,
            },
        },
    )
    production_environ = {
        "TBX_AGENT_TRUSTED_PROXY_HMAC_SECRET": "s" * 48,
        "TBX_AGENT_DATA_ROOT": str(data_root),
    }

    result = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    checks = _checks(result)

    assert result["ok"] is True
    assert result["deployment"]["technical_profile_contract_passed"] is True
    assert result["deployment"]["release_authorized"] is False
    assert checks["deployment.profile.contract"]["status"] == "pass"
    assert checks["llm.runtime.contract"]["status"] == "pass"
    assert checks["llm.runtime.contract"]["details"]["verified_bundle_files"] == 1
    assert checks["llm.evaluation.contract"]["status"] == "pass"
    assert checks["release.runtime_evidence.inventory"]["status"] == "pass"
    assert checks["release.runtime_evidence.semantic"]["status"] == "pass"
    assert (
        checks["release.runtime_evidence.semantic"]["details"]["precision_pair_record_count"] == 1
    )
    rendered = json.dumps(result)
    assert "s" * 48 not in rendered
    assert "k" * 48 not in rendered

    outside_result = root / "outside-data-root.json"
    _write_json(outside_result, {"status": "outside"})
    tampered_runtime_records = json.loads(json.dumps(runtime_records))
    tampered_runtime_records[1]["result_sha256"] = "0" * 64
    tampered_runtime_records[2]["result_path"] = str(outside_result)
    tampered_runtime_records[2]["result_sha256"] = _sha256(outside_result)
    restore_incident = json.loads(json.dumps(runtime_records[0]))
    restore_incident.update(
        {
            "kind": "qwen35_bf16_q4_constrained_narrator_paired_regression",
            "run_id": "synthetic-restore-incident",
            "status": "failed_restore_requires_operator",
        }
    )
    tampered_runtime_records.append(restore_incident)
    runtime_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in tampered_runtime_records) + "\n",
        encoding="utf-8",
    )
    tampered_receipts = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    semantic_failures = _checks(tampered_receipts)["release.runtime_evidence.semantic"]["details"][
        "failures"
    ]
    assert tampered_receipts["ok"] is False
    assert {failure["reason"] for failure in semantic_failures} >= {
        "result_sha256_mismatch",
        "result_path_outside_data_root",
        "required_runtime_records_missing",
    }
    missing_required = next(
        failure
        for failure in semantic_failures
        if failure["reason"] == "required_runtime_records_missing"
    )
    assert "operator_recovery_after_restore_incident" in missing_required["missing"]
    runtime_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in runtime_records) + "\n",
        encoding="utf-8",
    )

    tampered_coexistence = json.loads(json.dumps(coexistence_record))
    tampered_coexistence["selection_use"] = True
    tampered_coexistence["result_sha256"] = "0" * 64
    evaluation_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in (tampered_coexistence, pair_harness_record))
        + "\n",
        encoding="utf-8",
    )
    tampered_coexistence_receipt = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    coexistence_failures = _checks(tampered_coexistence_receipt)[
        "release.runtime_evidence.semantic"
    ]["details"]["failures"]
    assert tampered_coexistence_receipt["ok"] is False
    assert {failure["reason"] for failure in coexistence_failures} >= {
        "governance_declaration_not_explicitly_false",
        "result_sha256_mismatch",
    }
    evaluation_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in (coexistence_record, pair_harness_record))
        + "\n",
        encoding="utf-8",
    )

    tampered_pair_harness = json.loads(json.dumps(pair_harness_record))
    tampered_pair_harness["result_sha256"] = "0" * 64
    evaluation_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in (coexistence_record, tampered_pair_harness))
        + "\n",
        encoding="utf-8",
    )
    tampered_pair_receipt = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    pair_failures = _checks(tampered_pair_receipt)["release.runtime_evidence.semantic"]["details"][
        "failures"
    ]
    assert tampered_pair_receipt["ok"] is False
    assert "result_sha256_mismatch" in {failure["reason"] for failure in pair_failures}
    evaluation_ledger_path.write_text(
        "\n".join(json.dumps(record) for record in (coexistence_record, pair_harness_record))
        + "\n",
        encoding="utf-8",
    )

    missing_evidence_path = root / "scripts" / "test_llamacpp_recovery.ps1"
    missing_evidence_path.unlink()
    missing_evidence = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    missing_evidence_check = _checks(missing_evidence)["release.runtime_evidence.inventory"]
    assert missing_evidence["ok"] is False
    assert missing_evidence_check["status"] == "fail"
    assert missing_evidence_check["details"]["missing"] == ["scripts/test_llamacpp_recovery.ps1"]
    missing_evidence_path.write_text(
        "synthetic non-clinical recovery evidence\n",
        encoding="utf-8",
    )

    evaluation_path = root / "evaluation" / "llamacpp_eval_config.json"
    drifted = json.loads(evaluation_path.read_text(encoding="utf-8"))
    drifted.update(
        {
            "selection_use": True,
            "locked_or_hidden_test_used": True,
            "clinical_validation": True,
            "major_variables_changed": ["one", "two", "three"],
        }
    )
    drifted["narrator"]["runtime_config_sha256"] = "0" * 64
    drifted["metrics"].remove("emergency_recall")
    _write_json(evaluation_path, drifted)

    invalid = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    invalid_check = _checks(invalid)["llm.evaluation.contract"]
    assert invalid["ok"] is False
    assert invalid_check["status"] == "fail"
    assert set(invalid_check["details"]["mismatches"]) >= {
        "selection_use",
        "locked_or_hidden_test_used",
        "clinical_validation",
        "major_variables_changed",
        "narrator.runtime_config_sha256",
        "metrics",
    }

    evaluation_path.unlink()
    missing = preflight.run_preflight(
        project_root=root,
        environ=production_environ,
    )
    missing_checks = _checks(missing)
    assert missing["ok"] is False
    assert missing_checks["llm.evaluation.contract"]["details"] == {
        "active": True,
        "reason": "missing",
    }
    assert missing_checks["artifact_inventory.complete"]["status"] == "fail"


def _install_openai_compat_fixture(root: Path) -> dict[str, Path]:
    config_dir = root / "configs"
    runtime_root = root / "runtime" / "openai-compat-fixture"
    runtime_root.mkdir(parents=True, exist_ok=True)
    source_files = {
        "tbx_agent/llm/openai_compat.py": root / "src" / "tbx_agent" / "llm" / "openai_compat.py",
        "tbx_agent/llm/runtime_supervisor.py": root
        / "src"
        / "tbx_agent"
        / "llm"
        / "runtime_supervisor.py",
        "scripts/test_openai_compatible_api.py": root / "scripts" / "test_openai_compatible_api.py",
    }
    for name, path in source_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic source binding: {name}\n", encoding="utf-8")
    source_file_digests = {name: _sha256(path) for name, path in source_files.items()}
    source_tree_sha256 = hashlib.sha256(
        json.dumps(source_file_digests, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    source_binding = {
        "algorithm": "sha256(canonical relative-path to file-sha256 map)",
        "files": source_file_digests,
        "source_tree_sha256": source_tree_sha256,
    }

    runtime_config_path = config_dir / "llm_runtime.yaml"
    model_sha256 = "7" * 64
    runtime_payload = {
        "schema_version": 1,
        "runtime_id": "synthetic-openai-compat-runtime",
        "engine": "llama.cpp",
        "server_build": "b10517",
        "release_archive_sha256": "a" * 64,
        "binary_path": str((runtime_root / "llama-server.exe").resolve()),
        "binary_sha256": "b" * 64,
        "bundle_manifest_path": str((runtime_root / "bundle.json").resolve()),
        "bundle_manifest_sha256": "c" * 64,
        "api_key_file": str((runtime_root / "api.keys").resolve()),
        "model_id": "Qwen/Qwen3.5-4B",
        "model_alias": "tbx-qwen3.5-4b-q4-k-m",
        "model_path": str((runtime_root / "qwen.gguf").resolve()),
        "model_sha256": model_sha256,
        "quantization": "Q4_K_M",
        "host": "127.0.0.1",
        "port": 11435,
        "context_tokens": 8192,
        "parallel_slots": 1,
        "enable_thinking": False,
        "reasoning": "off",
        "max_input_tokens": 4096,
        "max_output_tokens": 512,
        "seed": 20260829,
        "allowed_roles": ["narrator", "evidence_composer"],
        "clinical_authority": False,
        "load_mmproj": False,
    }
    _write_json(runtime_config_path, runtime_payload)
    from tbx_agent.llm.runtime_supervisor import load_runtime_config

    runtime_config = load_runtime_config(runtime_config_path)
    runtime_raw_sha256 = _sha256(runtime_config_path)
    runtime_canonical_sha256 = runtime_config.canonical_sha256()
    source_revision = "a" * 40
    common_full_configuration = {
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_file_sha256": runtime_raw_sha256,
        "runtime_config_canonical_sha256": runtime_canonical_sha256,
        "runtime_id": runtime_config.runtime_id,
        "server_build": runtime_config.server_build,
        "model_alias": runtime_config.model_alias,
        "model_sha256": runtime_config.model_sha256,
        "openai_base_url": "http://127.0.0.1:11435/v1",
        "openai_sdk_version": "2.54.0",
        "openai_sdk_max_retries": 0,
        "http_client_trust_env": False,
        "http_client_follow_redirects": False,
        "tested_endpoints": [
            "/v1/models",
            "/v1/chat/completions:sync",
            "/v1/chat/completions:sse",
        ],
        "request_timeout_seconds": 120,
        "temperature": 0,
        "max_output_tokens": 16,
        "stream_include_usage": True,
        "chat_template_enable_thinking": False,
        "fixture_kind": "fixed_synthetic_non_medical_protocol_prompt",
        "prompt_sha256": "d" * 64,
        "expected_content_sha256": "e" * 64,
        "credential_source": "runtime_acl_key_file",
        "credential_persisted_in_receipt": False,
        "dataset_used": False,
        "seed": 20260829,
        "smoke_module_sha256": source_file_digests["tbx_agent/llm/openai_compat.py"],
        "smoke_entrypoint_sha256": source_file_digests["scripts/test_openai_compatible_api.py"],
        "runtime_asset_hashes_reverified": False,
        "served_process_binary_attested": False,
    }

    evidence_root = runtime_root / "evidence"
    evidence_root.mkdir()
    failed_result = evidence_root / "failed.json"
    passed_result = evidence_root / "passed.json"
    common_receipt = {
        "schema_version": 1,
        "kind": "llamacpp_openai_python_sdk_compatibility_smoke",
        "hypothesis": "synthetic protocol-only fixture",
        "seed": 20260829,
        "split_hash": None,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "source_binding": source_binding,
        "selection_use": False,
        "model_selection": False,
        "threshold_selection": False,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "clinical_validation": False,
        "major_variables_changed": [],
        "full_configuration": common_full_configuration,
        "peak_vram_bytes": None,
        "peak_vram_measurement": (
            "unavailable_not_measured_existing_server_protocol_compatibility_smoke"
        ),
    }
    failed_receipt = {
        **common_receipt,
        "run_id": "openai-compat-20260829T000000000000Z",
        "status": "failed_retained",
        "created_at": "2026-08-29T00:00:00+00:00",
        "metrics": {
            "models_list_success": False,
            "expected_alias_present": False,
            "chat_sync_success": False,
            "chat_sync_exact_content": False,
            "chat_stream_success": False,
            "chat_stream_exact_content": False,
            "stream_usage_chunk_count": 0,
        },
        "runtime": {
            "models_list_milliseconds": None,
            "chat_sync_milliseconds": None,
            "chat_stream_milliseconds": None,
            "total_milliseconds": 5.0,
        },
        "failure": {
            "stage": "load_acl_key",
            "reason": "runtime_or_sdk_error",
            "error_type": "PermissionError",
        },
    }
    passed_receipt = {
        **common_receipt,
        "run_id": "openai-compat-20260829T000001000000Z",
        "status": "passed_openai_compat_smoke",
        "created_at": "2026-08-29T00:00:01+00:00",
        "metrics": {
            "models_list_success": True,
            "expected_alias_present": True,
            "chat_sync_success": True,
            "chat_sync_exact_content": True,
            "chat_stream_success": True,
            "chat_stream_exact_content": True,
            "stream_usage_chunk_count": 1,
            "stream_chunk_count": 5,
            "stream_event_sequence": ["content", "finish", "usage"],
            "chat_sync_prompt_tokens": 15,
            "chat_sync_completion_tokens": 5,
            "chat_sync_total_tokens": 20,
            "chat_stream_prompt_tokens": 15,
            "chat_stream_completion_tokens": 5,
            "chat_stream_total_tokens": 20,
        },
        "runtime": {"total_milliseconds": 100.0},
        "failure": None,
    }
    _write_json(failed_result, failed_receipt)
    _write_json(passed_result, passed_receipt)

    contract_path = config_dir / "openai_compat_test.yaml"
    contract = {
        "schema_version": 1,
        "contract_id": "qwen35-4b-q4-k-m-llamacpp-openai-protocol-test-v1",
        "status": "protocol_test_only",
        "source_runtime_config": "configs/llm_runtime.yaml",
        "governance": {
            "selection_use": False,
            "locked_or_hidden_test_used": False,
            "clinical_validation": False,
            "clinical_authority": False,
            "agent_output": False,
            "intended_use": (
                "loopback_raw_model_protocol_testing_with_bearer_protected_generation"
            ),
            "prohibited_claims": [
                "tbx_agent_response",
                "diagnosis_or_exclusion",
                "treatment_recommendation",
                "clinical_validation",
            ],
        },
        "transport": {
            "implementation": "native_llama_server",
            "listen_origin": "http://127.0.0.1:11435",
            "api_prefix": "/v1",
            "openai_base_url": "http://127.0.0.1:11435/v1",
            "loopback_only": True,
            "public_network_exposure_allowed": False,
            "tls_terminated_here": False,
            "remote_access": {
                "allowed_without_gateway": False,
                "gateway_required": True,
                "gateway_implemented": False,
                "gateway_tls_required": True,
                "gateway_authentication": "independent_bearer_token",
                "internal_llama_key_reuse_allowed": False,
            },
        },
        "authentication": {
            "behavior": "endpoint_specific_native_llama_server_b10517",
            "models_endpoint": {
                "scheme": "none",
                "native_public_endpoint": True,
                "fake_bearer_observed_http_status": 200,
                "must_not_be_used_as_authentication_probe": True,
                "risk_mitigation": "loopback_only",
            },
            "generation_endpoint": {
                "scheme": "bearer",
                "invalid_bearer_observed_http_status": 401,
                "credential_purpose": "internal_llama_server_loopback_only",
            },
            "key_value_in_config_allowed": False,
            "key_value_in_logs_allowed": False,
            "key_value_in_command_line_allowed": False,
            "key_value_in_source_control_allowed": False,
        },
        "model": {
            "public_alias": runtime_config.model_alias,
            "upstream_alias": runtime_config.model_alias,
            "model_sha256": runtime_config.model_sha256,
            "engine": "llama.cpp",
            "server_build": "b10517",
            "modality": "text_only",
            "thinking": False,
            "clinical_authority": False,
        },
        "protocol": {
            "dialect": "openai_chat_completions",
            "endpoints": {
                "models": {"method": "GET", "path": "/v1/models"},
                "chat_completions": {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "synchronous_json": True,
                    "streaming_sse": True,
                    "terminal_sse_marker": "[DONE]",
                },
            },
            "supported_message_roles": ["system", "user", "assistant"],
            "message_content": "plain_text_only",
            "generation": {
                "temperature": 0,
                "enable_thinking": False,
                "maximum_input_tokens": 4096,
                "maximum_output_tokens": 512,
                "parallel_slots": 1,
            },
            "unsupported_capabilities": [
                "agent_tools",
                "function_calling",
                "image_or_multimodal_input",
                "embeddings",
                "files",
                "audio",
                "openai_responses_api",
            ],
        },
        "separation_boundary": {
            "tbx_agent_api_base_url": "http://127.0.0.1:8000",
            "raw_model_api_base_url": "http://127.0.0.1:11435/v1",
            "raw_output_enters_agent_memory": False,
            "raw_output_enters_case_or_report": False,
            "raw_output_passes_agent_safety_verifier": False,
        },
        "live_smoke": {
            "command": (
                "python scripts/test_openai_compatible_api.py --runtime-config "
                "configs/llm_runtime.yaml"
            ),
            "evidence_status": ("passed_engineering_protocol_smoke_with_failed_attempt_retained"),
            "publication_or_release_evidence": False,
            "clinical_validation": False,
            "seed": 20260829,
            "split_hash": None,
            "source_revision": source_revision,
            "source_tree_sha256": source_tree_sha256,
            "runtime_config_file_sha256": runtime_raw_sha256,
            "runtime_config_canonical_sha256": runtime_canonical_sha256,
            "sdk": {
                "version": "2.54.0",
                "max_retries": 0,
                "http_client_trust_env": False,
                "http_client_follow_redirects": False,
            },
            "passed_result": {
                "path": str(passed_result),
                "sha256": _sha256(passed_result),
                "status": "passed_openai_compat_smoke",
                "models_list_success": True,
                "chat_sync_success": True,
                "chat_stream_success": True,
                "stream_terminal_usage_observed": True,
                "peak_vram_bytes": None,
                "peak_vram_measurement": (
                    "unavailable_not_measured_existing_server_protocol_compatibility_smoke"
                ),
            },
            "retained_failed_result": {
                "path": str(failed_result),
                "sha256": _sha256(failed_result),
                "status": "failed_retained",
                "failure_stage": "load_acl_key",
                "failure_type": "PermissionError",
            },
        },
    }
    _write_json(contract_path, contract)
    return {
        "contract": contract_path,
        "runtime": runtime_config_path,
        "passed": passed_result,
        "failed": failed_result,
    }


def _openai_protocol_check(result: dict) -> dict:
    return _checks(result)["llm.openai_compat.protocol"]


def test_openai_compat_protocol_gate_passes_in_research_and_stays_non_deployable(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    paths = _install_openai_compat_fixture(root)

    research = preflight.run_preflight(project_root=root, environ={})
    research_check = _openai_protocol_check(research)
    assert research["ok"] is True
    assert research_check["status"] == "pass"
    assert research_check["details"]["profile"] == "research"
    assert research_check["details"]["engineering_protocol_only"] is True
    assert research_check["details"]["clinical_validation"] is False
    assert research_check["details"]["agent_output"] is False
    assert research_check["details"]["release_authorized"] is False
    assert research_check["details"]["deployment_safety_decision"] == "unchanged_no_go"
    assert research_check["details"]["retained_failed_result_sha256"] == _sha256(paths["failed"])

    override_root = root / "runtime" / "openai-compat-overrides"
    override_root.mkdir()
    passed_override = override_root / "passed.json"
    failed_override = override_root / "failed.json"
    passed_override.write_bytes(paths["passed"].read_bytes())
    failed_override.write_bytes(paths["failed"].read_bytes())
    overridden = preflight.run_preflight(
        project_root=root,
        environ={},
        openai_compat_result=passed_override,
        openai_compat_failed_result=failed_override,
    )
    overridden_check = _openai_protocol_check(overridden)
    assert overridden_check["status"] == "pass"
    assert overridden_check["details"]["passed_result_path"] == str(passed_override)
    assert overridden_check["details"]["retained_failed_result_path"] == str(failed_override)

    production = preflight.run_preflight(
        project_root=root,
        environ={"TBX_AGENT_DEPLOYMENT_PROFILE": "production"},
    )
    production_check = _openai_protocol_check(production)
    assert production["ok"] is False
    assert production["deployment"]["release_authorized"] is False
    assert production_check["status"] == "pass"
    assert production_check["details"]["profile"] == "production"
    assert production_check["details"]["deployment_safety_decision"] == "unchanged_no_go"


def test_openai_compat_protocol_gate_rejects_contract_runtime_drift(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    paths = _install_openai_compat_fixture(root)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract["model"]["public_alias"] = "drifted-alias"
    contract["transport"]["public_network_exposure_allowed"] = True
    contract["governance"]["agent_output"] = True
    _write_json(paths["contract"], contract)

    result = preflight.run_preflight(project_root=root, environ={})
    check = _openai_protocol_check(result)
    mismatch_fields = {
        failure.get("field")
        for failure in check["details"]["failures"]
        if failure["reason"] == "contract_runtime_mismatch"
    }
    assert result["ok"] is False
    assert check["status"] == "fail"
    assert mismatch_fields >= {
        "model.public_alias",
        "transport.public_network_exposure_allowed",
        "governance.agent_output",
    }


def test_openai_compat_protocol_gate_rejects_tampered_or_missing_receipts(
    tmp_path: Path,
) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    paths = _install_openai_compat_fixture(root)
    passed = json.loads(paths["passed"].read_text(encoding="utf-8"))
    passed["metrics"]["chat_sync_total_tokens"] = 999
    _write_json(paths["passed"], passed)
    paths["failed"].unlink()

    result = preflight.run_preflight(project_root=root, environ={})
    reasons = {
        failure["reason"] for failure in _openai_protocol_check(result)["details"]["failures"]
    }
    assert result["ok"] is False
    assert "passed_result_sha256_mismatch" in reasons
    assert "retained_failed_result_missing" in reasons


def test_openai_compat_protocol_gate_rejects_hash_valid_fake_success(tmp_path: Path) -> None:
    root, _ = _build_preflight_tree(tmp_path, backend="mock")
    paths = _install_openai_compat_fixture(root)
    fake = json.loads(paths["passed"].read_text(encoding="utf-8"))
    fake["metrics"]["chat_stream_success"] = False
    fake["metrics"]["stream_usage_chunk_count"] = 0
    fake["metrics"]["stream_event_sequence"] = ["content", "finish"]
    _write_json(paths["passed"], fake)
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    contract["live_smoke"]["passed_result"]["sha256"] = _sha256(paths["passed"])
    _write_json(paths["contract"], contract)

    result = preflight.run_preflight(project_root=root, environ={})
    failures = _openai_protocol_check(result)["details"]["failures"]
    reasons = {failure["reason"] for failure in failures}
    assert result["ok"] is False
    assert "passed_result_sha256_mismatch" not in reasons
    assert "passed_result_metric_not_true" in reasons
    assert "passed_result_usage_chunk_count_mismatch" in reasons
    assert "passed_result_stream_terminal_sequence_invalid" in reasons
