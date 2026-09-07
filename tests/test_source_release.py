from __future__ import annotations

import sys
import zipfile
from pathlib import Path, PurePosixPath

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_source_release import (  # noqa: E402
    ARCHIVE_PREFIX,
    DEPLOYMENT_RELEASE_FILES,
    REQUIRED_RELEASE_FILES,
    ReleasePolicyError,
    build_archive,
    collect_source_files,
    is_public_path,
    scan_content,
    validate_release_path,
)

SYSTEM_BENCH_RELEASE_FILES = {
    ".github/workflows/ci.yml",
    "docs/evaluation.md",
    "evaluation/system_bench_config.json",
    "evaluation/system_bench_config_v1_1.json",
    "evaluation/system_bench_config_v1_2.json",
    "evaluation/system_bench_config_v1_3.json",
    "evaluation/system_bench_config_v1_4.json",
    "evaluation/system_bench_config_v1_5.json",
    "evaluation/system_bench_config_v1_6.json",
    "evaluation/suites/system_v1/README.md",
    "evaluation/suites/system_v1/cases.jsonl",
    "evaluation/suites/system_v1/manifest.json",
    "evaluation/suites/system_v1_2/README.md",
    "evaluation/suites/system_v1_2/cases.jsonl",
    "evaluation/suites/system_v1_2/manifest.json",
    "evaluation/suites/system_v1_3/README.md",
    "evaluation/suites/system_v1_3/cases.jsonl",
    "evaluation/suites/system_v1_3/manifest.json",
    "evaluation/suites/system_v1_4/README.md",
    "evaluation/suites/system_v1_4/cases.jsonl",
    "evaluation/suites/system_v1_4/manifest.json",
    "evaluation/suites/system_v1_5/README.md",
    "evaluation/suites/system_v1_5/cases.jsonl",
    "evaluation/suites/system_v1_5/manifest.json",
    "evaluation/suites/system_v1_6/README.md",
    "evaluation/suites/system_v1_6/cases.jsonl",
    "evaluation/suites/system_v1_6/manifest.json",
    "evaluation/suites/trajectory_v1/README.md",
    "evaluation/suites/trajectory_v1/cases.jsonl",
    "evaluation/suites/trajectory_v1/manifest.json",
    "evaluation/suites/trajectory_v2/README.md",
    "evaluation/suites/trajectory_v2/cases.jsonl",
    "evaluation/suites/trajectory_v2/manifest.json",
    "evaluation/suites/trajectory_v3/README.md",
    "evaluation/suites/trajectory_v3/cases.jsonl",
    "evaluation/suites/trajectory_v3/manifest.json",
    "docs/agent_trajectory_evaluation.md",
    "src/tbx_agent/evaluation/trajectory.py",
    "src/tbx_agent/evaluation/system_bench.py",
    "src/tbx_agent/evaluation/system_bench_ci.py",
}

SOFTWARE_CONFORMANCE_RELEASE_FILES = {
    "evaluation/software_conformance_config_v1.json",
    "evaluation/suites/software_conformance_v1/README.md",
    "evaluation/suites/software_conformance_v1/cases.jsonl",
    "evaluation/suites/software_conformance_v1/manifest.json",
    "src/tbx_agent/evaluation/software_conformance.py",
}

OPTIONAL_VISION_RELEASE_FILES = {
    "docs/anatomy_spatial_evidence.md",
    "docs/medsam_refinement.md",
    "scripts/smoke_medsam_refinement.py",
}

ACTIVE_RETRIEVAL_EVAL_RELEASE_FILES = {
    "evaluation/retrieval/smoke_v5/config.json",
    "evaluation/retrieval/smoke_v5/qrels.jsonl",
    "evaluation/retrieval/smoke_v5/queries.jsonl",
    "evaluation/retrieval/smoke_v5/README.md",
    "evaluation/retrieval/core_guideline_seven_v2/config.json",
    "evaluation/retrieval/core_guideline_seven_v2/qrels.jsonl",
    "evaluation/retrieval/core_guideline_seven_v2/queries.jsonl",
    "evaluation/retrieval/core_guideline_seven_v2/README.md",
    "evaluation/retrieval/cdc_supplemental_v2/config.json",
    "evaluation/retrieval/cdc_supplemental_v2/qrels.jsonl",
    "evaluation/retrieval/cdc_supplemental_v2/queries.jsonl",
    "evaluation/retrieval/cdc_supplemental_v2/README.md",
}

MEDICAL_DIALOGUE_RUNTIME_RELEASE_FILES = {
    "docs/evaluation.md",
    "evaluation/fixtures/medical_dialogue_qa_v1.json",
    "scripts/evaluate_medical_dialogue_runtime.py",
}

INTERNAL_RELEASE_EXCLUSIONS = {
    "artifacts/system-bench/report.json",
    "docs/cards/system_eval_dataset_card.md",
    "evaluation/ledger.jsonl",
    "evaluation/runtime_operations_ledger.jsonl",
    "evaluation/suites/system_v1/deterministic_mock_candidate.json",
    "evaluation/suites/system_v1_2/candidate.json",
    "evaluation/suites/system_v1_2/report.json",
    "evaluation/suites/system_v1_3/candidate.json",
    "evaluation/suites/system_v1_3/report.json",
    "weights/rank03.pt",
}


def test_agent_system_bench_release_contract_is_explicit() -> None:
    assert SYSTEM_BENCH_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in SYSTEM_BENCH_RELEASE_FILES)
    assert all(not is_public_path(path) for path in INTERNAL_RELEASE_EXCLUSIONS)


def test_software_conformance_release_contract_is_explicit() -> None:
    assert SOFTWARE_CONFORMANCE_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in SOFTWARE_CONFORMANCE_RELEASE_FILES)


def test_active_retrieval_evaluation_fixtures_ship_in_source_release() -> None:
    assert ACTIVE_RETRIEVAL_EVAL_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in ACTIVE_RETRIEVAL_EVAL_RELEASE_FILES)


def test_medical_dialogue_runtime_evaluator_ships_with_its_fixture() -> None:
    assert MEDICAL_DIALOGUE_RUNTIME_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in MEDICAL_DIALOGUE_RUNTIME_RELEASE_FILES)


def test_deployment_surface_is_mandatory_in_every_source_release() -> None:
    assert DEPLOYMENT_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in DEPLOYMENT_RELEASE_FILES)
    assert "scripts/build_llamacpp_linux.sh" in DEPLOYMENT_RELEASE_FILES
    assert OPTIONAL_VISION_RELEASE_FILES <= DEPLOYMENT_RELEASE_FILES
    assert OPTIONAL_VISION_RELEASE_FILES <= REQUIRED_RELEASE_FILES
    assert all(is_public_path(path) for path in OPTIONAL_VISION_RELEASE_FILES)
    assert "scripts/evaluate_agent_runtime.py" in DEPLOYMENT_RELEASE_FILES
    assert "scripts/evaluate_medical_dialogue_runtime.py" in DEPLOYMENT_RELEASE_FILES


@pytest.mark.parametrize("path", [
    "src/tbx_agent/training/rank03_classifier.py",
    "src/tbx_agent/training/rank03_detector.py",
    "scripts/train_rank03.py",
    "src/tbx_agent/vision/train_segmentation.py",
])
def test_training_implementations_cannot_enter_inference_release(path: str) -> None:
    assert not is_public_path(path)
    with pytest.raises(ReleasePolicyError):
        validate_release_path(path)


def test_inference_distribution_has_no_training_entrypoint_or_package() -> None:
    import tomllib

    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "rank03-training" not in project["project"]["optional-dependencies"]
    assert all("training" not in value for value in project["project"]["scripts"].values())
    assert not (PROJECT_ROOT / "src/tbx_agent/training").exists()
    assert not (PROJECT_ROOT / "scripts/train_rank03.py").exists()


def test_linux_llamacpp_builder_is_explicit_and_candidate_only() -> None:
    script = (PROJECT_ROOT / "scripts/build_llamacpp_linux.sh").read_text(encoding="utf-8")
    assert "EXECUTE=0" in script
    assert "--execute" in script
    assert "candidate_unregistered" in script
    assert "download-llama-source" in script
    assert "Candidate only: no runtime identity" in script


def test_compose_real_profile_is_fail_closed_and_health_means_ready() -> None:
    compose_path = PROJECT_ROOT / "docker-compose.yml"
    compose_text = compose_path.read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)

    health_command = " ".join(compose["x-runtime"]["healthcheck"]["test"])
    assert "/readyz" in health_command
    assert "/livez" not in health_command

    required_variables = {
        "TBX_MODEL_CACHE",
        "TBX_DFINE_ROOT",
        "TBX_RANK03_RUNTIME_CONFIG_HOST",
        "TBX_LLM_RUNTIME_CONFIG_HOST",
        "TBX_LLAMA_BUNDLE_DIR",
        "TBX_LLAMA_BUNDLE_MANIFEST_HOST",
        "TBX_LLAMA_SERVER_BINARY_NAME",
        "LLAMA_CPP_API_KEY_FILE_HOST",
        "LLAMA_CPP_MODEL_SHA256",
        "TBX_EXTERNAL_LLM_URL",
    }
    for variable in required_variables:
        assert f"${{{variable}:?" in compose_text

    assert "${LLAMA_CPP_MODEL_SHA256:-}" not in compose_text
    api = compose["services"]["api-rank03"]
    assert compose["x-runtime"]["read_only"] is True
    assert compose["x-runtime"]["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in compose["x-runtime"]["security_opt"]
    assert api["environment"]["TBX_AGENT_LLM_RUNTIME_CONFIG"] == (
        "/run/tbx/llm_runtime.generated.yaml"
    )
    assert api["environment"]["TBX_ARTIFACT_ROOT"] == "/models"
    assert api["environment"]["TBX_AGENT_CASE_ARTIFACT_ROOT"] == "/data/cases"
    assert api["environment"]["TBX_AGENT_CONTOUR_REFINEMENT_BACKEND"] == (
        "${TBX_AGENT_CONTOUR_REFINEMENT_BACKEND:-none}"
    )
    assert api["environment"]["LLAMA_CPP_BUNDLE_MANIFEST_PATH"] == (
        "/run/tbx/llama-bundle-manifest.json"
    )
    mounted_targets = {item["target"] for item in api["volumes"]}
    assert {
        "/data",
        "/opt/llama",
        "/run/tbx/llm_runtime.generated.yaml",
        "/run/tbx/llama-bundle-manifest.json",
    } <= mounted_targets
    ui = compose["services"]["ui-rank03"]
    assert ui["read_only"] is True
    assert ui["cap_drop"] == ["ALL"]
    assert compose["x-build-control"]["args"]["TBX_INSTALL_EXTRAS"] == (
        "${TBX_UI_INSTALL_EXTRAS:-ui,dicom}"
    )
    assert compose["services"]["api-rank03"]["ports"] == ["127.0.0.1:${TBX_API_PORT:-8000}:8000"]
    assert compose["services"]["ui-rank03"]["ports"] == ["127.0.0.1:${TBX_UI_PORT:-8501}:8501"]


def test_docker_image_is_non_editable_and_unprivileged() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'python -m pip install ".[${TBX_INSTALL_EXTRAS}]"' in dockerfile
    assert "pip install --no-cache-dir -e" not in dockerfile
    assert "PIP_NO_CACHE_DIR=1" in dockerfile
    assert "USER tbx" in dockerfile
    assert "STOPSIGNAL SIGTERM" in dockerfile


@pytest.mark.parametrize("launcher", ["scripts/run_local.ps1", "scripts/run_local.sh"])
def test_native_launchers_wait_for_readiness_before_starting_ui(launcher: str) -> None:
    launcher_text = (PROJECT_ROOT / launcher).read_text(encoding="utf-8")
    assert "/readyz" in launcher_text
    assert "/livez" not in launcher_text


def test_native_launchers_preserve_an_explicit_external_artifact_cache() -> None:
    powershell = (PROJECT_ROOT / "scripts/run_local.ps1").read_text(encoding="utf-8")
    posix = (PROJECT_ROOT / "scripts/run_local.sh").read_text(encoding="utf-8")

    assert "Get-PlatformArtifactRoot" in powershell
    assert "GetEnvironmentVariable($Pair[0], 'Process')" in powershell
    assert "if ([string]::IsNullOrWhiteSpace($env:TBX_ARTIFACT_ROOT))" in powershell
    assert "platform_artifact_root" in posix
    assert 'eval "existing=\\${$key-}"' in posix
    assert 'if [ -z "${TBX_ARTIFACT_ROOT:-}" ]; then' in posix
    assert 'TBX_AGENT_DATA_ROOT/artifacts' not in powershell
    assert '$TBX_AGENT_DATA_ROOT/artifacts' not in posix


def test_windows_launcher_rebinds_clean_exec_wrapper_to_verified_listener() -> None:
    launcher_text = (PROJECT_ROOT / "scripts/run_local.ps1").read_text(encoding="utf-8")

    assert "WrapperExitedCleanly" in launcher_text
    assert "Get-VerifiedLoopbackListenerProcess" in launcher_text
    assert "ExpectedLlmExecutable" in launcher_text
    assert "Ready llama.cpp listener could not be bound" in launcher_text
    assert "did not become ready" in launcher_text


@pytest.mark.parametrize(
    "relative",
    [
        "evaluation/suites/system_v1_2/candidate.json",
        "evaluation/suites/system_v1_2/observations.jsonl",
        "evaluation/suites/system_v1_2/report.json",
        "evaluation/suites/system_v1_2/ledger.jsonl",
        "evaluation/suites/unreviewed/cases.jsonl",
    ],
)
def test_suite_allowlist_rejects_generated_and_unreviewed_files(relative: str) -> None:
    with pytest.raises(ReleasePolicyError, match="outside the public source allowlist"):
        validate_release_path(relative)


@pytest.mark.parametrize(
    "relative",
    [
        "tests/data/patient_rows.json",
        "src/tbx_agent/weights/checkpoint_manifest.json",
        "ui/runtime/session.json",
    ],
)
def test_nested_runtime_data_directories_fail_closed(relative: str) -> None:
    with pytest.raises(ReleasePolicyError, match="denied path component"):
        validate_release_path(relative)


def test_model_artifact_manager_source_package_remains_public() -> None:
    assert validate_release_path("src/tbx_agent/artifacts/manifest.py")


def test_workstation_profile_paths_are_rejected_in_every_public_file() -> None:
    local_path = b"C:" + rb"\Users\release-owner\Desktop\private-receipt.json"
    with pytest.raises(ReleasePolicyError, match="workstation-specific"):
        scan_content("docs/evaluation.md", b"load " + local_path)


def test_source_archive_contains_v12_contract_without_internal_outputs(tmp_path: Path) -> None:
    files = collect_source_files(PROJECT_ROOT)
    paths = {item.path for item in files}
    assert paths >= SYSTEM_BENCH_RELEASE_FILES
    assert INTERNAL_RELEASE_EXCLUSIONS.isdisjoint(paths)

    forbidden_components = {
        ".venv",
        "artifacts",
        "checkpoints",
        "datasets",
        "dist",
        "reports",
        "runtime",
        "weights",
    }
    assert not any(PurePosixPath(path).parts[0] in forbidden_components for path in paths)

    archive_path = tmp_path / "tbx-agent-source.zip"
    build_archive(files, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archived = {name.removeprefix(f"{ARCHIVE_PREFIX}/") for name in archive.namelist()}
    assert archived == paths
