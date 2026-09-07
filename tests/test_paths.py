from __future__ import annotations

from pathlib import Path

from tbx_agent.config import Settings, resolve_model_path
from tbx_agent.paths import default_runtime_root, discover_project_root, resolve_portable_path


def test_project_root_discovery_honors_explicit_override(tmp_path: Path) -> None:
    checkout = tmp_path / "mounted-checkout"
    assert discover_project_root(
        {"TBX_AGENT_PROJECT_ROOT": str(checkout)},
        source_checkout=tmp_path / "site-packages",
        cwd=tmp_path / "other",
    ) == checkout.resolve()


def test_project_root_discovery_supports_noneditable_wheel_from_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "knowledge").mkdir()
    (checkout / "configs" / "app.yaml").write_text("runtime: {}\n", encoding="utf-8")
    (checkout / "knowledge" / "source_manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )

    assert discover_project_root(
        {},
        source_checkout=tmp_path / "python" / "site-packages",
        cwd=checkout,
    ) == checkout.resolve()


def test_runtime_root_uses_platform_data_directories_without_workstation_probe(
    tmp_path: Path,
) -> None:
    local_app_data = tmp_path / "LocalAppData"
    assert default_runtime_root(
        {"LOCALAPPDATA": str(local_app_data)}, platform="win32"
    ) == local_app_data / "TBX-Agent" / "runtime"

    assert default_runtime_root(
        {"XDG_DATA_HOME": "/var/lib/test-user"}, platform="linux"
    ) == Path("/var/lib/test-user/tbx-agent")


def test_runtime_root_honors_explicit_data_root_before_legacy_alias(tmp_path: Path) -> None:
    data_root = tmp_path / "canonical-data"
    legacy_root = tmp_path / "legacy-runtime"
    assert default_runtime_root(
        {
            "TBX_AGENT_DATA_ROOT": str(data_root),
            "TBX_RUNTIME_ROOT": str(legacy_root),
        },
        platform="win32",
    ) == data_root


def test_runtime_reference_uses_explicit_external_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "runtime-on-any-volume"
    resolved = resolve_portable_path(
        "runtime://evaluation_runs/receipt.json",
        base=tmp_path / "project",
        environment={"TBX_AGENT_DATA_ROOT": str(data_root)},
    )
    assert resolved == (data_root / "evaluation_runs" / "receipt.json").resolve()


def test_case_storage_is_separate_from_immutable_model_cache(
    tmp_path: Path, monkeypatch,
) -> None:
    data_root = tmp_path / "runtime"
    model_root = tmp_path / "models"
    monkeypatch.setenv("TBX_AGENT_DATA_ROOT", str(data_root))
    monkeypatch.setenv("TBX_ARTIFACT_ROOT", str(model_root))
    monkeypatch.delenv("TBX_AGENT_CASE_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("TBX_AGENT_ARTIFACT_ROOT", raising=False)

    settings = Settings.from_env()

    assert settings.artifact_root == (data_root / "cases").resolve()
    assert settings.case_artifact_root == (data_root / "cases").resolve()
    assert settings.case_artifact_root != model_root.resolve()
    assert resolve_model_path(
        "artifact://rank03/classifier/best.pt",
        "TBX_TEST_MODEL_PATH",
        project_root=tmp_path,
        environment={"TBX_ARTIFACT_ROOT": str(model_root)},
    ) == (model_root / "rank03/classifier/best.pt").resolve()
