from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_command(launcher: str, script: Path, *, bundle: bool) -> list[str]:
    if launcher.endswith(".ps1"):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            pytest.skip("PowerShell is unavailable")
        command = [
            shell, "-NoProfile", "-NonInteractive", "-File", str(script),
            "-DryRun", "-Python", "must-not-be-executed", "-CacheDir", "artifacts",
        ]
        if bundle:
            command += [
                "-VisionBundle", "manual bundle.zip", "-RuntimeConfig", "runtime/config.json",
            ]
    else:
        shell = shutil.which("sh")
        git_shell = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/sh.exe"
        if shell is None and git_shell.is_file():
            shell = str(git_shell)
        if shell is None:
            pytest.skip("POSIX shell is unavailable")
        command = [
            shell, script.as_posix(), "--dry-run", "--python", "must-not-be-executed",
            "--cache-dir", "artifacts",
        ]
        if bundle:
            command += [
                "--vision-bundle", "manual bundle.zip", "--runtime-config", "runtime/config.json",
            ]
    return command


@pytest.mark.parametrize("launcher", ["bootstrap.ps1", "bootstrap.sh"])
@pytest.mark.parametrize("bundle", [False, True])
def test_bootstrap_dry_run_installs_lightweight_demo_or_explicit_bundle(
    tmp_path: Path, launcher: str, bundle: bool,
) -> None:
    script = tmp_path / "project" / "scripts" / launcher
    script.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "scripts" / launcher, script)
    result = subprocess.run(
        _bootstrap_command(launcher, script, bundle=bundle),
        cwd=tmp_path,
        env={**os.environ, "TBX_INSTALL_EXTRAS": "ui,dicom"},
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    actions = [line for line in result.stdout.splitlines() if line.startswith("> ")]
    assert any("[ui,dicom" in action for action in actions)
    assert "no files changed" in result.stdout
    assert not (script.parent.parent / ".env").exists()
    assert not (script.parent.parent / ".venv").exists()
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "runtime").exists()
    assert not any(
        "bootstrap_models.py" in action or "bootstrap_dfine.py" in action for action in actions
    )
    installs = [action for action in actions if "install_vision_bundle.py" in action]
    assert len(installs) == int(bundle)
    if bundle:
        assert "[ui,dicom,vision]" in result.stdout
        for argument in ("--bundle", "manual bundle.zip", "--artifact-root", "--runtime-config"):
            assert argument in installs[0]


@pytest.mark.parametrize("launcher", ["bootstrap.ps1", "bootstrap.sh"])
def test_bundle_bootstrap_requires_explicit_external_runtime_output(
    tmp_path: Path, launcher: str,
) -> None:
    command = _bootstrap_command(launcher, PROJECT_ROOT / "scripts" / launcher, bundle=True)
    result = subprocess.run(
        command[:-2], cwd=tmp_path, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=15,
    )
    assert result.returncode != 0
    assert "requires" in (result.stdout + result.stderr)
    assert not any(line.startswith("> ") for line in result.stdout.splitlines())


@pytest.fixture
def setup_audit():
    spec = importlib.util.spec_from_file_location(
        "release_setup_audit", PROJECT_ROOT / "scripts" / "check_setup.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_setup_audit_directs_missing_and_template_contracts_to_bundle_installer(setup_audit):
    for environment in (
        {},
        {"TBX_AGENT_RANK03_RUNTIME_CONFIG": str(PROJECT_ROOT / "configs" / "rank03_runtime.json")},
    ):
        check = setup_audit._check_rank03(environment)[0]
        assert check.status == "FAIL"
        assert "install_vision_bundle.py" in check.remediation
        assert "--runtime-config" in check.remediation
        assert "train" not in check.remediation.lower()


@pytest.mark.parametrize("mode", ["demo", "vision"])
def test_setup_demo_uses_deterministic_narrator_without_relaxing_real_mode(
    setup_audit, monkeypatch, tmp_path: Path, mode: str,
):
    monkeypatch.setattr(setup_audit, "_effective_environment", lambda _dotenv: {
        "LLM_PROVIDER": "openai",
        "TBX_AGENT_NARRATOR_BACKEND": "llama_cpp",
        "TBX_AGENT_RETRIEVAL_CONFIG": "unavailable-model-retrieval.yaml",
    })
    checks = setup_audit.run_checks(mode, tmp_path / "absent.env")
    llm_checks = [check for check in checks if check.name.startswith("llm")]
    assert len(llm_checks) == 1
    assert llm_checks[0].status == ("PASS" if mode == "demo" else "FAIL")
    if mode == "demo":
        assert "deterministic" in llm_checks[0].message
        assert not any(check.name.startswith("rag") and check.status == "FAIL" for check in checks)
