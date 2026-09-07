from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PROFILE = {
    "TBX_AGENT_VISION_BACKEND": "rank03",
    "TBX_AGENT_NARRATOR_BACKEND": "llama_cpp",
    "TBX_AGENT_REQUIRE_REAL_INFERENCE": "true",
    "TBX_AGENT_REQUIRE_LLM_INFERENCE": "true",
    "TBX_AGENT_OPENAI_ENABLED": "true",
    "TBX_AGENT_ANATOMY_BACKEND": "xrv_pspnet",
    "TBX_AGENT_REQUIRE_ANATOMY_INFERENCE": "true",
    "TBX_AGENT_CONTOUR_REFINEMENT_BACKEND": "medsam_hf",
    "LLM_PROVIDER": "openai",
    "TBX_AGENT_RETRIEVAL_CONFIG": "not-the-demo-retrieval.yaml",
}


def _demo_environment(launcher: str) -> dict[str, str]:
    # Execute only the launcher's real demo branch. This exercises shell
    # assignments without reading either local dotenv file or starting servers.
    text = (PROJECT_ROOT / "scripts" / launcher).read_text(encoding="utf-8")
    if launcher.endswith(".ps1"):
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if shell is None:
            pytest.skip("PowerShell is unavailable")
        match = re.search(r"(?ms)^if \(\$Demo\) \{.*?^\}", text)
        assert match is not None
        outputs = "\n".join(f"Write-Output $env:{name}" for name in REAL_PROFILE)
        command = [
            shell, "-NoProfile", "-NonInteractive", "-Command",
            "$Demo = $true\n$ProjectRoot = $env:TBX_AGENT_PROJECT_ROOT\n"
            + match.group() + "\n" + outputs,
        ]
    else:
        shell = shutil.which("sh")
        git_shell = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/sh.exe"
        if shell is None and git_shell.is_file():
            shell = str(git_shell)
        if shell is None:
            pytest.skip("POSIX shell is unavailable")
        match = re.search(r'(?ms)^if \[ "\$DEMO" -eq 1 \]; then.*?^fi', text)
        assert match is not None
        outputs = "\n".join(f'printf "%s\\n" "${{{name}}}"' for name in REAL_PROFILE)
        command = [shell, "-c", "DEMO=1\n" + match.group() + "\n" + outputs]
    result = subprocess.run(
        command,
        env={
            **os.environ, **REAL_PROFILE,
            "TBX_AGENT_PROJECT_ROOT": str(PROJECT_ROOT), "PROJECT_ROOT": str(PROJECT_ROOT),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return dict(zip(REAL_PROFILE, result.stdout.splitlines(), strict=True))


@pytest.mark.parametrize("launcher", ["run_local.ps1", "run_local.sh"])
def test_demo_overrides_real_profile_and_skips_model_warmup(tmp_path, monkeypatch, launcher):
    for name, value in _demo_environment(launcher).items():
        monkeypatch.setenv(name, value)
    settings = replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )
    assert settings.vision_backend == "mock"
    assert settings.narrator_backend == "none"
    assert settings.anatomy_backend == "none"
    assert settings.contour_refinement_backend == "none"
    assert settings.require_real_inference is False
    assert settings.require_llm_inference is False
    assert settings.anatomy_required is False
    assert settings.openai_enabled is False

    def unexpected_model_probe(*args, **kwargs):
        pytest.fail("demo lifespan must not warm up any model runtime")

    monkeypatch.setattr("tbx_agent.api.main.build_capability_snapshot", unexpected_model_probe)
    service = TBXAgentService(settings)
    assert service.anatomy is None
    assert service.refinement is None
    with TestClient(create_app(service)) as client:
        assert client.get("/livez").status_code == 200
