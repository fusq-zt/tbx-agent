from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tbx_agent.llm import medgemma_setup as subject


def _identity(tmp_path: Path) -> dict[str, str]:
    return {
        "runtime_id": "llamacpp-b10517-test",
        "release_archive_sha256": "1" * 64,
        "binary_path": str(tmp_path / "llama-server"),
        "binary_sha256": "2" * 64,
        "bundle_manifest_path": str(tmp_path / "bundle.json"),
        "bundle_manifest_sha256": "3" * 64,
    }


def _paths(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        api_key_file=tmp_path / "secrets/llama-server-api.keys",
        runtime_config=tmp_path / "config/llm_runtime.generated.yaml",
        environment_file=tmp_path / "config/llm.env",
    )


def test_configure_medgemma_verifies_in_place_and_writes_secret_free_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / subject.MODEL_FILENAME
    model.write_bytes(b"synthetic-medgemma-q4")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()
    paths = _paths(tmp_path / "runtime")
    monkeypatch.setattr(
        subject,
        "_runtime_identity",
        lambda **_kwargs: (paths, _identity(tmp_path)),
    )

    result = subject.configure_medgemma(
        model_path=model,
        runtime_root=tmp_path / "runtime",
        expected_sha256=expected,
        expected_size_bytes=model.stat().st_size,
        install_windows_runtime=False,
    )

    runtime = yaml.safe_load(paths.runtime_config.read_text(encoding="utf-8"))
    assert runtime["model_alias"] == subject.MODEL_ALIAS
    assert runtime["model_path"] == str(model.resolve())
    assert runtime["model_sha256"] == expected
    assert runtime["quantization"] == "Q4_K_M"
    assert runtime["load_mmproj"] is False
    env_text = paths.environment_file.read_text(encoding="utf-8")
    assert f"LLAMA_CPP_MODEL_PATH={model.resolve()}" in env_text
    assert f"LLAMA_CPP_MODEL_SHA256={expected}" in env_text
    assert "LLAMA_CPP_API_KEY=" not in env_text
    assert result["mmproj_loaded"] == "false"


def test_configure_medgemma_rejects_wrong_artifact_before_runtime_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / subject.MODEL_FILENAME
    model.write_bytes(b"wrong")
    runtime_called = False

    def unexpected_runtime(**_kwargs):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime setup must not run")

    monkeypatch.setattr(subject, "_runtime_identity", unexpected_runtime)
    with pytest.raises(subject.MedGemmaSetupError, match="SHA256 mismatch"):
        subject.configure_medgemma(
            model_path=model,
            runtime_root=tmp_path / "runtime",
            expected_sha256="0" * 64,
            expected_size_bytes=model.stat().st_size,
        )
    assert runtime_called is False


def test_configure_medgemma_rejects_wrong_size_before_hashing(tmp_path: Path) -> None:
    model = tmp_path / subject.MODEL_FILENAME
    model.write_bytes(b"too-small")
    with pytest.raises(subject.MedGemmaSetupError, match="size mismatch"):
        subject.configure_medgemma(
            model_path=model,
            runtime_root=tmp_path / "runtime",
            expected_size_bytes=model.stat().st_size + 1,
        )


def test_download_medgemma_uses_pinned_repo_revision_and_external_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    calls: list[dict[str, object]] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        destination = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"downloaded")
        return str(destination)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    downloaded = subject.download_medgemma(
        runtime_root=tmp_path / "runtime",
        token="test-token",
    )

    assert downloaded == (tmp_path / "runtime/models" / subject.MODEL_FILENAME).resolve()
    assert calls == [
        {
            "repo_id": subject.MODEL_REPO_ID,
            "filename": subject.MODEL_FILENAME,
            "revision": subject.MODEL_REVISION,
            "local_dir": (tmp_path / "runtime/models").resolve(),
            "token": "test-token",
        }
    ]
