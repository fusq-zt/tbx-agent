from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from tbx_agent.llm import bootstrap as subject
from tbx_agent.llm import bootstrap_cli


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contract(*, model: bytes = b"q4", bf16: bytes = b"bf16") -> dict:
    return {
        "qwen": {
            "repo_id": "Qwen/test",
            "revision": "a" * 40,
            "files": {},
        },
        "llama_cpp": {
            "project": "ggml-org/llama.cpp",
            "tag": "b10517",
            "commit": "b" * 40,
            "source_archive": {"sha256": "c" * 64},
            "platforms": {},
        },
        "conversion": {
            "output_name": "qwen3.5-4b-text-no-mtp-q4_k_m.gguf",
            "destination": "llm/qwen3.5-4b-text-no-mtp-q4_k_m.gguf",
            "expected_size_bytes": len(model),
            "expected_sha256": _digest(model),
            "bf16_size_bytes": len(bf16),
            "bf16_sha256": _digest(bf16),
            "converter_arguments": ["--outtype", "bf16", "--no-nextn"],
            "quantization": "Q4_K_M",
            "quantization_threads": 2,
        },
    }


def _paths(tmp_path: Path) -> subject.QwenBootstrapPaths:
    return subject.QwenBootstrapPaths(
        artifact_root=tmp_path / "artifacts", runtime_root=tmp_path / "runtime"
    )


def test_linux_source_download_action_accepts_an_external_runtime_root(tmp_path: Path) -> None:
    args = bootstrap_cli._parser().parse_args(
        ["--runtime-root", str(tmp_path / "runtime"), "download-llama-source"]
    )
    assert args.action == "download-llama-source"
    assert args.runtime_root == tmp_path / "runtime"


@pytest.mark.parametrize(
    "argv",
    [
        ["download-source"],
        ["build-model"],
        ["prepare"],
        ["prepare", "--model", "existing.gguf", "--build-from-official"],
        ["prepare", "--model", "existing.gguf", "--keep-bf16"],
    ],
)
def test_inference_cli_rejects_conversion_and_requires_an_existing_model(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        bootstrap_cli._parser().parse_args(argv)
    assert error.value.code == 2


def test_prepare_registers_existing_model_and_configures_registered_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _paths(tmp_path)
    contract = _contract()
    model = tmp_path / "existing.gguf"
    model.write_bytes(b"q4")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binary = bundle / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    archive = tmp_path / "release.tar"
    archive.write_bytes(b"release")
    subject.register_runtime_bundle(
        contract,
        paths,
        bundle_dir=bundle,
        binary_name="llama-server",
        expected_binary_sha256=_digest(b"binary"),
        release_archive=archive,
        expected_release_sha256=_digest(b"release"),
        platform_id="test-platform",
    )
    monkeypatch.setattr(bootstrap_cli, "load_source_contract", lambda _path: contract)
    monkeypatch.setattr(bootstrap_cli.sys, "platform", "linux")

    result = bootstrap_cli.main(
        [
            "--artifact-root", str(paths.artifact_root),
            "--runtime-root", str(paths.runtime_root),
            "prepare", "--model", str(model),
        ]
    )

    assert result == 0
    assert paths.model_path.read_bytes() == b"q4"
    assert subject.verify_prepared_runtime(paths.runtime_config)["model_sha256"] == _digest(b"q4")
    output = json.loads(capsys.readouterr().out)
    assert output["configured"] is True
    assert output["model_verified"] is True


def _zip(path: Path, files: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_checked_in_source_contract_is_strict_and_contains_only_public_pins() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = subject.load_source_contract(root / "configs/qwen_runtime_sources.yaml")

    assert contract["qwen"]["repo_id"] == "Qwen/Qwen3.5-4B"
    assert len(contract["qwen"]["revision"]) == 40
    assert contract["llama_cpp"]["tag"] == "b10517"
    assert contract["conversion"]["quantization"] == "Q4_K_M"
    encoded = json.dumps(contract)
    assert "HF_TOKEN" not in encoded
    assert "api_key" not in encoded.lower()


def test_register_model_rehashes_and_installs_atomically(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source_file = tmp_path / "external.gguf"
    source_file.write_bytes(b"q4")

    installed = subject.register_model(_contract(), paths, source_file)

    assert installed == paths.model_path.resolve()
    assert installed.read_bytes() == b"q4"
    assert not list(installed.parent.glob("*.copy"))
    source_file.write_bytes(b"wrong")
    with pytest.raises(subject.QwenBootstrapError, match="size mismatch"):
        subject.register_model(_contract(), _paths(tmp_path / "other"), source_file)


def test_safe_zip_extraction_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    _zip(archive, {"../escape": b"no"})

    with pytest.raises(subject.QwenBootstrapError, match="unsafe path"):
        subject._safe_extract_zip(archive, tmp_path / "out")

    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "out").exists()


def test_llama_source_reuse_is_reverified_against_the_pinned_archive(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    archive = tmp_path / "llama-source.zip"
    _zip(
        archive,
        {
            "llama.cpp-b10517/convert_hf_to_gguf.py": b"# converter\n",
            "llama.cpp-b10517/CMakeLists.txt": b"# cmake\n",
            "llama.cpp-b10517/src/core.cpp": b"// source\n",
        },
    )
    contract = _contract()
    contract["llama_cpp"]["source_archive"] = {
        "url": "https://example.test/llama-source.zip",
        "file": archive.name,
        "size_bytes": archive.stat().st_size,
        "sha256": subject.sha256_file(archive),
    }

    def downloader(_pin: subject.PinnedFile, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read_bytes())
        return destination

    installed = subject.install_llama_source(contract, paths, downloader=downloader)
    assert subject.install_llama_source(contract, paths, downloader=downloader) == installed

    (installed / "src/core.cpp").write_text("// tampered\n", encoding="utf-8")
    with pytest.raises(subject.QwenBootstrapError, match="differs from the pinned archive"):
        subject.install_llama_source(contract, paths, downloader=downloader)


def test_windows_runtime_install_combines_pinned_archives_and_writes_identity(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    server = b"server"
    runtime_zip = tmp_path / "runtime.zip"
    cudart_zip = tmp_path / "cudart.zip"
    _zip(runtime_zip, {"llama-server.exe": server, "llama-quantize.exe": b"quant"})
    _zip(cudart_zip, {"cudart.dll": b"cuda"})
    archives = [runtime_zip, cudart_zip]
    contract = _contract()
    archive_records = [
        {
            "url": f"https://example.test/{archive.name}",
            "file": archive.name,
            "size_bytes": archive.stat().st_size,
            "sha256": subject.sha256_file(archive),
        }
        for archive in archives
    ]
    records = [
        {
            "relative_path": name,
            "size_bytes": len(data),
            "sha256": _digest(data),
        }
        for name, data in sorted(
            {
                "llama-server.exe": server,
                "llama-quantize.exe": b"quant",
                "cudart.dll": b"cuda",
            }.items(),
            key=lambda item: item[0].casefold(),
        )
    ]
    manifest_payload = {
        "schema_version": 1,
        "runtime_id": "llama.cpp-b10517-win-cuda-13.3-x64",
        "build": 10517,
        "commit": "b" * 40,
        "release_archive_sha256": archive_records[0]["sha256"],
        "cudart_archive_sha256": archive_records[1]["sha256"],
        "source_archive_sha256": "c" * 64,
        "files": records,
    }
    expected_manifest = _digest(
        subject._bundle_manifest_bytes(manifest_payload, windows_receipt=True)
    )
    contract["llama_cpp"]["platforms"] = {
        "windows_cuda_13_3_x64": {
            "archives": archive_records,
            "binary": "llama-server.exe",
            "binary_sha256": _digest(server),
            "bundle_manifest_sha256": expected_manifest,
        }
    }

    def downloader(pin: subject.PinnedFile, destination: Path) -> Path:
        source = next(path for path in archives if path.name == pin.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    identity = subject.install_windows_cuda_runtime(
        contract, paths, downloader=downloader
    )

    assert Path(identity["binary_path"]).read_bytes() == server
    assert identity["bundle_manifest_sha256"] == expected_manifest
    assert subject.load_runtime_identity(paths) == identity


def test_register_runtime_requires_independent_archive_and_binary_pins(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binary = bundle / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    (bundle / "llama-quantize").write_bytes(b"quantize")
    release = tmp_path / "release.tar"
    release.write_bytes(b"receipt")

    identity = subject.register_runtime_bundle(
        _contract(),
        paths,
        bundle_dir=bundle,
        binary_name="llama-server",
        expected_binary_sha256=_digest(b"binary"),
        release_archive=release,
        expected_release_sha256=_digest(b"receipt"),
        platform_id="linux-cpu-x86_64",
    )

    assert identity["runtime_id"].endswith("linux-cpu-x86_64")
    with pytest.raises(subject.QwenBootstrapError, match="asserted SHA256"):
        subject.register_runtime_bundle(
            _contract(),
            paths,
            bundle_dir=bundle,
            binary_name="llama-server",
            expected_binary_sha256="0" * 64,
            release_archive=release,
            expected_release_sha256=_digest(b"receipt"),
            platform_id="linux-cpu-x86_64",
        )


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable-bit contract")
def test_register_runtime_rejects_non_executable_posix_binary(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binary = bundle / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o644)
    release = tmp_path / "release.tar"
    release.write_bytes(b"receipt")

    with pytest.raises(subject.QwenBootstrapError, match="not executable"):
        subject.register_runtime_bundle(
            _contract(),
            paths,
            bundle_dir=bundle,
            binary_name="llama-server",
            expected_binary_sha256=_digest(b"binary"),
            release_archive=release,
            expected_release_sha256=_digest(b"receipt"),
            platform_id="linux-cpu-x86_64",
        )


def test_configure_writes_secret_free_env_pointer_and_verifies_assets(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.model_path.parent.mkdir(parents=True)
    paths.model_path.write_bytes(b"q4")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    binary = bundle / "llama-server"
    binary.write_bytes(b"binary")
    binary.chmod(0o755)
    release = tmp_path / "release.tar"
    release.write_bytes(b"release")
    identity = subject.register_runtime_bundle(
        _contract(),
        paths,
        bundle_dir=bundle,
        binary_name="llama-server",
        expected_binary_sha256=_digest(b"binary"),
        release_archive=release,
        expected_release_sha256=_digest(b"release"),
        platform_id="test-platform",
    )
    template = Path(__file__).resolve().parents[1] / "configs/llm_runtime.yaml"

    config_path = subject.configure_runtime(
        _contract(), paths, runtime_identity=identity, template_path=template
    )

    env_text = paths.environment_file.read_text()
    key = paths.api_key_file.read_text().strip()
    assert len(key) >= 32
    assert key not in env_text
    assert "LLAMA_CPP_API_KEY_FILE=" in env_text
    attestation = subject.verify_prepared_runtime(config_path)
    assert attestation["model_sha256"] == _digest(b"q4")
    if os.name != "nt":
        assert stat_mode(paths.api_key_file) == 0o600


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
