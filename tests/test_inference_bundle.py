from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tbx_agent.artifacts.rank03_bundle import (
    BUNDLE_FILES,
    MAX_WEIGHT_BYTES,
    BundleError,
    download_and_install_rank03_bundle,
    install_rank03_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _encoded(value):
    return json.dumps(value, ensure_ascii=False).encode()


def _bundle(tmp_path: Path, *, mutate=None) -> Path:
    source = tmp_path / "bundle-input"
    source.mkdir()
    classifier = {"model": "convnext_tiny.fb_in1k", "input_size": 320, "amp": True}
    detector = {
        "model": "DFINE", "postprocessor": "DFINEPostProcessor", "num_classes": 1,
        "remap_mscoco_category": False, "eval_spatial_size": [512, 512], "use_focal_loss": True,
        "DFINE": {"backbone": "HGNetv2", "encoder": "HybridEncoder", "decoder": "DFINETransformer"},
        "DFINEPostProcessor": {"num_top_queries": 300},
        "HGNetv2": {"name": "B4", "pretrained": False, "return_idx": [1, 2, 3]},
        "HybridEncoder": {
            "hidden_dim": 256, "in_channels": [512, 1024, 2048], "feat_strides": [8, 16, 32],
            "nhead": 8, "num_encoder_layers": 1,
        },
        "DFINETransformer": {
            "hidden_dim": 256, "num_layers": 6, "num_levels": 3, "num_queries": 300,
            "feat_channels": [256, 256, 256], "feat_strides": [8, 16, 32], "reg_max": 32,
        },
    }
    runtime = {
        "schema_version": 1, "template": True, "model_bundle_id": "replaced-on-install",
        "source_revision": "synthetic-only",
        "classifier": {
            "architecture": "convnext_tiny.fb_in1k", "input_size": 320, "amp": True,
            "probability_order": ["healthy", "sick_non_tb", "tb"],
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "detector": {
            "architecture": "D-FINE-L HGNetv2-B4", "input_size": 512, "native_label": 0,
            "native_top_queries": 300, "export_floor": 0.05,
            "bbox_format": "xyxy_original_image_pixels", "tta": "none", "precision": "fp32",
        },
        "validation_scope": {"clinical_validation": False, "performance_claims_inherited": False},
    }
    payload = {
        # Deliberately opaque bytes: installation must never deserialize these.
        "classifier.pt": b"synthetic opaque classifier bytes",
        "detector.pt": b"synthetic opaque detector bytes",
        "classifier_config.json": _encoded(classifier),
        "detector_config.json": _encoded(detector),
    }
    manifest = {
        "schema_version": 1, "bundle_type": "rank03-inference-v1", "bundle_id": "synthetic-v1",
        "files": {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
            for name, raw in payload.items()
        },
        "runtime": runtime,
    }
    if mutate:
        mutate(manifest, payload)
    for name, raw in payload.items():
        (source / name).write_bytes(raw)
    (source / "manifest.json").write_bytes(_encoded(manifest))
    return source


def _zip(source: Path, *, compression=zipfile.ZIP_STORED) -> Path:
    archive = source.parent / "bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=compression) as output:
        for child in source.iterdir():
            output.write(child, arcname=child.name)
    return archive


def _install(source: Path, tmp_path: Path, **kwargs):
    return install_rank03_bundle(
        source, artifact_root=tmp_path / "artifacts", runtime_config=tmp_path / "runtime.json",
        **kwargs,
    )


@pytest.mark.parametrize("zipped", [False, True])
def test_bundle_installs_only_inference_files_and_binds_actual_hashes(tmp_path: Path, zipped: bool):
    def malicious_paths(manifest, _payload):
        manifest["runtime"]["classifier"]["checkpoint_path"] = "C:/untrusted/model.pt"
        manifest["runtime"]["detector"]["source_root"] = "C:/untrusted/code"

    source = _bundle(tmp_path, mutate=malicious_paths)
    source = _zip(source) if zipped else source
    result = _install(source, tmp_path)
    installed = Path(result["installed_directory"])
    assert {item.name for item in installed.iterdir()} == BUNDLE_FILES
    assert result["weights_deserialized"] is False
    assert result["environment"]["TBX_ARTIFACT_ROOT"] == str(tmp_path / "artifacts")
    runtime = json.loads(Path(result["runtime_config"]).read_text())
    assert runtime["template"] is False
    assert runtime["model_bundle_id"] == "synthetic-v1"
    assert runtime["classifier"]["checkpoint_path"] == (
        "artifact://rank03/bundles/synthetic-v1/classifier.pt"
    )
    assert runtime["classifier"]["checkpoint_sha256"] == hashlib.sha256(
        (installed / "classifier.pt").read_bytes()
    ).hexdigest()
    assert runtime["detector"]["source_root"] == "artifact://sources/D-FINE"
    assert runtime["detector"]["resolved_config_path"].endswith("/detector_config.json")


def test_zip_pin_and_per_file_hash_are_both_checked_before_publication(tmp_path: Path):
    source = _bundle(tmp_path)
    archive = _zip(source)
    with pytest.raises(BundleError, match="archive SHA-256 mismatch"):
        _install(archive, tmp_path, sha256="0" * 64)
    assert not (tmp_path / "runtime.json").exists()
    with zipfile.ZipFile(archive, "w") as output:
        for child in source.iterdir():
            raw = b"modified weight" if child.name == "classifier.pt" else child.read_bytes()
            output.writestr(child.name, raw)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(BundleError, match="size does not match"):
        _install(archive, tmp_path, sha256=digest)
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "artifacts" / "rank03").exists()


def test_local_zip_trusted_hash_success(tmp_path: Path):
    archive = _zip(_bundle(tmp_path))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    result = _install(archive, tmp_path, sha256=digest)
    assert result["archive_pin_verified"] is True
    assert result["bundle_sha256"] == digest


def test_verified_zip_snapshot_survives_input_replacement(tmp_path: Path, monkeypatch):
    from tbx_agent.artifacts import rank03_bundle

    source = _bundle(tmp_path)
    archive = _zip(source)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    snapshot_archive = rank03_bundle._snapshot_archive

    def replace_after_verification(path, snapshot, expected):
        result = snapshot_archive(path, snapshot, expected)
        path.write_bytes(b"input replaced after hash verification")
        return result

    monkeypatch.setattr(rank03_bundle, "_snapshot_archive", replace_after_verification)
    result = _install(archive, tmp_path, sha256=digest)
    assert result["bundle_sha256"] == digest
    assert (Path(result["installed_directory"]) / "classifier.pt").read_bytes() == (
        source / "classifier.pt"
    ).read_bytes()


def test_runtime_publication_failure_removes_only_owned_installation(tmp_path: Path, monkeypatch):
    source = _bundle(tmp_path)
    unrelated = tmp_path / "artifacts" / "preserved.txt"
    unrelated.parent.mkdir()
    unrelated.write_text("existing artifact", encoding="utf-8")

    def fail_config(_manifest):
        raise OSError("synthetic runtime publication failure")

    monkeypatch.setattr("tbx_agent.artifacts.rank03_bundle._runtime_config", fail_config)
    with pytest.raises(OSError, match="synthetic runtime publication"):
        _install(source, tmp_path)
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "artifacts" / "rank03" / "bundles" / "synthetic-v1").exists()
    assert unrelated.read_text() == "existing artifact"


def test_directory_hash_mismatch_leaves_no_installation(tmp_path: Path):
    source = _bundle(tmp_path)
    original = (source / "classifier.pt").read_bytes()
    (source / "classifier.pt").write_bytes(b"x" * len(original))
    with pytest.raises(BundleError, match="SHA-256 mismatch"):
        _install(source, tmp_path)
    assert not (tmp_path / "runtime.json").exists()
    assert list((tmp_path / "artifacts").iterdir()) == []


def test_existing_bundle_and_runtime_are_never_overwritten(tmp_path: Path):
    source = _bundle(tmp_path)
    first = _install(source, tmp_path)
    config = (tmp_path / "runtime.json").read_bytes()
    with pytest.raises(BundleError, match="runtime config already exists"):
        _install(source, tmp_path)
    with pytest.raises(BundleError, match="bundle already installed"):
        install_rank03_bundle(
            source, artifact_root=tmp_path / "artifacts", runtime_config=tmp_path / "new.json",
        )
    assert (tmp_path / "runtime.json").read_bytes() == config
    assert not (tmp_path / "new.json").exists()
    assert (Path(first["installed_directory"]) / "classifier.pt").read_bytes() == (
        source / "classifier.pt"
    ).read_bytes()


@pytest.mark.parametrize("name", ["../escape.pt", "/absolute.pt", "C:/absolute.pt", "extra.txt"])
def test_zip_traversal_absolute_and_extra_files_are_rejected(tmp_path: Path, name: str):
    archive = _zip(_bundle(tmp_path))
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(name, b"unauthorized")
    with pytest.raises(BundleError, match="five allowlisted"):
        _install(archive, tmp_path)
    assert not (tmp_path / "runtime.json").exists()
    assert not (tmp_path / "escape.pt").exists()


def test_duplicate_zip_entry_is_rejected(tmp_path: Path):
    archive = _zip(_bundle(tmp_path))
    with zipfile.ZipFile(archive, "a") as output, pytest.warns(UserWarning):
        output.writestr("classifier.pt", b"duplicate")
    with pytest.raises(BundleError, match="five allowlisted"):
        _install(archive, tmp_path)


def test_zip_symlink_is_rejected(tmp_path: Path):
    source = _bundle(tmp_path)
    archive = source.parent / "symlink.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for child in source.iterdir():
            info = zipfile.ZipInfo(child.name)
            if child.name == "classifier.pt":
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(info, child.read_bytes())
    with pytest.raises(BundleError, match="ZIP links"):
        _install(archive, tmp_path)


def test_zip_bomb_ratio_is_rejected_before_decompression(tmp_path: Path):
    def compressible(manifest, payload):
        payload["classifier.pt"] = b"0" * (512 * 1024)
        manifest["files"]["classifier.pt"] = {
            "sha256": hashlib.sha256(payload["classifier.pt"]).hexdigest(),
            "size_bytes": len(payload["classifier.pt"]),
        }

    archive = _zip(_bundle(tmp_path, mutate=compressible), compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(BundleError, match="compression ratio"):
        _install(archive, tmp_path)
    assert not (tmp_path / "runtime.json").exists()


@pytest.mark.parametrize("bundle_id", ["../escape", "C:\\escape", "con", "ends.", "UPPER"])
def test_bundle_id_cannot_escape_or_use_windows_device_paths(tmp_path: Path, bundle_id: str):
    source = _bundle(tmp_path, mutate=lambda manifest, _: manifest.update(bundle_id=bundle_id))
    with pytest.raises(BundleError, match="safe lowercase label"):
        _install(source, tmp_path)


def test_declared_oversize_and_unknown_manifest_fields_are_rejected(tmp_path: Path):
    source = _bundle(tmp_path)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["classifier.pt"]["size_bytes"] = MAX_WEIGHT_BYTES + 1
    path.write_bytes(_encoded(manifest))
    with pytest.raises(BundleError, match="excessive file size"):
        _install(source, tmp_path)
    manifest["training_command"] = "do-not-execute"
    path.write_bytes(_encoded(manifest))
    with pytest.raises(BundleError, match="schema exactly"):
        _install(source, tmp_path)


@pytest.mark.parametrize("scope,key,value", [
    ("classifier", "architecture", "another_model"),
    ("classifier", "input_size", 224),
    ("classifier", "device", "cuda"),
    ("detector", "export_floor", 0.99),
    ("detector", "native_top_queries", 100),
])
def test_runtime_cannot_change_frozen_inference_contract(
    tmp_path: Path, scope: str, key: str, value,
):
    source = _bundle(
        tmp_path, mutate=lambda manifest, _: manifest["runtime"][scope].update({key: value}),
    )
    with pytest.raises(BundleError, match="unsupported"):
        _install(source, tmp_path)


@pytest.mark.parametrize("config_file,mutation", [
    ("classifier_config.json", lambda value: value.update(epochs=100)),
    ("detector_config.json", lambda value: value.update(train_dataloader={})),
    ("detector_config.json", lambda value: value["HGNetv2"].update(pretrained=True)),
    ("detector_config.json", lambda value: value["HGNetv2"].update(local_model_dir="outside")),
])
def test_config_rejects_training_and_model_download_settings(
    tmp_path: Path, config_file: str, mutation,
):
    def altered(manifest, payload):
        value = json.loads(payload[config_file])
        mutation(value)
        payload[config_file] = _encoded(value)
        manifest["files"][config_file] = {
            "sha256": hashlib.sha256(payload[config_file]).hexdigest(),
            "size_bytes": len(payload[config_file]),
        }

    source = _bundle(tmp_path, mutate=altered)
    with pytest.raises(BundleError):
        _install(source, tmp_path)
    assert not (tmp_path / "runtime.json").exists()


def test_source_symlink_is_rejected_when_supported(tmp_path: Path):
    source = _bundle(tmp_path)
    target = tmp_path / "weight.pt"
    (source / "classifier.pt").rename(target)
    try:
        (source / "classifier.pt").symlink_to(target)
    except OSError:
        pytest.skip("filesystem does not permit creating test symlinks")
    with pytest.raises(BundleError, match="symlinks"):
        _install(source, tmp_path)


def test_artifacts_and_runtime_must_be_outside_repository(tmp_path: Path):
    source = _bundle(tmp_path)
    with pytest.raises(BundleError, match="outside the project"):
        install_rank03_bundle(
            source, artifact_root=PROJECT_ROOT / "forbidden",
            runtime_config=tmp_path / "config.json",
        )
    assert not (PROJECT_ROOT / "forbidden").exists()


def test_install_cli_never_imports_torch_or_training(tmp_path: Path):
    source = _bundle(tmp_path)
    code = """
import importlib.abc, runpy, sys
class DenyModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('tbx_agent.training'):
            raise RuntimeError('installer imported a model/training dependency')
sys.meta_path.insert(0, DenyModels())
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(PROJECT_ROOT / "scripts" / "install_vision_bundle.py"),
         "--bundle", str(source), "--artifact-root", str(tmp_path / "artifacts"),
         "--runtime-config", str(tmp_path / "runtime.json")],
        cwd=PROJECT_ROOT, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert json.loads(completed.stdout)["weights_deserialized"] is False


def test_https_download_requires_pin_and_installs_verified_response(tmp_path: Path, monkeypatch):
    archive = _zip(_bundle(tmp_path))
    raw = archive.read_bytes()

    class Response(io.BytesIO):
        def geturl(self):
            return "https://example.invalid/bundle.zip"

    calls = []

    def open_response(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(raw)

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_: SimpleNamespace(open=open_response),
    )
    with pytest.raises(BundleError, match="trusted lowercase"):
        download_and_install_rank03_bundle(
            "https://example.invalid/bundle.zip", sha256="",
            artifact_root=tmp_path / "artifacts", runtime_config=tmp_path / "runtime.json",
        )
    assert calls == []
    result = download_and_install_rank03_bundle(
        "https://example.invalid/bundle.zip", sha256=hashlib.sha256(raw).hexdigest(),
        artifact_root=tmp_path / "artifacts", runtime_config=tmp_path / "runtime.json",
    )
    assert result["archive_pin_verified"] is True
    assert calls == [("https://example.invalid/bundle.zip", 30)]


def test_https_redirect_cannot_downgrade_to_http():
    from tbx_agent.artifacts.rank03_bundle import _HTTPSRedirectHandler

    with pytest.raises(BundleError, match="HTTPS"):
        _HTTPSRedirectHandler().redirect_request(None, None, 302, "", {}, "http://example.invalid")
