from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from tbx_agent.artifacts import (
    ArtifactManager,
    DownloadError,
    FileLockTimeoutError,
    ManifestError,
    default_artifact_root,
    load_manifest,
)
from tbx_agent.artifacts.cli import main


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(
    path: Path,
    data: bytes,
    *,
    source: dict[str, object] | None = None,
    source_override: dict[str, object] | None = None,
    sha256: str | None = None,
) -> Path:
    artifact: dict[str, object] = {
        "description": "test artifact",
        "destination": "models/test.bin",
        "revision": "test-revision-1",
        "file": "test.bin",
        "size_bytes": len(data),
        "sha256": sha256 or _sha256(data),
        "license": {
            "id": "Apache-2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.txt",
            "redistributable": True,
            "notice": "test only",
        },
    }
    if source is not None:
        artifact["sources"] = [source]
    if source_override is not None:
        artifact["source_override"] = source_override
    payload = {
        "schema_version": 1,
        "groups": {"test_group": ["test_model"]},
        "artifacts": {"test_model": artifact},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


class _BytesHandler(BaseHTTPRequestHandler):
    data = b""
    requests: list[dict[str, str]] = []

    def do_GET(self) -> None:  # noqa: N802
        range_header = self.headers.get("Range", "")
        type(self).requests.append(
            {
                "path": self.path,
                "range": range_header,
                "authorization": self.headers.get("Authorization", ""),
            }
        )
        start = 0
        if range_header.startswith("bytes="):
            start = int(range_header.removeprefix("bytes=").split("-", 1)[0])
        body = type(self).data[start:]
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        if start:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(type(self).data) - 1}/{len(type(self).data)}"
            )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


@contextmanager
def _local_http(data: bytes) -> Iterator[tuple[str, type[_BytesHandler]]]:
    class Handler(_BytesHandler):
        pass

    Handler.data = data
    Handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_default_artifact_root_is_platform_specific_and_overridable(tmp_path: Path) -> None:
    windows_root = default_artifact_root(
        {"LOCALAPPDATA": str(tmp_path / "LocalAppData")}, platform="win32"
    )
    assert windows_root == tmp_path / "LocalAppData" / "TBX-Agent" / "artifacts"
    assert default_artifact_root(
        {"XDG_CACHE_HOME": "/var/cache/test"}, platform="linux"
    ) == Path("/var/cache/test/tbx-agent/artifacts")
    assert default_artifact_root(
        {"TBX_ARTIFACT_ROOT": str(tmp_path)}, platform="linux"
    ) == tmp_path


def test_public_manifest_has_pinned_anatomy_and_no_training_initializers() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_text = (project_root / "configs" / "model_sources.yaml").read_text(
        encoding="utf-8"
    )
    assert "D:\\" not in manifest_text
    manifest = load_manifest(project_root / "configs" / "model_sources.yaml")
    anatomy = manifest.artifacts["xrv_chestxdet_pspnet"]
    assert anatomy.size_bytes == 272988989
    assert anatomy.sha256 == "019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc"
    assert anatomy.sources[0].download_url().startswith("https://github.com/mlmed/")
    assert "rank03" not in manifest.groups
    assert not any(artifact_id.startswith("rank03_") for artifact_id in manifest.artifacts)
    assert "TBX_RANK03" not in manifest_text
    assert "training_prerequisites" not in manifest.groups
    assert "dfine_l_coco_pretrained" not in manifest.artifacts


def test_https_download_resumes_partial_and_installs_atomically(tmp_path: Path) -> None:
    data = (b"verified-model-bytes-" * 4096) + b"end"
    with _local_http(data) as (endpoint, handler):
        manifest_path = _write_manifest(
            tmp_path / "manifest.yaml",
            data,
            source={"kind": "https", "url": f"{endpoint}/model.bin"},
        )
        manager = ArtifactManager(load_manifest(manifest_path), tmp_path / "cache", retries=2)
        spec = manager.manifest.artifacts["test_model"]
        destination = manager.path_for(spec)
        destination.parent.mkdir(parents=True)
        partial = destination.with_name(destination.name + ".part")
        partial.write_bytes(data[:117])

        result = manager.download(spec)

        assert result.state == "valid"
        assert destination.read_bytes() == data
        assert not partial.exists()
        assert not destination.with_name(destination.name + ".lock").exists()
        assert handler.requests[0]["range"] == "bytes=117-"


def test_huggingface_source_uses_pinned_path_and_bearer_token_without_disclosure(
    tmp_path: Path,
) -> None:
    data = b"private-hf-artifact"
    with _local_http(data) as (endpoint, handler):
        manifest_path = _write_manifest(
            tmp_path / "manifest.yaml",
            data,
            source={
                "kind": "huggingface",
                "repo_id": "owner/private-model",
                "revision": "0123456789abcdef",
                "file": "weights/model.bin",
                "endpoint": endpoint,
                "token_env": "TEST_HF_TOKEN",
            },
        )
        secret = "hf_test_secret_must_not_leak"
        manager = ArtifactManager(
            load_manifest(manifest_path),
            tmp_path / "cache",
            environment={"TEST_HF_TOKEN": secret},
        )
        spec = manager.manifest.artifacts["test_model"]

        assert manager.download(spec).state == "valid"
        assert handler.requests == [
            {
                "path": "/owner/private-model/resolve/0123456789abcdef/weights/model.bin",
                "range": "",
                "authorization": f"Bearer {secret}",
            }
        ]
        description = manager.source_descriptions(spec)[0]
        assert description == (
            "huggingface:owner/private-model@0123456789abcdef/weights/model.bin"
        )
        assert secret not in description


def test_malformed_token_is_rejected_without_echoing_secret(tmp_path: Path) -> None:
    data = b"private-hf-artifact"
    with _local_http(data) as (endpoint, _handler):
        manifest_path = _write_manifest(
            tmp_path / "manifest.yaml",
            data,
            source={
                "kind": "huggingface",
                "repo_id": "owner/private-model",
                "revision": "0123456789abcdef",
                "file": "weights/model.bin",
                "endpoint": endpoint,
                "token_env": "TEST_HF_TOKEN",
            },
        )
        secret = "secret\r\nX-Injected: yes"
        manager = ArtifactManager(
            load_manifest(manifest_path),
            tmp_path / "cache",
            environment={"TEST_HF_TOKEN": secret},
        )

        with pytest.raises(DownloadError) as caught:
            manager.download(manager.manifest.artifacts["test_model"])

        assert secret not in str(caught.value)


def test_hash_mismatch_rejects_partial_and_never_installs(tmp_path: Path) -> None:
    data = b"wrong bytes"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        data,
        source={"kind": "file", "url": source.resolve().as_uri()},
        sha256="0" * 64,
    )
    manager = ArtifactManager(load_manifest(manifest_path), tmp_path / "cache")
    spec = manager.manifest.artifacts["test_model"]
    destination = manager.path_for(spec)

    with pytest.raises(DownloadError, match="SHA256 mismatch"):
        manager.download(spec)

    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()
    assert not destination.with_name(destination.name + ".lock").exists()


def test_user_trained_rank03_is_not_a_downloadable_manifest_target(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest = load_manifest(project_root / "configs" / "model_sources.yaml")
    manager = ArtifactManager(manifest, tmp_path, environment={})

    with pytest.raises(ManifestError, match="unknown artifact or group"):
        manager.select(["rank03"])


def test_environment_hf_override_requires_an_immutable_revision(tmp_path: Path) -> None:
    data = b"model"
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        data,
        source_override={
            "hf_repo_env": "MODEL_REPO",
            "hf_revision_env": "MODEL_REVISION",
            "hf_file": "weights/model.bin",
            "hf_token_env": "HF_TOKEN",
        },
    )
    manager = ArtifactManager(
        load_manifest(manifest_path),
        tmp_path / "cache",
        environment={"MODEL_REPO": "owner/private"},
    )

    with pytest.raises(ManifestError, match="MODEL_REVISION"):
        manager.source_descriptions(manager.manifest.artifacts["test_model"])


def test_environment_hf_override_downloads_from_private_repository(tmp_path: Path) -> None:
    data = b"private override"
    with _local_http(data) as (endpoint, handler):
        manifest_path = _write_manifest(
            tmp_path / "manifest.yaml",
            data,
            source_override={
                "hf_repo_env": "MODEL_REPO",
                "hf_revision_env": "MODEL_REVISION",
                "hf_file": "weights/model.bin",
                "hf_token_env": "HF_TOKEN",
                "hf_endpoint_env": "HF_ENDPOINT",
            },
        )
        manager = ArtifactManager(
            load_manifest(manifest_path),
            tmp_path / "cache",
            environment={
                "MODEL_REPO": "owner/private",
                "MODEL_REVISION": "deadbeef",
                "HF_TOKEN": "private-token",
                "HF_ENDPOINT": endpoint,
            },
        )

        assert manager.download(manager.manifest.artifacts["test_model"]).state == "valid"
        assert handler.requests[0]["path"] == (
            "/owner/private/resolve/deadbeef/weights/model.bin"
        )
        assert handler.requests[0]["authorization"] == "Bearer private-token"


def test_fresh_lock_times_out_without_overwriting_lock_owner(tmp_path: Path) -> None:
    data = b"model"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        data,
        source={"kind": "file", "url": source.resolve().as_uri()},
    )
    manager = ArtifactManager(
        load_manifest(manifest_path),
        tmp_path / "cache",
        lock_timeout_seconds=0.01,
    )
    spec = manager.manifest.artifacts["test_model"]
    destination = manager.path_for(spec)
    destination.parent.mkdir(parents=True)
    lock = destination.with_name(destination.name + ".lock")
    lock.write_text("another owner", encoding="utf-8")

    with pytest.raises(FileLockTimeoutError):
        manager.download(spec)

    assert lock.read_text(encoding="utf-8") == "another owner"


def test_manifest_rejects_remote_plain_http_and_path_traversal(tmp_path: Path) -> None:
    data = b"model"
    insecure = _write_manifest(
        tmp_path / "insecure.yaml",
        data,
        source={"kind": "https", "url": "http://example.com/model.bin"},
    )
    with pytest.raises(ManifestError, match="HTTPS"):
        load_manifest(insecure)

    payload = yaml.safe_load(insecure.read_text(encoding="utf-8"))
    payload["artifacts"]["test_model"]["sources"][0]["url"] = "https://example.com/model.bin"
    payload["artifacts"]["test_model"]["destination"] = "../escaped.bin"
    insecure.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="safe relative POSIX path"):
        load_manifest(insecure)


def test_cli_list_dry_run_download_and_verify_are_explicit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = b"cli artifact"
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    manifest_path = _write_manifest(
        tmp_path / "manifest.yaml",
        data,
        source={"kind": "file", "url": source.resolve().as_uri()},
    )
    cache = tmp_path / "cache"
    prefix = ["--manifest", str(manifest_path), "--cache-dir", str(cache)]

    assert main([*prefix, "list"]) == 0
    assert "test_model" in capsys.readouterr().out
    assert main([*prefix, "dry-run", "test_group"]) == 0
    assert "[missing]" in capsys.readouterr().out
    assert main([*prefix, "download", "test_model"]) == 0
    assert "[valid]" in capsys.readouterr().out
    assert main([*prefix, "verify", "test_group"]) == 0
    assert "[valid]" in capsys.readouterr().out
