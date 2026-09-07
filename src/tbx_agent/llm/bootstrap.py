"""Auditable Qwen3.5/llama.cpp acquisition without repository-local binaries.

The public source tree contains only pins and orchestration. Large inputs,
prebuilt GGUF files, native binaries, receipts, and API keys are written below
the operator-selected artifact/runtime roots.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

from ..artifacts import default_artifact_root
from ..artifacts.manager import sha256_file
from ..paths import default_runtime_root


class QwenBootstrapError(RuntimeError):
    """A pinned input or atomic installation step failed closed."""


@dataclass(frozen=True, slots=True)
class PinnedFile:
    name: str
    size_bytes: int
    sha256: str
    url: str | None = None


@dataclass(frozen=True, slots=True)
class QwenBootstrapPaths:
    artifact_root: Path
    runtime_root: Path

    @property
    def model_path(self) -> Path:
        return self.artifact_root / "llm/qwen3.5-4b-text-no-mtp-q4_k_m.gguf"

    @property
    def source_root(self) -> Path:
        return self.runtime_root / "sources/Qwen3.5-4B-851bf6e8"

    @property
    def llama_source_root(self) -> Path:
        return self.runtime_root / "sources/llama.cpp-b10517"

    @property
    def windows_bundle_root(self) -> Path:
        return self.runtime_root / "llama.cpp/b10517/bin"

    @property
    def bundle_manifest(self) -> Path:
        return self.runtime_root / "provenance/llama-b10517-bundle.json"

    @property
    def runtime_identity(self) -> Path:
        return self.runtime_root / "provenance/llama-b10517-runtime.json"

    @property
    def runtime_config(self) -> Path:
        return self.runtime_root / "config/llm_runtime.generated.yaml"

    @property
    def environment_file(self) -> Path:
        return self.runtime_root / "config/llm.env"

    @property
    def api_key_file(self) -> Path:
        return self.runtime_root / "secrets/llama-server-api.keys"


def load_source_contract(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise QwenBootstrapError("could not read the Qwen runtime source contract") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise QwenBootstrapError("Qwen runtime source contract schema is unsupported")
    for section in ("qwen", "llama_cpp", "conversion"):
        if not isinstance(payload.get(section), dict):
            raise QwenBootstrapError(f"Qwen runtime source contract is missing {section}")
    return payload


def _validate_pin(value: Mapping[str, Any], *, name: str, url: str | None = None) -> PinnedFile:
    size = value.get("size_bytes")
    digest = value.get("sha256")
    if not isinstance(size, int) or size < 0:
        raise QwenBootstrapError(f"invalid size pin for {name}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise QwenBootstrapError(f"invalid SHA256 pin for {name}")
    resolved_url = url if url is not None else value.get("url")
    if resolved_url is not None and (
        not isinstance(resolved_url, str) or urlsplit(resolved_url).scheme != "https"
    ):
        raise QwenBootstrapError(f"invalid HTTPS source for {name}")
    return PinnedFile(name=name, size_bytes=size, sha256=digest, url=resolved_url)


def _verify_file(path: Path, pin: PinnedFile) -> None:
    if not path.is_file():
        raise QwenBootstrapError(f"required file is missing: {pin.name}")
    if path.stat().st_size != pin.size_bytes:
        raise QwenBootstrapError(f"size mismatch: {pin.name}")
    if sha256_file(path) != pin.sha256:
        raise QwenBootstrapError(f"SHA256 mismatch: {pin.name}")


def _write_atomic(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        with suppress(OSError):
            path.chmod(mode)
    finally:
        temporary.unlink(missing_ok=True)


def _download_pinned(
    pin: PinnedFile,
    destination: Path,
    *,
    retries: int = 5,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download a public pinned file with resume and atomic installation."""

    if pin.url is None:
        raise QwenBootstrapError(f"no download URL is declared for {pin.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_file(destination, pin)
        return destination
    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size > pin.size_bytes:
        partial.unlink()
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "tbx-agent-qwen-bootstrap/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(pin.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                final = urlsplit(response.geturl())
                if final.scheme != "https":
                    raise QwenBootstrapError("official source redirected to an insecure URL")
                status = getattr(response, "status", 200)
                append = bool(offset and status == 206)
                with partial.open("ab" if append else "wb") as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if partial.stat().st_size == pin.size_bytes:
                _verify_file(partial, pin)
                os.replace(partial, destination)
                return destination
            if partial.stat().st_size > pin.size_bytes:
                partial.unlink(missing_ok=True)
                raise QwenBootstrapError(f"download exceeded pinned size: {pin.name}")
        except urllib.error.HTTPError as exc:
            if exc.code < 500 and exc.code not in {408, 416, 425, 429}:
                raise QwenBootstrapError(f"official download returned HTTP {exc.code}") from None
            if exc.code == 416:
                partial.unlink(missing_ok=True)
            if attempt == retries:
                raise QwenBootstrapError("official download exhausted its retries") from None
        except QwenBootstrapError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException):
            if attempt == retries:
                raise QwenBootstrapError("official download exhausted its retries") from None
        time.sleep(min(2 ** (attempt - 1), 8))
    raise QwenBootstrapError("official download did not complete")  # pragma: no cover


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    try:
        with zipfile.ZipFile(archive) as source:
            if len(source.infolist()) > 20_000:
                raise QwenBootstrapError("archive contains too many entries")
            for member in source.infolist():
                name = member.filename.replace("\\", "/")
                relative = PurePosixPath(name)
                if (
                    relative.is_absolute()
                    or not relative.parts
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise QwenBootstrapError("archive contains an unsafe path")
                folded = relative.as_posix().casefold()
                if folded in seen:
                    raise QwenBootstrapError("archive contains a case-insensitive duplicate")
                seen.add(folded)
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise QwenBootstrapError("archive contains a symbolic link")
                target = destination.joinpath(*relative.parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as reader, target.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _inventory(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_symlink() or not path.is_file():
            raise QwenBootstrapError("llama.cpp bundle must contain direct regular files only")
        records.append(
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise QwenBootstrapError("llama.cpp bundle is empty")
    return records


def _source_tree_inventory(directory: Path) -> list[dict[str, object]]:
    """Hash a source tree without following symlinks or accepting special files."""

    records: list[dict[str, object]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink():
            raise QwenBootstrapError("llama.cpp source tree contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise QwenBootstrapError("llama.cpp source tree contains a non-regular file")
        records.append(
            {
                "relative_path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise QwenBootstrapError("llama.cpp source tree is empty")
    return records


def _extracted_llama_source_root(extraction: Path) -> Path:
    roots = [child for child in extraction.iterdir() if child.is_dir()]
    if len(roots) != 1 or not (roots[0] / "convert_hf_to_gguf.py").is_file():
        raise QwenBootstrapError("llama.cpp source archive layout is unsupported")
    return roots[0]


def _bundle_manifest_bytes(payload: Mapping[str, Any], *, windows_receipt: bool) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if windows_receipt:
        encoded = encoded.replace("\n", "\r\n")
    return encoded.encode("utf-8")


def _save_runtime_identity(paths: QwenBootstrapPaths, identity: Mapping[str, str]) -> None:
    allowed = {
        "binary_path",
        "binary_sha256",
        "bundle_manifest_path",
        "bundle_manifest_sha256",
        "release_archive_sha256",
        "runtime_id",
    }
    if set(identity) != allowed or not all(isinstance(value, str) for value in identity.values()):
        raise QwenBootstrapError("runtime identity receipt is incomplete")
    _write_atomic(
        paths.runtime_identity,
        (json.dumps(dict(identity), indent=2, sort_keys=True) + "\n").encode(),
        mode=0o600,
    )


def load_runtime_identity(paths: QwenBootstrapPaths) -> dict[str, str]:
    try:
        payload = json.loads(paths.runtime_identity.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenBootstrapError("llama.cpp runtime identity has not been registered") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        raise QwenBootstrapError("llama.cpp runtime identity receipt is invalid")
    identity = dict(payload)
    allowed = {
        "binary_path",
        "binary_sha256",
        "bundle_manifest_path",
        "bundle_manifest_sha256",
        "release_archive_sha256",
        "runtime_id",
    }
    if set(identity) != allowed:
        raise QwenBootstrapError("llama.cpp runtime identity receipt is incomplete")
    return identity


def install_windows_cuda_runtime(
    contract: Mapping[str, Any],
    paths: QwenBootstrapPaths,
    *,
    downloader: Callable[[PinnedFile, Path], Path] = _download_pinned,
) -> dict[str, str]:
    """Install the one tested official Windows CUDA bundle atomically."""

    platform = contract["llama_cpp"]["platforms"]["windows_cuda_13_3_x64"]
    archives_raw = platform.get("archives")
    if not isinstance(archives_raw, list) or len(archives_raw) != 2:
        raise QwenBootstrapError("Windows llama.cpp archive contract is invalid")
    download_root = paths.runtime_root / "downloads/llama.cpp/b10517"
    archives: list[tuple[PinnedFile, Path]] = []
    for record in archives_raw:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise QwenBootstrapError("Windows llama.cpp archive record is invalid")
        pin = _validate_pin(record, name=record["file"])
        archives.append((pin, downloader(pin, download_root / pin.name)))

    target = paths.windows_bundle_root
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.install")
    if target.exists():
        binary = target / platform["binary"]
        _verify_file(
            binary,
            PinnedFile(
                name=platform["binary"],
                size_bytes=binary.stat().st_size if binary.is_file() else 0,
                sha256=platform["binary_sha256"],
            ),
        )
    else:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir()
        try:
            for _, archive in archives:
                extracted = temporary.parent / f".{archive.stem}.{uuid.uuid4().hex}.extract"
                _safe_extract_zip(archive, extracted)
                try:
                    for child in extracted.iterdir():
                        if child.is_dir() or child.is_symlink():
                            raise QwenBootstrapError(
                                "Windows release archive layout is unsupported"
                            )
                        destination = temporary / child.name
                        if destination.exists():
                            raise QwenBootstrapError("Windows release archives overlap")
                        os.replace(child, destination)
                finally:
                    shutil.rmtree(extracted, ignore_errors=True)
            binary = temporary / platform["binary"]
            _verify_file(
                binary,
                PinnedFile(
                    name=platform["binary"],
                    size_bytes=binary.stat().st_size if binary.is_file() else 0,
                    sha256=platform["binary_sha256"],
                ),
            )
            os.replace(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    records = _inventory(target)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "runtime_id": "llama.cpp-b10517-win-cuda-13.3-x64",
        "build": 10517,
        "commit": contract["llama_cpp"]["commit"],
        "release_archive_sha256": archives[0][0].sha256,
        "cudart_archive_sha256": archives[1][0].sha256,
        "source_archive_sha256": contract["llama_cpp"]["source_archive"]["sha256"],
        "files": records,
    }
    manifest_bytes = _bundle_manifest_bytes(payload, windows_receipt=True)
    expected_manifest = platform["bundle_manifest_sha256"]
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest:
        raise QwenBootstrapError("installed Windows bundle inventory does not match its pin")
    _write_atomic(paths.bundle_manifest, manifest_bytes, mode=0o644)
    identity = {
        "binary_path": str((target / platform["binary"]).resolve()),
        "binary_sha256": platform["binary_sha256"],
        "bundle_manifest_path": str(paths.bundle_manifest.resolve()),
        "bundle_manifest_sha256": expected_manifest,
        "release_archive_sha256": archives[0][0].sha256,
        "runtime_id": payload["runtime_id"],
    }
    _save_runtime_identity(paths, identity)
    return identity


def register_runtime_bundle(
    contract: Mapping[str, Any],
    paths: QwenBootstrapPaths,
    *,
    bundle_dir: Path,
    binary_name: str,
    expected_binary_sha256: str,
    release_archive: Path,
    expected_release_sha256: str,
    platform_id: str,
) -> dict[str, str]:
    """Register a user-built b10517 bundle with independent digest assertions."""

    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,63}", platform_id) is None:
        raise QwenBootstrapError("platform id is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", expected_binary_sha256) is None or re.fullmatch(
        r"[0-9a-f]{64}", expected_release_sha256
    ) is None:
        raise QwenBootstrapError("runtime registration requires lowercase SHA256 pins")
    bundle_dir = bundle_dir.expanduser().resolve(strict=True)
    binary = (bundle_dir / binary_name).resolve(strict=True)
    if binary.parent != bundle_dir or not binary.is_file():
        raise QwenBootstrapError("registered binary must be directly inside the bundle")
    if os.name != "nt" and not os.access(binary, os.X_OK):
        raise QwenBootstrapError("registered llama-server is not executable on this platform")
    if sha256_file(binary) != expected_binary_sha256:
        raise QwenBootstrapError("registered llama-server does not match its asserted SHA256")
    release_archive = release_archive.expanduser().resolve(strict=True)
    if sha256_file(release_archive) != expected_release_sha256:
        raise QwenBootstrapError("registered release archive does not match its asserted SHA256")
    payload = {
        "schema_version": 1,
        "runtime_id": f"llama.cpp-b10517-{platform_id}",
        "build": 10517,
        "commit": contract["llama_cpp"]["commit"],
        "release_archive_sha256": expected_release_sha256,
        "files": _inventory(bundle_dir),
    }
    manifest_bytes = _bundle_manifest_bytes(payload, windows_receipt=False)
    _write_atomic(paths.bundle_manifest, manifest_bytes, mode=0o644)
    identity = {
        "binary_path": str(binary),
        "binary_sha256": expected_binary_sha256,
        "bundle_manifest_path": str(paths.bundle_manifest.resolve()),
        "bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "release_archive_sha256": expected_release_sha256,
        "runtime_id": payload["runtime_id"],
    }
    _save_runtime_identity(paths, identity)
    return identity


def register_model(
    contract: Mapping[str, Any], paths: QwenBootstrapPaths, source: Path
) -> Path:
    """Atomically copy only the exact, previously audited Q4 derivative."""

    conversion = contract["conversion"]
    pin = _validate_pin(
        {
            "size_bytes": conversion["expected_size_bytes"],
            "sha256": conversion["expected_sha256"],
        },
        name=conversion["output_name"],
    )
    source = source.expanduser().resolve(strict=True)
    _verify_file(source, pin)
    destination = paths.model_path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_file(destination, pin)
        return destination
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.copy")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        _verify_file(temporary, pin)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def install_llama_source(
    contract: Mapping[str, Any],
    paths: QwenBootstrapPaths,
    *,
    downloader: Callable[[PinnedFile, Path], Path] = _download_pinned,
) -> Path:
    source = contract["llama_cpp"]["source_archive"]
    pin = _validate_pin(source, name=source["file"])
    archive = downloader(
        pin, paths.runtime_root / "downloads/llama.cpp/b10517" / source["file"]
    )
    target = paths.llama_source_root
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise QwenBootstrapError("existing llama.cpp source path is not a regular directory")
        verification = target.with_name(f".{target.name}.{uuid.uuid4().hex}.verify")
        try:
            _safe_extract_zip(archive, verification)
            expected_root = _extracted_llama_source_root(verification)
            if _source_tree_inventory(target) != _source_tree_inventory(expected_root):
                raise QwenBootstrapError(
                    "existing llama.cpp source directory differs from the pinned archive"
                )
        finally:
            shutil.rmtree(verification, ignore_errors=True)
        return target
    extraction = target.with_name(f".{target.name}.{uuid.uuid4().hex}.extract")
    try:
        _safe_extract_zip(archive, extraction)
        source_root = _extracted_llama_source_root(extraction)
        _source_tree_inventory(source_root)
        os.replace(source_root, target)
    finally:
        shutil.rmtree(extraction, ignore_errors=True)
    return target


def ensure_api_key(path: Path) -> Path:
    if path.exists():
        values = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not values or any(len(value) < 32 for value in values):
            raise QwenBootstrapError("existing llama.cpp API key file is invalid")
        return path
    value = secrets.token_urlsafe(48)
    _write_atomic(path, (value + "\n").encode("utf-8"), mode=0o600)
    return path


def configure_runtime(
    contract: Mapping[str, Any],
    paths: QwenBootstrapPaths,
    *,
    runtime_identity: Mapping[str, str],
    template_path: Path,
) -> Path:
    """Write the generated runtime contract and a secret-free env pointer file."""

    model = paths.model_path.resolve(strict=True)
    expected_model = contract["conversion"]["expected_sha256"]
    if sha256_file(model) != expected_model:
        raise QwenBootstrapError("configured Q4 model does not match its pin")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict) or template.get("schema_version") != 1:
        raise QwenBootstrapError("llama.cpp runtime template is invalid")
    template.update(
        {
            "runtime_id": runtime_identity["runtime_id"],
            "release_archive_sha256": runtime_identity["release_archive_sha256"],
            "binary_path": runtime_identity["binary_path"],
            "binary_sha256": runtime_identity["binary_sha256"],
            "bundle_manifest_path": runtime_identity["bundle_manifest_path"],
            "bundle_manifest_sha256": runtime_identity["bundle_manifest_sha256"],
            "api_key_file": str(ensure_api_key(paths.api_key_file).resolve()),
            "model_path": str(model),
            "model_sha256": expected_model,
        }
    )
    encoded = yaml.safe_dump(template, allow_unicode=True, sort_keys=False).encode("utf-8")
    _write_atomic(paths.runtime_config, encoded, mode=0o600)
    env_values = {
        "TBX_AGENT_LLM_RUNTIME_CONFIG": str(paths.runtime_config.resolve()),
        "LLAMA_CPP_BINARY_PATH": runtime_identity["binary_path"],
        "LLAMA_CPP_BUNDLE_MANIFEST_PATH": runtime_identity["bundle_manifest_path"],
        "LLAMA_CPP_API_KEY_FILE": str(paths.api_key_file.resolve()),
        "LLAMA_CPP_MODEL_PATH": str(model),
        "LLAMA_CPP_MODEL_SHA256": expected_model,
    }
    if any("\n" in value or "\r" in value for value in env_values.values()):
        raise QwenBootstrapError("runtime path contains an unsupported newline")
    env_bytes = "".join(f"{name}={value}\n" for name, value in env_values.items()).encode()
    _write_atomic(paths.environment_file, env_bytes, mode=0o600)
    return paths.runtime_config


def bootstrap_paths(
    *,
    artifact_root: Path | None = None,
    runtime_root: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> QwenBootstrapPaths:
    environment = os.environ if environment is None else environment
    return QwenBootstrapPaths(
        artifact_root=Path(artifact_root or default_artifact_root(environment)).expanduser(),
        runtime_root=Path(runtime_root or default_runtime_root(environment)).expanduser(),
    )


def verify_prepared_runtime(config_path: Path) -> dict[str, Any]:
    from .runtime_supervisor import load_runtime_config, verify_runtime_assets

    return verify_runtime_assets(load_runtime_config(config_path))


def public_status(paths: QwenBootstrapPaths) -> dict[str, object]:
    """Return paths and booleans only; never disclose the API key."""

    return {
        "artifact_root": str(paths.artifact_root.resolve(strict=False)),
        "runtime_root": str(paths.runtime_root.resolve(strict=False)),
        "model_present": paths.model_path.is_file(),
        "runtime_config_present": paths.runtime_config.is_file(),
        "environment_file_present": paths.environment_file.is_file(),
        "api_key_file_present": paths.api_key_file.is_file(),
    }


def run_command(
    argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run
) -> None:
    """Small shell-free seam used by tests and future service managers."""

    runner(list(argv), check=True, text=True)
