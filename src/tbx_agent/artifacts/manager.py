"""Atomic, resumable and hash-pinned model artifact downloads."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .manifest import ArtifactManifest, ArtifactSpec, ManifestError, SourceSpec


class ArtifactError(RuntimeError):
    """Base class for artifact acquisition failures."""


class MissingSourceError(ArtifactError):
    """No authorized source was configured for an artifact."""


class DownloadError(ArtifactError):
    """An artifact could not be downloaded or did not match its pin."""


class FileLockTimeoutError(ArtifactError):
    """Another process held an artifact lock beyond the allowed wait."""


@dataclass(frozen=True, slots=True)
class ArtifactStatus:
    artifact_id: str
    destination: str
    state: str
    expected_size_bytes: int
    observed_size_bytes: int | None
    expected_sha256: str
    observed_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_artifact_root(
    environment: Mapping[str, str] | None = None, *, platform: str | None = None
) -> Path:
    environment = os.environ if environment is None else environment
    override = environment.get("TBX_ARTIFACT_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        local_app_data = environment.get("LOCALAPPDATA", "").strip()
        base = (
            Path(local_app_data).expanduser()
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return base / "TBX-Agent" / "artifacts"
    xdg_cache = environment.get("XDG_CACHE_HOME", "").strip()
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "tbx-agent" / "artifacts"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _ArtifactLock(AbstractContextManager["_ArtifactLock"]):
    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float,
        stale_seconds: float,
        poll_seconds: float = 0.1,
    ) -> None:
        self.path = path
        self.timeout_seconds = timeout_seconds
        self.stale_seconds = stale_seconds
        self.poll_seconds = poll_seconds
        self._owned = False

    def __enter__(self) -> _ArtifactLock:
        started = time.monotonic()
        payload = json.dumps({"pid": os.getpid(), "created_unix": time.time()}).encode()
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._owned = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() - started >= self.timeout_seconds:
                    raise FileLockTimeoutError(
                        "timed out waiting for another artifact download process"
                    ) from None
                time.sleep(self.poll_seconds)

    def __exit__(self, *_: object) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False


class ArtifactManager:
    def __init__(
        self,
        manifest: ArtifactManifest,
        cache_dir: str | Path | None = None,
        *,
        environment: Mapping[str, str] | None = None,
        retries: int = 4,
        timeout_seconds: float = 60.0,
        lock_timeout_seconds: float = 600.0,
        stale_lock_seconds: float = 7200.0,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        self.manifest = manifest
        self.environment = dict(os.environ if environment is None else environment)
        self.cache_dir = Path(cache_dir or default_artifact_root(self.environment)).expanduser()
        self.retries = retries
        self.timeout_seconds = timeout_seconds
        self.lock_timeout_seconds = lock_timeout_seconds
        self.stale_lock_seconds = stale_lock_seconds

    def select(self, targets: list[str] | tuple[str, ...]) -> tuple[ArtifactSpec, ...]:
        return self.manifest.select(targets)

    def path_for(self, spec: ArtifactSpec) -> Path:
        root = self.cache_dir.resolve()
        path = root.joinpath(*spec.destination.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:  # defense in depth after manifest validation
            raise ManifestError(
                f"artifact destination escapes cache root: {spec.artifact_id}"
            ) from exc
        return path

    def verify(self, spec: ArtifactSpec) -> ArtifactStatus:
        path = self.path_for(spec)
        if not path.is_file():
            return ArtifactStatus(
                artifact_id=spec.artifact_id,
                destination=spec.destination.as_posix(),
                state="missing",
                expected_size_bytes=spec.size_bytes,
                observed_size_bytes=None,
                expected_sha256=spec.sha256,
                observed_sha256=None,
            )
        observed_size = path.stat().st_size
        if observed_size != spec.size_bytes:
            return ArtifactStatus(
                artifact_id=spec.artifact_id,
                destination=spec.destination.as_posix(),
                state="size_mismatch",
                expected_size_bytes=spec.size_bytes,
                observed_size_bytes=observed_size,
                expected_sha256=spec.sha256,
                observed_sha256=None,
            )
        observed_sha256 = sha256_file(path)
        state = "valid" if observed_sha256 == spec.sha256 else "hash_mismatch"
        return ArtifactStatus(
            artifact_id=spec.artifact_id,
            destination=spec.destination.as_posix(),
            state=state,
            expected_size_bytes=spec.size_bytes,
            observed_size_bytes=observed_size,
            expected_sha256=spec.sha256,
            observed_sha256=observed_sha256,
        )

    def source_descriptions(self, spec: ArtifactSpec) -> tuple[str, ...]:
        return tuple(
            source.public_description() for source in spec.resolve_sources(self.environment)
        )

    def download(self, spec: ArtifactSpec) -> ArtifactStatus:
        sources = spec.resolve_sources(self.environment)
        if not sources:
            raise MissingSourceError(
                f"{spec.artifact_id} has no authorized source; configure "
                f"{spec.source_configuration_hint}"
            )
        destination = self.path_for(spec)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock_path = destination.with_name(destination.name + ".lock")
        partial_path = destination.with_name(destination.name + ".part")
        with _ArtifactLock(
            lock_path,
            timeout_seconds=self.lock_timeout_seconds,
            stale_seconds=self.stale_lock_seconds,
        ):
            current = self.verify(spec)
            if current.state == "valid":
                return current
            if destination.exists():
                destination.unlink()
            if partial_path.exists() and partial_path.stat().st_size > spec.size_bytes:
                partial_path.unlink()

            failures: list[str] = []
            for source in sources:
                try:
                    self._acquire(source, partial_path, spec.size_bytes)
                    self._validate_partial(partial_path, spec)
                    os.replace(partial_path, destination)
                    result = self.verify(spec)
                    if result.state != "valid":  # pragma: no cover - os/filesystem fault
                        raise DownloadError(
                            f"{spec.artifact_id} failed verification after atomic installation"
                        )
                    return result
                except (DownloadError, OSError) as exc:
                    failures.append(f"{source.kind}: {exc}")
            summary = "; ".join(failures)
            raise DownloadError(f"{spec.artifact_id} download failed ({summary})")

    def _acquire(self, source: SourceSpec, partial_path: Path, expected_size: int) -> None:
        if source.kind == "file":
            self._copy_local(source, partial_path, expected_size)
            return
        self._download_http(source, partial_path, expected_size)

    def _copy_local(self, source: SourceSpec, partial_path: Path, expected_size: int) -> None:
        parsed = urlsplit(source.download_url())
        source_path = Path(unquote(parsed.path))
        if os.name == "nt" and source_path.as_posix().startswith("/"):
            source_path = Path(source_path.as_posix().lstrip("/"))
        if not source_path.is_file():
            raise DownloadError("offline mirror file does not exist")
        if source_path.stat().st_size != expected_size:
            raise DownloadError("offline mirror size does not match the manifest")
        offset = partial_path.stat().st_size if partial_path.exists() else 0
        with source_path.open("rb") as source_handle:
            source_handle.seek(offset)
            with partial_path.open("ab" if offset else "wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
                destination_handle.flush()
                os.fsync(destination_handle.fileno())

    def _download_http(
        self, source: SourceSpec, partial_path: Path, expected_size: int
    ) -> None:
        url = source.download_url()
        safe_source = source.public_description()
        token = self.environment.get(source.token_env, "") if source.token_env else ""
        if token and any(character in token for character in "\r\n"):
            raise DownloadError("authorization token environment variable is malformed")
        for attempt in range(1, self.retries + 1):
            offset = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"User-Agent": "tbx-agent-artifact-bootstrap/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            if token:
                headers["Authorization"] = f"Bearer {token}"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    final_scheme = urlsplit(response.geturl()).scheme
                    final_host = urlsplit(response.geturl()).hostname
                    if final_scheme != "https" and not (
                        final_scheme == "http"
                        and final_host in {"127.0.0.1", "::1", "localhost"}
                    ):
                        raise DownloadError("source redirected to an insecure URL")
                    status = getattr(response, "status", 200)
                    append = bool(offset and status == 206)
                    if offset and status == 200:
                        offset = 0
                    mode = "ab" if append else "wb"
                    with partial_path.open(mode) as handle:
                        while chunk := response.read(1024 * 1024):
                            handle.write(chunk)
                        handle.flush()
                        os.fsync(handle.fileno())
                size = partial_path.stat().st_size
                if size == expected_size:
                    return
                if size > expected_size:
                    partial_path.unlink(missing_ok=True)
                    raise DownloadError("server returned more bytes than declared")
                if attempt == self.retries:
                    raise DownloadError(
                        "incomplete transfer after "
                        f"{attempt} attempts: {size}/{expected_size} bytes"
                    )
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and partial_path.exists():
                    if partial_path.stat().st_size == expected_size:
                        return
                    partial_path.unlink(missing_ok=True)
                if exc.code < 500 and exc.code not in {408, 416, 425, 429}:
                    raise DownloadError(f"HTTP {exc.code} from {safe_source}") from None
                if attempt == self.retries:
                    raise DownloadError(f"HTTP {exc.code} from {safe_source}") from None
            except DownloadError:
                raise
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                if attempt == self.retries:
                    raise DownloadError(
                        f"transport failure from {safe_source}: {type(exc).__name__}"
                    ) from None
            time.sleep(min(2 ** (attempt - 1), 8))

    @staticmethod
    def _validate_partial(partial_path: Path, spec: ArtifactSpec) -> None:
        if not partial_path.is_file():
            raise DownloadError("download did not produce a temporary file")
        observed_size = partial_path.stat().st_size
        if observed_size != spec.size_bytes:
            raise DownloadError(
                f"size mismatch: expected {spec.size_bytes}, received {observed_size}"
            )
        observed_hash = sha256_file(partial_path)
        if observed_hash != spec.sha256:
            partial_path.unlink(missing_ok=True)
            raise DownloadError("SHA256 mismatch; rejected temporary download")
