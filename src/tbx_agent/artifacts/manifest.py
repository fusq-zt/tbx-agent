"""Strict schema for the public model-source manifest.

The manifest deliberately contains no credentials. Private repositories are
selected through named environment variables whose *values* are never copied
into status messages or receipts.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import yaml

_ARTIFACT_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """The artifact manifest is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class LicenseMetadata:
    identifier: str
    url: str | None
    redistributable: bool
    notice: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """One immutable artifact source.

    ``https`` sources contain a complete URL. ``huggingface`` sources are
    expanded to ``<endpoint>/<repo>/resolve/<revision>/<file>``. ``file`` is
    retained for offline mirrors and unit tests and is never used in the
    repository's public manifest.
    """

    kind: str
    url: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    file: str | None = None
    endpoint: str = "https://huggingface.co"
    token_env: str | None = None

    def download_url(self) -> str:
        if self.kind in {"https", "file"}:
            if self.url is None:  # pragma: no cover - guarded by parser
                raise ManifestError(f"{self.kind} source has no URL")
            return self.url
        if self.kind != "huggingface":  # pragma: no cover - guarded by parser
            raise ManifestError(f"unsupported source kind: {self.kind}")
        if not self.repo_id or not self.revision or not self.file:
            raise ManifestError("Hugging Face source is incomplete")
        repository = "/".join(quote(part, safe="") for part in self.repo_id.split("/"))
        revision = quote(self.revision, safe="")
        file_path = "/".join(quote(part, safe="") for part in self.file.split("/"))
        return f"{self.endpoint.rstrip('/')}/{repository}/resolve/{revision}/{file_path}"

    def public_description(self) -> str:
        if self.kind == "huggingface":
            return f"huggingface:{self.repo_id}@{self.revision}/{self.file}"
        if self.kind == "file":
            return "local-file-mirror"
        assert self.url is not None
        parsed = urlsplit(self.url)
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


@dataclass(frozen=True, slots=True)
class SourceOverride:
    url_env: str | None = None
    hf_repo_env: str | None = None
    hf_revision_env: str | None = None
    hf_file: str | None = None
    hf_file_env: str | None = None
    hf_token_env: str | None = None
    hf_endpoint_env: str | None = None

    @property
    def advertised_env_names(self) -> tuple[str, ...]:
        values = (
            self.url_env,
            self.hf_repo_env,
            self.hf_revision_env,
            self.hf_file_env,
            self.hf_token_env,
            self.hf_endpoint_env,
        )
        return tuple(value for value in values if value)

    @property
    def configuration_hint(self) -> str:
        choices: list[str] = []
        if self.url_env:
            choices.append(self.url_env)
        if self.hf_repo_env:
            hf_required = " + ".join(
                value
                for value in (self.hf_repo_env, self.hf_revision_env, self.hf_file_env)
                if value
            )
            if self.hf_file and self.hf_revision_env:
                hf_required = f"{self.hf_repo_env} + {self.hf_revision_env}"
            if self.hf_token_env:
                hf_required += f" ({self.hf_token_env} when private)"
            choices.append(hf_required)
        return " or ".join(choices)

    def resolve(self, environment: Mapping[str, str]) -> tuple[SourceSpec, ...]:
        resolved: list[SourceSpec] = []
        if self.url_env and environment.get(self.url_env, "").strip():
            url = environment[self.url_env].strip()
            _validate_remote_url(url, f"environment variable {self.url_env}")
            resolved.append(SourceSpec(kind="https", url=url))

        repo = _env_value(environment, self.hf_repo_env)
        revision = _env_value(environment, self.hf_revision_env)
        file_value = _env_value(environment, self.hf_file_env) or self.hf_file
        endpoint = _env_value(environment, self.hf_endpoint_env) or "https://huggingface.co"
        any_hf = any((repo, revision, _env_value(environment, self.hf_file_env)))
        if any_hf:
            missing: list[str] = []
            if not repo:
                missing.append(self.hf_repo_env or "HF repository")
            if not revision:
                missing.append(self.hf_revision_env or "HF revision")
            if not file_value:
                missing.append(self.hf_file_env or "HF file")
            if missing:
                raise ManifestError(
                    "incomplete Hugging Face source override; set " + ", ".join(missing)
                )
            _validate_repo_id(repo)
            _validate_revision(revision)
            _validate_posix_relative(file_value, "Hugging Face file")
            _validate_endpoint(endpoint)
            resolved.append(
                SourceSpec(
                    kind="huggingface",
                    repo_id=repo,
                    revision=revision,
                    file=file_value,
                    endpoint=endpoint.rstrip("/"),
                    token_env=self.hf_token_env,
                )
            )
        return tuple(resolved)


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    artifact_id: str
    destination: PurePosixPath
    sha256: str
    size_bytes: int
    revision: str
    source_file: str
    license: LicenseMetadata
    sources: tuple[SourceSpec, ...]
    source_override: SourceOverride | None
    description: str

    def resolve_sources(
        self, environment: Mapping[str, str] | None = None
    ) -> tuple[SourceSpec, ...]:
        environment = os.environ if environment is None else environment
        overrides = (
            self.source_override.resolve(environment) if self.source_override else ()
        )
        return (*overrides, *self.sources)

    @property
    def source_env_names(self) -> tuple[str, ...]:
        if self.source_override is None:
            return ()
        return self.source_override.advertised_env_names

    @property
    def source_configuration_hint(self) -> str:
        if self.source_override is None:
            return "an authorized source"
        return self.source_override.configuration_hint


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: int
    artifacts: Mapping[str, ArtifactSpec]
    groups: Mapping[str, tuple[str, ...]]
    manifest_path: Path

    def select(self, targets: list[str] | tuple[str, ...]) -> tuple[ArtifactSpec, ...]:
        selected: list[ArtifactSpec] = []
        seen: set[str] = set()
        for target in targets:
            if target in self.artifacts:
                identifiers = (target,)
            elif target in self.groups:
                identifiers = self.groups[target]
            else:
                raise ManifestError(f"unknown artifact or group: {target}")
            for artifact_id in identifiers:
                if artifact_id not in seen:
                    selected.append(self.artifacts[artifact_id])
                    seen.add(artifact_id)
        return tuple(selected)


def load_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ManifestError(f"could not read artifact manifest: {manifest_path.name}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("artifact manifest root must be a mapping")
    if raw.get("schema_version") != 1:
        raise ManifestError("artifact manifest schema_version must be 1")

    raw_artifacts = raw.get("artifacts")
    if not isinstance(raw_artifacts, dict) or not raw_artifacts:
        raise ManifestError("artifact manifest must contain a non-empty artifacts mapping")
    artifacts: dict[str, ArtifactSpec] = {}
    destinations: set[PurePosixPath] = set()
    for artifact_id, value in raw_artifacts.items():
        if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id):
            raise ManifestError(f"invalid artifact id: {artifact_id!r}")
        if not isinstance(value, dict):
            raise ManifestError(f"artifact {artifact_id} must be a mapping")
        spec = _parse_artifact(artifact_id, value)
        if spec.destination in destinations:
            raise ManifestError(f"duplicate artifact destination: {spec.destination}")
        destinations.add(spec.destination)
        artifacts[artifact_id] = spec

    raw_groups = raw.get("groups", {})
    if not isinstance(raw_groups, dict):
        raise ManifestError("groups must be a mapping")
    groups: dict[str, tuple[str, ...]] = {}
    for group_id, members in raw_groups.items():
        if not isinstance(group_id, str) or not _ARTIFACT_ID.fullmatch(group_id):
            raise ManifestError(f"invalid artifact group id: {group_id!r}")
        if not isinstance(members, list) or not members:
            raise ManifestError(f"artifact group {group_id} must be a non-empty list")
        if not all(isinstance(member, str) and member in artifacts for member in members):
            raise ManifestError(f"artifact group {group_id} references an unknown artifact")
        if len(set(members)) != len(members):
            raise ManifestError(f"artifact group {group_id} contains duplicates")
        groups[group_id] = tuple(members)

    return ArtifactManifest(
        schema_version=1,
        artifacts=artifacts,
        groups=groups,
        manifest_path=manifest_path,
    )


def _parse_artifact(artifact_id: str, value: Mapping[str, Any]) -> ArtifactSpec:
    destination_text = _required_string(value, "destination", artifact_id)
    destination = _validate_posix_relative(destination_text, f"{artifact_id}.destination")
    sha256 = _required_string(value, "sha256", artifact_id).lower()
    if not _SHA256.fullmatch(sha256):
        raise ManifestError(f"{artifact_id}.sha256 must be 64 lowercase hexadecimal characters")
    size_bytes = value.get("size_bytes")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ManifestError(f"{artifact_id}.size_bytes must be a non-negative integer")
    revision = _required_string(value, "revision", artifact_id)
    source_file = _required_string(value, "file", artifact_id)
    _validate_posix_relative(source_file, f"{artifact_id}.file")

    raw_license = value.get("license")
    if not isinstance(raw_license, dict):
        raise ManifestError(f"{artifact_id}.license must be a mapping")
    license_identifier = _required_string(raw_license, "id", f"{artifact_id}.license")
    license_url = raw_license.get("url")
    if license_url is not None:
        if not isinstance(license_url, str) or not license_url.strip():
            raise ManifestError(f"{artifact_id}.license.url must be a non-empty URL or null")
        _validate_remote_url(license_url, f"{artifact_id}.license.url", https_only=True)
    redistributable = raw_license.get("redistributable")
    if not isinstance(redistributable, bool):
        raise ManifestError(f"{artifact_id}.license.redistributable must be boolean")
    notice = raw_license.get("notice", "")
    if not isinstance(notice, str):
        raise ManifestError(f"{artifact_id}.license.notice must be text")
    license_metadata = LicenseMetadata(
        identifier=license_identifier,
        url=license_url,
        redistributable=redistributable,
        notice=notice,
    )

    raw_sources = value.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ManifestError(f"{artifact_id}.sources must be a list")
    sources = tuple(_parse_source(item, artifact_id) for item in raw_sources)
    raw_override = value.get("source_override")
    source_override = (
        _parse_source_override(raw_override, artifact_id) if raw_override is not None else None
    )
    if not sources and source_override is None:
        raise ManifestError(
            f"{artifact_id} has no source or environment-driven source_override"
        )
    if not redistributable and sources:
        raise ManifestError(
            f"{artifact_id} is not redistributable and cannot contain a public source"
        )
    description = value.get("description", "")
    if not isinstance(description, str):
        raise ManifestError(f"{artifact_id}.description must be text")
    return ArtifactSpec(
        artifact_id=artifact_id,
        destination=destination,
        sha256=sha256,
        size_bytes=size_bytes,
        revision=revision,
        source_file=source_file,
        license=license_metadata,
        sources=sources,
        source_override=source_override,
        description=description,
    )


def _parse_source(value: Any, artifact_id: str) -> SourceSpec:
    if not isinstance(value, dict):
        raise ManifestError(f"{artifact_id}.sources entries must be mappings")
    kind = value.get("kind")
    if kind == "https":
        url = _required_string(value, "url", f"{artifact_id}.source")
        _validate_remote_url(url, f"{artifact_id}.source.url")
        return SourceSpec(
            kind="https",
            url=url,
            revision=_optional_string(value, "revision", f"{artifact_id}.source"),
            file=_optional_string(value, "file", f"{artifact_id}.source"),
            token_env=_optional_env(value, "token_env", f"{artifact_id}.source"),
        )
    if kind == "huggingface":
        repo_id = _required_string(value, "repo_id", f"{artifact_id}.source")
        revision = _required_string(value, "revision", f"{artifact_id}.source")
        file_value = _required_string(value, "file", f"{artifact_id}.source")
        endpoint = value.get("endpoint", "https://huggingface.co")
        if not isinstance(endpoint, str):
            raise ManifestError(f"{artifact_id}.source.endpoint must be text")
        _validate_repo_id(repo_id)
        _validate_revision(revision)
        _validate_posix_relative(file_value, f"{artifact_id}.source.file")
        _validate_endpoint(endpoint)
        return SourceSpec(
            kind="huggingface",
            repo_id=repo_id,
            revision=revision,
            file=file_value,
            endpoint=endpoint.rstrip("/"),
            token_env=_optional_env(value, "token_env", f"{artifact_id}.source"),
        )
    if kind == "file":
        url = _required_string(value, "url", f"{artifact_id}.source")
        if not url.startswith("file://"):
            raise ManifestError(f"{artifact_id}.source file URL must start with file://")
        return SourceSpec(kind="file", url=url)
    raise ManifestError(f"{artifact_id}.source kind must be https, huggingface, or file")


def _parse_source_override(value: Any, artifact_id: str) -> SourceOverride:
    if not isinstance(value, dict):
        raise ManifestError(f"{artifact_id}.source_override must be a mapping")
    allowed = {
        "url_env",
        "hf_repo_env",
        "hf_revision_env",
        "hf_file",
        "hf_file_env",
        "hf_token_env",
        "hf_endpoint_env",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ManifestError(
            f"{artifact_id}.source_override has unknown keys: {', '.join(sorted(unknown))}"
        )
    kwargs: dict[str, str | None] = {}
    for key in allowed - {"hf_file"}:
        kwargs[key] = _optional_env(value, key, f"{artifact_id}.source_override")
    hf_file = value.get("hf_file")
    if hf_file is not None:
        if not isinstance(hf_file, str) or not hf_file.strip():
            raise ManifestError(f"{artifact_id}.source_override.hf_file must be text")
        _validate_posix_relative(hf_file, f"{artifact_id}.source_override.hf_file")
        kwargs["hf_file"] = hf_file
    override = SourceOverride(**kwargs)
    if not override.url_env and not override.hf_repo_env:
        raise ManifestError(
            f"{artifact_id}.source_override requires url_env or hf_repo_env"
        )
    if override.hf_repo_env and not override.hf_revision_env:
        raise ManifestError(
            f"{artifact_id}.source_override with hf_repo_env requires hf_revision_env"
        )
    if override.hf_repo_env and not (override.hf_file or override.hf_file_env):
        raise ManifestError(
            f"{artifact_id}.source_override with hf_repo_env requires hf_file or hf_file_env"
        )
    return override


def _required_string(value: Mapping[str, Any], key: str, context: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ManifestError(f"{context}.{key} must be non-empty text")
    return result.strip()


def _optional_string(value: Mapping[str, Any], key: str, context: str) -> str | None:
    result = value.get(key)
    if result is None:
        return None
    if not isinstance(result, str) or not result.strip():
        raise ManifestError(f"{context}.{key} must be non-empty text or null")
    return result.strip()


def _optional_env(value: Mapping[str, Any], key: str, context: str) -> str | None:
    result = _optional_string(value, key, context)
    if result is not None and not _ENV_NAME.fullmatch(result):
        raise ManifestError(f"{context}.{key} is not a valid environment variable name")
    return result


def _env_value(environment: Mapping[str, str], name: str | None) -> str | None:
    if not name:
        return None
    value = environment.get(name, "").strip()
    return value or None


def _validate_repo_id(value: str) -> None:
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ManifestError("Hugging Face repo_id must have the form owner/name")


def _validate_revision(value: str) -> None:
    if not value.strip() or any(character in value for character in "\\?#"):
        raise ManifestError("Hugging Face revision is invalid")


def _validate_posix_relative(value: str, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"{context} must be a safe relative POSIX path")
    if "\\" in value or ":" in path.parts[0]:
        raise ManifestError(f"{context} must not contain a local absolute path")
    return path


def _validate_endpoint(value: str) -> None:
    _validate_remote_url(value, "Hugging Face endpoint")
    if urlsplit(value).query:
        raise ManifestError("Hugging Face endpoint must not contain a query string")


def _validate_remote_url(value: str, context: str, *, https_only: bool = False) -> None:
    parsed = urlsplit(value)
    loopback_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not (loopback_http and not https_only):
        raise ManifestError(f"{context} must use HTTPS")
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ManifestError(f"{context} is malformed or contains credentials")
