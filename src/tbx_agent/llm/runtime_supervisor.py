from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..artifacts import default_artifact_root
from ..paths import resolve_portable_path


class LlamaCppRuntimeConfig(BaseModel):
    """Versioned, auditable launch contract for an externally supervised server."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    runtime_id: str = Field(min_length=1)
    engine: Literal["llama.cpp"]
    server_build: str = Field(min_length=1)
    release_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binary_path: Path
    binary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_manifest_path: Path
    bundle_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    api_key_file: Path
    model_id: str = Field(min_length=1)
    model_alias: str = Field(min_length=1)
    model_path: Path
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quantization: Literal["Q4_K_M"]
    host: Literal["127.0.0.1", "::1"] = "127.0.0.1"
    port: int = Field(default=11435, ge=1024, le=65535)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    parallel_slots: Literal[1] = 1
    batch_tokens: int = Field(default=512, ge=32, le=2048)
    micro_batch_tokens: int = Field(default=128, ge=16, le=512)
    generation_threads: int = Field(default=8, ge=1, le=64)
    batch_threads: int = Field(default=12, ge=1, le=64)
    gpu_layers: Literal["auto"] = "auto"
    fit_target_mib: int = Field(default=4096, ge=1024, le=7168)
    fit_context_tokens: int = Field(default=8192, ge=2048, le=32768)
    cache_ram_mib: int = Field(default=256, ge=0, le=2048)
    kv_cache_type: Literal["q8_0"] = "q8_0"
    flash_attention: bool = True
    continuous_batching: bool = False
    jinja: bool = True
    enable_thinking: bool = False
    reasoning: Literal["off"] = "off"
    metrics: bool = True
    web_ui: bool = False
    expose_slots: Literal[False] = False
    cors_origins: Literal["localhost"] = "localhost"
    request_timeout_seconds: int = Field(default=120, ge=1, le=600)
    max_input_tokens: int = Field(default=4096, ge=256, le=16384)
    max_output_tokens: int = Field(default=512, ge=32, le=2048)
    seed: int = 20260829
    allowed_roles: list[Literal["narrator", "evidence_composer"]]
    clinical_authority: Literal[False] = False
    load_mmproj: Literal[False] = False

    @field_validator("binary_path", "model_path", "api_key_file", "bundle_manifest_path")
    @classmethod
    def absolute_paths_only(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("runtime binary and model paths must be absolute")
        return value

    @field_validator("model_alias")
    @classmethod
    def alias_has_no_whitespace(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("model_alias must not contain whitespace")
        return value

    @model_validator(mode="after")
    def context_bounds(self) -> LlamaCppRuntimeConfig:
        if self.max_input_tokens + self.max_output_tokens > self.context_tokens:
            raise ValueError("input and output token limits exceed the server context")
        if self.fit_context_tokens != self.context_tokens:
            raise ValueError("fit_context_tokens must equal context_tokens")
        return self

    def canonical_sha256(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


_RUNTIME_PATH_ENV = {
    "binary_path": "LLAMA_CPP_BINARY_PATH",
    "bundle_manifest_path": "LLAMA_CPP_BUNDLE_MANIFEST_PATH",
    "api_key_file": "LLAMA_CPP_API_KEY_FILE",
    "model_path": "LLAMA_CPP_MODEL_PATH",
}


def _resolve_runtime_config_path(
    raw: object,
    *,
    base: Path,
    environment: Mapping[str, str],
) -> Path:
    text = str(raw).strip()
    prefix = "artifact://"
    if text.startswith(prefix):
        relative = PurePosixPath(text[len(prefix) :])
        if relative.is_absolute() or not relative.parts or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("invalid artifact path reference in LLM runtime config")
        root = default_artifact_root(environment).resolve(strict=False)
        candidate = root.joinpath(*relative.parts).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError("LLM artifact path reference escapes the cache root")
        return candidate
    return resolve_portable_path(text, base=base, environment=environment)


def load_runtime_config(
    path: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> LlamaCppRuntimeConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LLM runtime configuration must be a mapping")
    environment = os.environ if environment is None else environment
    for field, env_name in _RUNTIME_PATH_ENV.items():
        override = environment.get(env_name, "").strip()
        payload[field] = _resolve_runtime_config_path(
            override or payload.get(field, ""),
            base=path.parent,
            environment=environment,
        )
    return LlamaCppRuntimeConfig.model_validate(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime_assets(config: LlamaCppRuntimeConfig) -> dict[str, Any]:
    if not config.binary_path.is_file():
        raise FileNotFoundError(f"llama-server binary not found: {config.binary_path}")
    if not config.model_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {config.model_path}")
    if not config.bundle_manifest_path.is_file():
        raise FileNotFoundError(
            f"llama.cpp bundle manifest not found: {config.bundle_manifest_path}"
        )
    if not config.api_key_file.is_file():
        raise FileNotFoundError(f"llama-server API key file not found: {config.api_key_file}")
    keys = [
        line.strip()
        for line in config.api_key_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or any(len(key) < 32 for key in keys):
        raise ValueError("llama-server API key file must contain only strong non-empty keys")
    if os.getenv("TBX_AGENT_LOCAL_FAST_START", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return {
            "runtime_id": config.runtime_id,
            "runtime_config_sha256": config.canonical_sha256(),
            "binary_sha256": config.binary_sha256,
            "bundle_manifest_sha256": config.bundle_manifest_sha256,
            "verified_bundle_files": 0,
            "release_archive_sha256": config.release_archive_sha256,
            "model_sha256": config.model_sha256,
            "binary_size_bytes": config.binary_path.stat().st_size,
            "model_size_bytes": config.model_path.stat().st_size,
            "verification_mode": "local_fast_start",
        }
    binary_digest = _sha256_file(config.binary_path)
    if binary_digest != config.binary_sha256:
        raise ValueError("llama-server SHA256 does not match the runtime contract")
    manifest_bytes = config.bundle_manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_digest != config.bundle_manifest_sha256:
        raise ValueError("llama.cpp bundle manifest SHA256 does not match the runtime contract")
    try:
        bundle_manifest = json.loads(manifest_bytes)
        bundle_records = bundle_manifest["files"]
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("llama.cpp bundle manifest is invalid") from exc
    if not isinstance(bundle_records, list) or not bundle_records:
        raise ValueError("llama.cpp bundle manifest contains no files")
    bundle_root = config.binary_path.parent.resolve()
    verified_bundle_files = 0
    for record in bundle_records:
        if not isinstance(record, dict):
            raise ValueError("llama.cpp bundle manifest record is invalid")
        relative_path = record.get("relative_path")
        expected_digest = record.get("sha256")
        expected_size = record.get("size_bytes")
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_digest, str)
            or not isinstance(expected_size, int)
        ):
            raise ValueError("llama.cpp bundle manifest record is incomplete")
        bundle_file = (bundle_root / relative_path).resolve()
        try:
            bundle_file.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError("llama.cpp bundle manifest path escapes the bundle root") from exc
        if not bundle_file.is_file() or bundle_file.stat().st_size != expected_size:
            raise ValueError(f"llama.cpp bundle file size mismatch: {relative_path}")
        if _sha256_file(bundle_file) != expected_digest:
            raise ValueError(f"llama.cpp bundle file digest mismatch: {relative_path}")
        verified_bundle_files += 1
    digest = _sha256_file(config.model_path)
    if digest != config.model_sha256:
        raise ValueError("GGUF model SHA256 does not match the runtime contract")
    return {
        "runtime_id": config.runtime_id,
        "runtime_config_sha256": config.canonical_sha256(),
        "binary_sha256": binary_digest,
        "bundle_manifest_sha256": manifest_digest,
        "verified_bundle_files": verified_bundle_files,
        "release_archive_sha256": config.release_archive_sha256,
        "model_sha256": digest,
        "binary_size_bytes": config.binary_path.stat().st_size,
        "model_size_bytes": config.model_path.stat().st_size,
    }


def build_server_command(config: LlamaCppRuntimeConfig) -> list[str]:
    """Return a shell-free argv list; process lifecycle belongs to the service manager."""

    command = [
        str(config.binary_path),
        "-m",
        str(config.model_path),
        "--alias",
        config.model_alias,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "-c",
        str(config.context_tokens),
        "-np",
        str(config.parallel_slots),
        "-b",
        str(config.batch_tokens),
        "-ub",
        str(config.micro_batch_tokens),
        "-t",
        str(config.generation_threads),
        "-tb",
        str(config.batch_threads),
        "--n-gpu-layers",
        config.gpu_layers,
        "--fit",
        "on",
        "--fit-target",
        str(config.fit_target_mib),
        "--fit-ctx",
        str(config.fit_context_tokens),
        "-ctk",
        config.kv_cache_type,
        "-ctv",
        config.kv_cache_type,
        "--cache-ram",
        str(config.cache_ram_mib),
        "--reasoning",
        config.reasoning,
        "--cors-origins",
        config.cors_origins,
        "--api-key-file",
        str(config.api_key_file),
        "--timeout",
        str(config.request_timeout_seconds),
    ]
    command.extend(["-fa", "on" if config.flash_attention else "off"])
    command.append("-cb" if config.continuous_batching else "-nocb")
    command.append("--no-mmproj")
    command.append("--no-slots")
    if config.jinja:
        command.append("--jinja")
    # Keep one authoritative shell-free reasoning switch. Repeating it through
    # JSON chat-template kwargs is deprecated and its embedded quotes are not
    # preserved by Windows ``os.execv`` argument encoding.
    if config.metrics:
        command.append("--metrics")
    if not config.web_ui:
        command.append("--no-ui")
    return command
