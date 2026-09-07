from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..paths import default_runtime_root
from .runtime_supervisor import LlamaCppRuntimeConfig, load_runtime_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "llm_runtime.yaml"
SCHEMA_VERSION = 1
FALLBACK_SEED = 20260829
# MedGemma tokenizes the fixed sentinel more finely than the former Qwen
# runtime. This remains a tiny protocol-only ceiling, but leaves enough room
# for the sentinel plus a clean stop token in both sync and SSE modes.
MAX_OUTPUT_TOKENS = 64
FIXED_SYSTEM_PROMPT = (
    "You are participating in a deterministic local API protocol check. "
    "Return only the JSON object admitted by the supplied schema."
)
FIXED_USER_PROMPT = "Return the protocol-ready status object."
EXPECTED_CONTENT = '{"status":"OPENAI_COMPAT_OK"}'
EXPECTED_PAYLOAD = {"status": "OPENAI_COMPAT_OK"}
PROTOCOL_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "tbx_openai_compat_status",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "const": "OPENAI_COMPAT_OK"}
            },
            "required": ["status"],
            "additionalProperties": False,
        },
    },
}
OPENAI_COMPATIBLE_ENDPOINTS = (
    "/v1/models",
    "/v1/chat/completions",
)
OPENAI_COMPAT_SMOKE_ENDPOINTS = (
    "/v1/models",
    "/v1/chat/completions:sync",
    "/v1/chat/completions:sse",
)
HYPOTHESIS = (
    "The already-running loopback llama.cpp service advertising the configured alias and build "
    "satisfies the bounded OpenAI Python SDK contract for model discovery, deterministic chat "
    "completion, and usage-bearing chat SSE without changing runtime or model state."
)


def default_output_root(environment: Mapping[str, str] | None = None) -> Path:
    """Return the portable receipt root, with an optional dedicated override."""

    environment = os.environ if environment is None else environment
    override = environment.get("TBX_AGENT_OPENAI_COMPAT_OUTPUT_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return default_runtime_root(environment) / "evaluation_runs" / "openai-compat"


# Backward-compatible import for callers that only need the process-start snapshot.
DEFAULT_OUTPUT_ROOT = default_output_root()


class CompatibilityCheckError(RuntimeError):
    """A contract failure whose reason and details are safe to persist."""

    def __init__(self, reason: str, **safe_details: int | bool | str | None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.safe_details = safe_details


@dataclass(frozen=True)
class SmokeOutcome:
    receipt: dict[str, Any]
    result_path: Path
    result_sha256: str

    @property
    def passed(self) -> bool:
        return self.receipt["status"] == "passed_openai_compat_smoke"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _source_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    revision = completed.stdout.strip()
    return revision if len(revision) == 40 else "unavailable"


def _source_binding() -> dict[str, Any]:
    source_files = {
        "tbx_agent/llm/openai_compat.py": Path(__file__).resolve(),
        "tbx_agent/llm/runtime_supervisor.py": Path(__file__)
        .with_name("runtime_supervisor.py")
        .resolve(),
        "scripts/test_openai_compatible_api.py": (
            PROJECT_ROOT / "scripts" / "test_openai_compatible_api.py"
        ).resolve(),
    }
    file_digests = {name: _sha256_file(path) for name, path in source_files.items()}
    return {
        "algorithm": "sha256(canonical relative-path to file-sha256 map)",
        "files": file_digests,
        "source_tree_sha256": _canonical_sha256(file_digests),
    }


def _load_acl_key(config: LlamaCppRuntimeConfig) -> str:
    key_file = config.api_key_file
    if not key_file.is_file():
        raise FileNotFoundError("configured ACL key file is unavailable")
    if key_file.stat().st_size > 64 * 1024:
        raise ValueError("configured ACL key file exceeds the bounded size")
    keys = [
        line.strip()
        for line in key_file.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys or any(len(key) < 32 for key in keys):
        raise ValueError("configured ACL key file does not contain only strong keys")
    return keys[0]


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _positive_usage(usage: Any) -> dict[str, int]:
    prompt_tokens = _get(usage, "prompt_tokens")
    completion_tokens = _get(usage, "completion_tokens")
    total_tokens = _get(usage, "total_tokens")
    values = (prompt_tokens, completion_tokens, total_tokens)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise CompatibilityCheckError("invalid_usage_shape")
    if prompt_tokens <= 0 or completion_tokens <= 0 or total_tokens != sum(values[:2]):
        raise CompatibilityCheckError("invalid_usage_counts")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": FIXED_SYSTEM_PROMPT},
        {"role": "user", "content": FIXED_USER_PROMPT},
    ]


def _request_parameters(config: LlamaCppRuntimeConfig) -> dict[str, Any]:
    return {
        "model": config.model_alias,
        "messages": _messages(),
        "temperature": 0,
        "seed": config.seed,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": PROTOCOL_RESPONSE_FORMAT,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def _validate_server_build(response: Any, expected_build: str, reason: str) -> None:
    fingerprint = _get(response, "system_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint.startswith(f"{expected_build}-"):
        raise CompatibilityCheckError(reason)


def _validate_sync(response: Any, expected_alias: str, expected_build: str) -> dict[str, int]:
    if _get(response, "model") != expected_alias:
        raise CompatibilityCheckError("sync_response_alias_mismatch")
    _validate_server_build(response, expected_build, "sync_server_build_mismatch")
    choices = _get(response, "choices")
    if not isinstance(choices, Iterable) or isinstance(choices, (str, bytes, Mapping)):
        raise CompatibilityCheckError("sync_choices_missing")
    choices = list(choices)
    if len(choices) != 1:
        raise CompatibilityCheckError("sync_choice_count_mismatch", observed_count=len(choices))
    choice = choices[0]
    if _get(choice, "finish_reason") != "stop":
        raise CompatibilityCheckError("sync_finish_reason_mismatch")
    message = _get(choice, "message")
    if _get(message, "role") != "assistant":
        raise CompatibilityCheckError("sync_role_mismatch")
    content = _get(message, "content")
    if not isinstance(content, str):
        raise CompatibilityCheckError("sync_content_not_text")
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        parsed_content = None
    if parsed_content != EXPECTED_PAYLOAD:
        raise CompatibilityCheckError(
            "sync_content_mismatch",
            observed_content_length=len(content),
            observed_content_sha256=_sha256_bytes(content.encode("utf-8")),
        )
    reasoning = _get(message, "reasoning_content")
    if reasoning not in (None, ""):
        raise CompatibilityCheckError("sync_unexpected_reasoning_content")
    return _positive_usage(_get(response, "usage"))


def _validate_stream(
    stream: Iterable[Any], expected_alias: str, expected_build: str
) -> dict[str, Any]:
    content_parts: list[str] = []
    finish_reasons: list[str] = []
    usage_records: list[dict[str, int]] = []
    event_sequence: list[str] = []
    chunk_count = 0
    for chunk in stream:
        chunk_count += 1
        if usage_records:
            raise CompatibilityCheckError("stream_chunk_after_terminal_usage")
        if _get(chunk, "model") != expected_alias:
            raise CompatibilityCheckError("stream_response_alias_mismatch")
        _validate_server_build(chunk, expected_build, "stream_server_build_mismatch")
        usage = _get(chunk, "usage")
        if usage is not None:
            choices_with_usage = list(_get(chunk, "choices", []))
            if choices_with_usage:
                raise CompatibilityCheckError("stream_usage_chunk_has_choices")
            if finish_reasons != ["stop"]:
                raise CompatibilityCheckError("stream_usage_before_stop")
            usage_records.append(_positive_usage(usage))
            event_sequence.append("usage")
        choices = list(_get(chunk, "choices", []))
        if len(choices) > 1:
            raise CompatibilityCheckError(
                "stream_choice_count_exceeded", observed_count=len(choices)
            )
        for choice in choices:
            finish_reason = _get(choice, "finish_reason")
            if finish_reason is not None:
                if finish_reasons:
                    raise CompatibilityCheckError("stream_duplicate_finish_reason")
                finish_reasons.append(str(finish_reason))
                event_sequence.append("finish")
            delta = _get(choice, "delta")
            content = _get(delta, "content")
            if content is not None:
                if not isinstance(content, str):
                    raise CompatibilityCheckError("stream_content_not_text")
                if finish_reasons:
                    raise CompatibilityCheckError("stream_content_after_finish")
                content_parts.append(content)
                if content:
                    event_sequence.append("content")
            reasoning = _get(delta, "reasoning_content")
            if reasoning not in (None, ""):
                raise CompatibilityCheckError("stream_unexpected_reasoning_content")
    content = "".join(content_parts)
    try:
        parsed_content = json.loads(content)
    except json.JSONDecodeError:
        parsed_content = None
    if parsed_content != EXPECTED_PAYLOAD:
        raise CompatibilityCheckError(
            "stream_content_mismatch",
            observed_content_length=len(content),
            observed_content_sha256=_sha256_bytes(content.encode("utf-8")),
        )
    if finish_reasons != ["stop"]:
        raise CompatibilityCheckError(
            "stream_finish_reason_mismatch",
            observed_finish_count=len(finish_reasons),
        )
    if len(usage_records) != 1:
        raise CompatibilityCheckError(
            "stream_usage_chunk_count_mismatch",
            observed_count=len(usage_records),
        )
    return {
        "chunk_count": chunk_count,
        "usage_chunk_count": len(usage_records),
        "usage": usage_records[0],
        "event_sequence": event_sequence,
    }


def _safe_failure(exc: Exception, stage: str) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "stage": stage,
        "reason": (
            exc.reason if isinstance(exc, CompatibilityCheckError) else "runtime_or_sdk_error"
        ),
        "error_type": type(exc).__name__,
        "message": "OpenAI compatibility smoke failed; raw exception text is withheld.",
    }
    if isinstance(exc, CompatibilityCheckError):
        failure["safe_details"] = exc.safe_details
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and not isinstance(status_code, bool):
        failure["http_status_code"] = status_code
    return failure


def _write_json_atomic_exclusive(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable result already exists: {path}")
    temporary = path.with_suffix(".json.tmp")
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"immutable result already exists: {path}")
    os.replace(temporary, path)


def _openai_version() -> str:
    try:
        return importlib.metadata.version("openai")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def run_openai_compat_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_root: Path | None = None,
    client_factory: Callable[..., Any] | None = None,
    http_client_factory: Callable[..., Any] | None = None,
    created_at: datetime | None = None,
) -> SmokeOutcome:
    """Probe an already-running local server and retain a secret-free audit receipt."""

    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    stamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"openai-compat-{stamp}"
    resolved_output_root = default_output_root() if output_root is None else output_root
    run_root = resolved_output_root.expanduser().resolve() / stamp
    run_root.mkdir(parents=True, exist_ok=False)
    result_path = run_root / "result.json"
    resolved_config_path = config_path.resolve()
    prompt_fingerprint = _canonical_sha256(_messages())
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "llamacpp_openai_python_sdk_compatibility_smoke",
        "run_id": run_id,
        "status": "running",
        "hypothesis": HYPOTHESIS,
        "created_at": timestamp.isoformat(),
        "seed": FALLBACK_SEED,
        "split_hash": None,
        "source_revision": _source_revision(),
        "source_tree_sha256": None,
        "source_binding": None,
        "selection_use": False,
        "model_selection": False,
        "threshold_selection": False,
        "locked_or_hidden_test_used": False,
        "official_hidden_test_used": False,
        "clinical_validation": False,
        "major_variables_changed": [],
        "full_configuration": {
            "runtime_config_path": str(resolved_config_path),
            "runtime_config_file_sha256": None,
            "runtime_config_canonical_sha256": None,
            "runtime_id": None,
            "server_build": None,
            "model_alias": None,
            "model_sha256": None,
            "openai_base_url": None,
            "openai_sdk_version": _openai_version(),
            "openai_sdk_max_retries": 0,
            "http_client_trust_env": False,
            "http_client_follow_redirects": False,
            "tested_endpoints": list(OPENAI_COMPAT_SMOKE_ENDPOINTS),
            "request_timeout_seconds": None,
            "temperature": 0,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "stream_include_usage": True,
            "chat_template_enable_thinking": False,
            "fixture_kind": "fixed_synthetic_non_medical_protocol_prompt",
            "prompt_sha256": prompt_fingerprint,
            "expected_content_sha256": _sha256_bytes(EXPECTED_CONTENT.encode("utf-8")),
            "credential_source": "runtime_acl_key_file",
            "credential_persisted_in_receipt": False,
            "dataset_used": False,
            "seed": FALLBACK_SEED,
            "smoke_module_sha256": None,
            "smoke_entrypoint_sha256": None,
            "runtime_asset_hashes_reverified": False,
            "served_process_binary_attested": False,
        },
        "metrics": {
            "models_list_success": False,
            "expected_alias_present": False,
            "chat_sync_success": False,
            "chat_sync_exact_content": False,
            "chat_stream_success": False,
            "chat_stream_exact_content": False,
            "stream_usage_chunk_count": 0,
        },
        "runtime": {
            "models_list_milliseconds": None,
            "chat_sync_milliseconds": None,
            "chat_stream_milliseconds": None,
            "total_milliseconds": None,
        },
        "peak_vram_bytes": None,
        "peak_vram_measurement": (
            "unavailable_not_measured_existing_server_protocol_compatibility_smoke"
        ),
        "vram_measurement": {
            "available": False,
            "scope": "process",
            "reason": "not_measured_by_protocol_only_openai_compatibility_smoke",
        },
        "failure": None,
        "limitations": [
            "The smoke uses only fixed synthetic non-medical text and no dataset split.",
            "It does not start, stop, reconfigure, or benchmark the llama.cpp service.",
            (
                "It binds the declared runtime config and verifies the served alias/build, but "
                "does not rehash model/binary assets or attest the serving process executable."
            ),
            (
                "It verifies a bounded protocol contract, not open-ended generation or "
                "clinical quality."
            ),
            (
                "Peak process VRAM is unavailable because this protocol smoke does not "
                "sample GPU state."
            ),
        ],
    }
    started = time.perf_counter()
    client: Any | None = None
    http_client: Any | None = None
    stream: Any | None = None
    stage = "source_binding"
    try:
        source_binding = _source_binding()
        receipt["source_binding"] = source_binding
        receipt["source_tree_sha256"] = source_binding["source_tree_sha256"]
        receipt["full_configuration"].update(
            {
                "smoke_module_sha256": source_binding["files"]["tbx_agent/llm/openai_compat.py"],
                "smoke_entrypoint_sha256": source_binding["files"][
                    "scripts/test_openai_compatible_api.py"
                ],
            }
        )
        stage = "load_runtime_config"
        config = load_runtime_config(resolved_config_path)
        receipt["seed"] = config.seed
        base_url = f"http://{config.host}:{config.port}/v1"
        receipt["full_configuration"].update(
            {
                "runtime_config_file_sha256": _sha256_file(resolved_config_path),
                "runtime_config_canonical_sha256": config.canonical_sha256(),
                "runtime_id": config.runtime_id,
                "server_build": config.server_build,
                "model_alias": config.model_alias,
                "model_sha256": config.model_sha256,
                "openai_base_url": base_url,
                "request_timeout_seconds": config.request_timeout_seconds,
                "seed": config.seed,
            }
        )
        stage = "load_acl_key"
        api_key = _load_acl_key(config)
        if client_factory is None or http_client_factory is None:
            from openai import DefaultHttpxClient, OpenAI

            client_factory = client_factory or OpenAI
            http_client_factory = http_client_factory or DefaultHttpxClient
        stage = "construct_no_proxy_http_client"
        http_client = http_client_factory(
            trust_env=False,
            follow_redirects=False,
            timeout=float(config.request_timeout_seconds),
        )
        stage = "construct_openai_client"
        client = client_factory(
            api_key=api_key,
            base_url=base_url,
            timeout=float(config.request_timeout_seconds),
            max_retries=0,
            http_client=http_client,
        )

        stage = "models_list"
        step_started = time.perf_counter()
        models = client.models.list()
        receipt["runtime"]["models_list_milliseconds"] = (time.perf_counter() - step_started) * 1000
        model_ids = [
            str(model_id)
            for model in list(_get(models, "data", []))
            if (model_id := _get(model, "id")) is not None
        ]
        receipt["metrics"]["models_returned_count"] = len(model_ids)
        receipt["metrics"]["models_list_success"] = True
        alias_present = config.model_alias in model_ids
        receipt["metrics"]["expected_alias_present"] = alias_present
        if not alias_present:
            raise CompatibilityCheckError(
                "models_alias_mismatch",
                observed_model_count=len(model_ids),
            )

        stage = "chat_sync"
        step_started = time.perf_counter()
        sync_response = client.chat.completions.create(
            **_request_parameters(config),
            stream=False,
        )
        receipt["runtime"]["chat_sync_milliseconds"] = (time.perf_counter() - step_started) * 1000
        sync_usage = _validate_sync(sync_response, config.model_alias, config.server_build)
        receipt["metrics"].update(
            {
                "chat_sync_success": True,
                "chat_sync_exact_content": True,
                "chat_sync_prompt_tokens": sync_usage["prompt_tokens"],
                "chat_sync_completion_tokens": sync_usage["completion_tokens"],
                "chat_sync_total_tokens": sync_usage["total_tokens"],
            }
        )

        stage = "chat_stream"
        step_started = time.perf_counter()
        stream = client.chat.completions.create(
            **_request_parameters(config),
            stream=True,
            stream_options={"include_usage": True},
        )
        stream_result = _validate_stream(stream, config.model_alias, config.server_build)
        receipt["runtime"]["chat_stream_milliseconds"] = (time.perf_counter() - step_started) * 1000
        stream_usage = stream_result["usage"]
        receipt["metrics"].update(
            {
                "chat_stream_success": True,
                "chat_stream_exact_content": True,
                "stream_chunk_count": stream_result["chunk_count"],
                "stream_usage_chunk_count": stream_result["usage_chunk_count"],
                "stream_event_sequence": stream_result["event_sequence"],
                "chat_stream_prompt_tokens": stream_usage["prompt_tokens"],
                "chat_stream_completion_tokens": stream_usage["completion_tokens"],
                "chat_stream_total_tokens": stream_usage["total_tokens"],
            }
        )
        receipt["status"] = "passed_openai_compat_smoke"
    except Exception as exc:
        receipt["status"] = "failed_retained"
        receipt["failure"] = _safe_failure(exc, stage)
    finally:
        if stream is not None and callable(getattr(stream, "close", None)):
            try:
                stream.close()
            except Exception:
                receipt.setdefault("cleanup", {})["stream_close"] = "failed_without_raw_error"
        if client is not None and callable(getattr(client, "close", None)):
            try:
                client.close()
            except Exception:
                receipt.setdefault("cleanup", {})["client_close"] = "failed_without_raw_error"
        if http_client is not None and callable(getattr(http_client, "close", None)):
            try:
                http_client.close()
            except Exception:
                receipt.setdefault("cleanup", {})["http_client_close"] = "failed_without_raw_error"
        receipt["runtime"]["total_milliseconds"] = (time.perf_counter() - started) * 1000

    _write_json_atomic_exclusive(result_path, receipt)
    return SmokeOutcome(
        receipt=receipt,
        result_path=result_path,
        result_sha256=_sha256_file(result_path),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded OpenAI Python SDK compatibility smoke against the already-running "
            "pinned local llama.cpp service. The command never manages service lifecycle."
        )
    )
    parser.add_argument(
        "--runtime-config",
        "--config",
        dest="config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Pinned llama.cpp runtime YAML (`--config` remains a compatibility alias).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "receipt directory; defaults to TBX_AGENT_OPENAI_COMPAT_OUTPUT_ROOT or "
            "<platform runtime root>/evaluation_runs/openai-compat"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outcome = run_openai_compat_smoke(
        config_path=args.config,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "run_id": outcome.receipt["run_id"],
                "status": outcome.receipt["status"],
                "result_path": str(outcome.result_path),
                "result_sha256": outcome.result_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if outcome.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
