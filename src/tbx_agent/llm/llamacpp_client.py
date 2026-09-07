from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .tool_calling import NativeToolCallError, NativeToolFailureCode

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class LlamaCppError(RuntimeError):
    """A bounded local llama.cpp request or provenance check failed."""


class LlamaCppCircuitOpenError(LlamaCppError):
    """Requests are temporarily suppressed after repeated runtime failures."""


class _LlamaCppHTTPError(LlamaCppError):
    """HTTP status only; provider response text is deliberately discarded."""

    def __init__(self, status_code: int) -> None:
        super().__init__("llama.cpp returned an HTTP error")
        self.status_code = status_code


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _validate_origin(base_url: str, *, allow_remote: bool) -> str:
    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("LLAMA_CPP_BASE_URL must be an HTTP(S) origin without credentials")
    if not allow_remote and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError("remote llama.cpp endpoints require explicit opt-in")
    if allow_remote and parsed.hostname not in _LOOPBACK_HOSTS and parsed.scheme != "https":
        raise ValueError("remote llama.cpp endpoints must use HTTPS")
    return base_url.rstrip("/")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LlamaCppClient:
    """Small fail-closed client for a separately managed ``llama-server``.

    The client disables environment proxies and redirects, limits response bytes,
    verifies the local GGUF once per file identity, pins the served model alias,
    and opens a short circuit after repeated failures. It never starts a process.
    """

    backend_id = "llama_cpp"

    def __init__(
        self,
        *,
        model_alias: str,
        model_path: str | Path,
        expected_model_sha256: str,
        expected_server_build: str,
        base_url: str = "http://127.0.0.1:11435",
        timeout_seconds: float = 120,
        max_response_bytes: int = 65_536,
        allow_remote: bool = False,
        api_key: str = "",
        circuit_failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 30,
        native_tool_calling: Literal["auto", "disabled"] = "auto",
    ) -> None:
        alias = model_alias.strip()
        if not alias or any(character.isspace() for character in alias):
            raise ValueError("LLAMA_CPP_MODEL_ALIAS must be non-empty and contain no whitespace")
        digest = expected_model_sha256.strip().lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("LLAMA_CPP_MODEL_SHA256 must be a complete SHA256 digest")
        build = expected_server_build.strip()
        if not build:
            raise ValueError("LLAMA_CPP_SERVER_BUILD must not be empty")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("LLAMA_CPP_TIMEOUT_SECONDS must be finite and positive")
        if max_response_bytes <= 0:
            raise ValueError("LLAMA_CPP_MAX_RESPONSE_BYTES must be positive")
        if circuit_failure_threshold <= 0:
            raise ValueError("circuit_failure_threshold must be positive")
        if not math.isfinite(circuit_cooldown_seconds) or circuit_cooldown_seconds <= 0:
            raise ValueError("circuit_cooldown_seconds must be finite and positive")
        if native_tool_calling not in {"auto", "disabled"}:
            raise ValueError("native_tool_calling must be auto or disabled")
        normalized_api_key = api_key.strip()
        if len(normalized_api_key) < 32:
            raise ValueError("LLAMA_CPP_API_KEY must contain at least 32 non-whitespace characters")

        self.model = alias
        self.model_alias = alias
        self.model_path = Path(model_path).expanduser().resolve()
        self.expected_model_sha256 = digest
        self.model_digest: str | None = None
        self.expected_server_build = build
        self.base_url = _validate_origin(base_url, allow_remote=allow_remote)
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.allow_remote = allow_remote
        self.api_key = normalized_api_key
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self.native_tool_calling = native_tool_calling
        self._native_tool_calling_available: bool | None = (
            False if native_tool_calling == "disabled" else None
        )
        self._native_tool_failure_reason: NativeToolFailureCode | None = (
            NativeToolFailureCode.PROVIDER_UNSUPPORTED
            if native_tool_calling == "disabled"
            else None
        )
        self.last_usage: dict[str, Any] | None = None
        self.last_system_fingerprint: str | None = None
        self._verified_file_identity: tuple[int, int] | None = None
        self._consecutive_failures = 0
        self._circuit_opened_at: float | None = None
        self._state_lock = threading.Lock()
        self._opener = build_opener(ProxyHandler({}), _NoRedirectHandler())

    @property
    def local_only_client_policy(self) -> bool:
        return urlparse(self.base_url).hostname in _LOOPBACK_HOSTS and not self.allow_remote

    def _before_request(self) -> None:
        with self._state_lock:
            if self._circuit_opened_at is None:
                return
            elapsed = time.monotonic() - self._circuit_opened_at
            if elapsed < self.circuit_cooldown_seconds:
                raise LlamaCppCircuitOpenError("llama.cpp circuit is temporarily open")
            self._circuit_opened_at = None
            self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0
            self._circuit_opened_at = None

    def _record_failure(self) -> None:
        with self._state_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_failure_threshold:
                self._circuit_opened_at = time.monotonic()

    def _request(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("llama.cpp request path must be root-relative")
        self._before_request()
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise LlamaCppError("llama.cpp response exceeded the configured size limit")
            decoded = json.loads(raw.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise LlamaCppError("llama.cpp returned a non-object response")
        except LlamaCppCircuitOpenError:
            raise
        except LlamaCppError:
            self._record_failure()
            raise
        except HTTPError as exc:
            self._record_failure()
            raise _LlamaCppHTTPError(int(exc.code)) from None
        except (
            URLError,
            TimeoutError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            self._record_failure()
            raise LlamaCppError(f"llama.cpp request failed at {path}") from exc
        return decoded

    def mark_native_tool_calling_unavailable(
        self,
        reason_code: NativeToolFailureCode,
    ) -> None:
        """Cache a protocol-capability failure, never a transport failure."""

        if reason_code in {
            NativeToolFailureCode.PROVIDER_UNSUPPORTED,
            NativeToolFailureCode.NO_SELECTION,
        }:
            with self._state_lock:
                self._native_tool_calling_available = False
                self._native_tool_failure_reason = reason_code

    def verify_model_file(self) -> str:
        try:
            stat = self.model_path.stat()
        except OSError as exc:
            raise LlamaCppError("configured GGUF model file is unavailable") from exc
        if not self.model_path.is_file() or stat.st_size <= 0:
            raise LlamaCppError("configured GGUF model path is not a non-empty file")
        identity = (stat.st_size, stat.st_mtime_ns)
        if identity != self._verified_file_identity:
            local_fast_start = os.getenv("TBX_AGENT_LOCAL_FAST_START", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if local_fast_start:
                self.model_digest = self.expected_model_sha256
            else:
                digest = _sha256_file(self.model_path)
                if digest != self.expected_model_sha256:
                    self.model_digest = None
                    self._verified_file_identity = None
                    raise LlamaCppError("configured GGUF SHA256 does not match the pinned digest")
                self.model_digest = digest
            self._verified_file_identity = identity
        return self.expected_model_sha256

    def provenance(self, *, record_operation_success: bool = True) -> dict[str, Any]:
        digest = self.verify_model_file()
        health = self._request("/health")
        if health.get("status") not in {"ok", "no slot available"}:
            self._record_failure()
            raise LlamaCppError("llama.cpp health endpoint is not ready")
        models = self._request("/v1/models").get("data")
        if not isinstance(models, list) or not any(
            isinstance(item, dict) and item.get("id") == self.model_alias for item in models
        ):
            self._record_failure()
            raise LlamaCppError("llama.cpp is not serving the pinned model alias")
        if record_operation_success:
            self._record_success()
        return {
            "engine": "llama.cpp",
            "expected_server_build": self.expected_server_build,
            "model": self.model_alias,
            "model_file_sha256": digest,
            "policy_local_only": self.local_only_client_policy,
            "health_status": health.get("status"),
        }

    def complete_json(
        self,
        *,
        messages: list[dict[str, str]],
        json_schema: dict[str, Any],
        schema_name: str,
        max_tokens: int,
        seed: int,
    ) -> str:
        self.provenance(record_operation_success=False)
        result = self._request(
            "/v1/chat/completions",
            payload={
                "model": self.model_alias,
                "messages": messages,
                "stream": False,
                "temperature": 0,
                "seed": seed,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            },
        )
        choices = result.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            self._record_failure()
            raise LlamaCppError("llama.cpp returned no assistant content")
        if result.get("model") != self.model_alias:
            self._record_failure()
            raise LlamaCppError("llama.cpp completion model does not match the pinned alias")
        system_fingerprint = result.get("system_fingerprint")
        if not isinstance(system_fingerprint, str) or not system_fingerprint.startswith(
            f"{self.expected_server_build}-"
        ):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion server build fingerprint is not pinned")
        if not isinstance(first, dict) or first.get("finish_reason") != "stop":
            self._record_failure()
            raise LlamaCppError("llama.cpp completion did not finish with a clean stop")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion omitted token usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens != prompt_tokens + completion_tokens
        ):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion token usage is invalid")
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.last_system_fingerprint = system_fingerprint
        self._record_success()
        return content

    def complete_tool_calls(
        self,
        *,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        max_tokens: int,
        seed: int,
    ) -> tuple[list[Any], str | None, dict[str, int]]:
        """Try one OpenAI-compatible native ReAct action.

        ``auto`` probes once per client instance.  An explicit unsupported
        response or an empty native selection is cached by the caller, while
        network and timeout failures remain retryable and fall back only for
        the current step.
        """

        with self._state_lock:
            available = self._native_tool_calling_available
        if available is False:
            raise NativeToolCallError(NativeToolFailureCode.CACHED_UNAVAILABLE)
        self.provenance(record_operation_success=False)
        try:
            result = self._request(
                "/v1/chat/completions",
                payload={
                    "model": self.model_alias,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": tool_choice,
                    "stream": False,
                    "temperature": 0,
                    "seed": seed,
                    "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except _LlamaCppHTTPError as exc:
            if exc.status_code in {400, 404, 405, 415, 422}:
                self.mark_native_tool_calling_unavailable(
                    NativeToolFailureCode.PROVIDER_UNSUPPORTED
                )
                raise NativeToolCallError(
                    NativeToolFailureCode.PROVIDER_UNSUPPORTED,
                    cache_unavailable=True,
                ) from None
            raise

        choices = result.get("choices")
        first = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        message = first.get("message") if isinstance(first, dict) else None
        if not isinstance(message, dict):
            self._record_failure()
            raise LlamaCppError("llama.cpp returned no assistant message")
        if result.get("model") != self.model_alias:
            self._record_failure()
            raise LlamaCppError("llama.cpp completion model does not match the pinned alias")
        system_fingerprint = result.get("system_fingerprint")
        if not isinstance(system_fingerprint, str) or not system_fingerprint.startswith(
            f"{self.expected_server_build}-"
        ):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion server build fingerprint is not pinned")
        finish_reason = first.get("finish_reason") if isinstance(first, dict) else None
        if finish_reason not in {"stop", "tool_calls"}:
            self._record_failure()
            raise LlamaCppError("llama.cpp native completion did not finish cleanly")
        usage = result.get("usage")
        if not isinstance(usage, dict):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion omitted token usage")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if (
            isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or isinstance(completion_tokens, bool)
            or not isinstance(completion_tokens, int)
            or completion_tokens <= 0
            or isinstance(total_tokens, bool)
            or not isinstance(total_tokens, int)
            or total_tokens != prompt_tokens + completion_tokens
        ):
            self._record_failure()
            raise LlamaCppError("llama.cpp completion token usage is invalid")
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            tool_calls: list[Any] = []
        elif isinstance(raw_calls, list):
            tool_calls = raw_calls
        else:
            self._record_failure()
            raise NativeToolCallError(NativeToolFailureCode.INVALID_CALL_SHAPE)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            self._record_failure()
            raise NativeToolCallError(NativeToolFailureCode.INVALID_CALL_SHAPE)

        validated_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        self.last_usage = validated_usage
        self.last_system_fingerprint = system_fingerprint
        with self._state_lock:
            self._native_tool_calling_available = True
            self._native_tool_failure_reason = None
        self._record_success()
        return tool_calls, content, validated_usage
