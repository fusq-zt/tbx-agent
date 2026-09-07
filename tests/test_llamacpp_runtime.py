from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tbx_agent.config import _load_llama_cpp_api_key
from tbx_agent.llm.llamacpp_client import (
    LlamaCppCircuitOpenError,
    LlamaCppClient,
    LlamaCppError,
)
from tbx_agent.llm.runtime_supervisor import (
    LlamaCppRuntimeConfig,
    build_server_command,
    verify_runtime_assets,
)
from tbx_agent.llm.tool_calling import (
    NativeToolFailureCode,
    attempt_native_tool_selection,
    native_tool_definitions,
)
from tbx_agent.narrator import LlamaCppNarrator, NarrationError
from tbx_agent.schemas import AgentResponse, NarrationStatus, ResponseKind, Urgency


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _client(model_path: Path, **kwargs) -> LlamaCppClient:
    return LlamaCppClient(
        model_alias="tbx-qwen3.5-4b-q4-k-m",
        model_path=model_path,
        expected_model_sha256=_digest(model_path),
        expected_server_build="b10517",
        timeout_seconds=2,
        max_response_bytes=4096,
        api_key="k" * 48,
        **kwargs,
    )


def _response() -> AgentResponse:
    return AgentResponse(
        request_id="secret-request",
        trace_id="secret-trace",
        thread_id="secret-thread",
        response_kind=ResponseKind.NEXT_TEST_INFORMATION,
        summary="模型识别为结核类，建议进一步检查。",
        diagnostic_information=["结合症状、接触史和病原学检查进行综合评估。"],
        limitations=["本系统不能确诊或排除肺结核。"],
        urgency=Urgency.PROMPT_EVALUATION,
    )


def test_llamacpp_client_rejects_remote_plaintext_and_model_digest_drift(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    with pytest.raises(ValueError, match="remote llama.cpp"):
        _client(model, base_url="http://runtime.example.test")
    with pytest.raises(ValueError, match="HTTPS"):
        _client(model, base_url="http://runtime.example.test", allow_remote=True)

    client = LlamaCppClient(
        model_alias="qwen",
        model_path=model,
        expected_model_sha256="0" * 64,
        expected_server_build="b10517",
        api_key="k" * 48,
    )
    with pytest.raises(LlamaCppError, match="SHA256"):
        client.verify_model_file()


def test_llamacpp_provenance_pins_alias_and_local_file(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)

    def request(path: str, *, payload=None):
        assert payload is None
        if path == "/health":
            return {"status": "ok"}
        assert path == "/v1/models"
        return {"data": [{"id": "tbx-qwen3.5-4b-q4-k-m"}]}

    monkeypatch.setattr(client, "_request", request)
    provenance = client.provenance()
    assert provenance["model_file_sha256"] == _digest(model)
    assert provenance["policy_local_only"] is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model": "wrong-model"}, "pinned alias"),
        ({"system_fingerprint": "wrong-build"}, "build fingerprint"),
        ({"choices": [{"message": {"content": "{}"}, "finish_reason": "length"}]}, "clean stop"),
        (
            {"usage": {"prompt_tokens": 2, "completion_tokens": 0, "total_tokens": 2}},
            "token usage",
        ),
    ],
)
def test_llamacpp_completion_requires_model_stop_and_positive_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict,
    message: str,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)

    def request(path: str, *, payload=None):
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": client.model_alias}]}
        assert path == "/v1/chat/completions"
        result = {
            "model": client.model_alias,
            "system_fingerprint": "b10517-test",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
        result.update(override)
        return result

    monkeypatch.setattr(client, "_request", request)

    with pytest.raises(LlamaCppError, match=message):
        client.complete_json(
            messages=[{"role": "user", "content": "probe"}],
            json_schema={"type": "object"},
            schema_name="probe",
            max_tokens=8,
            seed=1,
        )


def test_llamacpp_native_tool_call_uses_openai_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)
    captured: dict = {}
    monkeypatch.setattr(client, "provenance", lambda **_kwargs: {"model": client.model})

    def request(path: str, *, payload=None):
        assert path == "/v1/chat/completions"
        captured.update(payload)
        return {
            "model": client.model_alias,
            "system_fingerprint": "b10517-test",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "classify_cxr",
                                    "arguments": "{}",
                                }
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        }

    monkeypatch.setattr(client, "_request", request)
    calls, content, usage = client.complete_tool_calls(
        messages=[{"role": "user", "content": "分析当前胸片"}],
        tools=native_tool_definitions(trusted_query="分析当前胸片"),
        tool_choice="auto",
        max_tokens=128,
        seed=1,
    )

    assert calls[0]["function"]["name"] == "classify_cxr"
    assert content is None
    assert usage["total_tokens"] == 11
    assert captured["tool_choice"] == "auto"
    assert "response_format" not in captured


def test_llamacpp_empty_native_response_is_probed_once_then_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)
    requests = 0
    monkeypatch.setattr(client, "provenance", lambda **_kwargs: {"model": client.model})

    def request(path: str, *, payload=None):
        nonlocal requests
        requests += 1
        assert path == "/v1/chat/completions"
        return {
            "model": client.model_alias,
            "system_fingerprint": "b10517-test",
            "choices": [
                {"message": {"content": None}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
        }

    monkeypatch.setattr(client, "_request", request)
    parameters = {
        "messages": [{"role": "user", "content": "test"}],
        "trusted_query": "test",
        "max_tokens": 32,
        "seed": 1,
    }
    first = attempt_native_tool_selection(client, **parameters)
    second = attempt_native_tool_selection(client, **parameters)

    assert first.failure_reason == NativeToolFailureCode.NO_SELECTION
    assert second.failure_reason == NativeToolFailureCode.CACHED_UNAVAILABLE
    assert requests == 1


def test_llamacpp_narrator_uses_schema_and_never_exposes_response_identifiers(
    tmp_path: Path, monkeypatch
):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    narrator = LlamaCppNarrator(
        model_alias="tbx-qwen3.5-4b-q4-k-m",
        model_path=str(model),
        expected_model_sha256=_digest(model),
        expected_server_build="b10517",
        api_key="k" * 48,
    )
    captured = {}

    def complete_json(**kwargs):
        captured.update(kwargs)
        narrator.client.model_digest = _digest(model)
        return json.dumps(
            {
                "summary": (
                    "请注意：模型识别为结核类，建议进一步检查。 "
                    "结合症状、接触史和病原学检查进行综合评估。"
                )
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(narrator.client, "complete_json", complete_json)
    rendered = narrator.narrate(_response())
    assert rendered.narration_status == NarrationStatus.APPLIED
    assert rendered.narrator_backend == "llama_cpp"
    assert rendered.narrator_model_digest == _digest(model)
    request_text = json.dumps(captured, ensure_ascii=False)
    assert "secret-request" not in request_text
    assert "secret-trace" not in request_text
    assert "secret-thread" not in request_text
    assert captured["json_schema"]["additionalProperties"] is False
    allowed = captured["json_schema"]["properties"]["summary"]["enum"]
    assert "模型识别为结核类，建议进一步检查。" in allowed
    assert (
        "模型识别为结核类，建议进一步检查。 "
        "结合症状、接触史和病原学检查进行综合评估。"
    ) in allowed
    assert all("本系统不能确诊或排除肺结核。" not in item for item in allowed)


def test_llamacpp_readiness_runs_and_caches_real_structured_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    narrator = LlamaCppNarrator(
        model_alias="tbx-qwen3.5-4b-q4-k-m",
        model_path=str(model),
        expected_model_sha256=_digest(model),
        expected_server_build="b10517",
        api_key="k" * 48,
    )
    calls = 0

    def provenance():
        narrator.client.model_digest = _digest(model)
        narrator.model_digest = _digest(model)
        return {"model": narrator.model, "model_file_sha256": _digest(model)}

    def complete_json(**kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["json_schema"]["properties"]["status"]["const"] == "ready"
        assert kwargs["max_tokens"] == 64
        narrator.client.last_usage = {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        }
        return '{"status":"ready"}'

    monkeypatch.setattr(narrator, "provenance", provenance)
    monkeypatch.setattr(narrator.client, "complete_json", complete_json)

    first = narrator.probe_generation()
    second = narrator.probe_generation()

    assert first["generation_probed"] is True
    assert first["cached"] is False
    assert second["cached"] is True
    assert calls == 1


def test_llamacpp_readiness_rejects_invalid_structured_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    narrator = LlamaCppNarrator(
        model_alias="tbx-qwen3.5-4b-q4-k-m",
        model_path=str(model),
        expected_model_sha256=_digest(model),
        expected_server_build="b10517",
        api_key="k" * 48,
    )
    monkeypatch.setattr(narrator, "provenance", lambda: {"model": narrator.model})
    monkeypatch.setattr(
        narrator.client, "complete_json", lambda **_kwargs: '{"status":"not-ready"}'
    )

    with pytest.raises(NarrationError, match="invalid value"):
        narrator.probe_generation()


def test_circuit_breaker_opens_after_bounded_failures(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)

    def fail(*args, **kwargs):
        raise OSError("simulated runtime failure")

    monkeypatch.setattr(client._opener, "open", fail)
    for _ in range(3):
        with pytest.raises(LlamaCppError):
            client._request("/health")
    with pytest.raises(LlamaCppCircuitOpenError):
        client._request("/health")


def test_chat_failures_open_circuit_even_when_health_and_models_succeed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"gguf-test")
    client = _client(model)
    chat_calls = 0

    def request(path: str, *, payload=None):
        nonlocal chat_calls
        client._before_request()
        if path == "/health":
            return {"status": "ok"}
        if path == "/v1/models":
            return {"data": [{"id": client.model_alias}]}
        chat_calls += 1
        client._record_failure()
        raise LlamaCppError("simulated chat failure")

    monkeypatch.setattr(client, "_request", request)
    kwargs = {
        "messages": [{"role": "user", "content": "probe"}],
        "json_schema": {"type": "object"},
        "schema_name": "probe",
        "max_tokens": 8,
        "seed": 1,
    }

    for _ in range(3):
        with pytest.raises(LlamaCppError, match="chat failure"):
            client.complete_json(**kwargs)
    with pytest.raises(LlamaCppCircuitOpenError):
        client.complete_json(**kwargs)
    assert chat_calls == 3


def test_runtime_contract_verifies_both_assets_and_builds_shell_free_argv(tmp_path: Path):
    binary = tmp_path / "llama-server.exe"
    model = tmp_path / "qwen.gguf"
    api_key_file = tmp_path / "llama-api-keys.txt"
    bundle_manifest = tmp_path / "bundle-manifest.json"
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    api_key_file.write_text("k" * 48 + "\n", encoding="utf-8")
    bundle_manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "relative_path": binary.name,
                        "size_bytes": binary.stat().st_size,
                        "sha256": _digest(binary),
                    }
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = LlamaCppRuntimeConfig(
        schema_version=1,
        runtime_id="qwen35-4b-q4-k-m-llamacpp-b10517",
        engine="llama.cpp",
        server_build="b10517",
        release_archive_sha256="a" * 64,
        binary_path=binary,
        binary_sha256=_digest(binary),
        bundle_manifest_path=bundle_manifest,
        bundle_manifest_sha256=_digest(bundle_manifest),
        api_key_file=api_key_file,
        model_id="Qwen/Qwen3.5-4B",
        model_alias="tbx-qwen3.5-4b-q4-k-m",
        model_path=model,
        model_sha256=_digest(model),
        quantization="Q4_K_M",
        allowed_roles=["narrator", "evidence_composer"],
        clinical_authority=False,
        load_mmproj=False,
    )
    attestation = verify_runtime_assets(config)
    command = build_server_command(config)
    assert attestation["binary_sha256"] == _digest(binary)
    assert attestation["verified_bundle_files"] == 1
    assert attestation["model_sha256"] == _digest(model)
    assert command[0] == str(binary)
    assert "--no-ui" in command
    assert "--metrics" in command
    assert "--jinja" in command
    assert "--fit-target" in command
    assert "--alias" in command
    assert "--no-mmproj" in command
    assert "--no-slots" in command
    assert "--api-key-file" in command
    assert command[command.index("-fa") + 1] == "on"
    assert command[command.index("-t") + 1] == "8"
    assert command[command.index("-tb") + 1] == "12"
    assert command[command.index("--reasoning") + 1] == "off"
    assert "--chat-template-kwargs" not in command


def test_llamacpp_key_file_is_required_only_for_enabled_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_API_KEY_FILE", raising=False)
    missing = tmp_path / "missing.keys"
    assert (
        _load_llama_cpp_api_key(
            narrator_backend="none",
            configured_file=str(missing),
            project_root=tmp_path,
        )
        == ""
    )
    with pytest.raises(ValueError, match="unreadable"):
        _load_llama_cpp_api_key(
            narrator_backend="llama_cpp",
            configured_file=str(missing),
            project_root=tmp_path,
        )


def test_llamacpp_key_file_accepts_one_strong_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LLAMA_CPP_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_CPP_API_KEY_FILE", raising=False)
    key_file = tmp_path / "llama.keys"
    key_file.write_text("# managed secret\n" + "k" * 48 + "\n", encoding="utf-8")
    assert (
        _load_llama_cpp_api_key(
            narrator_backend="llama_cpp",
            configured_file=str(key_file),
            project_root=tmp_path,
        )
        == "k" * 48
    )
