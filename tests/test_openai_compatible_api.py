from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from tbx_agent.llm import __main__ as llm_main
from tbx_agent.llm.openai_compat import (
    EXPECTED_CONTENT,
    FIXED_SYSTEM_PROMPT,
    FIXED_USER_PROMPT,
    _parse_args,
    default_output_root,
    run_openai_compat_smoke,
)
from tbx_agent.llm.runtime_supervisor import load_runtime_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALIAS = "tbx-medgemma-1.5-4b-it-q4-k-m"
SERVER_BUILD = "b10517"
SECRET = "s" * 48
FIXED_TIME = datetime(2026, 8, 29, 8, 30, 0, tzinfo=UTC)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=21, completion_tokens=5, total_tokens=26)


def _sync_response(*, alias: str = ALIAS, content: str = EXPECTED_CONTENT) -> SimpleNamespace:
    return SimpleNamespace(
        model=alias,
        system_fingerprint=f"{SERVER_BUILD}-test",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=content,
                    reasoning_content=None,
                ),
            )
        ],
        usage=_usage(),
    )


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self) -> None:
        self.closed = True


def _stream(*, alias: str = ALIAS, include_usage: bool = True) -> FakeStream:
    midpoint = len(EXPECTED_CONTENT) // 2
    chunks = [
        SimpleNamespace(
            model=alias,
            system_fingerprint=f"{SERVER_BUILD}-test",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(role="assistant", content=None, reasoning_content=None),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model=alias,
            system_fingerprint=f"{SERVER_BUILD}-test",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=EXPECTED_CONTENT[:midpoint], reasoning_content=None
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model=alias,
            system_fingerprint=f"{SERVER_BUILD}-test",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=EXPECTED_CONTENT[midpoint:], reasoning_content=None
                    ),
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            model=alias,
            system_fingerprint=f"{SERVER_BUILD}-test",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(content=None, reasoning_content=None),
                )
            ],
            usage=None,
        ),
    ]
    if include_usage:
        chunks.append(
            SimpleNamespace(
                model=alias,
                system_fingerprint=f"{SERVER_BUILD}-test",
                choices=[],
                usage=_usage(),
            )
        )
    return FakeStream(chunks)


class FakeCompletions:
    def __init__(self, *, sync_response: Any, stream: FakeStream) -> None:
        self.sync_response = sync_response
        self.stream = stream
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.stream if kwargs["stream"] else self.sync_response


class FakeClient:
    def __init__(
        self,
        *,
        model_ids: list[str] | None = None,
        sync_response: Any | None = None,
        stream: FakeStream | None = None,
    ) -> None:
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model_id) for model_id in (model_ids or [ALIAS])]
            )
        )
        self.completions = FakeCompletions(
            sync_response=sync_response or _sync_response(),
            stream=stream or _stream(),
        )
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, client: FakeClient) -> None:
        self.client = client
        self.kwargs: dict[str, Any] | None = None

    def __call__(self, **kwargs: Any) -> FakeClient:
        self.kwargs = kwargs
        return self.client


class FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeHttpFactory:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.client = FakeHttpClient()

    def __call__(self, **kwargs: Any) -> FakeHttpClient:
        self.kwargs = kwargs
        return self.client


def _runtime_config(tmp_path: Path) -> Path:
    key_file = tmp_path / "runtime.keys"
    key_file.write_text(f"# test-only ACL\n{SECRET}\n", encoding="utf-8")
    payload = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "llm_runtime.yaml").read_text(encoding="utf-8")
    )
    payload["api_key_file"] = str(key_file.resolve())
    config_path = tmp_path / "llm_runtime.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _run(tmp_path: Path, client: FakeClient):
    factory = FakeFactory(client)
    http_factory = FakeHttpFactory()
    outcome = run_openai_compat_smoke(
        config_path=_runtime_config(tmp_path),
        output_root=tmp_path / "results",
        client_factory=factory,
        http_client_factory=http_factory,
        created_at=FIXED_TIME,
    )
    return outcome, factory, http_factory


def test_openai_compat_success_records_bounded_secret_free_receipt(tmp_path: Path):
    client = FakeClient()
    outcome, factory, http_factory = _run(tmp_path, client)

    assert outcome.passed
    assert outcome.receipt["status"] == "passed_openai_compat_smoke"
    assert outcome.receipt["split_hash"] is None
    assert outcome.receipt["peak_vram_bytes"] is None
    assert outcome.receipt["vram_measurement"]["available"] is False
    assert outcome.receipt["metrics"]["expected_alias_present"] is True
    assert outcome.receipt["metrics"]["chat_sync_exact_content"] is True
    assert outcome.receipt["metrics"]["chat_stream_exact_content"] is True
    assert outcome.receipt["metrics"]["stream_usage_chunk_count"] == 1
    assert factory.kwargs is not None
    assert factory.kwargs["api_key"] == SECRET
    assert factory.kwargs["base_url"] == "http://127.0.0.1:11435/v1"
    assert factory.kwargs["timeout"] == 120.0
    assert factory.kwargs["max_retries"] == 0
    assert factory.kwargs["http_client"] is http_factory.client
    assert http_factory.kwargs == {
        "trust_env": False,
        "follow_redirects": False,
        "timeout": 120.0,
    }
    assert client.closed is True
    assert http_factory.client.closed is True
    assert client.completions.stream.closed is True
    assert len(client.completions.calls) == 2
    sync_call, stream_call = client.completions.calls
    assert sync_call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sync_call["response_format"]["type"] == "json_schema"
    assert (
        sync_call["response_format"]["json_schema"]["schema"]["properties"]["status"][
            "const"
        ]
        == "OPENAI_COMPAT_OK"
    )
    assert sync_call["seed"] == 20260829
    assert sync_call["max_tokens"] == 64
    assert sync_call["stream"] is False
    assert stream_call["stream"] is True
    assert stream_call["max_tokens"] == 64
    assert stream_call["stream_options"] == {"include_usage": True}
    serialized = outcome.result_path.read_text(encoding="utf-8")
    assert SECRET not in serialized
    assert FIXED_SYSTEM_PROMPT not in serialized
    assert FIXED_USER_PROMPT not in serialized
    assert hashlib.sha256(outcome.result_path.read_bytes()).hexdigest() == outcome.result_sha256


def test_openai_compat_retains_alias_drift_without_calling_chat(tmp_path: Path):
    client = FakeClient(model_ids=["drifted-model-alias"])
    outcome, _, _ = _run(tmp_path, client)

    assert outcome.passed is False
    assert outcome.receipt["status"] == "failed_retained"
    assert outcome.receipt["failure"]["stage"] == "models_list"
    assert outcome.receipt["failure"]["reason"] == "models_alias_mismatch"
    assert outcome.receipt["metrics"]["expected_alias_present"] is False
    assert client.completions.calls == []


def test_openai_compat_hashes_unexpected_content_instead_of_persisting_it(tmp_path: Path):
    unexpected = "DO_NOT_PERSIST_THIS_RESPONSE"
    client = FakeClient(sync_response=_sync_response(content=unexpected))
    outcome, _, _ = _run(tmp_path, client)

    assert outcome.receipt["status"] == "failed_retained"
    assert outcome.receipt["failure"]["reason"] == "sync_content_mismatch"
    serialized = json.dumps(outcome.receipt, ensure_ascii=False)
    assert unexpected not in serialized
    assert outcome.receipt["failure"]["safe_details"]["observed_content_sha256"] == (
        hashlib.sha256(unexpected.encode("utf-8")).hexdigest()
    )


def test_openai_compat_requires_one_usage_only_stream_chunk(tmp_path: Path):
    client = FakeClient(stream=_stream(include_usage=False))
    outcome, _, _ = _run(tmp_path, client)

    assert outcome.receipt["status"] == "failed_retained"
    assert outcome.receipt["failure"]["stage"] == "chat_stream"
    assert outcome.receipt["failure"]["reason"] == "stream_usage_chunk_count_mismatch"
    assert outcome.receipt["failure"]["safe_details"]["observed_count"] == 0


def test_failed_receipt_withholds_secret_prompt_and_raw_exception(tmp_path: Path):
    class FailingFactory:
        def __call__(self, **kwargs: Any):
            raise RuntimeError(f"{kwargs['api_key']} {FIXED_USER_PROMPT} raw transport detail")

    outcome = run_openai_compat_smoke(
        config_path=_runtime_config(tmp_path),
        output_root=tmp_path / "results",
        client_factory=FailingFactory(),
        http_client_factory=FakeHttpFactory(),
        created_at=FIXED_TIME,
    )
    serialized = outcome.result_path.read_text(encoding="utf-8")

    assert outcome.receipt["status"] == "failed_retained"
    assert outcome.receipt["failure"]["stage"] == "construct_openai_client"
    assert SECRET not in serialized
    assert FIXED_USER_PROMPT not in serialized
    assert "raw transport detail" not in serialized


def test_openai_compat_disables_environment_proxy_and_redirects(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://untrusted-proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted-proxy.invalid:8080")
    outcome, _, http_factory = _run(tmp_path, FakeClient())

    assert outcome.passed
    assert http_factory.kwargs is not None
    assert http_factory.kwargs["trust_env"] is False
    assert http_factory.kwargs["follow_redirects"] is False
    assert outcome.receipt["full_configuration"]["http_client_trust_env"] is False


def test_llm_info_exposes_sdk_base_url_and_supported_endpoints(tmp_path: Path, monkeypatch, capsys):
    config_path = _runtime_config(tmp_path)
    before = load_runtime_config(config_path).canonical_sha256()
    monkeypatch.setattr(
        "sys.argv",
        ["python -m tbx_agent.llm", "info", "--config", str(config_path)],
    )

    assert llm_main.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["openai_base_url"] == "http://127.0.0.1:11435/v1"
    assert payload["openai_compatible_endpoints"] == [
        "/v1/models",
        "/v1/chat/completions",
    ]
    assert "/v1/chat/completions:sse" in payload["openai_compat_smoke_endpoints"]
    assert payload["runtime_config_sha256"] == before
    assert load_runtime_config(config_path).canonical_sha256() == before


def test_openai_compat_cli_accepts_reserved_runtime_config_spelling(tmp_path: Path):
    runtime_config = tmp_path / "runtime.yaml"
    args = _parse_args(["--runtime-config", str(runtime_config), "--output-root", str(tmp_path)])
    assert args.config == runtime_config
    assert args.output_root == tmp_path


def test_openai_compat_output_root_uses_portable_runtime_and_dedicated_override(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    assert default_output_root({"TBX_AGENT_DATA_ROOT": str(runtime_root)}) == (
        runtime_root / "evaluation_runs" / "openai-compat"
    )

    dedicated = tmp_path / "compat-results"
    assert default_output_root(
        {
            "TBX_AGENT_DATA_ROOT": str(runtime_root),
            "TBX_AGENT_OPENAI_COMPAT_OUTPUT_ROOT": str(dedicated),
        }
    ) == dedicated


def test_openai_compat_cli_defers_default_output_root_to_runtime_environment() -> None:
    args = _parse_args([])
    assert args.output_root is None
