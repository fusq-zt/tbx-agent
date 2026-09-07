from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from tbx_agent.llm.tool_calling import NativeToolCallError, NativeToolFailureCode
from tbx_agent.narrator import (
    NarrationError,
    NarrationRejectedError,
    OpenAICompatibleNarrator,
    normalize_openai_compatible_base_url,
)
from tbx_agent.schemas import AgentResponse, NarrationStatus, ResponseKind, Urgency

SECRET = "test-provider-secret-must-not-leak"


def _authoritative_response(*, urgency: Urgency | None = Urgency.PROMPT_EVALUATION):
    return AgentResponse(
        request_id="private-request",
        trace_id="private-trace",
        thread_id="private-thread",
        case_id="private-case",
        response_kind=ResponseKind.NEXT_TEST_INFORMATION,
        summary="模型未识别为结核训练类别。",
        diagnostic_information=["持续咳嗽需要结合病史和进一步检查评估。"],
        limitations=["本系统不用于确诊或排除肺结核。"],
        urgency=urgency,
    )


class _FakeCompletions:
    def __init__(self, *, content: str, error: Exception | None = None, usage: Any = None):
        self.content = content
        self.error = error
        self.usage = usage or SimpleNamespace(prompt_tokens=37, completion_tokens=9)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.content),
                )
            ],
            usage=self.usage,
        )


class _FakeClient:
    def __init__(self, completions: _FakeCompletions):
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _narrator(
    *,
    content: str,
    error: Exception | None = None,
    usage: Any = None,
    base_url: str = "https://llm.example.test/v1",
):
    completions = _FakeCompletions(content=content, error=error, usage=usage)
    client = _FakeClient(completions)
    factory_arguments: dict[str, Any] = {}

    def factory(**kwargs):
        factory_arguments.update(kwargs)
        return client

    narrator = OpenAICompatibleNarrator(
        base_url=base_url,
        model="qwen-compatible",
        api_key=SECRET,
        timeout_seconds=7,
        max_response_bytes=4096,
        client_factory=factory,
    )
    return narrator, completions, client, factory_arguments


@pytest.mark.parametrize(
    "base_url",
    [
        "http://llm.example.test/v1",
        "https://user:password@llm.example.test/v1",
        "https://llm.example.test/v1?token=private",
        "https://llm.example.test/v1#fragment",
        "https://llm.example.test/v1\r\nX-Injected: yes",
        "https://llm.example.test/v1%0d%0aX-Injected",
    ],
)
def test_openai_compatible_url_rejects_unsafe_inputs(base_url: str) -> None:
    with pytest.raises(ValueError):
        normalize_openai_compatible_base_url(base_url)


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://127.0.0.1:8001/v1/", "http://127.0.0.1:8001/v1"),
        ("http://[::1]:8001/v1", "http://[::1]:8001/v1"),
        ("http://localhost:8001/v1", "http://localhost:8001/v1"),
        ("https://llm.example.test/openai/v1/", "https://llm.example.test/openai/v1"),
    ],
)
def test_openai_compatible_url_accepts_https_and_loopback_http(
    base_url: str,
    expected: str,
) -> None:
    assert normalize_openai_compatible_base_url(base_url) == expected


def test_openai_compatible_narration_is_allowlisted_and_records_usage() -> None:
    selected = "请注意：模型未识别为结核训练类别。 持续咳嗽需要结合病史和进一步检查评估。"
    narrator, completions, client, factory_arguments = _narrator(
        content=json.dumps({"summary": selected}, ensure_ascii=False)
    )
    original = _authoritative_response()

    rendered = narrator.narrate(original)

    assert rendered.summary == selected
    assert rendered.narrator_backend == "openai_compatible"
    assert rendered.narrator_model == "qwen-compatible"
    assert rendered.narrator_model_digest is None
    assert rendered.narration_status == NarrationStatus.APPLIED
    assert rendered.narrator_generation_invoked is True
    assert rendered.narrator_prompt_tokens == 37
    assert rendered.narrator_completion_tokens == 9
    assert rendered.model_dump(
        exclude={
            "summary",
            "narrator_backend",
            "narrator_model",
            "narrator_model_digest",
            "narrator_policy_id",
            "narration_status",
            "narrator_generation_invoked",
            "narrator_prompt_tokens",
            "narrator_completion_tokens",
        }
    ) == original.model_dump(
        exclude={
            "summary",
            "narrator_backend",
            "narrator_model",
            "narrator_model_digest",
            "narrator_policy_id",
            "narration_status",
            "narrator_generation_invoked",
            "narrator_prompt_tokens",
            "narrator_completion_tokens",
        }
    )

    assert factory_arguments == {
        "api_key": SECRET,
        "base_url": "https://llm.example.test/v1",
        "timeout": 7.0,
        "max_retries": 0,
    }
    request = completions.calls[0]
    assert request["model"] == "qwen-compatible"
    assert request["stream"] is False
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}
    request_text = json.dumps(request, ensure_ascii=False)
    for private_value in (
        SECRET,
        "private-request",
        "private-trace",
        "private-thread",
        "private-case",
    ):
        assert private_value not in request_text

    narrator.close()
    assert client.closed is True


def test_openai_compatible_narrator_rejects_unapproved_generated_claim() -> None:
    narrator, _, _, _ = _narrator(
        content=json.dumps({"summary": "模型已经确诊肺结核。"}, ensure_ascii=False)
    )

    with pytest.raises(NarrationRejectedError, match="approved evidence") as caught:
        narrator.narrate(_authoritative_response())

    assert "确诊肺结核" not in str(caught.value)


def test_openai_compatible_narrator_transport_error_never_leaks_secret() -> None:
    narrator, _, _, _ = _narrator(
        content="",
        error=RuntimeError(f"Authorization: Bearer {SECRET}"),
    )

    with pytest.raises(NarrationError, match="request failed") as caught:
        narrator.narrate(_authoritative_response())

    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


def test_openai_compatible_narrator_initialization_error_never_leaks_secret() -> None:
    def failing_factory(**_kwargs):
        raise RuntimeError(f"credential={SECRET}")

    with pytest.raises(NarrationError, match="initialization failed") as caught:
        OpenAICompatibleNarrator(
            base_url="https://llm.example.test/v1",
            model="qwen-compatible",
            api_key=SECRET,
            client_factory=failing_factory,
        )

    assert SECRET not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("content", "usage"),
    [
        ("not-json", SimpleNamespace(prompt_tokens=5, completion_tokens=3)),
        (
            json.dumps({"summary": "模型未识别为结核训练类别。"}, ensure_ascii=False),
            SimpleNamespace(prompt_tokens=None, completion_tokens=None),
        ),
    ],
)
def test_openai_compatible_narrator_rejects_invalid_json_or_usage(
    content: str,
    usage: Any,
) -> None:
    narrator, _, _, _ = _narrator(content=content, usage=usage)

    with pytest.raises(NarrationError):
        narrator.narrate(_authoritative_response())


def test_openai_compatible_narrator_skips_emergency_without_remote_call() -> None:
    narrator, completions, _, _ = _narrator(content="unused")

    rendered = narrator.narrate(_authoritative_response(urgency=Urgency.EMERGENCY))

    assert rendered.narration_status == NarrationStatus.SKIPPED_EMERGENCY
    assert rendered.narrator_backend == "openai_compatible"
    assert rendered.narrator_generation_invoked is False
    assert completions.calls == []


def test_openai_compatible_api_key_validation_does_not_echo_secret() -> None:
    injected = "private-value\r\nX-Injected: yes"

    with pytest.raises(ValueError) as caught:
        OpenAICompatibleNarrator(
            base_url="https://llm.example.test/v1",
            model="qwen-compatible",
            api_key=injected,
            client_factory=lambda **_kwargs: object(),
        )

    assert "private-value" not in str(caught.value)


def test_openai_compatible_native_tool_call_uses_standard_protocol() -> None:
    narrator, completions, _, _ = _narrator(content="unused")

    def native_create(**kwargs):
        completions.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                function=SimpleNamespace(
                                    name="classify_cxr",
                                    arguments="{}",
                                )
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=4),
        )

    completions.create = native_create
    calls, content, usage = narrator.complete_tool_calls(
        messages=[{"role": "user", "content": "分析当前胸片"}],
        tools=[{"type": "function", "function": {"name": "classify_cxr"}}],
        tool_choice="auto",
        max_tokens=128,
        seed=1,
    )

    assert calls[0].function.name == "classify_cxr"
    assert content is None
    assert usage == {"prompt_tokens": 13, "completion_tokens": 4}
    request = completions.calls[0]
    assert request["tools"][0]["function"]["name"] == "classify_cxr"
    assert request["tool_choice"] == "auto"
    assert "response_format" not in request


def test_explicit_openai_native_unsupported_response_is_cached() -> None:
    class UnsupportedError(RuntimeError):
        status_code = 405

    narrator, completions, _, _ = _narrator(
        content="",
        error=UnsupportedError("provider body must not escape"),
    )
    parameters = {
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "tool_choice": "auto",
        "max_tokens": 32,
        "seed": 1,
    }

    with pytest.raises(NativeToolCallError) as first:
        narrator.complete_tool_calls(**parameters)
    with pytest.raises(NativeToolCallError) as second:
        narrator.complete_tool_calls(**parameters)

    assert first.value.reason_code == NativeToolFailureCode.PROVIDER_UNSUPPORTED
    assert second.value.reason_code == NativeToolFailureCode.CACHED_UNAVAILABLE
    assert len(completions.calls) == 1


def test_transient_openai_native_failure_is_not_cached() -> None:
    class TransientError(RuntimeError):
        status_code = 503

    narrator, completions, _, _ = _narrator(
        content="",
        error=TransientError("temporary"),
    )
    parameters = {
        "messages": [{"role": "user", "content": "test"}],
        "tools": [],
        "tool_choice": "auto",
        "max_tokens": 32,
        "seed": 1,
    }

    for _ in range(2):
        with pytest.raises(NativeToolCallError) as caught:
            narrator.complete_tool_calls(**parameters)
        assert caught.value.reason_code == NativeToolFailureCode.REQUEST_FAILED

    assert len(completions.calls) == 2
