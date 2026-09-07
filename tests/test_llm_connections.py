from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from tbx_agent.llm.connections import (
    EphemeralLLMConnectionRegistry,
    LLMConnectionAccessError,
    LLMConnectionUnavailableError,
)

OWNER = "tenant:user-a"
USER = "user-a"
THREAD = "thread-a"
SECRET = "test-secret-that-must-never-be-serialized"


def _registry(*, clock=None) -> EphemeralLLMConnectionRegistry:
    kwargs = {
        "default_ttl_seconds": 60,
        "max_ttl_seconds": 600,
        "wall_clock": lambda: datetime(2026, 8, 31, tzinfo=UTC),
    }
    if clock is not None:
        kwargs["clock"] = clock
    return EphemeralLLMConnectionRegistry(**kwargs)


def _create(registry: EphemeralLLMConnectionRegistry, **kwargs):
    return registry.create(
        owner_scope=kwargs.pop("owner_scope", OWNER),
        user_id=kwargs.pop("user_id", USER),
        thread_id=kwargs.pop("thread_id", THREAD),
        base_url=kwargs.pop("base_url", "https://llm.example.test/v1"),
        model=kwargs.pop("model", "qwen-test"),
        api_key=kwargs.pop("api_key", SECRET),
        **kwargs,
    )


def test_registry_returns_only_non_sensitive_metadata_and_redacts_resolved_secret() -> None:
    registry = _registry()

    info = _create(registry)
    resolved = registry.resolve(
        info.connection_id,
        owner_scope=OWNER,
        user_id=USER,
        thread_id=THREAD,
    )

    assert info.connection_id.startswith("llmc_")
    assert info.provider == "openai_compatible"
    assert info.model == "qwen-test"
    assert resolved.info == info
    assert resolved.api_key.get_secret_value() == SECRET
    assert SECRET not in repr(info)
    assert SECRET not in repr(resolved)
    assert SECRET not in str(resolved)


def test_registry_expires_records_using_monotonic_ttl() -> None:
    current = [100.0]
    registry = _registry(clock=lambda: current[0])
    info = _create(registry, ttl_seconds=2)

    current[0] = 101.999
    assert registry.resolve(
        info.connection_id,
        owner_scope=OWNER,
        user_id=USER,
        thread_id=THREAD,
    ).info == info

    current[0] = 102.0
    with pytest.raises(LLMConnectionUnavailableError, match="unavailable"):
        registry.resolve(
            info.connection_id,
            owner_scope=OWNER,
            user_id=USER,
            thread_id=THREAD,
        )
    assert len(registry) == 0


@pytest.mark.parametrize(
    ("owner_scope", "user_id", "thread_id"),
    [
        ("tenant:user-b", USER, THREAD),
        (OWNER, "user-b", THREAD),
        (OWNER, USER, "thread-b"),
    ],
)
def test_registry_rejects_cross_identity_resolution(
    owner_scope: str,
    user_id: str,
    thread_id: str,
) -> None:
    registry = _registry()
    info = _create(registry)

    with pytest.raises(LLMConnectionAccessError, match="unavailable"):
        registry.resolve(
            info.connection_id,
            owner_scope=owner_scope,
            user_id=user_id,
            thread_id=thread_id,
        )


def test_registry_revoke_is_identity_bound_and_idempotent() -> None:
    registry = _registry()
    info = _create(registry)

    with pytest.raises(LLMConnectionAccessError):
        registry.revoke(
            info.connection_id,
            owner_scope=OWNER,
            user_id=USER,
            thread_id="other-thread",
        )
    assert registry.revoke(
        info.connection_id,
        owner_scope=OWNER,
        user_id=USER,
        thread_id=THREAD,
    )
    assert not registry.revoke(
        info.connection_id,
        owner_scope=OWNER,
        user_id=USER,
        thread_id=THREAD,
    )
    with pytest.raises(LLMConnectionUnavailableError):
        registry.resolve(
            info.connection_id,
            owner_scope=OWNER,
            user_id=USER,
            thread_id=THREAD,
        )


def test_registry_is_thread_safe_and_issues_unique_opaque_ids() -> None:
    registry = _registry()

    def create_one(index: int) -> str:
        return _create(registry, thread_id=f"thread-{index}").connection_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        connection_ids = list(executor.map(create_one, range(64)))

    assert len(connection_ids) == len(set(connection_ids)) == 64
    assert len(registry) == 64


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan"), 601])
def test_registry_rejects_invalid_or_excessive_ttl(ttl: float) -> None:
    registry = _registry()

    with pytest.raises(ValueError):
        _create(registry, ttl_seconds=ttl)


def test_registry_rejects_secret_header_injection_without_echoing_it() -> None:
    registry = _registry()
    injected = "private-value\r\nX-Injected: yes"

    with pytest.raises(ValueError) as caught:
        _create(registry, api_key=injected)

    assert injected not in str(caught.value)
    assert "private-value" not in str(caught.value)
