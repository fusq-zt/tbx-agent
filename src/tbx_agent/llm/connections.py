"""Ephemeral, identity-bound credentials for user-selected LLM endpoints.

The registry deliberately has no persistence adapter.  A raw credential lives only
inside a process-local ``SecretStr`` until it expires or is revoked; callers receive
an opaque connection identifier and must re-prove the owner/user/thread binding on
every lookup.  This keeps credentials out of case state, thread memory,
SQLite, audit events, and API response models.
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from pydantic import SecretStr

OPENAI_COMPATIBLE_PROVIDER = "openai_compatible"
_CONNECTION_ID_PREFIX = "llmc_"


class LLMConnectionError(RuntimeError):
    """Base class for safe-to-report connection registry failures."""


class LLMConnectionUnavailableError(LLMConnectionError):
    """The opaque connection is absent, expired, or already revoked."""


class LLMConnectionAccessError(LLMConnectionError):
    """The caller does not own the requested ephemeral connection."""


@dataclass(frozen=True, slots=True)
class LLMConnectionInfo:
    """Non-sensitive metadata suitable for returning to an API client."""

    connection_id: str
    provider: str
    base_url: str
    model: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ResolvedLLMConnection:
    """A short-lived lookup result whose secret remains redacted in repr/str."""

    info: LLMConnectionInfo
    api_key: SecretStr


@dataclass(frozen=True, slots=True)
class _ConnectionRecord:
    info: LLMConnectionInfo
    owner_scope: str
    user_id: str
    thread_id: str
    api_key: SecretStr
    expires_monotonic: float


class EphemeralLLMConnectionRegistry:
    """Thread-safe TTL vault for OpenAI-compatible endpoint credentials.

    The registry is intentionally process local.  Restarting the API invalidates
    every connection, which is preferable to persisting user-supplied API keys.
    ``clock`` and ``wall_clock`` are injectable solely for deterministic tests.
    """

    def __init__(
        self,
        *,
        default_ttl_seconds: float = 30 * 60,
        max_ttl_seconds: float = 24 * 60 * 60,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._validate_ttl(default_ttl_seconds, field="default_ttl_seconds")
        self._validate_ttl(max_ttl_seconds, field="max_ttl_seconds")
        if default_ttl_seconds > max_ttl_seconds:
            raise ValueError("default connection TTL must not exceed the maximum TTL")
        self.default_ttl_seconds = float(default_ttl_seconds)
        self.max_ttl_seconds = float(max_ttl_seconds)
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._records: dict[str, _ConnectionRecord] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _validate_ttl(value: float, *, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be a number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")

    @staticmethod
    def _identity_part(value: str, *, field: str, max_length: int) -> str:
        normalized = value.strip()
        if not normalized or len(normalized) > max_length:
            raise ValueError(f"{field} must be non-empty and at most {max_length} characters")
        if any(character in normalized for character in ("\r", "\n", "\x00")):
            raise ValueError(f"{field} contains a forbidden control character")
        return normalized

    @staticmethod
    def _secret(value: str | SecretStr) -> SecretStr:
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw, str):
            raise TypeError("api_key must be a string")
        if (
            not raw
            or len(raw) > 4096
            or any(character in raw for character in ("\r", "\n", "\x00"))
        ):
            raise ValueError("api_key is empty, too long, or contains a control character")
        return SecretStr(raw)

    def create(
        self,
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
        base_url: str,
        model: str,
        api_key: str | SecretStr,
        ttl_seconds: float | None = None,
    ) -> LLMConnectionInfo:
        """Create a credential record and return metadata without the secret."""

        owner = self._identity_part(owner_scope, field="owner_scope", max_length=256)
        user = self._identity_part(user_id, field="user_id", max_length=128)
        thread = self._identity_part(thread_id, field="thread_id", max_length=128)
        endpoint = self._identity_part(base_url, field="base_url", max_length=2048)
        model_id = self._identity_part(model, field="model", max_length=256)
        secret = self._secret(api_key)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        self._validate_ttl(ttl, field="ttl_seconds")
        if ttl > self.max_ttl_seconds:
            raise ValueError("ttl_seconds exceeds the configured maximum")

        now_monotonic = self._clock()
        now = self._wall_clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("wall_clock must return a timezone-aware datetime")
        connection_id = f"{_CONNECTION_ID_PREFIX}{secrets.token_urlsafe(32)}"
        info = LLMConnectionInfo(
            connection_id=connection_id,
            provider=OPENAI_COMPATIBLE_PROVIDER,
            base_url=endpoint,
            model=model_id,
            created_at=now,
            expires_at=now + timedelta(seconds=float(ttl)),
        )
        record = _ConnectionRecord(
            info=info,
            owner_scope=owner,
            user_id=user,
            thread_id=thread,
            api_key=secret,
            expires_monotonic=now_monotonic + float(ttl),
        )
        with self._lock:
            self._purge_expired_locked(now_monotonic)
            while connection_id in self._records:  # pragma: no cover - cryptographic collision.
                connection_id = f"{_CONNECTION_ID_PREFIX}{secrets.token_urlsafe(32)}"
                info = LLMConnectionInfo(
                    connection_id=connection_id,
                    provider=info.provider,
                    base_url=info.base_url,
                    model=info.model,
                    created_at=info.created_at,
                    expires_at=info.expires_at,
                )
                record = _ConnectionRecord(
                    info=info,
                    owner_scope=record.owner_scope,
                    user_id=record.user_id,
                    thread_id=record.thread_id,
                    api_key=record.api_key,
                    expires_monotonic=record.expires_monotonic,
                )
            self._records[connection_id] = record
        return info

    def resolve(
        self,
        connection_id: str,
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
    ) -> ResolvedLLMConnection:
        """Resolve one record only after checking its complete identity binding."""

        connection = self._identity_part(
            connection_id, field="connection_id", max_length=128
        )
        owner = self._identity_part(owner_scope, field="owner_scope", max_length=256)
        user = self._identity_part(user_id, field="user_id", max_length=128)
        thread = self._identity_part(thread_id, field="thread_id", max_length=128)
        with self._lock:
            now = self._clock()
            self._purge_expired_locked(now)
            record = self._records.get(connection)
            if record is None:
                raise LLMConnectionUnavailableError("LLM connection is unavailable")
            if (record.owner_scope, record.user_id, record.thread_id) != (
                owner,
                user,
                thread,
            ):
                raise LLMConnectionAccessError("LLM connection is unavailable to this identity")
            return ResolvedLLMConnection(info=record.info, api_key=record.api_key)

    def revoke(
        self,
        connection_id: str,
        *,
        owner_scope: str,
        user_id: str,
        thread_id: str,
    ) -> bool:
        """Delete an owned connection; return false when it is already unavailable."""

        connection = self._identity_part(
            connection_id, field="connection_id", max_length=128
        )
        owner = self._identity_part(owner_scope, field="owner_scope", max_length=256)
        user = self._identity_part(user_id, field="user_id", max_length=128)
        thread = self._identity_part(thread_id, field="thread_id", max_length=128)
        with self._lock:
            self._purge_expired_locked(self._clock())
            record = self._records.get(connection)
            if record is None:
                return False
            if (record.owner_scope, record.user_id, record.thread_id) != (
                owner,
                user,
                thread,
            ):
                raise LLMConnectionAccessError("LLM connection is unavailable to this identity")
            del self._records[connection]
            return True

    def clear(self) -> None:
        """Revoke all process-local connections when the host lifespan ends."""
        with self._lock:
            self._records.clear()

    def purge_expired(self) -> int:
        """Drop expired secrets and return the number removed."""

        with self._lock:
            return self._purge_expired_locked(self._clock())

    def _purge_expired_locked(self, now: float) -> int:
        expired = [
            connection_id
            for connection_id, record in self._records.items()
            if record.expires_monotonic <= now
        ]
        for connection_id in expired:
            del self._records[connection_id]
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked(self._clock())
            return len(self._records)
