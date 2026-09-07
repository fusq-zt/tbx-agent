from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from ..artifacts import default_artifact_root
from ..config import local_paths_overlap

if TYPE_CHECKING:
    from starlette.requests import Request

    from ..config import Settings


TENANT_HEADER = "x-tbx-tenant"
USER_HEADER = "x-tbx-user"
ACTOR_HEADER = "x-tbx-actor"
TIMESTAMP_HEADER = "x-tbx-timestamp"
NONCE_HEADER = "x-tbx-nonce"
SIGNATURE_HEADER = "x-tbx-signature"
SIGNED_HEADERS = (
    TENANT_HEADER,
    USER_HEADER,
    ACTOR_HEADER,
    TIMESTAMP_HEADER,
    NONCE_HEADER,
    SIGNATURE_HEADER,
)

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUBJECT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22,128}$")
_SIGNATURE_PATTERN = re.compile(r"^sha256=([0-9a-fA-F]{64})$")
_PLACEHOLDER_SECRETS = {
    "change-me",
    "changeme",
    "secret",
    "development",
    "replace-with-a-random-secret",
}


class AuthenticationError(Exception):
    """A safe-to-report authentication failure without claim or secret values."""

    def __init__(self, code: str = "trusted_proxy_authentication_failed") -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class TrustedIdentity:
    tenant_id: str
    user_id: str
    actor_id: str

    @property
    def owner_scope(self) -> str:
        return f"tenant:{self.tenant_id}"

    @property
    def metric_principal(self) -> str:
        material = f"{self.tenant_id}\0{self.user_id}\0{self.actor_id}".encode()
        return hashlib.sha256(material).hexdigest()


def _canonical_message(
    *,
    method: str,
    path: str,
    tenant_id: str,
    user_id: str,
    actor_id: str,
    timestamp: str,
    nonce: str,
) -> bytes:
    fields = (
        "TBX-HMAC-V1",
        method.upper(),
        path,
        timestamp,
        nonce,
        tenant_id,
        user_id,
        actor_id,
    )
    return "\n".join(fields).encode("utf-8")


def build_signed_proxy_headers(
    *,
    secret: str,
    method: str,
    path: str,
    tenant_id: str,
    user_id: str,
    actor_id: str,
    timestamp: int,
    nonce: str,
) -> dict[str, str]:
    """Build the exact trusted-proxy header contract.

    This helper is intended for the reverse-proxy integration test and does not
    make direct clients trusted. The secret must stay on the proxy and API host.
    """

    timestamp_text = str(timestamp)
    message = _canonical_message(
        method=method,
        path=path,
        tenant_id=tenant_id,
        user_id=user_id,
        actor_id=actor_id,
        timestamp=timestamp_text,
        nonce=nonce,
    )
    digest = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return {
        "X-TBX-Tenant": tenant_id,
        "X-TBX-User": user_id,
        "X-TBX-Actor": actor_id,
        "X-TBX-Timestamp": timestamp_text,
        "X-TBX-Nonce": nonce,
        "X-TBX-Signature": f"sha256={digest}",
    }


class _ReplayCache:
    def __init__(self, *, max_entries: int = 50_000) -> None:
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def accept_once(self, key: str, *, now: float, expires_at: float) -> bool:
        with self._lock:
            while self._entries:
                _, expiry = next(iter(self._entries.items()))
                if expiry >= now:
                    break
                self._entries.popitem(last=False)
            if key in self._entries:
                return False
            self._entries[key] = expires_at
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return True


class ProxyHMACAuthenticator:
    """Authenticate identity asserted by a separately trusted reverse proxy."""

    def __init__(
        self,
        *,
        secret: str,
        replay_window_seconds: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._secret = secret.encode("utf-8")
        self._window = replay_window_seconds
        self._clock = clock
        self._replay_cache = _ReplayCache()

    @staticmethod
    def _unique_headers(request: Request) -> dict[str, str]:
        values: dict[str, list[str]] = {name: [] for name in SIGNED_HEADERS}
        for raw_name, raw_value in request.scope.get("headers", []):
            name = raw_name.decode("latin-1").lower()
            if name in values:
                values[name].append(raw_value.decode("latin-1").strip())
        if any(len(items) != 1 for items in values.values()):
            raise AuthenticationError()
        return {name: items[0] for name, items in values.items()}

    def authenticate(self, request: Request) -> TrustedIdentity:
        headers = self._unique_headers(request)
        tenant_id = headers[TENANT_HEADER]
        user_id = headers[USER_HEADER]
        actor_id = headers[ACTOR_HEADER]
        timestamp_text = headers[TIMESTAMP_HEADER]
        nonce = headers[NONCE_HEADER]
        signature_match = _SIGNATURE_PATTERN.fullmatch(headers[SIGNATURE_HEADER])
        if (
            not _TENANT_PATTERN.fullmatch(tenant_id)
            or not _SUBJECT_PATTERN.fullmatch(user_id)
            or not _SUBJECT_PATTERN.fullmatch(actor_id)
            or not _NONCE_PATTERN.fullmatch(nonce)
            or signature_match is None
        ):
            raise AuthenticationError()
        try:
            timestamp = int(timestamp_text)
        except ValueError as exc:
            raise AuthenticationError() from exc
        now = self._clock()
        if abs(now - timestamp) > self._window:
            raise AuthenticationError()

        path = f"{request.scope.get('root_path', '')}{request.scope.get('path', '')}"
        raw_query = request.scope.get("query_string", b"")
        request_target = f"{path}?{raw_query.decode('latin-1')}" if raw_query else path
        expected = hmac.new(
            self._secret,
            _canonical_message(
                method=request.method,
                path=request_target,
                tenant_id=tenant_id,
                user_id=user_id,
                actor_id=actor_id,
                timestamp=timestamp_text,
                nonce=nonce,
            ),
            hashlib.sha256,
        ).hexdigest()
        supplied = signature_match.group(1).lower()
        if not hmac.compare_digest(expected, supplied):
            raise AuthenticationError()

        replay_key = hashlib.sha256(f"{tenant_id}\0{nonce}\0{supplied}".encode()).hexdigest()
        if not self._replay_cache.accept_once(
            replay_key,
            now=now,
            expires_at=now + self._window,
        ):
            raise AuthenticationError()
        return TrustedIdentity(
            tenant_id=tenant_id,
            user_id=user_id,
            actor_id=actor_id,
        )


def production_blockers(settings: Settings) -> tuple[str, ...]:
    """Return stable blocker codes; never return configured secret values."""

    blockers: list[str] = []
    if settings.deployment_profile not in {"research", "development", "production"}:
        return ("deployment_profile_invalid",)
    if settings.deployment_profile != "production":
        return ()
    if local_paths_overlap(
        settings.case_artifact_root,
        default_artifact_root(),
    ):
        blockers.append("case_artifact_root_overlaps_model_cache")
    if not settings.require_real_inference:
        blockers.append("production_requires_real_inference")
    if settings.vision_backend != "rank03":
        blockers.append("production_requires_rank03_backend")
    if not settings.require_llm_inference:
        blockers.append("production_requires_local_llm_inference")
    if settings.narrator_backend != "llama_cpp":
        blockers.append("production_requires_llama_cpp_backend")
    llama_origin = urlparse(settings.llama_cpp_base_url)
    if settings.llama_cpp_allow_remote or llama_origin.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        blockers.append("production_requires_loopback_llama_cpp")
    if settings.retain_uploaded_image:
        blockers.append("production_uploaded_image_retention_forbidden")
    if not settings.trusted_proxy_auth_enabled:
        blockers.append("trusted_proxy_auth_disabled")
    secret = settings.trusted_proxy_hmac_secret
    if len(secret.encode("utf-8")) < 32 or secret.strip().lower() in _PLACEHOLDER_SECRETS:
        blockers.append("trusted_proxy_hmac_secret_missing_or_weak")
    if not 5 <= settings.trusted_proxy_replay_window_seconds <= 300:
        blockers.append("trusted_proxy_replay_window_out_of_range")
    if settings.max_request_body_bytes <= settings.max_upload_bytes:
        blockers.append("request_body_limit_must_exceed_upload_limit")
    if settings.max_concurrent_requests < 1:
        blockers.append("max_concurrent_requests_invalid")
    if settings.rate_limit_requests_per_minute < 1 or settings.rate_limit_burst < 1:
        blockers.append("rate_limit_invalid")
    if (
        settings.metrics_enabled
        and not settings.metrics_allow_loopback
        and len(settings.metrics_admin_token.encode("utf-8")) < 32
    ):
        blockers.append("metrics_admin_guard_missing")
    return tuple(blockers)


def bearer_token_matches(authorization: str | None, expected_token: str) -> bool:
    if not authorization or not expected_token:
        return False
    scheme, separator, supplied = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected_token.encode("utf-8"))


def is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    import ipaddress

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def pseudonymous_client_key(parts: Mapping[str, str | None]) -> str:
    material = "\0".join(f"{key}={value or ''}" for key, value in sorted(parts.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
