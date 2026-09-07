from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from math import ceil

from starlette.requests import Request


@dataclass(frozen=True, slots=True)
class TrafficGuardError(Exception):
    status_code: int
    detail: str
    retry_after: int | None = None


class BackpressureGate:
    """A process-local, non-blocking bound on expensive in-flight requests."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("backpressure capacity must be positive")
        self._capacity = capacity
        self._inflight = 0
        self._lock = threading.Lock()

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def try_enter(self) -> bool:
        with self._lock:
            if self._inflight >= self._capacity:
                return False
            self._inflight += 1
            return True

    def leave(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    """Bounded, process-local limiter keyed only by pseudonymous digests."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        burst: int,
        max_keys: int = 20_000,
    ) -> None:
        if requests_per_minute < 1 or burst < 1 or max_keys < 1:
            raise ValueError("rate limiter values must be positive")
        self._rate_per_second = requests_per_minute / 60.0
        self._burst = float(burst)
        self._max_keys = max_keys
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    @property
    def retry_after_seconds(self) -> int:
        return max(1, ceil(1.0 / self._rate_per_second))

    def allow(self, key: str, *, now: float | None = None) -> bool:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self._max_keys:
                    oldest = min(self._buckets, key=lambda item: self._buckets[item].updated_at)
                    del self._buckets[oldest]
                bucket = _Bucket(tokens=self._burst, updated_at=observed_at)
                self._buckets[key] = bucket
            elapsed = max(0.0, observed_at - bucket.updated_at)
            bucket.tokens = min(
                self._burst,
                bucket.tokens + elapsed * self._rate_per_second,
            )
            bucket.updated_at = observed_at
            if bucket.tokens < 1.0:
                return False
            bucket.tokens -= 1.0
            return True


def validate_request_size(
    request: Request,
    *,
    max_bytes: int,
    require_content_length: bool,
) -> None:
    raw_content_lengths = [
        value.decode("latin-1").strip()
        for name, value in request.scope.get("headers", [])
        if name.decode("latin-1").lower() == "content-length"
    ]
    transfer_encodings = [
        value.decode("latin-1").strip()
        for name, value in request.scope.get("headers", [])
        if name.decode("latin-1").lower() == "transfer-encoding"
    ]
    if len(raw_content_lengths) > 1 or (raw_content_lengths and transfer_encodings):
        raise TrafficGuardError(400, "ambiguous request framing")
    if transfer_encodings and require_content_length:
        raise TrafficGuardError(411, "content-length required")
    if not raw_content_lengths:
        if require_content_length and request.method in {"POST", "PUT", "PATCH"}:
            raise TrafficGuardError(411, "content-length required")
        return
    try:
        content_length = int(raw_content_lengths[0])
    except ValueError as exc:
        raise TrafficGuardError(400, "invalid content-length") from exc
    if content_length < 0:
        raise TrafficGuardError(400, "invalid content-length")
    if content_length > max_bytes:
        raise TrafficGuardError(413, "request body too large")
