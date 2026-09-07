from __future__ import annotations

import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_SAFE_LABEL = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_LATENCY_BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


@dataclass(slots=True)
class _RouteMetric:
    count: int = 0
    errors: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_buckets: Counter[int] = field(default_factory=Counter)


class InProcessMetrics:
    """Small single-process metrics store with no request or clinical payloads."""

    def __init__(self) -> None:
        self._started_wall = datetime.now(UTC)
        self._started_monotonic = time.monotonic()
        self._totals: Counter[str] = Counter()
        self._routes: defaultdict[tuple[str, str, str], _RouteMetric] = defaultdict(_RouteMetric)
        self._fallbacks: Counter[str] = Counter()
        self._lock = threading.Lock()

    def observe_request(
        self,
        *,
        method: str,
        route_template: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        status_class = f"{status_code // 100}xx"
        key = (method.upper(), route_template, status_class)
        with self._lock:
            metric = self._routes[key]
            metric.count += 1
            metric.latency_sum_ms += latency_ms
            metric.latency_max_ms = max(metric.latency_max_ms, latency_ms)
            for boundary in _LATENCY_BUCKETS_MS:
                if latency_ms <= boundary:
                    metric.latency_buckets[boundary] += 1
            self._totals["requests"] += 1
            if status_code >= 400:
                metric.errors += 1
                self._totals["errors"] += 1

    def record_guard_event(self, event: str) -> None:
        if event not in {
            "authentication_failed",
            "identity_mismatch",
            "rate_limited",
            "backpressure_rejected",
            "body_rejected",
        }:
            raise ValueError("unsupported guard metric")
        with self._lock:
            self._totals[event] += 1

    def record_fallback(self, component: str) -> None:
        if not _SAFE_LABEL.fullmatch(component):
            raise ValueError("fallback component must be a stable low-cardinality label")
        with self._lock:
            self._fallbacks[component] += 1
            self._totals["fallbacks"] += 1

    def snapshot(self, *, inflight: int) -> dict[str, Any]:
        with self._lock:
            routes = []
            for (method, route, status_class), metric in sorted(self._routes.items()):
                routes.append(
                    {
                        "method": method,
                        "route": route,
                        "status_class": status_class,
                        "count": metric.count,
                        "errors": metric.errors,
                        "latency_sum_ms": round(metric.latency_sum_ms, 3),
                        "latency_max_ms": round(metric.latency_max_ms, 3),
                        "latency_buckets": {
                            f"le_{boundary}_ms": metric.latency_buckets[boundary]
                            for boundary in _LATENCY_BUCKETS_MS
                        },
                    }
                )
            return {
                "schema_version": "tbx-agent-inprocess-metrics-v1",
                "started_at": self._started_wall.isoformat(),
                "uptime_seconds": round(time.monotonic() - self._started_monotonic, 3),
                "totals": dict(sorted(self._totals.items())),
                "fallbacks": dict(sorted(self._fallbacks.items())),
                "gauges": {"inflight_requests": inflight},
                "routes": routes,
                "scope": "single_process",
                "contains_request_or_clinical_identifiers": False,
            }
