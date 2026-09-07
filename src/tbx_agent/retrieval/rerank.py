from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

from .contracts import RetrievalDocument
from .errors import BackendUnavailableError, ContractError

RerankTransport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class RerankScore:
    chunk_id: str
    score: float


class Reranker(Protocol):
    @property
    def fingerprint(self) -> str: ...

    def score(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
    ) -> tuple[RerankScore, ...]: ...


def _default_transport(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            raw = response.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise BackendUnavailableError("reranker endpoint response exceeds 2 MiB")
        result = json.loads(raw)
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError(
            f"reranker endpoint unavailable: {type(exc).__name__}"
        ) from exc
    if not isinstance(result, dict):
        raise BackendUnavailableError("reranker response must be a JSON object")
    return result


class BgeRerankerAdapter:
    """Adapter for an explicitly hosted bge-reranker-v2-m3 `/rerank` endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        model_sha256: str | None = None,
        timeout_seconds: float = 20.0,
        api_key_env: str | None = None,
        require_loopback: bool = True,
        transport: RerankTransport | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ContractError("reranker endpoint must be an http(s) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ContractError("reranker endpoint cannot contain credentials, query, or fragment")
        if require_loopback and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ContractError("reranker endpoint must be loopback unless explicitly approved")
        if not model_id.strip() or not 0 < timeout_seconds <= 120:
            raise ContractError("invalid reranker model_id or timeout")
        if model_sha256 is not None and (
            len(model_sha256) != 64 or any(char not in "0123456789abcdef" for char in model_sha256)
        ):
            raise ContractError("reranker model_sha256 must be a lowercase SHA-256 digest")
        self.endpoint = endpoint.rstrip("/")
        self.model_id = model_id
        self.model_sha256 = model_sha256
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env
        self._transport = transport or _default_transport

    @property
    def fingerprint(self) -> str:
        return f"{self.model_id}@{self.model_sha256 or 'UNVERIFIED'}"

    def score(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
    ) -> tuple[RerankScore, ...]:
        if not query.strip() or not documents:
            raise ContractError("reranker requires a query and at least one document")
        headers: dict[str, str] = {}
        if self.api_key_env:
            secret = os.environ.get(self.api_key_env)
            if not secret:
                raise BackendUnavailableError(
                    f"reranker credential environment variable is unset: {self.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {secret}"
        result = self._transport(
            self.endpoint,
            {
                "model": self.model_id,
                "query": query,
                "documents": [document.embedding_text for document in documents],
                "top_n": len(documents),
                "return_documents": False,
            },
            headers,
            self.timeout_seconds,
        )
        if result.get("model") not in {None, self.model_id}:
            raise BackendUnavailableError("reranker endpoint served an unexpected model_id")
        raw_results = result.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(documents):
            raise BackendUnavailableError("reranker returned the wrong item count")
        output: list[RerankScore] = []
        seen: set[int] = set()
        try:
            for item in raw_results:
                index = int(item["index"])
                score = float(item["relevance_score"])
                if index in seen or not 0 <= index < len(documents) or not math.isfinite(score):
                    raise ValueError("invalid index or score")
                seen.add(index)
                output.append(RerankScore(documents[index].chunk_id, score))
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendUnavailableError("reranker returned an invalid result schema") from exc
        return tuple(sorted(output, key=lambda item: (-item.score, item.chunk_id)))


class LocalCallableReranker:
    def __init__(
        self,
        function: Callable[[str, Sequence[str]], Sequence[float]],
        *,
        model_id: str,
        model_sha256: str | None,
    ) -> None:
        if not model_id.strip():
            raise ContractError("reranker model_id must be non-empty")
        if model_sha256 is not None and (
            len(model_sha256) != 64 or any(char not in "0123456789abcdef" for char in model_sha256)
        ):
            raise ContractError("reranker model_sha256 must be a lowercase SHA-256 digest")
        self.function = function
        self.model_id = model_id
        self.model_sha256 = model_sha256

    @property
    def fingerprint(self) -> str:
        return f"{self.model_id}@{self.model_sha256 or 'UNVERIFIED'}"

    def score(
        self,
        query: str,
        documents: Sequence[RetrievalDocument],
    ) -> tuple[RerankScore, ...]:
        if not query.strip() or not documents:
            raise ContractError("reranker requires a query and at least one document")
        try:
            scores = tuple(
                float(value)
                for value in self.function(
                    query, [document.embedding_text for document in documents]
                )
            )
        except Exception as exc:
            raise BackendUnavailableError(
                f"local reranker callable failed: {type(exc).__name__}"
            ) from exc
        if len(scores) != len(documents) or any(not math.isfinite(value) for value in scores):
            raise BackendUnavailableError("local reranker returned invalid scores")
        return tuple(
            sorted(
                (
                    RerankScore(document.chunk_id, score)
                    for document, score in zip(documents, scores, strict=True)
                ),
                key=lambda item: (-item.score, item.chunk_id),
            )
        )
