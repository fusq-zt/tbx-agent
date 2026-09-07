from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from ..paths import resolve_portable_path
from .engine import RetrievalEngineConfig
from .errors import ContractError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be a mapping")
    return value


def _strict_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise ContractError(f"unknown {name} keys: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class DenseRuntimeConfig:
    enabled: bool
    adapter: Literal["bge_m3_local", "openai_compatible_http", "local_callable"]
    endpoint: str | None
    model_id: str
    model_sha256: str | None
    revision: str | None
    dimensions: int
    normalized: bool
    timeout_seconds: float
    require_loopback: bool
    query_prefix: str
    document_prefix: str
    model_path_env: str | None = None
    cache_dir_env: str | None = None
    device_env: str | None = None

    def __post_init__(self) -> None:
        if self.adapter not in {
            "bge_m3_local",
            "openai_compatible_http",
            "local_callable",
        }:
            raise ContractError("unsupported dense adapter")
        if self.enabled and self.adapter == "openai_compatible_http" and not self.endpoint:
            raise ContractError("enabled HTTP dense retrieval requires endpoint")
        if self.enabled and not self.model_sha256:
            raise ContractError("enabled dense retrieval requires a pinned model_sha256")
        if self.enabled and not self.revision:
            raise ContractError("enabled dense retrieval requires a pinned revision")
        if self.model_sha256 is not None and not _SHA256_RE.fullmatch(self.model_sha256):
            raise ContractError("dense model_sha256 must be a lowercase SHA-256 digest")
        if self.dimensions < 1 or not self.normalized:
            raise ContractError("dense embeddings require positive dimensions and normalization")
        environment_names = (self.model_path_env, self.cache_dir_env, self.device_env)
        if self.adapter == "bge_m3_local" and any(not value for value in environment_names):
            raise ContractError(
                "BGE-M3 local adapter requires model/cache/device environment names"
            )


@dataclass(frozen=True, slots=True)
class VectorRuntimeConfig:
    backend: Literal["qdrant_local", "sqlite_vec", "qdrant_server"]
    qdrant_path: Path | None
    sqlite_root: Path | None
    maximum_exact_candidates: int
    qdrant_base_url: str | None
    qdrant_collection: str | None

    def __post_init__(self) -> None:
        if self.backend not in {"qdrant_local", "sqlite_vec", "qdrant_server"}:
            raise ContractError("unsupported vector-store backend")
        if self.maximum_exact_candidates < 1:
            raise ContractError("maximum_exact_candidates must be positive")


@dataclass(frozen=True, slots=True)
class RerankerRuntimeConfig:
    enabled: bool
    adapter: Literal["http", "local_callable"]
    endpoint: str | None
    model_id: str
    model_sha256: str | None
    timeout_seconds: float
    require_loopback: bool

    def __post_init__(self) -> None:
        if self.adapter not in {"http", "local_callable"}:
            raise ContractError("unsupported reranker adapter")
        if self.enabled and self.adapter == "http" and not self.endpoint:
            raise ContractError("enabled HTTP reranker requires endpoint")
        if self.enabled and not self.model_sha256:
            raise ContractError("enabled reranker requires a pinned model_sha256")
        if self.model_sha256 is not None and not _SHA256_RE.fullmatch(self.model_sha256):
            raise ContractError("reranker model_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeConfig:
    schema_version: int
    engine: RetrievalEngineConfig
    dense: DenseRuntimeConfig
    vector_store: VectorRuntimeConfig
    reranker: RerankerRuntimeConfig


def load_retrieval_config(path: Path) -> RetrievalRuntimeConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot load retrieval config: {path}") from exc
    root = _mapping(raw, "retrieval config")
    _strict_keys(root, {"schema_version", "engine", "dense", "vector_store", "reranker"}, "root")
    if root.get("schema_version") != 1:
        raise ContractError("unsupported retrieval config schema_version")

    engine_raw = _mapping(root.get("engine"), "engine")
    _strict_keys(engine_raw, set(RetrievalEngineConfig.__dataclass_fields__), "engine")
    engine = RetrievalEngineConfig(**engine_raw)

    dense_raw = _mapping(root.get("dense"), "dense")
    _strict_keys(dense_raw, set(DenseRuntimeConfig.__dataclass_fields__), "dense")
    dense = DenseRuntimeConfig(**dense_raw)

    vector_raw = _mapping(root.get("vector_store"), "vector_store")
    _strict_keys(vector_raw, set(VectorRuntimeConfig.__dataclass_fields__), "vector_store")
    sqlite_root = vector_raw.get("sqlite_root")
    qdrant_path = vector_raw.get("qdrant_path")
    try:
        vector_raw["sqlite_root"] = (
            resolve_portable_path(str(sqlite_root), base=path.parent)
            if sqlite_root
            else None
        )
        vector_raw["qdrant_path"] = (
            resolve_portable_path(str(qdrant_path), base=path.parent)
            if qdrant_path
            else None
        )
    except ValueError as exc:
        raise ContractError("sqlite_root contains an invalid runtime reference") from exc
    vector = VectorRuntimeConfig(**vector_raw)
    if vector.backend == "sqlite_vec" and vector.sqlite_root is None:
        raise ContractError("sqlite_vec backend requires sqlite_root")
    if vector.backend == "qdrant_local" and (
        vector.qdrant_path is None or not vector.qdrant_collection
    ):
        raise ContractError("qdrant_local backend requires qdrant_path and collection")
    if vector.backend == "qdrant_server" and (
        not vector.qdrant_base_url or not vector.qdrant_collection
    ):
        raise ContractError("qdrant_server backend requires base_url and collection")

    reranker_raw = _mapping(root.get("reranker"), "reranker")
    _strict_keys(reranker_raw, set(RerankerRuntimeConfig.__dataclass_fields__), "reranker")
    reranker = RerankerRuntimeConfig(**reranker_raw)
    return RetrievalRuntimeConfig(
        schema_version=1,
        engine=engine,
        dense=dense,
        vector_store=vector,
        reranker=reranker,
    )
