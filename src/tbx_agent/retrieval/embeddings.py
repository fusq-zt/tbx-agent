from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from .contracts import EmbeddingBatch, EmbeddingProvenance
from .errors import BackendUnavailableError, ContractError, OptionalDependencyError

EmbeddingPurpose = Literal["query", "document"]
JsonTransport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]
BgeModelLoader = Callable[..., Any]


def _with_prefixes(
    provenance: EmbeddingProvenance,
    query_prefix: str | None,
    document_prefix: str | None,
) -> EmbeddingProvenance:
    """Bind effective preprocessing, including legacy explicit adapter options."""

    return replace(
        provenance,
        query_prefix=provenance.query_prefix if query_prefix is None else query_prefix,
        document_prefix=provenance.document_prefix if document_prefix is None else document_prefix,
    )


class EmbeddingAdapter(Protocol):
    @property
    def provenance(self) -> EmbeddingProvenance: ...

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> EmbeddingBatch: ...


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
            raw = response.read(8 * 1024 * 1024 + 1)
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        raise BackendUnavailableError(
            f"embedding endpoint unavailable: {type(exc).__name__}"
        ) from exc
    try:
        if len(raw) > 8 * 1024 * 1024:
            raise BackendUnavailableError("embedding endpoint response exceeds 8 MiB")
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendUnavailableError("embedding endpoint returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise BackendUnavailableError("embedding endpoint response must be a JSON object")
    return value


def _validate_endpoint(url: str, *, require_loopback: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ContractError("embedding endpoint must be an http(s) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ContractError("embedding endpoint cannot contain credentials, query, or fragment")
    if require_loopback and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ContractError("embedding endpoint must be loopback unless explicitly approved")


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in vector)
    norm = math.sqrt(sum(value * value for value in converted))
    if not math.isfinite(norm) or norm == 0:
        raise BackendUnavailableError("embedding endpoint returned a zero/non-finite vector")
    return tuple(value / norm for value in converted)


def local_artifact_sha256(path: Path) -> str:
    """Hash one local model file or directory without trusting a sidecar manifest.

    Directory hashes bind both each relative path and its bytes.  This can be
    expensive for a large model, so adapters calculate it only at the first
    explicit embedding call, never at import or application startup.
    """

    artifact = path.expanduser().resolve()
    if artifact.is_file():
        files = (artifact,)
        root = artifact.parent
    elif artifact.is_dir():
        files = tuple(
            sorted(
                (item for item in artifact.rglob("*") if item.is_file()),
                key=lambda item: item.relative_to(artifact).as_posix(),
            )
        )
        root = artifact
    else:
        raise BackendUnavailableError(f"local embedding artifact is missing: {artifact}")
    if not files:
        raise BackendUnavailableError("local embedding artifact directory is empty")
    digest = sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        try:
            with file_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise BackendUnavailableError(
                f"cannot hash local embedding artifact: {relative}"
            ) from exc
        digest.update(b"\0")
    return digest.hexdigest()


def _default_bge_m3_loader(
    *,
    model_path: Path,
    cache_dir: Path | None,
    device: str,
    revision: str,
) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            "FlagEmbedding is not installed; install the retrieval extra"
        ) from exc
    options: dict[str, Any] = {
        "use_fp16": device.startswith("cuda"),
        "trust_remote_code": False,
        "local_files_only": True,
        "revision": revision,
    }
    if cache_dir is not None:
        options["cache_dir"] = str(cache_dir)
    if device != "auto":
        options["devices"] = [device]
    return BGEM3FlagModel(str(model_path), **options)


class BgeM3LocalEmbeddingAdapter:
    """Lazy, offline-only BGE-M3 dense embedding adapter.

    The exact local artifact tree, revision and output contract are pinned.
    Construction has no filesystem, model, torch, or FlagEmbedding side
    effects.  The optional dependency and model are touched only on the first
    explicit :meth:`embed` call.
    """

    def __init__(
        self,
        *,
        provenance: EmbeddingProvenance,
        model_path_env: str = "RAG_EMBEDDING_MODEL_PATH",
        cache_dir_env: str = "RAG_EMBEDDING_CACHE_DIR",
        device_env: str = "RAG_EMBEDDING_DEVICE",
        model_path: Path | None = None,
        cache_dir: Path | None = None,
        device: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
        model_loader: BgeModelLoader | None = None,
    ) -> None:
        if not provenance.normalized:
            raise ContractError("BGE-M3 retrieval embeddings must be normalized")
        if provenance.model_sha256 is None or provenance.revision is None:
            raise ContractError("BGE-M3 requires a pinned local artifact hash and revision")
        if not all(value.strip() for value in (model_path_env, cache_dir_env, device_env)):
            raise ContractError("BGE-M3 environment variable names must be non-empty")
        if not 1 <= batch_size <= 512 or not 1 <= max_length <= 8192:
            raise ContractError("invalid BGE-M3 batch size or maximum sequence length")
        self._provenance = _with_prefixes(provenance, query_prefix, document_prefix)
        self.model_path_env = model_path_env
        self.cache_dir_env = cache_dir_env
        self.device_env = device_env
        self._model_path_override = model_path
        self._cache_dir_override = cache_dir
        self._device_override = device
        self.query_prefix = self.provenance.query_prefix
        self.document_prefix = self.provenance.document_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self._model_loader = model_loader or _default_bge_m3_loader
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _optional_path(override: Path | None, environment_name: str) -> Path | None:
        if override is not None:
            return override.expanduser().resolve()
        value = os.environ.get(environment_name)
        return Path(value).expanduser().resolve() if value else None

    def _runtime_inputs(self) -> tuple[Path, Path | None, str]:
        model_path = self._optional_path(self._model_path_override, self.model_path_env)
        if model_path is None:
            raise BackendUnavailableError(
                f"BGE-M3 model path environment variable is unset: {self.model_path_env}"
            )
        cache_dir = self._optional_path(self._cache_dir_override, self.cache_dir_env)
        if cache_dir is not None and not cache_dir.is_dir():
            raise BackendUnavailableError(f"BGE-M3 cache directory is missing: {cache_dir}")
        device = self._device_override or os.environ.get(self.device_env)
        if not device:
            raise BackendUnavailableError(
                f"BGE-M3 device environment variable is unset: {self.device_env}"
            )
        device = device.strip().lower()
        if not (
            device in {"auto", "cpu", "mps"}
            or (device == "cuda")
            or (device.startswith("cuda:") and device[5:].isdigit())
        ):
            raise BackendUnavailableError(f"unsupported BGE-M3 device: {device}")
        return model_path, cache_dir, device

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            model_path, cache_dir, device = self._runtime_inputs()
            actual_hash = local_artifact_sha256(model_path)
            if actual_hash != self.provenance.model_sha256:
                raise BackendUnavailableError(
                    "BGE-M3 local artifact hash does not match the pinned model_sha256"
                )
            try:
                model = self._model_loader(
                    model_path=model_path,
                    cache_dir=cache_dir,
                    device=device,
                    revision=self.provenance.revision,
                )
            except (OptionalDependencyError, BackendUnavailableError):
                raise
            except Exception as exc:
                raise BackendUnavailableError(
                    f"BGE-M3 local model load failed: {type(exc).__name__}"
                ) from exc
            self._model = model
            return model

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ContractError("embedding input must contain non-empty texts")
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        prepared = [f"{prefix}{text}" for text in texts]
        model = self._load()
        try:
            encoded = model.encode(
                prepared,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            if not isinstance(encoded, Mapping) or "dense_vecs" not in encoded:
                raise TypeError("missing dense_vecs")
            raw_vectors = encoded["dense_vecs"]
            if hasattr(raw_vectors, "tolist"):
                raw_vectors = raw_vectors.tolist()
            vectors = tuple(_normalize(vector) for vector in raw_vectors)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"BGE-M3 local embedding failed: {type(exc).__name__}"
            ) from exc
        if len(vectors) != len(texts):
            raise BackendUnavailableError("BGE-M3 returned the wrong item count")
        return EmbeddingBatch(vectors=vectors, provenance=self.provenance)


class OpenAICompatibleEmbeddingAdapter:
    """Explicit HTTP adapter; it never downloads or imports a model."""

    def __init__(
        self,
        *,
        endpoint: str,
        provenance: EmbeddingProvenance,
        timeout_seconds: float = 15.0,
        api_key_env: str | None = None,
        require_loopback: bool = True,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        transport: JsonTransport | None = None,
    ) -> None:
        _validate_endpoint(endpoint, require_loopback=require_loopback)
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise ContractError("embedding timeout must be in (0, 120] seconds")
        if not provenance.normalized:
            raise ContractError("retrieval embeddings must declare normalized=true")
        self.endpoint = endpoint.rstrip("/")
        self._provenance = _with_prefixes(provenance, query_prefix, document_prefix)
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env
        self.query_prefix = self.provenance.query_prefix
        self.document_prefix = self.provenance.document_prefix
        self._transport = transport or _default_transport

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ContractError("embedding input must contain non-empty texts")
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        prepared = [f"{prefix}{text}" for text in texts]
        headers: dict[str, str] = {}
        if self.api_key_env:
            secret = os.environ.get(self.api_key_env)
            if not secret:
                raise BackendUnavailableError(
                    f"embedding API credential environment variable is unset: {self.api_key_env}"
                )
            headers["Authorization"] = f"Bearer {secret}"
        response = self._transport(
            self.endpoint,
            {"model": self.provenance.model_id, "input": prepared, "encoding_format": "float"},
            headers,
            self.timeout_seconds,
        )
        response_model = response.get("model")
        if response_model is not None and response_model != self.provenance.model_id:
            raise BackendUnavailableError("embedding endpoint served an unexpected model_id")
        raw_data = response.get("data")
        if not isinstance(raw_data, list) or len(raw_data) != len(prepared):
            raise BackendUnavailableError("embedding endpoint returned the wrong item count")
        try:
            ordered = sorted(raw_data, key=lambda item: int(item["index"]))
            if [int(item["index"]) for item in ordered] != list(range(len(prepared))):
                raise ValueError("non-contiguous indices")
            vectors = tuple(_normalize(item["embedding"]) for item in ordered)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendUnavailableError(
                "embedding endpoint returned an invalid data schema"
            ) from exc
        return EmbeddingBatch(vectors=vectors, provenance=self.provenance)


class LocalCallableEmbeddingAdapter:
    """Boundary for a preloaded local model supplied by the deployment process."""

    def __init__(
        self,
        function: Callable[[Sequence[str], EmbeddingPurpose], Sequence[Sequence[float]]],
        *,
        provenance: EmbeddingProvenance,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        if not provenance.normalized:
            raise ContractError("retrieval embeddings must declare normalized=true")
        self.function = function
        self._provenance = _with_prefixes(provenance, query_prefix, document_prefix)
        self.query_prefix = self.provenance.query_prefix
        self.document_prefix = self.provenance.document_prefix

    @property
    def provenance(self) -> EmbeddingProvenance:
        return self._provenance

    def embed(
        self,
        texts: Sequence[str],
        *,
        purpose: EmbeddingPurpose,
    ) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ContractError("embedding input must contain non-empty texts")
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        try:
            raw_vectors = self.function([f"{prefix}{text}" for text in texts], purpose)
            vectors = tuple(_normalize(vector) for vector in raw_vectors)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise BackendUnavailableError(
                f"local embedding callable failed: {type(exc).__name__}"
            ) from exc
        if len(vectors) != len(texts):
            raise BackendUnavailableError("local embedding callable returned the wrong item count")
        return EmbeddingBatch(vectors=vectors, provenance=self.provenance)
