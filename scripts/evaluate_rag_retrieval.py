#!/usr/bin/env python3
"""Evaluate fixed RAG queries across BM25, dense, and hybrid retrieval modes.

Exit codes: 0 for completed retrieval execution, 1 for per-query retrieval
failures retained in the reports, and 2 for configuration or report I/O errors.
Ranking scores remain descriptive; this command does not add a quality gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbx_agent.evaluation.run_artifacts import (  # noqa: E402
    publish_report,
    require_unrecorded_destination,
    write_snapshot,
)
from tbx_agent.experiment_recorder import ExperimentRecorder  # noqa: E402
from tbx_agent.retrieval import (  # noqa: E402
    BgeM3LocalEmbeddingAdapter,
    ContractError,
    DenseRetriever,
    EmbeddingProvenance,
    HybridRetriever,
    OpenAICompatibleEmbeddingAdapter,
    QdrantLocalConfig,
    QdrantLocalVectorStore,
    RetrievalError,
    corpus_sha256,
    evaluate_retrieval_modes,
    load_curated_corpus,
    load_evaluation_queries,
    load_index_manifest,
    load_qrels,
    load_retrieval_config,
    render_benchmark_markdown,
    validate_index_manifest,
)

_ALLOWED_MODES = {"bm25", "dense", "hybrid"}
_RUNTIME_DISTRIBUTIONS = (
    "tbx-agent",
    "FlagEmbedding",
    "qdrant-client",
    "torch",
    "transformers",
    "numpy",
    "psutil",
)


class _UnavailableRetriever:
    def __init__(self, error: RetrievalError) -> None:
        self.error = error

    def retrieve(self, _query: str, *, top_k: int, filters: Any = None):
        del top_k, filters
        raise self.error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "retrieval" / "smoke_v5" / "queries.jsonl",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "retrieval" / "smoke_v5" / "qrels.jsonl",
    )
    parser.add_argument("--knowledge-dir", type=Path, default=PROJECT_ROOT / "knowledge")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "retrieval.yaml",
    )
    parser.add_argument(
        "--suite-config",
        type=Path,
        help=(
            "Fixture config.json binding queries, qrels, corpus, seed, and retrieval config. "
            "When omitted, a sibling config.json is discovered if queries and qrels share a "
            "directory."
        ),
    )
    parser.add_argument(
        "--allow-retrieval-config-override",
        action="store_true",
        help=(
            "Explicitly authorize evaluating a retrieval config whose SHA-256 differs from "
            "the suite-bound config; the override is recorded in the report."
        ),
    )
    parser.add_argument(
        "--modes",
        default="bm25,dense,hybrid",
        help="Comma-separated subset of bm25,dense,hybrid.",
    )
    parser.add_argument(
        "--k",
        default="1,3,5,10",
        help="Comma-separated ranking cutoffs.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON report path.")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Markdown report path; defaults to the JSON path with a .md suffix.",
    )
    parser.add_argument("--model-path", type=Path, help="Explicit offline BGE-M3 snapshot.")
    parser.add_argument("--cache-dir", type=Path, help="External BGE-M3 cache directory.")
    parser.add_argument("--device", help="BGE device override; otherwise use configured env.")
    parser.add_argument(
        "--ledger", type=Path, help="Shared experiment ledger; defaults beside output."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Replace named reports after archiving old bytes; run evidence stays immutable.",
    )
    return parser


def _parse_csv(value: str, *, allowed: set[str] | None = None) -> tuple[str, ...]:
    items = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not items:
        raise ContractError("comma-separated option cannot be empty")
    if allowed is not None and not set(items).issubset(allowed):
        raise ContractError(f"unsupported values: {sorted(set(items).difference(allowed))}")
    return items


def _parse_k(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _parse_csv(value))
    except ValueError as exc:
        raise ContractError("--k must contain integers") from exc
    if any(item < 1 or item > 100 for item in values):
        raise ContractError("--k values must be in [1, 100]")
    return values


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_text(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"suite config requires non-empty {key}")
    return value.strip()


def _required_sha256(mapping: dict[str, Any], key: str) -> str:
    value = _required_text(mapping, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"suite config {key} must be a lowercase SHA-256 digest")
    return value


def _configured_path(config_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return candidate.resolve()


def _discover_suite_config(
    args: argparse.Namespace,
    *,
    queries_path: Path,
    qrels_path: Path,
) -> tuple[Path | None, str]:
    if args.suite_config is not None:
        return args.suite_config.expanduser().resolve(), "explicit"
    if queries_path.parent == qrels_path.parent:
        candidate = queries_path.parent / "config.json"
        if candidate.is_file():
            return candidate.resolve(), "auto_discovered"
    return None, "not_provided"


def _load_suite_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load suite config: {path}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ContractError("suite config must be a schema_version=1 JSON object")
    return raw


def _validate_suite_config(
    args: argparse.Namespace,
    *,
    queries_path: Path,
    qrels_path: Path,
    queries: Any,
    qrels: Any,
    knowledge_dir: Path,
    snapshot: Any,
    expected_generation: str,
    retrieval_config_path: Path,
) -> dict[str, Any]:
    suite_path, discovery = _discover_suite_config(
        args,
        queries_path=queries_path,
        qrels_path=qrels_path,
    )
    if suite_path is None:
        if args.allow_retrieval_config_override:
            raise ContractError(
                "--allow-retrieval-config-override requires a validated --suite-config"
            )
        return {
            "status": "not_provided",
            "discovery": discovery,
            "path": None,
            "sha256": None,
            "retrieval_config_override_used": False,
        }

    suite = _load_suite_config(suite_path)
    seed = suite.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ContractError("suite config seed must be a non-negative integer")
    experiment_id = _required_text(suite, "experiment_id")
    hypothesis = _required_text(suite, "hypothesis")
    purpose = _required_text(suite, "purpose")
    prohibited_uses = suite.get("prohibited_uses")
    if (
        not isinstance(prohibited_uses, list)
        or not prohibited_uses
        or any(not isinstance(item, str) or not item.strip() for item in prohibited_uses)
    ):
        raise ContractError("suite config prohibited_uses must be a non-empty string list")
    configured_queries = _configured_path(
        suite_path,
        _required_text(suite, "queries_path"),
    )
    configured_qrels = _configured_path(
        suite_path,
        _required_text(suite, "qrels_path"),
    )
    if configured_queries != queries_path:
        raise ContractError("--queries does not match the suite-bound queries_path")
    if configured_qrels != qrels_path:
        raise ContractError("--qrels does not match the suite-bound qrels_path")

    expected_queries_sha256 = _required_sha256(suite, "queries_sha256")
    expected_qrels_sha256 = _required_sha256(suite, "qrels_file_sha256")
    actual_queries_sha256 = _sha256_file(queries_path)
    actual_qrels_sha256 = _sha256_file(qrels_path)
    if actual_queries_sha256 != expected_queries_sha256:
        raise ContractError("queries SHA-256 does not match the suite config")
    if actual_qrels_sha256 != expected_qrels_sha256:
        raise ContractError("qrels SHA-256 does not match the suite config")

    configured_query_count = suite.get("query_count")
    if configured_query_count is None:
        expected_query_count = sum(
            bool(line.strip())
            for line in configured_queries.read_text(encoding="utf-8").splitlines()
        )
        query_count_source = "sha256_bound_queries_file"
    elif isinstance(configured_query_count, bool) or not isinstance(configured_query_count, int):
        raise ContractError("suite config query_count must be an integer")
    else:
        expected_query_count = configured_query_count
        query_count_source = "suite_config"
    if expected_query_count < 1 or len(queries) != expected_query_count:
        raise ContractError("query_count does not match the suite config")
    if len(qrels) != expected_query_count:
        raise ContractError("qrels count does not match the suite config query_count")

    query_identity = queries[0]
    for key, actual in (
        ("suite_id", query_identity.suite_id),
        ("suite_version", query_identity.suite_version),
        ("split_id", query_identity.split_id),
    ):
        if _required_text(suite, key) != actual:
            raise ContractError(f"{key} does not match the suite config")
    for key, actual in (
        ("answerable_query_count", sum(query.answerable for query in queries)),
        ("no_answer_query_count", sum(not query.answerable for query in queries)),
    ):
        configured = suite.get(key)
        if configured is not None and (
            isinstance(configured, bool) or not isinstance(configured, int) or configured != actual
        ):
            raise ContractError(f"{key} does not match the suite config")

    configured_knowledge_dir = _configured_path(
        suite_path,
        _required_text(suite, "knowledge_dir"),
    )
    expected_snapshot_id = _required_text(suite, "expected_snapshot_id")
    expected_source_manifest = _required_sha256(
        suite,
        "expected_source_manifest_sha256",
    )
    expected_chunks = _required_sha256(suite, "expected_chunks_sha256")
    expected_corpus_generation = _required_text(
        suite,
        "expected_corpus_generation_id",
    )
    if snapshot.snapshot_id != expected_snapshot_id:
        raise ContractError("knowledge snapshot_id does not match the suite config")
    if snapshot.source_manifest_sha256 != expected_source_manifest:
        raise ContractError("source manifest SHA-256 does not match the suite config")
    if snapshot.chunks_sha256 != expected_chunks:
        raise ContractError("chunks SHA-256 does not match the suite config")
    if expected_generation != expected_corpus_generation:
        raise ContractError("corpus generation does not match the suite config")

    bound_retrieval_config = _configured_path(
        suite_path,
        _required_text(suite, "retrieval_config_path"),
    )
    expected_retrieval_sha256 = _required_sha256(suite, "retrieval_config_sha256")
    actual_retrieval_sha256 = _sha256_file(retrieval_config_path)
    retrieval_config_matches = actual_retrieval_sha256 == expected_retrieval_sha256
    if not retrieval_config_matches and not args.allow_retrieval_config_override:
        raise ContractError(
            "retrieval config differs from the suite binding; pass "
            "--allow-retrieval-config-override for an explicit comparison"
        )
    return {
        "status": "verified",
        "discovery": discovery,
        "path": str(suite_path),
        "sha256": _sha256_file(suite_path),
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "purpose": purpose,
        "prohibited_uses": prohibited_uses,
        "seed": seed,
        "query_count": expected_query_count,
        "query_count_source": query_count_source,
        "knowledge_dir": {
            "bound_path": str(configured_knowledge_dir),
            "actual_path": str(knowledge_dir),
            "identity_verified": True,
        },
        "retrieval_config_binding": {
            "bound_path": str(bound_retrieval_config),
            "bound_sha256": expected_retrieval_sha256,
            "actual_path": str(retrieval_config_path),
            "actual_sha256": actual_retrieval_sha256,
            "content_matches": retrieval_config_matches,
        },
        "retrieval_config_override_authorized": bool(
            args.allow_retrieval_config_override
        ),
        "retrieval_config_override_used": not retrieval_config_matches,
    }


def _source_tree_manifest() -> dict[str, Any]:
    candidates = [PROJECT_ROOT / "pyproject.toml"]
    candidates.extend((PROJECT_ROOT / "src").rglob("*.py"))
    candidates.extend((PROJECT_ROOT / "scripts").rglob("*.py"))
    files = tuple(
        sorted(
            (path.resolve() for path in candidates if path.is_file()),
            key=lambda path: path.relative_to(PROJECT_ROOT).as_posix(),
        )
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(relative_path\\0file_sha256\\n)",
        "scope": ["pyproject.toml", "src/**/*.py", "scripts/**/*.py"],
        "file_count": len(files),
        "sha256": digest.hexdigest(),
    }


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _peak_rss_bytes() -> tuple[int | None, str]:
    if not sys.platform.startswith("win"):
        try:
            import resource

            maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                return maximum, "resource.ru_maxrss_bytes"
            return maximum * 1024, "resource.ru_maxrss_kib"
        except (ImportError, OSError, ValueError):
            pass
    try:
        import psutil

        memory = psutil.Process().memory_info()
        peak = getattr(memory, "peak_wset", None)
        if peak is not None:
            return int(peak), "psutil.peak_wset"
    except (ImportError, OSError, ValueError):
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class _ProcessMemoryCounters(ctypes.Structure):
                _fields_ = (
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                )

            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(_ProcessMemoryCounters),
                wintypes.DWORD,
            )
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            success = psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            if success:
                return int(counters.PeakWorkingSetSize), "win32.PeakWorkingSetSize"
        except Exception:  # noqa: BLE001 - best-effort cross-platform provenance
            pass
    return None, "unavailable"


def _cuda_peak_vram() -> dict[str, Any]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {"available": False, "reason": "torch_not_loaded"}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "reason": "cuda_unavailable"}
        devices = []
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": str(torch.cuda.get_device_name(index)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:  # noqa: BLE001 - provenance must not fail the benchmark
        return {
            "available": False,
            "reason": f"cuda_measurement_failed:{type(exc).__name__}",
        }


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "description": platform.platform(),
        },
        "dependencies": _distribution_versions(),
    }


def _governance_markdown(report: dict[str, Any]) -> str:
    suite = report["suite_validation"]
    binding = suite.get("retrieval_config_binding") or {}
    resources = report["resource_usage"]
    peak_rss = resources["process_peak_rss_bytes"]
    peak_rss_text = "unavailable" if peak_rss is None else str(peak_rss)
    lines = [
        "## Run governance and environment",
        "",
        f"- Suite validation: `{suite['status']}` ({suite['discovery']})",
        f"- Suite config SHA-256: `{suite.get('sha256') or 'not_provided'}`",
        (
            "- Retrieval config override used: "
            f"`{str(bool(suite.get('retrieval_config_override_used'))).lower()}`"
        ),
        f"- Bound retrieval config SHA-256: `{binding.get('bound_sha256', 'not_bound')}`",
        f"- Actual retrieval config SHA-256: `{binding.get('actual_sha256', 'not_bound')}`",
        (
            "- Source-tree manifest SHA-256: "
            f"`{report['source_tree_manifest']['sha256']}`"
        ),
        f"- Process peak RSS bytes: `{peak_rss_text}`",
        (
            "- CUDA peak VRAM available: "
            f"`{str(bool(resources['cuda_peak_vram']['available'])).lower()}`"
        ),
        f"- Wall time seconds: `{report['wall_time_seconds']}`",
        "",
    ]
    return "\n".join(lines)


def _embedding_adapter(runtime: Any, args: argparse.Namespace) -> Any:
    dense = runtime.dense
    provenance = EmbeddingProvenance(
        provider=dense.adapter,
        model_id=dense.model_id,
        dimensions=dense.dimensions,
        normalized=dense.normalized,
        model_sha256=dense.model_sha256,
        revision=dense.revision,
        query_prefix=dense.query_prefix,
        document_prefix=dense.document_prefix,
    )
    if dense.adapter == "bge_m3_local":
        if not dense.model_path_env or not dense.cache_dir_env or not dense.device_env:
            raise ContractError("BGE-M3 environment variable names are not configured")
        return BgeM3LocalEmbeddingAdapter(
            provenance=provenance,
            model_path_env=dense.model_path_env,
            cache_dir_env=dense.cache_dir_env,
            device_env=dense.device_env,
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            device=args.device,
            query_prefix=dense.query_prefix,
            document_prefix=dense.document_prefix,
        )
    if dense.adapter == "openai_compatible_http" and dense.endpoint:
        return OpenAICompatibleEmbeddingAdapter(
            endpoint=dense.endpoint,
            provenance=provenance,
            timeout_seconds=dense.timeout_seconds,
            require_loopback=dense.require_loopback,
            query_prefix=dense.query_prefix,
            document_prefix=dense.document_prefix,
        )
    raise ContractError("evaluation cannot instantiate the configured dense adapter")


def _build_retrievers(
    args: argparse.Namespace,
    modes: tuple[str, ...],
    snapshot: Any,
    runtime: Any,
) -> dict[str, Any]:
    retrievers: dict[str, Any] = {}
    if "bm25" in modes:
        retrievers["bm25"] = HybridRetriever(
            snapshot.documents,
            candidate_limit=runtime.engine.candidate_limit,
            sparse_minimum_score=runtime.engine.sparse_minimum_score,
        )
    requested_dense = bool({"dense", "hybrid"}.intersection(modes))
    if not requested_dense:
        return retrievers
    try:
        if not runtime.dense.enabled:
            raise ContractError("dense retrieval is disabled in the selected config")
        vector = runtime.vector_store
        if vector.backend != "qdrant_local" or vector.qdrant_path is None:
            raise ContractError("evaluation dense modes require qdrant_local")
        if not vector.qdrant_collection:
            raise ContractError("Qdrant local collection is not configured")
        embedder = _embedding_adapter(runtime, args)
        portable_manifest = load_index_manifest(vector.qdrant_path / "index-manifest.json")
        validate_index_manifest(
            portable_manifest,
            snapshot.documents,
            embedder.provenance,
            source_manifest_sha256=snapshot.source_manifest_sha256,
            chunks_sha256=snapshot.chunks_sha256,
        )
        store = QdrantLocalVectorStore(
            QdrantLocalConfig(
                root=vector.qdrant_path,
                collection=vector.qdrant_collection,
            )
        )
        dense = DenseRetriever(snapshot.documents, embedder=embedder, vector_store=store)
    except RetrievalError as exc:
        for mode in modes:
            if mode in {"dense", "hybrid"}:
                retrievers[mode] = _UnavailableRetriever(exc)
        return retrievers
    if "dense" in modes:
        retrievers["dense"] = dense
    if "hybrid" in modes:
        retrievers["hybrid"] = HybridRetriever(
            snapshot.documents,
            dense_retriever=dense,
            candidate_limit=runtime.engine.candidate_limit,
            sparse_weight=runtime.engine.sparse_weight,
            dense_weight=runtime.engine.dense_weight,
            rrf_rank_constant=runtime.engine.rrf_rank_constant,
            sparse_minimum_score=runtime.engine.sparse_minimum_score,
            fallback_to_sparse=runtime.engine.fallback_to_sparse,
        )
    return retrievers


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    wall_started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    modes = _parse_csv(args.modes, allowed=_ALLOWED_MODES)
    k_values = _parse_k(args.k)
    queries_path = args.queries.expanduser().resolve()
    qrels_path = args.qrels.expanduser().resolve()
    knowledge_dir = args.knowledge_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    queries = load_evaluation_queries(queries_path)
    qrels = load_qrels(qrels_path)
    snapshot = load_curated_corpus(knowledge_dir)
    runtime = load_retrieval_config(config_path)
    expected_generation = f"sparse-{corpus_sha256(snapshot.documents)[:24]}"
    if any(case.corpus_generation_id != expected_generation for case in qrels):
        raise ContractError("qrels corpus generation does not match the reviewed snapshot")
    suite_validation = _validate_suite_config(
        args,
        queries_path=queries_path,
        qrels_path=qrels_path,
        queries=queries,
        qrels=qrels,
        knowledge_dir=knowledge_dir,
        snapshot=snapshot,
        expected_generation=expected_generation,
        retrieval_config_path=config_path,
    )
    retrievers = _build_retrievers(args, modes, snapshot, runtime)
    report = evaluate_retrieval_modes(
        queries,
        qrels,
        retrievers,
        k_values=k_values,
    )
    source_tree_manifest = _source_tree_manifest()
    report.update(
        {
            "started_at": started_at,
            "purpose": "engineering_retrieval_comparison_not_clinical_validation",
            "requested_modes": list(modes),
            "suite_validation": suite_validation,
            "runtime_environment": _runtime_environment(),
            "source_tree_manifest": source_tree_manifest,
            "provenance": {
                "queries_path": str(queries_path),
                "queries_sha256": _sha256_file(queries_path),
                "qrels_path": str(qrels_path),
                "qrels_sha256": _sha256_file(qrels_path),
                "knowledge_snapshot_id": snapshot.snapshot_id,
                "source_manifest_sha256": snapshot.source_manifest_sha256,
                "chunks_sha256": snapshot.chunks_sha256,
                "corpus_generation_id": expected_generation,
                "retrieval_config_path": str(config_path),
                "retrieval_config_sha256": _sha256_file(config_path),
                "source_tree_manifest_sha256": source_tree_manifest["sha256"],
            },
        }
    )
    peak_rss, peak_rss_source = _peak_rss_bytes()
    report["resource_usage"] = {
        "process_peak_rss_bytes": peak_rss,
        "process_peak_rss_source": peak_rss_source,
        "cuda_peak_vram": _cuda_peak_vram(),
    }
    report["wall_time_seconds"] = round(time.perf_counter() - wall_started, 6)
    report["finished_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return report


def _write_reports(
    report: dict[str, Any],
    json_path: Path,
    markdown_path: Path,
    *,
    force: bool,
    run_dir: Path,
) -> dict[str, Any]:
    json_path = json_path.expanduser().resolve()
    markdown_path = markdown_path.expanduser().resolve()
    if json_path == markdown_path:
        raise ContractError("JSON and Markdown outputs must be different paths")
    existing = [path for path in (json_path, markdown_path) if path.exists()]
    if existing and not force:
        raise ContractError(f"refusing to overwrite report: {existing[0]}")
    markdown = render_benchmark_markdown(report).rstrip() + "\n\n" + _governance_markdown(report)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    artifacts = {
        "json_report": write_snapshot(run_dir / "report.json", payload.encode("utf-8")),
        "markdown_report": write_snapshot(run_dir / "report.md", markdown.encode("utf-8")),
    }
    for key, destination in (("json_report", json_path), ("markdown_report", markdown_path)):
        publish_report(
            Path(artifacts[key]["path"]), destination,
            force=force, archive_dir=run_dir / "previous_reports" / key,
        )
    return artifacts


def _recording_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Capture local configuration before evaluation can fail or load a model."""
    files = {}
    suite_path = args.suite_config
    if suite_path is None and args.queries.parent.resolve() == args.qrels.parent.resolve():
        candidate = args.queries.parent / "config.json"
        if candidate.is_file():
            suite_path = candidate
    suite = {}
    for name, path in (
        ("retrieval_config", args.config), ("suite_config", suite_path),
        ("queries", args.queries), ("qrels", args.qrels),
    ):
        if path is None:
            files[name] = {"available": False, "reason": "not_provided"}
            continue
        entry = {"path": str(path.expanduser().resolve())}
        try:
            raw = path.read_bytes()
            entry["sha256"] = hashlib.sha256(raw).hexdigest()
            if name in {"retrieval_config", "suite_config"}:
                entry["contents"] = raw.decode("utf-8-sig")
            if name == "suite_config":
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    suite = parsed
        except (OSError, UnicodeError, ValueError) as exc:
            entry["unavailable_reason"] = type(exc).__name__
        files[name] = entry
    source = _source_tree_manifest()
    return {
        "hypothesis": suite.get(
            "hypothesis", "Compare retrieval modes on the supplied fixed suite"
        ),
        "seed": suite.get("seed"),
        "seed_reason": None if "seed" in suite else "suite_seed_unavailable",
        "split_hash": files["queries"].get("sha256"),
        "split_hash_kind": "evaluation_queries_sha256_not_training_split",
        "configuration": {
            "arguments": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
            "input_files": files,
        },
        "source": {"revision": f"source-tree:{source['sha256']}", "manifest": source},
        "environment": _runtime_environment(),
        "peak_vram": _cuda_peak_vram(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    recorder = None
    report = None
    measurements = {}
    artifacts = None
    run_id = f"rag-{uuid.uuid4().hex}"
    try:
        markdown = args.markdown_output or args.output.with_suffix(".md")
        for destination in (args.output, markdown, args.ledger):
            if destination is not None:
                require_unrecorded_destination(destination)
        output_parent = args.output.expanduser().resolve().parent
        recorder = ExperimentRecorder(
            run_dir=output_parent / "evaluation-runs" / run_id,
            ledger_path=args.ledger or output_parent / "experiment-ledger.sqlite3",
            run_id=run_id, task="rag_retrieval", metadata=_recording_metadata(args),
        )
        if args.output.resolve() == markdown.resolve():
            raise ContractError("JSON and Markdown outputs must be different paths")
        ledger_files = {
            Path(str(recorder.ledger_path) + suffix)
            for suffix in ("", "-journal", "-wal", "-shm")
        }
        for destination in (args.output, markdown):
            resolved = destination.expanduser().resolve()
            if resolved in ledger_files or resolved.is_relative_to(recorder.run_dir):
                raise ContractError("report output cannot replace experiment records")
        if not args.force and (args.output.exists() or markdown.exists()):
            raise ContractError("refusing to overwrite existing reports; use a new output path")
        report = evaluate(args)
        report["experiment"] = {
            "run_id": run_id, "experiment_id": recorder.experiment_id,
            "run_dir": str(recorder.run_dir), "ledger_path": str(recorder.ledger_path),
        }
        mode_statuses = {mode: result["status"] for mode, result in report["modes"].items()}
        has_failures = any(status != "completed" for status in mode_statuses.values())
        measurements = {
            "metrics": {"modes": report["modes"], "suite_validation": report["suite_validation"]},
            "environment": report["runtime_environment"],
            "peak_vram": report["resource_usage"]["cuda_peak_vram"],
            "runtime_seconds": report["wall_time_seconds"],
        }
        artifacts = _write_reports(
            report, args.output, markdown, force=args.force, run_dir=recorder.run_dir,
        )
        if has_failures:
            recorder.fail(
                RuntimeError("per_mode_retrieval_failures"), artifacts=artifacts, **measurements
            )
        else:
            recorder.complete(artifacts=artifacts, **measurements)
    except BaseException as exc:
        if recorder is not None:
            try:
                if artifacts is None:
                    artifacts = {
                        key: {"path": str(path), "sha256": _sha256_file(path)}
                        for key, name in (("json_report", "report.json"),
                                          ("markdown_report", "report.md"))
                        if (path := recorder.run_dir / name).is_file()
                    }
                failure_measurements = measurements or {
                    "environment": _runtime_environment(), "peak_vram": _cuda_peak_vram(),
                }
                recorder.fail(exc, artifacts=artifacts, **failure_measurements)
            except Exception as recording_error:
                print(f"Could not finalize experiment ledger: {recording_error}", file=sys.stderr)
        print(f"RAG retrieval evaluation failed: {exc}", file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt | SystemExit):
            raise
        return 2
    summary = {
        "status": "completed_with_failures" if has_failures else "completed",
        "query_count": report["query_count"],
        "modes": mode_statuses,
        "suite_validation": report["suite_validation"]["status"],
        "retrieval_config_override_used": report["suite_validation"].get(
            "retrieval_config_override_used",
            False,
        ),
        "json_report": str(args.output.resolve()),
        "markdown_report": str(markdown.resolve()),
        "experiment": report["experiment"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
