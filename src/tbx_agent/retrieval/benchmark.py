from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contracts import MetadataFilter, RetrievedChunk
from .errors import (
    BackendUnavailableError,
    ContractError,
    GenerationMismatchError,
    OptionalDependencyError,
    StaleIndexError,
)
from .evaluation import QrelCase, RetrievedReference, evaluate_rankings

Clock = Callable[[], int]
_QUERY_V1_KEYS = {
    "schema_version",
    "suite_id",
    "suite_version",
    "split_id",
    "query_id",
    "text",
}
_QUERY_V2_KEYS = _QUERY_V1_KEYS | {"guideline_scope", "answerable"}


class ChunkRetriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedChunk]: ...


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    schema_version: int
    suite_id: str
    suite_version: str
    split_id: str
    query_id: str
    text: str
    guideline_scope: tuple[str, ...] = ()
    answerable: bool = True

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ContractError("unsupported evaluation query schema_version")
        if not isinstance(self.answerable, bool):
            raise ContractError("evaluation query answerable must be boolean")
        if not all(
            value.strip()
            for value in (
                self.suite_id,
                self.suite_version,
                self.split_id,
                self.query_id,
                self.text,
            )
        ):
            raise ContractError("evaluation query fields must be non-empty")
        if any(not isinstance(value, str) for value in self.guideline_scope):
            raise ContractError("guideline_scope must contain strings")
        normalized_scope = tuple(
            dict.fromkeys(value.strip() for value in self.guideline_scope if value.strip())
        )
        if len(normalized_scope) != len(self.guideline_scope):
            raise ContractError("guideline_scope must contain unique non-empty values")
        object.__setattr__(self, "guideline_scope", normalized_scope)
        if self.schema_version == 1 and (self.guideline_scope or not self.answerable):
            raise ContractError(
                "evaluation query schema v1 does not support scope/no-answer labels"
            )
        if self.schema_version == 2 and not self.guideline_scope:
            raise ContractError("evaluation query schema v2 requires guideline_scope")

    @property
    def query_sha256(self) -> str:
        return hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()


def load_evaluation_queries(path: Path) -> tuple[EvaluationQuery, ...]:
    rows: list[EvaluationQuery] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ContractError(f"evaluation query schema mismatch at line {line_number}")
            schema_version = raw.get("schema_version")
            expected_keys = _QUERY_V1_KEYS if schema_version == 1 else _QUERY_V2_KEYS
            if set(raw) != expected_keys:
                raise ContractError(f"evaluation query schema mismatch at line {line_number}")
            if schema_version == 2:
                if not isinstance(raw["answerable"], bool) or not isinstance(
                    raw["guideline_scope"], list
                ):
                    raise ContractError(f"evaluation query schema mismatch at line {line_number}")
                raw["guideline_scope"] = tuple(raw["guideline_scope"])
            rows.append(EvaluationQuery(**raw))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ContractError(f"cannot load evaluation queries: {path}") from exc
    if not rows or len({row.query_id for row in rows}) != len(rows):
        raise ContractError("evaluation queries must be non-empty with unique query IDs")
    identities = {(row.suite_id, row.suite_version, row.split_id) for row in rows}
    if len(identities) != 1:
        raise ContractError("evaluation queries must share suite and split identities")
    if len({row.schema_version for row in rows}) != 1:
        raise ContractError("evaluation queries must share one schema_version")
    return tuple(rows)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def classify_retrieval_failure(error: Exception) -> str:
    if isinstance(error, StaleIndexError):
        return "stale_index"
    if isinstance(error, GenerationMismatchError):
        return "generation_mismatch"
    if isinstance(error, OptionalDependencyError):
        return "optional_dependency_missing"
    if isinstance(error, BackendUnavailableError):
        return "backend_unavailable"
    if isinstance(error, ContractError):
        return "contract_violation"
    return "unexpected_error"


def _citation_hit_at_k(
    qrels: Sequence[QrelCase],
    rankings: Mapping[str, Sequence[RetrievedReference]],
    k: int,
) -> float:
    hits = 0
    for case in qrels:
        retrieved_sources = {item.source_id for item in rankings[case.query_id][:k]}
        hits += bool(retrieved_sources.intersection(case.relevant_source_ids))
    return hits / len(qrels)


def evaluate_retrieval_modes(
    queries: Sequence[EvaluationQuery],
    qrels: Sequence[QrelCase],
    retrievers: Mapping[str, ChunkRetriever],
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
    clock_ns: Clock = time.perf_counter_ns,
) -> dict[str, Any]:
    """Evaluate isolated retrievers while retaining per-mode/query failures."""

    if not retrievers:
        raise ContractError("at least one retrieval mode is required")
    normalized_k = tuple(sorted(set(int(value) for value in k_values)))
    if not normalized_k or normalized_k[0] < 1 or normalized_k[-1] > 100:
        raise ContractError("evaluation k values must be unique integers in [1, 100]")
    query_ids = {query.query_id for query in queries}
    if query_ids != {case.query_id for case in qrels}:
        raise ContractError("evaluation queries and qrels must cover identical query IDs")
    query_identity = {(item.suite_id, item.suite_version) for item in queries}
    qrel_identity = {(item.suite_id, item.suite_version) for item in qrels}
    if query_identity != qrel_identity:
        raise ContractError("evaluation queries and qrels must share suite identity")
    qrels_by_id = {case.query_id: case for case in qrels}
    for query in queries:
        case = qrels_by_id[query.query_id]
        if query.answerable != case.answerable:
            raise ContractError(f"answerable label mismatch for {query.query_id}")
    maximum_k = normalized_k[-1]
    mode_reports: dict[str, Any] = {}
    for mode, retriever in sorted(retrievers.items()):
        rankings: dict[str, tuple[RetrievedReference, ...]] = {}
        observations: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        failures: list[dict[str, str]] = []
        for query in queries:
            start = clock_ns()
            error: Exception | None = None
            hits: Sequence[RetrievedChunk] = ()
            try:
                if query.schema_version == 1:
                    hits = retriever.retrieve(query.text, top_k=maximum_k)
                else:
                    hits = retriever.retrieve(
                        query.text,
                        top_k=maximum_k,
                        filters=MetadataFilter(
                            required_claim_scopes_any=query.guideline_scope,
                        ),
                    )
                ids = [hit.chunk_id for hit in hits]
                if len(ids) != len(set(ids)):
                    raise ContractError("retriever returned duplicate chunk IDs")
                if query.schema_version == 2:
                    scope = set(query.guideline_scope)
                    leaked = [
                        hit.chunk_id
                        for hit in hits
                        if not scope.intersection(hit.document.allowed_claim_scopes)
                    ]
                    if leaked:
                        raise ContractError(
                            "retriever violated guideline_scope hard filter: " + ", ".join(leaked)
                        )
            except Exception as exc:  # noqa: BLE001 - benchmark retains backend failures
                error = exc
                hits = ()
            latency_ms = (clock_ns() - start) / 1_000_000
            latencies_ms.append(latency_ms)
            references = tuple(
                RetrievedReference(chunk_id=hit.chunk_id, source_id=hit.source_id) for hit in hits
            )
            rankings[query.query_id] = references
            failure: dict[str, str] | None = None
            if error is not None:
                failure = {
                    "category": classify_retrieval_failure(error),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
                failures.append({"query_id": query.query_id, **failure})
            elif getattr(retriever, "last_fallback_reason", None):
                failure = {
                    "category": "backend_fallback",
                    "error_type": "",
                    "message": str(retriever.last_fallback_reason),
                }
                failures.append({"query_id": query.query_id, **failure})
            elif not references and query.answerable:
                failure = {
                    "category": "empty_result",
                    "error_type": "",
                    "message": "retriever returned no chunks",
                }
                failures.append({"query_id": query.query_id, **failure})
            case = qrels_by_id[query.query_id]
            retrieved_ids = [reference.chunk_id for reference in references]
            query_errors: list[str] = []
            if error is not None:
                query_errors.append("retrieval_failure")
            elif query.answerable:
                relevant = set(case.relevance)
                matched = relevant.intersection(retrieved_ids)
                if not references:
                    query_errors.append("false_abstention")
                if not matched:
                    query_errors.append("relevant_not_retrieved_at_max_k")
                elif len(matched) != len(relevant):
                    query_errors.append("incomplete_recall_at_max_k")
                if set(case.hard_negative_chunk_ids).intersection(retrieved_ids):
                    query_errors.append("hard_negative_retrieved_at_max_k")
            elif references:
                query_errors.append("failed_no_answer_abstention")
            observations.append(
                {
                    "query_id": query.query_id,
                    "query_sha256": query.query_sha256,
                    "latency_ms": latency_ms,
                    "retrieved_chunk_ids": [reference.chunk_id for reference in references],
                    "guideline_scope": list(query.guideline_scope),
                    "answerable": query.answerable,
                    "abstained": error is None and not references,
                    "citation_source_hit": bool(
                        set(case.relevant_source_ids).intersection(
                            reference.source_id for reference in references
                        )
                    ),
                    "failure": failure,
                    "errors": query_errors,
                }
            )
        answerable_qrels = tuple(case for case in qrels if case.answerable)
        answerable_rankings = {case.query_id: rankings[case.query_id] for case in answerable_qrels}
        metrics_by_k: dict[str, Any] = {}
        for k in normalized_k:
            if not answerable_qrels:
                raise ContractError("evaluation suite must contain at least one answerable query")
            metrics = evaluate_rankings(qrels, rankings, k=k).to_dict()
            metrics_by_k[str(k)] = {
                "recall_at_k": metrics["recall_at_k"],
                "mrr_at_k": metrics["mrr_at_k"],
                "ndcg_at_k": metrics["ndcg_at_k"],
                "citation_hit_at_k": _citation_hit_at_k(answerable_qrels, answerable_rankings, k),
                "source_recall_at_k": metrics["source_recall_at_k"],
                "hard_negative_rejection_at_k": metrics["hard_negative_rejection_at_k"],
            }
        failure_counts = Counter(item["category"] for item in failures)
        no_answer_observations = [item for item in observations if not item["answerable"]]
        no_answer_correct = sum(bool(item["abstained"]) for item in no_answer_observations)
        mode_reports[mode] = {
            "status": "completed" if not failures else "completed_with_failures",
            "query_count": len(queries),
            "metrics_by_k": metrics_by_k,
            "latency_ms": {
                "p50": _percentile(latencies_ms, 0.50),
                "p95": _percentile(latencies_ms, 0.95),
                "minimum": min(latencies_ms) if latencies_ms else None,
                "maximum": max(latencies_ms) if latencies_ms else None,
            },
            "failure_counts": dict(sorted(failure_counts.items())),
            "failures": failures,
            "answerable_query_count": len(answerable_qrels),
            "no_answer_query_count": len(no_answer_observations),
            "no_answer_abstention_accuracy": (
                no_answer_correct / len(no_answer_observations) if no_answer_observations else None
            ),
            "query_errors": [
                {"query_id": item["query_id"], "errors": item["errors"]}
                for item in observations
                if item["errors"]
            ],
            "observations": observations,
        }
    return {
        "schema_version": queries[0].schema_version,
        "suite_id": queries[0].suite_id,
        "suite_version": queries[0].suite_version,
        "split_id": queries[0].split_id,
        "query_count": len(queries),
        "answerable_query_count": sum(query.answerable for query in queries),
        "no_answer_query_count": sum(not query.answerable for query in queries),
        "k_values": list(normalized_k),
        "modes": mode_reports,
    }


def render_benchmark_markdown(report: Mapping[str, Any]) -> str:
    k_values = [int(value) for value in report["k_values"]]
    maximum_k = max(k_values)
    lines = [
        "# TBX-Agent RAG retrieval evaluation",
        "",
        (
            f"Suite `{report['suite_id']}` / `{report['suite_version']}`, "
            f"split `{report['split_id']}`, queries: {report['query_count']}."
        ),
        "",
        "| Mode | Status | "
        + " | ".join(f"Recall@{k}" for k in k_values)
        + (
            f" | MRR@{maximum_k} | nDCG@{maximum_k} | Citation hit@K | "
            "No-answer abstention | p50 ms | p95 ms | Errors |"
        ),
        "|---|---:|" + "---:|" * len(k_values) + "---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in sorted(report["modes"].items()):
        metrics = result["metrics_by_k"]
        at_maximum = metrics[str(maximum_k)]
        recalls = " | ".join(f"{metrics[str(k)]['recall_at_k']:.4f}" for k in k_values)
        abstention = result.get("no_answer_abstention_accuracy")
        abstention_text = "n/a" if abstention is None else f"{abstention:.4f}"
        lines.append(
            f"| {mode} | {result['status']} | {recalls} | "
            f"{at_maximum['mrr_at_k']:.4f} | {at_maximum['ndcg_at_k']:.4f} | "
            f"{at_maximum['citation_hit_at_k']:.4f} | {abstention_text} | "
            f"{result['latency_ms']['p50'] or 0.0:.3f} | "
            f"{result['latency_ms']['p95'] or 0.0:.3f} | {len(result.get('query_errors', []))} |"
        )
    lines.extend(["", "## Failure classification", ""])
    for mode, result in sorted(report["modes"].items()):
        counts = result["failure_counts"]
        rendered = ", ".join(f"`{name}`: {count}" for name, count in counts.items())
        lines.append(f"- **{mode}**: {rendered or 'none'}")
    lines.extend(["", "## Per-query errors", ""])
    for mode, result in sorted(report["modes"].items()):
        errors = result.get("query_errors", [])
        if not errors:
            lines.append(f"- **{mode}**: none")
            continue
        for item in errors:
            lines.append(f"- **{mode} / {item['query_id']}**: " + ", ".join(item["errors"]))
    lines.extend(
        [
            "",
            "This fixed fixture is an engineering retrieval benchmark, not clinical validation.",
            "",
        ]
    )
    return "\n".join(lines)
