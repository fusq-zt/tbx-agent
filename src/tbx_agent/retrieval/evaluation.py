from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ContractError


@dataclass(frozen=True, slots=True)
class QrelCase:
    schema_version: int
    suite_id: str
    suite_version: str
    corpus_generation_id: str
    query_id: str
    relevance: dict[str, int]
    relevant_source_ids: tuple[str, ...]
    hard_negative_chunk_ids: tuple[str, ...]
    answerable: bool = True

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ContractError("unsupported qrels schema_version")
        if not isinstance(self.answerable, bool):
            raise ContractError("qrels answerable must be boolean")
        if self.schema_version == 1 and not self.answerable:
            raise ContractError("qrels schema v1 cases must be answerable")
        if not all(
            value.strip()
            for value in (
                self.suite_id,
                self.suite_version,
                self.corpus_generation_id,
                self.query_id,
            )
        ):
            raise ContractError("qrels identifiers must be non-empty")
        if any(value not in {1, 2, 3} for value in self.relevance.values()):
            raise ContractError("qrels relevance must contain integer grades 1..3")
        if self.answerable and not self.relevance:
            raise ContractError("answerable qrels must contain relevance judgments")
        if not self.answerable and self.relevance:
            raise ContractError("no-answer qrels cannot contain relevant chunks")
        if any(not chunk_id.strip() for chunk_id in self.relevance):
            raise ContractError("qrels relevant chunk IDs must be non-empty")
        if self.answerable and not self.relevant_source_ids:
            raise ContractError("qrels must identify at least one relevant source")
        if not self.answerable and self.relevant_source_ids:
            raise ContractError("no-answer qrels cannot contain relevant sources")
        if len(set(self.relevant_source_ids)) != len(self.relevant_source_ids):
            raise ContractError("qrels relevant_source_ids must be unique")
        if len(set(self.hard_negative_chunk_ids)) != len(self.hard_negative_chunk_ids):
            raise ContractError("qrels hard_negative_chunk_ids must be unique")
        if set(self.relevance).intersection(self.hard_negative_chunk_ids):
            raise ContractError("a chunk cannot be both relevant and a hard negative")


@dataclass(frozen=True, slots=True)
class RetrievedReference:
    chunk_id: str
    source_id: str


@dataclass(frozen=True, slots=True)
class RetrievalMetricReport:
    schema_version: int
    suite_id: str
    suite_version: str
    corpus_generation_id: str
    qrels_sha256: str
    k: int
    query_count: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    source_recall_at_k: float
    hard_negative_rejection_at_k: float | None
    hard_negative_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _qrels_sha256(cases: Sequence[QrelCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = asdict(case)
        # Preserve the frozen v1 digest and report semantics exactly.
        if case.schema_version == 1:
            payload.pop("answerable")
        digest.update(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_qrels(path: Path) -> tuple[QrelCase, ...]:
    cases: list[QrelCase] = []
    try:
        for _line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            raw["relevant_source_ids"] = tuple(raw.get("relevant_source_ids", ()))
            raw["hard_negative_chunk_ids"] = tuple(raw.get("hard_negative_chunk_ids", ()))
            if raw.get("schema_version") == 2 and "answerable" not in raw:
                raise ContractError(f"qrels v2 requires answerable at line {_line_number}")
            cases.append(QrelCase(**raw))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise ContractError(
            f"invalid qrels JSONL near line {_line_number if '_line_number' in locals() else 0}"
        ) from exc
    if not cases:
        raise ContractError("qrels file contains no cases")
    if len({case.query_id for case in cases}) != len(cases):
        raise ContractError("qrels query_id values must be unique")
    identities = {(case.suite_id, case.suite_version, case.corpus_generation_id) for case in cases}
    if len(identities) != 1:
        raise ContractError("all qrels rows must share suite and corpus versions")
    if len({case.schema_version for case in cases}) != 1:
        raise ContractError("all qrels rows must share one schema_version")
    return tuple(cases)


def evaluate_rankings(
    qrels: Sequence[QrelCase],
    rankings: Mapping[str, Sequence[RetrievedReference]],
    *,
    k: int,
) -> RetrievalMetricReport:
    if not qrels or k < 1:
        raise ContractError("evaluation requires qrels and k >= 1")
    if set(rankings) != {case.query_id for case in qrels}:
        raise ContractError("rankings must exactly cover qrels query_id values")
    identity = {(case.suite_id, case.suite_version, case.corpus_generation_id) for case in qrels}
    if len(identity) != 1:
        raise ContractError("qrels suite/corpus versions are inconsistent")
    evaluable_qrels = tuple(case for case in qrels if case.answerable)
    if not evaluable_qrels:
        raise ContractError("ranking metrics require at least one answerable qrel")
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    source_recalls: list[float] = []
    rejected_hard_negatives = 0
    hard_negative_count = 0
    for case in evaluable_qrels:
        retrieved = tuple(rankings[case.query_id][:k])
        retrieved_ids = [item.chunk_id for item in retrieved]
        if len(set(retrieved_ids)) != len(retrieved_ids):
            raise ContractError(f"ranking for {case.query_id} contains duplicate chunk_id")
        relevant_ids = set(case.relevance)
        recalls.append(len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids))
        first_relevant = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if first_relevant is None else 1.0 / first_relevant)
        gains = [case.relevance.get(chunk_id, 0) for chunk_id in retrieved_ids]
        discounted_gain = sum(
            (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(gains, start=1)
        )
        ideal = sorted(case.relevance.values(), reverse=True)[:k]
        ideal_discounted_gain = sum(
            (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1)
        )
        ndcgs.append(discounted_gain / ideal_discounted_gain)
        relevant_sources = set(case.relevant_source_ids)
        if relevant_sources:
            retrieved_sources = {item.source_id for item in retrieved}
            source_recalls.append(
                len(relevant_sources.intersection(retrieved_sources)) / len(relevant_sources)
            )
        else:
            source_recalls.append(1.0)
        for hard_negative in case.hard_negative_chunk_ids:
            hard_negative_count += 1
            rejected_hard_negatives += int(hard_negative not in retrieved_ids)
    suite_id, suite_version, corpus_generation_id = next(iter(identity))
    query_count = len(evaluable_qrels)
    return RetrievalMetricReport(
        schema_version=1,
        suite_id=suite_id,
        suite_version=suite_version,
        corpus_generation_id=corpus_generation_id,
        qrels_sha256=_qrels_sha256(qrels),
        k=k,
        query_count=query_count,
        recall_at_k=sum(recalls) / query_count,
        mrr_at_k=sum(reciprocal_ranks) / query_count,
        ndcg_at_k=sum(ndcgs) / query_count,
        source_recall_at_k=sum(source_recalls) / query_count,
        hard_negative_rejection_at_k=(
            rejected_hard_negatives / hard_negative_count if hard_negative_count else None
        ),
        hard_negative_count=hard_negative_count,
    )
