from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import RankedItem, RetrievalDocument
from .errors import ContractError
from .tokenization import tokenize


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    chunk_id: str
    score: float
    backend_ranks: dict[str, int]
    backend_scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    candidates: tuple[FusedCandidate, ...]
    suppressed_chunk_ids: tuple[str, ...]


def weighted_reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RankedItem]],
    *,
    weights: Mapping[str, float],
    rank_constant: int = 60,
) -> tuple[FusedCandidate, ...]:
    """Fuse ranks without assuming incomparable backend scores are calibrated."""

    if rank_constant < 1:
        raise ContractError("RRF rank_constant must be positive")
    active = {name: ranking for name, ranking in rankings.items() if ranking}
    if not active:
        return ()
    if set(active).difference(weights):
        raise ContractError("every active ranking must have an explicit RRF weight")
    if any(weights[name] <= 0 for name in active):
        raise ContractError("active RRF weights must be positive")
    accumulators: dict[str, float] = {}
    backend_ranks: dict[str, dict[str, int]] = {}
    backend_scores: dict[str, dict[str, float]] = {}
    for backend, ranking in sorted(active.items()):
        seen: set[str] = set()
        for expected_rank, item in enumerate(ranking, start=1):
            if item.backend != backend or item.rank != expected_rank:
                raise ContractError(f"ranking contract violation for backend {backend}")
            if item.chunk_id in seen:
                raise ContractError(f"backend {backend} returned duplicate chunk_id")
            seen.add(item.chunk_id)
            accumulators[item.chunk_id] = accumulators.get(item.chunk_id, 0.0) + (
                weights[backend] / (rank_constant + item.rank)
            )
            backend_ranks.setdefault(item.chunk_id, {})[backend] = item.rank
            backend_scores.setdefault(item.chunk_id, {})[backend] = item.score
    return tuple(
        FusedCandidate(
            chunk_id=chunk_id,
            score=score,
            backend_ranks=backend_ranks[chunk_id],
            backend_scores=backend_scores[chunk_id],
        )
        for chunk_id, score in sorted(
            accumulators.items(),
            key=lambda item: (
                -item[1],
                min(backend_ranks[item[0]].values()),
                item[0],
            ),
        )
    )


def _shingles(text: str, size: int = 4) -> frozenset[tuple[str, ...]]:
    tokens = tokenize(text)
    if len(tokens) < size:
        return frozenset({tokens}) if tokens else frozenset()
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def _jaccard(left: frozenset[tuple[str, ...]], right: frozenset[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    return len(left.intersection(right)) / max(1, len(left.union(right)))


def deduplicate_candidates(
    candidates: Sequence[FusedCandidate],
    documents: Mapping[str, RetrievalDocument],
    *,
    near_duplicate_threshold: float = 0.92,
    near_duplicates_across_sources: bool = False,
) -> DeduplicationResult:
    if not 0 <= near_duplicate_threshold <= 1:
        raise ContractError("near_duplicate_threshold must be in [0, 1]")
    kept: list[FusedCandidate] = []
    kept_shingles: dict[str, frozenset[tuple[str, ...]]] = {}
    suppressed: list[str] = []
    for candidate in candidates:
        document = documents.get(candidate.chunk_id)
        if document is None:
            raise ContractError(f"ranking references unknown chunk_id: {candidate.chunk_id}")
        duplicate = False
        shingles = _shingles(document.text)
        for earlier in kept:
            earlier_document = documents[earlier.chunk_id]
            if document.content_sha256 == earlier_document.content_sha256:
                duplicate = True
                break
            compare_near = near_duplicates_across_sources or (
                document.source_id == earlier_document.source_id
            )
            if compare_near and _jaccard(shingles, kept_shingles[earlier.chunk_id]) >= (
                near_duplicate_threshold
            ):
                duplicate = True
                break
        if duplicate:
            suppressed.append(candidate.chunk_id)
        else:
            kept.append(candidate)
            kept_shingles[candidate.chunk_id] = shingles
    return DeduplicationResult(tuple(kept), tuple(suppressed))
