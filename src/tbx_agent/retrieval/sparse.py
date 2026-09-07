from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

from .contracts import RankedItem, RetrievalDocument, RetrievalFilter
from .errors import ContractError
from .tokenization import tokenize


class BM25Index:
    """Auditable BM25 baseline with deterministic tie-breaking."""

    def __init__(
        self,
        documents: Iterable[RetrievalDocument],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ContractError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        ordered = sorted(documents, key=lambda item: item.chunk_id)
        if not ordered:
            raise ContractError("BM25 index requires at least one document")
        if len({item.chunk_id for item in ordered}) != len(ordered):
            raise ContractError("duplicate chunk_id in BM25 corpus")
        self.documents = tuple(ordered)
        self._tokens = tuple(tokenize(item.embedding_text) for item in ordered)
        self._frequency = tuple(Counter(tokens) for tokens in self._tokens)
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )
        self._average_length = sum(len(tokens) for tokens in self._tokens) / len(self._tokens)

    def _score(self, query_tokens: tuple[str, ...], index: int) -> float:
        frequencies = self._frequency[index]
        length = len(self._tokens[index])
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_frequency = math.log(
                1.0 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            normalization = frequency + self.k1 * (
                1.0 - self.b + self.b * length / max(self._average_length, 1.0)
            )
            score += inverse_frequency * frequency * (self.k1 + 1.0) / normalization
        return score

    def search(
        self,
        query: str,
        *,
        filters: RetrievalFilter | None = None,
        limit: int = 20,
        minimum_score: float = 0.0,
    ) -> tuple[RankedItem, ...]:
        if not query.strip():
            return ()
        if limit < 1 or minimum_score < 0:
            raise ContractError("limit must be positive and minimum_score non-negative")
        policy = filters or RetrievalFilter()
        query_tokens = tokenize(query)
        scored = [
            (document.chunk_id, self._score(query_tokens, index))
            for index, document in enumerate(self.documents)
            if policy.admits(document)
        ]
        ranked = sorted(
            ((chunk_id, score) for chunk_id, score in scored if score > minimum_score),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
        return tuple(
            RankedItem(chunk_id=chunk_id, score=score, backend="bm25", rank=rank)
            for rank, (chunk_id, score) in enumerate(ranked, start=1)
        )
