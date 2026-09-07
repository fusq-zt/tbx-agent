from __future__ import annotations


class RetrievalError(RuntimeError):
    """Base class for retrieval subsystem failures."""


class ContractError(RetrievalError, ValueError):
    """An input violates a retrieval contract or integrity invariant."""


class OptionalDependencyError(RetrievalError):
    """An explicitly configured optional backend is not installed."""


class BackendUnavailableError(RetrievalError):
    """A configured local backend did not respond with a valid result."""


class GenerationMismatchError(RetrievalError):
    """The query/runtime fingerprint does not match the vector generation."""


class StaleIndexError(GenerationMismatchError):
    """A persisted index no longer matches its corpus or embedding identity."""
