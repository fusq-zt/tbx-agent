"""Versioned retrieval building blocks for TBX-Agent.

The package is intentionally not wired into the clinical-support service by import
side effect. Deployment code must explicitly construct reviewed documents, model
adapters, and a promoted vector generation.
"""

from .benchmark import (
    EvaluationQuery,
    classify_retrieval_failure,
    evaluate_retrieval_modes,
    load_evaluation_queries,
    render_benchmark_markdown,
)
from .config import RetrievalRuntimeConfig, load_retrieval_config
from .contracts import (
    EmbeddingBatch,
    EmbeddingProvenance,
    MetadataFilter,
    RetrievalDocument,
    RetrievalFilter,
    RetrievalHit,
    RetrievalQuery,
    RetrievalReceipt,
    RetrievalResult,
    RetrievedChunk,
    sha256_text,
)
from .corpus import (
    CorpusSnapshot,
    load_curated_corpus,
    reviewed_document_from_ingested_chunk,
)
from .embeddings import (
    BgeM3LocalEmbeddingAdapter,
    LocalCallableEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
    local_artifact_sha256,
)
from .engine import RetrievalEngine, RetrievalEngineConfig
from .errors import (
    BackendUnavailableError,
    ContractError,
    GenerationMismatchError,
    OptionalDependencyError,
    RetrievalError,
    StaleIndexError,
)
from .evaluation import (
    QrelCase,
    RetrievalMetricReport,
    RetrievedReference,
    evaluate_rankings,
    load_qrels,
)
from .fusion import deduplicate_candidates, weighted_reciprocal_rank_fusion
from .indexing import (
    IndexManifest,
    build_index_manifest,
    corpus_sha256,
    load_index_manifest,
    validate_index_manifest,
    write_index_manifest,
)
from .rerank import BgeRerankerAdapter, LocalCallableReranker
from .retrievers import DenseRetriever, HybridRetriever
from .sparse import BM25Index
from .vectorstores import (
    QdrantLocalConfig,
    QdrantLocalVectorStore,
    QdrantServerConfig,
    QdrantServerVectorStore,
    SQLiteVecStore,
    VectorGenerationManifest,
)

__all__ = [
    "BM25Index",
    "BackendUnavailableError",
    "BgeM3LocalEmbeddingAdapter",
    "BgeRerankerAdapter",
    "ContractError",
    "CorpusSnapshot",
    "DenseRetriever",
    "EmbeddingBatch",
    "EmbeddingProvenance",
    "EvaluationQuery",
    "GenerationMismatchError",
    "HybridRetriever",
    "IndexManifest",
    "LocalCallableEmbeddingAdapter",
    "LocalCallableReranker",
    "MetadataFilter",
    "OpenAICompatibleEmbeddingAdapter",
    "OptionalDependencyError",
    "QdrantLocalConfig",
    "QdrantLocalVectorStore",
    "QdrantServerConfig",
    "QdrantServerVectorStore",
    "QrelCase",
    "RetrievalDocument",
    "RetrievalEngine",
    "RetrievalEngineConfig",
    "RetrievalError",
    "RetrievalFilter",
    "RetrievalHit",
    "RetrievalMetricReport",
    "RetrievalQuery",
    "RetrievalReceipt",
    "RetrievalResult",
    "RetrievalRuntimeConfig",
    "RetrievedReference",
    "RetrievedChunk",
    "SQLiteVecStore",
    "StaleIndexError",
    "VectorGenerationManifest",
    "deduplicate_candidates",
    "build_index_manifest",
    "classify_retrieval_failure",
    "corpus_sha256",
    "evaluate_rankings",
    "evaluate_retrieval_modes",
    "load_evaluation_queries",
    "load_qrels",
    "load_curated_corpus",
    "load_index_manifest",
    "load_retrieval_config",
    "local_artifact_sha256",
    "reviewed_document_from_ingested_chunk",
    "render_benchmark_markdown",
    "sha256_text",
    "validate_index_manifest",
    "weighted_reciprocal_rank_fusion",
    "write_index_manifest",
]
