from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from tbx_agent.retrieval import (
    BackendUnavailableError,
    BgeM3LocalEmbeddingAdapter,
    BgeRerankerAdapter,
    BM25Index,
    ContractError,
    DenseRetriever,
    EmbeddingProvenance,
    GenerationMismatchError,
    HybridRetriever,
    IndexManifest,
    LocalCallableEmbeddingAdapter,
    LocalCallableReranker,
    MetadataFilter,
    OpenAICompatibleEmbeddingAdapter,
    QdrantLocalConfig,
    QdrantLocalVectorStore,
    QdrantServerConfig,
    QdrantServerVectorStore,
    QrelCase,
    RetrievalDocument,
    RetrievalEngine,
    RetrievalEngineConfig,
    RetrievalFilter,
    RetrievalQuery,
    RetrievalReceipt,
    RetrievedReference,
    SQLiteVecStore,
    StaleIndexError,
    build_index_manifest,
    corpus_sha256,
    evaluate_rankings,
    load_curated_corpus,
    load_index_manifest,
    load_qrels,
    load_retrieval_config,
    local_artifact_sha256,
    sha256_text,
    validate_index_manifest,
    write_index_manifest,
)
from tbx_agent.retrieval.contracts import RankedItem, canonical_json


def _document(
    chunk_id: str,
    text: str,
    *,
    source_id: str = "china-guide",
    jurisdiction: str = "China",
    publication_date: str | None = "2024-10-01",
    topics: tuple[str, ...] = ("diagnosis",),
    scopes: tuple[str, ...] = ("diagnostic_support",),
    review_status: str = "approved",
    retrievable: bool = True,
) -> RetrievalDocument:
    return RetrievalDocument(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        content_sha256=sha256_text(text),
        source_sha256=sha256_text(f"source:{source_id}"),
        title="合成测试指南",
        locator=f"section {chunk_id}",
        jurisdiction=jurisdiction,
        publication_date=publication_date,
        topics=topics,
        allowed_claim_scopes=scopes,
        review_status=review_status,  # type: ignore[arg-type]
        retrievable=retrievable,
    )


def _provenance(model_id: str = "BAAI/bge-m3") -> EmbeddingProvenance:
    return EmbeddingProvenance(
        provider="test-local",
        model_id=model_id,
        dimensions=2,
        normalized=True,
        model_sha256=sha256_text(model_id),
        revision="synthetic-v1",
    )


def test_document_contract_rejects_content_tampering_and_invalid_date():
    document = _document("c1", "病原学检查用于合成测试")

    with pytest.raises(ContractError, match="content hash mismatch"):
        replace(document, text="已被篡改")
    with pytest.raises(ContractError, match="publication_date"):
        replace(document, publication_date="2024/10/01")


def test_bm25_is_deterministic_and_filters_before_scoring():
    documents = (
        _document("approved", "痰标本病原学检查是进一步检查的一部分"),
        _document("foreign", "病原学检查", jurisdiction="Global"),
        _document("pending", "病原学检查", review_status="pending_medical_review"),
        _document("blocked", "病原学检查", retrievable=False),
        _document("old", "病原学检查", publication_date="2010-01-01"),
    )
    index = BM25Index(reversed(documents))
    policy = RetrievalFilter(
        jurisdictions=("China",),
        required_claim_scopes_any=("diagnostic_support",),
        published_on_or_after="2020-01-01",
    )

    first = index.search("需要做什么病原学检查", filters=policy)
    second = index.search("需要做什么病原学检查", filters=policy)

    assert first == second
    assert [item.chunk_id for item in first] == ["approved"]


def test_openai_embedding_adapter_is_loopback_pinned_and_schema_checked():
    observed: dict[str, object] = {}

    def transport(url, payload, headers, timeout):
        observed.update(url=url, payload=payload, headers=headers, timeout=timeout)
        return {
            "model": "BAAI/bge-m3",
            "data": [
                {"index": 1, "embedding": [0.0, 2.0]},
                {"index": 0, "embedding": [3.0, 0.0]},
            ],
        }

    adapter = OpenAICompatibleEmbeddingAdapter(
        endpoint="http://127.0.0.1:7998/v1/embeddings",
        provenance=_provenance(),
        query_prefix="query: ",
        transport=transport,
    )
    result = adapter.embed(("甲", "乙"), purpose="query")

    assert result.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert observed["payload"]["input"] == ["query: 甲", "query: 乙"]  # type: ignore[index]
    assert result.provenance.query_prefix == "query: "
    assert result.provenance.fingerprint != _provenance().fingerprint
    with pytest.raises(ContractError, match="loopback"):
        OpenAICompatibleEmbeddingAdapter(
            endpoint="http://example.invalid/v1/embeddings",
            provenance=_provenance(),
        )


def test_bge_m3_local_adapter_is_lazy_offline_and_reuses_one_loaded_model(
    tmp_path: Path,
):
    model_path = tmp_path / "bge-m3-snapshot"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"bge_m3"}', encoding="utf-8")
    provenance = EmbeddingProvenance(
        provider="FlagEmbedding",
        model_id="BAAI/bge-m3",
        dimensions=2,
        normalized=True,
        model_sha256=local_artifact_sha256(model_path),
        revision="frozen-test-revision",
    )
    loader_calls: list[dict[str, object]] = []

    class Model:
        def encode(self, texts, **options):
            assert options["return_sparse"] is False
            assert texts == ["query: 甲", "query: 乙"]
            return {"dense_vecs": [[3.0, 0.0], [0.0, 2.0]]}

    def load_model(**values):
        loader_calls.append(values)
        return Model()

    adapter = BgeM3LocalEmbeddingAdapter(
        provenance=provenance,
        model_path=model_path,
        cache_dir=tmp_path,
        device="cpu",
        query_prefix="query: ",
        model_loader=load_model,
    )

    assert adapter.loaded is False
    assert loader_calls == []
    first = adapter.embed(("甲", "乙"), purpose="query")
    second = adapter.embed(("甲", "乙"), purpose="query")

    assert first.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert first.provenance.query_prefix == "query: "
    assert first.provenance.fingerprint != provenance.fingerprint
    assert second == first
    assert adapter.loaded is True
    assert len(loader_calls) == 1
    assert loader_calls[0]["revision"] == "frozen-test-revision"
    assert loader_calls[0]["device"] == "cpu"


@pytest.mark.parametrize("prefix_field", ["query_prefix", "document_prefix"])
def test_changed_embedding_prefix_rejects_previous_index(prefix_field: str):
    documents = (_document("c1", "synthetic evidence"),)
    provenance = _provenance()
    manifest = build_index_manifest(documents, provenance, index_backend="synthetic")
    observed = []

    def embed(texts, purpose):
        observed.append((texts, purpose))
        return [(1.0, 0.0) for _ in texts]

    adapter = LocalCallableEmbeddingAdapter(
        embed, provenance=provenance, **{prefix_field: "changed: "}
    )
    purpose = "query" if prefix_field == "query_prefix" else "document"
    batch = adapter.embed(("evidence",), purpose=purpose)

    assert observed == [(["changed: evidence"], purpose)]
    assert batch.provenance.fingerprint == adapter.provenance.fingerprint
    with pytest.raises(StaleIndexError, match="embedding_fingerprint"):
        validate_index_manifest(manifest, documents, adapter.provenance)


def test_adapter_inherits_prefixes_from_provenance():
    provenance = replace(_provenance(), query_prefix="query: ", document_prefix="passage: ")
    observed = []

    def embed(texts, purpose):
        observed.append(texts)
        return [(1.0, 0.0) for _ in texts]

    adapter = LocalCallableEmbeddingAdapter(embed, provenance=provenance)
    adapter.embed(("one",), purpose="query")
    adapter.embed(("two",), purpose="document")
    assert observed == [["query: one"], ["passage: two"]]
    assert adapter.provenance == provenance


def test_legacy_index_without_prefix_attestation_requires_rebuild():
    documents = (_document("c1", "synthetic evidence"),)
    provenance = _provenance()
    manifest = build_index_manifest(documents, provenance, index_backend="synthetic")
    legacy = asdict(provenance)
    legacy.pop("query_prefix")
    legacy.pop("document_prefix")
    legacy_fingerprint = sha256_text(
        json.dumps(legacy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    old_manifest = replace(
        manifest, embedding_fingerprint=legacy_fingerprint, embedding_provenance=legacy
    )
    with pytest.raises(StaleIndexError, match="embedding_fingerprint"):
        validate_index_manifest(old_manifest, documents, provenance)


def test_bge_m3_local_adapter_fails_closed_before_optional_import(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("RAG_EMBEDDING_MODEL_PATH", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_DEVICE", raising=False)
    provenance = EmbeddingProvenance(
        provider="FlagEmbedding",
        model_id="BAAI/bge-m3",
        dimensions=2,
        normalized=True,
        model_sha256=sha256_text("missing-local-model"),
        revision="frozen-test-revision",
    )
    loader_called = False

    def forbidden_loader(**_values):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("loader must not run without explicit configuration")

    adapter = BgeM3LocalEmbeddingAdapter(
        provenance=provenance,
        model_loader=forbidden_loader,
    )

    with pytest.raises(BackendUnavailableError, match="model path environment variable"):
        adapter.embed(("测试",), purpose="query")
    assert loader_called is False
    assert adapter.loaded is False


def test_bge_reranker_validates_indices_and_model_identity():
    documents = (_document("c1", "甲"), _document("c2", "乙"))
    adapter = BgeRerankerAdapter(
        endpoint="http://127.0.0.1:7999/rerank",
        model_sha256=sha256_text("reranker"),
        transport=lambda *_: {
            "model": "BAAI/bge-reranker-v2-m3",
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
            ],
        },
    )

    assert [item.chunk_id for item in adapter.score("测试", documents)] == ["c2", "c1"]


def test_sqlite_vec_generation_is_atomic_filtered_and_content_addressed(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    china = _document("china", "中国指南病原学检测")
    global_document = _document(
        "global", "全球指南影像检查", jurisdiction="Global", source_id="who"
    )
    provenance = _provenance()
    store = SQLiteVecStore(tmp_path / "vectors")

    # Input order is deliberately not chunk-id order; vectors must stay attached.
    manifest = store.build_generation(
        (global_document, china),
        ((0.0, 1.0), (1.0, 0.0)),
        embedding_provenance=provenance,
    )
    second = store.build_generation(
        (global_document, china),
        ((0.0, 1.0), (1.0, 0.0)),
        embedding_provenance=provenance,
    )
    hits = store.search(
        (1.0, 0.0),
        filters=RetrievalFilter(jurisdictions=("China",)),
        limit=5,
    )
    pointer = json.loads((tmp_path / "vectors" / "current.json").read_text(encoding="utf-8"))

    assert manifest == second
    assert pointer["generation_id"] == manifest.generation_id
    assert manifest.vector_count == 2
    assert [item.chunk_id for item in hits] == ["china"]


def test_engine_hybrid_reranking_and_exact_duplicate_suppression(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    first = _document("a", "结核病原学检查证据")
    duplicate = _document("b", "结核病原学检查证据", source_id="duplicate-source")
    other = _document("c", "结核影像辅助筛查")
    documents = (first, duplicate, other)
    provenance = _provenance()

    def embed(texts, purpose):
        return [(1.0, 0.0) if "病原学" in text else (0.0, 1.0) for text in texts]

    embedder = LocalCallableEmbeddingAdapter(embed, provenance=provenance)
    store = SQLiteVecStore(tmp_path / "vectors")
    document_batch = embedder.embed([item.embedding_text for item in documents], purpose="document")
    store.build_generation(
        documents,
        document_batch.vectors,
        embedding_provenance=provenance,
    )
    reranker = LocalCallableReranker(
        lambda _query, texts: [0.1 if "病原学" in text else 0.9 for text in texts],
        model_id="synthetic-reranker",
        model_sha256=sha256_text("synthetic-reranker"),
    )
    engine = RetrievalEngine(documents, embedder=embedder, vector_store=store, reranker=reranker)

    result = engine.retrieve(RetrievalQuery("q1", "结核检查", top_k=3))

    assert result.receipt.backend_status == {
        "bm25": "used",
        "dense": "used",
        "reranker": "used",
    }
    assert "b" in result.receipt.suppressed_duplicate_ids
    assert [item.document.chunk_id for item in result.hits] == ["c", "a"]
    assert all(item.backend_ranks for item in result.hits)


def test_engine_falls_back_deterministically_on_embedding_generation_mismatch(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    document = _document("c1", "病原学检查")
    original = _provenance("model-a")
    store = SQLiteVecStore(tmp_path / "vectors")
    store.build_generation((document,), ((1.0, 0.0),), embedding_provenance=original)
    changed = _provenance("model-b")
    embedder = LocalCallableEmbeddingAdapter(
        lambda _texts, _purpose: [(1.0, 0.0)], provenance=changed
    )
    engine = RetrievalEngine((document,), embedder=embedder, vector_store=store)

    result = engine.retrieve(RetrievalQuery("q1", "病原学"))

    assert result.receipt.backend_status["dense"] == "mismatch"
    assert result.receipt.fallback_reason == "dense:GenerationMismatchError"
    assert result.receipt.corpus_generation_id.startswith("sparse-")
    assert [item.document.chunk_id for item in result.hits] == ["c1"]


def test_engine_can_fail_closed_instead_of_sparse_fallback(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    document = _document("c1", "病原学检查")
    store = SQLiteVecStore(tmp_path / "vectors")
    store.build_generation((document,), ((1.0, 0.0),), embedding_provenance=_provenance("a"))
    embedder = LocalCallableEmbeddingAdapter(
        lambda _texts, _purpose: [(1.0, 0.0)], provenance=_provenance("b")
    )
    config = RetrievalEngineConfig(fallback_to_sparse=False)

    with pytest.raises(GenerationMismatchError):
        RetrievalEngine((document,), config=config, embedder=embedder, vector_store=store).retrieve(
            RetrievalQuery("q1", "病原学")
        )


def test_qdrant_adapter_boundary_prohibits_local_mode_and_insecure_remote_http():
    values = {
        "collection": "tbx-guidelines-v1",
        "dimensions": 1024,
        "generation_id": "generation-1",
        "corpus_sha256": sha256_text("corpus"),
        "embedding_fingerprint": sha256_text("embedding"),
    }
    with pytest.raises(ContractError, match="local/in-memory"):
        QdrantServerConfig(base_url="http://localhost/:memory:", **values)
    with pytest.raises(ContractError, match="HTTPS"):
        QdrantServerConfig(base_url="http://qdrant.internal:6333", **values)
    assert QdrantServerConfig(base_url="http://127.0.0.1:6333", **values).collection == (
        "tbx-guidelines-v1"
    )


def test_versioned_retrieval_metrics_include_hard_negative_rejection(tmp_path: Path):
    cases = (
        QrelCase(
            schema_version=1,
            suite_id="synthetic-retrieval",
            suite_version="1.0.0",
            corpus_generation_id="corpus-v1",
            query_id="q1",
            relevance={"c1": 3, "c2": 1},
            relevant_source_ids=("s1", "s2"),
            hard_negative_chunk_ids=("hard",),
        ),
    )
    rankings = {
        "q1": (
            RetrievedReference("c1", "s1"),
            RetrievedReference("hard", "noise"),
        )
    }

    report = evaluate_rankings(cases, rankings, k=2)

    assert report.recall_at_k == 0.5
    assert report.mrr_at_k == 1.0
    assert report.source_recall_at_k == 0.5
    assert report.hard_negative_rejection_at_k == 0.0
    assert 0 < report.ndcg_at_k < 1

    qrels_path = tmp_path / "qrels.jsonl"
    qrels_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_id": "synthetic-retrieval",
                "suite_version": "1.0.0",
                "corpus_generation_id": "corpus-v1",
                "query_id": "q1",
                "relevance": {"c1": 3},
                "relevant_source_ids": ["s1"],
                "hard_negative_chunk_ids": ["hard"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_qrels(qrels_path)[0].suite_version == "1.0.0"


def test_checked_in_config_is_sparse_safe_until_model_hashes_are_pinned():
    config_path = Path(__file__).parents[1] / "configs" / "retrieval.yaml"
    config = load_retrieval_config(config_path)

    assert config.schema_version == 1
    assert config.engine.retrieval_mode == "sparse"
    assert config.dense.model_id == "BAAI/bge-m3"
    assert config.dense.adapter == "bge_m3_local"
    assert config.dense.enabled is False
    assert config.vector_store.backend == "qdrant_local"
    assert config.vector_store.qdrant_path is not None
    assert config.vector_store.qdrant_collection == "tbx-guidelines"
    assert config.reranker.enabled is False


def test_dense_retriever_uses_injected_embeddings_and_unified_metadata_filter():
    china = _document("china", "中国结核病原学检查", topics=("diagnosis",))
    who = _document(
        "who",
        "WHO tuberculosis diagnostic test",
        source_id="who-guide",
        jurisdiction="WHO",
        topics=("diagnosis",),
    )
    treatment = _document("treatment", "结核治疗教育", topics=("treatment",))
    provenance = _provenance()
    adapter = LocalCallableEmbeddingAdapter(
        lambda texts, purpose: [(1.0, 0.0) for _ in texts],
        provenance=provenance,
    )
    retriever = DenseRetriever(
        (who, treatment, china),
        embedder=adapter,
        document_vectors={
            "china": (1.0, 0.0),
            "who": (1.0, 0.0),
            "treatment": (0.0, 1.0),
        },
    )

    hits = retriever.retrieve(
        "病原学检查",
        filters=MetadataFilter(
            topics_any=("diagnosis",),
            jurisdictions=("China",),
            required_claim_scopes_any=("diagnostic_support",),
        ),
        top_k=3,
    )

    assert [hit.chunk_id for hit in hits] == ["china"]
    assert hits[0].metadata["jurisdiction"] == "China"
    assert hits[0].metadata["topics"] == ("diagnosis",)
    assert hits[0].backend_ranks == {"dense": 1}


def test_hybrid_rrf_has_stable_chunk_id_tie_break():
    first = _document("a", "alpha shared evidence")
    second = _document("b", "alpha shared evidence", source_id="source-b")
    provenance = _provenance()
    adapter = LocalCallableEmbeddingAdapter(
        lambda texts, purpose: [(1.0, 0.0) for _ in texts],
        provenance=provenance,
    )
    dense = DenseRetriever(
        (second, first),
        embedder=adapter,
        document_vectors={"a": (0.9, 0.1), "b": (1.0, 0.0)},
    )
    hybrid = HybridRetriever((second, first), dense_retriever=dense)

    first_run = hybrid.retrieve("alpha", top_k=2)
    second_run = hybrid.retrieve("alpha", top_k=2)

    assert [item.chunk_id for item in first_run] == ["a", "b"]
    assert first_run == second_run
    assert first_run[0].backend_ranks == {"bm25": 1, "dense": 2}
    assert first_run[1].backend_ranks == {"bm25": 2, "dense": 1}


def test_index_manifest_detects_content_metadata_and_embedding_staleness(tmp_path: Path):
    document = _document("chunk", "病原学检查")
    provenance = _provenance("embedding-a")
    manifest = build_index_manifest(
        (document,),
        provenance,
        index_backend="qdrant_local",
        source_manifest_sha256=sha256_text("manifest"),
        chunks_sha256=sha256_text("chunks"),
    )
    path = tmp_path / "index-manifest.json"
    write_index_manifest(path, manifest)

    loaded = load_index_manifest(path)
    assert isinstance(loaded, IndexManifest)
    assert loaded.manifest_sha256 == manifest.manifest_sha256
    validate_index_manifest(
        loaded,
        (document,),
        provenance,
        source_manifest_sha256=sha256_text("manifest"),
        chunks_sha256=sha256_text("chunks"),
    )
    with pytest.raises(StaleIndexError, match="corpus_sha256"):
        validate_index_manifest(
            loaded,
            (replace(document, jurisdiction="WHO"),),
            provenance,
        )
    with pytest.raises(StaleIndexError, match="embedding_fingerprint"):
        validate_index_manifest(loaded, (document,), _provenance("embedding-b"))


def test_dense_retriever_rejects_a_stale_manifest_before_query():
    original = _document("chunk", "原始证据")
    changed_text = "发生变化的证据"
    changed = replace(
        original,
        text=changed_text,
        content_sha256=sha256_text(changed_text),
    )
    provenance = _provenance()
    manifest = build_index_manifest(
        (original,), provenance, index_backend="in_memory_exact"
    )
    adapter = LocalCallableEmbeddingAdapter(
        lambda texts, purpose: [(1.0, 0.0) for _ in texts],
        provenance=provenance,
    )

    with pytest.raises(StaleIndexError, match="corpus_sha256"):
        DenseRetriever(
            (changed,),
            embedder=adapter,
            document_vectors={"chunk": (1.0, 0.0)},
            index_manifest=manifest,
        )


def test_hybrid_falls_back_to_bm25_when_injected_dense_query_fails():
    document = _document("chunk", "病原学检查")
    provenance = _provenance()

    def fail_query(texts, purpose):
        if purpose == "query":
            raise RuntimeError("synthetic embedding outage")
        return [(1.0, 0.0) for _ in texts]

    dense = DenseRetriever(
        (document,),
        embedder=LocalCallableEmbeddingAdapter(fail_query, provenance=provenance),
        document_vectors={"chunk": (1.0, 0.0)},
    )
    hybrid = HybridRetriever((document,), dense_retriever=dense)

    hits = hybrid.retrieve("病原学检查")

    assert [item.chunk_id for item in hits] == ["chunk"]
    assert hybrid.last_backend_status == {"bm25": "used", "dense": "unavailable"}
    assert hybrid.last_fallback_reason == "dense:BackendUnavailableError"


def test_curated_snapshot_loads_once_into_unified_retrieval_documents():
    project_root = Path(__file__).parents[1]
    snapshot = load_curated_corpus(project_root / "knowledge")

    assert snapshot.snapshot_id
    assert len(snapshot.documents) == len(snapshot.chunks)
    assert all(item.retrievable for item in snapshot.documents)
    assert all(item.review_status == "approved" for item in snapshot.documents)
    assert len(snapshot.source_manifest_sha256) == 64
    assert len(snapshot.chunks_sha256) == 64


def test_qdrant_local_config_is_path_based_and_does_not_import_a_model(tmp_path: Path):
    config = QdrantLocalConfig(root=tmp_path / "qdrant", collection="tbx-guidelines-v1")

    assert config.root == tmp_path / "qdrant"
    assert config.collection == "tbx-guidelines-v1"


def test_qdrant_local_generation_is_promoted_and_metadata_filtered_with_fake_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class Box:
        def __init__(self, **values):
            self.__dict__.update(values)

    class Models:
        FieldCondition = Box
        MatchValue = Box
        MatchAny = Box
        DatetimeRange = Box
        Filter = Box
        VectorParams = Box
        PointStruct = Box
        Distance = SimpleNamespace(COSINE="cosine")

    class Client:
        def __init__(self, *, path: str):
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            self.points_path = self.path / "points.json"
            self.points = (
                json.loads(self.points_path.read_text(encoding="utf-8"))
                if self.points_path.is_file()
                else []
            )

        def create_collection(self, **_kwargs):
            return True

        def upsert(self, *, points, **_kwargs):
            self.points.extend(
                {
                    "id": point.id,
                    "vector": point.vector,
                    "payload": point.payload,
                }
                for point in points
            )
            self.points_path.write_text(json.dumps(self.points), encoding="utf-8")

        def query_points(self, *, query, **_kwargs):
            points = [
                SimpleNamespace(
                    payload=point["payload"],
                    score=sum(
                        left * right
                        for left, right in zip(query, point["vector"], strict=True)
                    ),
                )
                for point in self.points
            ]
            return SimpleNamespace(points=points)

        def close(self):
            return None

    monkeypatch.setattr(
        QdrantLocalVectorStore,
        "_load_qdrant",
        staticmethod(lambda: (Client, Models, "test-qdrant")),
    )
    china = _document("china", "中国指南")
    who = _document("who", "WHO guide", source_id="who", jurisdiction="WHO")
    config = QdrantLocalConfig(root=tmp_path / "qdrant", collection="tbx-guidelines")
    store = QdrantLocalVectorStore(config)

    manifest = store.build_generation(
        (who, china),
        ((0.0, 1.0), (1.0, 0.0)),
        embedding_provenance=_provenance(),
    )
    reopened = QdrantLocalVectorStore(config)
    hits = reopened.search(
        (1.0, 0.0),
        filters=MetadataFilter(jurisdictions=("China",)),
        limit=5,
    )

    assert reopened.manifest == manifest
    assert manifest.store_type == "qdrant-local"
    assert manifest.vector_count == 2
    assert [item.chunk_id for item in hits] == ["china"]


@pytest.mark.parametrize(
    "backend", ["bm25", "dense-memory", "hybrid-memory", "sqlite-vec", "qdrant-local"]
)
def test_required_source_is_filtered_before_backend_limit(tmp_path: Path, backend: str):
    target = _document("target", "keyword with extra context", source_id="requested")
    distractors = tuple(
        _document(f"distractor-{number}", f"keyword keyword {number}", source_id="other")
        for number in range(12)
    )
    documents = (*distractors, target)
    vectors = (*((1.0, 0.0) for _ in distractors), (0.0, 1.0))
    policy = MetadataFilter(required_source_ids=("requested",))
    adapter = LocalCallableEmbeddingAdapter(
        lambda texts, _purpose: [(1.0, 0.0) for _ in texts], provenance=_provenance()
    )
    if backend == "bm25":
        store = BM25Index(documents)
        unconstrained = store.search("keyword", limit=1)
        filtered = store.search("keyword", filters=policy, limit=1)
    elif backend in {"dense-memory", "hybrid-memory"}:
        dense = DenseRetriever(
            documents,
            embedder=adapter,
            document_vectors=dict(zip((item.chunk_id for item in documents), vectors, strict=True)),
        )
        retriever = (
            HybridRetriever(documents, dense_retriever=dense, candidate_limit=1)
            if backend == "hybrid-memory"
            else dense
        )
        unconstrained = retriever.retrieve("keyword", top_k=1)
        filtered = retriever.retrieve("keyword", filters=policy, top_k=1)
    else:
        pytest.importorskip("sqlite_vec" if backend == "sqlite-vec" else "qdrant_client")
        store = (
            SQLiteVecStore(tmp_path / backend)
            if backend == "sqlite-vec"
            else QdrantLocalVectorStore(QdrantLocalConfig(root=tmp_path / backend))
        )
        store.build_generation(documents, vectors, embedding_provenance=_provenance())
        unconstrained = store.search((1.0, 0.0), filters=MetadataFilter(), limit=1)
        filtered = store.search((1.0, 0.0), filters=policy, limit=1)
        if isinstance(store, QdrantLocalVectorStore):
            store.close()
    assert unconstrained and unconstrained[0].chunk_id != "target"
    assert [item.chunk_id for item in filtered] == ["target"]


def test_qdrant_server_source_filter_is_sent_before_limit(monkeypatch: pytest.MonkeyPatch):
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self, _limit):
            return b'{"result":{"points":[{"payload":{"chunk_id":"target"},"score":0.5}]}}'

    def urlopen(request, **_kwargs):
        observed.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    store = QdrantServerVectorStore(
        QdrantServerConfig(
            base_url="http://localhost:6333",
            collection="synthetic",
            dimensions=2,
            generation_id="synthetic-v1",
            corpus_sha256=sha256_text("synthetic"),
            embedding_fingerprint=_provenance().fingerprint,
        )
    )
    assert store.search(
        (1.0, 0.0), filters=MetadataFilter(required_source_ids=("requested",)), limit=1
    )[0].chunk_id == "target"
    assert observed["limit"] == 1
    assert {"key": "source_id", "match": {"any": ["requested"]}} in observed["filter"]["must"]


@pytest.mark.parametrize(
    "filters",
    [
        RetrievalFilter(topics_any=("diagnosis",)),
        RetrievalFilter(jurisdictions=("China",)),
        RetrievalFilter(required_claim_scopes_any=("diagnostic_support",)),
        RetrievalFilter(published_on_or_after="2020-01-01"),
        RetrievalFilter(published_on_or_before="2030-01-01"),
        RetrievalFilter(allowed_review_statuses=("approved", "superseded")),
        RetrievalFilter(retrievable_only=False),
        RetrievalFilter(required_source_ids=("china-guide",)),
    ],
)
def test_receipt_binds_every_filter_even_when_results_are_unchanged(filters: RetrievalFilter):
    engine = RetrievalEngine((_document("one", "keyword"),))
    original = engine.retrieve(RetrievalQuery("same-id", "keyword", top_k=1))
    changed = engine.retrieve(RetrievalQuery("same-id", "keyword", top_k=1, filters=filters))
    assert original.hits == changed.hits
    assert original.receipt.request_sha256 != changed.receipt.request_sha256
    assert original.receipt.retrieval_id != changed.receipt.retrieval_id


def test_receipt_binds_top_k_and_canonicalizes_equivalent_filter_order():
    engine = RetrievalEngine((_document("one", "keyword"),))
    first_query = RetrievalQuery(
        "same-id", "keyword", top_k=1,
        filters=RetrievalFilter(jurisdictions=("China", "WHO")),
    )
    first = engine.retrieve(first_query)
    equivalent = engine.retrieve(
        replace(first_query, filters=RetrievalFilter(jurisdictions=("WHO", "China")))
    )
    more = engine.retrieve(replace(first_query, top_k=2))
    assert first.hits == more.hits
    assert first.receipt == equivalent.receipt
    assert first.receipt.retrieval_id != more.receipt.retrieval_id
    assert first.receipt.schema_version == 2
    assert first.receipt.top_k == 1
    assert first.receipt.returned_chunks == ({
        "chunk_id": "one", "source_id": "china-guide",
        "content_sha256": sha256_text("keyword"),
        "source_sha256": sha256_text("source:china-guide"),
    },)


def test_receipt_result_identity_changes_with_backend_result_order():
    documents = (_document("one", "first evidence"), _document("two", "second evidence"))

    class MutableStore:
        flip = False
        manifest = SimpleNamespace(
            generation_id="synthetic-v1",
            manifest_sha256=sha256_text("synthetic-manifest"),
            corpus_sha256=corpus_sha256(documents),
            embedding_fingerprint=_provenance().fingerprint,
        )

        def search(self, *_args, **_kwargs):
            ids = ("two", "one") if self.flip else ("one", "two")
            return tuple(
                RankedItem(chunk_id, 1.0 / rank, "dense", rank)
                for rank, chunk_id in enumerate(ids, start=1)
            )

    store = MutableStore()
    engine = RetrievalEngine(
        documents,
        config=RetrievalEngineConfig(retrieval_mode="dense"),
        embedder=LocalCallableEmbeddingAdapter(
            lambda texts, _purpose: [(1.0, 0.0) for _ in texts], provenance=_provenance()
        ),
        vector_store=store,
    )
    query = RetrievalQuery("same-id", "evidence", top_k=2)
    first = engine.retrieve(query)
    store.flip = True
    second = engine.retrieve(query)
    assert first.receipt.request_sha256 == second.receipt.request_sha256
    assert first.receipt.retrieval_id != second.receipt.retrieval_id
    assert first.receipt.returned_chunks == tuple(reversed(second.receipt.returned_chunks))


def test_receipt_json_roundtrip_preserves_v1_identity_and_checks_v2_tampering():
    receipt = RetrievalEngine((_document("one", "keyword"),)).retrieve(
        RetrievalQuery("same-id", "keyword", top_k=1)
    ).receipt
    payload = json.loads(json.dumps(receipt.to_dict()))
    assert RetrievalReceipt.from_dict(payload) == receipt
    legacy = dict(payload)
    for key in ("top_k", "filters", "request_sha256", "returned_chunks", "retrieval_id"):
        legacy.pop(key)
    legacy["schema_version"] = 1
    legacy["retrieval_id"] = sha256_text(canonical_json(legacy))[:32]
    old = RetrievalReceipt.from_dict(legacy)
    assert old.schema_version == 1 and old.top_k is None and old.returned_chunks is None
    assert json.loads(json.dumps(old.to_dict())) == legacy
    tampered = json.loads(json.dumps(payload))
    tampered["returned_chunks"][0]["content_sha256"] = sha256_text("tampered")
    with pytest.raises(ContractError, match="identity hash mismatch"):
        RetrievalReceipt.from_dict(tampered)
    payload["top_k"] = 2
    with pytest.raises(ContractError, match="request hash mismatch"):
        RetrievalReceipt.from_dict(payload)


def test_qdrant_local_serializes_actual_clients_across_store_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("qdrant_client")
    config = QdrantLocalConfig(root=tmp_path / "qdrant-concurrent")
    builder = QdrantLocalVectorStore(config)
    builder.build_generation(
        (_document("one", "synthetic evidence"),), ((1.0, 0.0),),
        embedding_provenance=_provenance(),
    )
    first_store, second_store = QdrantLocalVectorStore(config), QdrantLocalVectorStore(config)
    client_class, models, version = builder._load_qdrant()
    first_entered, release_first, second_started = Event(), Event(), Event()
    close_started = Event()
    opened, closed = [], []

    class DelayedClient:
        def __init__(self, **kwargs):
            self.inner = client_class(**kwargs)
            self.first = not opened
            opened.append(self)

        def query_points(self, **kwargs):
            if self.first:
                first_entered.set()
                assert release_first.wait(timeout=5)
            return self.inner.query_points(**kwargs)

        def close(self):
            self.inner.close()
            closed.append(self)

    monkeypatch.setattr(
        QdrantLocalVectorStore, "_load_qdrant",
        staticmethod(lambda: (DelayedClient, models, version)),
    )

    def search(store, started=None):
        if started is not None:
            started.set()
        return store.search((1.0, 0.0), filters=MetadataFilter(), limit=1)

    def close_first():
        close_started.set()
        first_store.close()

    with ThreadPoolExecutor(max_workers=3) as executor:
        first = executor.submit(search, first_store)
        try:
            assert first_entered.wait(timeout=5)
            second = executor.submit(search, second_store, second_started)
            assert second_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
            assert len(opened) == 1
            closing = executor.submit(close_first)
            assert close_started.wait(timeout=5)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.1)
        finally:
            release_first.set()
        assert first.result(timeout=5)[0].chunk_id == "one"
        assert second.result(timeout=5)[0].chunk_id == "one"
        closing.result(timeout=5)
    assert len(opened) == len(closed) == 2
    with pytest.raises(BackendUnavailableError, match="closed"):
        search(first_store)
    first_store.close()
    second_store.close()
    builder.close()


def test_qdrant_local_releases_client_on_failure_and_can_query_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("qdrant_client")
    store = QdrantLocalVectorStore(QdrantLocalConfig(root=tmp_path / "qdrant-outage"))
    store.build_generation(
        (_document("one", "synthetic evidence"),), ((1.0, 0.0),),
        embedding_provenance=_provenance(),
    )
    client_class, models, version = store._load_qdrant()
    closed = []

    class FailingClient:
        def __init__(self, **kwargs):
            self.inner = client_class(**kwargs)

        def query_points(self, **_kwargs):
            raise RuntimeError("synthetic query outage")

        def close(self):
            self.inner.close()
            closed.append(True)

    monkeypatch.setattr(
        QdrantLocalVectorStore, "_load_qdrant",
        staticmethod(lambda: (FailingClient, models, version)),
    )
    with pytest.raises(BackendUnavailableError):
        store.search((1.0, 0.0), filters=MetadataFilter(), limit=1)
    assert closed == [True]
    monkeypatch.setattr(
        QdrantLocalVectorStore, "_load_qdrant",
        staticmethod(lambda: (client_class, models, version)),
    )
    assert store.search((1.0, 0.0), filters=MetadataFilter(), limit=1)[0].chunk_id == "one"
    store.close()
    store.close()
    with pytest.raises(BackendUnavailableError, match="closed"):
        store.search((1.0, 0.0), filters=MetadataFilter(), limit=1)


def test_qdrant_local_manifest_is_pinned_until_a_new_store_is_created(tmp_path: Path):
    pytest.importorskip("qdrant_client")
    config = QdrantLocalConfig(root=tmp_path / "qdrant-generations")
    builder = QdrantLocalVectorStore(config)
    documents = (_document("one", "first evidence"), _document("two", "second evidence"))
    first_manifest = builder.build_generation(
        documents, ((1.0, 0.0), (0.0, 1.0)), embedding_provenance=_provenance(),
    )
    pinned = QdrantLocalVectorStore(config)
    assert pinned.manifest == first_manifest
    second_manifest = builder.build_generation(
        documents, ((0.0, 1.0), (1.0, 0.0)), embedding_provenance=_provenance(),
    )
    assert second_manifest.generation_id != first_manifest.generation_id
    assert pinned.search((1.0, 0.0), filters=MetadataFilter(), limit=1)[0].chunk_id == "one"
    pinned.close()
    reopened = QdrantLocalVectorStore(config)
    assert reopened.manifest == second_manifest
    assert reopened.search((1.0, 0.0), filters=MetadataFilter(), limit=1)[0].chunk_id == "two"
    reopened.close()
    builder.close()
