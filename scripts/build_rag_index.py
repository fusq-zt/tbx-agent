#!/usr/bin/env python3
"""Build an immutable Qdrant-local index from the reviewed knowledge snapshot.

Both supported embedding paths are explicit: a pinned loopback OpenAI-compatible
endpoint, or a pinned, offline local BGE-M3 artifact.  Neither path downloads a
model.  BGE/torch are imported lazily only after a non-dry-run build is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tbx_agent.retrieval import (  # noqa: E402
    BgeM3LocalEmbeddingAdapter,
    ContractError,
    EmbeddingProvenance,
    OpenAICompatibleEmbeddingAdapter,
    QdrantLocalConfig,
    QdrantLocalVectorStore,
    RetrievalError,
    build_index_manifest,
    load_curated_corpus,
    load_retrieval_config,
    validate_index_manifest,
    write_index_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knowledge-dir",
        type=Path,
        default=None,
        help="Reviewed knowledge directory containing source_manifest.json and chunks.jsonl.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Compatibility alias for the reviewed knowledge/chunks.jsonl input.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "retrieval.yaml",
        help="Retrieval runtime config with pinned embedding provenance.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the Qdrant-local root from retrieval.yaml.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Explicit offline BGE-M3 snapshot path; otherwise use its configured env variable.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="External BGE cache path; otherwise use its configured env variable.",
    )
    parser.add_argument(
        "--device",
        help="BGE execution device (auto, cpu, mps, cuda, or cuda:N); env is used if omitted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow atomically promoting a generation when the output already has a current index.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate corpus/config and print the prospective identity without embedding.",
    )
    return parser


def _knowledge_directory(args: argparse.Namespace) -> Path:
    if args.input is not None:
        source = args.input.expanduser().resolve()
        if source.name != "chunks.jsonl":
            raise ContractError("--input must name a reviewed chunks.jsonl file")
        if args.knowledge_dir is not None and args.knowledge_dir.resolve() != source.parent:
            raise ContractError("--input and --knowledge-dir identify different snapshots")
        return source.parent
    return (args.knowledge_dir or PROJECT_ROOT / "knowledge").expanduser().resolve()


def build(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.batch_size <= 512:
        raise ContractError("batch-size must be between 1 and 512")
    knowledge_dir = _knowledge_directory(args)
    snapshot = load_curated_corpus(knowledge_dir)
    runtime = load_retrieval_config(args.config)
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
    preview = build_index_manifest(
        snapshot.documents,
        provenance,
        index_backend="qdrant-local",
        source_manifest_sha256=snapshot.source_manifest_sha256,
        chunks_sha256=snapshot.chunks_sha256,
    )
    if args.dry_run:
        return {
            "status": "validated_dry_run",
            "snapshot_id": snapshot.snapshot_id,
            "chunk_count": len(snapshot.documents),
            "prospective_generation_id": preview.generation_id,
            "embedding_fingerprint": provenance.fingerprint,
            "dense_enabled": dense.enabled,
            "vector_backend": runtime.vector_store.backend,
            "knowledge_dir": str(knowledge_dir),
        }
    if not dense.enabled:
        raise ContractError(
            "dense.enabled is false; pin the embedding artifact hash and enable it before build"
        )
    vector = runtime.vector_store
    vector_root = args.output.expanduser().resolve() if args.output else vector.qdrant_path
    if vector.backend != "qdrant_local" or vector_root is None:
        raise ContractError("index builder requires vector_store.backend=qdrant_local")
    if not vector.qdrant_collection:
        raise ContractError("Qdrant local collection is not configured")

    if (vector_root / "current.json").exists() and not args.force:
        raise ContractError("output already has a promoted index; pass --force to re-promote")
    if dense.adapter == "openai_compatible_http" and dense.endpoint:
        embedder = OpenAICompatibleEmbeddingAdapter(
            endpoint=dense.endpoint,
            provenance=provenance,
            timeout_seconds=dense.timeout_seconds,
            require_loopback=dense.require_loopback,
            query_prefix=dense.query_prefix,
            document_prefix=dense.document_prefix,
        )
    elif dense.adapter == "bge_m3_local":
        if not dense.model_path_env or not dense.cache_dir_env or not dense.device_env:
            raise ContractError("BGE-M3 environment variable names are not configured")
        embedder = BgeM3LocalEmbeddingAdapter(
            provenance=provenance,
            model_path_env=dense.model_path_env,
            cache_dir_env=dense.cache_dir_env,
            device_env=dense.device_env,
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            device=args.device,
            query_prefix=dense.query_prefix,
            document_prefix=dense.document_prefix,
            batch_size=args.batch_size,
        )
    else:
        raise ContractError("index builder requires bge_m3_local or openai_compatible_http")
    vectors: list[tuple[float, ...]] = []
    documents = snapshot.documents
    for start in range(0, len(documents), args.batch_size):
        batch_documents = documents[start : start + args.batch_size]
        batch = embedder.embed(
            tuple(document.embedding_text for document in batch_documents),
            purpose="document",
        )
        if batch.provenance.fingerprint != provenance.fingerprint:
            raise ContractError("embedding provenance changed while building the index")
        vectors.extend(batch.vectors)

    store = QdrantLocalVectorStore(
        QdrantLocalConfig(
            root=vector_root,
            collection=vector.qdrant_collection,
        )
    )
    promoted = store.build_generation(
        documents,
        vectors,
        embedding_provenance=provenance,
    )
    manifest = build_index_manifest(
        documents,
        provenance,
        index_backend=promoted.store_type,
        source_manifest_sha256=snapshot.source_manifest_sha256,
        chunks_sha256=snapshot.chunks_sha256,
    )
    if manifest.corpus_sha256 != promoted.corpus_sha256:
        raise ContractError("promoted Qdrant corpus identity differs from reviewed chunks")
    manifest = type(manifest)(
        **{
            **manifest.to_dict(),
            "generation_id": promoted.generation_id,
        }
    )
    validate_index_manifest(
        manifest,
        documents,
        provenance,
        source_manifest_sha256=snapshot.source_manifest_sha256,
        chunks_sha256=snapshot.chunks_sha256,
    )
    manifest_path = vector_root / "index-manifest.json"
    write_index_manifest(manifest_path, manifest)
    return {
        "status": "completed",
        "snapshot_id": snapshot.snapshot_id,
        "chunk_count": len(documents),
        "generation_id": promoted.generation_id,
        "corpus_sha256": promoted.corpus_sha256,
        "embedding_fingerprint": promoted.embedding_fingerprint,
        "index_manifest": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(args)
    except (RetrievalError, OSError) as exc:
        print(f"RAG index build failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
