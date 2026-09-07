"""Fail-closed BM25 engineering-smoke evaluator for the checked-in RAG snapshot.

The qrels exercised here are deliberately non-expert engineering fixtures.  This
runner must not be used for retriever/model selection or clinical claims.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tbx_agent.knowledge import GuidelineRetriever
from tbx_agent.retrieval import RetrievedReference, evaluate_rankings, load_qrels


class SmokeBenchError(RuntimeError):
    """Raised when a smoke-evaluation contract is violated."""


_CONFIG_KEYS = {
    "schema_version",
    "experiment_id",
    "hypothesis",
    "purpose",
    "prohibited_uses",
    "seed",
    "split_id",
    "suite_id",
    "suite_version",
    "k",
    "queries_path",
    "queries_sha256",
    "qrels_path",
    "qrels_file_sha256",
    "knowledge_dir",
    "expected_snapshot_id",
    "expected_source_manifest_sha256",
    "expected_chunks_sha256",
    "expected_corpus_generation_id",
    "retrieval_config_path",
    "retrieval_config_sha256",
    "backend_contract",
    "regression_policy",
}
_QUERY_KEYS = {
    "schema_version",
    "suite_id",
    "suite_version",
    "split_id",
    "query_id",
    "text",
}
_METRICS = (
    "recall_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "source_recall_at_k",
    "hard_negative_rejection_at_k",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeBenchError(f"cannot load smoke config: {path}") from exc
    if not isinstance(raw, dict) or set(raw) != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS.difference(raw if isinstance(raw, dict) else {}))
        extra = sorted(set(raw if isinstance(raw, dict) else {}).difference(_CONFIG_KEYS))
        raise SmokeBenchError(f"smoke config keys mismatch; missing={missing}, extra={extra}")
    if raw["schema_version"] != 1 or raw["purpose"] != "engineering_smoke_only":
        raise SmokeBenchError("unsupported smoke config schema or purpose")
    if not isinstance(raw["seed"], int) or not isinstance(raw["k"], int) or raw["k"] < 1:
        raise SmokeBenchError("seed must be an integer and k must be a positive integer")
    if not isinstance(raw["prohibited_uses"], list) or len(raw["prohibited_uses"]) < 3:
        raise SmokeBenchError("smoke config must retain explicit prohibited uses")
    backend = raw["backend_contract"]
    if backend != {
        "backend": "deterministic_bm25",
        "retrieval_version": "tbx-retrieval-v1",
        "dense_enabled": False,
        "reranker_enabled": False,
    }:
        raise SmokeBenchError("only the frozen deterministic BM25 backend is allowed")
    policy = raw["regression_policy"]
    supported_policies = (
        {
            "paired_same_suite_only": True,
            "maximum_absolute_metric_drop": 0.0,
        },
        {
            "paired_same_suite_and_corpus_generation_only": True,
            "maximum_absolute_metric_drop": 0.0,
        },
    )
    if policy not in supported_policies:
        raise SmokeBenchError("unsupported regression policy")
    return raw


def _resolve(config_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise SmokeBenchError("artifact paths must be non-empty strings")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def _load_queries(path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SmokeBenchError(f"cannot read query split: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SmokeBenchError(f"invalid query JSONL at line {line_number}") from exc
        if not isinstance(raw, dict) or set(raw) != _QUERY_KEYS:
            raise SmokeBenchError(f"query schema mismatch at line {line_number}")
        if raw["schema_version"] != 1:
            raise SmokeBenchError(f"unsupported query schema at line {line_number}")
        for key in ("suite_id", "suite_version", "split_id", "query_id", "text"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                raise SmokeBenchError(f"invalid {key} at query line {line_number}")
        if (
            raw["suite_id"] != config["suite_id"]
            or raw["suite_version"] != config["suite_version"]
            or raw["split_id"] != config["split_id"]
        ):
            raise SmokeBenchError(f"query identity mismatch at line {line_number}")
        rows.append(raw)
    query_ids = [str(row["query_id"]) for row in rows]
    if not rows or len(query_ids) != len(set(query_ids)):
        raise SmokeBenchError("query split must be non-empty with unique query IDs")
    return tuple(rows)


def _source_identity(project_root: Path) -> dict[str, Any]:
    source_files = [
        project_root / "src" / "tbx_agent" / "knowledge.py",
        project_root / "retrieval" / "smoke_bench.py",
    ]
    source_files.extend(
        sorted((project_root / "src" / "tbx_agent" / "retrieval").glob("*.py"))
    )
    digest = hashlib.sha256()
    for source_file in source_files:
        if not source_file.is_file():
            raise SmokeBenchError(f"evaluation source file is missing: {source_file}")
        relative = source_file.relative_to(project_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_file.read_bytes())
        digest.update(b"\0")
    evaluation_source_sha256 = digest.hexdigest()

    try:
        revision = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain", "--", "."],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        # A source ZIP intentionally has no .git directory. The evaluated
        # implementation is still content-addressed instead of being recorded
        # as an unverifiable or fabricated Git revision.
        revision = None
        status = None
    if revision is not None and len(revision) != 40:
        raise SmokeBenchError("source revision is not a full Git commit")

    source_revision = (
        revision
        if revision is not None
        else f"evaluation-source-sha256:{evaluation_source_sha256}"
    )
    return {
        "git_revision": revision,
        "source_revision": source_revision,
        "source_revision_kind": "git_commit" if revision is not None else "content_digest",
        "workspace_dirty_for_tbx_agent": bool(status.strip()) if status is not None else None,
        "evaluation_source_sha256": evaluation_source_sha256,
        "evaluation_source_file_count": len(source_files),
    }


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SmokeBenchError(f"cannot read ledger: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SmokeBenchError(f"ledger contains blank line at {line_number}")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SmokeBenchError(f"ledger JSON invalid at line {line_number}") from exc
        if not isinstance(event, dict):
            raise SmokeBenchError(f"ledger event is not an object at line {line_number}")
        claimed = event.get("event_sha256")
        material = dict(event)
        material.pop("event_sha256", None)
        actual = _sha256_bytes(_canonical_json(material).encode("utf-8"))
        if claimed != actual or event.get("previous_event_sha256") != previous:
            raise SmokeBenchError(f"ledger chain invalid at line {line_number}")
        if event.get("status") not in {"completed", "regressed_retained", "failed_retained"}:
            raise SmokeBenchError(f"ledger status invalid at line {line_number}")
        previous = claimed
        events.append(event)
    return events


def _append_ledger(path: Path, material: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SmokeBenchError(f"ledger lock already exists: {lock}") from exc
    try:
        os.close(lock_fd)
        events = _load_ledger(path)
        material = dict(material)
        material["previous_event_sha256"] = events[-1]["event_sha256"] if events else None
        material["event_sha256"] = _sha256_bytes(
            _canonical_json(material).encode("utf-8")
        )
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(material) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return material
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _compare_baseline(report: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeBenchError(f"cannot load baseline report: {baseline_path}") from exc
    compatibility_keys = (
        "suite_id",
        "suite_version",
        "split_id",
        "query_split_sha256",
        "qrels_file_sha256",
        "qrels_canonical_sha256",
        "corpus_generation_id",
        "snapshot_id",
        "retrieval_config_sha256",
        "evaluation_source_sha256",
    )
    current_provenance = report["provenance"]
    baseline_provenance = baseline.get("provenance", {})
    mismatches = [
        key
        for key in compatibility_keys
        if baseline_provenance.get(key) != current_provenance.get(key)
    ]
    if mismatches:
        raise SmokeBenchError(f"baseline is not paired-compatible: {mismatches}")
    if [row["query_id"] for row in baseline.get("observations", [])] != [
        row["query_id"] for row in report["observations"]
    ]:
        raise SmokeBenchError("baseline query set/order is not paired-compatible")
    deltas: dict[str, float | None] = {}
    regressed: list[str] = []
    for metric in _METRICS:
        current = report["metrics"].get(metric)
        previous = baseline.get("metrics", {}).get(metric)
        if current is None and previous is None:
            deltas[metric] = None
            continue
        if not isinstance(current, (int, float)) or not isinstance(previous, (int, float)):
            raise SmokeBenchError(f"baseline metric {metric} is invalid")
        delta = float(current) - float(previous)
        deltas[metric] = delta
        if delta < 0:
            regressed.append(metric)
    return {
        "baseline_run_id": baseline.get("run_id"),
        "baseline_report_sha256": _sha256_file(baseline_path),
        "paired_compatible": True,
        "metric_deltas": deltas,
        "regressed_metrics": regressed,
        "regression_detected": bool(regressed),
        "scope": "engineering_regression_only_not_model_selection",
    }


def run_smoke_bench(
    config_path: Path,
    output_path: Path,
    ledger_path: Path,
    *,
    baseline_path: Path | None = None,
) -> dict[str, Any]:
    started_at = _utc_now()
    start_ns = time.perf_counter_ns()
    config_sha256 = "unavailable"
    config: dict[str, Any] | None = None
    source: dict[str, Any] | None = None
    run_id = f"retrieval-smoke-failed-{started_at.replace(':', '').replace('-', '')}"
    try:
        if output_path.exists():
            raise SmokeBenchError(f"refusing to overwrite report: {output_path}")
        config = _load_config(config_path)
        config_sha256 = _sha256_file(config_path)
        queries_path = _resolve(config_path, config["queries_path"])
        qrels_path = _resolve(config_path, config["qrels_path"])
        knowledge_dir = _resolve(config_path, config["knowledge_dir"])
        retrieval_config_path = _resolve(config_path, config["retrieval_config_path"])
        if _sha256_file(queries_path) != config["queries_sha256"]:
            raise SmokeBenchError("query split hash mismatch")
        if _sha256_file(qrels_path) != config["qrels_file_sha256"]:
            raise SmokeBenchError("qrels file hash mismatch")
        if _sha256_file(retrieval_config_path) != config["retrieval_config_sha256"]:
            raise SmokeBenchError("retrieval config hash mismatch")
        queries = _load_queries(queries_path, config)
        qrels = load_qrels(qrels_path)
        if {row["query_id"] for row in queries} != {case.query_id for case in qrels}:
            raise SmokeBenchError("query split and qrels query IDs differ")
        if any(
            case.suite_id != config["suite_id"]
            or case.suite_version != config["suite_version"]
            or case.corpus_generation_id != config["expected_corpus_generation_id"]
            for case in qrels
        ):
            raise SmokeBenchError("qrels suite or corpus identity mismatch")

        retriever = GuidelineRetriever(knowledge_dir)
        if retriever.snapshot_id != config["expected_snapshot_id"]:
            raise SmokeBenchError("knowledge snapshot ID mismatch")
        if retriever.manifest_sha256 != config["expected_source_manifest_sha256"]:
            raise SmokeBenchError("source manifest hash mismatch")
        if retriever.chunks_sha256 != config["expected_chunks_sha256"]:
            raise SmokeBenchError("knowledge chunks hash mismatch")
        if retriever.corpus_generation_id != config["expected_corpus_generation_id"]:
            raise SmokeBenchError("BM25 corpus generation mismatch")
        if retriever.retrieval_version != config["backend_contract"]["retrieval_version"]:
            raise SmokeBenchError("retrieval implementation version mismatch")
        if retriever.retrieval_config.dense.enabled or retriever.retrieval_config.reranker.enabled:
            raise SmokeBenchError("smoke runner prohibits dense retrieval and reranking")

        project_root = Path(__file__).resolve().parents[1]
        source = _source_identity(project_root)
        rankings: dict[str, tuple[RetrievedReference, ...]] = {}
        observations: list[dict[str, Any]] = []
        qrels_by_id = {case.query_id: case for case in qrels}
        for query in queries:
            query_start = time.perf_counter_ns()
            hits = retriever.retrieve(str(query["text"]), top_k=int(config["k"]))
            query_duration_ms = (time.perf_counter_ns() - query_start) / 1_000_000
            references = tuple(
                RetrievedReference(hit.citation.chunk_id, hit.citation.source_id) for hit in hits
            )
            rankings[str(query["query_id"])] = references
            case = qrels_by_id[str(query["query_id"])]
            observations.append(
                {
                    "query_id": query["query_id"],
                    "query_sha256": _sha256_bytes(str(query["text"]).strip().encode("utf-8")),
                    "duration_ms": query_duration_ms,
                    "retrieved": [asdict(reference) for reference in references],
                    "relevant_retrieved": sorted(
                        set(case.relevance).intersection(item.chunk_id for item in references)
                    ),
                    "hard_negatives_retrieved": sorted(
                        set(case.hard_negative_chunk_ids).intersection(
                            item.chunk_id for item in references
                        )
                    ),
                }
            )
        metrics = evaluate_rankings(qrels, rankings, k=int(config["k"])).to_dict()
        finished_at = _utc_now()
        runtime_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        provenance = {
            "suite_id": config["suite_id"],
            "suite_version": config["suite_version"],
            "split_id": config["split_id"],
            "seed": config["seed"],
            "query_split_sha256": config["queries_sha256"],
            "qrels_file_sha256": config["qrels_file_sha256"],
            "qrels_canonical_sha256": metrics["qrels_sha256"],
            "corpus_generation_id": retriever.corpus_generation_id,
            "snapshot_id": retriever.snapshot_id,
            "source_manifest_sha256": retriever.manifest_sha256,
            "chunks_sha256": retriever.chunks_sha256,
            "retrieval_config_sha256": config["retrieval_config_sha256"],
            "config_sha256": config_sha256,
            **source,
        }
        run_material = {
            "provenance": provenance,
            "started_at": started_at,
        }
        run_digest = _sha256_bytes(_canonical_json(run_material).encode("utf-8"))
        run_id = f"retrieval-smoke-{run_digest[:20]}"
        report: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "completed",
            "hypothesis": config["hypothesis"],
            "purpose": config["purpose"],
            "prohibited_uses": config["prohibited_uses"],
            "experiment_config": config,
            "provenance": provenance,
            "runtime": {
                "wall_clock_seconds": runtime_seconds,
                "peak_vram_mb": None,
                "peak_vram_measurement": "unknown_not_measured",
                "execution_device": "cpu_bm25",
            },
            "metrics": metrics,
            "observations": observations,
            "comparison": None,
            "interpretation": (
                "Non-expert engineering smoke fixture only; no medical relevance, clinical, "
                "or model-selection inference is permitted."
            ),
            "release_gate": None,
        }
        if baseline_path is not None:
            comparison = _compare_baseline(report, baseline_path)
            report["comparison"] = comparison
            if comparison["regression_detected"]:
                report["status"] = "regressed_retained"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        report_sha256 = _sha256_file(output_path)
        _append_ledger(
            ledger_path,
            {
                "schema_version": 1,
                "event_id": run_id,
                "recorded_at": _utc_now(),
                "status": report["status"],
                "hypothesis": config["hypothesis"],
                "config_sha256": config_sha256,
                "seed": config["seed"],
                "split_id": config["split_id"],
                "split_sha256": config["queries_sha256"],
                "qrels_sha256": metrics["qrels_sha256"],
                "source_revision": source["source_revision"],
                "metrics": metrics,
                "runtime": report["runtime"],
                "report_path": str(output_path.resolve()),
                "report_sha256": report_sha256,
                "error": None,
            },
        )
        return report
    except Exception as exc:
        runtime_seconds = (time.perf_counter_ns() - start_ns) / 1_000_000_000
        try:
            _append_ledger(
                ledger_path,
                {
                    "schema_version": 1,
                    "event_id": run_id,
                    "recorded_at": _utc_now(),
                    "status": "failed_retained",
                    "hypothesis": config["hypothesis"] if config else "unavailable",
                    "config_sha256": config_sha256,
                    "seed": config["seed"] if config else None,
                    "split_id": config["split_id"] if config else None,
                    "split_sha256": config["queries_sha256"] if config else None,
                    "qrels_sha256": None,
                    "source_revision": source["source_revision"] if source else None,
                    "metrics": None,
                    "runtime": {
                        "wall_clock_seconds": runtime_seconds,
                        "peak_vram_mb": None,
                        "peak_vram_measurement": "unknown_not_measured",
                        "execution_device": "cpu_bm25",
                    },
                    "report_path": None,
                    "report_sha256": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )
        except Exception as ledger_exc:
            message = f"{exc}; additionally could not retain failure: {ledger_exc}"
            raise SmokeBenchError(message) from exc
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_smoke_bench(
            args.config.resolve(),
            args.output.resolve(),
            args.ledger.resolve(),
            baseline_path=args.baseline_report.resolve() if args.baseline_report else None,
        )
    except Exception as exc:
        print(f"retrieval smoke failed: {exc}", file=sys.stderr)
        return 2
    print(
        _canonical_json(
            {
                "run_id": report["run_id"],
                "status": report["status"],
                "metrics": report["metrics"],
            }
        )
    )
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
