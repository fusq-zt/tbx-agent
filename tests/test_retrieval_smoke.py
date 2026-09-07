from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import retrieval.smoke_bench as smoke_bench_module
from retrieval.smoke_bench import SmokeBenchError, run_smoke_bench

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "evaluation" / "retrieval" / "smoke_v5" / "config.json"


def _load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_checked_in_bm25_smoke_is_reproducible_and_content_addressed(tmp_path: Path):
    first_report_path = tmp_path / "first.json"
    ledger_path = tmp_path / "ledger.jsonl"
    first = run_smoke_bench(CONFIG_PATH, first_report_path, ledger_path)
    second = run_smoke_bench(
        CONFIG_PATH,
        tmp_path / "second.json",
        ledger_path,
        baseline_path=first_report_path,
    )

    assert first["status"] == "completed"
    assert first["purpose"] == "engineering_smoke_only"
    assert first["release_gate"] is None
    assert first["provenance"]["corpus_generation_id"] == ("sparse-1b305b3eb573708ed7989a3e")
    assert first["provenance"]["query_split_sha256"] == (
        "ba5437889740e96a8011ec0af1cbf3a8e2b765b2d74548bdb38ca6889423c3c9"
    )
    assert first["runtime"]["peak_vram_mb"] is None
    assert first["runtime"]["peak_vram_measurement"] == "unknown_not_measured"
    assert first["provenance"]["source_revision_kind"] in {"git_commit", "content_digest"}
    assert first["provenance"]["source_revision"]
    assert first["metrics"] == second["metrics"]
    assert second["comparison"]["paired_compatible"] is True
    assert second["comparison"]["regression_detected"] is False
    assert second["comparison"]["metric_deltas"] == {
        "hard_negative_rejection_at_k": 0.0,
        "mrr_at_k": 0.0,
        "ndcg_at_k": 0.0,
        "recall_at_k": 0.0,
        "source_recall_at_k": 0.0,
    }
    assert len(first["observations"]) == 31
    assert all("text" not in row for row in first["observations"])

    events = _load_events(ledger_path)
    assert [event["status"] for event in events] == ["completed", "completed"]
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    for event in events:
        material = dict(event)
        claimed = material.pop("event_sha256")
        canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == claimed


def test_source_zip_without_git_uses_content_addressed_revision(monkeypatch) -> None:
    def unavailable_git(*args, **kwargs):
        raise FileNotFoundError("git metadata intentionally absent")

    monkeypatch.setattr(smoke_bench_module.subprocess, "run", unavailable_git)
    identity = smoke_bench_module._source_identity(PROJECT_ROOT)

    assert identity["git_revision"] is None
    assert identity["source_revision_kind"] == "content_digest"
    assert identity["source_revision"] == (
        f"evaluation-source-sha256:{identity['evaluation_source_sha256']}"
    )
    assert identity["workspace_dirty_for_tbx_agent"] is None


def test_query_drift_fails_closed_and_is_retained(tmp_path: Path):
    original = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    query_path = tmp_path / "queries.jsonl"
    qrels_path = CONFIG_PATH.parent / "qrels.jsonl"
    query_path.write_bytes((CONFIG_PATH.parent / "queries.jsonl").read_bytes() + b"\n")
    original["queries_path"] = str(query_path)
    original["qrels_path"] = str(qrels_path.resolve())
    original["knowledge_dir"] = str((PROJECT_ROOT / "knowledge").resolve())
    original["retrieval_config_path"] = str((PROJECT_ROOT / "configs" / "retrieval.yaml").resolve())
    drifted_config = tmp_path / "config.json"
    drifted_config.write_text(json.dumps(original), encoding="utf-8")
    ledger_path = tmp_path / "ledger.jsonl"

    with pytest.raises(SmokeBenchError, match="query split hash mismatch"):
        run_smoke_bench(drifted_config, tmp_path / "report.json", ledger_path)

    events = _load_events(ledger_path)
    assert len(events) == 1
    assert events[0]["status"] == "failed_retained"
    assert events[0]["seed"] == 20260901
    assert events[0]["split_sha256"] == original["queries_sha256"]
    assert events[0]["report_path"] is None


def test_tampered_ledger_is_not_extended(tmp_path: Path):
    ledger_path = tmp_path / "ledger.jsonl"
    run_smoke_bench(CONFIG_PATH, tmp_path / "first.json", ledger_path)
    event = json.loads(ledger_path.read_text(encoding="utf-8"))
    event["metrics"]["recall_at_k"] = 0.0
    ledger_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(SmokeBenchError, match="additionally could not retain failure"):
        run_smoke_bench(CONFIG_PATH, tmp_path / "second.json", ledger_path)

    assert len(ledger_path.read_text(encoding="utf-8").splitlines()) == 1
