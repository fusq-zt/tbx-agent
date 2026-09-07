from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_rag_retrieval.py"


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("rag_recording_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events(tmp_path):
    with sqlite3.connect(tmp_path / "experiment-ledger.sqlite3") as connection:
        rows = connection.execute("SELECT payload_json FROM events ORDER BY rowid").fetchall()
    return [json.loads(row[0]) for row in rows]


def test_config_failure_creates_a_failed_record_without_a_success_report(tmp_path, cli):
    output = tmp_path / "result.json"
    assert cli.main([
        "--queries", str(tmp_path / "missing.jsonl"), "--modes", "bm25",
        "--output", str(output),
    ]) == 2
    events = _events(tmp_path)
    assert [event["event_type"] for event in events] == ["running", "failed"]
    assert events[-1]["failure"]["type"] == "ContractError"
    assert events[-1]["environment"]["python"]["version"]
    assert events[-1]["peak_vram"]["available"] is False
    assert events[-1]["metrics_reason"]
    assert events[0]["metadata"]["configuration"]["input_files"]["queries"][
        "unavailable_reason"
    ] == "FileNotFoundError"
    assert not output.exists()


def test_force_preserves_old_reports_and_both_immutable_runs(tmp_path, cli):
    output = tmp_path / "result.json"
    arguments = ["--modes", "bm25", "--output", str(output)]
    assert cli.main(arguments) == 0
    original = output.read_bytes()
    original_markdown = output.with_suffix(".md").read_bytes()
    first = json.loads(original)
    assert cli.main([*arguments, "--force"]) == 0
    second = json.loads(output.read_bytes())
    first_dir = Path(first["experiment"]["run_dir"])
    second_dir = Path(second["experiment"]["run_dir"])
    assert first_dir != second_dir
    assert (first_dir / "report.json").read_bytes() == original
    assert (first_dir / "report.md").read_bytes() == original_markdown
    assert (second_dir / "report.json").read_bytes() == output.read_bytes()
    backup_json = next((second_dir / "previous_reports" / "json_report").iterdir())
    assert backup_json.read_bytes() == original
    backup_markdown = next((second_dir / "previous_reports" / "markdown_report").iterdir())
    assert backup_markdown.read_bytes() == original_markdown
    assert [event["event_type"] for event in _events(tmp_path)] == [
        "running", "complete", "running", "complete",
    ]


def test_refused_overwrite_is_recorded_before_evaluation(tmp_path, cli, monkeypatch):
    output = tmp_path / "result.json"
    output.write_bytes(b"previous report")
    monkeypatch.setattr(cli, "evaluate", lambda args: pytest.fail("evaluation must not start"))
    assert cli.main(["--modes", "bm25", "--output", str(output)]) == 2
    assert output.read_bytes() == b"previous report"
    assert _events(tmp_path)[-1]["event_type"] == "failed"


def test_publish_failure_retains_computed_metrics_and_archived_report(tmp_path, cli, monkeypatch):
    def fail_publish(*args, **kwargs):
        raise OSError("synthetic publish failure")

    monkeypatch.setattr(cli, "publish_report", fail_publish)
    assert cli.main(["--modes", "bm25", "--output", str(tmp_path / "result.json")]) == 2
    event = _events(tmp_path)[-1]
    assert event["event_type"] == "failed"
    assert event["metrics"]["modes"]["bm25"]["status"] == "completed"
    assert event["failure"]["message"] == "synthetic publish failure"
    assert len(event["artifacts"]["json_report"]["sha256"]) == 64
    archive = next((tmp_path / "evaluation-runs").glob("*/report.json"))
    assert json.loads(archive.read_text(encoding="utf-8"))["modes"]["bm25"]["status"] == "completed"


def test_interrupt_gets_a_terminal_event_and_is_reraised(tmp_path, cli, monkeypatch):
    def interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "evaluate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli.main(["--modes", "bm25", "--output", str(tmp_path / "result.json")])
    assert [event["event_type"] for event in _events(tmp_path)] == ["running", "interrupted"]


def test_mode_failure_records_failure_with_report_hashes(tmp_path, cli):
    assert cli.main(["--modes", "dense", "--output", str(tmp_path / "result.json")]) == 1
    event = _events(tmp_path)[-1]
    assert event["event_type"] == "failed"
    assert event["metrics"]["modes"]["dense"]["status"] == "completed_with_failures"
    assert len(event["artifacts"]["json_report"]["sha256"]) == 64


def test_force_cannot_replace_the_experiment_ledger(tmp_path, cli):
    ledger = tmp_path / "experiment-ledger.sqlite3"
    assert cli.main(["--modes", "bm25", "--output", str(ledger), "--force"]) == 2
    assert _events(tmp_path)[-1]["event_type"] == "failed"
    assert not ledger.with_suffix(".md").exists()


@pytest.mark.parametrize("filename", ["report.json", "experiment.sqlite3", "nested/report.json"])
def test_force_cannot_modify_a_previous_run_tree(tmp_path, cli, filename):
    output = tmp_path / "result.json"
    assert cli.main(["--modes", "bm25", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    archive = Path(report["experiment"]["run_dir"])
    before = {path.relative_to(archive): path.read_bytes()
              for path in archive.rglob("*") if path.is_file()}
    assert cli.main([
        "--modes", "bm25", "--output", str(archive / filename), "--force",
    ]) == 2
    after = {path.relative_to(archive): path.read_bytes()
             for path in archive.rglob("*") if path.is_file()}
    assert before == after
