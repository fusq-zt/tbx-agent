from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tbx_agent.experiment_recorder import ExperimentRecorder, ExperimentRecordingError


def _recorder(tmp_path: Path, run_id: str = "synthetic", **kwargs) -> ExperimentRecorder:
    return ExperimentRecorder(
        run_dir=tmp_path / "runs" / run_id,
        ledger_path=tmp_path / "ledger.sqlite3",
        run_id=run_id,
        task="synthetic_test",
        metadata=kwargs.get("metadata", {"hypothesis": "synthetic", "seed": 17}),
    )


def _events(path: Path, experiment_id: str) -> list[dict]:
    connection = sqlite3.connect(path)
    try:
        return [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload_json FROM events WHERE experiment_id=? ORDER BY sequence",
                (experiment_id,),
            )
        ]
    finally:
        connection.close()


def _epoch(recorder: ExperimentRecorder, epoch: int = 1) -> dict:
    return recorder.record_epoch(
        epoch,
        metrics={"macro_f1": 0.5, "epochs_completed": epoch},
        environment={"device": "synthetic_cpu"},
        peak_vram={"available": True, "value_mb": 0.0, "reason": "cpu_execution"},
    )


def test_events_preserve_metadata_metrics_and_interrupt_in_both_databases(tmp_path: Path):
    metadata = {"hypothesis": "synthetic", "configuration": {"seed": 17}}
    recorder = _recorder(tmp_path, metadata=metadata)
    metadata["configuration"]["seed"] = 999
    first = recorder.latest_event
    assert first["metadata"]["configuration"]["seed"] == 17
    assert first["metrics"] is None
    assert first["metrics_reason"] == "no_completed_measurement"
    assert first["peak_vram"]["reason"] == "not_measured"
    epoch = _epoch(recorder)
    terminal = recorder.fail(KeyboardInterrupt())
    assert terminal["event_type"] == "interrupted"
    assert terminal["metrics"] == epoch["metrics"]
    assert terminal["environment"] == epoch["environment"]
    assert terminal["peak_vram"] == epoch["peak_vram"]
    assert terminal["runtime_seconds"] >= epoch["runtime_seconds"]
    events = _events(recorder.ledger_path, recorder.experiment_id)
    assert events == _events(recorder.events_path, recorder.experiment_id)
    assert [event["event_type"] for event in events] == ["running", "epoch", "interrupted"]


@pytest.mark.parametrize("error", [RuntimeError("synthetic failure"), SystemExit(2)])
def test_early_failure_records_explicit_unavailable_measurements(tmp_path: Path, error):
    recorder = _recorder(tmp_path)
    result = recorder.fail(error)
    assert result["event_type"] == ("interrupted" if isinstance(error, SystemExit) else "failed")
    assert result["failure"]["type"] == type(error).__name__
    assert result["metrics"] is None
    assert result["metrics_reason"]
    assert result["environment"]["reason"]
    assert result["peak_vram"]["reason"]


def test_failure_retains_fingerprints_of_already_written_artifacts(tmp_path: Path):
    recorder = _recorder(tmp_path)
    artifacts = {"report": {"sha256": "a" * 64, "path": "synthetic-report.json"}}
    result = recorder.fail(RuntimeError("publication failure"), artifacts=artifacts)
    assert result["artifacts"] == artifacts
    assert recorder.latest_event["artifacts"] == artifacts


def test_completed_run_cannot_be_reused_or_mutated(tmp_path: Path):
    recorder = _recorder(tmp_path)
    _epoch(recorder)
    recorder.complete(metrics={"selection_value": 0.5}, environment={}, peak_vram={})
    with pytest.raises(ExperimentRecordingError, match="terminal"):
        recorder.fail(RuntimeError("later"))
    with pytest.raises(ExperimentRecordingError, match="terminal"):
        _epoch(recorder, 2)
    with pytest.raises(ExperimentRecordingError, match="not empty"):
        _recorder(tmp_path)
    for path in (recorder.events_path, recorder.ledger_path):
        with sqlite3.connect(path) as connection:
            for statement in (
                "DELETE FROM events",
                "UPDATE events SET event_type='running'",
                "DELETE FROM experiments",
                "UPDATE experiments SET metadata_json='{}'",
                "INSERT OR REPLACE INTO events SELECT * FROM events WHERE sequence=1",
            ):
                with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                    connection.execute(statement)
            with pytest.raises(sqlite3.IntegrityError, match="identity already exists"):
                connection.execute("INSERT OR REPLACE INTO experiments SELECT * FROM experiments")


def test_duplicate_central_run_identity_rolls_back_without_running_orphan(tmp_path: Path):
    original = _recorder(tmp_path)
    duplicate_dir = tmp_path / "duplicate"
    with pytest.raises(sqlite3.IntegrityError):
        ExperimentRecorder(
            run_dir=duplicate_dir,
            ledger_path=original.ledger_path,
            run_id=original.run_id,
            task=original.task,
            metadata={},
        )
    assert not (duplicate_dir / "experiment.sqlite3").exists()
    with sqlite3.connect(original.ledger_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1


def test_failed_second_database_write_rolls_back_central_append(tmp_path: Path):
    recorder = _recorder(tmp_path)
    with sqlite3.connect(recorder.events_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_epoch BEFORE INSERT ON events "
            "WHEN NEW.event_type='epoch' BEGIN SELECT RAISE(ABORT, 'synthetic disk failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="synthetic disk failure"):
        _epoch(recorder)
    assert len(_events(recorder.ledger_path, recorder.experiment_id)) == 1
    assert _events(recorder.ledger_path, recorder.experiment_id) == _events(
        recorder.events_path, recorder.experiment_id
    )


def test_concurrent_writers_keep_every_event_without_overwrite(tmp_path: Path):
    def record(index: int) -> ExperimentRecorder:
        recorder = _recorder(tmp_path, run_id=f"parallel-{index}")
        for epoch in range(1, 4):
            _epoch(recorder, epoch)
        recorder.complete(metrics={"selection_value": index}, environment={}, peak_vram={})
        return recorder

    with ThreadPoolExecutor(max_workers=8) as executor:
        recorders = list(executor.map(record, range(12)))
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0] == 12
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 60
    assert len({recorder.experiment_id for recorder in recorders}) == 12
    for recorder in recorders:
        assert _events(recorder.ledger_path, recorder.experiment_id) == _events(
            recorder.events_path, recorder.experiment_id
        )


def test_invalid_event_does_not_append_or_replace_previous_measurements(tmp_path: Path):
    recorder = _recorder(tmp_path)
    _epoch(recorder)
    with pytest.raises(ExperimentRecordingError, match="increase"):
        _epoch(recorder)
    with pytest.raises(ExperimentRecordingError, match="positive integer"):
        _epoch(recorder, 1.5)
    with pytest.raises(ValueError):
        recorder.complete(metrics={"score": float("nan")}, environment={}, peak_vram={})
    assert len(_events(recorder.ledger_path, recorder.experiment_id)) == 2


def test_constructor_failure_does_not_commit_a_running_event(tmp_path: Path, monkeypatch):
    original = ExperimentRecorder._insert

    def broken(self, connection, **kwargs):
        original(self, connection, **kwargs)
        raise RuntimeError("synthetic setup failure after inserts")

    monkeypatch.setattr(ExperimentRecorder, "_insert", broken)
    with pytest.raises(RuntimeError, match="synthetic setup failure"):
        _recorder(tmp_path)
    assert not (tmp_path / "runs/synthetic/experiment.sqlite3").exists()
    with sqlite3.connect(tmp_path / "ledger.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()[0]
            == 0
        )
