"""Append-only experiment events without importing a model or training framework."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TERMINAL = {"complete", "failed", "interrupted"}


class ExperimentRecordingError(RuntimeError):
    """An experiment identity or event transition violates the ledger contract."""


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


class ExperimentRecorder:
    """Record a new run in a central ledger and its own immutable event database.

    Both databases use rollback journals on the same filesystem. SQLite's
    attached-database transaction commits each event to both or to neither.
    Run directories must be new/empty; existing runs are never resumed or reset.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        ledger_path: Path,
        run_id: str,
        task: str,
        metadata: dict[str, Any],
    ) -> None:
        if not run_id.strip() or not task.strip():
            raise ExperimentRecordingError("run_id and task must be non-empty")
        # Freeze caller-owned metadata before touching the filesystem.
        metadata_json = _json(metadata)
        frozen_metadata = json.loads(metadata_json)
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.ledger_path = Path(ledger_path).expanduser().resolve()
        self.events_path = self.run_dir / "experiment.sqlite3"
        if self.ledger_path == self.events_path or self.ledger_path.is_relative_to(self.run_dir):
            raise ExperimentRecordingError("central ledger must be outside the run directory")
        self.run_id = run_id
        self.task = task
        self.experiment_id = uuid.uuid4().hex
        self._started = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if any(self.run_dir.iterdir()):
            raise ExperimentRecordingError("run directory is not empty; choose a new run directory")
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if self.run_dir.stat().st_dev != self.ledger_path.parent.stat().st_dev:
            raise ExperimentRecordingError("run events and central ledger must share a filesystem")
        # Exclusive creation reserves an empty directory against another recorder.
        try:
            with self.events_path.open("xb"):
                pass
        except FileExistsError as exc:
            raise ExperimentRecordingError("run directory is already reserved") from exc
        try:
            with self._transaction() as connection:
                created_at = datetime.now(UTC).isoformat()
                for schema in ("main", "run"):
                    self._create_schema(connection, schema)
                    connection.execute(
                        f"INSERT INTO {schema}.experiments VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            self.experiment_id,
                            run_id,
                            task,
                            str(self.run_dir),
                            metadata_json,
                            created_at,
                        ),
                    )
                self._insert(
                    connection,
                    event_type="running",
                    metrics=None,
                    environment=frozen_metadata.get("environment"),
                    peak_vram=frozen_metadata.get("peak_vram"),
                    runtime_seconds=0.0,
                    extra={"metadata": frozen_metadata},
                )
        except BaseException:
            # The transaction has rolled back; do not leave a false running run.
            self.events_path.unlink(missing_ok=True)
            raise

    @contextmanager
    def _transaction(self):
        connection = sqlite3.connect(str(self.ledger_path), timeout=30, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("ATTACH DATABASE ? AS run", (str(self.events_path),))
            connection.execute("PRAGMA run.journal_mode=DELETE")
            connection.execute("PRAGMA run.synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection, schema: str) -> None:
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS {schema}.experiments (
                experiment_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task TEXT NOT NULL,
                run_dir TEXT NOT NULL UNIQUE,
                metadata_json TEXT NOT NULL,
                created_at_utc TEXT NOT NULL,
                UNIQUE(task, run_id)
            )"""
        )
        connection.execute(
            f"""CREATE TABLE IF NOT EXISTS {schema}.events (
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                epoch INTEGER,
                payload_json TEXT NOT NULL,
                PRIMARY KEY(experiment_id, sequence)
            )"""
        )
        for table in ("experiments", "events"):
            for action in ("UPDATE", "DELETE"):
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS {schema}.{table}_no_{action.lower()}
                    BEFORE {action} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'experiment ledger is append-only'); END"""
                )
        # REPLACE can bypass DELETE triggers on a connection with SQLite's
        # default recursive_triggers setting, so reject collisions before INSERT.
        connection.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {schema}.experiments_no_replace
            BEFORE INSERT ON experiments WHEN EXISTS (
                SELECT 1 FROM experiments WHERE experiment_id=NEW.experiment_id
                OR run_dir=NEW.run_dir OR (task=NEW.task AND run_id=NEW.run_id)
            ) BEGIN SELECT RAISE(ABORT, 'experiment identity already exists'); END"""
        )
        connection.execute(
            f"""CREATE TRIGGER IF NOT EXISTS {schema}.events_no_replace
            BEFORE INSERT ON events WHEN EXISTS (
                SELECT 1 FROM events WHERE experiment_id=NEW.experiment_id
                AND sequence=NEW.sequence
            ) BEGIN SELECT RAISE(ABORT, 'experiment ledger is append-only'); END"""
        )

    @property
    def latest_event(self) -> dict[str, Any]:
        with closing(sqlite3.connect(str(self.events_path), timeout=30)) as connection:
            row = connection.execute(
                "SELECT payload_json FROM events WHERE experiment_id=? "
                "ORDER BY sequence DESC LIMIT 1",
                (self.experiment_id,),
            ).fetchone()
        if row is None:
            raise ExperimentRecordingError("experiment has no committed events")
        return json.loads(row[0])

    def _insert(
        self,
        connection: sqlite3.Connection,
        *,
        event_type: str,
        metrics: dict[str, Any] | None,
        environment: dict[str, Any] | None,
        peak_vram: dict[str, Any] | None,
        runtime_seconds: float | None,
        epoch: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = connection.execute(
            "SELECT payload_json FROM main.events WHERE experiment_id=? "
            "ORDER BY sequence DESC LIMIT 1",
            (self.experiment_id,),
        ).fetchone()
        previous = json.loads(row[0]) if row else {}
        if previous.get("event_type") in _TERMINAL:
            raise ExperimentRecordingError("experiment already has a terminal event")
        if event_type == "epoch":
            if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1:
                raise ExperimentRecordingError("epoch must be a positive integer")
            latest_epoch = connection.execute(
                "SELECT MAX(epoch) FROM main.events WHERE experiment_id=?",
                (self.experiment_id,),
            ).fetchone()[0]
            if latest_epoch is not None and epoch <= latest_epoch:
                raise ExperimentRecordingError("epoch events must increase without overwriting")
        elapsed = (
            time.perf_counter() - self._started if runtime_seconds is None else runtime_seconds
        )
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ExperimentRecordingError("runtime_seconds must be finite and non-negative")
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "task": self.task,
            "sequence": int(previous.get("sequence", 0)) + 1,
            "event_type": event_type,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "runtime_seconds": float(elapsed),
            "epoch": epoch,
            "metrics": metrics if metrics is not None else previous.get("metrics"),
            "metrics_reason": None
            if metrics is not None
            else previous.get("metrics_reason", "no_completed_measurement"),
            "environment": environment
            if environment is not None
            else previous.get("environment", {"available": False, "reason": "not_measured"}),
            "peak_vram": peak_vram
            if peak_vram is not None
            else previous.get(
                "peak_vram", {"available": False, "value_mb": None, "reason": "not_measured"}
            ),
            **(extra or {}),
        }
        encoded = _json(payload)
        values = (self.experiment_id, payload["sequence"], event_type, epoch, encoded)
        for schema in ("main", "run"):
            connection.execute(f"INSERT INTO {schema}.events VALUES (?, ?, ?, ?, ?)", values)
        return payload

    def record_epoch(
        self,
        epoch: int,
        *,
        metrics: dict[str, Any],
        environment: dict[str, Any],
        peak_vram: dict[str, Any],
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            return self._insert(
                connection,
                event_type="epoch",
                epoch=epoch,
                metrics=metrics,
                environment=environment,
                peak_vram=peak_vram,
                runtime_seconds=runtime_seconds,
            )

    def complete(
        self,
        *,
        metrics: dict[str, Any],
        environment: dict[str, Any],
        peak_vram: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        with self._transaction() as connection:
            return self._insert(
                connection,
                event_type="complete",
                metrics=metrics,
                environment=environment,
                peak_vram=peak_vram,
                runtime_seconds=runtime_seconds,
                extra={"artifacts": artifacts},
            )

    def fail(
        self,
        error: BaseException,
        *,
        metrics: dict[str, Any] | None = None,
        environment: dict[str, Any] | None = None,
        peak_vram: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        runtime_seconds: float | None = None,
    ) -> dict[str, Any]:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        with self._transaction() as connection:
            return self._insert(
                connection,
                event_type=status,
                metrics=metrics,
                environment=environment,
                peak_vram=peak_vram,
                runtime_seconds=runtime_seconds,
                extra={
                    "failure": {"type": type(error).__name__, "message": str(error)},
                    "artifacts": artifacts,
                },
            )
