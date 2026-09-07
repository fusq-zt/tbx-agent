from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .anatomy_runs import AnatomyRunRecord, AnatomyRunStatus, RefinementRunStatus
from .schemas import (
    ActiveScreeningSession,
    CaseRecord,
    ClassificationExecutionStatus,
    ReviewOrigin,
    ReviewRecord,
    ReviewStatus,
    ThreadState,
    UserPreferences,
    utc_now,
)

ModelT = TypeVar("ModelT", bound=BaseModel)
STORAGE_SCHEMA_VERSION = 4


class AccessDeniedError(PermissionError):
    pass


class VersionConflictError(RuntimeError):
    pass


def _dump(model: BaseModel) -> str:
    return model.model_dump_json()


def _load(model_type: type[ModelT], payload: str) -> ModelT:
    return model_type.model_validate_json(payload)


class SQLiteStore:
    """Small, explicit business-state store; no similarity search is used for cases."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA trusted_schema=OFF")
            self._conn.execute("PRAGMA wal_autocheckpoint=1000")
            self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                owner_scope TEXT NOT NULL,
                user_id TEXT,
                image_sha256 TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                review_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            );

            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(owner_scope, user_id, thread_id)
            );

            CREATE TABLE IF NOT EXISTS preferences (
                owner_scope TEXT NOT NULL,
                user_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(owner_scope, user_id)
            );

            CREATE TABLE IF NOT EXISTS screening_sessions (
                session_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                user_id TEXT,
                owner_scope TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                request_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_role TEXT NOT NULL,
                action TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                case_id TEXT,
                details TEXT NOT NULL,
                previous_event_hash TEXT,
                event_hash TEXT
            );

            CREATE TABLE IF NOT EXISTS anatomy_runs (
                run_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                user_id TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                generation_key TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                status TEXT NOT NULL,
                version INTEGER NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(case_id) REFERENCES cases(case_id)
            );

            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._migrate_subject_isolation()
            self._conn.execute(
                """CREATE INDEX IF NOT EXISTS idx_anatomy_case_updated
                ON anatomy_runs(owner_scope, user_id, case_id, updated_at DESC)"""
            )
            self._conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_anatomy_completed_generation
                ON anatomy_runs(owner_scope, user_id, case_id, generation_key)
                WHERE status='completed'"""
            )
            audit_columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(audit_events)")
            }
            if "previous_event_hash" not in audit_columns:
                self._conn.execute("ALTER TABLE audit_events ADD COLUMN previous_event_hash TEXT")
            if "event_hash" not in audit_columns:
                self._conn.execute("ALTER TABLE audit_events ADD COLUMN event_hash TEXT")
            self._conn.execute(
                """INSERT INTO schema_metadata(key, value) VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (str(STORAGE_SCHEMA_VERSION),),
            )
            self._conn.execute(f"PRAGMA user_version={STORAGE_SCHEMA_VERSION}")
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def _table_columns(self, table: str) -> dict[str, sqlite3.Row]:
        return {str(row["name"]): row for row in self._conn.execute(f"PRAGMA table_info({table})")}

    def _migrate_subject_isolation(self) -> None:
        """Migrate only identities already explicit in trusted persisted state.

        Rows whose subject or tenant cannot be proven remain quarantined/unbound.
        The migration never assigns a current caller to historical data.
        """

        case_columns = self._table_columns("cases")
        if "user_id" not in case_columns:
            self._conn.execute("ALTER TABLE cases ADD COLUMN user_id TEXT")
        for row in self._conn.execute(
            "SELECT case_id, owner_scope, user_id, payload FROM cases WHERE user_id IS NULL"
        ).fetchall():
            try:
                case = _load(CaseRecord, row["payload"])
            except Exception:
                continue
            if case.owner_scope == row["owner_scope"] and case.user_id is not None:
                self._conn.execute(
                    "UPDATE cases SET user_id=? WHERE case_id=? AND user_id IS NULL",
                    (case.user_id, row["case_id"]),
                )
        self._conn.execute("DROP INDEX IF EXISTS idx_cases_owner_sha")
        self._conn.execute(
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_cases_owner_user_sha
            ON cases(owner_scope, user_id, image_sha256) WHERE user_id IS NOT NULL"""
        )

        thread_columns = self._table_columns("threads")
        thread_pk = [
            name
            for name, row in sorted(
                thread_columns.items(), key=lambda item: int(item[1]["pk"] or 999)
            )
            if int(row["pk"] or 0) > 0
        ]
        if thread_pk == ["thread_id"]:
            if self._table_columns("threads_legacy_global_v2"):
                raise RuntimeError("legacy thread quarantine table already exists")
            self._conn.execute("ALTER TABLE threads RENAME TO threads_legacy_global_v2")
            self._conn.execute(
                """CREATE TABLE threads (
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    owner_scope TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_scope, user_id, thread_id)
                )"""
            )
            legacy_threads = self._conn.execute(
                """SELECT thread_id, user_id, owner_scope, payload, updated_at
                FROM threads_legacy_global_v2"""
            ).fetchall()
            for row in legacy_threads:
                try:
                    state = _load(ThreadState, row["payload"])
                except Exception:
                    continue
                if (
                    state.thread_id == row["thread_id"]
                    and state.user_id == row["user_id"]
                    and state.owner_scope == row["owner_scope"]
                ):
                    self._conn.execute(
                        """INSERT INTO threads
                        (thread_id, user_id, owner_scope, payload, updated_at)
                        VALUES (?, ?, ?, ?, ?)""",
                        (
                            row["thread_id"],
                            row["user_id"],
                            row["owner_scope"],
                            row["payload"],
                            row["updated_at"],
                        ),
                    )

        preference_columns = self._table_columns("preferences")
        if "owner_scope" not in preference_columns:
            if self._table_columns("preferences_legacy_unscoped_v2"):
                raise RuntimeError("legacy preference quarantine table already exists")
            self._conn.execute("ALTER TABLE preferences RENAME TO preferences_legacy_unscoped_v2")
            self._conn.execute(
                """CREATE TABLE preferences (
                    owner_scope TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_scope, user_id)
                )"""
            )

        screening_columns = self._table_columns("screening_sessions")
        if "user_id" not in screening_columns:
            self._conn.execute("ALTER TABLE screening_sessions ADD COLUMN user_id TEXT")
        for row in self._conn.execute(
            """SELECT session_id, owner_scope, user_id, payload
            FROM screening_sessions WHERE user_id IS NULL"""
        ).fetchall():
            try:
                session = _load(ActiveScreeningSession, row["payload"])
            except Exception:
                continue
            if session.owner_scope == row["owner_scope"]:
                self._conn.execute(
                    """UPDATE screening_sessions SET user_id=?
                    WHERE session_id=? AND user_id IS NULL""",
                    (session.user_id, row["session_id"]),
                )

    @staticmethod
    def _event_hash(previous_hash: str, payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(previous_hash.encode("ascii") + b"\0" + canonical).hexdigest()

    def audit(
        self,
        *,
        request_id: str,
        actor_id: str,
        actor_role: str,
        action: str,
        owner_scope: str,
        case_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        safe_details = details or {}
        occurred_at = datetime.now(UTC).isoformat()
        details_json = json.dumps(
            safe_details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            previous_row = self._conn.execute(
                "SELECT event_hash FROM audit_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            previous_hash = (
                str(previous_row["event_hash"])
                if previous_row is not None and previous_row["event_hash"]
                else "0" * 64
            )
            event_payload = {
                "occurred_at": occurred_at,
                "request_id": request_id,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "action": action,
                "owner_scope": owner_scope,
                "case_id": case_id,
                "details": json.loads(details_json),
            }
            event_hash = self._event_hash(previous_hash, event_payload)
            self._conn.execute(
                """INSERT INTO audit_events (
                    occurred_at, request_id, actor_id, actor_role,
                    action, owner_scope, case_id, details,
                    previous_event_hash, event_hash
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    occurred_at,
                    request_id,
                    actor_id,
                    actor_role,
                    action,
                    owner_scope,
                    case_id,
                    details_json,
                    previous_hash,
                    event_hash,
                ),
            )
            self._conn.commit()

    def verify_audit_chain(self) -> dict[str, Any]:
        """Verify append order and payload integrity without exposing event contents."""

        with self._lock:
            rows = self._conn.execute(
                """SELECT event_id, occurred_at, request_id, actor_id, actor_role,
                action, owner_scope, case_id, details, previous_event_hash, event_hash
                FROM audit_events ORDER BY event_id ASC"""
            ).fetchall()
        expected_previous = "0" * 64
        for row in rows:
            if row["previous_event_hash"] != expected_previous or not row["event_hash"]:
                return {
                    "valid": False,
                    "event_count": len(rows),
                    "first_invalid_event_id": row["event_id"],
                }
            try:
                details = json.loads(row["details"])
            except json.JSONDecodeError:
                return {
                    "valid": False,
                    "event_count": len(rows),
                    "first_invalid_event_id": row["event_id"],
                }
            payload = {
                "occurred_at": row["occurred_at"],
                "request_id": row["request_id"],
                "actor_id": row["actor_id"],
                "actor_role": row["actor_role"],
                "action": row["action"],
                "owner_scope": row["owner_scope"],
                "case_id": row["case_id"],
                "details": details,
            }
            calculated = self._event_hash(expected_previous, payload)
            if calculated != row["event_hash"]:
                return {
                    "valid": False,
                    "event_count": len(rows),
                    "first_invalid_event_id": row["event_id"],
                }
            expected_previous = calculated
        return {
            "valid": True,
            "event_count": len(rows),
            "head_event_hash": expected_previous if rows else None,
        }

    def integrity_check(self) -> dict[str, Any]:
        with self._lock:
            result = str(self._conn.execute("PRAGMA quick_check").fetchone()[0])
            foreign_key_rows = self._conn.execute("PRAGMA foreign_key_check").fetchall()
            user_version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        audit = self.verify_audit_chain()
        return {
            "sqlite_quick_check": result,
            "foreign_key_violation_count": len(foreign_key_rows),
            "schema_version": user_version,
            "audit_chain_valid": audit["valid"],
            "ok": result == "ok" and not foreign_key_rows and audit["valid"],
        }

    def backup_to(self, destination: str | Path) -> dict[str, Any]:
        """Create a consistent, non-overwriting SQLite backup for restore testing."""

        target = Path(destination).expanduser().resolve()
        if target == self.path.resolve():
            raise ValueError("backup destination must differ from the live database")
        if target.exists():
            raise FileExistsError(f"backup destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup_connection = sqlite3.connect(target)
        try:
            with self._lock:
                self._conn.backup(backup_connection)
            backup_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            backup_connection.close()
        return {
            "path": str(target),
            "size_bytes": target.stat().st_size,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }

    def save_case(self, case: CaseRecord) -> CaseRecord:
        if case.user_id is None:
            raise AccessDeniedError("new or updated cases require an explicit subject binding")
        case.updated_at = utc_now()
        with self._lock:
            existing = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256
                FROM cases WHERE case_id=?""",
                (case.case_id,),
            ).fetchone()
            if existing is not None:
                if existing["owner_scope"] != case.owner_scope:
                    raise AccessDeniedError("cannot change case owner scope")
                if existing["user_id"] is None:
                    raise AccessDeniedError("cannot claim a legacy unbound case")
                if existing["user_id"] != case.user_id:
                    raise AccessDeniedError("cannot change case subject binding")
                if existing["image_sha256"] != case.image_sha256:
                    raise VersionConflictError("cannot replace the image identity of a case")
            try:
                self._conn.execute(
                    """INSERT INTO cases
                    (case_id, owner_scope, user_id, image_sha256, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        owner_scope=excluded.owner_scope,
                        user_id=excluded.user_id,
                        image_sha256=excluded.image_sha256,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at""",
                    (
                        case.case_id,
                        case.owner_scope,
                        case.user_id,
                        case.image_sha256,
                        _dump(case),
                        case.updated_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise VersionConflictError(
                    "an assessment with this subject/image identity already exists"
                ) from exc
            self._conn.commit()
        return case

    def commit_localization_state(
        self,
        candidate: CaseRecord,
        *,
        expected_version: int,
    ) -> tuple[CaseRecord, bool]:
        """Atomically persist one localization observation and derived state.

        The classification evidence and its original fusion record are immutable
        across this write. Concurrent workers either reuse the same successful
        generation or receive a version conflict; a failed attempt can never
        overwrite an already completed result.
        """

        if candidate.user_id is None:
            raise AccessDeniedError("localization updates require a bound subject")
        incoming = candidate.localization_evidence
        if incoming.status == "not_requested":
            raise ValueError("localization commit requires an attempted execution state")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT owner_scope, user_id, image_sha256, payload FROM cases WHERE case_id=?",
                    (candidate.case_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"case not found: {candidate.case_id}")
                if (
                    row["owner_scope"] != candidate.owner_scope
                    or row["user_id"] is None
                    or row["user_id"] != candidate.user_id
                ):
                    raise AccessDeniedError("case localization subject binding mismatch")
                if row["image_sha256"] != candidate.image_sha256:
                    raise VersionConflictError("cannot replace localization image identity")

                current = _load(CaseRecord, row["payload"])
                existing = current.localization_evidence
                if (
                    existing.status in {"completed", "completed_no_detection"}
                    and existing.generation_key == incoming.generation_key
                ):
                    self._conn.commit()
                    return current, True
                if (
                    existing.status in {"completed", "completed_no_detection"}
                    and incoming.status == "failed"
                ):
                    self._conn.commit()
                    return current, True
                if current.record_version != expected_version:
                    raise VersionConflictError("case localization version conflict")
                if (
                    current.vision_evidence != candidate.vision_evidence
                    or current.fusion_decision != candidate.fusion_decision
                ):
                    raise VersionConflictError(
                        "localization cannot mutate classification or fusion evidence"
                    )

                candidate.record_version = expected_version + 1
                candidate.updated_at = utc_now()
                self._conn.execute(
                    """UPDATE cases SET payload=?, updated_at=?
                    WHERE case_id=? AND owner_scope=? AND user_id=? AND image_sha256=?""",
                    (
                        _dump(candidate),
                        candidate.updated_at.isoformat(),
                        candidate.case_id,
                        candidate.owner_scope,
                        candidate.user_id,
                        candidate.image_sha256,
                    ),
                )
                self._conn.commit()
                return candidate, False
            except Exception:
                self._conn.rollback()
                raise

    def commit_classification_state(
        self,
        candidate: CaseRecord,
        *,
        expected_version: int,
    ) -> tuple[CaseRecord, bool]:
        """Atomically persist one on-demand classification observation.

        A successful result for the same immutable generation is reused.  A
        later failure cannot erase successful evidence, and this transition is
        forbidden from changing localization evidence or the uploaded image.
        """

        if candidate.user_id is None:
            raise AccessDeniedError("classification updates require a bound subject")
        incoming_status = candidate.classification_status
        if incoming_status == ClassificationExecutionStatus.NOT_REQUESTED:
            raise ValueError("classification commit requires an attempted execution state")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT owner_scope, user_id, image_sha256, payload FROM cases WHERE case_id=?",
                    (candidate.case_id,),
                ).fetchone()
                if row is None:
                    raise KeyError(f"case not found: {candidate.case_id}")
                if (
                    row["owner_scope"] != candidate.owner_scope
                    or row["user_id"] is None
                    or row["user_id"] != candidate.user_id
                ):
                    raise AccessDeniedError("case classification subject binding mismatch")
                if row["image_sha256"] != candidate.image_sha256:
                    raise VersionConflictError("cannot replace classification image identity")

                current = _load(CaseRecord, row["payload"])
                if (
                    current.classification_status
                    == ClassificationExecutionStatus.COMPLETED
                    and current.classification_generation_key
                    == candidate.classification_generation_key
                ):
                    self._conn.commit()
                    return current, True
                if (
                    current.classification_status
                    == ClassificationExecutionStatus.COMPLETED
                    and incoming_status
                    in {
                        ClassificationExecutionStatus.FAILED,
                        ClassificationExecutionStatus.UNAVAILABLE,
                    }
                ):
                    self._conn.commit()
                    return current, True
                if current.record_version != expected_version:
                    raise VersionConflictError("case classification version conflict")
                if current.localization_evidence != candidate.localization_evidence:
                    raise VersionConflictError(
                        "classification cannot mutate localization evidence"
                    )
                immutable_upload = (
                    "image_artifact_ref",
                    "image_sha256",
                    "image_width",
                    "image_height",
                    "image_source_format",
                    "input_transform_id",
                    "image_quality_status",
                    "image_quality_codes",
                    "consent_scope",
                )
                if any(
                    getattr(current, field) != getattr(candidate, field)
                    for field in immutable_upload
                ):
                    raise VersionConflictError(
                        "classification cannot mutate uploaded image evidence"
                    )

                candidate.record_version = expected_version + 1
                candidate.updated_at = utc_now()
                cursor = self._conn.execute(
                    """UPDATE cases SET payload=?, updated_at=?
                    WHERE case_id=? AND owner_scope=? AND user_id=? AND image_sha256=?""",
                    (
                        _dump(candidate),
                        candidate.updated_at.isoformat(),
                        candidate.case_id,
                        candidate.owner_scope,
                        candidate.user_id,
                        candidate.image_sha256,
                    ),
                )
                if cursor.rowcount != 1:
                    raise VersionConflictError("case classification update lost its target")
                self._conn.commit()
                return candidate, False
            except Exception:
                self._conn.rollback()
                raise

    def create_anatomy_run(self, run: AnatomyRunRecord) -> AnatomyRunRecord:
        """Insert an immutable pending anatomy generation request."""

        if run.status != AnatomyRunStatus.PENDING or run.record_version != 1:
            raise ValueError("new anatomy runs must be pending at version one")
        run.updated_at = utc_now()
        with self._lock:
            case_row = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256 FROM cases
                WHERE case_id=?""",
                (run.case_id,),
            ).fetchone()
            if case_row is None:
                raise KeyError(f"case not found: {run.case_id}")
            if (
                case_row["owner_scope"] != run.owner_scope
                or case_row["user_id"] is None
                or case_row["user_id"] != run.user_id
                or case_row["image_sha256"] != run.image_sha256
            ):
                raise AccessDeniedError("anatomy run identity differs from its case")
            try:
                self._conn.execute(
                    """INSERT INTO anatomy_runs(
                        run_id, case_id, owner_scope, user_id, image_sha256,
                        generation_key, backend_id, status, version, payload, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run.run_id,
                        run.case_id,
                        run.owner_scope,
                        run.user_id,
                        run.image_sha256,
                        run.generation_key,
                        run.backend_id,
                        run.status.value,
                        run.record_version,
                        _dump(run),
                        run.updated_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise VersionConflictError("anatomy run identity already exists") from exc
            self._conn.commit()
        return run

    def update_anatomy_run(
        self,
        run: AnatomyRunRecord,
        *,
        expected_version: int,
    ) -> AnatomyRunRecord:
        """Advance an anatomy run using an optimistic lifecycle transition."""

        allowed = {
            AnatomyRunStatus.PENDING: {
                AnatomyRunStatus.RUNNING,
                AnatomyRunStatus.TECHNICAL_FAILURE,
            },
            AnatomyRunStatus.RUNNING: {
                AnatomyRunStatus.COMPLETED,
                AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE,
                AnatomyRunStatus.TECHNICAL_FAILURE,
            },
        }
        with self._lock:
            row = self._conn.execute(
                """SELECT case_id, owner_scope, user_id, image_sha256,
                generation_key, backend_id, status, version
                FROM anatomy_runs WHERE run_id=?""",
                (run.run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"anatomy run not found: {run.run_id}")
            identity = (
                run.case_id,
                run.owner_scope,
                run.user_id,
                run.image_sha256,
                run.generation_key,
                run.backend_id,
            )
            observed = tuple(
                row[key]
                for key in (
                    "case_id",
                    "owner_scope",
                    "user_id",
                    "image_sha256",
                    "generation_key",
                    "backend_id",
                )
            )
            if identity != observed:
                raise AccessDeniedError("cannot change anatomy run identity")
            if int(row["version"]) != expected_version:
                raise VersionConflictError("anatomy run was modified concurrently")
            prior = AnatomyRunStatus(str(row["status"]))
            if run.status not in allowed.get(prior, set()):
                raise VersionConflictError(
                    f"invalid anatomy run transition: {prior.value}->{run.status.value}"
                )
            run.record_version = expected_version + 1
            run.updated_at = utc_now()
            try:
                cursor = self._conn.execute(
                    """UPDATE anatomy_runs SET status=?, version=?, payload=?, updated_at=?
                    WHERE run_id=? AND version=?""",
                    (
                        run.status.value,
                        run.record_version,
                        _dump(run),
                        run.updated_at.isoformat(),
                        run.run_id,
                        expected_version,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise VersionConflictError(
                    "an equivalent completed anatomy generation already exists"
                ) from exc
            if cursor.rowcount != 1:
                raise VersionConflictError("anatomy run update lost an optimistic-lock race")
            self._conn.commit()
        return run

    def get_anatomy_run(
        self,
        run_id: str,
        owner_scope: str,
        *,
        subject_user_id: str,
    ) -> AnatomyRunRecord:
        with self._lock:
            row = self._conn.execute(
                """SELECT case_id, owner_scope, user_id, image_sha256,
                generation_key, backend_id, status, version, payload
                FROM anatomy_runs WHERE run_id=?""",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"anatomy run not found: {run_id}")
        if row["owner_scope"] != owner_scope or row["user_id"] != subject_user_id:
            raise AccessDeniedError("anatomy run is outside the caller subject scope")
        run = _load(AnatomyRunRecord, row["payload"])
        persisted = (
            row["case_id"],
            row["owner_scope"],
            row["user_id"],
            row["image_sha256"],
            row["generation_key"],
            row["backend_id"],
            row["status"],
            int(row["version"]),
        )
        represented = (
            run.case_id,
            run.owner_scope,
            run.user_id,
            run.image_sha256,
            run.generation_key,
            run.backend_id,
            run.status.value,
            run.record_version,
        )
        if persisted != represented:
            raise AccessDeniedError("anatomy run persisted identity is inconsistent")
        return run

    def find_anatomy_generation(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
        generation_key: str,
    ) -> AnatomyRunRecord | None:
        """Return a reusable completed or active generation, never a failed run."""

        with self._lock:
            row = self._conn.execute(
                """SELECT run_id FROM anatomy_runs
                WHERE case_id=? AND owner_scope=? AND user_id=? AND generation_key=?
                AND status IN ('completed', 'running', 'pending')
                ORDER BY CASE status
                    WHEN 'completed' THEN 0 WHEN 'running' THEN 1 ELSE 2 END,
                    updated_at DESC LIMIT 1""",
                (case_id, owner_scope, user_id, generation_key),
            ).fetchone()
        if row is None:
            return None
        return self.get_anatomy_run(
            str(row["run_id"]),
            owner_scope,
            subject_user_id=user_id,
        )

    def list_anatomy_runs(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
        limit: int = 20,
    ) -> list[AnatomyRunRecord]:
        if not 1 <= limit <= 100:
            raise ValueError("anatomy run limit must be in [1, 100]")
        with self._lock:
            rows = self._conn.execute(
                """SELECT run_id FROM anatomy_runs
                WHERE case_id=? AND owner_scope=? AND user_id=?
                ORDER BY updated_at DESC LIMIT ?""",
                (case_id, owner_scope, user_id, limit),
            ).fetchall()
        return [
            self.get_anatomy_run(
                str(row["run_id"]),
                owner_scope,
                subject_user_id=user_id,
            )
            for row in rows
        ]

    def find_latest_completed_anatomy_run(
        self,
        *,
        case_id: str,
        owner_scope: str,
        user_id: str,
    ) -> AnatomyRunRecord | None:
        """Return the newest completed run within the exact subject boundary."""

        with self._lock:
            row = self._conn.execute(
                """SELECT run_id FROM anatomy_runs
                WHERE case_id=? AND owner_scope=? AND user_id=?
                AND status IN ('completed', 'completed_with_refinement_failure')
                ORDER BY updated_at DESC LIMIT 1""",
                (case_id, owner_scope, user_id),
            ).fetchone()
        if row is None:
            return None
        return self.get_anatomy_run(
            str(row["run_id"]),
            owner_scope,
            subject_user_id=user_id,
        )

    def reconcile_interrupted_anatomy_runs(
        self,
        *,
        error_code: str = "worker_interrupted",
    ) -> list[AnatomyRunRecord]:
        """Atomically fail non-terminal runs left by a prior service process.

        The application currently supports one API process per SQLite database.
        Under that explicit deployment contract, a non-terminal row observed at
        process startup cannot still have a live in-process worker. Marking every
        such row in one transaction prevents jobs from remaining pending forever.
        """

        if not error_code.strip() or len(error_code) > 128:
            raise ValueError("anatomy reconciliation error code is invalid")
        reconciled: list[AnatomyRunRecord] = []
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    """SELECT run_id, status, version, payload FROM anatomy_runs
                    WHERE status IN ('pending', 'running')
                    ORDER BY updated_at, run_id"""
                ).fetchall()
                for row in rows:
                    run = _load(AnatomyRunRecord, row["payload"])
                    if (
                        run.run_id != row["run_id"]
                        or run.status.value != row["status"]
                        or run.record_version != int(row["version"])
                    ):
                        raise AccessDeniedError(
                            "interrupted anatomy run persisted identity is inconsistent"
                        )
                    refinement_update = (
                        {
                            "refinement_status": RefinementRunStatus.TECHNICAL_FAILURE,
                            "refinement_error_code": "anatomy_worker_interrupted",
                        }
                        if run.refinement_status == RefinementRunStatus.PENDING
                        else {}
                    )
                    failed = AnatomyRunRecord.model_validate(
                        run.model_copy(
                            update={
                                "status": AnatomyRunStatus.TECHNICAL_FAILURE,
                                "error_code": error_code,
                                "record_version": run.record_version + 1,
                                "updated_at": utc_now(),
                                **refinement_update,
                            }
                        ).model_dump()
                    )
                    cursor = self._conn.execute(
                        """UPDATE anatomy_runs
                        SET status=?, version=?, payload=?, updated_at=?
                        WHERE run_id=? AND version=? AND status IN ('pending', 'running')""",
                        (
                            failed.status.value,
                            failed.record_version,
                            _dump(failed),
                            failed.updated_at.isoformat(),
                            failed.run_id,
                            run.record_version,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise VersionConflictError(
                            "interrupted anatomy run changed during reconciliation"
                        )
                    reconciled.append(failed)
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()
        return reconciled

    def find_case_by_hash(
        self, owner_scope: str, user_id: str, image_sha256: str
    ) -> CaseRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256, payload FROM cases
                WHERE owner_scope=? AND user_id=? AND image_sha256=?""",
                (owner_scope, user_id, image_sha256),
            ).fetchone()
        if row is None:
            return None
        case = _load(CaseRecord, row["payload"])
        if (
            case.owner_scope != row["owner_scope"]
            or case.user_id != row["user_id"]
            or case.user_id != user_id
            or case.image_sha256 != row["image_sha256"]
        ):
            raise AccessDeniedError("case subject binding is inconsistent")
        return case

    def get_case(
        self,
        case_id: str,
        owner_scope: str,
        *,
        subject_user_id: str | None = None,
    ) -> CaseRecord:
        with self._lock:
            row = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256, payload
                FROM cases WHERE case_id=?""",
                (case_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"case not found: {case_id}")
        if row["owner_scope"] != owner_scope:
            raise AccessDeniedError("case is outside the caller owner scope")
        case = _load(CaseRecord, row["payload"])
        if (
            row["user_id"] is None
            or case.owner_scope != row["owner_scope"]
            or case.user_id != row["user_id"]
            or case.image_sha256 != row["image_sha256"]
        ):
            raise AccessDeniedError("case has no valid subject binding")
        if subject_user_id is not None and row["user_id"] != subject_user_id:
            raise AccessDeniedError("case is outside the caller subject scope")
        return case

    def save_review(self, review: ReviewRecord) -> ReviewRecord:
        with self._lock:
            case_row = self._conn.execute(
                "SELECT owner_scope, user_id FROM cases WHERE case_id=?",
                (review.case_id,),
            ).fetchone()
            if case_row is None:
                raise KeyError(f"case not found: {review.case_id}")
            if case_row["owner_scope"] != review.owner_scope:
                raise AccessDeniedError("review owner scope differs from its case")
            if case_row["user_id"] is None:
                raise AccessDeniedError("review case has no subject binding")
            existing = self._conn.execute(
                "SELECT case_id, owner_scope FROM reviews WHERE review_id=?",
                (review.review_id,),
            ).fetchone()
            if existing is not None and (
                existing["case_id"] != review.case_id
                or existing["owner_scope"] != review.owner_scope
            ):
                raise AccessDeniedError("cannot change review case identity or owner scope")
            self._conn.execute(
                """INSERT INTO reviews
                (review_id, case_id, owner_scope, version, status, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    version=excluded.version,
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at""",
                (
                    review.review_id,
                    review.case_id,
                    review.owner_scope,
                    review.version,
                    review.status.value,
                    _dump(review),
                    utc_now().isoformat(),
                ),
            )
            self._conn.commit()
        return review

    def save_case_with_review(
        self, case: CaseRecord, review: ReviewRecord
    ) -> tuple[CaseRecord, ReviewRecord]:
        """Atomically create/update one case and its pending review record."""

        if (
            review.case_id != case.case_id
            or review.owner_scope != case.owner_scope
            or case.review_id != review.review_id
            or case.review_status != ReviewStatus.PENDING
        ):
            raise ValueError("case and review identities or pending state do not match")
        case.updated_at = utc_now()
        if case.user_id is None:
            raise AccessDeniedError("case-review creation requires a subject binding")
        with self._lock:
            existing_case = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256
                FROM cases WHERE case_id=?""",
                (case.case_id,),
            ).fetchone()
            if existing_case is not None:
                if existing_case["owner_scope"] != case.owner_scope:
                    raise AccessDeniedError("cannot change case owner scope")
                if existing_case["user_id"] is None:
                    raise AccessDeniedError("cannot claim a legacy unbound case")
                if existing_case["user_id"] != case.user_id:
                    raise AccessDeniedError("cannot change case subject binding")
                if existing_case["image_sha256"] != case.image_sha256:
                    raise VersionConflictError("cannot replace the image identity of a case")
            existing_review = self._conn.execute(
                "SELECT case_id, owner_scope FROM reviews WHERE review_id=?",
                (review.review_id,),
            ).fetchone()
            if existing_review is not None and (
                existing_review["case_id"] != review.case_id
                or existing_review["owner_scope"] != review.owner_scope
            ):
                raise AccessDeniedError("cannot change review case identity or owner scope")
            try:
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO cases
                    (case_id, owner_scope, user_id, image_sha256, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(case_id) DO UPDATE SET
                        owner_scope=excluded.owner_scope,
                        user_id=excluded.user_id,
                        image_sha256=excluded.image_sha256,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at""",
                        (
                            case.case_id,
                            case.owner_scope,
                            case.user_id,
                            case.image_sha256,
                            _dump(case),
                            case.updated_at.isoformat(),
                        ),
                    )
                    self._conn.execute(
                        """INSERT INTO reviews
                    (review_id, case_id, owner_scope, version, status, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_id) DO UPDATE SET
                        version=excluded.version,
                        status=excluded.status,
                        payload=excluded.payload,
                        updated_at=excluded.updated_at""",
                        (
                            review.review_id,
                            review.case_id,
                            review.owner_scope,
                            review.version,
                            review.status.value,
                            _dump(review),
                            utc_now().isoformat(),
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise VersionConflictError(
                    "an assessment or review identity already exists"
                ) from exc
        return case, review

    def ensure_batch_review(
        self,
        review: ReviewRecord,
    ) -> tuple[CaseRecord, ReviewRecord]:
        """Create one batch review once without reopening an existing task.

        The caller supplies a deterministic review identifier derived from the
        tenant, batch and immutable case identity. ``INSERT OR IGNORE`` makes a
        concurrent retry idempotent across API processes; an existing completed
        task is returned unchanged rather than being reset to pending.
        """

        if review.origin != ReviewOrigin.BATCH_SCREENING:
            raise ValueError("batch enrollment requires a batch-screening review")
        if review.status != ReviewStatus.PENDING or review.version != 1:
            raise ValueError("new batch reviews must be pending at version one")
        if review.batch_id is None or review.batch_item_id is None:
            raise ValueError("batch review provenance is incomplete")

        def validate_existing(row: sqlite3.Row) -> ReviewRecord:
            persisted = _load(ReviewRecord, row["payload"])
            if (
                persisted.review_id != review.review_id
                or persisted.case_id != review.case_id
                or persisted.owner_scope != review.owner_scope
                or persisted.origin != ReviewOrigin.BATCH_SCREENING
                or persisted.batch_id != review.batch_id
                or persisted.batch_item_id != review.batch_item_id
            ):
                raise AccessDeniedError("batch review identity or provenance mismatch")
            if persisted.status.value != row["status"] or persisted.version != int(
                row["version"]
            ):
                raise AccessDeniedError("batch review persisted state is inconsistent")
            return persisted

        with self._lock:
            case_row = self._conn.execute(
                """SELECT owner_scope, user_id, image_sha256, payload
                FROM cases WHERE case_id=?""",
                (review.case_id,),
            ).fetchone()
            if case_row is None:
                raise KeyError(f"case not found: {review.case_id}")
            case = _load(CaseRecord, case_row["payload"])
            if (
                case_row["owner_scope"] != review.owner_scope
                or case_row["user_id"] is None
                or case.owner_scope != case_row["owner_scope"]
                or case.user_id != case_row["user_id"]
                or case.image_sha256 != case_row["image_sha256"]
            ):
                raise AccessDeniedError("batch review case identity is inconsistent")

            existing = self._conn.execute(
                """SELECT case_id, owner_scope, version, status, payload
                FROM reviews WHERE review_id=?""",
                (review.review_id,),
            ).fetchone()
            if existing is not None:
                return case, validate_existing(existing)

            case.review_id = review.review_id
            case.review_status = ReviewStatus.PENDING
            case.record_version += 1
            case.updated_at = utc_now()
            with self._conn:
                cursor = self._conn.execute(
                    """INSERT OR IGNORE INTO reviews
                    (review_id, case_id, owner_scope, version, status, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        review.review_id,
                        review.case_id,
                        review.owner_scope,
                        review.version,
                        review.status.value,
                        _dump(review),
                        case.updated_at.isoformat(),
                    ),
                )
                if cursor.rowcount == 1:
                    self._conn.execute(
                        """UPDATE cases SET payload=?, updated_at=?
                        WHERE case_id=? AND owner_scope=?""",
                        (
                            _dump(case),
                            case.updated_at.isoformat(),
                            case.case_id,
                            case.owner_scope,
                        ),
                    )

            if cursor.rowcount == 1:
                return case, review

            # Another process won the deterministic insert race. Reload both
            # records and return the winner without changing its state.
            existing = self._conn.execute(
                """SELECT case_id, owner_scope, version, status, payload
                FROM reviews WHERE review_id=?""",
                (review.review_id,),
            ).fetchone()
            if existing is None:
                raise VersionConflictError("batch review insert race did not produce a record")
            persisted_case = self.get_case(review.case_id, review.owner_scope)
            return persisted_case, validate_existing(existing)

    def get_review(
        self,
        review_id: str,
        owner_scope: str,
        *,
        subject_user_id: str | None = None,
    ) -> ReviewRecord:
        with self._lock:
            row = self._conn.execute(
                """SELECT r.case_id, r.owner_scope, r.payload,
                c.user_id AS case_user_id
                FROM reviews AS r JOIN cases AS c ON c.case_id=r.case_id
                WHERE r.review_id=?""",
                (review_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"review not found: {review_id}")
        if row["owner_scope"] != owner_scope:
            raise AccessDeniedError("review is outside the caller owner scope")
        if row["case_user_id"] is None:
            raise AccessDeniedError("review case has no subject binding")
        if subject_user_id is not None and row["case_user_id"] != subject_user_id:
            raise AccessDeniedError("review is outside the caller subject scope")
        review = _load(ReviewRecord, row["payload"])
        if review.case_id != row["case_id"] or review.owner_scope != row["owner_scope"]:
            raise AccessDeniedError("review persisted identity is inconsistent")
        return review

    def complete_review(
        self,
        review: ReviewRecord,
        *,
        expected_version: int,
    ) -> ReviewRecord:
        with self._lock:
            row = self._conn.execute(
                """SELECT r.case_id, r.owner_scope, r.version, r.status,
                c.user_id AS case_user_id FROM reviews AS r
                JOIN cases AS c ON c.case_id=r.case_id WHERE r.review_id=?""",
                (review.review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"review not found: {review.review_id}")
            if row["case_id"] != review.case_id or row["owner_scope"] != review.owner_scope:
                raise AccessDeniedError("review identity or owner scope mismatch")
            if row["case_user_id"] is None:
                raise AccessDeniedError("review case has no subject binding")
            if row["status"] != ReviewStatus.PENDING.value:
                raise VersionConflictError("review is no longer pending")
            if int(row["version"]) != expected_version:
                raise VersionConflictError("review was modified by another reviewer")
            review.version = expected_version + 1
            review.status = ReviewStatus.COMPLETED
            cursor = self._conn.execute(
                """UPDATE reviews SET version=?, status=?, payload=?, updated_at=?
                WHERE review_id=? AND version=?""",
                (
                    review.version,
                    review.status.value,
                    _dump(review),
                    utc_now().isoformat(),
                    review.review_id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise VersionConflictError("review update lost an optimistic-lock race")
            self._conn.commit()
        return review

    def complete_review_with_case_update(
        self,
        review: ReviewRecord,
        *,
        expected_version: int,
    ) -> tuple[ReviewRecord, CaseRecord]:
        """Commit the review and its case status in one SQLite transaction."""

        with self._lock:
            row = self._conn.execute(
                """SELECT r.case_id, r.owner_scope, r.version, r.status,
                c.user_id AS case_user_id FROM reviews AS r
                JOIN cases AS c ON c.case_id=r.case_id WHERE r.review_id=?""",
                (review.review_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"review not found: {review.review_id}")
            if row["case_id"] != review.case_id or row["owner_scope"] != review.owner_scope:
                raise AccessDeniedError("review identity or owner scope mismatch")
            if row["case_user_id"] is None:
                raise AccessDeniedError("review case has no subject binding")
            if row["status"] != ReviewStatus.PENDING.value:
                raise VersionConflictError("review is no longer pending")
            if int(row["version"]) != expected_version:
                raise VersionConflictError("review was modified by another reviewer")
            case_row = self._conn.execute(
                """SELECT owner_scope, user_id, payload
                FROM cases WHERE case_id=?""",
                (review.case_id,),
            ).fetchone()
            if case_row is None:
                raise KeyError(f"case not found: {review.case_id}")
            if case_row["owner_scope"] != review.owner_scope:
                raise AccessDeniedError("review owner scope differs from its case")
            case = _load(CaseRecord, case_row["payload"])
            if (
                case_row["user_id"] is None
                or case.owner_scope != case_row["owner_scope"]
                or case.user_id != case_row["user_id"]
            ):
                raise AccessDeniedError("review case has no valid subject binding")
            review.version = expected_version + 1
            review.status = ReviewStatus.COMPLETED
            case.review_status = ReviewStatus.COMPLETED
            case.record_version += 1
            case.updated_at = utc_now()
            with self._conn:
                cursor = self._conn.execute(
                    """UPDATE reviews SET version=?, status=?, payload=?, updated_at=?
                    WHERE review_id=? AND version=?""",
                    (
                        review.version,
                        review.status.value,
                        _dump(review),
                        utc_now().isoformat(),
                        review.review_id,
                        expected_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise VersionConflictError("review update lost an optimistic-lock race")
                self._conn.execute(
                    "UPDATE cases SET payload=?, updated_at=? WHERE case_id=? AND owner_scope=?",
                    (
                        _dump(case),
                        case.updated_at.isoformat(),
                        case.case_id,
                        case.owner_scope,
                    ),
                )
        return review, case

    def list_pending_reviews(
        self,
        owner_scope: str,
        *,
        subject_user_id: str | None = None,
        origin: ReviewOrigin | None = None,
    ) -> list[ReviewRecord]:
        subject_clause = "" if subject_user_id is None else " AND c.user_id=?"
        args: tuple[str, ...] = (owner_scope, ReviewStatus.PENDING.value)
        if subject_user_id is not None:
            args += (subject_user_id,)
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.payload FROM reviews AS r
                JOIN cases AS c ON c.case_id=r.case_id
                WHERE r.owner_scope=? AND r.status=? AND c.user_id IS NOT NULL"""
                + subject_clause
                + " ORDER BY r.updated_at ASC",
                args,
            ).fetchall()
        reviews = [_load(ReviewRecord, row["payload"]) for row in rows]
        return reviews if origin is None else [item for item in reviews if item.origin == origin]

    def save_thread(self, state: ThreadState) -> ThreadState:
        state.updated_at = utc_now()
        state.recent_messages = state.recent_messages[-12:]
        with self._lock:
            existing = self._conn.execute(
                """SELECT user_id, owner_scope FROM threads
                WHERE owner_scope=? AND user_id=? AND thread_id=?""",
                (state.owner_scope, state.user_id, state.thread_id),
            ).fetchone()
            if existing is not None and (
                existing["user_id"] != state.user_id or existing["owner_scope"] != state.owner_scope
            ):
                raise AccessDeniedError("cannot change thread identity or owner scope")
            self._conn.execute(
                """INSERT INTO threads(thread_id, user_id, owner_scope, payload, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(owner_scope, user_id, thread_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at""",
                (
                    state.thread_id,
                    state.user_id,
                    state.owner_scope,
                    _dump(state),
                    state.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return state

    def get_or_create_thread(self, thread_id: str, user_id: str, owner_scope: str) -> ThreadState:
        with self._lock:
            row = self._conn.execute(
                """SELECT user_id, owner_scope, payload FROM threads
                WHERE owner_scope=? AND user_id=? AND thread_id=?""",
                (owner_scope, user_id, thread_id),
            ).fetchone()
        if row is None:
            return self.save_thread(
                ThreadState(thread_id=thread_id, user_id=user_id, owner_scope=owner_scope)
            )
        if row["user_id"] != user_id or row["owner_scope"] != owner_scope:
            raise AccessDeniedError("thread identity or owner scope mismatch")
        return _load(ThreadState, row["payload"])

    def get_preferences(self, user_id: str, *, owner_scope: str | None = None) -> UserPreferences:
        if owner_scope is None:
            raise AccessDeniedError("preference access requires an owner scope")
        with self._lock:
            row = self._conn.execute(
                """SELECT payload FROM preferences
                WHERE owner_scope=? AND user_id=?""",
                (owner_scope, user_id),
            ).fetchone()
        if row is None:
            return UserPreferences(user_id=user_id)
        preferences = _load(UserPreferences, row["payload"])
        if preferences.user_id != user_id:
            raise AccessDeniedError("preference subject binding is inconsistent")
        return preferences

    def save_preferences(
        self, preferences: UserPreferences, *, owner_scope: str | None = None
    ) -> UserPreferences:
        if owner_scope is None:
            raise AccessDeniedError("preference writes require an owner scope")
        preferences.updated_at = utc_now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO preferences(owner_scope, user_id, payload, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(owner_scope, user_id) DO UPDATE SET payload=excluded.payload,
                updated_at=excluded.updated_at""",
                (
                    owner_scope,
                    preferences.user_id,
                    _dump(preferences),
                    preferences.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return preferences

    def save_screening_session(self, session: ActiveScreeningSession) -> ActiveScreeningSession:
        session.updated_at = utc_now()
        with self._lock:
            thread_row = self._conn.execute(
                """SELECT user_id, owner_scope FROM threads
                WHERE owner_scope=? AND user_id=? AND thread_id=?""",
                (session.owner_scope, session.user_id, session.thread_id),
            ).fetchone()
            if thread_row is None:
                raise KeyError(f"thread not found: {session.thread_id}")
            if (
                thread_row["user_id"] != session.user_id
                or thread_row["owner_scope"] != session.owner_scope
            ):
                raise AccessDeniedError("screening session differs from its thread identity")
            existing = self._conn.execute(
                """SELECT thread_id, user_id, owner_scope
                FROM screening_sessions WHERE session_id=?""",
                (session.session_id,),
            ).fetchone()
            if existing is not None and (
                existing["thread_id"] != session.thread_id
                or existing["user_id"] != session.user_id
                or existing["owner_scope"] != session.owner_scope
            ):
                raise AccessDeniedError("cannot change screening session identity or owner scope")
            self._conn.execute(
                """INSERT INTO screening_sessions
                (session_id, thread_id, user_id, owner_scope, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at""",
                (
                    session.session_id,
                    session.thread_id,
                    session.user_id,
                    session.owner_scope,
                    _dump(session),
                    session.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return session

    def get_screening_session(
        self,
        session_id: str,
        owner_scope: str,
        *,
        subject_user_id: str | None = None,
    ) -> ActiveScreeningSession:
        with self._lock:
            row = self._conn.execute(
                """SELECT owner_scope, user_id, payload
                FROM screening_sessions WHERE session_id=?""",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"screening session not found: {session_id}")
        if row["owner_scope"] != owner_scope:
            raise AccessDeniedError("screening session is outside the caller owner scope")
        session = _load(ActiveScreeningSession, row["payload"])
        if (
            row["user_id"] is None
            or session.owner_scope != row["owner_scope"]
            or session.user_id != row["user_id"]
        ):
            raise AccessDeniedError("screening session has no valid subject binding")
        if subject_user_id is not None and row["user_id"] != subject_user_id:
            raise AccessDeniedError("screening session is outside the caller subject scope")
        return session

    def audit_count(self, action: str | None = None) -> int:
        query = "SELECT COUNT(*) AS n FROM audit_events"
        args: tuple[Any, ...] = ()
        if action:
            query += " WHERE action=?"
            args = (action,)
        with self._lock:
            return int(self._conn.execute(query, args).fetchone()["n"])
