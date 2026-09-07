from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

from tbx_agent.anatomy_runs import AnatomyRunRecord, AnatomyRunStatus
from tbx_agent.schemas import (
    ActiveScreeningSession,
    CaseRecord,
    ReviewOrigin,
    ReviewRecord,
    ReviewStatus,
    ThreadState,
    UserPreferences,
)
from tbx_agent.storage import AccessDeniedError, SQLiteStore, VersionConflictError


@pytest.fixture
def store(tmp_path) -> Iterator[SQLiteStore]:
    instance = SQLiteStore(tmp_path / "state.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def _case(
    case_id: str,
    owner_scope: str,
    image_sha256: str,
    *,
    user_id: str | None = "user",
) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        owner_scope=owner_scope,
        user_id=user_id,
        image_artifact_ref=f"artifact://{case_id}/cxr.png",
        image_sha256=image_sha256,
        image_width=512,
        image_height=512,
        consent_scope="screening_only",
    )


def _review(review_id: str, case: CaseRecord) -> ReviewRecord:
    return ReviewRecord(
        review_id=review_id,
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        trigger_reasons=["rank03_branch_disagreement"],
    )


def _batch_review(review_id: str, case: CaseRecord) -> ReviewRecord:
    return ReviewRecord(
        review_id=review_id,
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        trigger_reasons=["batch_non_tb_abnormal"],
        origin=ReviewOrigin.BATCH_SCREENING,
        batch_id="batch-001",
        batch_item_id="item-001",
    )


def _anatomy_run(run_id: str, case: CaseRecord, generation: str) -> AnatomyRunRecord:
    return AnatomyRunRecord(
        run_id=run_id,
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id=case.user_id or "user",
        image_sha256=case.image_sha256,
        generation_key=generation * 64,
        backend_id="test-anatomy",
    )


def test_startup_reconciliation_atomically_fails_interrupted_anatomy_runs(
    store: SQLiteStore,
) -> None:
    case = store.save_case(_case("case-worker-restart", "tenant:restart", "9" * 64))
    pending = store.create_anatomy_run(_anatomy_run("run-pending", case, "a"))
    running = store.create_anatomy_run(_anatomy_run("run-running", case, "b"))
    running.status = AnatomyRunStatus.RUNNING
    running = store.update_anatomy_run(running, expected_version=1)

    reconciled = store.reconcile_interrupted_anatomy_runs()

    assert {run.run_id for run in reconciled} == {pending.run_id, running.run_id}
    failed_pending = store.get_anatomy_run(
        pending.run_id,
        case.owner_scope,
        subject_user_id=case.user_id or "user",
    )
    failed_running = store.get_anatomy_run(
        running.run_id,
        case.owner_scope,
        subject_user_id=case.user_id or "user",
    )
    assert failed_pending.status == AnatomyRunStatus.TECHNICAL_FAILURE
    assert failed_pending.error_code == "worker_interrupted"
    assert failed_pending.record_version == 2
    assert failed_running.status == AnatomyRunStatus.TECHNICAL_FAILURE
    assert failed_running.error_code == "worker_interrupted"
    assert failed_running.record_version == 3
    assert store.reconcile_interrupted_anatomy_runs() == []


def test_case_subject_image_unique_race_is_a_domain_conflict(
    store: SQLiteStore,
) -> None:
    first = _case("case-generation-one", "tenant:race", "8" * 64)
    second = _case("case-generation-two", "tenant:race", "8" * 64)
    store.save_case(first)

    with pytest.raises(VersionConflictError, match="subject/image identity"):
        store.save_case(second)

    assert store.find_case_by_hash("tenant:race", "user", "8" * 64) == first


def _screening_session(
    session_id: str,
    *,
    thread_id: str,
    user_id: str,
    owner_scope: str,
) -> ActiveScreeningSession:
    return ActiveScreeningSession(
        session_id=session_id,
        thread_id=thread_id,
        user_id=user_id,
        owner_scope=owner_scope,
        consent=True,
        guideline_rule_version="active-screening-test-v1",
        status="collecting",
    )


def test_case_hash_lookup_and_reads_require_exact_owner_scope(store: SQLiteStore) -> None:
    shared_hash = "d" * 64
    parent = _case("case-parent", "tenant:alpha", shared_hash)
    child = _case("case-child", "tenant:alpha:child", shared_hash)
    store.save_case(parent)
    store.save_case(child)

    assert store.get_case(parent.case_id, "tenant:alpha").case_id == parent.case_id
    assert store.get_case(child.case_id, "tenant:alpha:child").case_id == child.case_id
    assert store.find_case_by_hash("tenant:alpha", "user", shared_hash).case_id == parent.case_id
    assert (
        store.find_case_by_hash("tenant:alpha:child", "user", shared_hash).case_id == child.case_id
    )
    assert store.find_case_by_hash("tenant:alph", "user", shared_hash) is None

    for wrong_scope in ("tenant:alph", "tenant:alpha:child", "tenant:alpha "):
        with pytest.raises(AccessDeniedError):
            store.get_case(parent.case_id, wrong_scope)


def test_review_reads_and_pending_lists_require_exact_owner_scope(
    store: SQLiteStore,
) -> None:
    parent = store.save_case(_case("case-a", "tenant:a", "a" * 64))
    child = store.save_case(_case("case-a-child", "tenant:a:child", "b" * 64))
    parent_review = store.save_review(_review("review-a", parent))
    child_review = store.save_review(_review("review-a-child", child))

    assert store.get_review(parent_review.review_id, "tenant:a") == parent_review
    assert store.get_review(child_review.review_id, "tenant:a:child") == child_review
    assert [item.review_id for item in store.list_pending_reviews("tenant:a")] == [
        parent_review.review_id
    ]
    assert [item.review_id for item in store.list_pending_reviews("tenant:a:child")] == [
        child_review.review_id
    ]
    assert store.list_pending_reviews("tenant") == []

    with pytest.raises(AccessDeniedError):
        store.get_review(parent_review.review_id, "tenant:a:child")


def test_thread_ids_are_composite_isolated_by_tenant_and_user(
    store: SQLiteStore,
) -> None:
    created = store.get_or_create_thread("thread-1", "user-1", "tenant:a")

    assert store.get_or_create_thread("thread-1", "user-1", "tenant:a") == created
    other_user = store.get_or_create_thread("thread-1", "user-2", "tenant:a")
    other_tenant = store.get_or_create_thread("thread-1", "user-1", "tenant:a:child")
    assert other_user.user_id == "user-2"
    assert other_tenant.owner_scope == "tenant:a:child"
    assert created is not other_user


def test_existing_case_scope_cannot_be_taken_over(store: SQLiteStore) -> None:
    original = store.save_case(_case("case-protected", "tenant:owner", "f" * 64))
    forged = original.model_copy(update={"owner_scope": "tenant:attacker"})

    with pytest.raises(AccessDeniedError, match="cannot change case owner scope"):
        store.save_case(forged)

    persisted = store.get_case(original.case_id, "tenant:owner")
    assert persisted.owner_scope == "tenant:owner"
    assert persisted.image_sha256 == original.image_sha256
    with pytest.raises(AccessDeniedError):
        store.get_case(original.case_id, "tenant:attacker")


def test_same_thread_id_in_another_scope_cannot_overwrite_original(
    store: SQLiteStore,
) -> None:
    original = store.get_or_create_thread("thread-protected", "user-owner", "tenant:owner")
    forged = original.model_copy(
        update={"user_id": "user-attacker", "owner_scope": "tenant:attacker"}
    )

    store.save_thread(forged)

    persisted = store.get_or_create_thread(
        original.thread_id, original.user_id, original.owner_scope
    )
    assert persisted.user_id == "user-owner"
    assert persisted.owner_scope == "tenant:owner"
    isolated = store.get_or_create_thread(forged.thread_id, forged.user_id, forged.owner_scope)
    assert isolated.user_id == "user-attacker"


def test_existing_review_identity_and_scope_cannot_be_taken_over(
    store: SQLiteStore,
) -> None:
    owner_case = store.save_case(_case("case-owner", "tenant:owner", "1" * 64))
    attacker_case = store.save_case(_case("case-attacker", "tenant:attacker", "2" * 64))
    original = store.save_review(_review("review-protected", owner_case))
    forged = _review(original.review_id, attacker_case)

    with pytest.raises(AccessDeniedError, match="cannot change review case identity"):
        store.save_review(forged)

    persisted = store.get_review(original.review_id, "tenant:owner")
    assert persisted.case_id == owner_case.case_id
    assert persisted.owner_scope == owner_case.owner_scope
    with pytest.raises(AccessDeniedError):
        store.get_review(original.review_id, "tenant:attacker")


def test_existing_screening_session_scope_cannot_be_taken_over(
    store: SQLiteStore,
) -> None:
    store.get_or_create_thread("thread-owner", "user-owner", "tenant:owner")
    store.get_or_create_thread("thread-attacker", "user-attacker", "tenant:attacker")
    original = store.save_screening_session(
        _screening_session(
            "session-protected",
            thread_id="thread-owner",
            user_id="user-owner",
            owner_scope="tenant:owner",
        )
    )
    forged = _screening_session(
        original.session_id,
        thread_id="thread-attacker",
        user_id="user-attacker",
        owner_scope="tenant:attacker",
    )

    with pytest.raises(AccessDeniedError, match="cannot change screening session identity"):
        store.save_screening_session(forged)

    persisted = store.get_screening_session(original.session_id, "tenant:owner")
    assert persisted.thread_id == "thread-owner"
    assert persisted.owner_scope == "tenant:owner"
    with pytest.raises(AccessDeniedError):
        store.get_screening_session(original.session_id, "tenant:attacker")


def test_complete_review_rejects_forged_scope_before_update(
    store: SQLiteStore,
) -> None:
    case = store.save_case(_case("case-forged-review", "tenant:owner", "3" * 64))
    original = store.save_review(_review("review-forged-scope", case))
    forged = original.model_copy(update={"owner_scope": "tenant:attacker"})
    forged.reviewer_decision = "indeterminate"

    with pytest.raises(AccessDeniedError, match="review identity or owner scope mismatch"):
        store.complete_review(forged, expected_version=original.version)

    persisted = store.get_review(original.review_id, "tenant:owner")
    assert persisted.status is ReviewStatus.PENDING
    assert persisted.version == 1
    assert persisted.reviewer_decision is None


def test_batch_review_provenance_is_required_and_pending_filter_is_explicit(
    store: SQLiteStore,
) -> None:
    with pytest.raises(ValueError, match="require batch_id"):
        ReviewRecord(
            review_id="incomplete-batch-review",
            case_id="case-incomplete",
            owner_scope="tenant:batch",
            trigger_reasons=["test"],
            origin=ReviewOrigin.BATCH_SCREENING,
        )

    legacy_case = store.save_case(_case("legacy-case", "tenant:batch", "1" * 64))
    batch_case = store.save_case(_case("batch-case", "tenant:batch", "2" * 64))
    legacy = store.save_review(_review("legacy-review", legacy_case))
    _, batch = store.ensure_batch_review(_batch_review("batch-review", batch_case))

    assert {item.review_id for item in store.list_pending_reviews("tenant:batch")} == {
        legacy.review_id,
        batch.review_id,
    }
    assert [
        item.review_id
        for item in store.list_pending_reviews(
            "tenant:batch",
            origin=ReviewOrigin.BATCH_SCREENING,
        )
    ] == [batch.review_id]


def test_batch_review_retry_does_not_reopen_completed_task(store: SQLiteStore) -> None:
    case = store.save_case(_case("batch-complete-case", "tenant:batch-complete", "3" * 64))
    proposed = _batch_review("batch-complete-review", case)
    _, pending = store.ensure_batch_review(proposed)
    pending.reviewer_decision = "indeterminate"
    pending.reviewed_by = "reviewer-1"
    pending.reviewed_at = datetime.now(UTC)
    completed, _ = store.complete_review_with_case_update(pending, expected_version=1)

    persisted_case, persisted_review = store.ensure_batch_review(
        _batch_review("batch-complete-review", case)
    )

    assert completed.status is ReviewStatus.COMPLETED
    assert persisted_review.status is ReviewStatus.COMPLETED
    assert persisted_review.version == 2
    assert persisted_case.review_status is ReviewStatus.COMPLETED


def test_complete_review_enforces_optimistic_lock_and_preserves_winner(
    store: SQLiteStore,
) -> None:
    case = store.save_case(_case("case-review", "tenant:review", "e" * 64))
    store.save_review(_review("review-1", case))
    first_reader = store.get_review("review-1", "tenant:review")
    stale_reader = store.get_review("review-1", "tenant:review")

    first_reader.reviewer_decision = "keep_model_flagged"
    first_reader.reviewer_note = "first reviewer committed"
    first_reader.reviewed_by = "reviewer-1"
    first_reader.reviewed_at = datetime.now(UTC)
    completed = store.complete_review(first_reader, expected_version=1)

    assert completed.version == 2
    assert completed.status is ReviewStatus.COMPLETED
    assert store.list_pending_reviews("tenant:review") == []

    stale_reader.reviewer_decision = "indeterminate"
    stale_reader.reviewer_note = "stale write must not win"
    stale_reader.reviewed_by = "reviewer-2"
    stale_reader.reviewed_at = datetime.now(UTC)
    with pytest.raises(VersionConflictError):
        store.complete_review(stale_reader, expected_version=1)

    persisted = store.get_review("review-1", "tenant:review")
    assert persisted.version == 2
    assert persisted.status is ReviewStatus.COMPLETED
    assert persisted.reviewer_decision == "keep_model_flagged"
    assert persisted.reviewer_note == "first reviewer committed"
    assert persisted.reviewed_by == "reviewer-1"


def test_case_review_creation_and_completion_update_case_atomically(
    store: SQLiteStore,
) -> None:
    case = _case("case-atomic", "tenant:atomic", "7" * 64)
    review = _review("review-atomic", case)
    case.review_id = review.review_id
    case.review_status = ReviewStatus.PENDING
    store.save_case_with_review(case, review)

    loaded = store.get_review(review.review_id, case.owner_scope)
    loaded.reviewer_decision = "indeterminate"
    completed, updated_case = store.complete_review_with_case_update(loaded, expected_version=1)

    assert completed.status is ReviewStatus.COMPLETED
    assert updated_case.review_status is ReviewStatus.COMPLETED
    assert store.get_case(case.case_id, case.owner_scope).record_version == 2


def test_case_review_transaction_rolls_back_review_when_case_insert_fails(
    store: SQLiteStore,
) -> None:
    store.save_case(_case("existing-case", "tenant:atomic", "8" * 64))
    duplicate = _case("duplicate-case", "tenant:atomic", "8" * 64)
    review = _review("orphan-review", duplicate)
    duplicate.review_id = review.review_id
    duplicate.review_status = ReviewStatus.PENDING

    # Storage deliberately normalizes the database-specific uniqueness error so
    # API callers receive a stable conflict contract instead of leaking sqlite.
    with pytest.raises(VersionConflictError):
        store.save_case_with_review(duplicate, review)

    with pytest.raises(KeyError):
        store.get_review(review.review_id, duplicate.owner_scope)
    with pytest.raises(KeyError):
        store.get_case(duplicate.case_id, duplicate.owner_scope)


def test_subject_scoped_case_review_screening_and_preferences_are_isolated(
    store: SQLiteStore,
) -> None:
    shared_hash = "c" * 64
    alice = store.save_case(_case("case-alice", "tenant:shared", shared_hash, user_id="alice"))
    bob = store.save_case(_case("case-bob", "tenant:shared", shared_hash, user_id="bob"))
    assert store.find_case_by_hash("tenant:shared", "alice", shared_hash) == alice
    assert store.find_case_by_hash("tenant:shared", "bob", shared_hash) == bob
    with pytest.raises(AccessDeniedError, match="subject scope"):
        store.get_case(alice.case_id, alice.owner_scope, subject_user_id="bob")

    alice_review = store.save_review(_review("review-alice", alice))
    store.save_review(_review("review-bob", bob))
    assert (
        store.get_review(
            alice_review.review_id,
            alice.owner_scope,
            subject_user_id="alice",
        )
        == alice_review
    )
    with pytest.raises(AccessDeniedError, match="subject scope"):
        store.get_review(
            alice_review.review_id,
            alice.owner_scope,
            subject_user_id="bob",
        )
    assert [
        item.review_id
        for item in store.list_pending_reviews(alice.owner_scope, subject_user_id="alice")
    ] == [alice_review.review_id]

    store.get_or_create_thread("shared-thread", "alice", "tenant:shared")
    session = store.save_screening_session(
        _screening_session(
            "session-alice",
            thread_id="shared-thread",
            user_id="alice",
            owner_scope="tenant:shared",
        )
    )
    with pytest.raises(AccessDeniedError, match="subject scope"):
        store.get_screening_session(
            session.session_id,
            session.owner_scope,
            subject_user_id="bob",
        )

    store.save_preferences(
        UserPreferences(user_id="same-user", language="en"),
        owner_scope="tenant:a",
    )
    store.save_preferences(
        UserPreferences(user_id="same-user", report_detail="detailed"),
        owner_scope="tenant:b",
    )
    assert store.get_preferences("same-user", owner_scope="tenant:a").language == "en"
    assert store.get_preferences("same-user", owner_scope="tenant:b").report_detail == "detailed"
    with pytest.raises(AccessDeniedError, match="owner scope"):
        store.get_preferences("same-user")


def test_v2_migration_backfills_only_explicit_subjects_and_quarantines_legacy(
    tmp_path,
) -> None:
    database = tmp_path / "legacy-v2.sqlite3"
    explicit_case = _case("explicit-case", "tenant:legacy", "1" * 64, user_id="alice")
    unbound_case = _case("unbound-case", "tenant:legacy", "2" * 64, user_id=None)
    explicit_thread = ThreadState(
        thread_id="thread-explicit",
        user_id="alice",
        owner_scope="tenant:legacy",
        active_intent="preserved",
    )
    mismatched_thread = ThreadState(
        thread_id="thread-mismatch",
        user_id="mallory",
        owner_scope="tenant:legacy",
        active_intent="must-not-migrate",
    )
    explicit_session = _screening_session(
        "session-explicit",
        thread_id=explicit_thread.thread_id,
        user_id="alice",
        owner_scope="tenant:legacy",
    )
    legacy_preference = UserPreferences(user_id="alice", report_detail="detailed")
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                owner_scope TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX idx_cases_owner_sha
                ON cases(owner_scope, image_sha256);
            CREATE TABLE threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE preferences (
                user_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE screening_sessions (
                session_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                owner_scope TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            PRAGMA user_version=2;
            """
        )
        now = datetime.now(UTC).isoformat()
        connection.executemany(
            """INSERT INTO cases
            (case_id, owner_scope, image_sha256, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    explicit_case.case_id,
                    explicit_case.owner_scope,
                    explicit_case.image_sha256,
                    explicit_case.model_dump_json(),
                    now,
                ),
                (
                    unbound_case.case_id,
                    unbound_case.owner_scope,
                    unbound_case.image_sha256,
                    unbound_case.model_dump_json(),
                    now,
                ),
            ],
        )
        connection.executemany(
            """INSERT INTO threads
            (thread_id, user_id, owner_scope, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            [
                (
                    explicit_thread.thread_id,
                    explicit_thread.user_id,
                    explicit_thread.owner_scope,
                    explicit_thread.model_dump_json(),
                    now,
                ),
                (
                    mismatched_thread.thread_id,
                    "alice",
                    mismatched_thread.owner_scope,
                    mismatched_thread.model_dump_json(),
                    now,
                ),
            ],
        )
        connection.execute(
            """INSERT INTO preferences(user_id, payload, updated_at)
            VALUES (?, ?, ?)""",
            (legacy_preference.user_id, legacy_preference.model_dump_json(), now),
        )
        connection.execute(
            """INSERT INTO screening_sessions
            (session_id, thread_id, owner_scope, payload, updated_at)
            VALUES (?, ?, ?, ?, ?)""",
            (
                explicit_session.session_id,
                explicit_session.thread_id,
                explicit_session.owner_scope,
                explicit_session.model_dump_json(),
                now,
            ),
        )

    migrated = SQLiteStore(database)
    try:
        assert migrated.integrity_check()["schema_version"] == 4
        assert (
            migrated.get_case(
                explicit_case.case_id,
                explicit_case.owner_scope,
                subject_user_id="alice",
            ).case_id
            == explicit_case.case_id
        )
        with pytest.raises(AccessDeniedError, match="no valid subject"):
            migrated.get_case(unbound_case.case_id, unbound_case.owner_scope)
        with pytest.raises(AccessDeniedError, match="legacy unbound"):
            migrated.save_case(unbound_case.model_copy(update={"user_id": "alice"}))

        preserved = migrated.get_or_create_thread(
            explicit_thread.thread_id,
            explicit_thread.user_id,
            explicit_thread.owner_scope,
        )
        assert preserved.active_intent == "preserved"
        not_attributed = migrated.get_or_create_thread(
            mismatched_thread.thread_id,
            "alice",
            mismatched_thread.owner_scope,
        )
        assert not_attributed.active_intent is None
        assert (
            migrated.get_screening_session(
                explicit_session.session_id,
                explicit_session.owner_scope,
                subject_user_id="alice",
            ).session_id
            == explicit_session.session_id
        )

        assert (
            migrated.get_preferences("alice", owner_scope="tenant:legacy").report_detail
            == "standard"
        )
        with sqlite3.connect(database) as connection:
            assert (
                connection.execute(
                    """SELECT user_id FROM cases WHERE case_id='unbound-case'"""
                ).fetchone()[0]
                is None
            )
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM preferences_legacy_unscoped_v2"
                ).fetchone()[0]
                == 1
            )
            assert (
                connection.execute("SELECT COUNT(*) FROM threads_legacy_global_v2").fetchone()[0]
                == 2
            )
    finally:
        migrated.close()


def test_audit_hash_chain_detects_payload_tampering(store: SQLiteStore) -> None:
    for index in range(2):
        store.audit(
            request_id=f"request-{index}",
            actor_id="user-hash",
            actor_role="user",
            action="synthetic_test",
            owner_scope="tenant:hash",
            details={"index": index},
        )
    verified = store.verify_audit_chain()
    assert verified["valid"] is True
    assert verified["event_count"] == 2
    assert len(verified["head_event_hash"]) == 64

    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE audit_events SET details='{}' WHERE event_id=1")
        connection.commit()
    tampered = store.verify_audit_chain()
    assert tampered["valid"] is False
    assert tampered["first_invalid_event_id"] == 1


def test_integrity_check_and_non_overwriting_online_backup(store: SQLiteStore, tmp_path) -> None:
    store.save_case(_case("backup-case", "tenant:backup", "9" * 64))
    store.audit(
        request_id="backup-request",
        actor_id="backup-user",
        actor_role="user",
        action="backup_test",
        owner_scope="tenant:backup",
    )
    assert store.integrity_check() == {
        "sqlite_quick_check": "ok",
        "foreign_key_violation_count": 0,
        "schema_version": 4,
        "audit_chain_valid": True,
        "ok": True,
    }

    destination = tmp_path / "backups" / "state.sqlite3"
    receipt = store.backup_to(destination)
    assert receipt["path"] == str(destination.resolve())
    assert receipt["size_bytes"] > 0
    assert len(receipt["sha256"]) == 64
    with pytest.raises(FileExistsError):
        store.backup_to(destination)

    restored = SQLiteStore(destination)
    try:
        assert restored.get_case("backup-case", "tenant:backup").case_id == "backup-case"
        assert restored.integrity_check()["ok"] is True
    finally:
        restored.close()
