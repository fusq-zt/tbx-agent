from __future__ import annotations

import io
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.schemas import (
    CaseRecord,
    NarrationStatus,
    ReviewOrigin,
    ReviewRecord,
    ReviewStatus,
)
from tbx_agent.security.identity import build_signed_proxy_headers, production_blockers
from tbx_agent.service import TBXAgentService
from tbx_agent.vision import fuse_rank03, validate_image
from tbx_agent.vision.mock import MockRank03Backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROXY_SECRET = "test-only-proxy-secret-with-at-least-32-bytes"
METRICS_TOKEN = "test-only-metrics-token-with-at-least-32-bytes"


def _settings(tmp_path: Path, **overrides) -> Settings:
    settings = replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        openai_enabled=False,
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
    )
    return replace(settings, **overrides)


def _production_settings(tmp_path: Path, **overrides) -> Settings:
    return _settings(
        tmp_path,
        deployment_profile="production",
        trusted_proxy_auth_enabled=True,
        trusted_proxy_hmac_secret=PROXY_SECRET,
        vision_backend="rank03",
        narrator_backend="llama_cpp",
        require_real_inference=True,
        require_llm_inference=True,
        retain_uploaded_image=False,
        **overrides,
    )


class _Rank03ContractTestDouble:
    """Deterministic test double that emits the exact frozen evidence identity."""

    backend_id = "tbx11k-rank03-official-a-v1"
    runtime_contract = "rank03-frozen-runtime-v1"
    synthetic = False

    def __init__(self, policy: dict, runtime_config: dict):
        self.policy = policy
        self.runtime_config = runtime_config
        self._delegate = MockRank03Backend(policy, runtime_config)
        self._classifier = object()
        self._detector = object()

    def probe_runtime(self) -> dict:
        return {"classifier_loaded": True, "detector_loaded": True}

    def infer(self, *, case_id, image):
        evidence = self._delegate.infer(case_id=case_id, image=image)
        bundle = self.runtime_config["model_bundle_id"]
        return evidence.model_copy(
            update={
                "run_id": f"contract-test-{uuid.uuid4()}",
                "classifier_model_id": f"{bundle}:convnext_tiny",
                "classifier_checkpoint_sha256": self.runtime_config["classifier"][
                    "checkpoint_sha256"
                ],
                "classifier_weight_version": self.runtime_config["classifier"]["checkpoint_sha256"],
                "detector_model_id": f"{bundle}:dfine_l",
                "detector_checkpoint_sha256": self.runtime_config["detector"]["checkpoint_sha256"],
                "preprocessing_version": "rank03-frozen-official-submission-v1",
                "artifact_refs": ["device:test", "contract_test_double"],
            }
        )


class _LlamaCppContractTestDouble:
    backend_id = "llama_cpp"
    runtime_contract = "llama-cpp-grounded-generation-v1"
    synthetic = False
    policy_id = "tbx-grounded-evidence-synthesis-v2"

    def __init__(self, settings: Settings | None = None):
        self.model = (
            settings.llama_cpp_model_alias
            if settings
            else "tbx-medgemma-1.5-4b-it-q4-k-m"
        )
        self.model_digest = settings.llama_cpp_model_sha256 if settings else "1" * 64
        self.probe_calls = 0

    def probe_generation(self) -> dict:
        self.probe_calls += 1
        return {
            "generation_probed": True,
            "generation_invoked": True,
            "model": self.model,
            "model_file_sha256": self.model_digest,
            "prompt_tokens": 3,
            "completion_tokens": 2,
        }

    def narrate(self, response):
        return response.model_copy(
            update={
                "narrator_backend": self.backend_id,
                "narrator_model": self.model,
                "narrator_model_digest": self.model_digest,
                "narrator_policy_id": self.policy_id,
                "narration_status": NarrationStatus.APPLIED,
                "narrator_generation_invoked": True,
                "narrator_prompt_tokens": 3,
                "narrator_completion_tokens": 2,
            }
        )


class _FailingVisionProbe(_Rank03ContractTestDouble):
    def probe_runtime(self) -> dict:
        raise RuntimeError("simulated load failure")


class _FailingGenerationProbe(_LlamaCppContractTestDouble):
    @staticmethod
    def probe_generation() -> dict:
        raise RuntimeError("simulated generation failure")


class _SyntheticEvidenceBehindAttestedBackend(_Rank03ContractTestDouble):
    def infer(self, *, case_id, image):
        return self._delegate.infer(case_id=case_id, image=image)


class _QualityDriftEvidenceBehindAttestedBackend(_Rank03ContractTestDouble):
    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        return evidence.model_copy(
            update={
                "image_quality_status": "transport_valid",
                "image_quality_codes": [],
            }
        )


def _production_service(tmp_path: Path, **overrides) -> TBXAgentService:
    settings = _production_settings(tmp_path, **overrides)
    service = TBXAgentService(_settings(tmp_path))
    service.settings = settings
    service.vision = _Rank03ContractTestDouble(service.policy, service.runtime_config)
    service.narrator = _LlamaCppContractTestDouble(settings)
    return service


def _headers(
    method: str,
    path: str,
    *,
    nonce: str | None = None,
    tenant_id: str = "clinic-a",
    user_id: str = "patient-1",
    actor_id: str = "clinician-1",
) -> dict[str, str]:
    return build_signed_proxy_headers(
        secret=PROXY_SECRET,
        method=method,
        path=path,
        tenant_id=tenant_id,
        user_id=user_id,
        actor_id=actor_id,
        timestamp=int(time.time()),
        nonce=nonce or uuid.uuid4().hex,
    )


def _agent_body(**overrides) -> dict:
    body = {
        "thread_id": "security-test-thread",
        "message": "胸片异常后要做什么检查？",
    }
    body.update(overrides)
    return body


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=(40, 60, 80)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_production_startup_fails_closed_without_proxy_contract(tmp_path: Path) -> None:
    unsafe = _settings(
        tmp_path,
        deployment_profile="production",
        trusted_proxy_auth_enabled=False,
        trusted_proxy_hmac_secret="do-not-leak-me",
    )
    with pytest.raises(RuntimeError) as caught:
        create_app(TBXAgentService(unsafe))
    assert "trusted_proxy_auth_disabled" in str(caught.value)
    assert "trusted_proxy_hmac_secret_missing_or_weak" in str(caught.value)
    assert "do-not-leak-me" not in str(caught.value)


def test_production_cannot_disable_real_vision_or_local_llm(tmp_path: Path) -> None:
    unsafe = _settings(
        tmp_path,
        deployment_profile="production",
        trusted_proxy_auth_enabled=True,
        trusted_proxy_hmac_secret=PROXY_SECRET,
    )
    with pytest.raises(RuntimeError) as caught:
        create_app(TBXAgentService(unsafe))
    message = str(caught.value)
    assert "production_requires_real_inference" in message
    assert "production_requires_rank03_backend" in message
    assert "production_requires_local_llm_inference" in message
    assert "production_requires_llama_cpp_backend" in message


def test_production_rejects_remote_llama_cpp_even_with_explicit_opt_in(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    service.settings = replace(
        service.settings,
        llama_cpp_base_url="https://llm.example.invalid",
        llama_cpp_allow_remote=True,
    )

    with pytest.raises(RuntimeError, match="production_requires_loopback_llama_cpp"):
        create_app(service)


@pytest.mark.parametrize("case_relation", ["same", "inside", "contains"])
def test_production_rejects_case_and_model_root_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_relation: str,
) -> None:
    model_root = tmp_path / "models"
    case_root = {
        "same": model_root,
        "inside": model_root / "cases",
        "contains": tmp_path,
    }[case_relation]
    monkeypatch.setenv("TBX_ARTIFACT_ROOT", str(model_root))
    settings = _production_settings(tmp_path, artifact_root=case_root)

    assert "case_artifact_root_overlaps_model_cache" in production_blockers(settings)


def test_readiness_fails_when_rank03_weights_cannot_load(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    service.vision = _FailingVisionProbe(service.policy, service.runtime_config)

    response = TestClient(create_app(service)).get("/readyz")

    assert response.status_code == 503
    assert response.json()["runtime_verified"] is False
    assert "required_components_degraded" in response.json()["blockers"]


def test_readiness_requires_a_real_structured_llm_generation(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    service.narrator = _FailingGenerationProbe()

    response = TestClient(create_app(service)).get("/readyz")

    assert response.status_code == 503
    assert response.json()["llm_generation_probed"] is False
    assert response.json()["runtime_verified"] is False


def test_public_health_endpoints_reuse_startup_attestation(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    client = TestClient(create_app(service))

    health = client.get("/healthz").json()
    manifest = client.get("/v1/system/manifest").json()
    assert health["vision_backend"] == "rank03"
    assert manifest["vision_backend"] == "rank03"
    assert health["vision_model_display_name"] == "TBX-CXR Vision v1"
    assert manifest["vision_model_display_name"] == "TBX-CXR Vision v1"

    for _ in range(4):
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/v1/system/capabilities").status_code == 200
        assert client.get("/v1/system/manifest").status_code == 200

    assert service.narrator.probe_calls == 1


def test_attested_backend_cannot_persist_synthetic_evidence(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    service.vision = _SyntheticEvidenceBehindAttestedBackend(service.policy, service.runtime_config)
    client = TestClient(create_app(service))
    path = "/v1/assessments/cxr"
    payload = _png()

    response = client.post(
        path,
        files={"file": ("cxr.png", payload, "image/png")},
        data={"consent_to_process": "true", "attested_chest_radiograph": "true"},
        headers=_headers("POST", path),
    )

    assert response.status_code == 200
    case_id = response.json()["case"]["case_id"]
    agent_path = "/v1/agent/respond"
    classified = client.post(
        agent_path,
        json={
            "thread_id": "attestation-test",
            "message": "这张胸片有没有结核病？",
            "case_id": case_id,
        },
        headers=_headers("POST", agent_path),
    )
    assert classified.status_code == 200
    assert classified.json()["execution_receipt"]["status"] in {"failed", "unavailable"}
    persisted = service.store.get_case(
        case_id,
        "tenant:clinic-a",
        subject_user_id="patient-1",
    )
    assert persisted.vision_evidence is None
    assert persisted.classification_status in {"failed", "unavailable"}


def test_attested_backend_cannot_drop_input_quality_abstention(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    service.vision = _QualityDriftEvidenceBehindAttestedBackend(
        service.policy,
        service.runtime_config,
    )
    client = TestClient(create_app(service))
    path = "/v1/assessments/cxr"

    response = client.post(
        path,
        files={"file": ("cxr.png", _png(), "image/png")},
        data={"consent_to_process": "true", "attested_chest_radiograph": "true"},
        headers=_headers("POST", path),
    )

    assert response.status_code == 200
    case_id = response.json()["case"]["case_id"]
    agent_path = "/v1/agent/respond"
    classified = client.post(
        agent_path,
        json={
            "thread_id": "quality-attestation-test",
            "message": "这张胸片有没有结核病？",
            "case_id": case_id,
        },
        headers=_headers("POST", agent_path),
    )
    assert classified.status_code == 200
    assert classified.json()["execution_receipt"]["status"] in {"failed", "unavailable"}
    persisted = service.store.get_case(
        case_id,
        "tenant:clinic-a",
        subject_user_id="patient-1",
    )
    assert persisted.vision_evidence is None


def test_real_runtime_never_reuses_forged_historical_evidence(tmp_path: Path) -> None:
    service = _production_service(tmp_path)
    payload = _png()
    image = validate_image(payload, max_bytes=service.settings.max_upload_bytes)
    case_id = str(uuid.uuid4())
    evidence = service.vision.infer(case_id=case_id, image=image)
    forged = evidence.model_copy(update={"run_id": f"mock-{uuid.uuid4()}"})
    case = CaseRecord(
        case_id=case_id,
        owner_scope="tenant:clinic-a",
        user_id="patient-1",
        image_artifact_ref=str(
            image.persist_original(payload, service.settings.case_artifact_root, case_id)
        ),
        image_sha256=image.sha256,
        image_width=image.width,
        image_height=image.height,
        image_source_format=image.source_format,
        input_transform_id=image.input_transform_id,
        image_quality_status=image.quality_status,
        image_quality_codes=list(image.quality_warnings),
        consent_scope="cxr_auxiliary_screening",
        vision_evidence=forged,
        fusion_decision=fuse_rank03(forged, service.policy),
    )
    service.store.save_case(case)
    path = "/v1/assessments/cxr"

    response = TestClient(create_app(service)).post(
        path,
        files={"file": ("cxr.png", payload, "image/png")},
        data={"consent_to_process": "true", "attested_chest_radiograph": "true"},
        headers=_headers("POST", path),
    )

    assert response.status_code == 200
    agent_path = "/v1/agent/respond"
    classified = TestClient(create_app(service)).post(
        agent_path,
        json={
            "thread_id": "forged-history-test",
            "message": "这张胸片有没有结核病？",
            "case_id": case_id,
        },
        headers=_headers("POST", agent_path),
    )
    assert classified.status_code == 200
    refreshed = service.store.get_case(
        case_id,
        "tenant:clinic-a",
        subject_user_id="patient-1",
    )
    assert refreshed.vision_evidence is not None
    assert not refreshed.vision_evidence.run_id.startswith("mock-")


def test_production_derives_identity_and_rejects_unsigned_or_spoofed_claims(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(_production_service(tmp_path)))
    path = "/v1/agent/respond"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["deployment_profile"] == "production"
    assert ready.json()["blockers"] == []

    unsigned = client.post(path, json=_agent_body())
    assert unsigned.status_code == 401
    assert PROXY_SECRET not in unsigned.text

    successful = client.post(path, json=_agent_body(), headers=_headers("POST", path))
    assert successful.status_code == 200

    spoofed = client.post(
        path,
        json=_agent_body(owner_scope="tenant:attacker", user_id="attacker"),
        headers=_headers("POST", path),
    )
    assert spoofed.status_code == 403
    assert spoofed.json() == {"detail": "identity claim mismatch"}


def test_production_identity_is_applied_to_multipart_without_client_claims(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(_production_service(tmp_path)))
    path = "/v1/assessments/cxr"
    response = client.post(
        path,
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
        headers=_headers("POST", path),
    )

    assert response.status_code == 200
    assert response.json()["case"]["owner_scope"] == "tenant:clinic-a"
    assert "image_artifact_ref" not in response.json()["case"]
    assert str(tmp_path) not in response.text
    assert response.json()["response"]["response_kind"] == "case_explanation"
    assert response.json()["case"]["classification_status"] == "not_requested"


def test_production_case_screening_review_and_report_reads_are_subject_scoped(
    tmp_path: Path,
) -> None:
    service = _production_service(tmp_path)
    client = TestClient(create_app(service))
    upload_path = "/v1/assessments/cxr"
    assessment = client.post(
        upload_path,
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
        headers=_headers("POST", upload_path, user_id="patient-1"),
    )
    assert assessment.status_code == 200
    case_id = assessment.json()["case"]["case_id"]

    case_path = f"/v1/cases/{case_id}"
    own_case = client.get(case_path, headers=_headers("GET", case_path, user_id="patient-1"))
    foreign_case = client.get(case_path, headers=_headers("GET", case_path, user_id="patient-2"))
    assert own_case.status_code == 200
    assert "image_artifact_ref" not in own_case.json()
    assert str(tmp_path) not in own_case.text
    assert foreign_case.status_code == 403

    report_path = f"/v1/cases/{case_id}/reports"
    report = client.post(
        report_path,
        json={},
        headers=_headers("POST", report_path, user_id="patient-1"),
    )
    assert report.status_code == 200
    json_path = report.json()["json_download_url"]
    own_report = client.get(json_path, headers=_headers("GET", json_path, user_id="patient-1"))
    foreign_report = client.get(json_path, headers=_headers("GET", json_path, user_id="patient-2"))
    assert own_report.status_code == 200
    assert "image_artifact_ref" not in own_report.text
    assert str(tmp_path) not in own_report.text
    assert foreign_report.status_code == 403

    start_path = "/v1/screening/sessions"
    started = client.post(
        start_path,
        json={"thread_id": "subject-screen", "consent": True},
        headers=_headers("POST", start_path, user_id="patient-1"),
    )
    assert started.status_code == 200
    session_id = started.json()["session"]["session_id"]
    session_path = f"/v1/screening/sessions/{session_id}"
    assert (
        client.get(
            session_path,
            headers=_headers("GET", session_path, user_id="patient-1"),
        ).status_code
        == 200
    )
    assert (
        client.get(
            session_path,
            headers=_headers("GET", session_path, user_id="patient-2"),
        ).status_code
        == 403
    )

    review_case = CaseRecord(
        case_id="subject-review-case",
        owner_scope="tenant:clinic-a",
        user_id="patient-1",
        image_artifact_ref=str(tmp_path / "must-not-leak.png"),
        image_sha256="d" * 64,
        image_width=512,
        image_height=512,
        consent_scope="test",
        review_id="subject-review",
        review_status=ReviewStatus.PENDING,
    )
    review = ReviewRecord(
        review_id="subject-review",
        case_id=review_case.case_id,
        owner_scope=review_case.owner_scope,
        trigger_reasons=["subject_scope_test"],
        origin=ReviewOrigin.BATCH_SCREENING,
        batch_id="subject-batch",
        batch_item_id="subject-item",
    )
    service.store.save_case_with_review(review_case, review)
    legacy_case = CaseRecord(
        case_id="legacy-interactive-review-case",
        owner_scope="tenant:clinic-a",
        user_id="patient-1",
        image_artifact_ref=str(tmp_path / "legacy-must-not-leak.png"),
        image_sha256="e" * 64,
        image_width=512,
        image_height=512,
        consent_scope="test",
        review_id="legacy-interactive-review",
        review_status=ReviewStatus.PENDING,
    )
    legacy_review = ReviewRecord(
        review_id="legacy-interactive-review",
        case_id=legacy_case.case_id,
        owner_scope=legacy_case.owner_scope,
        trigger_reasons=["legacy_subject_scope_test"],
    )
    service.store.save_case_with_review(legacy_case, legacy_review)
    pending_path = "/v1/reviews/pending"
    own_pending = client.get(
        pending_path,
        headers=_headers("GET", pending_path, user_id="patient-1"),
    )
    foreign_pending = client.get(
        pending_path,
        headers=_headers("GET", pending_path, user_id="patient-2"),
    )
    assert review.review_id in [item["review_id"] for item in own_pending.json()]
    assert legacy_review.review_id not in [item["review_id"] for item in own_pending.json()]
    assert foreign_pending.json() == []

    review_path = f"/v1/reviews/{review.review_id}"
    assert (
        client.get(
            review_path,
            headers=_headers("GET", review_path, user_id="patient-2"),
        ).status_code
        == 403
    )
    completion = {
        "expected_version": 1,
        "decision": "indeterminate",
        "note": "subject-bound review",
    }
    assert (
        client.patch(
            review_path,
            json=completion,
            headers=_headers("PATCH", review_path, user_id="patient-2"),
        ).status_code
        == 403
    )
    assert (
        client.patch(
            review_path,
            json=completion,
            headers=_headers("PATCH", review_path, user_id="patient-1"),
        ).status_code
        == 200
    )


def test_assessment_generation_conflict_has_stable_409_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = _production_service(tmp_path)
    client = TestClient(create_app(service))
    path = "/v1/assessments/cxr"

    def upload():
        return client.post(
            path,
            files={"file": ("cxr.png", _png(), "image/png")},
            data={
                "consent_to_process": "true",
                "attested_chest_radiograph": "true",
            },
            headers=_headers("POST", path, user_id="patient-1"),
        )

    assert upload().status_code == 200
    monkeypatch.setattr(service, "_can_reuse", lambda _case, **_kwargs: False)
    conflict = upload()

    assert conflict.status_code == 409
    assert conflict.json()["error_code"] == ("assessment_generation_migration_required")


def test_signed_identity_nonce_is_single_use(tmp_path: Path) -> None:
    client = TestClient(create_app(_production_service(tmp_path)))
    path = "/v1/agent/respond"
    headers = _headers("POST", path, nonce="unique-replay-nonce-0001")

    first = client.post(path, json=_agent_body(), headers=headers)
    replay = client.post(path, json=_agent_body(), headers=headers)

    assert first.status_code == 200
    assert replay.status_code == 401
    assert replay.json() == {"detail": "trusted proxy authentication failed"}


def test_signature_covers_raw_query_string(tmp_path: Path) -> None:
    client = TestClient(create_app(_production_service(tmp_path)))
    path = "/v1/cases/not-present"
    target = f"{path}?owner_scope=tenant%3Aclinic-a"

    correctly_signed = client.get(target, headers=_headers("GET", target))
    path_only_signature = client.get(target, headers=_headers("GET", path))

    assert correctly_signed.status_code == 404
    assert path_only_signature.status_code == 401


def test_stale_signature_and_signature_for_other_path_are_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(_production_service(tmp_path)))
    path = "/v1/agent/respond"
    stale = build_signed_proxy_headers(
        secret=PROXY_SECRET,
        method="POST",
        path=path,
        tenant_id="clinic-a",
        user_id="patient-1",
        actor_id="clinician-1",
        timestamp=int(time.time()) - 600,
        nonce="unique-stale-nonce-0001",
    )
    wrong_path = _headers("POST", "/v1/screening/sessions")

    assert client.post(path, json=_agent_body(), headers=stale).status_code == 401
    assert client.post(path, json=_agent_body(), headers=wrong_path).status_code == 401


def test_liveness_readiness_headers_and_development_compatibility(tmp_path: Path) -> None:
    client = TestClient(create_app(TBXAgentService(_settings(tmp_path))))

    live = client.get("/livez", headers={"X-Request-ID": "accepted-request-id"})
    ready = client.get("/readyz")
    response = client.post(
        "/v1/agent/respond",
        json=_agent_body(user_id="user", owner_scope="tenant:user"),
    )

    assert live.status_code == 200
    assert live.headers["X-Request-ID"] == "accepted-request-id"
    assert live.headers["X-Content-Type-Options"] == "nosniff"
    assert live.headers["X-Frame-Options"] == "DENY"
    assert ready.status_code == 200
    assert ready.json()["optional_narrator_probed"] is False
    assert response.status_code == 200
    missing_claims = client.post("/v1/agent/respond", json=_agent_body())
    assert missing_claims.status_code == 422


def test_metrics_are_guarded_deidentified_and_do_not_leak_secrets(tmp_path: Path) -> None:
    app = create_app(
        _production_service(
            tmp_path,
            metrics_enabled=True,
            metrics_allow_loopback=False,
            metrics_admin_token=METRICS_TOKEN,
        )
    )
    app.state.metrics.record_fallback("narrator_error")
    client = TestClient(app)
    client.get("/livez")

    denied = client.get("/metrics")
    metrics = client.get("/metrics", headers={"Authorization": f"Bearer {METRICS_TOKEN}"})

    assert denied.status_code == 403
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["contains_request_or_clinical_identifiers"] is False
    assert payload["scope"] == "single_process"
    assert payload["fallbacks"] == {"narrator_error": 1}
    serialized = metrics.text
    assert PROXY_SECRET not in serialized
    assert METRICS_TOKEN not in serialized
    assert "patient-1" not in serialized
    assert "clinic-a" not in serialized


def test_content_length_limit_rejects_before_payload_parsing(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            _production_service(
                tmp_path,
                max_upload_bytes=10,
                max_request_body_bytes=100,
            )
        )
    )
    response = client.post(
        "/v1/agent/respond",
        content=b"x" * 101,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large"}


def test_process_rate_limit_is_fail_fast(tmp_path: Path) -> None:
    client = TestClient(
        create_app(
            _production_service(
                tmp_path,
                rate_limit_requests_per_minute=1,
                rate_limit_burst=1,
            )
        )
    )
    path = "/v1/agent/respond"

    first = client.post(path, json=_agent_body(), headers=_headers("POST", path))
    limited = client.post(path, json=_agent_body(), headers=_headers("POST", path))

    assert first.status_code == 200
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"
