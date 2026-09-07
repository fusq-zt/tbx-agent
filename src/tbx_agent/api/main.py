from __future__ import annotations

import importlib.util
import io
import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi import Path as PathParameter
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageChops, ImageFilter
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from ..anatomy_runs import AnatomyRunStatus
from ..capabilities import build_capability_snapshot
from ..config import Settings
from ..langgraph_runtime import plan_react_provenance
from ..llm.connections import (
    EphemeralLLMConnectionRegistry,
    LLMConnectionAccessError,
    LLMConnectionUnavailableError,
)
from ..narrator import (
    NARRATOR_POLICY_ID,
    NarrationError,
    OpenAICompatibleNarrator,
)
from ..observability import InProcessMetrics, bind_request_id, reset_request_id
from ..orchestration import TBXAgentGraph
from ..product_identity import VISION_MODEL_DISPLAY_NAME
from ..schemas import AgentResponse, NarrationStatus, ResponseKind, ReviewOrigin
from ..screening import ScreeningError
from ..security.identity import (
    AuthenticationError,
    ProxyHMACAuthenticator,
    TrustedIdentity,
    bearer_token_matches,
    is_loopback_host,
    production_blockers,
    pseudonymous_client_key,
)
from ..security.traffic import (
    BackpressureGate,
    TokenBucketRateLimiter,
    TrafficGuardError,
    validate_request_size,
)
from ..service import (
    AnatomyNotConfiguredError,
    AssessmentStateConflictError,
    ConsentRequiredError,
    TBXAgentService,
)
from ..storage import AccessDeniedError, VersionConflictError
from ..vision.anatomy import (
    AnatomyBackendError,
    LungSide,
    decode_binary_mask,
)
from ..vision.base import VisionBackendError
from ..vision.image_validator import ImageValidationError
from .models import (
    AgentQuery,
    LLMConnectionCreateRequest,
    ReportRequest,
    ReviewCompleteRequest,
    ScreeningAnswerRequest,
    ScreeningCancelRequest,
    ScreeningStartRequest,
)


@lru_cache(maxsize=1)
def get_service() -> TBXAgentService:
    return TBXAgentService(Settings.from_env())


_PUBLIC_PATHS = {
    "/healthz",
    "/livez",
    "/readyz",
    "/metrics",
    "/v1/system/capabilities",
    "/v1/system/manifest",
    "/docs",
    "/openapi.json",
    "/redoc",
}

_PUBLICLY_HIDDEN_CLASSIFIER_SCORE_FIELDS = frozenset(
    {
        "class_probability_order",
        "class_probabilities",
        "top1_score",
        "top2_score",
        "top1_top2_margin",
        "classifier_threshold",
    }
)


def _redact_internal_artifact_references(value):
    """Project internal case evidence to the score-free public API representation."""

    if isinstance(value, dict):
        return {
            key: _redact_internal_artifact_references(item)
            for key, item in value.items()
            if key != "image_artifact_ref"
            and key not in _PUBLICLY_HIDDEN_CLASSIFIER_SCORE_FIELDS
        }
    if isinstance(value, list):
        return [_redact_internal_artifact_references(item) for item in value]
    return value


def _jsonable_agent_value(value: Any) -> Any:
    """Serialize public Agent runtime models without reconstructing provenance."""

    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable_agent_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_agent_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _jsonable_agent_value(item)
            for key, item in value.items()
        }
    return value


def _render_anatomy_boundary_png(evidence, selected: set[LungSide]) -> bytes:
    """Render transparent source-coordinate boundaries without the source CXR."""

    colors = {
        LungSide.LEFT: (20, 184, 166, 230),
        LungSide.RIGHT: (245, 158, 11, 230),
    }
    canvas = Image.new("RGBA", (evidence.image_width, evidence.image_height), (0, 0, 0, 0))
    for item in evidence.masks:
        if item.structure not in selected:
            continue
        decoded = decode_binary_mask(item.payload)
        mono = Image.new("L", (evidence.image_width, evidence.image_height))
        mono.putdata([255 if value else 0 for row in decoded for value in row])
        # A narrow morphological gradient remains legible without obscuring the
        # radiograph that the client composites beneath it.
        outside = mono.filter(ImageFilter.MaxFilter(5))
        inside = mono.filter(ImageFilter.MinFilter(5))
        boundary = ImageChops.subtract(outside, inside)
        layer = Image.new("RGBA", canvas.size, colors[item.structure])
        layer.putalpha(boundary)
        canvas = Image.alpha_composite(canvas, layer)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _render_refinement_contours_png(evidence) -> bytes:
    """Render transparent source-coordinate MedSAM contours without the CXR."""

    palette = (
        (239, 68, 68, 235),
        (37, 99, 235, 235),
        (168, 85, 247, 235),
        (6, 182, 212, 235),
    )
    canvas = Image.new("RGBA", (evidence.image_width, evidence.image_height), (0, 0, 0, 0))
    for item in evidence.items:
        if item.mask is None:
            continue
        decoded = decode_binary_mask(item.mask)
        mono = Image.new("L", canvas.size)
        mono.putdata([255 if value else 0 for row in decoded for value in row])
        boundary = ImageChops.subtract(
            mono.filter(ImageFilter.MaxFilter(5)),
            mono.filter(ImageFilter.MinFilter(5)),
        )
        color = palette[item.detection_index % len(palette)]
        fill = Image.new("RGBA", canvas.size, (*color[:3], 48))
        fill.putalpha(mono.point(lambda value: 48 if value else 0))
        outline = Image.new("RGBA", canvas.size, color)
        outline.putalpha(boundary)
        canvas = Image.alpha_composite(canvas, fill)
        canvas = Image.alpha_composite(canvas, outline)
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


_ANATOMY_EVIDENCE_AVAILABLE_STATUSES = {
    AnatomyRunStatus.COMPLETED,
    AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE,
}


def create_app(service: TBXAgentService | None = None) -> FastAPI:
    settings = service.settings if service is not None else Settings.from_env()
    inference_blockers: list[str] = []
    if settings.require_real_inference and settings.vision_backend != "rank03":
        inference_blockers.append("real_vision_backend_required")
    if settings.require_llm_inference and settings.narrator_backend != "llama_cpp":
        inference_blockers.append("local_llm_backend_required")
    if settings.anatomy_required and settings.anatomy_backend == "none":
        inference_blockers.append("anatomy_backend_required")
    if service is not None and settings.require_real_inference:
        backend_id = str(getattr(service.vision, "backend_id", ""))
        if (
            "mock" in backend_id.casefold()
            or bool(getattr(service.vision, "synthetic", True))
            or getattr(service.vision, "runtime_contract", None) != "rank03-frozen-runtime-v1"
        ):
            inference_blockers.append("synthetic_vision_backend_injected")
    if service is not None and settings.require_llm_inference:
        narrator_id = str(getattr(service.narrator, "backend_id", ""))
        if (
            narrator_id != "llama_cpp"
            or bool(getattr(service.narrator, "synthetic", True))
            or getattr(service.narrator, "runtime_contract", None)
            != "llama-cpp-grounded-generation-v1"
        ):
            inference_blockers.append("required_llm_runtime_not_injected")
    if inference_blockers:
        raise RuntimeError(f"real inference gate failed: {','.join(inference_blockers)}")
    blockers = production_blockers(settings)
    if blockers:
        # Only stable blocker codes are emitted; configured secret values and
        # identity claims can never appear in this exception.
        raise RuntimeError(f"production security gate failed: {','.join(blockers)}")
    runtime = service

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Own runtime startup/drain; explicitly injected services stay caller-owned."""
        nonlocal runtime, graph_runtime
        try:
            if (
                settings.require_real_inference
                or settings.require_llm_inference
                or settings.anatomy_required
                or settings.anatomy_backend != "none"
            ):
                service_ = await run_in_threadpool(svc)
                snapshot = await run_in_threadpool(build_capability_snapshot, service_)
                if snapshot.status != "ok" or not snapshot.runtime_verified:
                    raise RuntimeError("required real inference runtime failed startup warmup")
                _app.state.capability_snapshot = snapshot
            yield
        finally:
            try:
                if service is None and runtime is not None:
                    await run_in_threadpool(runtime.close)
            finally:
                llm_connections.clear()
                if service is None:
                    runtime = None
                    graph_runtime = None
                    if hasattr(_app.state, "capability_snapshot"):
                        del _app.state.capability_snapshot

    app = FastAPI(
        title="TBX-Agent",
        version="0.1.0",
        description=(
            "肺结核辅助筛查与诊疗信息支持。不是医疗器械成品，不用于确诊或排除，不提供个体化处方。"
        ),
        docs_url=None if settings.deployment_profile == "production" else "/docs",
        redoc_url=None if settings.deployment_profile == "production" else "/redoc",
        openapi_url=(None if settings.deployment_profile == "production" else "/openapi.json"),
        lifespan=lifespan,
    )
    graph_runtime: TBXAgentGraph | None = None
    llm_connections = EphemeralLLMConnectionRegistry()
    metrics = InProcessMetrics()
    backpressure = BackpressureGate(settings.max_concurrent_requests)
    capability_lock = Lock()
    runtime_lock = Lock()

    rate_limiter = TokenBucketRateLimiter(
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst=settings.rate_limit_burst,
    )
    authenticator = (
        ProxyHMACAuthenticator(
            secret=settings.trusted_proxy_hmac_secret,
            replay_window_seconds=settings.trusted_proxy_replay_window_seconds,
        )
        if settings.trusted_proxy_auth_enabled
        else None
    )
    app.state.metrics = metrics
    app.state.settings = settings
    app.state.llm_connections = llm_connections

    def is_protected_path(request: Request) -> bool:
        return request.url.path not in _PUBLIC_PATHS

    def route_template(request: Request) -> str:
        route = request.scope.get("route")
        template = getattr(route, "path", None)
        return template if isinstance(template, str) else "unmatched"

    def add_security_headers(response, request_id: str) -> None:
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cache-Control"] = "no-store"
        if settings.deployment_profile == "production":
            response.headers["Content-Security-Policy"] = (
                "default-src 'none'; frame-ancestors 'none'"
            )
        if settings.hsts_enabled:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    @app.middleware("http")
    async def platform_guard(request: Request, call_next):
        request_id, request_id_token = bind_request_id(request.headers.get("X-Request-ID"))
        started = perf_counter()
        entered = False
        response = None
        status_code = 500
        try:
            validate_request_size(
                request,
                max_bytes=settings.max_request_body_bytes,
                require_content_length=settings.deployment_profile == "production",
            )
            identity: TrustedIdentity | None = None
            protected = is_protected_path(request)
            if protected and authenticator is not None:
                identity = authenticator.authenticate(request)
                request.state.trusted_identity = identity
            elif protected and settings.deployment_profile == "production":
                # Defensive invariant: create_app already refuses this state.
                raise AuthenticationError()

            if protected:
                client_host = request.client.host if request.client else None
                principal = (
                    identity.metric_principal
                    if identity is not None
                    else pseudonymous_client_key({"client": client_host})
                )
                if not rate_limiter.allow(principal):
                    metrics.record_guard_event("rate_limited")
                    raise TrafficGuardError(
                        429,
                        "request rate exceeded",
                        retry_after=rate_limiter.retry_after_seconds,
                    )
                if not backpressure.try_enter():
                    metrics.record_guard_event("backpressure_rejected")
                    raise TrafficGuardError(503, "request capacity exhausted", retry_after=1)
                entered = True
            response = await call_next(request)
            status_code = response.status_code
        except AuthenticationError:
            metrics.record_guard_event("authentication_failed")
            status_code = 401
            response = JSONResponse(
                status_code=status_code,
                content={"detail": "trusted proxy authentication failed"},
            )
        except TrafficGuardError as exc:
            if exc.status_code in {400, 411, 413}:
                metrics.record_guard_event("body_rejected")
            status_code = exc.status_code
            response = JSONResponse(
                status_code=status_code,
                content={"detail": exc.detail},
            )
            if exc.retry_after is not None:
                response.headers["Retry-After"] = str(exc.retry_after)
        finally:
            if entered:
                backpressure.leave()
            latency_ms = (perf_counter() - started) * 1000
            metrics.observe_request(
                method=request.method,
                route_template=route_template(request),
                status_code=status_code,
                latency_ms=latency_ms,
            )
            reset_request_id(request_id_token)
        add_security_headers(response, request_id)
        return response

    def svc() -> TBXAgentService:
        nonlocal runtime
        if runtime is None:
            # Construct from the exact settings object that passed this app's
            # production gate; never borrow a process-global service created
            # under a different deployment profile.
            with runtime_lock:
                if runtime is None:
                    runtime = TBXAgentService(settings)
        return runtime

    def current_capability_snapshot():
        cached = getattr(app.state, "capability_snapshot", None)
        if cached is not None:
            return cached
        with capability_lock:
            cached = getattr(app.state, "capability_snapshot", None)
            if cached is None:
                cached = build_capability_snapshot(svc())
                app.state.capability_snapshot = cached
        return cached

    def resolved_identity(
        request: Request,
        *,
        owner_scope: str | None,
        user_id: str | None = None,
        actor_id: str | None = None,
        require_user: bool = False,
        require_actor: bool = False,
    ) -> tuple[str, str | None, str | None]:
        trusted: TrustedIdentity | None = getattr(request.state, "trusted_identity", None)
        if trusted is not None:
            if (
                (owner_scope is not None and owner_scope != trusted.owner_scope)
                or (user_id is not None and user_id != trusted.user_id)
                or (actor_id is not None and actor_id != trusted.actor_id)
            ):
                metrics.record_guard_event("identity_mismatch")
                raise HTTPException(status_code=403, detail="identity claim mismatch")
            return trusted.owner_scope, trusted.user_id, trusted.actor_id
        if owner_scope is None:
            raise HTTPException(status_code=422, detail="owner_scope is required")
        if require_user and user_id is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        if require_actor and actor_id is None:
            raise HTTPException(status_code=422, detail="actor_id is required")
        return owner_scope, user_id, actor_id

    def agent_graph() -> TBXAgentGraph:
        nonlocal graph_runtime
        if graph_runtime is None:
            graph_runtime = TBXAgentGraph(svc())
        return graph_runtime

    @app.exception_handler(AccessDeniedError)
    async def access_denied_handler(_request, exc: AccessDeniedError):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    @app.exception_handler(VersionConflictError)
    async def version_conflict_handler(_request, exc: VersionConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(AssessmentStateConflictError)
    async def assessment_state_conflict_handler(_request, exc: AssessmentStateConflictError):
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(exc),
                "error_code": "assessment_generation_migration_required",
            },
        )

    @app.exception_handler(VisionBackendError)
    async def vision_backend_error_handler(_request, _exc: VisionBackendError):
        app.state.capability_snapshot = None
        return JSONResponse(
            status_code=503,
            content={
                "detail": "真实影像模型推理未完成；系统没有生成替代或模拟结果。",
                "error_code": "required_vision_inference_failed",
            },
        )

    @app.exception_handler(AnatomyNotConfiguredError)
    async def anatomy_not_configured_handler(_request, _exc: AnatomyNotConfiguredError):
        return JSONResponse(
            status_code=409,
            content={
                "detail": "该部署未启用可选解剖分割。",
                "error_code": "anatomy_not_configured",
            },
        )

    @app.exception_handler(AnatomyBackendError)
    async def anatomy_backend_error_handler(_request, _exc: AnatomyBackendError):
        app.state.capability_snapshot = None
        return JSONResponse(
            status_code=503,
            content={
                "detail": "解剖分割运行时或权重不可用；既有 rank03 结果未被改变。",
                "error_code": "anatomy_backend_unavailable",
            },
        )

    @app.exception_handler(NarrationError)
    async def narrator_error_handler(_request, _exc: NarrationError):
        app.state.capability_snapshot = None
        return JSONResponse(
            status_code=503,
            content={
                "detail": "所选大模型推理未完成；系统没有回退到其他模型或模拟回答。",
                "error_code": "required_llm_inference_failed",
            },
        )

    @app.get("/livez")
    def liveness() -> dict:
        return {
            "status": "alive",
            "service": "TBX-Agent",
            "version": "0.1.0",
        }

    @app.get("/readyz")
    def readiness():
        readiness_blockers = list(production_blockers(settings))
        capability_snapshot = None
        if not readiness_blockers:
            try:
                capability_snapshot = current_capability_snapshot()
            except Exception:  # noqa: BLE001 - readiness must fail closed
                readiness_blockers.append("required_components_unavailable")
        if capability_snapshot is not None and capability_snapshot.status != "ok":
            readiness_blockers.append("required_components_degraded")
        llm_generation_probed = bool(
            capability_snapshot is not None
            and any(
                component.component_id == "llm_evidence_composer"
                and component.state.value == "ready"
                and component.loaded is True
                for component in capability_snapshot.components
            )
        )
        payload = {
            "status": "ready" if not readiness_blockers else "not_ready",
            "service": "TBX-Agent",
            "version": "0.1.0",
            "deployment_profile": settings.deployment_profile,
            "blockers": readiness_blockers,
            "runtime_verified": bool(
                capability_snapshot is not None and capability_snapshot.runtime_verified
            ),
            "llm_generation_probed": llm_generation_probed,
            # Backward-compatible field; it now reflects a completed structured
            # generation probe instead of merely echoing configuration.
            "optional_narrator_probed": llm_generation_probed,
            "clinical_validation": False,
        }
        return JSONResponse(
            status_code=200 if not readiness_blockers else 503,
            content=payload,
        )

    @app.get("/metrics")
    def process_metrics(request: Request):
        if not settings.metrics_enabled:
            raise HTTPException(status_code=404, detail="not found")
        client_host = request.client.host if request.client else None
        loopback_allowed = settings.metrics_allow_loopback and is_loopback_host(client_host)
        admin_allowed = bearer_token_matches(
            request.headers.get("Authorization"), settings.metrics_admin_token
        )
        if not loopback_allowed and not admin_allowed:
            raise HTTPException(status_code=403, detail="metrics access denied")
        return metrics.snapshot(inflight=backpressure.inflight)

    @app.get("/healthz")
    def health() -> dict:
        service_ = svc()
        capability_snapshot = current_capability_snapshot()
        return {
            "status": capability_snapshot.status,
            "service": "TBX-Agent",
            "version": "0.1.0",
            "vision_backend": service_.settings.vision_backend,
            "vision_model_display_name": VISION_MODEL_DISPLAY_NAME,
            "narrator_backend": service_.settings.narrator_backend,
            "mode": capability_snapshot.mode,
            "deployment_profile": settings.deployment_profile,
            "required_components_ready": capability_snapshot.status == "ok",
            "clinical_validation": False,
        }

    @app.get("/v1/system/capabilities")
    def system_capabilities() -> dict:
        return current_capability_snapshot().model_dump(mode="json")

    @app.get("/v1/system/manifest")
    def system_manifest() -> dict:
        service_ = svc()
        capability_snapshot = current_capability_snapshot()
        policy = service_.policy
        reference_split_sha256 = policy.get(
            "reference_split_sha256", policy.get("selection_split_sha256")
        )
        narrator_backend = service_.settings.narrator_backend
        narrator_model = None
        narrator_digest = None
        if narrator_backend == "ollama":
            narrator_model = service_.settings.ollama_model
            narrator_digest = service_.settings.ollama_model_digest
        elif narrator_backend == "llama_cpp":
            narrator_model = service_.settings.llama_cpp_model_alias
            narrator_digest = service_.settings.llama_cpp_model_sha256
        elif narrator_backend == "openai":
            narrator_model = service_.settings.openai_model
        real_contract = bool(
            service_.settings.require_real_inference
            and service_.settings.require_llm_inference
            and service_.settings.vision_backend == "rank03"
            and service_.settings.narrator_backend == "llama_cpp"
        )
        dicom_enabled = all(
            importlib.util.find_spec(module) is not None for module in ("numpy", "pydicom")
        )
        return {
            "vision_backend": service_.settings.vision_backend,
            "vision_model_display_name": VISION_MODEL_DISPLAY_NAME,
            "inference_contract": "real_required" if real_contract else "test_or_optional",
            "runtime_verified": capability_snapshot.runtime_verified,
            "inference_mode": (
                "real_runtime_verified"
                if real_contract and capability_snapshot.runtime_verified
                else "real_contract_not_verified"
                if real_contract
                else "test_or_optional"
            ),
            "model_bundle_id": service_.runtime_config["model_bundle_id"],
            "model_source_revision": service_.runtime_config["source_revision"],
            "classifier_checkpoint_sha256": service_.runtime_config["classifier"][
                "checkpoint_sha256"
            ],
            "detector_checkpoint_sha256": service_.runtime_config["detector"]["checkpoint_sha256"],
            "fusion_policy_id": policy["policy_id"],
            "classifier_rule": policy.get("classifier_rule"),
            "classifier_routes": policy.get("classifier_routes", {}),
            "detector_role": policy.get("detector_role"),
            "input_contract": {
                "max_upload_bytes": service_.settings.max_upload_bytes,
                "raster_formats": ["PNG", "JPEG"],
                "dicom": {
                    "enabled": dicom_enabled,
                    "part10_preamble_required": True,
                    "modalities": ["CR", "DX"],
                    "single_frame": True,
                    "monochrome_only": True,
                    "compressed_transfer_syntax": False,
                    "raw_dicom_persisted_as_case_artifact": False,
                    "persisted_derivative": "metadata_free_png",
                    "input_transform_id": "dicom-crdx-windowed-rgb-v1",
                },
                "raster_input_transform_id": "raster-exif-transpose-rgb-v1",
                "quality_contract": "technical_and_coarse_domain_checks_only",
            },
            "policy_selection_design": policy.get("selection_design"),
            "policy_selection_metrics": policy.get("selection_metrics"),
            "policy_heldout_metrics": policy.get("heldout_metrics"),
            "reference_split_sha256": reference_split_sha256,
            # Backward-compatible alias for clients that consumed the V1 manifest.
            "selection_split_sha256": reference_split_sha256,
            "knowledge_snapshot_id": service_.retriever.snapshot_id,
            "knowledge_manifest_sha256": service_.retriever.manifest_sha256,
            "knowledge_chunks_sha256": service_.retriever.chunks_sha256,
            "narrator": {
                "backend": narrator_backend,
                "model": narrator_model,
                "expected_model_digest": narrator_digest,
                "policy_id": NARRATOR_POLICY_ID,
                "local_only_client_policy": (
                    (narrator_backend == "ollama" and not service_.settings.ollama_allow_remote)
                    or (
                        narrator_backend == "llama_cpp"
                        and not service_.settings.llama_cpp_allow_remote
                    )
                ),
            },
            "agent_orchestration": plan_react_provenance(),
            "source_decisions": service_.retriever.manifest["sources"],
            "clinical_validation": False,
            "warning": policy["warning"],
        }

    @app.post("/v1/assessments/cxr")
    async def assess_cxr(
        request: Request,
        file: Annotated[
            UploadFile,
            File(description="已去标识化 PNG/JPEG 或无压缩单帧 CR/DX DICOM 胸部X线图像"),
        ],
        consent_to_process: Annotated[bool, Form()],
        attested_chest_radiograph: Annotated[bool, Form()],
        user_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        owner_scope: Annotated[str | None, Form(min_length=1, max_length=256)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:  # defensive type narrowing
            raise HTTPException(status_code=422, detail="user_id is required")
        service_ = await run_in_threadpool(svc)
        payload = await file.read(service_.settings.max_upload_bytes + 1)
        try:
            case, response = await run_in_threadpool(
                service_.assess_cxr,
                payload,
                user_id=user_id_,
                owner_scope=owner_scope_,
                consent_to_process=consent_to_process,
                attested_chest_radiograph=attested_chest_radiograph,
            )
        except (ConsentRequiredError, ImageValidationError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "case": _redact_internal_artifact_references(case.model_dump(mode="json")),
            "response": response.model_dump(mode="json"),
        }

    @app.post("/v1/batches/{batch_id}/assessments/cxr")
    async def assess_batch_cxr(
        request: Request,
        batch_id: Annotated[str, PathParameter(min_length=1, max_length=128)],
        file: Annotated[
            UploadFile,
            File(description="批量筛查中的 PNG/JPEG 或无压缩单帧 CR/DX DICOM 胸部X线图像"),
        ],
        batch_item_id: Annotated[str, Form(min_length=1, max_length=128)],
        consent_to_process: Annotated[bool, Form()],
        attested_chest_radiograph: Annotated[bool, Form()],
        user_id: Annotated[str | None, Form(min_length=1, max_length=128)] = None,
        owner_scope: Annotated[str | None, Form(min_length=1, max_length=256)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        service_ = await run_in_threadpool(svc)
        payload = await file.read(service_.settings.max_upload_bytes + 1)
        try:
            case, response, review = await run_in_threadpool(
                service_.assess_cxr_for_batch,
                payload,
                batch_id=batch_id,
                batch_item_id=batch_item_id,
                user_id=user_id_,
                owner_scope=owner_scope_,
                consent_to_process=consent_to_process,
                attested_chest_radiograph=attested_chest_radiograph,
            )
        except (ConsentRequiredError, ImageValidationError, ValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "batch_id": batch_id,
            "batch_item_id": batch_item_id,
            "enqueued_for_review": review is not None,
            "case": _redact_internal_artifact_references(case.model_dump(mode="json")),
            "response": response.model_dump(mode="json"),
            "review": review.model_dump(mode="json") if review is not None else None,
        }

    @app.get("/v1/cases/{case_id}")
    def get_case(
        case_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            case = svc().store.get_case(case_id, owner_scope_, subject_user_id=user_id_)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _redact_internal_artifact_references(case.model_dump(mode="json"))

    @app.post("/v1/cases/{case_id}/anatomy-runs")
    async def create_anatomy_run(
        case_id: str,
        request: Request,
        file: Annotated[
            UploadFile | None,
            File(description="可选：未保留衍生图时重新上传完全相同的 PNG/JPEG 或 DICOM"),
        ] = None,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ):
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        service_ = await run_in_threadpool(svc)
        payload = (
            await file.read(service_.settings.max_upload_bytes + 1)
            if file is not None
            else None
        )
        try:
            run = await run_in_threadpool(
                service_.request_anatomy_run,
                case_id=case_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
                payload=payload,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ImageValidationError, ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        status_code = 200 if run.reused_existing_run else 202
        return JSONResponse(
            status_code=status_code,
            content=run.model_dump(mode="json"),
        )

    @app.get("/v1/cases/{case_id}/anatomy-runs")
    def list_anatomy_runs(
        case_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> list[dict]:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            svc().store.get_case(
                case_id,
                owner_scope_,
                subject_user_id=user_id_,
            )
            runs = svc().store.list_anatomy_runs(
                case_id=case_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return [run.model_dump(mode="json") for run in runs]

    @app.get("/v1/cases/{case_id}/anatomy-runs/{run_id}")
    def get_anatomy_run(
        case_id: str,
        run_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            run = svc().get_anatomy_run(
                run_id=run_id,
                case_id=case_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return run.model_dump(mode="json")

    @app.get("/v1/cases/{case_id}/anatomy-runs/{run_id}/boundary.png")
    def get_anatomy_boundary(
        case_id: str,
        run_id: str,
        request: Request,
        structure: Annotated[
            Literal["combined", "left_lung", "right_lung"],
            Query(),
        ] = "combined",
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> Response:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            run = svc().get_anatomy_run(
                run_id=run_id,
                case_id=case_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run.status not in _ANATOMY_EVIDENCE_AVAILABLE_STATUSES or run.evidence is None:
            raise HTTPException(status_code=409, detail="anatomy run is not completed")
        selected = (
            set(LungSide)
            if structure == "combined"
            else {LungSide(structure)}
        )
        return Response(
            content=_render_anatomy_boundary_png(run.evidence, selected),
            media_type="image/png",
            headers={
                "X-Anatomy-Run-ID": run.run_id,
                "X-Anatomy-Routing-Effect": "none",
            },
        )

    @app.get("/v1/cases/{case_id}/anatomy-runs/{run_id}/contours.png")
    def get_refinement_contours(
        case_id: str,
        run_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> Response:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            run = svc().get_anatomy_run(
                run_id=run_id,
                case_id=case_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if run.refinement_evidence is None:
            raise HTTPException(status_code=409, detail="contour refinement is unavailable")
        return Response(
            content=_render_refinement_contours_png(run.refinement_evidence),
            media_type="image/png",
            headers={
                "X-Anatomy-Run-ID": run.run_id,
                "X-Refinement-Routing-Effect": "none",
                "X-Clinical-Validation": "false",
            },
        )

    @app.post("/v1/llm/connections")
    def create_llm_connection(
        request: LLMConnectionCreateRequest,
        http_request: Request,
    ) -> dict:
        """Validate and retain one user-selected endpoint credential in memory only."""

        owner_scope, user_id, _ = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            require_user=True,
        )
        if user_id is None:  # defensive narrowing
            raise HTTPException(status_code=422, detail="user_id is required")
        narrator: OpenAICompatibleNarrator | None = None
        try:
            narrator = OpenAICompatibleNarrator(
                base_url=request.base_url,
                model=request.model,
                api_key=request.api_key,
            )
            probe = narrator.narrate(
                AgentResponse(
                    request_id="llm-connection-probe",
                    trace_id="llm-connection-probe",
                    thread_id=request.thread_id,
                    response_kind=ResponseKind.CASE_EXPLANATION,
                    summary="连接测试通过。",
                )
            )
            if (
                probe.narration_status != NarrationStatus.APPLIED
                or probe.narrator_backend != narrator.backend_id
                or probe.narrator_model != narrator.model
                or probe.narrator_generation_invoked is not True
            ):
                raise NarrationError("OpenAI-compatible endpoint probe was not attested")
            info = llm_connections.create(
                owner_scope=owner_scope,
                user_id=user_id,
                thread_id=request.thread_id,
                base_url=narrator.base_url,
                model=narrator.model,
                api_key=request.api_key,
            )
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="OpenAI 兼容 API 地址、模型名或密钥格式不正确。",
            ) from None
        finally:
            if narrator is not None:
                narrator.close()
        return {
            "connection_id": info.connection_id,
            "provider": info.provider,
            "base_url": info.base_url,
            "model": info.model,
            "created_at": info.created_at.isoformat(),
            "expires_at": info.expires_at.isoformat(),
            "credential_persistence": "process_memory_only",
        }

    @app.delete("/v1/llm/connections/{connection_id}")
    def revoke_llm_connection(
        connection_id: str,
        request: Request,
        thread_id: Annotated[str, Query(min_length=1, max_length=128)],
        owner_scope: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
        user_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            revoked = llm_connections.revoke(
                connection_id,
                owner_scope=owner_scope_,
                user_id=user_id_,
                thread_id=thread_id,
            )
        except LLMConnectionAccessError:
            raise HTTPException(status_code=404, detail="LLM connection is unavailable") from None
        return {"revoked": revoked}

    @app.post("/v1/agent/respond")
    def respond(request: AgentQuery, http_request: Request) -> dict:
        owner_scope, user_id, _ = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            require_user=True,
        )
        request_payload = request.model_dump(
            exclude={"llm_provider", "llm_connection_id"}
        )
        request_payload.update({"owner_scope": owner_scope, "user_id": user_id})
        narrator_override: OpenAICompatibleNarrator | None = None
        try:
            if request.llm_provider == "openai_compatible":
                if user_id is None or request.llm_connection_id is None:
                    raise HTTPException(status_code=422, detail="LLM connection is required")
                try:
                    resolved = llm_connections.resolve(
                        request.llm_connection_id,
                        owner_scope=owner_scope,
                        user_id=user_id,
                        thread_id=request.thread_id,
                    )
                except LLMConnectionAccessError:
                    raise HTTPException(
                        status_code=404,
                        detail="LLM connection is unavailable",
                    ) from None
                except LLMConnectionUnavailableError:
                    raise HTTPException(
                        status_code=409,
                        detail="LLM connection expired or was revoked; reconnect before retrying",
                    ) from None
                narrator_override = OpenAICompatibleNarrator(
                    base_url=resolved.info.base_url,
                    model=resolved.info.model,
                    api_key=resolved.api_key,
                )
            result = agent_graph().invoke_with_receipt(
                request_payload,
                narrator_override=narrator_override,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        finally:
            if narrator_override is not None:
                narrator_override.close()
        if any(tool_result.receipt.fallback_used for tool_result in result.tool_results):
            metrics.record_fallback("agent_tool")
        narration_status = result.response.narration_status.value
        if narration_status == "fallback_error":
            metrics.record_fallback("narrator_error")
        elif narration_status == "rejected_by_safety":
            metrics.record_fallback("narrator_safety")
        payload = result.response.model_dump(mode="json")
        # Additive field: existing AgentResponse keys and their meanings remain
        # unchanged, while clients that need provenance can persist the receipt.
        payload["execution_receipt"] = _jsonable_agent_value(result.receipt)
        payload["execution_receipts"] = [
            tool_result.receipt.model_dump(mode="json")
            for tool_result in result.tool_results
        ]
        payload["execution_plan"] = _jsonable_agent_value(result.execution_plan)
        payload["reflection"] = _jsonable_agent_value(result.reflection)
        # The controller runtime is the source of truth for the public v2 trace.
        # Do not reconstruct decisions, reflection, or fallback state here.
        payload["agent_trace"] = _jsonable_agent_value(result.trace)
        return _redact_internal_artifact_references(payload)

    @app.post("/v1/screening/sessions")
    def start_screening(request: ScreeningStartRequest, http_request: Request) -> dict:
        owner_scope, user_id, _ = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            require_user=True,
        )
        request_payload = request.model_dump()
        request_payload.update({"owner_scope": owner_scope, "user_id": user_id})
        try:
            session, response = svc().start_active_screening(**request_payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "session": session.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }

    @app.post("/v1/screening/sessions/{session_id}/answers")
    def answer_screening(
        session_id: str,
        request: ScreeningAnswerRequest,
        http_request: Request,
    ) -> dict:
        owner_scope, user_id, _ = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            require_user=True,
        )
        request_payload = request.model_dump()
        request_payload.update({"owner_scope": owner_scope, "user_id": user_id})
        try:
            session, response = svc().answer_active_screening(
                session_id=session_id, **request_payload
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ScreeningError, PermissionError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "session": session.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }

    @app.post("/v1/screening/sessions/{session_id}/cancel")
    def cancel_screening(
        session_id: str,
        request: ScreeningCancelRequest,
        http_request: Request,
    ) -> dict:
        owner_scope, user_id, _ = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            require_user=True,
        )
        request_payload = request.model_dump()
        request_payload.update({"owner_scope": owner_scope, "user_id": user_id})
        try:
            session, response = svc().cancel_active_screening(
                session_id=session_id, **request_payload
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ScreeningError, PermissionError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "session": session.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }

    @app.get("/v1/screening/sessions/{session_id}")
    def get_screening_session(
        session_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            session = svc().store.get_screening_session(
                session_id, owner_scope_, subject_user_id=user_id_
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        response = svc().build_active_screening_response(
            session,
            request_id="read-only",
            trace_id="read-only",
            user_id=user_id_,
            owner_scope=owner_scope_,
        )
        return {
            "session": session.model_dump(mode="json"),
            "response": response.model_dump(mode="json"),
        }

    @app.get("/v1/reviews/pending")
    def pending_reviews(
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> list[dict]:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        return [
            item.model_dump(mode="json")
            for item in svc().store.list_pending_reviews(
                owner_scope_,
                subject_user_id=user_id_,
                origin=ReviewOrigin.BATCH_SCREENING,
            )
        ]

    @app.patch("/v1/reviews/{review_id}")
    def complete_review(
        review_id: str,
        request: ReviewCompleteRequest,
        http_request: Request,
    ) -> dict:
        owner_scope, subject_user_id, reviewer_id = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            actor_id=request.reviewer_id,
            require_user=True,
            require_actor=True,
        )
        request_payload = request.model_dump()
        request_payload.update(
            {
                "owner_scope": owner_scope,
                "reviewer_id": reviewer_id,
                "subject_user_id": subject_user_id,
                "allow_cross_subject": False,
            }
        )
        request_payload.pop("user_id", None)
        try:
            review = svc().complete_review(review_id=review_id, **request_payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return review.model_dump(mode="json")

    @app.get("/v1/reviews/{review_id}")
    def get_review(
        review_id: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ) -> dict:
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            review = svc().store.get_review(review_id, owner_scope_, subject_user_id=user_id_)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return review.model_dump(mode="json")

    @app.post("/v1/cases/{case_id}/reports")
    def create_report(
        case_id: str,
        request: ReportRequest,
        http_request: Request,
    ) -> dict:
        owner_scope, subject_user_id, actor_id = resolved_identity(
            http_request,
            owner_scope=request.owner_scope,
            user_id=request.user_id,
            actor_id=request.actor_id,
            require_user=True,
            require_actor=True,
        )
        request_payload = request.model_dump()
        request_payload.update(
            {
                "owner_scope": owner_scope,
                "actor_id": actor_id,
                "subject_user_id": subject_user_id,
                "allow_cross_subject": False,
            }
        )
        request_payload.pop("user_id", None)
        try:
            artifact = svc().create_report(case_id=case_id, **request_payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        report_id = artifact["report_id"]
        return {
            "report_id": report_id,
            "generated_at": artifact["generated_at"],
            "markdown_download_url": (f"/v1/cases/{case_id}/reports/{report_id}/markdown"),
            "json_download_url": f"/v1/cases/{case_id}/reports/{report_id}/json",
        }

    @app.get("/v1/cases/{case_id}/reports/{report_id}/{artifact_format}")
    def download_report(
        case_id: str,
        report_id: str,
        artifact_format: str,
        request: Request,
        owner_scope: Annotated[str | None, Query(min_length=1)] = None,
        user_id: Annotated[str | None, Query(min_length=1)] = None,
    ):
        owner_scope_, user_id_, _ = resolved_identity(
            request,
            owner_scope=owner_scope,
            user_id=user_id,
            require_user=True,
        )
        if user_id_ is None:
            raise HTTPException(status_code=422, detail="user_id is required")
        try:
            svc().store.get_case(case_id, owner_scope_, subject_user_id=user_id_)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        suffixes = {"markdown": ".md", "json": ".json"}
        suffix = suffixes.get(artifact_format)
        if suffix is None:
            raise HTTPException(status_code=404, detail="unsupported report artifact")
        report_root = (svc().settings.case_artifact_root / case_id / "reports").resolve()
        artifact_path = (report_root / f"{report_id}{suffix}").resolve()
        if artifact_path.parent != report_root or not artifact_path.is_file():
            raise HTTPException(status_code=404, detail="report artifact not found")
        if artifact_format == "json":
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status_code=500, detail="report artifact is unreadable"
                ) from exc
            return JSONResponse(
                content=_redact_internal_artifact_references(payload),
                headers={"Content-Disposition": 'attachment; filename="tbx-agent-report.json"'},
            )
        media_type = "text/markdown"
        filename = "tbx-agent-report.md"
        return FileResponse(
            path=Path(artifact_path),
            media_type=media_type,
            filename=filename,
        )

    return app


class _LazyASGIApp:
    """Delay real-runtime construction until the serving process receives ASGI traffic."""

    def __init__(self) -> None:
        self._instance: FastAPI | None = None
        self._lock = Lock()

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = create_app()
        await self._instance(scope, receive, send)


app = _LazyASGIApp()


def run() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "tbx_agent.api.main:app",
        host=os.getenv("TBX_AGENT_BIND_HOST", "127.0.0.1"),
        port=int(os.getenv("TBX_AGENT_BIND_PORT", "8000")),
        reload=False,
    )
