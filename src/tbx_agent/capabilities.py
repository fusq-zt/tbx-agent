from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from .langgraph_runtime import plan_react_provenance
from .llm.tool_calling import HighLevelToolName
from .schemas import StrictModel, utc_now
from .tools.contracts import ToolName

_MODEL_TOOL_HANDLERS = {
    HighLevelToolName.CLASSIFY_CXR: ToolName.CLASSIFY_CURRENT_CXR,
    HighLevelToolName.LOCALIZE_CXR: ToolName.LOCALIZE_CURRENT_CXR,
    HighLevelToolName.ANALYZE_LUNG_ANATOMY: ToolName.INSPECT_ANATOMICAL_CONTEXT,
    HighLevelToolName.SEARCH_TB_KNOWLEDGE: ToolName.SEARCH_TB_KNOWLEDGE,
}


class ComponentState(StrEnum):
    READY = "ready"
    CONFIGURED = "configured_not_probed"
    DISABLED = "disabled"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class CapabilityComponent(StrictModel):
    component_id: str
    state: ComponentState
    required: bool
    implementation: str
    detail: str
    loaded: bool | None = None
    synthetic: bool = False


class CapabilitySnapshot(StrictModel):
    status: str
    mode: str
    runtime_verified: bool = False
    checked_at: datetime = Field(default_factory=utc_now)
    components: list[CapabilityComponent]


def build_capability_snapshot(service: Any) -> CapabilitySnapshot:
    """Build a non-sensitive, fail-closed runtime capability snapshot.

    A deployment that requires real vision or LLM inference must prove those
    components are available. Explicit test deployments may still configure mock
    vision or omit the narrator without those optional components becoming ready.
    """

    backend = service.vision
    backend_id = str(getattr(backend, "backend_id", type(backend).__name__))
    is_mock = service.settings.vision_backend == "mock" or bool(
        getattr(backend, "synthetic", False)
    )
    classifier_loaded = getattr(backend, "_classifier", None) is not None
    detector_loaded = getattr(backend, "_detector", None) is not None
    real_vision_required = bool(service.settings.require_real_inference)
    vision_state = ComponentState.READY
    if is_mock:
        model_detail = (
            "synthetic backend is forbidden by the active real-inference contract"
            if real_vision_required
            else "deterministic synthetic test backend"
        )
        model_loaded = True
    else:
        runtime_contract = str(getattr(backend, "runtime_contract", ""))
        if real_vision_required and runtime_contract != "rank03-frozen-runtime-v1":
            vision_state = ComponentState.UNAVAILABLE
            model_loaded = False
            model_detail = "required rank03 runtime attestation is absent"
        else:
            probe = getattr(backend, "probe_runtime", None)
            try:
                if not callable(probe):
                    raise RuntimeError("rank03 runtime probe is unavailable")
                probe_receipt = probe()
                if (
                    not isinstance(probe_receipt, dict)
                    or probe_receipt.get("classifier_loaded") is not True
                    or probe_receipt.get("detector_loaded") is not True
                ):
                    raise RuntimeError("rank03 runtime probe receipt is invalid")
            except Exception:  # noqa: BLE001 - readiness must fail closed
                vision_state = ComponentState.UNAVAILABLE
                model_loaded = False
                model_detail = "rank03 dependencies or frozen weights failed to load"
            else:
                classifier_loaded = getattr(backend, "_classifier", None) is not None
                detector_loaded = getattr(backend, "_detector", None) is not None
                model_loaded = classifier_loaded and detector_loaded
                vision_state = ComponentState.READY if model_loaded else ComponentState.UNAVAILABLE
                model_detail = (
                    "frozen classifier and detector weights loaded; readiness does not "
                    "manufacture an image result"
                )

    components = [
        CapabilityComponent(
            component_id="rank03_image_assessment",
            state=(
                ComponentState.UNAVAILABLE if is_mock and real_vision_required else vision_state
            ),
            required=True,
            implementation=backend_id,
            detail=model_detail,
            loaded=model_loaded,
            synthetic=is_mock,
        ),
        CapabilityComponent(
            component_id="guideline_retrieval",
            state=ComponentState.READY,
            required=True,
            implementation=(
                f"{service.retriever.snapshot_id}/{service.retriever.retrieval_version}"
            ),
            detail=(
                f"reviewed offline snapshot with {len(service.retriever.chunks)} chunks; "
                f"metadata-gated BM25 generation {service.retriever.corpus_generation_id}; "
                "dense/reranker disabled pending pinned BGE artifacts and qrels gates"
            ),
            loaded=True,
        ),
        CapabilityComponent(
            component_id="active_screening",
            state=ComponentState.READY,
            required=False,
            implementation=service.screening.bank.guideline_rule_version,
            detail="consent-gated deterministic questionnaire state machine",
            loaded=True,
        ),
        CapabilityComponent(
            component_id="case_storage",
            state=ComponentState.READY,
            required=True,
            implementation="sqlite",
            detail="case, review, screening, and audit storage initialized",
            loaded=True,
        ),
    ]

    anatomy = getattr(service, "anatomy", None)
    anatomy_required = bool(getattr(service.settings, "anatomy_required", False))
    anatomy_operational = False
    if anatomy is None:
        components.append(
            CapabilityComponent(
                component_id="anatomy_segmentation",
                state=(
                    ComponentState.UNAVAILABLE if anatomy_required else ComponentState.DISABLED
                ),
                required=anatomy_required,
                implementation="none",
                detail=(
                    "required anatomy backend is not configured"
                    if anatomy_required
                    else "optional anatomy evidence is disabled"
                ),
                loaded=False,
            )
        )
    else:
        try:
            # A configured model-visible tool must be proven usable, even when
            # it is optional to the deployment as a whole. A dependency-only
            # probe previously reported ``configured_not_probed`` while the
            # pinned checkpoint path was missing, so the Agent advertised a
            # tool that could only fail on first use.
            probe = anatomy.probe_runtime(load=True)
        except Exception:  # noqa: BLE001 - capability boundary fails closed
            anatomy_state = ComponentState.UNAVAILABLE
            anatomy_loaded = False
            anatomy_detail = "configured anatomy runtime failed its startup probe"
        else:
            anatomy_loaded = bool(probe.loaded)
            anatomy_state = {
                "yes": ComponentState.READY,
                "no": ComponentState.UNAVAILABLE,
                "unverified": ComponentState.CONFIGURED,
            }[probe.available]
            anatomy_operational = (
                anatomy_state == ComponentState.READY and anatomy_loaded
            )
            anatomy_detail = (
                "routing-neutral paired lung masks and 2-D lung-field localization; "
                "not a lobe estimate and not clinically validated"
                if anatomy_operational
                else "configured anatomy runtime failed its startup probe"
            )
        components.append(
            CapabilityComponent(
                component_id="anatomy_segmentation",
                state=anatomy_state,
                required=anatomy_required,
                implementation=str(getattr(anatomy, "backend_id", "unknown")),
                detail=anatomy_detail,
                loaded=anatomy_loaded,
            )
        )

    refinement = getattr(service, "refinement", None)
    if refinement is None:
        components.append(
            CapabilityComponent(
                component_id="contour_refinement",
                state=ComponentState.DISABLED,
                required=False,
                implementation="none",
                detail="optional MedSAM detector-box contour visualization is disabled",
                loaded=False,
            )
        )
    else:
        try:
            probe = refinement.probe_runtime(load=False)
        except Exception:  # noqa: BLE001 - capability boundary fails closed
            refinement_state = ComponentState.UNAVAILABLE
            refinement_loaded = False
            refinement_detail = "optional contour-refinement runtime probe failed"
        else:
            refinement_loaded = bool(probe.loaded)
            refinement_state = {
                "yes": ComponentState.READY,
                "no": ComponentState.UNAVAILABLE,
                "unverified": ComponentState.CONFIGURED,
            }[probe.available]
            refinement_detail = (
                "D-FINE box-prompt contours constrained to PSPNet lung masks; "
                "visualization only, routing-neutral, and not clinically validated"
            )
        components.append(
            CapabilityComponent(
                component_id="contour_refinement",
                state=refinement_state,
                required=False,
                implementation=str(getattr(refinement, "backend_id", "unknown")),
                detail=refinement_detail,
                loaded=refinement_loaded,
            )
        )

    narrator = service.narrator
    narrator_required = bool(service.settings.require_llm_inference)
    if narrator is None:
        components.append(
            CapabilityComponent(
                component_id="llm_evidence_composer",
                state=(
                    ComponentState.UNAVAILABLE if narrator_required else ComponentState.DISABLED
                ),
                required=narrator_required,
                implementation="none",
                detail=(
                    "required local LLM is not configured"
                    if narrator_required
                    else "LLM composition is disabled for this test deployment"
                ),
                loaded=False,
            )
        )
    else:
        narrator_state = ComponentState.CONFIGURED
        narrator_loaded: bool | None = None
        narrator_detail = "configured but not probed in this deployment"
        if narrator_required:
            try:
                generation_probe = getattr(narrator, "probe_generation", None)
                if not callable(generation_probe):
                    raise RuntimeError("required LLM generation probe is unavailable")
                probe_receipt = generation_probe()
                if (
                    not isinstance(probe_receipt, dict)
                    or probe_receipt.get("generation_probed") is not True
                    or probe_receipt.get("generation_invoked") is not True
                    or probe_receipt.get("model") != service.settings.llama_cpp_model_alias
                    or probe_receipt.get("model_file_sha256")
                    != service.settings.llama_cpp_model_sha256
                    or isinstance(probe_receipt.get("prompt_tokens"), bool)
                    or not isinstance(probe_receipt.get("prompt_tokens"), int)
                    or probe_receipt["prompt_tokens"] <= 0
                    or isinstance(probe_receipt.get("completion_tokens"), bool)
                    or not isinstance(probe_receipt.get("completion_tokens"), int)
                    or probe_receipt["completion_tokens"] <= 0
                ):
                    raise RuntimeError("required LLM generation probe receipt is invalid")
            except Exception:  # noqa: BLE001 - readiness must fail closed
                narrator_state = ComponentState.UNAVAILABLE
                narrator_loaded = False
                narrator_detail = "required local LLM structured generation probe failed"
            else:
                narrator_state = ComponentState.READY
                narrator_loaded = True
                narrator_detail = (
                    "pinned local LLM artifact, serving alias, and structured generation verified"
                )
        components.append(
            CapabilityComponent(
                component_id="llm_evidence_composer",
                state=narrator_state,
                required=narrator_required,
                implementation=str(getattr(narrator, "backend_id", "unknown")),
                detail=narrator_detail,
                loaded=narrator_loaded,
            )
        )

    registry = getattr(service, "tool_registry", None)
    if registry is not None:
        orchestration = plan_react_provenance()
        components.append(
            CapabilityComponent(
                component_id="agent_orchestration",
                state=ComponentState.READY,
                required=True,
                implementation=str(orchestration["policy_id"]),
                detail=(
                    "LangGraph Plan + ReAct; native tool_calls first with strict JSON "
                    "Schema fallback; one model-visible tool per decision; SQLiteStore "
                    "owns business state and no durable graph checkpointer is configured"
                ),
                loaded=True,
            )
        )
        status_by_name = {tool.name: tool for tool in registry.statuses()}
        for public_name, internal_name in _MODEL_TOOL_HANDLERS.items():
            tool = status_by_name.get(internal_name.value)
            tool_required = (
                anatomy_required
                if public_name == HighLevelToolName.ANALYZE_LUNG_ANATOMY
                else True
            )
            if tool is None:
                components.append(
                    CapabilityComponent(
                        component_id=f"agent_tool:{public_name.value}",
                        state=ComponentState.UNAVAILABLE,
                        required=tool_required,
                        implementation="model_visible_high_level_tool",
                        detail="audited internal handler is not registered",
                        loaded=False,
                    )
                )
                continue
            tool_state = {
                "ready": ComponentState.READY,
                "degraded": ComponentState.DEGRADED,
                "unavailable": ComponentState.UNAVAILABLE,
            }[tool.availability.value]
            tool_detail = (
                f"audited handler={internal_name.value}; timeout={tool.timeout_ms}ms; "
                "runtime binds case identifiers and validates permissions"
            )
            if (
                public_name == HighLevelToolName.ANALYZE_LUNG_ANATOMY
                and not anatomy_operational
            ):
                tool_state = ComponentState.UNAVAILABLE
                tool_detail = "anatomy handler withheld because its startup probe is not ready"
            components.append(
                CapabilityComponent(
                    component_id=f"agent_tool:{public_name.value}",
                    state=tool_state,
                    required=tool_required,
                    implementation="model_visible_high_level_tool",
                    detail=tool_detail,
                    loaded=tool_state != ComponentState.UNAVAILABLE,
                )
            )

    required_ready = all(item.state == ComponentState.READY for item in components if item.required)
    real_vision_verified = any(
        item.component_id == "rank03_image_assessment"
        and item.state == ComponentState.READY
        and item.loaded is True
        and not item.synthetic
        for item in components
    )
    required_llm_verified = not narrator_required or any(
        item.component_id == "llm_evidence_composer"
        and item.state == ComponentState.READY
        and item.loaded is True
        for item in components
    )
    runtime_verified = (
        real_vision_required and real_vision_verified and required_llm_verified and required_ready
    )
    return CapabilitySnapshot(
        status="ok" if required_ready else "degraded",
        mode=(
            "synthetic_demo"
            if is_mock
            else "real_rank03_with_local_llm"
            if narrator_required
            else "real_rank03"
        ),
        runtime_verified=runtime_verified,
        components=components,
    )
