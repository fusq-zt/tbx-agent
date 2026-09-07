"""Structured case state and deterministic post-action guards.

This module deliberately has no dependency on the service, planner, storage, or
vision backends.  It turns an existing ``CaseRecord`` (or a compatible object)
into the bounded, serializable state that an Agent controller is allowed to
inspect.  The functions here do not execute tools and never alter model output.

In particular, an empty detection list is not enough to infer that D-FINE ran.
The explicit detector execution marker (or a future ``localization_evidence``
record) distinguishes ``not_requested`` from ``completed_no_detection``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStatus(StrEnum):
    """Lifecycle state for one evidence source."""

    NOT_REQUESTED = "not_requested"
    NOT_RUN = "not_run"
    AVAILABLE = "available"
    COMPLETED = "completed"
    COMPLETED_NO_DETECTION = "completed_no_detection"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    EVIDENCE_GAP = "evidence_gap"
    STALE = "stale"


class AgentAction(StrEnum):
    """Finite action space exposed to the bounded controller."""

    CLASSIFY_CURRENT_CXR = "classify_current_cxr"
    GET_CLASSIFICATION_EVIDENCE = "get_classification_evidence"
    LOCALIZE_CURRENT_CXR = "localize_current_cxr"
    INSPECT_IMAGE_QUALITY = "inspect_image_quality"
    RETRIEVE_PRIOR_STUDIES = "retrieve_prior_studies"
    INSPECT_ANATOMICAL_CONTEXT = "inspect_anatomical_context"
    SEARCH_TB_KNOWLEDGE = "search_tb_knowledge"
    SEARCH_TB_GUIDANCE = "search_tb_knowledge"
    # Python compatibility alias for older callers.  Persisted traces written
    # by previous releases are accepted by ``_missing_`` below, while all new
    # actions and receipts expose the query-first public capability name.
    RETRIEVE_GUIDELINE = "search_tb_knowledge"
    DESCRIBE_CAPABILITIES = "describe_capabilities"
    EMERGENCY_TRIAGE = "emergency_triage"
    STOP = "stop"
    REFER_TO_HUMAN = "refer_to_human"

    @classmethod
    def _missing_(cls, value: object):
        if value in {"retrieve_guideline", "search_tb_guidance"}:
            return cls.SEARCH_TB_KNOWLEDGE
        return None


class ScreeningDisposition(StrEnum):
    """System disposition; separate from the classifier's immutable output."""

    UNRESOLVED = "unresolved"
    SCREEN_NEGATIVE = "screen_negative"
    SCREEN_POSITIVE = "screen_positive"
    NON_TB_ABNORMAL = "non_tb_abnormal"
    REVIEW_REQUIRED = "review_required"
    TECHNICAL_FAILURE = "technical_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ConflictFlag(StrEnum):
    """Deterministic cross-evidence conflicts and uncertainty signals."""

    CLASSIFIER_POSITIVE_DETECTOR_NEGATIVE = "classifier_positive_detector_negative"
    CLASSIFIER_NEGATIVE_DETECTOR_POSITIVE = "classifier_negative_detector_positive"
    CLASSIFICATION_LOW_MARGIN = "classification_low_margin"
    CLASSIFICATION_NEAR_THRESHOLD = "classification_near_threshold"
    TECHNICAL_QUALITY_CONFLICT = "technical_quality_conflict"
    TOOL_FAILURE_CONFLICT = "tool_failure_conflict"


class EvidenceGap(StrEnum):
    """Structured reason why the current task is not yet supportable."""

    CLASSIFICATION_MISSING = "classification_missing"
    CLASSIFICATION_FAILED = "classification_failed"
    LOCALIZATION_NOT_RUN = "localization_not_run"
    LOCALIZATION_FAILED = "localization_failed"
    LOCALIZATION_UNSUPPORTED = "localization_unsupported"
    QUALITY_EVIDENCE_MISSING = "quality_evidence_missing"
    QUALITY_INSPECTION_FAILED = "quality_inspection_failed"
    DIAGNOSTIC_GUIDANCE_MISSING = "diagnostic_guidance_missing"
    DIAGNOSTIC_GUIDANCE_FAILED = "diagnostic_guidance_failed"
    TREATMENT_GUIDANCE_MISSING = "treatment_guidance_missing"
    TREATMENT_GUIDANCE_FAILED = "treatment_guidance_failed"
    PRIOR_STUDY_NOT_CHECKED = "prior_study_not_checked"
    PRIOR_IMAGE_NOT_FOUND = "prior_image_not_found"
    PRIOR_RETRIEVAL_FAILED = "prior_retrieval_failed"
    LONGITUDINAL_EVIDENCE_MISSING = "longitudinal_evidence_missing"
    LONGITUDINAL_MODEL_UNAVAILABLE = "longitudinal_model_unavailable"
    ANATOMY_EVIDENCE_MISSING = "anatomy_evidence_missing"
    ANATOMY_TOOL_FAILED = "anatomy_tool_failed"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"


class CalibrationStatus(StrEnum):
    """Probability calibration availability for classification scores."""

    UNAVAILABLE = "unavailable"


class EvidenceKind(StrEnum):
    """Evidence requirements supplied by the structured task layer."""

    CLASSIFICATION = "classification"
    LOCALIZATION = "localization"
    QUALITY = "quality"
    DIAGNOSTIC = "diagnostic"
    TREATMENT = "treatment"
    PRIOR = "prior"
    LONGITUDINAL = "longitudinal"
    ANATOMY = "anatomy"


class _DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceState(_DomainModel):
    """One evidence slot without backend-specific payloads."""

    status: EvidenceStatus
    source_action: AgentAction | None = None
    item_count: int | None = Field(default=None, ge=0)
    reason_codes: list[str] = Field(default_factory=list)


class CaseState(_DomainModel):
    """Approved structured context for one bounded Agent task."""

    case_id: str | None = None
    subject_id: str | None = None
    current_task: str = Field(min_length=1)
    active_intent: str | None = None
    required_evidence: list[EvidenceKind] = Field(default_factory=list)

    classification_evidence: EvidenceState
    quality_evidence: EvidenceState
    localization_evidence: EvidenceState
    diagnostic_guideline_evidence: EvidenceState
    treatment_guideline_evidence: EvidenceState
    prior_evidence: EvidenceState
    longitudinal_evidence: EvidenceState
    anatomical_evidence: EvidenceState

    predicted_class: str | None = None
    classification_scores: dict[str, float] = Field(default_factory=dict)
    top1_class: str | None = None
    top1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top2_class: str | None = None
    top2_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top1_top2_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_status: CalibrationStatus = CalibrationStatus.UNAVAILABLE
    classifier_operating_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    classification_low_margin_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    classification_threshold_proximity: float = Field(default=0.05, ge=0.0, le=1.0)
    quality_conflict_codes: list[str] = Field(default_factory=list)

    uncertainty_flags: list[str] = Field(default_factory=list)
    conflict_flags: list[ConflictFlag] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    screening_disposition: ScreeningDisposition = ScreeningDisposition.UNRESOLVED

    tool_budget: int = Field(default=4, ge=0)
    tool_calls_used: int = Field(default=0, ge=0)
    completed_actions: list[AgentAction] = Field(default_factory=list)
    failed_actions: list[AgentAction] = Field(default_factory=list)
    last_action: AgentAction | None = None

    task_complete: bool = False
    human_review_required: bool = False
    human_review_reasons: list[str] = Field(default_factory=list)
    terminal_action: AgentAction | None = None
    stop_reason: str | None = None

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.tool_budget - self.tool_calls_used)


_MISSING = object()
_SATISFIED_STATUSES = {
    EvidenceStatus.AVAILABLE,
    EvidenceStatus.COMPLETED,
    EvidenceStatus.COMPLETED_NO_DETECTION,
}
_DETECTOR_NOT_REQUESTED_REF = "detector_execution:not_requested"
_DETECTOR_COMPLETED_REFS = {
    "detector_execution:on_demand_completed",
    "detector_execution:completed",
}


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _coerce_status(value: EvidenceStatus | str) -> EvidenceStatus:
    if isinstance(value, EvidenceStatus):
        return value
    aliases = {
        "pending": EvidenceStatus.NOT_RUN,
        "running": EvidenceStatus.NOT_RUN,
        "success": EvidenceStatus.COMPLETED,
        "succeeded": EvidenceStatus.COMPLETED,
        "technical_failure": EvidenceStatus.FAILED,
        "unavailable": EvidenceStatus.UNSUPPORTED,
        "disabled": EvidenceStatus.UNSUPPORTED,
        "not_found": EvidenceStatus.EVIDENCE_GAP,
        "completed_with_refinement_failure": EvidenceStatus.COMPLETED,
    }
    normalized = str(_enum_value(value)).strip().casefold()
    if normalized in aliases:
        return aliases[normalized]
    return EvidenceStatus(normalized)


def _action_list(values: Iterable[AgentAction | str]) -> list[AgentAction]:
    return list(dict.fromkeys(AgentAction(_enum_value(value)) for value in values))


def _slot(
    status: EvidenceStatus | str,
    action: AgentAction,
    *,
    item_count: int | None = None,
    reason_codes: Iterable[str] = (),
) -> EvidenceState:
    return EvidenceState(
        status=_coerce_status(status),
        source_action=action,
        item_count=item_count,
        reason_codes=list(dict.fromkeys(str(code) for code in reason_codes if str(code))),
    )


def _duck_typed_slot(
    value: Any,
    *,
    action: AgentAction,
    fallback: EvidenceStatus,
) -> EvidenceState:
    if value is None:
        return _slot(fallback, action)
    if isinstance(value, EvidenceState):
        return value.model_copy(deep=True)
    raw_status = _read(value, "status", _MISSING)
    status = fallback if raw_status is _MISSING else _coerce_status(raw_status)
    raw_items = _read(value, "detections", _MISSING)
    item_count = _read(value, "item_count", None)
    if (
        item_count is None
        and raw_items is not _MISSING
        and raw_items is not None
        and status in {EvidenceStatus.COMPLETED, EvidenceStatus.COMPLETED_NO_DETECTION}
    ):
        item_count = len(raw_items)
    if status == EvidenceStatus.COMPLETED and item_count == 0:
        status = EvidenceStatus.COMPLETED_NO_DETECTION
    reasons = _read(value, "reason_codes", ()) or ()
    return _slot(status, action, item_count=item_count, reason_codes=reasons)


def _localization_slot(case: Any, vision: Any) -> EvidenceState:
    explicit = _read(case, "localization_evidence", _MISSING)
    if explicit is not _MISSING:
        return _duck_typed_slot(
            explicit,
            action=AgentAction.LOCALIZE_CURRENT_CXR,
            fallback=EvidenceStatus.NOT_REQUESTED,
        )
    if vision is None:
        return _slot(EvidenceStatus.NOT_RUN, AgentAction.LOCALIZE_CURRENT_CXR)

    refs = {str(item) for item in (_read(vision, "artifact_refs", ()) or ())}
    detections = _read(vision, "detections", ()) or ()
    failed = any(ref.startswith("detector_execution:failed") for ref in refs)
    completed = bool(refs & _DETECTOR_COMPLETED_REFS) or any(
        ref.startswith("detector_execution:on_demand_completed") for ref in refs
    )
    if failed:
        return _slot(
            EvidenceStatus.FAILED,
            AgentAction.LOCALIZE_CURRENT_CXR,
            reason_codes=("detector_execution_failed",),
        )
    if completed:
        status = (
            EvidenceStatus.COMPLETED
            if len(detections) > 0
            else EvidenceStatus.COMPLETED_NO_DETECTION
        )
        return _slot(
            status,
            AgentAction.LOCALIZE_CURRENT_CXR,
            item_count=len(detections),
        )
    if _DETECTOR_NOT_REQUESTED_REF in refs:
        return _slot(EvidenceStatus.NOT_REQUESTED, AgentAction.LOCALIZE_CURRENT_CXR)

    # Legacy records predate the lazy-execution marker and ran D-FINE eagerly.
    status = (
        EvidenceStatus.COMPLETED
        if len(detections) > 0
        else EvidenceStatus.COMPLETED_NO_DETECTION
    )
    return _slot(status, AgentAction.LOCALIZE_CURRENT_CXR, item_count=len(detections))


def _classification_summary(vision: Any) -> tuple[
    str | None,
    dict[str, float],
    str | None,
    float | None,
    str | None,
    float | None,
    float | None,
]:
    if vision is None:
        return None, {}, None, None, None, None, None
    raw_scores = _read(vision, "class_probabilities", {}) or {}
    scores = {
        str(_enum_value(name)): float(score)
        for name, score in raw_scores.items()
        if isinstance(score, (int, float)) and math.isfinite(score)
    }
    order = [str(item) for item in (_read(vision, "class_probability_order", ()) or ())]
    order_index = {name: index for index, name in enumerate(order)}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], order_index.get(item[0], 999)))
    top1_class, top1_score = ranked[0] if ranked else (None, None)
    top2_class, top2_score = ranked[1] if len(ranked) > 1 else (None, None)
    margin = (
        max(0.0, top1_score - top2_score)
        if top1_score is not None and top2_score is not None
        else None
    )
    # The immutable model output is read only from VisionEvidence.  A fusion or
    # localization record is intentionally never consulted for this field.
    predicted = _enum_value(_read(vision, "predicted_class", None))
    predicted_class = None if predicted is None else str(predicted)
    return (
        predicted_class,
        scores,
        top1_class,
        top1_score,
        top2_class,
        top2_score,
        margin,
    )


def _classification_slot(case: Any, vision: Any) -> EvidenceState:
    raw_execution_status = _read(case, "classification_status", _MISSING)
    if raw_execution_status is not _MISSING:
        execution_status = str(_enum_value(raw_execution_status)).casefold()
        if execution_status == "not_requested":
            return _slot(
                EvidenceStatus.NOT_REQUESTED,
                AgentAction.CLASSIFY_CURRENT_CXR,
            )
        if execution_status == "failed":
            return _slot(
                EvidenceStatus.FAILED,
                AgentAction.CLASSIFY_CURRENT_CXR,
                reason_codes=(
                    _read(case, "classification_error_code", None)
                    or "classification_failed",
                ),
            )
        if execution_status == "unavailable":
            return _slot(
                EvidenceStatus.UNSUPPORTED,
                AgentAction.CLASSIFY_CURRENT_CXR,
                reason_codes=(
                    _read(case, "classification_error_code", None)
                    or "classification_unavailable",
                ),
            )
    if vision is None:
        return _slot(EvidenceStatus.NOT_RUN, AgentAction.CLASSIFY_CURRENT_CXR)
    predicted = _read(vision, "predicted_class", None)
    scores = _read(vision, "class_probabilities", {}) or {}
    if predicted is None or not scores:
        return _slot(
            EvidenceStatus.EVIDENCE_GAP,
            AgentAction.CLASSIFY_CURRENT_CXR,
            reason_codes=("classification_unresolved",),
        )
    return _slot(EvidenceStatus.AVAILABLE, AgentAction.CLASSIFY_CURRENT_CXR)


def _quality_slot(case: Any, vision: Any) -> EvidenceState:
    raw_status_value = _read(case, "image_quality_status", _MISSING)
    if raw_status_value is _MISSING or str(raw_status_value).casefold() == "unknown":
        if vision is None:
            return _slot(EvidenceStatus.NOT_RUN, AgentAction.INSPECT_IMAGE_QUALITY)
        raw_status_value = _read(vision, "image_quality_status", "")
        codes = _read(vision, "image_quality_codes", ()) or ()
    else:
        codes = _read(case, "image_quality_codes", ()) or ()
    raw_status = str(_enum_value(raw_status_value)).casefold()
    if raw_status == "technical_failure":
        return _slot(
            EvidenceStatus.FAILED,
            AgentAction.INSPECT_IMAGE_QUALITY,
            reason_codes=("technical_failure", *codes),
        )
    if raw_status in {"not_evaluated", "not_run", ""}:
        return _slot(EvidenceStatus.NOT_RUN, AgentAction.INSPECT_IMAGE_QUALITY)
    return _slot(
        EvidenceStatus.AVAILABLE,
        AgentAction.INSPECT_IMAGE_QUALITY,
        reason_codes=codes,
    )


def _disposition_for_classification(
    classification: EvidenceState,
    quality: EvidenceState,
    predicted_class: str | None,
) -> ScreeningDisposition:
    if quality.status == EvidenceStatus.FAILED or classification.status == EvidenceStatus.FAILED:
        return ScreeningDisposition.TECHNICAL_FAILURE
    if classification.status not in _SATISFIED_STATUSES or predicted_class is None:
        return ScreeningDisposition.INSUFFICIENT_EVIDENCE
    if predicted_class == "tb":
        return ScreeningDisposition.SCREEN_POSITIVE
    if predicted_class == "sick_non_tb":
        return ScreeningDisposition.NON_TB_ABNORMAL
    if predicted_class == "healthy":
        return ScreeningDisposition.SCREEN_NEGATIVE
    return ScreeningDisposition.INSUFFICIENT_EVIDENCE


def build_case_state(
    case: Any,
    task: str,
    active_intent: str | None = None,
    *,
    tool_budget: int = 4,
    tool_calls_used: int = 0,
    completed_actions: Iterable[AgentAction | str] = (),
    failed_actions: Iterable[AgentAction | str] = (),
    last_action: AgentAction | str | None = None,
    diagnostic_status: EvidenceStatus | str = EvidenceStatus.NOT_REQUESTED,
    treatment_status: EvidenceStatus | str = EvidenceStatus.NOT_REQUESTED,
    prior_status: EvidenceStatus | str = EvidenceStatus.NOT_REQUESTED,
    longitudinal_status: EvidenceStatus | str = EvidenceStatus.NOT_REQUESTED,
    anatomy_status: EvidenceStatus | str = EvidenceStatus.NOT_REQUESTED,
    required_evidence: Iterable[EvidenceKind | str] | None = None,
    classification_low_margin_threshold: float | None = None,
    classification_threshold_proximity: float = 0.05,
    quality_conflict_codes: Iterable[str] = (),
) -> CaseState:
    """Build controller state from the current ``CaseRecord`` contract.

    ``case`` is intentionally duck typed so a future CaseRecord can add explicit
    localization/prior/anatomy evidence without creating a schema import cycle.
    Existing records are interpreted from ``vision_evidence.artifact_refs``.
    """

    if not task.strip():
        raise ValueError("task must not be empty")
    vision = _read(case, "vision_evidence", None) if case is not None else None
    classification = _classification_slot(case, vision)
    quality = _quality_slot(case, vision)
    localization = _localization_slot(case, vision)
    (
        predicted_class,
        scores,
        top1_class,
        top1_score,
        top2_class,
        top2_score,
        margin,
    ) = _classification_summary(vision)
    threshold = _read(vision, "classifier_threshold", None) if vision is not None else None

    diagnostic = _duck_typed_slot(
        _read(case, "diagnostic_guideline_evidence", None),
        action=AgentAction.RETRIEVE_GUIDELINE,
        fallback=_coerce_status(diagnostic_status),
    )
    treatment = _duck_typed_slot(
        _read(case, "treatment_guideline_evidence", None),
        action=AgentAction.RETRIEVE_GUIDELINE,
        fallback=_coerce_status(treatment_status),
    )
    prior = _duck_typed_slot(
        _read(case, "prior_evidence", None),
        action=AgentAction.RETRIEVE_PRIOR_STUDIES,
        fallback=_coerce_status(prior_status),
    )
    longitudinal = _duck_typed_slot(
        _read(case, "longitudinal_evidence", None),
        action=AgentAction.RETRIEVE_PRIOR_STUDIES,
        fallback=_coerce_status(longitudinal_status),
    )
    anatomy = _duck_typed_slot(
        _read(case, "anatomical_evidence", None),
        action=AgentAction.INSPECT_ANATOMICAL_CONTEXT,
        fallback=_coerce_status(anatomy_status),
    )
    requirements = (
        _requirements_from_intent(active_intent)
        if required_evidence is None
        else tuple(EvidenceKind(_enum_value(item)) for item in required_evidence)
    )

    state = CaseState(
        case_id=_read(case, "case_id", None),
        subject_id=_read(case, "user_id", None),
        current_task=task.strip(),
        active_intent=active_intent,
        required_evidence=list(dict.fromkeys(requirements)),
        classification_evidence=classification,
        quality_evidence=quality,
        localization_evidence=localization,
        diagnostic_guideline_evidence=diagnostic,
        treatment_guideline_evidence=treatment,
        prior_evidence=prior,
        longitudinal_evidence=longitudinal,
        anatomical_evidence=anatomy,
        predicted_class=predicted_class,
        classification_scores=scores,
        top1_class=top1_class,
        top1_score=top1_score,
        top2_class=top2_class,
        top2_score=top2_score,
        top1_top2_margin=margin,
        calibration_status=CalibrationStatus.UNAVAILABLE,
        classifier_operating_threshold=threshold,
        classification_low_margin_threshold=classification_low_margin_threshold,
        classification_threshold_proximity=classification_threshold_proximity,
        quality_conflict_codes=list(
            dict.fromkeys(str(code) for code in quality_conflict_codes if str(code))
        ),
        screening_disposition=_disposition_for_classification(
            classification, quality, predicted_class
        ),
        tool_budget=tool_budget,
        tool_calls_used=tool_calls_used,
        completed_actions=_action_list(completed_actions),
        failed_actions=_action_list(failed_actions),
        last_action=None if last_action is None else AgentAction(_enum_value(last_action)),
    )
    return detect_evidence_gaps(detect_conflicts(state))


def detect_conflicts(state: CaseState) -> CaseState:
    """Return a copy with deterministic conflicts derived from current evidence."""

    conflicts: list[ConflictFlag] = []
    uncertainty: list[str] = []
    requirements = set(_task_requirements(state))
    classification_relevant = EvidenceKind.CLASSIFICATION in requirements
    visual_quality_relevant = bool(
        requirements
        & {
            EvidenceKind.CLASSIFICATION,
            EvidenceKind.LOCALIZATION,
            EvidenceKind.QUALITY,
            EvidenceKind.ANATOMY,
        }
    )

    # D-FINE is an advisory, on-demand localizer. A displayed box or an empty
    # localization result is not an independent disease classifier and must
    # never contradict or override the immutable three-class argmax result.

    if (
        classification_relevant
        and state.classification_low_margin_threshold is not None
        and state.top1_top2_margin is not None
        and state.top1_top2_margin <= state.classification_low_margin_threshold
    ):
        # The deployed classifier uses the immutable three-class argmax
        # decision.  A small score margin is useful context, but it is not an
        # independent contradiction and must not open an interactive review.
        uncertainty.append(ConflictFlag.CLASSIFICATION_LOW_MARGIN.value)
    tb_score = state.classification_scores.get("tb")
    if (
        classification_relevant
        and state.classifier_operating_threshold is not None
        and tb_score is not None
        and abs(tb_score - state.classifier_operating_threshold)
        <= state.classification_threshold_proximity
    ):
        uncertainty.append(ConflictFlag.CLASSIFICATION_NEAR_THRESHOLD.value)

    if (
        classification_relevant
        and state.classification_scores
        and state.calibration_status == CalibrationStatus.UNAVAILABLE
    ):
        uncertainty.append("classification_calibration_unavailable")
    if visual_quality_relevant and state.quality_evidence.reason_codes:
        uncertainty.append("input_quality_warning")

    configured_quality_conflict = bool(
        set(state.quality_evidence.reason_codes) & set(state.quality_conflict_codes)
    )
    if visual_quality_relevant and state.predicted_class is not None and (
        state.quality_evidence.status == EvidenceStatus.FAILED or configured_quality_conflict
    ):
        conflicts.append(ConflictFlag.TECHNICAL_QUALITY_CONFLICT)
    evidence_slots = {
        EvidenceKind.CLASSIFICATION: state.classification_evidence,
        EvidenceKind.QUALITY: state.quality_evidence,
        EvidenceKind.LOCALIZATION: state.localization_evidence,
        EvidenceKind.DIAGNOSTIC: state.diagnostic_guideline_evidence,
        EvidenceKind.TREATMENT: state.treatment_guideline_evidence,
        EvidenceKind.PRIOR: state.prior_evidence,
        EvidenceKind.LONGITUDINAL: state.longitudinal_evidence,
        EvidenceKind.ANATOMY: state.anatomical_evidence,
    }
    if state.failed_actions or any(
        evidence_slots[kind].status == EvidenceStatus.FAILED for kind in requirements
    ):
        conflicts.append(ConflictFlag.TOOL_FAILURE_CONFLICT)

    disposition = _disposition_for_classification(
        state.classification_evidence,
        state.quality_evidence,
        state.predicted_class,
    )
    if conflicts and disposition != ScreeningDisposition.TECHNICAL_FAILURE:
        disposition = ScreeningDisposition.REVIEW_REQUIRED
    return state.model_copy(
        update={
            "conflict_flags": list(dict.fromkeys(conflicts)),
            "uncertainty_flags": list(dict.fromkeys(uncertainty)),
            "screening_disposition": disposition,
        },
        deep=True,
    )


def _requirements_from_intent(active_intent: str | None) -> tuple[EvidenceKind, ...]:
    """Legacy bridge from an already-structured intent, never from raw task text."""

    intent = (active_intent or "").casefold()
    intent_requirements = {
        "localization": (EvidenceKind.LOCALIZATION,),
        "case_localization": (EvidenceKind.LOCALIZATION,),
        "image_quality": (EvidenceKind.QUALITY,),
        "quality": (EvidenceKind.QUALITY,),
        "diagnostic": (EvidenceKind.DIAGNOSTIC,),
        "next_test": (EvidenceKind.DIAGNOSTIC,),
        "treatment": (EvidenceKind.TREATMENT,),
        "comparison": (EvidenceKind.PRIOR, EvidenceKind.LONGITUDINAL),
        "longitudinal": (EvidenceKind.PRIOR, EvidenceKind.LONGITUDINAL),
        "anatomy": (EvidenceKind.ANATOMY,),
    }
    return intent_requirements.get(intent, (EvidenceKind.CLASSIFICATION,))


def _task_requirements(state: CaseState) -> tuple[EvidenceKind, ...]:
    return tuple(state.required_evidence) or (EvidenceKind.CLASSIFICATION,)


def _gap_for_slot(name: EvidenceKind, slot: EvidenceState) -> EvidenceGap | None:
    if slot.status in _SATISFIED_STATUSES:
        return None
    if name == EvidenceKind.CLASSIFICATION:
        return (
            EvidenceGap.CLASSIFICATION_FAILED
            if slot.status == EvidenceStatus.FAILED
            else EvidenceGap.CLASSIFICATION_MISSING
        )
    if name == EvidenceKind.LOCALIZATION:
        if slot.status == EvidenceStatus.FAILED:
            return EvidenceGap.LOCALIZATION_FAILED
        if slot.status == EvidenceStatus.UNSUPPORTED:
            return EvidenceGap.LOCALIZATION_UNSUPPORTED
        return EvidenceGap.LOCALIZATION_NOT_RUN
    if name == EvidenceKind.QUALITY:
        return (
            EvidenceGap.QUALITY_INSPECTION_FAILED
            if slot.status == EvidenceStatus.FAILED
            else EvidenceGap.QUALITY_EVIDENCE_MISSING
        )
    if name == EvidenceKind.DIAGNOSTIC:
        return (
            EvidenceGap.DIAGNOSTIC_GUIDANCE_FAILED
            if slot.status == EvidenceStatus.FAILED
            else EvidenceGap.DIAGNOSTIC_GUIDANCE_MISSING
        )
    if name == EvidenceKind.TREATMENT:
        return (
            EvidenceGap.TREATMENT_GUIDANCE_FAILED
            if slot.status == EvidenceStatus.FAILED
            else EvidenceGap.TREATMENT_GUIDANCE_MISSING
        )
    if name == EvidenceKind.PRIOR:
        if slot.status == EvidenceStatus.FAILED:
            return EvidenceGap.PRIOR_RETRIEVAL_FAILED
        if slot.status in {EvidenceStatus.EVIDENCE_GAP, EvidenceStatus.UNSUPPORTED}:
            return EvidenceGap.PRIOR_IMAGE_NOT_FOUND
        return EvidenceGap.PRIOR_STUDY_NOT_CHECKED
    if name == EvidenceKind.LONGITUDINAL:
        if slot.status == EvidenceStatus.UNSUPPORTED:
            return EvidenceGap.LONGITUDINAL_MODEL_UNAVAILABLE
        return EvidenceGap.LONGITUDINAL_EVIDENCE_MISSING
    if name == EvidenceKind.ANATOMY:
        return (
            EvidenceGap.ANATOMY_TOOL_FAILED
            if slot.status == EvidenceStatus.FAILED
            else EvidenceGap.ANATOMY_EVIDENCE_MISSING
        )
    raise ValueError(f"unknown evidence slot: {name}")


def detect_evidence_gaps(state: CaseState) -> CaseState:
    """Return a copy with only the gaps relevant to the current user task."""

    slots = {
        EvidenceKind.CLASSIFICATION: state.classification_evidence,
        EvidenceKind.LOCALIZATION: state.localization_evidence,
        EvidenceKind.QUALITY: state.quality_evidence,
        EvidenceKind.DIAGNOSTIC: state.diagnostic_guideline_evidence,
        EvidenceKind.TREATMENT: state.treatment_guideline_evidence,
        EvidenceKind.PRIOR: state.prior_evidence,
        EvidenceKind.LONGITUDINAL: state.longitudinal_evidence,
        EvidenceKind.ANATOMY: state.anatomical_evidence,
    }
    gaps: list[EvidenceGap] = []
    for name in _task_requirements(state):
        # Longitudinal analysis cannot begin until a prior study has actually
        # been found.  Reporting only the current blocking gap also supports a
        # one-action-per-step controller.
        if name == EvidenceKind.LONGITUDINAL and EvidenceKind.PRIOR in _task_requirements(state):
            prior_gap = _gap_for_slot(EvidenceKind.PRIOR, slots[EvidenceKind.PRIOR])
            if prior_gap is not None:
                continue
        gap = _gap_for_slot(name, slots[name])
        if gap is not None:
            gaps.append(gap)
    return state.model_copy(update={"evidence_gaps": list(dict.fromkeys(gaps))}, deep=True)


def post_action_check(state: CaseState) -> CaseState:
    """Apply bounded STOP/HUMAN_REVIEW rules after one action.

    This is a deterministic evidence-completeness check, not free-form
    reflection.  It never changes ``predicted_class`` or classification scores.
    """

    checked = detect_evidence_gaps(detect_conflicts(state))
    gaps = list(checked.evidence_gaps)
    explicit_human = (
        checked.last_action == AgentAction.REFER_TO_HUMAN
        or AgentAction.REFER_TO_HUMAN in checked.completed_actions
    )
    unrecoverable_gaps = {
        EvidenceGap.CLASSIFICATION_FAILED,
        EvidenceGap.LOCALIZATION_FAILED,
        EvidenceGap.LOCALIZATION_UNSUPPORTED,
        EvidenceGap.QUALITY_INSPECTION_FAILED,
        EvidenceGap.DIAGNOSTIC_GUIDANCE_FAILED,
        EvidenceGap.TREATMENT_GUIDANCE_FAILED,
        EvidenceGap.PRIOR_RETRIEVAL_FAILED,
        EvidenceGap.ANATOMY_TOOL_FAILED,
    }
    terminal_gap_stops = {
        EvidenceGap.PRIOR_IMAGE_NOT_FOUND,
        EvidenceGap.LONGITUDINAL_MODEL_UNAVAILABLE,
    }
    reasons = [flag.value for flag in checked.conflict_flags]
    reasons.extend(gap.value for gap in gaps if gap in unrecoverable_gaps)
    if explicit_human:
        reasons.append("controller_requested_human_review")

    has_terminal_gap = any(gap in terminal_gap_stops for gap in gaps)
    budget_exhausted = checked.remaining_tool_calls == 0 and bool(gaps) and not has_terminal_gap
    if budget_exhausted:
        if EvidenceGap.TOOL_BUDGET_EXHAUSTED not in gaps:
            gaps.append(EvidenceGap.TOOL_BUDGET_EXHAUSTED)
        reasons.append(EvidenceGap.TOOL_BUDGET_EXHAUSTED.value)

    must_refer = bool(checked.conflict_flags) or explicit_human or budget_exhausted or any(
        gap in unrecoverable_gaps for gap in gaps
    )
    if must_refer:
        disposition = checked.screening_disposition
        if disposition != ScreeningDisposition.TECHNICAL_FAILURE:
            disposition = ScreeningDisposition.REVIEW_REQUIRED
        return checked.model_copy(
            update={
                "evidence_gaps": list(dict.fromkeys(gaps)),
                "screening_disposition": disposition,
                "task_complete": False,
                "human_review_required": True,
                "human_review_reasons": list(dict.fromkeys(reasons)),
                "terminal_action": AgentAction.REFER_TO_HUMAN,
                "stop_reason": "human_review_required",
            },
            deep=True,
        )

    if has_terminal_gap:
        return checked.model_copy(
            update={
                "screening_disposition": ScreeningDisposition.INSUFFICIENT_EVIDENCE,
                "task_complete": False,
                "human_review_required": False,
                "human_review_reasons": [],
                "terminal_action": AgentAction.STOP,
                "stop_reason": "required_evidence_unavailable",
            },
            deep=True,
        )

    explicit_stop = (
        checked.last_action == AgentAction.STOP or AgentAction.STOP in checked.completed_actions
    )
    if not gaps:
        return checked.model_copy(
            update={
                "task_complete": True,
                "human_review_required": False,
                "human_review_reasons": [],
                "terminal_action": AgentAction.STOP,
                "stop_reason": "task_evidence_complete",
            },
            deep=True,
        )
    if explicit_stop:
        return checked.model_copy(
            update={
                "screening_disposition": ScreeningDisposition.INSUFFICIENT_EVIDENCE,
                "task_complete": False,
                "human_review_required": False,
                "human_review_reasons": [],
                "terminal_action": AgentAction.STOP,
                "stop_reason": "controller_stopped_with_evidence_gap",
            },
            deep=True,
        )
    return checked.model_copy(
        update={
            "task_complete": False,
            "human_review_required": False,
            "human_review_reasons": [],
            "terminal_action": None,
            "stop_reason": None,
        },
        deep=True,
    )


__all__ = [
    "AgentAction",
    "CalibrationStatus",
    "CaseState",
    "ConflictFlag",
    "EvidenceGap",
    "EvidenceKind",
    "EvidenceState",
    "EvidenceStatus",
    "ScreeningDisposition",
    "build_case_state",
    "detect_conflicts",
    "detect_evidence_gaps",
    "post_action_check",
]
