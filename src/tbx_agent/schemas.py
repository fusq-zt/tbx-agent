from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# D-FINE exposes at most one candidate per object query.  The frozen rank03
# detector contract uses 300 queries, so accepting a longer list would describe
# evidence that this backend cannot have produced and would make downstream
# per-candidate workers vulnerable to unbounded fan-out.
MAX_DETECTION_CANDIDATES = 300


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ActorRole(StrEnum):
    USER = "user"
    REVIEWER = "reviewer"
    ADMIN = "admin"


class VisualResult(StrEnum):
    MODEL_FLAGGED = "model_flagged"
    MODEL_NOT_FLAGGED = "model_not_flagged"
    NON_TB_ABNORMAL = "non_tb_abnormal"
    INDETERMINATE = "indeterminate"
    TECHNICAL_FAILURE = "technical_failure"
    PENDING_HUMAN_REVIEW = "pending_human_review"


class ClassifierClass(StrEnum):
    HEALTHY = "healthy"
    SICK_NON_TB = "sick_non_tb"
    TB = "tb"


class ClassificationExecutionStatus(StrEnum):
    """Persisted lifecycle of the on-demand ConvNeXt classification tool."""

    NOT_REQUESTED = "not_requested"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class ReviewStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    COMPLETED = "completed"


class ReviewOrigin(StrEnum):
    LEGACY_INTERACTIVE = "legacy_interactive"
    BATCH_SCREENING = "batch_screening"


class ResponseKind(StrEnum):
    GENERAL_ANSWER = "general_answer"
    VISUAL_SCREENING_RESULT = "visual_screening_result"
    LOCALIZATION_RESULT = "localization_result"
    DIAGNOSTIC_INFORMATION = "diagnostic_information"
    NEXT_TEST_INFORMATION = "next_test_information"
    TREATMENT_EDUCATION = "treatment_education"
    ACTIVE_SCREENING_QUESTION = "active_screening_question"
    ACTIVE_SCREENING_SUMMARY = "active_screening_summary"
    SCREENING_REPORT = "screening_report"
    CASE_EXPLANATION = "case_explanation"
    CAPABILITY_STATEMENT = "capability_statement"
    EMERGENCY_ESCALATION = "emergency_escalation"
    SAFE_ABSTENTION = "safe_abstention"


class Urgency(StrEnum):
    EMERGENCY = "emergency"
    PROMPT_EVALUATION = "prompt_evaluation"
    PRIORITY_SCREENING = "priority_screening"
    ROUTINE_INFORMATION = "routine_information"


class NarrationStatus(StrEnum):
    NOT_CONFIGURED = "not_configured"
    APPLIED = "applied"
    SKIPPED_EMERGENCY = "skipped_emergency"
    SKIPPED_RESPONSE_KIND = "skipped_response_kind"
    FALLBACK_ERROR = "fallback_error"
    REJECTED_BY_SAFETY = "rejected_by_safety"


class GuidelineAnswerStatus(StrEnum):
    """How completely the reviewed snapshot answers one guideline question."""

    ANSWERED = "ANSWERED"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class Citation(StrictModel):
    chunk_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=1_000)
    organization: str = Field(min_length=1, max_length=512)
    publication_year: int = Field(ge=1900, le=2100)
    section: str = Field(min_length=1, max_length=1_000)
    locator: str = Field(min_length=1, max_length=1_000)
    url: str = Field(min_length=1, max_length=2_048)
    support_text: str = Field(min_length=1, max_length=4_000)


class RetrievedGuidelineEvidence(StrictModel):
    """Immutable, user-visible evidence returned by this retrieval turn.

    ``text`` is the reviewed chunk body, not an LLM paraphrase.  The remaining
    fields preserve enough retrieval and source provenance to audit why the
    chunk was admitted for the requested guideline scope.
    """

    chunk_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=12_000)
    source_id: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=1_000)
    organization: str = Field(min_length=1, max_length=512)
    publication_year: int = Field(ge=1900, le=2100)
    section: str = Field(min_length=1, max_length=1_000)
    locator: str = Field(min_length=1, max_length=1_000)
    page: str | None = Field(default=None, min_length=1, max_length=128)
    url: str = Field(min_length=1, max_length=2_048)
    score: float = Field(ge=0.0)
    lexical_score: float = Field(ge=0.0)
    semantic_score: float = Field(ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GroundedGuidelineClaim(StrictModel):
    """An extractive guideline claim bound to evidence from the current turn."""

    text: str = Field(min_length=1, max_length=12_000)
    chunk_ids: list[str] = Field(min_length=1, max_length=8)

    @field_validator("chunk_ids")
    @classmethod
    def chunk_ids_are_unique(cls, value: list[str]):
        if any(not item.strip() for item in value):
            raise ValueError("guideline claim chunk ids must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("guideline claim chunk ids must be unique")
        return value


class DetectionEvidence(StrictModel):
    bbox_xyxy: tuple[float, float, float, float]
    score: float = Field(ge=0.0, le=1.0)
    label: str = "tb_suspicious_region"

    @field_validator("bbox_xyxy")
    @classmethod
    def valid_box(cls, value: tuple[float, float, float, float]):
        x1, y1, x2, y2 = value
        if not all(math.isfinite(item) for item in value):
            raise ValueError("bbox coordinates must be finite")
        if x1 < 0 or y1 < 0:
            raise ValueError("bbox coordinates must be non-negative")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must have positive area")
        return value


class LocalizationEvidence(StrictModel):
    """Independent, version-bound D-FINE execution state.

    An empty detection list is meaningful only when ``status`` is
    ``completed_no_detection``.  ``not_requested`` and ``failed`` are distinct
    states and can therefore never be mistaken for a negative detector result.
    """

    status: Literal[
        "not_requested",
        "completed",
        "completed_no_detection",
        "failed",
        "unsupported",
        "stale",
    ] = "not_requested"
    run_id: str | None = Field(default=None, min_length=1, max_length=256)
    generation_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    case_id: str | None = Field(default=None, min_length=1, max_length=256)
    image_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    detector_model_id: str | None = Field(default=None, min_length=1, max_length=512)
    detector_checkpoint_sha256: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    preprocessing_version: str | None = Field(default=None, min_length=1, max_length=256)
    detections: list[DetectionEvidence] = Field(
        default_factory=list,
        max_length=MAX_DETECTION_CANDIDATES,
    )
    runtime_ms: int | None = Field(default=None, ge=0)
    attempt_count: int = Field(default=0, ge=0)
    error_code: str | None = Field(default=None, min_length=1, max_length=128)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def execution_state_is_unambiguous(self):
        completed = self.status in {"completed", "completed_no_detection"}
        identity = (
            self.run_id,
            self.generation_key,
            self.case_id,
            self.image_sha256,
            self.detector_model_id,
            self.detector_checkpoint_sha256,
            self.preprocessing_version,
        )
        if self.status == "not_requested":
            if any(identity) or self.detections or self.runtime_ms is not None:
                raise ValueError("not-requested localization cannot claim execution evidence")
            if self.attempt_count != 0 or self.error_code is not None:
                raise ValueError("not-requested localization cannot claim an attempt")
        elif completed:
            if any(value is None for value in identity) or self.runtime_ms is None:
                raise ValueError("completed localization requires immutable execution identity")
            if self.error_code is not None or self.attempt_count < 1:
                raise ValueError("completed localization has an invalid attempt/error state")
            if self.status == "completed" and not self.detections:
                raise ValueError("completed localization requires at least one detection")
            if self.status == "completed_no_detection" and self.detections:
                raise ValueError("completed-no-detection localization must have no detections")
        elif self.status == "failed":
            if any(value is None for value in identity[1:]) or self.attempt_count < 1:
                raise ValueError("failed localization requires target identity and an attempt")
            if self.error_code is None or self.detections:
                raise ValueError("failed localization requires an error and no detections")
        elif self.detections:
            raise ValueError("non-completed localization cannot contain detections")
        return self


class VisionEvidence(StrictModel):
    run_id: str
    case_id: str
    image_sha256: str
    image_quality_status: str
    image_quality_codes: list[str] = Field(default_factory=list, max_length=32)
    image_source_format: Literal["PNG", "JPEG", "DICOM", "UNKNOWN"] = "UNKNOWN"
    input_transform_id: str = Field(default="legacy-unknown", min_length=1, max_length=256)
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    classifier_model_id: str
    classifier_checkpoint_sha256: str
    class_probability_order: list[str]
    class_probabilities: dict[str, float]
    top1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top2_score: float | None = Field(default=None, ge=0.0, le=1.0)
    top1_top2_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_weight_version: str | None = Field(default=None, min_length=1, max_length=128)
    execution_status: Literal["completed"] = "completed"
    classifier_decision_rule: Literal[
        "legacy_p_tb_threshold", "native_three_class_argmax", "p_tb_gte_threshold"
    ] = "legacy_p_tb_threshold"
    predicted_class: ClassifierClass | None = None
    classifier_argmax_tied: bool = False
    classifier_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_flagged: bool
    detector_model_id: str
    detector_checkpoint_sha256: str
    detector_decision_role: Literal["legacy_vote", "advisory_localization_only"] = "legacy_vote"
    detector_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    detections: list[DetectionEvidence] = Field(max_length=MAX_DETECTION_CANDIDATES)
    detector_flagged: bool | None
    preprocessing_version: str
    threshold_config_version: str
    runtime_ms: int = Field(ge=0)
    artifact_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("class_probabilities")
    @classmethod
    def probabilities_are_valid(cls, value: dict[str, float]):
        if not value:
            raise ValueError("class probabilities are required")
        if any(
            not math.isfinite(probability) or probability < 0.0 or probability > 1.0
            for probability in value.values()
        ):
            raise ValueError("probabilities must be finite and in [0, 1]")
        if abs(sum(value.values()) - 1.0) > 1e-5:
            raise ValueError("class probabilities must sum to one")
        return value

    @field_validator("image_quality_codes")
    @classmethod
    def quality_codes_are_unique_and_nonempty(cls, value: list[str]):
        if any(not code.strip() for code in value):
            raise ValueError("image quality codes must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("image quality codes must be unique")
        return value

    @model_validator(mode="after")
    def decision_contract_is_consistent(self):
        expected_transform = {
            "PNG": "raster-exif-transpose-rgb-v1",
            "JPEG": "raster-exif-transpose-rgb-v1",
            "DICOM": "dicom-crdx-windowed-rgb-v1",
            "UNKNOWN": "legacy-unknown",
        }[self.image_source_format]
        if self.input_transform_id != expected_transform:
            raise ValueError("image source format and input transform identity are inconsistent")
        expected_order = [item.value for item in ClassifierClass]
        if self.class_probability_order != expected_order:
            raise ValueError(f"class probability order must be {expected_order}")
        if set(self.class_probabilities) != set(expected_order):
            raise ValueError("class probability keys must match the declared class order")

        ranked = sorted(self.class_probabilities.values(), reverse=True)
        expected_top1 = ranked[0]
        expected_top2 = ranked[1]
        expected_margin = expected_top1 - expected_top2
        score_fields = (
            ("top1_score", self.top1_score, expected_top1),
            ("top2_score", self.top2_score, expected_top2),
            ("top1_top2_margin", self.top1_top2_margin, expected_margin),
        )
        for name, observed, expected in score_fields:
            if observed is None:
                object.__setattr__(self, name, expected)
            elif abs(observed - expected) > 1e-6:
                raise ValueError(f"{name} does not match class probabilities")
        if self.classifier_weight_version is None:
            object.__setattr__(
                self,
                "classifier_weight_version",
                self.classifier_checkpoint_sha256,
            )
        elif self.classifier_weight_version != self.classifier_checkpoint_sha256:
            raise ValueError("classifier weight version must match checkpoint identity")

        if self.classifier_decision_rule in {
            "native_three_class_argmax",
            "p_tb_gte_threshold",
        }:
            if self.classifier_decision_rule == "native_three_class_argmax":
                if self.classifier_threshold is not None:
                    raise ValueError(
                        "native argmax evidence must not contain a classifier threshold"
                    )
            elif self.classifier_threshold is None:
                raise ValueError("p_tb threshold evidence requires a classifier threshold")

            # Threshold routing still records the frozen model's native argmax as
            # descriptive evidence. It is never used to override the threshold flag.
            if self.classifier_threshold is not None:
                expected_flag = self.class_probabilities["tb"] >= self.classifier_threshold
            else:
                expected_flag = None
            maximum = max(self.class_probabilities.values())
            winners = [
                ClassifierClass(name)
                for name in expected_order
                if self.class_probabilities[name] == maximum
            ]
            tied = len(winners) != 1
            # Native argmax is stable in the declared class order.  A tie is
            # retained as uncertainty metadata, not converted into a separate
            # routing class or interactive review requirement.
            expected_prediction = winners[0]
            if self.classifier_argmax_tied is not tied:
                raise ValueError("classifier argmax tie state is inconsistent with probabilities")
            compatible_legacy_tie = tied and self.predicted_class is None
            if self.predicted_class != expected_prediction and not compatible_legacy_tie:
                raise ValueError("classifier predicted class is inconsistent with argmax")
            if expected_flag is None:
                expected_flag = expected_prediction == ClassifierClass.TB
            if self.classifier_flagged is not expected_flag:
                if self.classifier_decision_rule == "native_three_class_argmax":
                    raise ValueError("classifier flag must be derived from the native argmax class")
                raise ValueError("classifier flag is inconsistent with its p_tb threshold")
        else:
            if self.classifier_threshold is None:
                raise ValueError("legacy classifier evidence requires a threshold")
            if self.predicted_class is not None or self.classifier_argmax_tied:
                raise ValueError("legacy classifier evidence cannot claim a native argmax result")
            expected_flag = self.class_probabilities["tb"] >= self.classifier_threshold
            if self.classifier_flagged is not expected_flag:
                raise ValueError("legacy classifier flag is inconsistent with its threshold")

        expected_detector_role = (
            "advisory_localization_only"
            if self.classifier_decision_rule in {"native_three_class_argmax", "p_tb_gte_threshold"}
            else "legacy_vote"
        )
        if self.detector_decision_role != expected_detector_role:
            raise ValueError("classifier decision rule and detector decision role must be paired")
        if self.detector_decision_role == "advisory_localization_only":
            if self.detector_threshold is not None or self.detector_flagged is not None:
                raise ValueError("advisory detector evidence cannot contain a decision threshold")
        else:
            if self.detector_threshold is None or self.detector_flagged is None:
                raise ValueError("legacy detector evidence requires a threshold and flag")
            maximum_score = max((item.score for item in self.detections), default=0.0)
            if self.detector_flagged is not (maximum_score >= self.detector_threshold):
                raise ValueError("legacy detector flag is inconsistent with its threshold")
        for detection in self.detections:
            _, _, x2, y2 = detection.bbox_xyxy
            if x2 > self.image_width or y2 > self.image_height:
                raise ValueError("detection bbox must remain within the source image")
        return self


class FusionDecision(StrictModel):
    policy_id: str
    visual_result: VisualResult
    review_required: bool
    review_reasons: list[str] = Field(default_factory=list)
    classifier_decision_rule: Literal[
        "legacy_p_tb_threshold", "native_three_class_argmax", "p_tb_gte_threshold"
    ] = "legacy_p_tb_threshold"
    predicted_class: ClassifierClass | None = None
    classifier_flagged: bool
    detector_decision_role: Literal["legacy_vote", "advisory_localization_only"] = "legacy_vote"
    detector_flagged: bool | None
    max_detector_score: float | None = Field(default=None, ge=0.0, le=1.0)
    clinical_validation: bool = False

    @model_validator(mode="after")
    def routing_contract_is_consistent(self):
        if self.clinical_validation:
            raise ValueError("dataset routing evidence cannot claim clinical validation")
        expected_role = (
            "advisory_localization_only"
            if self.classifier_decision_rule in {"native_three_class_argmax", "p_tb_gte_threshold"}
            else "legacy_vote"
        )
        if self.detector_decision_role != expected_role:
            raise ValueError("fusion classifier rule and detector role must be paired")
        if self.review_required is not (self.visual_result == VisualResult.PENDING_HUMAN_REVIEW):
            raise ValueError("fusion review flag must match the pending-review route")
        if self.classifier_decision_rule == "native_three_class_argmax":
            expected_flag = self.predicted_class == ClassifierClass.TB
            if self.classifier_flagged is not expected_flag:
                raise ValueError("native-argmax fusion flag must follow the tb training class")
            if (
                self.visual_result == VisualResult.MODEL_FLAGGED
                and self.predicted_class != ClassifierClass.TB
            ):
                raise ValueError("flagged native-argmax fusion requires the tb training class")
            if (
                self.visual_result == VisualResult.MODEL_NOT_FLAGGED
                and self.predicted_class != ClassifierClass.HEALTHY
            ):
                raise ValueError(
                    "not-flagged native-argmax fusion requires the healthy training class"
                )
            if (
                self.visual_result == VisualResult.NON_TB_ABNORMAL
                and self.predicted_class != ClassifierClass.SICK_NON_TB
            ):
                raise ValueError(
                    "non-TB-abnormal native-argmax fusion requires the sick_non_tb training class"
                )
        return self


class ReviewRecord(StrictModel):
    review_id: str
    case_id: str
    owner_scope: str
    trigger_reasons: list[str]
    origin: ReviewOrigin = ReviewOrigin.LEGACY_INTERACTIVE
    batch_id: str | None = Field(default=None, min_length=1, max_length=128)
    batch_item_id: str | None = Field(default=None, min_length=1, max_length=128)
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_decision: (
        Literal[
            "keep_model_flagged",
            "keep_model_not_flagged",
            "indeterminate",
            "technical_repeat_required",
        ]
        | None
    ) = None
    reviewer_note: str | None = Field(default=None, max_length=2_000)
    reviewed_by: str | None = Field(default=None, min_length=1, max_length=128)
    reviewed_at: datetime | None = None
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def batch_provenance_is_complete(self):
        batch_fields = (self.batch_id, self.batch_item_id)
        if self.origin == ReviewOrigin.BATCH_SCREENING and any(
            value is None for value in batch_fields
        ):
            raise ValueError("batch-screening reviews require batch_id and batch_item_id")
        if self.origin == ReviewOrigin.LEGACY_INTERACTIVE and any(
            value is not None for value in batch_fields
        ):
            raise ValueError("legacy interactive reviews cannot claim batch provenance")
        return self


class CaseRecord(StrictModel):
    case_id: str
    owner_scope: str
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    image_artifact_ref: str
    image_sha256: str
    image_width: int
    image_height: int
    image_source_format: Literal["PNG", "JPEG", "DICOM", "UNKNOWN"] = "UNKNOWN"
    input_transform_id: str = Field(default="legacy-unknown", min_length=1, max_length=256)
    image_quality_status: str = "unknown"
    image_quality_codes: list[str] = Field(default_factory=list, max_length=32)
    consent_scope: str
    classification_status: ClassificationExecutionStatus = (
        ClassificationExecutionStatus.NOT_REQUESTED
    )
    classification_generation_key: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    classification_attempt_count: int = Field(default=0, ge=0)
    classification_error_code: str | None = Field(default=None, min_length=1, max_length=128)
    vision_evidence: VisionEvidence | None = None
    localization_evidence: LocalizationEvidence = Field(default_factory=LocalizationEvidence)
    fusion_decision: FusionDecision | None = None
    screening_disposition: Literal[
        "screen_negative",
        "screen_positive",
        "non_tb_abnormal",
        "review_required",
        "technical_failure",
        "insufficient_evidence",
    ] = "insufficient_evidence"
    conflict_flags: list[str] = Field(default_factory=list, max_length=32)
    uncertainty_flags: list[str] = Field(default_factory=list, max_length=32)
    evidence_gaps: list[str] = Field(default_factory=list, max_length=32)
    human_review_required: bool = False
    human_review_reason_codes: list[str] = Field(default_factory=list, max_length=32)
    review_id: str | None = None
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    active_screening_session_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    record_version: int = 1

    @model_validator(mode="after")
    def classification_state_is_explicit(self):
        status = self.classification_status
        if status == ClassificationExecutionStatus.NOT_REQUESTED:
            if self.vision_evidence is not None and self.fusion_decision is not None:
                # Backward compatibility for records written before the explicit
                # classification lifecycle existed.
                object.__setattr__(
                    self,
                    "classification_status",
                    ClassificationExecutionStatus.COMPLETED,
                )
                if self.classification_attempt_count == 0:
                    object.__setattr__(self, "classification_attempt_count", 1)
                return self
            if (
                self.vision_evidence is None
                and self.fusion_decision is not None
                and self.fusion_decision.visual_result == VisualResult.TECHNICAL_FAILURE
            ):
                object.__setattr__(
                    self,
                    "classification_status",
                    ClassificationExecutionStatus.FAILED,
                )
                object.__setattr__(self, "classification_attempt_count", 1)
                object.__setattr__(
                    self,
                    "classification_error_code",
                    "legacy_classification_failure",
                )
                return self
            if self.vision_evidence is not None or self.fusion_decision is not None:
                raise ValueError("classification evidence and fusion must be stored together")
            if self.classification_attempt_count != 0 or self.classification_error_code:
                raise ValueError("not-requested classification cannot claim an attempt")
        elif status == ClassificationExecutionStatus.COMPLETED:
            if self.vision_evidence is None or self.fusion_decision is None:
                raise ValueError("completed classification requires evidence and fusion")
            if self.classification_attempt_count < 1 or self.classification_error_code:
                raise ValueError("completed classification has invalid attempt/error state")
        else:
            if self.vision_evidence is not None:
                raise ValueError("failed or unavailable classification cannot contain evidence")
            if self.classification_attempt_count < 1 or not self.classification_error_code:
                raise ValueError("failed or unavailable classification requires an error")
        return self


class UserPreferences(StrictModel):
    user_id: str
    language: Literal["zh-CN", "en"] = "zh-CN"
    report_detail: Literal["concise", "standard", "detailed"] = "standard"
    show_raw_model_scores: bool = False
    preferred_report_format: Literal["structured", "narrative"] = "structured"
    guideline_jurisdiction: list[Literal["China", "WHO", "US"]] = Field(
        default_factory=lambda: ["China", "WHO"]
    )
    updated_at: datetime = Field(default_factory=utc_now)


class ThreadMemoryEvent(StrictModel):
    """Non-reversible turn reference; free-text content is never stored here."""

    role: Literal["user", "assistant"]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=128)


class GuidelineTaskMemory(StrictModel):
    """Structured dimensions of the last successfully retrieved guideline task."""

    scope: Literal[
        "screening",
        "cad_interpretation",
        "diagnostic_testing",
        "treatment_education",
        "infection_control",
        "special_population",
    ]
    subtopic: str | None = Field(default=None, min_length=1, max_length=128)
    population: list[str] = Field(default_factory=list, max_length=16)
    product_terms: list[str] = Field(default_factory=list, max_length=16)
    scenario_tags: list[str] = Field(default_factory=list, max_length=16)


class ThreadState(StrictModel):
    thread_id: str
    user_id: str
    owner_scope: str
    current_case_id: str | None = None
    active_intent: str | None = None
    # Optional by design: thread rows written by releases before structured
    # guideline memory remain valid and simply use scope-level continuation.
    recent_guideline_task: GuidelineTaskMemory | None = None
    active_screening_session_id: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    recent_messages: list[ThreadMemoryEvent] = Field(default_factory=list)
    tool_call_counts: dict[str, int] = Field(default_factory=dict)
    pending_review_id: str | None = None
    memory_policy_id: str = "tbx-thread-memory-digest-only-v1"
    updated_at: datetime = Field(default_factory=utc_now)


class ScreeningQuestion(StrictModel):
    question_id: str = Field(min_length=1, max_length=128)
    text_zh: str = Field(min_length=1, max_length=1_000)
    answer_type: Literal["boolean", "integer", "single_choice", "multi_choice", "text"]
    choices: list[str] = Field(default_factory=list, max_length=64)
    sensitive: bool = True
    source_id: str = Field(min_length=1, max_length=128)
    locator: str = Field(min_length=1, max_length=512)
    ask_if: dict[str, Any] = Field(default_factory=dict)


class ScreeningSummary(StrictModel):
    urgency: Urgency
    triggers: list[str]
    information_gaps: list[str]
    next_steps: list[str]
    citations: list[Citation]


class ActiveScreeningSession(StrictModel):
    session_id: str
    thread_id: str
    user_id: str
    owner_scope: str
    case_id: str | None = None
    consent: bool
    guideline_rule_version: str
    status: Literal["consent_pending", "collecting", "complete", "cancelled", "needs_clarification"]
    answers: dict[str, Any] = Field(default_factory=dict)
    next_question_id: str | None = None
    result: ScreeningSummary | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AgentResponse(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    trace_id: str = Field(min_length=1, max_length=128)
    thread_id: str = Field(min_length=1, max_length=256)
    case_id: str | None = Field(default=None, min_length=1, max_length=256)
    response_kind: ResponseKind
    summary: str = Field(min_length=1, max_length=2_000)
    visual_result: VisualResult | None = None
    predicted_class: ClassifierClass | None = None
    visual_evidence_notes: list[str] = Field(default_factory=list, max_length=32)
    diagnostic_information: list[str] = Field(default_factory=list, max_length=32)
    next_step_information: list[str] = Field(default_factory=list, max_length=32)
    treatment_education: list[str] = Field(default_factory=list, max_length=32)
    limitations: list[str] = Field(default_factory=list, max_length=32)
    citations: list[Citation] = Field(default_factory=list, max_length=16)
    review_status: ReviewStatus | None = None
    next_question: ScreeningQuestion | None = None
    urgency: Urgency | None = None
    reused_existing_assessment: bool = False
    safety_policy_id: str = "tbx-agent-safety-v1"
    narrator_backend: str | None = None
    narrator_model: str | None = None
    narrator_model_digest: str | None = None
    narrator_policy_id: str | None = None
    narration_status: NarrationStatus = NarrationStatus.NOT_CONFIGURED
    narrator_generation_invoked: bool = False
    narrator_prompt_tokens: int | None = Field(default=None, ge=1)
    narrator_completion_tokens: int | None = Field(default=None, ge=1)
    source_query: str | None = Field(default=None, min_length=1, max_length=20_000)
    guideline_scope: str | None = Field(default=None, min_length=1, max_length=128)
    guideline_subtopic: str | None = Field(default=None, min_length=1, max_length=128)
    answer_status: GuidelineAnswerStatus | None = None
    retrieved_evidence: list[RetrievedGuidelineEvidence] = Field(
        default_factory=list,
        max_length=8,
    )
    claims: list[GroundedGuidelineClaim] = Field(
        default_factory=list,
        max_length=8,
    )
    evidence_gap: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def guideline_claims_are_grounded_in_this_turn(self):
        evidence_by_id = {item.chunk_id: item for item in self.retrieved_evidence}
        if len(evidence_by_id) != len(self.retrieved_evidence):
            raise ValueError("retrieved guideline evidence chunk ids must be unique")
        for claim in self.claims:
            if any(chunk_id not in evidence_by_id for chunk_id in claim.chunk_ids):
                raise ValueError("guideline claim cites evidence outside the current turn")
            # Claims are deliberately extractive. This validation prevents any
            # narrator or response composer from filling gaps with model memory.
            if not any(
                claim.text.strip() in evidence_by_id[chunk_id].text
                for chunk_id in claim.chunk_ids
            ):
                raise ValueError("guideline claim text is not present in its cited evidence")
        if self.answer_status in {
            GuidelineAnswerStatus.ANSWERED,
            GuidelineAnswerStatus.PARTIAL,
        } and not self.claims:
            raise ValueError("answered or partial guideline responses require grounded claims")
        if self.answer_status == GuidelineAnswerStatus.INSUFFICIENT_EVIDENCE:
            if self.claims:
                raise ValueError("insufficient-evidence responses cannot assert guideline claims")
            if not self.evidence_gap:
                raise ValueError("insufficient-evidence responses require an explicit gap")
        if self.answer_status in {
            GuidelineAnswerStatus.ANSWERED,
            GuidelineAnswerStatus.PARTIAL,
        } and not self.retrieved_evidence:
            raise ValueError("grounded guideline answers require retrieved evidence")
        return self
