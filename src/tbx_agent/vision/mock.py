from __future__ import annotations

import hashlib
import time
import uuid

from ..schemas import ClassifierClass, DetectionEvidence, VisionEvidence
from .image_validator import ValidatedImage
from .rank03 import _native_classifier_argmax, _policy_decision_fields


class MockRank03Backend:
    """Deterministic demo backend; outputs are visibly labelled as synthetic."""

    backend_id = "rank03-mock-v1"
    runtime_contract = "synthetic-test-runtime-v1"
    synthetic = True

    def __init__(self, policy: dict, runtime_config: dict):
        self.policy = policy
        self.runtime_config = runtime_config
        self.call_count = 0
        self.localization_call_count = 0

    def _mock_detections(
        self, *, case_id: str, image: ValidatedImage
    ) -> list[DetectionEvidence]:
        seed = int(hashlib.sha256((image.sha256 + case_id).encode()).hexdigest()[:8], 16)
        detector_score = 0.01 + ((seed >> 11) % 9000) / 10000.0
        detections: list[DetectionEvidence] = []
        if detector_score >= float(self.runtime_config["detector"]["export_floor"]):
            detections.append(
                DetectionEvidence(
                    bbox_xyxy=(
                        image.width * 0.25,
                        image.height * 0.20,
                        image.width * 0.72,
                        image.height * 0.78,
                    ),
                    score=min(detector_score, 0.999),
                )
            )
        return detections

    def infer(self, *, case_id: str, image: ValidatedImage) -> VisionEvidence:
        started = time.perf_counter()
        self.call_count += 1
        seed = int(hashlib.sha256((image.sha256 + case_id).encode()).hexdigest()[:8], 16)
        p_tb = 0.01 + (seed % 9000) / 10000.0
        p_sick = min(0.85, 0.15 + ((seed >> 5) % 5000) / 10000.0)
        p_healthy = max(0.001, 1.0 - p_tb - p_sick)
        total = p_tb + p_sick + p_healthy
        probabilities = {
            "healthy": p_healthy / total,
            "sick_non_tb": p_sick / total,
            "tb": p_tb / total,
        }
        detections: list[DetectionEvidence] = []
        order = [item.value for item in ClassifierClass]
        predicted_class, argmax_tied = _native_classifier_argmax(probabilities, order)
        decision_fields = _policy_decision_fields(
            self.policy,
            probabilities=probabilities,
            predicted_class=predicted_class,
            argmax_tied=argmax_tied,
            detections=detections,
        )
        return VisionEvidence(
            run_id=f"mock-{uuid.uuid4()}",
            case_id=case_id,
            image_sha256=image.sha256,
            image_quality_status=image.quality_status,
            image_quality_codes=list(image.quality_warnings),
            image_source_format=image.source_format,
            input_transform_id=image.input_transform_id,
            image_width=image.width,
            image_height=image.height,
            classifier_model_id="MOCK_ONLY__convnext_tiny",
            classifier_checkpoint_sha256="mock-not-a-checkpoint",
            class_probability_order=order,
            class_probabilities=probabilities,
            classifier_decision_rule=decision_fields["classifier_decision_rule"],
            predicted_class=decision_fields["predicted_class"],
            classifier_argmax_tied=decision_fields["classifier_argmax_tied"],
            classifier_threshold=decision_fields["classifier_threshold"],
            classifier_flagged=decision_fields["classifier_flagged"],
            detector_model_id="MOCK_ONLY__dfine_l",
            detector_checkpoint_sha256="mock-not-a-checkpoint",
            detector_decision_role=decision_fields["detector_decision_role"],
            detector_threshold=decision_fields["detector_threshold"],
            detections=detections,
            detector_flagged=decision_fields["detector_flagged"],
            preprocessing_version="mock-v1-not-rank03-preprocessing",
            threshold_config_version=str(self.policy["policy_id"]),
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
            artifact_refs=["synthetic_demo_output", "detector_execution:not_requested"],
        )

    def localize(
        self, *, case_id: str, image: ValidatedImage
    ) -> list[DetectionEvidence]:
        self.localization_call_count += 1
        return self._mock_detections(case_id=case_id, image=image)
