"""Synthetic geometry for orchestration tests, never a medical accuracy benchmark."""
from __future__ import annotations

import io
import uuid

from PIL import Image

from tbx_agent.schemas import ClassifierClass, DetectionEvidence
from tbx_agent.vision.anatomy import (
    AnatomyBackendUnavailable,
    AnatomyEvidence,
    AnatomyMask,
    AnatomyRuntimeProbe,
    LungSide,
    build_generation_key,
    encode_binary_mask,
    evaluate_lung_masks,
)
from tbx_agent.vision.mock import MockRank03Backend
from tbx_agent.vision.rank03 import _policy_decision_fields


def synthetic_png(color: tuple[int, int, int] = (48, 68, 88)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class ConversationVision(MockRank03Backend):
    """One candidate in image right, overlapping the synthetic left middle lung."""

    def __init__(self, policy: dict, runtime_config: dict, *, healthy=False, empty=False):
        super().__init__(policy, runtime_config)
        self.healthy = healthy
        self.empty = empty

    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        predicted = ClassifierClass.HEALTHY if self.healthy else ClassifierClass.TB
        probabilities = {"healthy": 0.90 if self.healthy else 0.04,
                         "sick_non_tb": 0.06, "tb": 0.04 if self.healthy else 0.90}
        fields = _policy_decision_fields(
            self.policy, probabilities=probabilities, predicted_class=predicted,
            argmax_tied=False, detections=[],
        )
        return evidence.model_copy(update={
            **fields, "class_probabilities": probabilities,
            "top1_score": 0.90, "top2_score": 0.06, "top1_top2_margin": 0.84,
        })

    def localize(self, *, case_id, image):
        self.localization_call_count += 1
        return [] if self.empty else [
            DetectionEvidence(bbox_xyxy=(300, 250, 380, 310), score=0.90)
        ]


class ConversationAnatomy:
    backend_id = "synthetic-conversation-paired-lungs"
    loaded = True

    def __init__(self, *, fail=False):
        self.fail = fail
        self.call_count = 0

    def probe_runtime(self, *, load=False):
        return AnatomyRuntimeProbe(
            backend_id=self.backend_id, loaded=True, available="yes",
            detail="Synthetic geometry for software tests only",
        )

    def generation_key_for(self, image_sha256):
        return build_generation_key(
            image_sha256=image_sha256, model_weight_sha256="a" * 64,
            preprocessing_id="synthetic-source-space-v1", policy_id="paired-lung-qc-v1",
            backend_id=self.backend_id, parameters={"model_state_dict_sha256": "b" * 64},
        )

    def infer(self, *, case_id, image):
        self.call_count += 1
        if self.fail:
            raise AnatomyBackendUnavailable("Injected failure in synthetic anatomy backend")
        left = [[70 <= y < 450 and 280 <= x < 450 for x in range(512)] for y in range(512)]
        right = [[70 <= y < 450 and 60 <= x < 240 for x in range(512)] for y in range(512)]
        return AnatomyEvidence(
            run_id=f"synthetic-{uuid.uuid4()}", case_id=case_id, image_sha256=image.sha256,
            image_width=512, image_height=512, backend_id=self.backend_id,
            model_id="SYNTHETIC_PAIRED_LUNGS", model_weight_sha256="a" * 64,
            model_state_dict_sha256="b" * 64, preprocessing_id="synthetic-source-space-v1",
            policy_id="paired-lung-qc-v1", generation_key=self.generation_key_for(image.sha256),
            masks=[AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
                   AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right))],
            qc=evaluate_lung_masks(left, right), runtime_ms=2,
        )
