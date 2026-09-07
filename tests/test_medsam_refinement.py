from __future__ import annotations

import hashlib

import pytest
from PIL import Image

from tbx_agent.vision.anatomy import (
    AnatomyEvidence,
    AnatomyMask,
    AnatomyQCStatus,
    LungSide,
    build_generation_key,
    decode_binary_mask,
    encode_binary_mask,
    evaluate_lung_masks,
)
from tbx_agent.vision.image_validator import ValidatedImage
from tbx_agent.vision.refinement import (
    DetectionContourEvidence,
    DetectionRefinementStatus,
    HFMedSAMBoxRefinementBackend,
    HFMedSAMConfig,
    RefinementBackendError,
    RefinementBackendUnavailable,
)


def _image() -> ValidatedImage:
    digest = hashlib.sha256(b"medsam-test-image").hexdigest()
    return ValidatedImage(
        image=Image.new("L", (30, 30), color=64),
        sha256=digest,
        width=30,
        height=30,
        source_format="PNG",
        input_transform_id="raster-exif-transpose-rgb-v1",
        quality_status="transport_valid",
        quality_warnings=(),
        artifact_bytes=b"medsam-test-image",
    )


def _anatomy(*, failed_qc: bool = False) -> AnatomyEvidence:
    left = [[False] * 30 for _ in range(30)]
    right = [[False] * 30 for _ in range(30)]
    for y in range(3, 27):
        for x in range(17, 26):
            left[y][x] = True
        for x in range(4, 13):
            right[y][x] = True
    qc = evaluate_lung_masks(left, right)
    if failed_qc:
        qc = qc.model_copy(update={"status": AnatomyQCStatus.FAIL, "codes": ["test_fail"]})
    image = _image()
    weight = hashlib.sha256(b"pspnet-test-weight").hexdigest()
    generation = build_generation_key(
        image_sha256=image.sha256,
        model_weight_sha256=weight,
        preprocessing_id="test-anatomy-preprocess",
        policy_id="paired-lung-qc-v1",
        backend_id="test-pspnet",
    )
    return AnatomyEvidence(
        run_id="test-anatomy-run",
        case_id="case-1",
        image_sha256=image.sha256,
        image_width=30,
        image_height=30,
        backend_id="test-pspnet",
        model_id="test-pspnet",
        model_weight_sha256=weight,
        model_state_dict_sha256=weight,
        preprocessing_id="test-anatomy-preprocess",
        policy_id="paired-lung-qc-v1",
        generation_key=generation,
        masks=[
            AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
            AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
        ],
        qc=qc,
        runtime_ms=0,
    )


def _config(tmp_path) -> HFMedSAMConfig:
    return HFMedSAMConfig(
        artifact_dir=tmp_path,
        expected_weight_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_preprocessor_sha256="c" * 64,
    )


class _StaticMedSAM(HFMedSAMBoxRefinementBackend):
    def _load(self) -> None:
        self._model = object()
        self._state_dict_digest = "d" * 64

    def _infer_masks(self, *, image, boxes):
        self.observed_batch_sizes = [
            *getattr(self, "observed_batch_sizes", []),
            len(boxes),
        ]
        results = []
        for _ in boxes:
            mask = [[False] * image.width for _ in range(image.height)]
            for y in range(4, 10):
                for x in range(18, 24):
                    mask[y][x] = True
            results.append(mask)
        return results


class _EmptyMedSAM(_StaticMedSAM):
    def _infer_masks(self, *, image, boxes):
        return [[[False] * image.width for _ in range(image.height)] for _ in boxes]


def test_medsam_refinement_is_lung_constrained_ordered_and_routing_neutral(tmp_path):
    backend = _StaticMedSAM(config=_config(tmp_path))
    boxes = [(17.0, 3.0, 25.0, 12.0), (13.0, 3.0, 16.0, 8.0)]
    evidence = backend.refine(
        case_id="case-1",
        image=_image(),
        boxes=boxes,
        anatomy=_anatomy(),
    )

    assert evidence.routing_effect == "none"
    assert evidence.clinical_validation is False
    assert evidence.prompt_source == "dfine_xyxy_source_image_pixels"
    assert evidence.lung_constraint == "pspnet_left_right_union"
    assert [item.detection_index for item in evidence.items] == [0, 1]
    assert evidence.items[0].status == DetectionRefinementStatus.REFINED
    assert evidence.items[1].status == DetectionRefinementStatus.OUTSIDE_LUNGS
    refined = decode_binary_mask(evidence.items[0].mask)
    assert all(not refined[y][x] for y in range(30) for x in range(17))
    assert evidence.items[0].note == "visualization_only_nonvalidated_contour"


def test_refinement_identity_changes_with_dfine_prompts(tmp_path):
    backend = _StaticMedSAM(config=_config(tmp_path))
    anatomy = _anatomy()
    first = backend.generation_key_for(
        image_sha256=_image().sha256,
        boxes=[(17.0, 3.0, 25.0, 12.0)],
        anatomy_generation_key=anatomy.generation_key,
    )
    second = backend.generation_key_for(
        image_sha256=_image().sha256,
        boxes=[(17.0, 3.0, 24.0, 12.0)],
        anatomy_generation_key=anatomy.generation_key,
    )
    assert first != second


def test_empty_or_implausible_medsam_mask_abstains_without_partial_mask(tmp_path):
    evidence = _EmptyMedSAM(config=_config(tmp_path)).refine(
        case_id="case-1",
        image=_image(),
        boxes=[(17.0, 3.0, 25.0, 12.0)],
        anatomy=_anatomy(),
    )
    item = evidence.items[0]
    assert item.status == DetectionRefinementStatus.MASK_QC_FAILED
    assert item.mask is None
    assert item.lung_constrained_pixels == 0
    assert item.note == "refinement_mask_qc_failed"


def test_failed_anatomy_qc_blocks_medsam_inference(tmp_path):
    backend = _StaticMedSAM(config=_config(tmp_path))
    with pytest.raises(RefinementBackendError, match="passed QC"):
        backend.refine(
            case_id="case-1",
            image=_image(),
            boxes=[(17.0, 3.0, 25.0, 12.0)],
            anatomy=_anatomy(failed_qc=True),
        )


def test_unverified_or_tampered_medsam_artifacts_fail_closed(tmp_path):
    backend = HFMedSAMBoxRefinementBackend(config=_config(tmp_path))
    with pytest.raises(RefinementBackendUnavailable, match="missing"):
        backend._verify_artifacts()
    (tmp_path / "pytorch_model.bin").write_bytes(b"wrong")
    (tmp_path / "config.json").write_bytes(b"wrong")
    (tmp_path / "preprocessor_config.json").write_bytes(b"wrong")
    with pytest.raises(RefinementBackendUnavailable, match="SHA-256"):
        backend._verify_artifacts()


def test_refinement_dto_cannot_claim_routing_or_validation():
    with pytest.raises(ValueError):
        DetectionContourEvidence(
            detection_index=0,
            bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
            status="outside_lungs",
            raw_mask_pixels=0,
            lung_constrained_pixels=0,
            note="detection_box_has_no_lung_overlap",
            routing_effect="changed",
        )
    with pytest.raises(ValueError):
        DetectionContourEvidence(
            detection_index=0,
            bbox_xyxy=(1.0, 1.0, 2.0, 2.0),
            status="outside_lungs",
            raw_mask_pixels=0,
            lung_constrained_pixels=0,
            note="detection_box_has_no_lung_overlap",
            clinical_validation=True,
        )


def test_three_hundred_dfine_boxes_are_covered_with_bounded_batches(tmp_path):
    backend = _StaticMedSAM(config=_config(tmp_path))
    boxes = [(17.0, 3.0, 25.0, 12.0)] * 300
    evidence = backend.refine(
        case_id="case-1",
        image=_image(),
        boxes=boxes,
        anatomy=_anatomy(),
    )

    assert len(evidence.items) == 300
    assert [item.detection_index for item in evidence.items] == list(range(300))
    assert evidence.capacity_abstained_count == 276
    assert sum(
        item.status == DetectionRefinementStatus.CAPACITY_ABSTAINED
        for item in evidence.items
    ) == 276
    assert all(size <= 4 for size in backend.observed_batch_sizes)
    assert sum(backend.observed_batch_sizes) == 24
