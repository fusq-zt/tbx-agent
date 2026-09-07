from __future__ import annotations

import hashlib
import io
import threading
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from tbx_agent.schemas import (
    MAX_DETECTION_CANDIDATES,
    ClassifierClass,
    DetectionEvidence,
    VisionEvidence,
    VisualResult,
)
from tbx_agent.vision.base import VisionBackendError
from tbx_agent.vision.fusion import fuse_rank03
from tbx_agent.vision.image_validator import ImageValidationError, validate_image
from tbx_agent.vision.mock import MockRank03Backend
from tbx_agent.vision.rank03 import Rank03Backend, _native_classifier_argmax

ACTIVE_THRESHOLD = 0.9775281548500061
ACTIVE_THRESHOLD_POLICY = {
    "policy_id": "rank03-agent-screening-demo-shenzhen-p-tb-threshold-det-advisory-v3",
    "classifier_rule": "p_tb_gte_threshold",
    "classifier_threshold": ACTIVE_THRESHOLD,
    "detector_role": "advisory_localization_only",
}
ACTIVE_NATIVE_ARGMAX_POLICY = {
    "policy_id": "rank03-user-trained-native-argmax-v2",
    "classifier_rule": "native_three_class_argmax",
    "detector_role": "advisory_localization_only",
    "quality_warning": "retain_argmax_with_advisory",
}


def _image_bytes(
    width: int = 512,
    height: int = 512,
    *,
    image_format: str = "PNG",
    low_dynamic_range: bool = False,
) -> bytes:
    if low_dynamic_range:
        image = Image.new("RGB", (width, height), color=(128, 128, 128))
    else:
        image = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    stream = io.BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


def _dicom_bytes(
    *,
    width: int = 512,
    height: int = 512,
    frames: int = 1,
    photometric: str = "MONOCHROME2",
    burned_in: str = "NO",
    compressed: bool = False,
) -> bytes:
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import ExplicitVRLittleEndian, JPEGBaseline8Bit, generate_uid

    transfer_syntax = JPEGBaseline8Bit if compressed else ExplicitVRLittleEndian
    sop_class_uid = generate_uid()
    sop_instance_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class_uid
    dataset.SOPInstanceUID = sop_instance_uid
    dataset.Modality = "DX"
    dataset.BodyPartExamined = "CHEST"
    dataset.ViewPosition = "PA"
    dataset.BurnedInAnnotation = burned_in
    dataset.PatientName = "TEST^PHI"
    dataset.PatientID = "SHOULD-NOT-PERSIST"
    dataset.Rows = height
    dataset.Columns = width
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.WindowCenter = 2048
    dataset.WindowWidth = 4096
    if frames != 1:
        dataset.NumberOfFrames = str(frames)
    row = np.linspace(0, 4095, width, dtype=np.uint16)
    pixels = np.tile(row, (frames, height, 1)) if frames != 1 else np.tile(row, (height, 1))
    if compressed:
        dataset.PixelData = encapsulate([pixels.tobytes()])
        dataset["PixelData"].is_undefined_length = True
    else:
        dataset.PixelData = pixels.tobytes()
    output = io.BytesIO()
    dataset.save_as(output, enforce_file_format=True)
    return output.getvalue()


def _evidence(
    *,
    classifier_flagged: bool,
    detector_flagged: bool,
    quality_status: str = "transport_valid",
) -> VisionEvidence:
    probabilities = (
        {"healthy": 0.1, "sick_non_tb": 0.1, "tb": 0.8}
        if classifier_flagged
        else {"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1}
    )
    detector_score = 0.8 if detector_flagged else 0.2
    return VisionEvidence(
        run_id="run-vision-test",
        case_id="case-vision-test",
        image_sha256="a" * 64,
        image_quality_status=quality_status,
        image_width=512,
        image_height=512,
        classifier_model_id="test:convnext_tiny",
        classifier_checkpoint_sha256="b" * 64,
        class_probability_order=["healthy", "sick_non_tb", "tb"],
        class_probabilities=probabilities,
        classifier_threshold=0.5,
        classifier_flagged=classifier_flagged,
        detector_model_id="test:dfine_l",
        detector_checkpoint_sha256="c" * 64,
        detector_threshold=0.5,
        detections=[
            DetectionEvidence(
                bbox_xyxy=(100.0, 120.0, 300.0, 360.0),
                score=detector_score,
                label="tb_lesion_candidate",
            )
        ],
        detector_flagged=detector_flagged,
        preprocessing_version="test-preprocessing-v1",
        threshold_config_version="test-policy-v1",
        runtime_ms=1,
    )


def _native_evidence(
    probabilities: dict[str, float],
    *,
    detector_score: float | None = 0.8,
    quality_status: str = "transport_valid",
    quality_codes: list[str] | None = None,
) -> VisionEvidence:
    order = [item.value for item in ClassifierClass]
    predicted_class, tied = _native_classifier_argmax(probabilities, order)
    detections = (
        [
            DetectionEvidence(
                bbox_xyxy=(100.0, 120.0, 300.0, 360.0),
                score=detector_score,
                label="tb_lesion_candidate",
            )
        ]
        if detector_score is not None
        else []
    )
    return VisionEvidence(
        run_id="run-native-vision-test",
        case_id="case-native-vision-test",
        image_sha256="d" * 64,
        image_quality_status=quality_status,
        image_quality_codes=quality_codes or [],
        image_width=512,
        image_height=512,
        classifier_model_id="test:convnext_tiny",
        classifier_checkpoint_sha256="b" * 64,
        class_probability_order=order,
        class_probabilities=probabilities,
        classifier_decision_rule="native_three_class_argmax",
        predicted_class=predicted_class,
        classifier_argmax_tied=tied,
        classifier_threshold=None,
        classifier_flagged=predicted_class == ClassifierClass.TB,
        detector_model_id="test:dfine_l",
        detector_checkpoint_sha256="c" * 64,
        detector_decision_role="advisory_localization_only",
        detector_threshold=None,
        detections=detections,
        detector_flagged=None,
        preprocessing_version="test-native-preprocessing-v1",
        threshold_config_version="test-native-policy-v2",
        runtime_ms=1,
    )


def _threshold_evidence(
    probabilities: dict[str, float],
    *,
    threshold: float = ACTIVE_THRESHOLD,
    detector_score: float | None = 0.8,
    quality_status: str = "transport_valid",
) -> VisionEvidence:
    order = [item.value for item in ClassifierClass]
    predicted_class, tied = _native_classifier_argmax(probabilities, order)
    detections = (
        [
            DetectionEvidence(
                bbox_xyxy=(100.0, 120.0, 300.0, 360.0),
                score=detector_score,
                label="tb_lesion_candidate",
            )
        ]
        if detector_score is not None
        else []
    )
    return VisionEvidence(
        run_id="run-threshold-vision-test",
        case_id="case-threshold-vision-test",
        image_sha256="e" * 64,
        image_quality_status=quality_status,
        image_width=512,
        image_height=512,
        classifier_model_id="test:convnext_tiny",
        classifier_checkpoint_sha256="b" * 64,
        class_probability_order=order,
        class_probabilities=probabilities,
        classifier_decision_rule="p_tb_gte_threshold",
        predicted_class=predicted_class,
        classifier_argmax_tied=tied,
        classifier_threshold=threshold,
        classifier_flagged=probabilities["tb"] >= threshold,
        detector_model_id="test:dfine_l",
        detector_checkpoint_sha256="c" * 64,
        detector_decision_role="advisory_localization_only",
        detector_threshold=None,
        detections=detections,
        detector_flagged=None,
        preprocessing_version="test-threshold-preprocessing-v1",
        threshold_config_version=(
            "rank03-agent-screening-demo-shenzhen-p-tb-threshold-det-advisory-v3"
        ),
        runtime_ms=1,
    )


def test_validate_image_accepts_exact_rank03_transport_domain() -> None:
    payload = _image_bytes()

    validated = validate_image(payload, max_bytes=len(payload))

    assert validated.sha256 == hashlib.sha256(validated.artifact_bytes).hexdigest()
    assert validated.artifact_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert validated.source_format == "PNG"
    assert validated.input_transform_id == "raster-exif-transpose-rgb-v1"
    assert validated.image.mode == "RGB"
    assert validated.image.size == (512, 512)
    assert (validated.width, validated.height) == (512, 512)
    assert validated.quality_status == "transport_valid"
    assert validated.quality_warnings == ()


def test_validate_image_collects_domain_and_quality_warnings() -> None:
    payload = _image_bytes(400, 800, low_dynamic_range=True)

    validated = validate_image(payload, max_bytes=len(payload) + 1)

    assert validated.quality_status == "warning"
    assert set(validated.quality_warnings) == {
        "resolution_below_512",
        "outside_rank03_validated_512x512_domain",
        "unusual_aspect_ratio",
        "very_low_dynamic_range",
    }


def test_rank03_evidence_preserves_quality_reason_codes(monkeypatch) -> None:
    image = validate_image(_image_bytes(300, 400), max_bytes=10_000_000)
    backend = object.__new__(Rank03Backend)
    backend._lock = threading.Semaphore(1)
    backend.policy = {
        "policy_id": "argmax-quality-codes-test",
        "classifier_rule": "native_three_class_argmax",
        "detector_role": "advisory_localization_only",
    }
    backend.config = {
        "model_bundle_id": "test-rank03",
        "classifier": {
            "checkpoint_sha256": "a" * 64,
            "probability_order": ["healthy", "sick_non_tb", "tb"],
        },
        "detector": {"checkpoint_sha256": "b" * 64},
    }
    backend._device = SimpleNamespace(type="cpu")
    monkeypatch.setattr(
        backend,
        "_classifier_infer",
        lambda _image: (
            {"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1},
            ClassifierClass.HEALTHY,
            False,
        ),
    )
    monkeypatch.setattr(backend, "_detector_infer", lambda _image: [])

    evidence = backend.infer(case_id="quality-codes-case", image=image)

    assert evidence.image_quality_status == "warning"
    assert set(evidence.image_quality_codes) == set(image.quality_warnings)
    assert evidence.image_source_format == "PNG"
    assert evidence.input_transform_id == "raster-exif-transpose-rgb-v1"


def test_vision_evidence_rejects_source_transform_provenance_mismatch() -> None:
    payload = _native_evidence(
        {"healthy": 0.8, "sick_non_tb": 0.1, "tb": 0.1}
    ).model_dump()
    payload.update(
        {
            "image_source_format": "DICOM",
            "input_transform_id": "raster-exif-transpose-rgb-v1",
        }
    )

    with pytest.raises(ValidationError, match="source format and input transform"):
        VisionEvidence.model_validate(payload)


def test_validate_image_decodes_dicom_to_metadata_free_stable_png(tmp_path) -> None:
    payload = _dicom_bytes()

    first = validate_image(payload, max_bytes=len(payload))
    second = validate_image(payload, max_bytes=len(payload))

    assert first.source_format == "DICOM"
    assert first.input_transform_id == "dicom-crdx-windowed-rgb-v1"
    assert first.image.mode == "RGB"
    assert first.image.size == (512, 512)
    assert first.quality_status == "transport_valid"
    assert first.quality_warnings == ()
    assert first.artifact_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"TEST^PHI" not in first.artifact_bytes
    assert b"SHOULD-NOT-PERSIST" not in first.artifact_bytes
    assert first.sha256 == hashlib.sha256(first.artifact_bytes).hexdigest()
    assert first.sha256 == second.sha256
    assert first.sha256 != hashlib.sha256(payload).hexdigest()

    path = first.persist_original(payload, tmp_path, "case-dicom")
    assert path.suffix == ".png"
    assert path.read_bytes() == first.artifact_bytes
    assert b"TEST^PHI" not in path.read_bytes()


def test_validate_image_dicom_monochrome1_inverts_display() -> None:
    mono2 = validate_image(_dicom_bytes(), max_bytes=2 * 1024 * 1024)
    mono1 = validate_image(
        _dicom_bytes(photometric="MONOCHROME1"),
        max_bytes=2 * 1024 * 1024,
    )

    assert mono2.image.getpixel((0, 0))[0] < mono2.image.getpixel((511, 0))[0]
    assert mono1.image.getpixel((0, 0))[0] > mono1.image.getpixel((511, 0))[0]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"frames": 2}, "仅接受单帧"),
        ({"burned_in": "YES"}, "烧录标注"),
        ({"compressed": True}, "暂不接受压缩 DICOM"),
    ],
)
def test_validate_image_rejects_unsupported_or_unsafe_dicom(options, message) -> None:
    payload = _dicom_bytes(**options)
    with pytest.raises(ImageValidationError, match=message):
        validate_image(payload, max_bytes=len(payload))


@pytest.mark.parametrize(
    ("payload", "max_bytes", "message"),
    [
        (b"", 1024, "上传文件为空"),
        (b"not-an-image", 1024, "无法安全解码"),
    ],
)
def test_validate_image_rejects_invalid_transport(
    payload: bytes, max_bytes: int, message: str
) -> None:
    with pytest.raises(ImageValidationError, match=message):
        validate_image(payload, max_bytes=max_bytes)


def test_validate_image_rejects_oversize_and_unsupported_decoded_formats() -> None:
    png = _image_bytes(64, 64)
    with pytest.raises(ImageValidationError, match="文件超过"):
        validate_image(png, max_bytes=len(png) - 1)

    bmp = _image_bytes(64, 64, image_format="BMP")
    with pytest.raises(ImageValidationError, match="仅接受 PNG/JPEG"):
        validate_image(bmp, max_bytes=len(bmp))


def test_probability_schema_rejects_non_normalized_mock_contract() -> None:
    invalid = _evidence(classifier_flagged=False, detector_flagged=False).model_dump()
    invalid["class_probabilities"] = {"healthy": 0.8, "sick_non_tb": 0.3, "tb": 0.1}

    with pytest.raises(ValidationError, match="sum to one"):
        VisionEvidence.model_validate(invalid)


def test_vision_evidence_enforces_dfine_three_hundred_candidate_contract() -> None:
    payload = _native_evidence(
        {"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05}
    ).model_dump()
    candidate = DetectionEvidence(bbox_xyxy=(1.0, 1.0, 2.0, 2.0), score=0.5)
    payload["detections"] = [candidate.model_dump()] * MAX_DETECTION_CANDIDATES

    evidence = VisionEvidence.model_validate(payload)
    assert len(evidence.detections) == 300

    payload["detections"].append(candidate.model_dump())
    with pytest.raises(ValidationError, match="at most 300"):
        VisionEvidence.model_validate(payload)


def test_mock_backend_is_deterministic_and_visibly_synthetic() -> None:
    payload = _image_bytes()
    image = validate_image(payload, max_bytes=len(payload))
    policy = {
        "policy_id": "rank03-agent-screening-demo-cls-argmax-det-advisory-v2",
        "classifier_rule": "native_three_class_argmax",
        "detector_role": "advisory_localization_only",
    }
    runtime_config = {"detector": {"export_floor": 0.05}}
    backend = MockRank03Backend(policy, runtime_config)

    first = backend.infer(case_id="case-mock", image=image)
    second = backend.infer(case_id="case-mock", image=image)

    assert backend.call_count == 2
    assert first.run_id != second.run_id
    assert first.run_id.startswith("mock-")
    assert first.class_probabilities == second.class_probabilities
    assert first.detections == second.detections
    assert first.classifier_model_id.startswith("MOCK_ONLY__")
    assert first.detector_model_id.startswith("MOCK_ONLY__")
    assert first.artifact_refs == [
        "synthetic_demo_output",
        "detector_execution:not_requested",
    ]
    assert first.detections == []
    assert backend.localization_call_count == 0
    assert backend.localize(case_id="case-mock", image=image) == backend.localize(
        case_id="case-mock", image=image
    )
    assert backend.localization_call_count == 2
    assert sum(first.class_probabilities.values()) == pytest.approx(1.0)
    assert first.classifier_decision_rule == "native_three_class_argmax"
    assert first.classifier_threshold is None
    assert first.classifier_flagged is (first.predicted_class == ClassifierClass.TB)
    assert first.detector_decision_role == "advisory_localization_only"
    assert first.image_source_format == "PNG"
    assert first.input_transform_id == "raster-exif-transpose-rgb-v1"
    assert first.detector_threshold is None
    assert first.detector_flagged is None


def test_mock_backend_supports_active_threshold_with_advisory_detector() -> None:
    payload = _image_bytes()
    image = validate_image(payload, max_bytes=len(payload))
    backend = MockRank03Backend(
        ACTIVE_THRESHOLD_POLICY,
        {"detector": {"export_floor": 0.05}},
    )

    evidence = backend.infer(case_id="active-threshold-case", image=image)

    assert evidence.classifier_decision_rule == "p_tb_gte_threshold"
    assert evidence.classifier_threshold == pytest.approx(ACTIVE_THRESHOLD)
    assert evidence.classifier_flagged is (evidence.class_probabilities["tb"] >= ACTIVE_THRESHOLD)
    assert evidence.detector_decision_role == "advisory_localization_only"
    assert evidence.detector_threshold is None
    assert evidence.detector_flagged is None


def test_active_threshold_comparison_is_inclusive_and_detector_is_advisory() -> None:
    at_cutoff = {
        "healthy": 1.0 - ACTIVE_THRESHOLD,
        "sick_non_tb": 0.0,
        "tb": ACTIVE_THRESHOLD,
    }
    evidence = _threshold_evidence(at_cutoff, detector_score=None)

    decision = fuse_rank03(evidence, ACTIVE_THRESHOLD_POLICY)

    assert evidence.classifier_flagged is True
    assert decision.visual_result is VisualResult.MODEL_FLAGGED
    assert decision.detector_flagged is None

    below_cutoff = {"healthy": 0.06, "sick_non_tb": 0.04, "tb": 0.9}
    without_box = fuse_rank03(
        _threshold_evidence(below_cutoff, detector_score=None),
        ACTIVE_THRESHOLD_POLICY,
    )
    with_high_box = fuse_rank03(
        _threshold_evidence(below_cutoff, detector_score=0.99),
        ACTIVE_THRESHOLD_POLICY,
    )
    assert without_box.predicted_class is ClassifierClass.TB
    assert without_box.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert with_high_box.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert with_high_box.max_detector_score == pytest.approx(0.99)
    assert "rank03_branch_disagreement" not in with_high_box.review_reasons


def test_active_threshold_evidence_and_policy_fail_closed_on_contract_drift() -> None:
    probabilities = {"healthy": 0.06, "sick_non_tb": 0.04, "tb": 0.9}
    payload = _threshold_evidence(probabilities).model_dump()
    payload["classifier_flagged"] = True
    with pytest.raises(ValidationError, match="inconsistent with its p_tb threshold"):
        VisionEvidence.model_validate(payload)

    evidence = _threshold_evidence(probabilities)
    drifted_policy = {**ACTIVE_THRESHOLD_POLICY, "classifier_threshold": 0.9}
    decision = fuse_rank03(evidence, drifted_policy)
    assert decision.visual_result is VisualResult.TECHNICAL_FAILURE
    assert decision.review_reasons == ["fusion_policy_evidence_mismatch"]


def test_quality_warning_overrides_active_threshold_route() -> None:
    evidence = _threshold_evidence(
        {"healthy": 0.005, "sick_non_tb": 0.005, "tb": 0.99},
        quality_status="warning",
    )

    decision = fuse_rank03(evidence, ACTIVE_THRESHOLD_POLICY)

    assert evidence.classifier_flagged is True
    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_reasons == ["image_quality_warning"]


def test_active_threshold_policy_rejects_detector_decision_fields() -> None:
    payload = _image_bytes()
    image = validate_image(payload, max_bytes=len(payload))
    policy = {
        **ACTIVE_THRESHOLD_POLICY,
        "detector_rule": "max_category_agnostic_score_gte_threshold",
        "detector_threshold": 0.2,
    }
    backend = MockRank03Backend(policy, {"detector": {"export_floor": 0.05}})

    with pytest.raises(VisionBackendError, match="advisory detector policy cannot contain"):
        backend.infer(case_id="invalid-active-threshold", image=image)


def test_mock_backend_reproduces_explicit_legacy_threshold_policy() -> None:
    payload = _image_bytes()
    image = validate_image(payload, max_bytes=len(payload))
    policy = {
        "policy_id": "legacy-sens98-v1",
        "classifier_rule": "p_tb_gte_threshold",
        "classifier_threshold": 0.4,
        "detector_rule": "max_category_agnostic_score_gte_threshold",
        "detector_threshold": 0.2,
    }
    backend = MockRank03Backend(policy, {"detector": {"export_floor": 0.05}})

    evidence = backend.infer(case_id="legacy-case", image=image)

    assert evidence.classifier_decision_rule == "legacy_p_tb_threshold"
    assert evidence.predicted_class is None
    assert evidence.classifier_threshold == pytest.approx(0.4)
    assert evidence.classifier_flagged is (evidence.class_probabilities["tb"] >= 0.4)
    assert evidence.detector_decision_role == "legacy_vote"
    assert evidence.detector_threshold == pytest.approx(0.2)
    max_score = max((item.score for item in evidence.detections), default=0.0)
    assert evidence.detector_flagged is (max_score >= 0.2)


def test_mock_backend_rejects_mixed_native_and_legacy_policy() -> None:
    payload = _image_bytes()
    image = validate_image(payload, max_bytes=len(payload))
    backend = MockRank03Backend(
        {
            "policy_id": "invalid-mixed-policy",
            "classifier_rule": "native_three_class_argmax",
            "detector_role": "legacy_vote",
        },
        {"detector": {"export_floor": 0.05}},
    )

    with pytest.raises(VisionBackendError, match="requires detector_role"):
        backend.infer(case_id="mixed-case", image=image)


def test_fusion_routes_branch_disagreement_to_human_review() -> None:
    evidence = _evidence(classifier_flagged=True, detector_flagged=False)

    decision = fuse_rank03(evidence, {"policy_id": "fusion-test-v1"})

    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_required is True
    assert decision.review_reasons == ["rank03_branch_disagreement"]
    assert decision.max_detector_score == pytest.approx(0.2)
    assert decision.clinical_validation is False


def test_fusion_quality_warning_overrides_branch_agreement() -> None:
    evidence = _evidence(
        classifier_flagged=False,
        detector_flagged=False,
        quality_status="warning",
    )

    decision = fuse_rank03(evidence, {"policy_id": "fusion-test-v1"})

    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_required is True
    assert decision.review_reasons == ["image_quality_warning"]


def test_fusion_preserves_both_disagreement_and_quality_reasons() -> None:
    evidence = _evidence(
        classifier_flagged=False,
        detector_flagged=True,
        quality_status="warning",
    )

    decision = fuse_rank03(evidence, {"policy_id": "fusion-test-v1"})

    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_reasons == [
        "rank03_branch_disagreement",
        "image_quality_warning",
    ]


@pytest.mark.parametrize(
    ("probabilities", "expected_class", "expected_result", "expected_reason"),
    [
        (
            {"healthy": 0.8, "sick_non_tb": 0.16, "tb": 0.04},
            ClassifierClass.HEALTHY,
            VisualResult.MODEL_NOT_FLAGGED,
            None,
        ),
        (
            {"healthy": 0.1, "sick_non_tb": 0.8, "tb": 0.1},
            ClassifierClass.SICK_NON_TB,
            VisualResult.NON_TB_ABNORMAL,
            None,
        ),
        (
            {"healthy": 0.1, "sick_non_tb": 0.2, "tb": 0.7},
            ClassifierClass.TB,
            VisualResult.MODEL_FLAGGED,
            None,
        ),
    ],
)
def test_native_argmax_routes_three_classes_without_classifier_threshold(
    probabilities: dict[str, float],
    expected_class: ClassifierClass,
    expected_result: VisualResult,
    expected_reason: str | None,
) -> None:
    evidence = _native_evidence(probabilities)

    decision = fuse_rank03(
        evidence,
        {"policy_id": "rank03-agent-screening-demo-cls-argmax-det-advisory-v2"},
    )

    assert evidence.predicted_class is expected_class
    assert evidence.classifier_threshold is None
    assert decision.predicted_class is expected_class
    assert decision.visual_result is expected_result
    assert decision.review_required is (expected_result == VisualResult.PENDING_HUMAN_REVIEW)
    assert decision.review_reasons == ([expected_reason] if expected_reason else [])


def test_native_exact_argmax_tie_uses_stable_class_order_without_review() -> None:
    evidence = _native_evidence({"healthy": 0.45, "sick_non_tb": 0.45, "tb": 0.1})

    decision = fuse_rank03(evidence, {"policy_id": "argmax-v2"})

    assert evidence.predicted_class is ClassifierClass.HEALTHY
    assert evidence.classifier_argmax_tied is True
    assert evidence.classifier_flagged is False
    assert decision.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert decision.review_required is False
    assert decision.review_reasons == []


def test_advisory_detector_score_does_not_change_native_argmax_fusion() -> None:
    probabilities = {"healthy": 0.7, "sick_non_tb": 0.2, "tb": 0.1}
    no_candidate = fuse_rank03(
        _native_evidence(probabilities, detector_score=None), {"policy_id": "argmax-v2"}
    )
    high_score = fuse_rank03(
        _native_evidence(probabilities, detector_score=0.99), {"policy_id": "argmax-v2"}
    )

    assert no_candidate.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert high_score.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert no_candidate.detector_flagged is None
    assert high_score.detector_flagged is None
    assert high_score.max_detector_score == pytest.approx(0.99)
    assert "rank03_branch_disagreement" not in high_score.review_reasons


def test_historical_quality_warning_still_overrides_native_healthy_argmax() -> None:
    evidence = _native_evidence(
        {"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05},
        quality_status="warning",
    )

    decision = fuse_rank03(evidence, {"policy_id": "argmax-v2"})

    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_reasons == ["image_quality_warning"]


@pytest.mark.parametrize(
    ("probabilities", "expected_result"),
    [
        (
            {"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05},
            VisualResult.MODEL_NOT_FLAGGED,
        ),
        (
            {"healthy": 0.05, "sick_non_tb": 0.8, "tb": 0.15},
            VisualResult.NON_TB_ABNORMAL,
        ),
        (
            {"healthy": 0.05, "sick_non_tb": 0.15, "tb": 0.8},
            VisualResult.MODEL_FLAGGED,
        ),
    ],
)
def test_active_native_argmax_retains_route_under_quality_warning(
    probabilities: dict[str, float],
    expected_result: VisualResult,
) -> None:
    evidence = _native_evidence(
        probabilities,
        quality_status="warning",
        quality_codes=["resolution_outside_validated_512"],
    )

    decision = fuse_rank03(evidence, ACTIVE_NATIVE_ARGMAX_POLICY)

    assert decision.visual_result is expected_result
    assert decision.review_required is False
    assert decision.review_reasons == []
    assert evidence.image_quality_status == "warning"
    assert evidence.image_quality_codes == ["resolution_outside_validated_512"]


def test_active_native_argmax_warning_and_tie_retain_stable_argmax() -> None:
    evidence = _native_evidence(
        {"healthy": 0.45, "sick_non_tb": 0.45, "tb": 0.1},
        quality_status="warning",
        quality_codes=["resolution_outside_validated_512"],
    )

    decision = fuse_rank03(evidence, ACTIVE_NATIVE_ARGMAX_POLICY)

    assert decision.visual_result is VisualResult.MODEL_NOT_FLAGGED
    assert decision.predicted_class is ClassifierClass.HEALTHY
    assert decision.review_required is False
    assert decision.review_reasons == []


def test_active_native_argmax_technical_failure_still_abstains() -> None:
    evidence = _native_evidence(
        {"healthy": 0.05, "sick_non_tb": 0.15, "tb": 0.8},
        quality_status="technical_failure",
    )

    decision = fuse_rank03(evidence, ACTIVE_NATIVE_ARGMAX_POLICY)

    assert decision.visual_result is VisualResult.TECHNICAL_FAILURE
    assert decision.review_required is False
    assert decision.review_reasons == ["technical_failure"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"classifier_threshold": 0.027}, "must not contain a classifier threshold"),
        ({"predicted_class": "tb"}, "inconsistent with argmax"),
        ({"classifier_flagged": True}, "flag must be derived"),
        ({"detector_threshold": 0.07}, "cannot contain a decision threshold"),
        ({"detector_flagged": False}, "cannot contain a decision threshold"),
        ({"detector_decision_role": "legacy_vote"}, "must be paired"),
        (
            {"class_probability_order": ["tb", "healthy", "sick_non_tb"]},
            "probability order",
        ),
    ],
)
def test_native_evidence_schema_rejects_inconsistent_decision_contract(
    mutation: dict[str, object], message: str
) -> None:
    payload = _native_evidence({"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05}).model_dump()
    payload.update(mutation)

    with pytest.raises(ValidationError, match=message):
        VisionEvidence.model_validate(payload)


def test_legacy_evidence_payload_defaults_remain_loadable_and_fusable() -> None:
    payload = _evidence(classifier_flagged=True, detector_flagged=False).model_dump()
    for field in (
        "classifier_decision_rule",
        "predicted_class",
        "classifier_argmax_tied",
        "detector_decision_role",
    ):
        payload.pop(field)

    evidence = VisionEvidence.model_validate(payload)
    decision = fuse_rank03(evidence, {"policy_id": "legacy-policy-v1"})

    assert evidence.classifier_decision_rule == "legacy_p_tb_threshold"
    assert evidence.predicted_class is None
    assert evidence.detector_decision_role == "legacy_vote"
    assert decision.visual_result is VisualResult.PENDING_HUMAN_REVIEW
    assert decision.review_reasons == ["rank03_branch_disagreement"]


def test_native_argmax_helper_rejects_probability_order_drift() -> None:
    probabilities = {"healthy": 0.7, "sick_non_tb": 0.2, "tb": 0.1}

    with pytest.raises(VisionBackendError, match="类别顺序"):
        _native_classifier_argmax(probabilities, ["tb", "sick_non_tb", "healthy"])


def test_fusion_fails_closed_when_active_policy_receives_legacy_evidence() -> None:
    evidence = _evidence(classifier_flagged=False, detector_flagged=False)

    decision = fuse_rank03(
        evidence,
        {
            "policy_id": "argmax-active-v2",
            "classifier_rule": "native_three_class_argmax",
            "detector_role": "advisory_localization_only",
        },
    )

    assert decision.visual_result is VisualResult.TECHNICAL_FAILURE
    assert decision.review_reasons == ["fusion_policy_evidence_mismatch"]


def test_probability_schema_rejects_non_finite_values() -> None:
    payload = _native_evidence({"healthy": 0.8, "sick_non_tb": 0.15, "tb": 0.05}).model_dump()
    payload["class_probabilities"] = {
        "healthy": float("nan"),
        "sick_non_tb": 0.15,
        "tb": 0.05,
    }

    with pytest.raises(ValidationError, match="finite"):
        VisionEvidence.model_validate(payload)
