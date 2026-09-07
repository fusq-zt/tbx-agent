from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import tbx_agent.vision.anatomy.localization as localization_module
from tbx_agent.anatomy_runs import build_anatomy_pipeline_generation_key
from tbx_agent.vision.anatomy import (
    AnatomyBackendUnavailable,
    AnatomyEvidence,
    AnatomyMask,
    AnatomyQCPolicy,
    AnatomyQCStatus,
    LungFieldZone,
    LungSide,
    TorchXRayVisionPSPNetBackend,
    XRVPSPNetConfig,
    build_chat_spatial_summary,
    build_generation_key,
    build_spatial_summary,
    decode_binary_mask,
    encode_binary_mask,
    evaluate_lung_masks,
    localize_bbox_to_lung_fields,
    localize_detection_boxes,
)
from tbx_agent.vision.anatomy.xrv_pspnet import _restore_mask_to_source


def _paired_masks(width: int = 30, height: int = 30):
    left = [[False] * width for _ in range(height)]
    right = [[False] * width for _ in range(height)]
    for y in range(3, 27):
        for x in range(17, 26):
            left[y][x] = True
        for x in range(4, 13):
            right[y][x] = True
    return left, right


def _evidence(*, qc_status: AnatomyQCStatus = AnatomyQCStatus.PASS):
    left, right = _paired_masks()
    qc = evaluate_lung_masks(left, right)
    if qc_status != qc.status:
        qc = qc.model_copy(update={"status": qc_status, "codes": ["forced_test_status"]})
    image_hash = hashlib.sha256(b"synthetic-image").hexdigest()
    weight_hash = hashlib.sha256(b"synthetic-weight").hexdigest()
    key = build_generation_key(
        image_sha256=image_hash,
        model_weight_sha256=weight_hash,
        preprocessing_id="synthetic-v1",
        policy_id="paired-lung-qc-v1",
        backend_id="synthetic",
    )
    return AnatomyEvidence(
        run_id="synthetic-run",
        case_id="synthetic-case",
        image_sha256=image_hash,
        image_width=30,
        image_height=30,
        backend_id="synthetic",
        model_id="synthetic-model",
        model_weight_sha256=weight_hash,
        model_state_dict_sha256=weight_hash,
        preprocessing_id="synthetic-v1",
        policy_id="paired-lung-qc-v1",
        generation_key=key,
        masks=[
            AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
            AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
        ],
        qc=qc,
        runtime_ms=0,
    )


def test_compact_rle_round_trip_and_public_dto_contains_no_path():
    left, _ = _paired_masks()
    encoded = encode_binary_mask(left)
    assert decode_binary_mask(encoded) == left
    dumped = json.dumps(encoded.model_dump(mode="json"))
    assert "path" not in dumped.lower()
    assert len(encoded.counts_b64) < 200


def test_rle_tamper_is_rejected():
    left, _ = _paired_masks()
    encoded = encode_binary_mask(left)
    tampered = encoded.model_copy(update={"foreground_pixels": encoded.foreground_pixels - 1})
    with pytest.raises(ValueError, match="foreground count"):
        decode_binary_mask(tampered)


def test_qc_accepts_plausible_pair_and_reports_source_space_metrics():
    left, right = _paired_masks()
    report = evaluate_lung_masks(left, right, source_width=30, source_height=30)
    assert report.status == AnatomyQCStatus.PASS
    assert report.codes == []
    assert report.metrics["left_right_area_ratio"] == pytest.approx(1.0)


def test_qc_fails_empty_and_implausibly_imbalanced_lungs():
    left, right = _paired_masks()
    empty = [[False] * 30 for _ in range(30)]
    empty_report = evaluate_lung_masks(empty, right)
    assert empty_report.status == AnatomyQCStatus.FAIL
    assert "left_lung_empty" in empty_report.codes

    for y in range(3, 27):
        for x in range(18, 25):
            left[y][x] = False
    imbalance = evaluate_lung_masks(left, right)
    assert imbalance.status == AnatomyQCStatus.FAIL
    assert "left_right_area_ratio_out_of_range" in imbalance.codes


def test_qc_warns_on_crop_and_boundary_contact():
    left, right = _paired_masks()
    for y in range(3, 27):
        left[y][29] = True
    policy = AnatomyQCPolicy(max_border_fraction=0.0)
    report = evaluate_lung_masks(
        left,
        right,
        policy=policy,
        preprocessing_crop_fraction=0.2,
    )
    assert report.status == AnatomyQCStatus.WARNING
    assert "lung_mask_touches_image_border" in report.codes
    assert "preprocessing_crop_fraction_high" in report.codes


def test_qc_reports_mask_canvas_outside_source_coordinates():
    left, right = _paired_masks()
    report = evaluate_lung_masks(left, right, source_width=20, source_height=20)
    assert report.status == AnatomyQCStatus.FAIL
    assert report.codes == ["mask_canvas_out_of_source_bounds"]


def test_source_geometry_restore_does_not_stretch_center_crop():
    small = [[True, False], [False, True]]
    restored = _restore_mask_to_source(small, source_width=6, source_height=4)
    assert len(restored) == 4
    assert all(len(row) == 6 for row in restored)
    assert all(not row[0] and not row[5] for row in restored)


@pytest.mark.parametrize(
    ("bbox", "side", "zone"),
    [
        ((18.0, 4.0, 24.0, 9.0), LungSide.LEFT, LungFieldZone.UPPER),
        ((5.0, 12.0, 11.0, 18.0), LungSide.RIGHT, LungFieldZone.MIDDLE),
        ((18.0, 21.0, 24.0, 26.0), LungSide.LEFT, LungFieldZone.LOWER),
    ],
)
def test_detector_box_maps_to_side_and_two_dimensional_lung_field(bbox, side, zone):
    result = localize_bbox_to_lung_fields(bbox, anatomy=_evidence())
    assert result.status == "localized"
    assert len(result.assignments) == 1
    assert result.assignments[0].lung == side
    assert result.assignments[0].primary_zone == zone
    assert result.note == "two_dimensional_lung_field_not_lobe"


def test_box_can_map_bilaterally_and_outside_box_stays_unlocalized():
    bilateral = localize_bbox_to_lung_fields((10.0, 10.0, 20.0, 20.0), anatomy=_evidence())
    assert bilateral.status == "localized"
    assert {item.lung for item in bilateral.assignments} == {LungSide.LEFT, LungSide.RIGHT}

    outside = localize_bbox_to_lung_fields((13.0, 3.0, 17.0, 8.0), anatomy=_evidence())
    assert outside.status == "outside_lungs"


def test_failed_anatomy_qc_blocks_spatial_claims():
    result = localize_bbox_to_lung_fields(
        (18.0, 4.0, 24.0, 9.0),
        anatomy=_evidence(qc_status=AnatomyQCStatus.FAIL),
    )
    assert result.status == "invalid_anatomy"
    assert result.assignments == []


def test_three_hundred_boxes_are_deterministic_and_decode_each_lung_once(monkeypatch):
    anatomy = _evidence()
    templates = [
        (18.0, 4.0, 24.0, 9.0),
        (5.0, 12.0, 11.0, 18.0),
        (10.0, 10.0, 20.0, 20.0),
        (13.0, 3.0, 17.0, 8.0),
    ]
    boxes = [templates[index % len(templates)] for index in range(300)]
    decode_calls = 0
    real_decode = localization_module.decode_binary_mask

    def counted_decode(payload):
        nonlocal decode_calls
        decode_calls += 1
        return real_decode(payload)

    monkeypatch.setattr(localization_module, "decode_binary_mask", counted_decode)

    first = localize_detection_boxes(boxes, anatomy=anatomy)
    assert decode_calls == 2
    second = localize_detection_boxes(boxes, anatomy=anatomy)
    assert decode_calls == 4

    assert first == second
    assert len(first) == 300
    assert [item.bbox_xyxy for item in first] == boxes
    assert [item.status for item in first[:4]] == [
        "localized",
        "localized",
        "localized",
        "outside_lungs",
    ]
    monkeypatch.setattr(localization_module, "decode_binary_mask", real_decode)
    assert first[:4] == [
        localize_bbox_to_lung_fields(box, anatomy=anatomy) for box in templates
    ]


def test_localization_batch_rejects_more_than_dfine_query_contract(monkeypatch):
    decode_calls = 0

    def counted_decode(_payload):
        nonlocal decode_calls
        decode_calls += 1
        raise AssertionError("over-limit input must be rejected before mask decoding")

    monkeypatch.setattr(localization_module, "decode_binary_mask", counted_decode)
    boxes = [(18.0, 4.0, 24.0, 9.0)] * 301

    with pytest.raises(ValueError, match="D-FINE limit of 300"):
        localize_detection_boxes(boxes, anatomy=_evidence())
    assert decode_calls == 0


def test_localization_preserves_empty_batch_and_invalid_geometry_fail_fast(monkeypatch):
    def unexpected_decode(_payload):
        raise AssertionError("this path must not decode an anatomy mask")

    monkeypatch.setattr(localization_module, "decode_binary_mask", unexpected_decode)
    invalid_policy = localization_module.LungFieldLocalizationPolicy(
        minimum_box_overlap_fraction=2.0
    )

    assert localize_detection_boxes([], anatomy=_evidence(), policy=invalid_policy) == []
    with pytest.raises(ValueError, match="bbox must have positive area"):
        localize_bbox_to_lung_fields((2.0, 2.0, 1.0, 3.0), anatomy=_evidence())
    with pytest.raises(ValueError, match="bbox must have positive area"):
        localize_detection_boxes(
            [(2.0, 2.0, 1.0, 3.0)],
            anatomy=_evidence(),
        )


def test_spatial_summary_is_deterministic_non_diagnostic_and_complete():
    anatomy = _evidence()
    locations = [
        localize_bbox_to_lung_fields((18.0, 4.0, 24.0, 9.0), anatomy=anatomy),
        localize_bbox_to_lung_fields((13.0, 3.0, 17.0, 8.0), anatomy=anatomy),
    ]

    summary = build_spatial_summary(locations, anatomy_qc_status=anatomy.qc.status)

    assert summary.candidate_count == 2
    assert summary.localized_count == 1
    assert summary.outside_lungs_count == 1
    assert summary.invalid_anatomy_count == 0
    assert summary.routing_effect == "none"
    assert summary.clinical_validation is False
    assert "左侧肺野" in summary.statements[0]
    assert "上肺野" in summary.statements[0]
    assert "二维" in summary.statements[0]
    assert "不代表肺叶或病灶诊断" in summary.statements[0]
    assert "不等同于肺外异常或病变判断" in summary.statements[1]
    assert summary.statements[-1] == "肺野分割和候选定位不改变胸片三分类结果。"


def test_chat_spatial_summary_groups_duplicate_candidates_without_overlap_metrics():
    anatomy = _evidence()
    locations = [
        localize_bbox_to_lung_fields((5.0, 4.0, 11.0, 9.0), anatomy=anatomy),
        localize_bbox_to_lung_fields((18.0, 4.0, 24.0, 9.0), anatomy=anatomy),
        localize_bbox_to_lung_fields((18.0, 4.0, 24.0, 9.0), anatomy=anatomy),
    ]

    summary = build_chat_spatial_summary(
        locations,
        anatomy_qc_status=anatomy.qc.status,
    )

    assert summary == "候选区域位于右上肺野和左上肺野（2 个）。"
    assert "交叠" not in summary
    assert "二维投影" not in summary
    assert "不代表肺叶" not in summary

    # The detailed audit representation remains untouched.
    detailed = build_spatial_summary(locations, anatomy_qc_status=anatomy.qc.status)
    assert "框内肺野掩膜交叠" in detailed.statements[0]
    assert "占该侧肺野" in detailed.statements[0]


def test_spatial_summary_abstains_when_anatomy_qc_fails():
    anatomy = _evidence(qc_status=AnatomyQCStatus.FAIL)
    location = localize_bbox_to_lung_fields(
        (18.0, 4.0, 24.0, 9.0),
        anatomy=anatomy,
    )

    summary = build_spatial_summary([location], anatomy_qc_status=anatomy.qc.status)

    assert summary.invalid_anatomy_count == 1
    assert all("左侧肺野" not in statement for statement in summary.statements)
    assert "停止空间定位" in summary.statements[0]


def test_generation_key_covers_image_weight_preprocessing_policy_and_parameters():
    base = {
        "image_sha256": "1" * 64,
        "model_weight_sha256": "2" * 64,
        "preprocessing_id": "pre-v1",
        "policy_id": "qc-v1",
        "backend_id": "backend-v1",
        "parameters": {"threshold": 0.5},
    }
    original = build_generation_key(**base)
    for name, replacement in (
        ("image_sha256", "3" * 64),
        ("model_weight_sha256", "4" * 64),
        ("preprocessing_id", "pre-v2"),
        ("policy_id", "qc-v2"),
        ("backend_id", "backend-v2"),
        ("parameters", {"threshold": 0.6}),
    ):
        changed = dict(base)
        changed[name] = replacement
        assert build_generation_key(**changed) != original


def test_pipeline_generation_key_covers_detector_and_spatial_policy_identity():
    base = {
        "anatomy_generation_key": "1" * 64,
        "detector_run_id": "detector-run-v1",
        "detector_checkpoint_sha256": "2" * 64,
        "detector_boxes": [(1.0, 2.0, 10.0, 20.0)],
        "localization_policy_id": "localization-v1",
        "localization_minimum_box_overlap_fraction": 0.01,
        "presentation_policy_id": "presentation-v1",
        "refinement_generation_key": None,
    }
    original = build_anatomy_pipeline_generation_key(**base)
    replacements = {
        "anatomy_generation_key": "3" * 64,
        "detector_run_id": "detector-run-v2",
        "detector_checkpoint_sha256": "4" * 64,
        "detector_boxes": [(2.0, 2.0, 10.0, 20.0)],
        "localization_policy_id": "localization-v2",
        "localization_minimum_box_overlap_fraction": 0.02,
        "presentation_policy_id": "presentation-v2",
        "refinement_generation_key": "5" * 64,
    }
    for field, replacement in replacements.items():
        changed = {**base, field: replacement}
        assert build_anatomy_pipeline_generation_key(**changed) != original


def test_pipeline_generation_key_rejects_invalid_identity():
    with pytest.raises(ValueError, match="complete SHA-256"):
        build_anatomy_pipeline_generation_key(
            anatomy_generation_key="not-a-digest",
            detector_run_id=None,
            detector_checkpoint_sha256=None,
            detector_boxes=[],
            localization_policy_id="localization-v1",
            localization_minimum_box_overlap_fraction=0.01,
            presentation_policy_id="presentation-v1",
        )


def test_xrv_backend_has_explainable_failure_when_optional_dependency_missing(monkeypatch):
    backend = TorchXRayVisionPSPNetBackend()
    real_import = __import__("importlib").import_module

    def fake_import(name: str):
        if name == "torchxrayvision":
            raise ImportError("not installed")
        return real_import(name)

    monkeypatch.setattr("tbx_agent.vision.anatomy.xrv_pspnet.importlib.import_module", fake_import)
    with pytest.raises(AnatomyBackendUnavailable, match="optional"):
        backend._load()


def test_xrv_runtime_probe_does_not_load_weights_by_default(monkeypatch):
    backend = TorchXRayVisionPSPNetBackend()
    monkeypatch.setattr(
        "tbx_agent.vision.anatomy.xrv_pspnet.importlib.util.find_spec",
        lambda name: None,
    )
    probe = backend.probe_runtime()
    assert probe.available == "no"
    assert probe.loaded is False
    assert backend.loaded is False


def test_xrv_rejects_wrong_checkpoint_file_hash_before_deserialization(tmp_path, monkeypatch):
    weight_path = tmp_path / "pspnet_chestxray_best_model_4.pth"
    weight_path.write_bytes(b"not-the-pinned-weight")
    constructor_called = False

    def constructor(**kwargs):
        nonlocal constructor_called
        constructor_called = True
        raise AssertionError("an unverified checkpoint must never be deserialized")

    fake_xrv = SimpleNamespace(
        baseline_models=SimpleNamespace(
            chestx_det=SimpleNamespace(PSPNet=constructor),
        )
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    def fake_import(name: str):
        if name == "torch":
            return fake_torch
        if name == "torchxrayvision":
            return fake_xrv
        raise ImportError(name)

    monkeypatch.setattr("tbx_agent.vision.anatomy.xrv_pspnet.importlib.import_module", fake_import)
    backend = TorchXRayVisionPSPNetBackend(
        config=XRVPSPNetConfig(
            cache_dir=tmp_path,
            expected_weight_file_sha256="0" * 64,
        )
    )
    with pytest.raises(AnatomyBackendUnavailable, match="SHA-256"):
        backend._load()
    assert constructor_called is False


def test_public_anatomy_evidence_cannot_claim_a_routing_effect():
    payload = _evidence().model_dump(mode="json")
    payload["routing_effect"] = "changes_argmax"
    with pytest.raises(ValueError):
        AnatomyEvidence.model_validate(payload)
