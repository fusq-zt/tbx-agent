from __future__ import annotations

import io
import json
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tbx_agent.anatomy_runs import (
    AnatomyRunRecord,
    AnatomyRunStatus,
    RefinementRunStatus,
)
from tbx_agent.api.main import create_app
from tbx_agent.capabilities import build_capability_snapshot
from tbx_agent.config import Settings
from tbx_agent.schemas import DetectionEvidence
from tbx_agent.service import TBXAgentService
from tbx_agent.storage import SQLiteStore
from tbx_agent.tools import ToolInvocation, ToolName
from tbx_agent.vision.anatomy import (
    AnatomyBackendUnavailable,
    AnatomyEvidence,
    AnatomyMask,
    AnatomyRuntimeProbe,
    LungSide,
    build_chat_spatial_summary,
    build_generation_key,
    build_spatial_summary,
    encode_binary_mask,
    evaluate_lung_masks,
    localize_detection_boxes,
)
from tbx_agent.vision.display import select_display_detections
from tbx_agent.vision.refinement import (
    ContourRefinementEvidence,
    DetectionContourEvidence,
    DetectionRefinementStatus,
    RefinementBackendUnavailable,
    RefinementRuntimeProbe,
    canonical_sha256,
    detector_box_digest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "a" * 64
MODEL_STATE_SHA256 = "b" * 64


def _settings(tmp_path: Path) -> Settings:
    return replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        anatomy_backend="xrv_pspnet",
        anatomy_required=False,
        anatomy_max_workers=1,
    )


def _png(color: tuple[int, int, int] = (40, 60, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _lung_masks() -> tuple[list[list[bool]], list[list[bool]]]:
    left = [[False] * 512 for _ in range(512)]
    right = [[False] * 512 for _ in range(512)]
    for y in range(70, 450):
        for x in range(280, 450):
            left[y][x] = True
        for x in range(60, 240):
            right[y][x] = True
    return left, right


class _StaticAnatomyBackend:
    backend_id = "static-anatomy-test"
    loaded = True

    @staticmethod
    def probe_runtime(*, load: bool = False) -> AnatomyRuntimeProbe:
        return AnatomyRuntimeProbe(
            backend_id="static-anatomy-test",
            loaded=True,
            available="yes",
            detail="test backend ready",
        )

    def generation_key_for(self, image_sha256: str) -> str:
        return build_generation_key(
            image_sha256=image_sha256,
            model_weight_sha256=MODEL_SHA256,
            preprocessing_id="static-source-space-v1",
            policy_id="paired-lung-qc-v1",
            backend_id=self.backend_id,
            parameters={"model_state_dict_sha256": MODEL_STATE_SHA256},
        )

    def infer(self, *, case_id, image) -> AnatomyEvidence:
        left, right = _lung_masks()
        return AnatomyEvidence(
            run_id=f"backend-local-{uuid.uuid4()}",
            case_id=case_id,
            image_sha256=image.sha256,
            image_width=image.width,
            image_height=image.height,
            backend_id=self.backend_id,
            model_id="static-paired-lung-model",
            model_weight_sha256=MODEL_SHA256,
            model_state_dict_sha256=MODEL_STATE_SHA256,
            preprocessing_id="static-source-space-v1",
            policy_id="paired-lung-qc-v1",
            generation_key=self.generation_key_for(image.sha256),
            masks=[
                AnatomyMask(structure=LungSide.LEFT, payload=encode_binary_mask(left)),
                AnatomyMask(structure=LungSide.RIGHT, payload=encode_binary_mask(right)),
            ],
            qc=evaluate_lung_masks(left, right),
            runtime_ms=2,
        )


class _FailingAnatomyBackend(_StaticAnatomyBackend):
    def infer(self, *, case_id, image):
        raise AnatomyBackendUnavailable("simulated optional runtime failure")


class _UnavailableAtStartupAnatomyBackend:
    backend_id = "unavailable-anatomy-test"
    loaded = False

    def __init__(self):
        self.probe_load_arguments: list[bool] = []

    def probe_runtime(self, *, load: bool = False) -> AnatomyRuntimeProbe:
        self.probe_load_arguments.append(load)
        return AnatomyRuntimeProbe(
            backend_id=self.backend_id,
            loaded=False,
            available="no",
            detail="test checkpoint is unavailable",
        )


class _StaticRefinementBackend:
    backend_id = "static-medsam-refinement-test"
    loaded = True

    @staticmethod
    def probe_runtime(*, load: bool = False) -> RefinementRuntimeProbe:
        return RefinementRuntimeProbe(
            backend_id="static-medsam-refinement-test",
            loaded=True,
            available="yes",
            detail="test refiner ready",
        )

    def generation_key_for(self, *, image_sha256, boxes, anatomy_generation_key):
        return canonical_sha256(
            {
                "backend": self.backend_id,
                "image": image_sha256,
                "boxes": boxes,
                "anatomy": anatomy_generation_key,
            }
        )

    def refine(self, *, case_id, image, boxes, anatomy):
        items = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(value) for value in box)
            mask = [[False] * image.width for _ in range(image.height)]
            for y in range(max(0, y1), min(image.height, max(y1 + 1, y2))):
                for x in range(max(0, x1), min(image.width, max(x1 + 1, x2))):
                    mask[y][x] = True
            encoded = encode_binary_mask(mask)
            items.append(
                DetectionContourEvidence(
                    detection_index=index,
                    bbox_xyxy=box,
                    status=DetectionRefinementStatus.REFINED,
                    mask=encoded,
                    raw_mask_pixels=encoded.foreground_pixels,
                    lung_constrained_pixels=encoded.foreground_pixels,
                    mask_prompt_overlap_fraction=1.0,
                    note="visualization_only_nonvalidated_contour",
                )
            )
        return ContourRefinementEvidence(
            case_id=case_id,
            image_sha256=image.sha256,
            image_width=image.width,
            image_height=image.height,
            backend_id=self.backend_id,
            model_id="static-medsam-test",
            model_revision="f" * 40,
            model_weight_sha256="c" * 64,
            model_state_dict_sha256="d" * 64,
            model_config_sha256="e" * 64,
            preprocessor_config_sha256="f" * 64,
            preprocessing_id="static-medsam-preprocess-v1",
            policy_id="static-medsam-policy-v1",
            generation_key=self.generation_key_for(
                image_sha256=image.sha256,
                boxes=boxes,
                anatomy_generation_key=anatomy.generation_key,
            ),
            anatomy_generation_key=anatomy.generation_key,
            anatomy_mask_digest="a" * 64,
            detector_box_digest=detector_box_digest(boxes),
            items=items,
            max_prompts_per_run=24,
            max_batch_size=4,
            capacity_abstained_count=0,
            runtime_ms=1,
        )


class _FailingRefinementBackend(_StaticRefinementBackend):
    def refine(self, *, case_id, image, boxes, anatomy):
        raise RefinementBackendUnavailable("simulated MedSAM runtime failure")


def _assessment(service: TBXAgentService):
    payload = _png()
    case = service.assess_cxr(
        payload,
        user_id="patient-1",
        owner_scope="tenant:patient-1",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )[0]
    case, _ = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
        payload=payload,
    )
    service.respond_with_tool(
        selected_tool="localize_current_cxr",
        message="定位候选区域",
        thread_id="anatomy-test",
        user_id="patient-1",
        owner_scope=case.owner_scope,
        case_id=case.case_id,
    )
    return service.store.get_case(
        case.case_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )


def _wait_for_terminal(service: TBXAgentService, case_id: str, run_id: str):
    for _ in range(200):
        run = service.get_anatomy_run(
            run_id=run_id,
            case_id=case_id,
            owner_scope="tenant:patient-1",
            user_id="patient-1",
        )
        if run.status.value in {
            "completed",
            "completed_with_refinement_failure",
            "technical_failure",
        }:
            return run
        time.sleep(0.01)
    raise AssertionError("anatomy worker did not reach a terminal state")


def test_anatomy_run_is_cached_and_cannot_change_rank03_routing(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(service)
    fusion_before = case.fusion_decision.model_dump(mode="json")

    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    completed = _wait_for_terminal(service, case.case_id, requested.run_id)

    assert completed.status.value == "completed"
    assert completed.evidence is not None
    assert completed.evidence.run_id == requested.run_id
    assert completed.evidence.routing_effect == "none"
    assert completed.clinical_validation is False
    assert len(completed.detector_locations) == len(case.localization_evidence.detections)
    assert completed.spatial_summary is not None
    assert completed.spatial_summary.candidate_count == len(case.localization_evidence.detections)
    assert completed.spatial_summary.routing_effect == "none"
    assert completed.localization_policy_id == "detector-lung-field-localization-v1"
    assert completed.anatomy_generation_key == completed.evidence.generation_key
    assert completed.generation_key != completed.anatomy_generation_key

    chat_response = service._tool_inspect_anatomical_context(  # noqa: SLF001
        ToolInvocation(
            tool_name=ToolName.INSPECT_ANATOMICAL_CONTEXT.value,
            message="候选区域位于哪个肺野？",
            thread_id="anatomy-chat-summary",
            user_id="patient-1",
            owner_scope=case.owner_scope,
            request_id="anatomy-chat-request",
            trace_id="anatomy-chat-trace",
            routing_policy_id="test-router",
            case_id=case.case_id,
        )
    )
    assert len(chat_response.summary) < 120
    assert "框内肺野掩膜交叠" not in chat_response.summary
    assert "占该侧肺野" not in chat_response.summary
    assert "二维投影" not in chat_response.summary
    assert "不代表肺叶" not in chat_response.summary
    # The same completed run still retains the precise ratios for audit/detail.
    assert completed.spatial_summary is not None
    if any(item.status == "localized" for item in completed.detector_locations):
        assert any("框内肺野掩膜交叠" in item for item in completed.spatial_summary.statements)

    explanation = service.respond(
        message="解释这个胸片模型结果和候选框位置",
        thread_id="anatomy-explanation-thread",
        user_id="patient-1",
        owner_scope="tenant:patient-1",
        case_id=case.case_id,
    )
    assert "候选" in explanation.summary
    assert explanation.visual_result == case.fusion_decision.visual_result
    assert explanation.predicted_class == case.fusion_decision.predicted_class
    unchanged = service.store.get_case(
        case.case_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )
    unchanged_fusion = unchanged.fusion_decision.model_dump(mode="json")
    for routing_field in (
        "visual_result",
        "review_required",
        "review_reasons",
        "predicted_class",
        "classifier_flagged",
        "detector_flagged",
    ):
        assert unchanged_fusion[routing_field] == fusion_before[routing_field]
    assert unchanged.localization_evidence.status in {
        "completed",
        "completed_no_detection",
    }
    assert "detector_execution:not_requested" in unchanged.vision_evidence.artifact_refs

    # Reading already-completed localization/anatomy evidence must not mutate
    # either generation identity. An identical explicit request therefore
    # reuses the same completed run instead of manufacturing a fresh run.
    refreshed = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    assert refreshed.run_id == completed.run_id
    assert refreshed.reused_existing_run is True
    refreshed_completed = _wait_for_terminal(service, case.case_id, refreshed.run_id)
    assert len(refreshed_completed.detector_locations) == len(
        unchanged.localization_evidence.detections
    )
    assert refreshed_completed.spatial_summary.candidate_count == len(
        unchanged.localization_evidence.detections
    )

    reused = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    assert reused.run_id == refreshed.run_id
    assert reused.reused_existing_run is True

    with pytest.raises(ValueError, match="does not match"):
        service.request_anatomy_run(
            case_id=case.case_id,
            owner_scope=case.owner_scope,
            user_id="patient-1",
            payload=_png((90, 30, 10)),
        )


def test_controller_answers_explicit_anatomy_query_from_completed_run_without_rerun(
    tmp_path,
):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(service)
    localized = service.respond_with_controller(
        message="病灶候选区域在哪里？",
        thread_id="cached-anatomy-localization-setup",
        user_id="patient-1",
        owner_scope=case.owner_scope,
        case_id=case.case_id,
    )
    assert localized.execution_plan["source"] == "plan_react"
    assert localized.execution_plan["tool_names"] == []
    assert localized.tool_results == []
    case = service.store.get_case(
        case.case_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    completed = _wait_for_terminal(service, case.case_id, requested.run_id)
    runs_before = service.store.list_anatomy_runs(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )

    result = service.respond_with_controller(
        message="这些候选框位于哪个肺区？",
        thread_id="cached-anatomy-controller",
        user_id="patient-1",
        owner_scope=case.owner_scope,
        case_id=case.case_id,
    )

    assert result.execution_plan["source"] == "plan_react"
    selected = select_display_detections(
        [item.model_dump(mode="json") for item in case.localization_evidence.detections],
        image_width=case.image_width,
    )
    visible_locations = [
        completed.detector_locations[item.raw_index]
        for item in selected
        if item.raw_index < len(completed.detector_locations)
    ]
    assert result.response.summary == build_chat_spatial_summary(
        visible_locations,
        anatomy_qc_status=completed.spatial_summary.anatomy_qc_status,
    )
    assert "当前病例：" not in result.response.summary
    assert "框内肺野掩膜交叠" not in result.response.summary
    assert "占该侧肺野" not in result.response.summary
    assert result.response.visual_result == case.fusion_decision.visual_result
    assert result.response.predicted_class == case.fusion_decision.predicted_class

    # A cache hit is evidence reuse, not a tool execution in this turn.
    assert result.execution_plan["tool_names"] == []
    assert result.execution_plan["steps"] == []
    assert result.tool_results == []
    assert result.receipt is None
    assert result.audit_action == "agent_no_tool"

    runs_after = service.store.list_anatomy_runs(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    assert [run.run_id for run in runs_after] == [run.run_id for run in runs_before]
    # Detailed overlap metrics remain persisted for audit and the case detail page.
    if any(item.status == "localized" for item in completed.detector_locations):
        assert any(
            "框内肺野掩膜交叠" in statement
            for statement in completed.spatial_summary.statements
        )


def test_cached_compound_request_projects_visual_evidence_without_fake_receipts(
    tmp_path,
):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(service)
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    _wait_for_terminal(service, case.case_id, requested.run_id)
    classifier_calls = service.vision.call_count
    localization_calls = service.vision.localization_call_count
    runs_before = service.store.list_anatomy_runs(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )

    result = service.respond_with_controller(
        message=(
            "这是一个 68 岁男性，近期咳嗽、乏力，家里有人以前得过肺结核。"
            "我上传了他的胸片。请先告诉我当前模型筛查结果；如果有可疑区域，"
            "说明它大概在哪一侧和哪个肺野区域；再结合指南告诉我下一步一般需要"
            "做什么检查。请把模型证据和指南建议分开写，如果现有证据不足就明确"
            "指出，不要替我下最终诊断。"
        ),
        thread_id="cached-compound-evidence",
        user_id="patient-1",
        owner_scope=case.owner_scope,
        case_id=case.case_id,
    )

    assert result.execution_plan["tool_names"] == ["search_tb_knowledge"]
    assert [item.receipt.model_tool_name for item in result.tool_results] == [
        "search_tb_knowledge"
    ]
    assert result.execution_plan["cached_evidence"] == [
        "classification",
        "localization",
        "lung_anatomy",
    ]
    assert result.execution_plan["composition_mode"] == (
        "deterministic_compound_evidence"
    )
    initial_steps = result.execution_plan["initial_plan"]["steps"]
    assert [item["evidence_need"] for item in initial_steps] == [
        "classification",
        "localization",
        "lung_anatomy",
        "tb_knowledge",
    ]
    assert all(item["evidence_need"] != "none" for item in initial_steps)
    assert result.response.summary.startswith("模型证据\n")
    assert "\n\n指南建议\n" in result.response.summary
    assert "NAAT" in result.response.summary
    assert result.response.predicted_class == case.fusion_decision.predicted_class
    selected = select_display_detections(
        [item.model_dump(mode="json") for item in case.localization_evidence.detections],
        image_width=case.image_width,
    )
    if selected:
        assert result.response.visual_evidence_notes
    else:
        assert "候选区域" in result.response.summary
    assert result.response.citations
    assert result.response.next_step_information
    assert service.vision.call_count == classifier_calls
    assert service.vision.localization_call_count == localization_calls
    runs_after = service.store.list_anatomy_runs(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    assert [item.run_id for item in runs_after] == [item.run_id for item in runs_before]


def test_anatomy_chat_matches_visible_candidates_but_keeps_dense_raw_details(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(service)
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    completed = _wait_for_terminal(service, case.case_id, requested.run_id)
    assert completed.evidence is not None

    detections = [
        DetectionEvidence(bbox_xyxy=(60, 80, 220, 250), score=0.95),
        DetectionEvidence(bbox_xyxy=(64, 84, 224, 254), score=0.90),
        DetectionEvidence(bbox_xyxy=(80, 270, 220, 420), score=0.85),
        DetectionEvidence(bbox_xyxy=(300, 80, 430, 250), score=0.40),
    ]
    locations = localize_detection_boxes(
        [item.bbox_xyxy for item in detections],
        anatomy=completed.evidence,
    )
    detailed_summary = build_spatial_summary(
        locations,
        anatomy_qc_status=completed.evidence.qc.status,
    )
    projected_case = case.model_copy(
        deep=True,
        update={
            "localization_evidence": case.localization_evidence.model_copy(
                update={"detections": detections}
            )
        },
    )
    projected_run = completed.model_copy(
        deep=True,
        update={
            "detector_locations": locations,
            "spatial_summary": detailed_summary,
        },
    )

    selected = select_display_detections(
        [item.model_dump(mode="json") for item in detections],
        image_width=projected_case.image_width,
    )
    assert [item.raw_index for item in selected] == [0]
    response = service._anatomy_run_response(  # noqa: SLF001
        projected_run,
        case=projected_case,
        request_id="display-aligned-anatomy",
        trace_id="display-aligned-anatomy-trace",
        thread_id="display-aligned-anatomy-thread",
    )

    assert response.summary == build_chat_spatial_summary(
        [locations[0]],
        anatomy_qc_status=completed.evidence.qc.status,
    )
    assert response.summary != build_chat_spatial_summary(
        locations,
        anatomy_qc_status=completed.evidence.qc.status,
    )
    assert len(projected_run.detector_locations) == 4
    assert projected_run.spatial_summary.candidate_count == 4
    assert sum(
        "框内肺野掩膜交叠" in statement
        for statement in projected_run.spatial_summary.statements
    ) == 4


def test_anatomy_failure_is_isolated_and_persisted(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_FailingAnatomyBackend(),
    )
    case = _assessment(service)
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    failed = _wait_for_terminal(service, case.case_id, requested.run_id)
    assert failed.status.value == "technical_failure"
    assert failed.error_code == "backend_unavailable"
    persisted_case = service.store.get_case(
        case.case_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )
    assert persisted_case.vision_evidence is not None
    assert persisted_case.fusion_decision is not None


def test_enabled_anatomy_is_loaded_during_capability_probe_and_tool_fails_closed(
    tmp_path,
):
    backend = _UnavailableAtStartupAnatomyBackend()
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=backend,
    )

    snapshot = build_capability_snapshot(service)
    components = {item.component_id: item for item in snapshot.components}
    anatomy = components["anatomy_segmentation"]
    anatomy_tool = components["agent_tool:analyze_lung_anatomy"]

    assert backend.probe_load_arguments
    assert all(backend.probe_load_arguments)
    assert anatomy.state.value == "unavailable"
    assert anatomy.loaded is False
    assert anatomy_tool.state.value == "unavailable"
    assert anatomy_tool.loaded is False
    assert anatomy_tool.required is False
    tool_status = {
        item.name: item for item in service.tool_registry.statuses()
    }[ToolName.INSPECT_ANATOMICAL_CONTEXT.value]
    assert tool_status.availability.value == "unavailable"
    assert tool_status.detail_code == "health_check_not_ready"


def test_report_explicitly_records_absent_optional_anatomy_evidence(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(service)

    report = service.create_report(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        actor_id="patient-1",
    )
    payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    markdown = Path(report["markdown_path"]).read_text(encoding="utf-8")

    assert payload["schema_version"] == "tbx.report.v4"
    assert payload["anatomy_evidence_summary"] is None
    assert "未生成可用的肺野分割/空间关系证据" in markdown


def test_refinement_success_is_persisted_without_changing_rank03_route(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
        refinement_backend=_StaticRefinementBackend(),
    )
    case = _assessment(service)
    fusion_before = case.fusion_decision.model_dump(mode="json")
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    completed = _wait_for_terminal(service, case.case_id, requested.run_id)

    assert completed.status == AnatomyRunStatus.COMPLETED
    assert completed.refinement_status == RefinementRunStatus.COMPLETED
    assert completed.refinement_evidence is not None
    assert completed.refinement_evidence.routing_effect == "none"
    assert completed.refinement_evidence.clinical_validation is False
    unchanged = service.store.get_case(
        case.case_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )
    assert unchanged.fusion_decision.model_dump(mode="json") == fusion_before

    report = service.create_report(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        actor_id="patient-1",
    )
    payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    summary = payload["anatomy_evidence_summary"]
    assert summary["status"] == "completed"
    assert summary["routing_effect"] == "none"
    assert summary["clinical_validation"] is False
    assert summary["anatomy_model_identity"]["model_id"] == "static-paired-lung-model"
    assert summary["refinement"]["status"] == "completed"
    assert summary["refinement"]["model_identity"]["model_id"] == "static-medsam-test"
    assert "masks" not in summary
    assert "detector_locations" not in summary
    assert "refinement_evidence" not in summary


def test_refinement_failure_degrades_only_optional_branch_and_can_retry(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
        refinement_backend=_FailingRefinementBackend(),
    )
    case = _assessment(service)
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    degraded = _wait_for_terminal(service, case.case_id, requested.run_id)

    assert degraded.status == AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE
    assert degraded.evidence is not None
    assert degraded.spatial_summary is not None
    assert degraded.refinement_status == RefinementRunStatus.TECHNICAL_FAILURE
    assert degraded.refinement_evidence is None
    assert degraded.refinement_error_code == "backend_unavailable"
    assert (
        service.store.find_latest_completed_anatomy_run(
            case_id=case.case_id,
            owner_scope=case.owner_scope,
            user_id="patient-1",
        ).run_id
        == degraded.run_id
    )

    report = service.create_report(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        actor_id="patient-1",
    )
    payload = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    summary = payload["anatomy_evidence_summary"]
    assert summary["status"] == "completed_with_refinement_failure"
    assert summary["spatial_summary"] is not None
    assert summary["refinement"] == {
        "status": "technical_failure",
        "backend_id": "static-medsam-refinement-test",
        "error_code": "backend_unavailable",
        "model_identity": None,
        "prompt_count": 0,
        "refined_count": 0,
        "capacity_abstained_count": 0,
    }

    retried = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    assert retried.run_id != degraded.run_id
    assert retried.reused_existing_run is False


def test_service_startup_marks_orphaned_worker_run_and_audits_it(tmp_path):
    settings = _settings(tmp_path)
    store = SQLiteStore(settings.db_path)
    seed_service = TBXAgentService(
        settings,
        store=store,
        anatomy_backend=_StaticAnatomyBackend(),
    )
    case = _assessment(seed_service)
    pending = AnatomyRunRecord(
        run_id="orphaned-anatomy-run",
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
        image_sha256=case.image_sha256,
        generation_key="c" * 64,
        backend_id="static-anatomy-test",
    )
    store.create_anatomy_run(pending)

    TBXAgentService(
        settings,
        store=store,
        anatomy_backend=_StaticAnatomyBackend(),
    )

    failed = store.get_anatomy_run(
        pending.run_id,
        case.owner_scope,
        subject_user_id="patient-1",
    )
    assert failed.status == AnatomyRunStatus.TECHNICAL_FAILURE
    assert failed.error_code == "worker_interrupted"
    assert store.audit_count("anatomy_worker_interrupted") == 1


def test_anatomy_api_returns_async_record_and_transparent_boundary(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
        refinement_backend=_StaticRefinementBackend(),
    )
    client = TestClient(create_app(service))
    assessment = client.post(
        "/v1/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "user_id": "patient-1",
            "owner_scope": "tenant:patient-1",
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
    )
    assert assessment.status_code == 200
    case_id = assessment.json()["case"]["case_id"]
    params = {"owner_scope": "tenant:patient-1", "user_id": "patient-1"}
    created = client.post(f"/v1/cases/{case_id}/anatomy-runs", params=params)
    assert created.status_code in {200, 202}
    run_id = created.json()["run_id"]

    for _ in range(200):
        fetched = client.get(
            f"/v1/cases/{case_id}/anatomy-runs/{run_id}",
            params=params,
        )
        assert fetched.status_code == 200
        if fetched.json()["status"] == "completed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("anatomy API run did not complete")

    completed_payload = fetched.json()
    assert completed_payload["spatial_summary"]["routing_effect"] == "none"
    assert completed_payload["spatial_summary"]["statements"]
    assert completed_payload["localization_policy_id"] == ("detector-lung-field-localization-v1")
    assert completed_payload["refinement_status"] == "completed"
    assert completed_payload["refinement_evidence"]["routing_effect"] == "none"

    boundary = client.get(
        f"/v1/cases/{case_id}/anatomy-runs/{run_id}/boundary.png",
        params={**params, "structure": "combined"},
    )
    assert boundary.status_code == 200
    assert boundary.headers["content-type"] == "image/png"
    assert boundary.headers["x-anatomy-routing-effect"] == "none"
    with Image.open(io.BytesIO(boundary.content)) as image:
        assert image.size == (512, 512)
        assert image.mode == "RGBA"

    contours = client.get(
        f"/v1/cases/{case_id}/anatomy-runs/{run_id}/contours.png",
        params=params,
    )
    assert contours.status_code == 200
    assert contours.headers["content-type"] == "image/png"
    assert contours.headers["x-refinement-routing-effect"] == "none"
    assert contours.headers["x-clinical-validation"] == "false"
    with Image.open(io.BytesIO(contours.content)) as image:
        assert image.size == (512, 512)
        assert image.mode == "RGBA"

    denied = client.get(
        f"/v1/cases/{case_id}/anatomy-runs/{run_id}",
        params={"owner_scope": "tenant:patient-1", "user_id": "other"},
    )
    assert denied.status_code == 403


def test_refinement_failure_api_keeps_boundary_and_withholds_contours(tmp_path):
    service = TBXAgentService(
        _settings(tmp_path),
        anatomy_backend=_StaticAnatomyBackend(),
        refinement_backend=_FailingRefinementBackend(),
    )
    case = _assessment(service)
    requested = service.request_anatomy_run(
        case_id=case.case_id,
        owner_scope=case.owner_scope,
        user_id="patient-1",
    )
    degraded = _wait_for_terminal(service, case.case_id, requested.run_id)
    assert degraded.status == AnatomyRunStatus.COMPLETED_WITH_REFINEMENT_FAILURE

    client = TestClient(create_app(service))
    params = {"owner_scope": case.owner_scope, "user_id": "patient-1"}
    boundary = client.get(
        f"/v1/cases/{case.case_id}/anatomy-runs/{degraded.run_id}/boundary.png",
        params=params,
    )
    contours = client.get(
        f"/v1/cases/{case.case_id}/anatomy-runs/{degraded.run_id}/contours.png",
        params=params,
    )

    assert boundary.status_code == 200
    assert boundary.headers["x-anatomy-routing-effect"] == "none"
    assert contours.status_code == 409
