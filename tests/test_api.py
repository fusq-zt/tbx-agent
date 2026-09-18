from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.schemas import ClassifierClass
from tbx_agent.service import TBXAgentService
from tbx_agent.vision import MockRank03Backend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicTBBackend(MockRank03Backend):
    """Make the batch-review contract independent of a random case UUID."""

    def infer(self, *, case_id, image):
        evidence = super().infer(case_id=case_id, image=image)
        return evidence.model_copy(
            update={
                "class_probabilities": {
                    "healthy": 0.01,
                    "sick_non_tb": 0.09,
                    "tb": 0.90,
                },
                "predicted_class": ClassifierClass.TB,
                "classifier_argmax_tied": False,
                "classifier_flagged": True,
                "top1_score": 0.90,
                "top2_score": 0.09,
                "top1_top2_margin": 0.81,
            }
        )


def _client(tmp_path: Path, *, deterministic_tb: bool = False) -> TestClient:
    base = Settings.from_env()
    settings = replace(
        base,
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        openai_enabled=False,
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        anatomy_backend="none",
        anatomy_required=False,
    )
    backend = (
        _DeterministicTBBackend(settings.fusion_policy(), settings.rank03_config())
        if deterministic_tb
        else None
    )
    return TestClient(create_app(TBXAgentService(settings, vision_backend=backend)))


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), color=(50, 70, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_langgraph_execution(payload: dict, *, used_tools: list[str]) -> None:
    plan = payload["execution_plan"]
    assert plan["framework"] == "langgraph"
    assert plan["tool_names"] == used_tools
    node_trace = plan["graph_node_trace"]
    assert node_trace[:3] == ["load_context", "plan", "decide"]
    assert node_trace[-1] == "finalize"
    assert all(
        node in {
            "load_context",
            "plan",
            "decide",
            "execute_tool",
            "observe",
            "replan",
            "finalize",
        }
        for node in node_trace
    )


def test_general_question_without_configured_llm_never_returns_case_state(tmp_path):
    response = _client(tmp_path).post(
        "/v1/agent/respond",
        json={
            "thread_id": "thread-general-no-llm",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "1+1 = ？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_kind"] == "general_answer"
    assert payload["execution_receipt"] is None
    assert payload["execution_receipts"] == []
    _assert_langgraph_execution(payload, used_tools=[])
    assert "模型" in payload["summary"]
    assert "未连接" in payload["summary"]
    assert "分类未运行" not in payload["summary"]
    assert "定位未运行" not in payload["summary"]


def test_health_manifest_assessment_agent_and_screening(tmp_path):
    client = _client(tmp_path)
    health = client.get("/healthz").json()
    assert health["clinical_validation"] is False
    assert health["required_components_ready"] is True
    assert health["mode"] == "synthetic_demo"
    assert health["vision_backend"] == "mock"
    assert health["vision_model_display_name"] == "TBX-CXR Vision v1"
    capabilities = client.get("/v1/system/capabilities")
    assert capabilities.status_code == 200
    capability_payload = capabilities.json()
    assert capability_payload["status"] == "ok"
    assert capability_payload["mode"] == "synthetic_demo"
    components = {item["component_id"]: item for item in capability_payload["components"]}
    assert components["rank03_image_assessment"]["synthetic"] is True
    assert components["guideline_retrieval"]["state"] == "ready"
    assert components["llm_evidence_composer"]["state"] == "disabled"
    assert components["contour_refinement"]["state"] == "disabled"
    assert components["contour_refinement"]["required"] is False
    assert components["contour_refinement"]["synthetic"] is False
    assert components["agent_tool:analyze_lung_anatomy"]["state"] == "unavailable"
    assert components["agent_tool:analyze_lung_anatomy"]["required"] is False
    assert {
        component_id.removeprefix("agent_tool:")
        for component_id in components
        if component_id.startswith("agent_tool:")
    } == {
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    }
    orchestration_component = components["agent_orchestration"]
    assert orchestration_component["implementation"] == "tbx-react-first-v4"
    assert "LangGraph ReAct-first" in orchestration_component["detail"]
    assert "TaskSpec" not in orchestration_component["detail"]
    assert str(tmp_path) not in capabilities.text
    manifest = client.get("/v1/system/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.json()
    assert manifest_payload["vision_backend"] == "mock"
    assert manifest_payload["vision_model_display_name"] == "TBX-CXR Vision v1"
    assert manifest_payload["knowledge_snapshot_id"]
    assert manifest_payload["fusion_policy_id"] == "rank03-user-trained-native-argmax-v2"
    assert manifest_payload["classifier_rule"] == "native_three_class_argmax"
    assert "classifier_threshold" not in manifest_payload
    assert "classifier_threshold_comparison" not in manifest_payload
    assert manifest_payload["classifier_routes"] == {
        "healthy": "model_not_flagged",
        "sick_non_tb": "non_tb_abnormal",
        "tb": "model_flagged",
    }
    assert manifest_payload["detector_role"] == "advisory_localization_only"
    assert manifest_payload["input_contract"] == {
        "max_upload_bytes": 20 * 1024 * 1024,
        "raster_formats": ["PNG", "JPEG"],
        "dicom": {
            "enabled": True,
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
    }
    assert manifest_payload["policy_selection_design"] is None
    assert manifest_payload["policy_selection_metrics"] is None
    assert manifest_payload["policy_heldout_metrics"] is None
    assert manifest_payload["reference_split_sha256"] is None
    assert manifest_payload["selection_split_sha256"] is None
    assert manifest_payload["narrator"]["backend"] == "none"
    orchestration = manifest_payload["agent_orchestration"]
    assert orchestration["framework"] == "langgraph"
    assert orchestration["policy_id"] == "tbx-react-first-v4"
    assert orchestration["strategy"] == "react_first_optional_plan"
    assert orchestration["graph_nodes"] == [
        "load_context",
        "plan",
        "decide",
        "execute_tool",
        "observe",
        "replan",
        "finalize",
    ]
    assert orchestration["model_visible_tools"] == [
        "classify_cxr",
        "localize_cxr",
        "analyze_lung_anatomy",
        "search_tb_knowledge",
    ]
    assert orchestration["tool_calling"] == {
        "preferred": "strict_json_schema_decision",
        "fallback": "rule_plan_after_model_error",
    }
    assert orchestration["durable_langgraph_checkpointer"] is False
    assert orchestration["business_state_authority"] == "SQLiteStore"
    assert "OLLAMA_BASE_URL" not in manifest.text
    assert "127.0.0.1:11434" not in manifest.text

    assessment = client.post(
        "/v1/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "user_id": "user",
            "owner_scope": "tenant:user",
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
    )
    assert assessment.status_code == 200
    assessment_payload = assessment.json()
    assert "image_artifact_ref" not in assessment_payload["case"]
    assert str(tmp_path) not in assessment.text
    assert assessment_payload["response"]["summary"] == "胸片已载入，尚未运行分类。"
    assert assessment_payload["response"]["predicted_class"] is None
    assert assessment_payload["case"]["classification_status"] == "not_requested"
    assert assessment_payload["case"]["vision_evidence"] is None
    assert assessment_payload["case"]["fusion_decision"] is None

    case_id = assessment_payload["case"]["case_id"]
    case_read = client.get(
        f"/v1/cases/{case_id}",
        params={"owner_scope": "tenant:user", "user_id": "user"},
    )
    assert case_read.status_code == 200
    assert "image_artifact_ref" not in case_read.json()
    report = client.post(
        f"/v1/cases/{case_id}/reports",
        json={
            "owner_scope": "tenant:user",
            "user_id": "user",
            "actor_id": "user",
        },
    )
    assert report.status_code == 200
    report_payload = report.json()
    assert "markdown_path" not in report_payload
    assert "json_path" not in report_payload
    markdown = client.get(
        report_payload["markdown_download_url"],
        params={"owner_scope": "tenant:user", "user_id": "user"},
    )
    assert markdown.status_code == 200
    assert "TBX-Agent" in markdown.text
    report_json = client.get(
        report_payload["json_download_url"],
        params={"owner_scope": "tenant:user", "user_id": "user"},
    )
    assert report_json.status_code == 200
    assert "image_artifact_ref" not in report_json.text
    assert str(tmp_path) not in report_json.text
    denied = client.get(
        report_payload["json_download_url"],
        params={"owner_scope": "tenant:other", "user_id": "user"},
    )
    assert denied.status_code == 403

    agent = client.post(
        "/v1/agent/respond",
        json={
            "thread_id": "thread",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "胸片异常后要做什么检查？",
        },
    )
    assert agent.status_code == 200
    assert agent.json()["citations"]
    receipt = agent.json()["execution_receipt"]
    assert receipt["status"] == "succeeded"
    assert receipt["tool_name"] == "search_tb_knowledge"
    assert receipt["model_tool_name"] == "search_tb_knowledge"
    _assert_langgraph_execution(
        agent.json(),
        used_tools=["search_tb_knowledge"],
    )
    assert "message" not in receipt

    screening = client.post(
        "/v1/screening/sessions",
        json={
            "thread_id": "thread",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "consent": True,
        },
    )
    assert screening.status_code == 200
    assert screening.json()["response"]["next_question"]["question_id"] == "emergency_red_flags"


def test_upload_requires_both_consent_and_cxr_attestation(tmp_path):
    response = _client(tmp_path).post(
        "/v1/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "user_id": "user",
            "owner_scope": "tenant:user",
            "consent_to_process": "false",
            "attested_chest_radiograph": "true",
        },
    )
    assert response.status_code == 422


def test_case_followups_classify_on_demand_then_reuse_structured_evidence(tmp_path):
    client = _client(tmp_path)
    assessment = client.post(
        "/v1/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data={
            "user_id": "case-user",
            "owner_scope": "tenant:case-user",
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
    )
    assert assessment.status_code == 200
    case_id = assessment.json()["case"]["case_id"]

    answers = []
    expected_tools = (
        ("classify_current_cxr", "classify_cxr"),
        ("localize_current_cxr", "localize_cxr"),
        (None, None),
    )
    for message, (expected_internal_tool, expected_public_tool) in zip(
        ("这张胸片有没有结核病？", "病灶在哪？", "为什么这样分类？"),
        expected_tools,
        strict=True,
    ):
        result = client.post(
            "/v1/agent/respond",
            json={
                "thread_id": "case-thread",
                "user_id": "case-user",
                "owner_scope": "tenant:case-user",
                "message": message,
                "case_id": case_id,
            },
        )
        assert result.status_code == 200
        payload = result.json()
        public_json = json.dumps(payload, ensure_ascii=False)
        assert '"class_probabilities"' not in public_json
        assert '"class_probability_order"' not in public_json
        assert '"top1_score"' not in public_json
        assert '"top2_score"' not in public_json
        assert '"top1_top2_margin"' not in public_json
        assert '"classifier_threshold"' not in public_json
        assert payload["case_id"] == case_id
        if expected_public_tool is None:
            assert payload["execution_receipt"] is None
            assert payload["execution_receipts"] == []
            _assert_langgraph_execution(payload, used_tools=[])
        else:
            assert payload["execution_receipt"]["tool_name"] == expected_internal_tool
            assert (
                payload["execution_receipt"]["model_tool_name"]
                == expected_public_tool
            )
            _assert_langgraph_execution(payload, used_tools=[expected_public_tool])
        answers.append(payload)

    assert answers[0]["execution_plan"]["steps"][0]["tool_name"] == (
        "classify_cxr"
    )
    assert answers[0]["execution_plan"]["steps"][0]["internal_tool_name"] == (
        "classify_current_cxr"
    )
    assert answers[1]["diagnostic_information"] == []
    assert answers[1]["next_step_information"] == []
    assert answers[1]["citations"] == []
    rationale_text = "\n".join(
        [answers[2]["summary"], *answers[2]["visual_evidence_notes"]]
    )
    assert any(
        label in answers[2]["summary"]
        for label in ("健康类", "非结核异常类", "结核类")
    )
    assert "分类模型" in answers[2]["summary"]
    assert "%" not in rationale_text
    assert "相对得分" not in rationale_text
    assert "相对分数" not in rationale_text
    assert all("D-FINE" not in item for item in answers[2]["visual_evidence_notes"])


def test_agent_tool_free_turn_has_no_receipt_and_keeps_v2_trace(tmp_path):
    response = _client(tmp_path).post(
        "/v1/agent/respond",
        json={
            "thread_id": "social-thread",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "你好",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_receipt"] is None
    assert payload["execution_receipts"] == []
    _assert_langgraph_execution(payload, used_tools=[])
    assert payload["reflection"] is None
    assert payload["agent_trace"]["trace_version"] == "tbx-agent-trace-v2"
    assert payload["agent_trace"]["terminal"]["reason_code"] == "react_answered"


def test_only_batch_multipart_assessment_enrolls_pending_review(tmp_path):
    client = _client(tmp_path, deterministic_tb=True)
    identity = {"user_id": "batch-user", "owner_scope": "tenant:batch-user"}
    upload = {
        **identity,
        "consent_to_process": "true",
        "attested_chest_radiograph": "true",
    }

    interactive = client.post(
        "/v1/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data=upload,
    )
    assert interactive.status_code == 200
    interactive_payload = interactive.json()
    assert interactive_payload["case"]["review_id"] is None
    assert interactive_payload["case"]["review_status"] == "not_required"
    assert client.get("/v1/reviews/pending", params=identity).json() == []

    batch_upload = {**upload, "batch_item_id": "item-001"}
    first = client.post(
        "/v1/batches/batch-001/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data=batch_upload,
    )
    second = client.post(
        "/v1/batches/batch-001/assessments/cxr",
        files={"file": ("cxr.png", _png(), "image/png")},
        data=batch_upload,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    for payload in (first_payload, second_payload):
        public_case_json = json.dumps(payload["case"], ensure_ascii=False)
        assert '"class_probabilities"' not in public_case_json
        assert '"class_probability_order"' not in public_case_json
        assert '"top1_score"' not in public_case_json
        assert '"top2_score"' not in public_case_json
        assert '"top1_top2_margin"' not in public_case_json
        assert '"classifier_threshold"' not in public_case_json
    assert first_payload["case"]["case_id"] == interactive_payload["case"]["case_id"]
    assert first_payload["enqueued_for_review"] is True
    assert first_payload["review"]["origin"] == "batch_screening"
    assert first_payload["review"]["batch_id"] == "batch-001"
    assert first_payload["review"]["batch_item_id"] == "item-001"
    assert second_payload["review"]["review_id"] == first_payload["review"]["review_id"]

    stored_case = client.get(
        f"/v1/cases/{first_payload['case']['case_id']}", params=identity
    )
    assert stored_case.status_code == 200
    stored_case_json = stored_case.text
    assert '"class_probabilities"' not in stored_case_json
    assert '"class_probability_order"' not in stored_case_json
    assert '"top1_score"' not in stored_case_json
    assert '"top2_score"' not in stored_case_json
    assert '"top1_top2_margin"' not in stored_case_json
    assert '"classifier_threshold"' not in stored_case_json

    pending = client.get("/v1/reviews/pending", params=identity)
    assert pending.status_code == 200
    assert [item["review_id"] for item in pending.json()] == [
        first_payload["review"]["review_id"]
    ]
