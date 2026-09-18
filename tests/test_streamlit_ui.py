from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests
from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ENTRYPOINT = PROJECT_ROOT / "ui" / "streamlit_app.py"
UI_STYLESHEET = PROJECT_ROOT / "ui" / "assets" / "tbx_agent.css"


@dataclass(frozen=True)
class RecordedRequest:
    method: str
    path: str
    kwargs: dict


def _json_response(payload: object, *, status_code: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.encoding = "utf-8"
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    response._content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return response


def _bytes_response(content: bytes, *, content_type: str) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = content_type
    response._content = content
    return response


class RequestsStub:
    def __init__(self) -> None:
        self.calls: list[RecordedRequest] = []
        self.anatomy_polls = 0
        self.anatomy_terminal_status = "completed"
        self.anatomy_refinement_status = "disabled"
        self.anatomy_capacity_abstained_count = 0
        self.assessment_payload = _upload_payload()
        self.classified_case_payload = _assessment_payload()["case"]
        self.assessment_status_code = 200
        self.agent_tool_name = "classify_current_cxr"
        self.agent_execution_receipts: list[dict] | None = None
        self.agent_execution_plan: object = {
            "tool_names": ["classify_current_cxr"]
        }
        self.agent_response_overrides: dict[str, object] = {}
        self.case_read_payload: dict | None = None
        self.screening_answers: list[object] = []
        self.screening_case_id: str | None = None
        self.pending_reviews: list[dict] = []
        self.batch_enqueue = False
        self.vision_component_state = "ready"
        self.llm_component_state = "ready"

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        normalized_method = method.upper()
        path = urlsplit(url).path
        self.calls.append(RecordedRequest(normalized_method, path, kwargs))

        if normalized_method == "GET" and path == "/healthz":
            return _json_response(
                {
                    "status": "ok",
                    "service": "TBX-Agent",
                    "required_components_ready": (
                        self.vision_component_state == "ready"
                        and self.llm_component_state == "ready"
                    ),
                    "vision_backend": "rank03",
                    "vision_model_display_name": (
                        "TBX-CXR Vision v1（三分类 argmax 分流；D-FINE 仅定位）"
                    ),
                    "narrator_backend": "llama_cpp",
                    "mode": "real_rank03_with_local_llm",
                    "clinical_validation": False,
                }
            )
        if normalized_method == "GET" and path == "/v1/system/capabilities":
            return _json_response(
                {
                    "status": "ok",
                    "mode": "real_rank03_with_local_llm",
                    "runtime_verified": True,
                    "components": [
                        {
                            "component_id": "rank03_image_assessment",
                            "state": self.vision_component_state,
                            "synthetic": False,
                        },
                        {
                            "component_id": "llm_evidence_composer",
                            "state": self.llm_component_state,
                            "required": True,
                            "loaded": self.llm_component_state == "ready",
                            "synthetic": False,
                        },
                        {
                            "component_id": "anatomy_segmentation",
                            "state": "configured_not_probed",
                            "required": False,
                            "synthetic": False,
                        },
                    ],
                }
            )
        if normalized_method == "GET" and path == "/v1/system/manifest":
            return _json_response(
                {
                    "classifier_rule": "native_three_class_argmax",
                    "detector_role": "advisory_localization_only",
                    "vision_model_display_name": (
                        "TBX-CXR Vision v1（三分类 argmax 分流；D-FINE 仅定位）"
                    ),
                    "narrator": {"backend": "llama_cpp"},
                }
            )
        if normalized_method == "POST" and path == "/v1/assessments/cxr":
            return _json_response(
                self.assessment_payload,
                status_code=self.assessment_status_code,
            )
        if normalized_method == "POST" and path == "/v1/agent/respond":
            receipt = {
                "tool_name": self.agent_tool_name,
                "status": "succeeded",
                "attempt": 1,
            }
            receipts = (
                self.agent_execution_receipts
                if self.agent_execution_receipts is not None
                else [receipt]
            )
            agent_response = {
                "request_id": "request-contract",
                "summary": "已结合当前胸片结果生成辅助解释。",
                "visual_result": "model_not_flagged",
                "review_status": "not_required",
                "visual_evidence_notes": [
                    "候选框 1 与左侧肺野的上肺野二维投影相交。"
                ],
                **self.agent_response_overrides,
            }
            return _json_response(
                {
                    "response": agent_response,
                    "execution_receipt": receipt,
                    "execution_receipts": receipts,
                    "execution_plan": self.agent_execution_plan,
                    "reflection": {
                        "triggered": True,
                        "message": "internal fallback reflection",
                    },
                    "agent_trace": {
                        "reflection": {
                            "triggered": True,
                            "message": "internal fallback reflection",
                        },
                        "steps": [
                            {
                                "phase": "context",
                                "label": "构建最小必要上下文",
                                "status": "completed",
                            },
                            {
                                "phase": "tool",
                                "tool_name": self.agent_tool_name,
                                "label": "读取当前病例的结构化影像证据",
                                "status": "completed",
                            },
                            {
                                "phase": "verify",
                                "label": "校验工具契约、证据与安全边界",
                                "status": "completed",
                            },
                            {
                                "phase": "compose",
                                "label": "生成受证据约束的回答",
                                "status": "completed",
                            },
                        ]
                    },
                }
            )
        if normalized_method == "GET" and path == "/v1/cases/case-contract":
            return _json_response(
                self.case_read_payload or self.classified_case_payload
            )
        if normalized_method == "POST" and path == "/v1/llm/connections":
            return _json_response(
                {
                    "connection_id": "llmc-ui-contract",
                    "provider": "openai_compatible",
                    "base_url": kwargs["json"]["base_url"],
                    "model": kwargs["json"]["model"],
                    "created_at": "2026-08-31T00:00:00+00:00",
                    "expires_at": "2026-08-31T00:30:00+00:00",
                    "credential_persistence": "process_memory_only",
                }
            )
        if normalized_method == "DELETE" and path == "/v1/llm/connections/llmc-ui-contract":
            return _json_response({"revoked": True})
        if normalized_method == "POST" and re.fullmatch(
            r"/v1/batches/[^/]+/assessments/cxr", path
        ):
            batch_id = path.split("/")[3]
            assessment = _assessment_payload()
            review = (
                {
                    "review_id": "review-ui-contract",
                    "case_id": "case-contract",
                    "origin": "batch_screening",
                    "batch_id": batch_id,
                    "batch_item_id": kwargs["data"]["batch_item_id"],
                    "trigger_reasons": ["batch_model_flagged"],
                    "status": "pending",
                    "version": 1,
                }
                if self.batch_enqueue
                else None
            )
            return _json_response(
                {
                    "batch_id": batch_id,
                    "batch_item_id": kwargs["data"]["batch_item_id"],
                    "enqueued_for_review": self.batch_enqueue,
                    **assessment,
                    "review": review,
                }
            )
        if normalized_method == "POST" and path == "/v1/screening/sessions":
            self.screening_case_id = kwargs["json"].get("case_id")
            return _json_response(_screening_payload(case_id=self.screening_case_id))
        if (
            normalized_method == "POST"
            and path == "/v1/screening/sessions/screening-contract/answers"
        ):
            answer = kwargs["json"]["answer"]
            self.screening_answers.append(answer)
            return _json_response(
                _screening_payload(
                    answers={"persistent_cough": answer},
                    question_id="age_group",
                    question_text="请选择年龄分组",
                    answer_type="single_choice",
                    case_id=self.screening_case_id,
                    choices=["儿童", "成人"],
                )
            )
        if (
            normalized_method == "POST"
            and path == "/v1/screening/sessions/screening-contract/cancel"
        ):
            return _json_response(
                _screening_payload(status="cancelled", next_question=False)
            )
        if normalized_method == "POST" and path == "/v1/cases/case-contract/anatomy-runs":
            return _json_response(_anatomy_payload("pending"), status_code=202)
        if (
            normalized_method == "GET"
            and path == "/v1/cases/case-contract/anatomy-runs/anatomy-contract"
        ):
            self.anatomy_polls += 1
            return _json_response(
                _anatomy_payload(
                    self.anatomy_terminal_status,
                    refinement_status=self.anatomy_refinement_status,
                    capacity_abstained_count=self.anatomy_capacity_abstained_count,
                )
            )
        if (
            normalized_method == "GET"
            and path
            == "/v1/cases/case-contract/anatomy-runs/anatomy-contract/boundary.png"
        ):
            response = _bytes_response(_transparent_boundary(), content_type="image/png")
            response.headers["X-Anatomy-Routing-Effect"] = "none"
            return response
        if (
            normalized_method == "GET"
            and path
            == "/v1/cases/case-contract/anatomy-runs/anatomy-contract/contours.png"
        ):
            response = _bytes_response(_transparent_contours(), content_type="image/png")
            response.headers["X-Refinement-Routing-Effect"] = "none"
            response.headers["X-Clinical-Validation"] = "false"
            return response
        if normalized_method == "POST" and path == "/v1/cases/case-contract/reports":
            return _json_response(
                {
                    "report_id": "report-contract",
                    "markdown_download_url": "/v1/reports/report-contract/markdown",
                    "json_download_url": "/v1/reports/report-contract/json",
                }
            )
        if normalized_method == "GET" and path == "/v1/reviews/pending":
            return _json_response(self.pending_reviews)
        if normalized_method == "PATCH" and path.startswith("/v1/reviews/"):
            return _json_response(
                {
                    "review_id": path.rsplit("/", 1)[-1],
                    "status": "completed",
                    "version": kwargs["json"]["expected_version"] + 1,
                }
            )

        raise AssertionError(f"unexpected UI request: {normalized_method} {path}")

    def get(self, url: str, **kwargs) -> requests.Response:
        path = urlsplit(url).path
        self.calls.append(RecordedRequest("GET", path, kwargs))
        if path.endswith("/markdown"):
            return _bytes_response(b"# TBX-Agent report\n", content_type="text/markdown")
        if path.endswith("/json"):
            return _bytes_response(
                b'{"report_id":"report-contract"}', content_type="application/json"
            )
        raise AssertionError(f"unexpected UI artifact request: GET {path}")


@pytest.fixture()
def requests_stub(monkeypatch: pytest.MonkeyPatch) -> RequestsStub:
    stub = RequestsStub()
    monkeypatch.setattr(requests, "request", stub.request)
    monkeypatch.setattr(requests, "get", stub.get)
    return stub


def _run_app() -> AppTest:
    return AppTest.from_file(str(UI_ENTRYPOINT), default_timeout=15).run()


def _element_by_label(elements, label: str):
    matches = [element for element in elements if getattr(element, "label", None) == label]
    assert len(matches) == 1, f"expected one element labelled {label!r}, found {len(matches)}"
    return matches[0]


def _visible_text(app: AppTest) -> str:
    groups = (
        app.markdown,
        app.caption,
        app.warning,
        app.error,
        app.info,
        app.success,
        app.subheader,
        app.title,
    )
    return "\n".join(str(element.value) for group in groups for element in group)


def _png_bytes(color: tuple[int, int, int] = (32, 48, 64)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _assessment_payload(
    *,
    synthetic: bool = False,
    visual_result: str = "model_not_flagged",
) -> dict:
    classifier_model_id = (
        "mock-classifier" if synthetic else "tbx11k_rank03_dfineL_convnextT:convnext_tiny"
    )
    detector_model_id = (
        "mock-detector" if synthetic else "tbx11k_rank03_dfineL_convnextT:dfine_l"
    )
    return {
        "case": {
            "case_id": "case-contract",
            "review_status": "not_required",
            "classification_status": "completed",
            "vision_evidence": {
                "run_id": "run-contract",
                "class_probabilities": {
                    "healthy": 0.8,
                    "sick_non_tb": 0.15,
                    "tb": 0.05,
                },
                "classifier_probability_order": ["healthy", "sick_non_tb", "tb"],
                "predicted_class": "healthy",
                "classifier_decision_rule": "native_three_class_argmax",
                "classifier_model_id": classifier_model_id,
                "detector_model_id": detector_model_id,
                "detector_decision_role": "advisory_localization_only",
                "detections": [],
                "image_quality_status": "transport_valid",
                "runtime_ms": 1.0,
            },
            "localization_evidence": {
                "status": "not_requested",
                "detections": [],
            },
            "fusion_decision": {
                "visual_result": visual_result,
                "predicted_class": "healthy",
                "classifier_decision_rule": "native_three_class_argmax",
                "detector_decision_role": "advisory_localization_only",
                "policy_id": "test-policy",
                "clinical_validation": False,
            },
        },
        "response": {
            "summary": "结构化模型证据已返回。",
            "visual_result": visual_result,
            "predicted_class": "healthy",
            "review_status": "not_required",
            "limitations": ["不能用于确诊或排除肺结核。"],
        },
    }


def _upload_payload() -> dict:
    payload = _assessment_payload()
    case = payload["case"]
    case["classification_status"] = "not_requested"
    case["vision_evidence"] = None
    case["fusion_decision"] = None
    payload["response"] = {
        "summary": "胸片已载入，尚未运行分类。",
        "visual_result": None,
        "predicted_class": None,
        "review_status": "not_required",
        "limitations": ["不能用于确诊或排除肺结核。"],
    }
    return payload


def _screening_payload(
    *,
    status: str = "collecting",
    answers: dict[str, object] | None = None,
    question_id: str = "persistent_cough",
    question_text: str = "是否持续咳嗽？",
    answer_type: str = "boolean",
    choices: list[str] | None = None,
    next_question: bool = True,
    case_id: str | None = None,
    include_visual: bool = False,
) -> dict:
    response: dict[str, object] = {
        "summary": "主动筛查已开始。",
        "next_step_information": [],
    }
    if next_question:
        response["next_question"] = {
            "question_id": question_id,
            "text_zh": question_text,
            "answer_type": answer_type,
            "choices": choices or [],
        }
    if case_id and include_visual:
        response.update(
            {
                "case_id": case_id,
                "visual_result": "model_not_flagged",
                "predicted_class": "healthy",
                "review_status": "not_required",
                "visual_evidence_notes": ["当前胸片模型分流为健康样本训练类别。"],
            }
        )
    return {
        "session": {
            "session_id": "screening-contract",
            "status": status,
            "case_id": case_id,
            "answers": answers or {},
            "next_question_id": question_id if next_question else None,
        },
        "response": response,
    }


def _anatomy_payload(
    status: str,
    *,
    refinement_status: str = "disabled",
    capacity_abstained_count: int = 0,
) -> dict:
    payload = {
        "run_id": "anatomy-contract",
        "case_id": "case-contract",
        "status": status,
        "routing_effect": "none",
        "clinical_validation": False,
        "refinement_status": refinement_status,
    }
    if status in {"completed", "completed_with_refinement_failure"}:
        payload.update(
            {
                "detector_locations": [
                    {
                        "bbox_xyxy": [4.0, 4.0, 20.0, 20.0],
                        "status": "localized",
                        "note": "two_dimensional_lung_field_not_lobe",
                        "assignments": [
                            {
                                "lung": "left_lung",
                                "primary_zone": "upper_lung_field",
                                "zone_fractions": {
                                    "upper_lung_field": 1.0,
                                    "middle_lung_field": 0.0,
                                    "lower_lung_field": 0.0,
                                },
                                "intersection_pixels": 32,
                                "box_overlap_fraction": 0.5,
                                "lung_overlap_fraction": 0.05,
                            }
                        ],
                    }
                ],
                "spatial_summary": {
                    "policy_id": "detector-lung-field-presentation-v1",
                    "anatomy_qc_status": "pass",
                    "candidate_count": 1,
                    "localized_count": 1,
                    "outside_lungs_count": 0,
                    "invalid_anatomy_count": 0,
                    "statements": [
                        "候选框 1 与左侧肺野的上肺野二维投影相交。",
                        "肺野分割和候选定位不改变胸片三分类结果。",
                    ],
                    "routing_effect": "none",
                    "clinical_validation": False,
                },
            }
        )
    if refinement_status == "completed":
        payload["refinement_evidence"] = {
            "capacity_abstained_count": capacity_abstained_count,
            "routing_effect": "none",
            "clinical_validation": False,
        }
    elif refinement_status == "technical_failure":
        payload["refinement_error_code"] = "backend_unavailable"
    return payload


def _transparent_boundary() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(4, 28):
        image.putpixel((10, y), (25, 180, 170, 255))
        image.putpixel((22, y), (255, 160, 50, 255))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _transparent_contours() -> bytes:
    buffer = io.BytesIO()
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(12, 20):
        image.putpixel((x, 12), (255, 80, 90, 255))
        image.putpixel((x, 20), (255, 80, 90, 255))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _confirm_uploaded_image(app: AppTest) -> None:
    labels = {item.label for item in app.checkbox}
    assert "我确认该胸片已去除姓名、证件号等可识别信息" not in labels
    assert "我同意将该胸片发送给当前本地服务进行辅助筛查" not in labels


def _seed_matching_assessment(app: AppTest, payload: dict | None = None) -> None:
    image_bytes = _png_bytes()
    image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    app.session_state["assessment_image_bytes"] = image_bytes
    app.session_state["assessment_image_name"] = "cxr.png"
    app.session_state["current_image_sha256"] = image_sha256
    app.session_state["assessment_upload_sha256"] = image_sha256
    app.session_state["assessment_confirmation_sha256"] = image_sha256
    app.session_state["assessment_user_id"] = app.session_state["user_id"]
    app.session_state["assessment_owner_scope"] = app.session_state["owner_scope"]
    app.session_state["image_deidentified_attestation"] = True
    app.session_state["image_processing_consent"] = True
    app.session_state["case_id"] = "case-contract"
    app.session_state["assessment_result"] = payload or _assessment_payload()
    app.run()


def test_initial_workspace_renders_minimal_safe_navigation(requests_stub: RequestsStub) -> None:
    app = _run_app()

    assert not app.exception
    assert not app.tabs
    navigation = _element_by_label(app.radio, "导航")
    assert navigation.options == ["病例工作台", "病例详情", "批量筛查"]
    assert navigation.value == "病例工作台"

    visible_text = _visible_text(app)
    assert "TBX-Agent" in visible_text
    assert "肺结核胸片辅助筛查 · 非诊断" not in visible_text
    assert "Agent" in visible_text
    assert "Image" in visible_text
    assert "上传胸片后，在左侧输入问题即可开始真实推理。" in visible_text
    home_surface = visible_text + "\n" + "\n".join(
        repr(element.value) for element in app.dataframe
    )
    assert "三分类 argmax 分流；D-FINE 仅定位" not in home_surface
    assert "rank03" not in home_surface.casefold()
    assert "开始主动筛查" in [button.label for button in app.button]
    assert "运行辅助筛查" not in [button.label for button in app.button]
    assert not app.checkbox
    assert len(app.chat_input) == 1
    assert app.chat_input[0].value is None


def test_user_chat_message_is_anchored_right_with_right_side_avatar() -> None:
    css = UI_STYLESHEET.read_text(encoding="utf-8")
    user_message_rule = re.search(
        r'\[data-testid="stChatMessage"\]:has\('
        r'\[data-testid="stChatMessageAvatarUser"\]\)\s*\{(?P<body>.*?)\}',
        css,
        flags=re.DOTALL,
    )
    user_content_rule = re.search(
        r'\[data-testid="stChatMessage"\]:has\('
        r'\[data-testid="stChatMessageAvatarUser"\]\)\s*'
        r'\[data-testid="stChatMessageContent"\]\s*\{(?P<body>.*?)\}',
        css,
        flags=re.DOTALL,
    )

    assert user_message_rule is not None
    assert user_content_rule is not None
    assert "flex-direction: row-reverse" in user_message_rule.group("body")
    assert "justify-content: flex-start" in user_message_rule.group("body")
    assert "margin-left: auto" in user_content_rule.group("body")
    assert "width: auto !important" in user_content_rule.group("body")
    assert "text-align: left" in user_content_rule.group("body")


def test_general_chat_remains_enabled_when_vision_is_unavailable(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.vision_component_state = "unavailable"
    requests_stub.agent_execution_receipts = []
    requests_stub.agent_execution_plan = {"tool_names": []}
    requests_stub.agent_response_overrides = {
        "response_kind": "general_answer",
        "summary": "2",
    }

    app = _run_app()

    assert not app.exception
    assert app.chat_input[0].disabled is False
    assert "通用问答仍可使用" in _visible_text(app)
    app.chat_input[0].set_value("1+1 = ？").run()
    agent_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path == "/v1/agent/respond"
    ]
    assert len(agent_calls) == 1
    assert agent_calls[0].kwargs["json"]["message"] == "1+1 = ？"
    assert not [
        call for call in requests_stub.calls if call.path == "/v1/assessments/cxr"
    ]

    primary_markdown = "\n".join(
        element.value
        for element in app.markdown
        if "<style>" not in element.value
        and "tbx-sidebar" not in element.value
        and not element.value.startswith("#####")
    )
    primary_plain_text = re.sub(r"<[^>]+>", " ", primary_markdown)
    assert len(re.sub(r"\s+", "", primary_plain_text)) <= 80


def test_settings_switches_agent_turn_to_ephemeral_openai_compatible_connection(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    _element_by_label(app.button, "⚙ 设置").click().run()
    provider = _element_by_label(app.radio, "回答模型")
    assert provider.value == "local_medgemma"
    provider.set_value("openai_compatible").run()

    _element_by_label(app.text_input, "模型").set_value("remote-instruct")
    _element_by_label(app.text_input, "API Key").set_value("test-ui-secret")
    _element_by_label(app.button, "连接并测试").click().run()
    assert not app.exception
    assert app.session_state["llm_provider"] == "openai_compatible"
    assert app.session_state["llm_connection_id"] == "llmc-ui-contract"
    assert "test-ui-secret" not in repr(dict(app.session_state.filtered_state))

    connection_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path == "/v1/llm/connections"
    ]
    assert len(connection_calls) == 1
    assert connection_calls[0].kwargs["json"] == {
        "thread_id": app.session_state["thread_id"],
        "user_id": app.session_state["user_id"],
        "owner_scope": app.session_state["owner_scope"],
        "base_url": "https://api.openai.com/v1",
        "model": "remote-instruct",
        "api_key": "test-ui-secret",
    }

    app.chat_input[0].set_value("需要做什么检查？").run()
    agent_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path == "/v1/agent/respond"
    ]
    assert agent_calls[-1].kwargs["json"]["llm_provider"] == "openai_compatible"
    assert agent_calls[-1].kwargs["json"]["llm_connection_id"] == "llmc-ui-contract"
    assert "api_key" not in agent_calls[-1].kwargs["json"]


def test_batch_screening_is_the_only_path_that_reveals_review_workspace(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.batch_enqueue = True
    app = _run_app()
    navigation = _element_by_label(app.radio, "导航")
    assert "复核工作台" not in navigation.options
    navigation.set_value("批量筛查").run()

    uploader = _element_by_label(app.file_uploader, "上传胸片或 DICOM")
    uploader.upload("one.png", _png_bytes(), "image/png")
    uploader.upload("two.png", _png_bytes((48, 56, 64)), "image/png").run()
    _element_by_label(app.button, "开始批量筛查").click().run()
    assert not app.exception

    batch_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and "/v1/batches/" in call.path
    ]
    assert len(batch_calls) == 2
    assert all(call.path.endswith("/assessments/cxr") for call in batch_calls)
    assert all(call.kwargs["data"]["consent_to_process"] == "true" for call in batch_calls)
    assert "复核工作台" in _element_by_label(app.radio, "导航").options
    assert "打开本批次复核工作台" in [button.label for button in app.button]


def test_image_submission_requires_no_extra_confirmation_clicks(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()

    chat_input = app.chat_input[0]
    assert chat_input.disabled is False
    _confirm_uploaded_image(app)
    assert not [call for call in requests_stub.calls if call.path == "/v1/assessments/cxr"]


def test_active_screening_runs_inside_chat_with_direct_typed_answers(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()

    navigation = _element_by_label(app.radio, "导航")
    assert "主动筛查" not in navigation.options
    assert not app.checkbox

    _element_by_label(app.button, "开始主动筛查").click().run()

    assert not app.exception
    assert app.session_state["screening_session"]["status"] == "collecting"
    assert app.chat_input[0].disabled is True
    assert not app.checkbox
    assert "是否持续咳嗽？" in _visible_text(app)
    start_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path == "/v1/screening/sessions"
    ]
    assert len(start_calls) == 1
    assert start_calls[0].kwargs["json"] == {
        "thread_id": app.session_state["thread_id"],
        "user_id": app.session_state["user_id"],
        "owner_scope": app.session_state["owner_scope"],
        "case_id": None,
        "consent": True,
    }

    _element_by_label(app.button, "是").click().run()

    assert not app.exception
    assert requests_stub.screening_answers == [True]
    answer_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST"
        and call.path == "/v1/screening/sessions/screening-contract/answers"
    ]
    assert len(answer_calls) == 1
    assert answer_calls[0].kwargs["json"] == {
        "user_id": app.session_state["user_id"],
        "owner_scope": app.session_state["owner_scope"],
        "question_id": "persistent_cough",
        "answer": True,
    }
    assert app.session_state["screening_session"]["answers"] == {
        "persistent_cough": True
    }
    assert app.chat_input[0].disabled is True
    assert "请选择年龄分组" in _visible_text(app)

    _element_by_label(app.button, "退出主动筛查").click().run()

    assert not app.exception
    assert "screening_session" not in app.session_state
    assert app.chat_input[0].disabled is False
    cancel_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST"
        and call.path == "/v1/screening/sessions/screening-contract/cancel"
    ]
    assert len(cancel_calls) == 1
    assert cancel_calls[0].kwargs["json"] == {
        "user_id": app.session_state["user_id"],
        "owner_scope": app.session_state["owner_scope"],
    }


def test_active_screening_registers_uploaded_xray_without_classifying_it(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()

    _element_by_label(app.button, "开始主动筛查").click().run()

    assert not app.exception
    workflow_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST"
        and call.path in {"/v1/assessments/cxr", "/v1/screening/sessions"}
    ]
    assert [call.path for call in workflow_calls] == [
        "/v1/assessments/cxr",
        "/v1/screening/sessions",
    ]
    assert workflow_calls[1].kwargs["json"]["case_id"] == "case-contract"
    assert app.session_state["case_id"] == "case-contract"
    assert app.session_state["screening_session"]["case_id"] == "case-contract"
    assert app.session_state["assessment_result"]["case"]["classification_status"] == (
        "not_requested"
    )
    assert "当前胸片 ·" not in _visible_text(app)


@pytest.mark.parametrize(
    ("visual_result", "class_name", "safe_message"),
    [
        ("model_flagged", "flagged", "建议结合病原学检查和专业人员判读"),
        ("model_not_flagged", "not-flagged", "分类器最高分对应非结核训练类别"),
        ("pending_human_review", "review", "单病例不会进入复核工作台"),
        ("technical_failure", "failure", "未形成可用模型结果"),
    ],
)
def test_visual_result_card_uses_explicit_non_diagnostic_semantics(
    requests_stub: RequestsStub,
    visual_result: str,
    class_name: str,
    safe_message: str,
) -> None:
    app = _run_app()
    _seed_matching_assessment(
        app,
        _assessment_payload(visual_result=visual_result),
    )

    main_text = _visible_text(app)
    assert "系统分流" not in main_text
    assert "TB 类别相对分数" not in main_text
    _element_by_label(app.radio, "导航").set_value("病例详情").run()

    visible_text = _visible_text(app)
    assert "真实运行时" in visible_text
    assert safe_message in visible_text
    assert "三分类模型输出" not in visible_text
    assert "softmax 分数" not in visible_text
    assert "80.0%" not in visible_text
    assert "15.0%" not in visible_text
    assert "5.0%" not in visible_text
    assert all(metric.label != "TB 类别相对分数" for metric in app.metric)
    assert all(expander.label != "查看精确模型输出" for expander in app.expander)
    result_markup = "\n".join(
        element.value
        for element in app.markdown
        if "tbx-result-banner" in element.value
        and "<style>" not in element.value
    )
    assert f"tbx-result-banner--{class_name}" in result_markup
    if visual_result == "model_not_flagged":
        assert "tbx-result-banner--flagged" not in result_markup


def test_unfinished_localization_notice_is_not_hidden_by_classification_card(
    requests_stub: RequestsStub,
) -> None:
    notice = "分类已完成。本轮尚未完成：候选区域标注。"
    requests_stub.agent_execution_plan = {
        "unfinished_evidence": ["localization"],
        "final_plan": {"steps": [
            {"evidence_need": "classification", "status": "completed"},
            {"evidence_need": "localization", "status": "pending"},
        ]},
    }
    requests_stub.agent_response_overrides = {"summary": notice}
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    app.chat_input[0].set_value("分析胸片").run()
    assert not app.exception
    assert notice in _visible_text(app)


def test_model_not_flagged_chat_result_is_one_plain_conclusion(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    app.chat_input[0].set_value("分析胸片").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "胸片分类" in visible_text
    assert app.session_state["assessment_result"]["case"]["classification_status"] == (
        "completed"
    )
    workflow_paths = [
        call.path
        for call in requests_stub.calls
        if call.path
        in {"/v1/assessments/cxr", "/v1/agent/respond", "/v1/cases/case-contract"}
    ]
    assert workflow_paths == [
        "/v1/assessments/cxr",
        "/v1/agent/respond",
        "/v1/cases/case-contract",
    ]
    result_markup = "\n".join(
        element.value
        for element in app.markdown
        if "tbx-compact-result" in element.value and "<style>" not in element.value
    )
    assert "本次模型未识别为结核类。" in result_markup
    assert result_markup.count("本次模型未识别为结核类。") == 1
    for verbose_warning in (
        "不等于胸片正常",
        "不能排除肺结核",
        "完整证据链",
        "单一症状、胸片、涂片或AI结果都不够",
    ):
        assert verbose_warning not in visible_text
    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图"]
    assert layer_control.value == "原图"
    assert "显示肺野图层" not in [button.label for button in app.button]


def test_compound_agent_answer_does_not_repeat_the_compact_classification_card(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_execution_receipts = [
        {"tool_name": "classify_current_cxr", "status": "succeeded", "attempt": 1},
        {
            "tool_name": "retrieve_diagnostic_guidance",
            "status": "succeeded",
            "attempt": 1,
        },
    ]
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()

    app.chat_input[0].set_value(
        "这张胸片有没有结核病？下一步做什么检查？"
    ).run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "胸片分类" in visible_text
    assert "检查与诊断信息" in visible_text
    assert "已结合当前胸片结果生成辅助解释。" in visible_text
    compact_cards = [
        element.value
        for element in app.markdown
        if "tbx-compact-result" in element.value and "<style>" not in element.value
    ]
    assert compact_cards == []


def test_zero_receipts_hide_tool_chain_and_ignore_internal_plan_shape(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "describe_agent_capabilities"
    requests_stub.agent_execution_receipts = []
    requests_stub.agent_execution_plan = {
        "tool_names": ["describe_agent_capabilities"],
        "steps": [
            {"phase": "context", "label": "构建最小必要上下文"},
            {"phase": "verify", "label": "校验工具契约、证据与安全边界"},
        ],
    }

    app = _run_app()
    app.chat_input[0].set_value("你能做什么？").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "已结合当前胸片结果生成辅助解释。" in visible_text
    assert "本轮工具链" not in visible_text
    assert "工具：能力说明" not in visible_text
    assert "构建最小必要上下文" not in visible_text
    assert "校验工具契约、证据与安全边界" not in visible_text
    assert "internal fallback reflection" not in visible_text


def test_tool_chain_uses_public_labels_from_execution_receipts_only(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_execution_receipts = [
        {"tool_name": "classify_current_cxr", "status": "succeeded", "attempt": 1},
        {
            "tool_name": "inspect_anatomical_context",
            "status": "succeeded",
            "attempt": 1,
        },
        {
            "tool_name": "describe_agent_capabilities",
            "status": "succeeded",
            "attempt": 1,
        },
    ]

    app = _run_app()
    app.chat_input[0].set_value("说明可用分析").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "本轮工具链" in visible_text
    assert "胸片分类" in visible_text
    assert "肺野结构分析" in visible_text
    assert "工具：能力说明" not in visible_text


def test_plan_react_chat_shows_only_relevant_initial_plan_objectives(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_execution_receipts = [
        {
            "model_tool_name": "search_tb_knowledge",
            "tool_name": "search_tb_knowledge",
            "status": "succeeded",
            "attempt": 1,
        }
    ]
    requests_stub.agent_execution_plan = {
        "source": "plan_react",
        "initial_plan": {
            "goal": "解释痰涂片阴性结果",
            "steps": [
                {
                    "id": "p1",
                    "objective": "检索痰涂片阴性的结核知识",
                    "evidence_need": "tb_knowledge",
                    "status": "pending",
                },
                {
                    "id": "p2",
                    "objective": "直接回答当前问题",
                    "evidence_need": "none",
                    "status": "pending",
                },
                {
                    "id": "p3",
                    "objective": "构建最小必要上下文",
                    "evidence_need": "none",
                    "status": "pending",
                },
            ],
        },
        "hidden_reasoning_persisted": False,
    }
    requests_stub.agent_response_overrides = {
        "summary": "一次痰涂片阴性不能排除肺结核。",
        "answer_status": "ANSWERED",
        "guideline_subtopic": "negative_test_interpretation",
        "claims": [],
        "visual_evidence_notes": [],
    }

    app = _run_app()
    app.chat_input[0].set_value("痰涂片阴性能排除肺结核吗？").run()

    assert not app.exception
    plan_markup = "\n".join(
        item.value for item in app.markdown if "tbx-turn-plan" in item.value
    )
    assert "检索痰涂片阴性的结核知识" in plan_markup
    assert "直接回答当前问题" not in plan_markup
    assert "构建最小必要上下文" not in plan_markup
    assert "hidden_reasoning" not in plan_markup
    tool_markup = "\n".join(
        item.value for item in app.markdown if "tbx-tool-strip" in item.value
    )
    assert "结核知识检索 · 阴性结果解释" in tool_markup
    assert "search_tb_knowledge" not in tool_markup


def test_chat_hides_serialized_agent_state_from_answer_and_plan(
    requests_stub: RequestsStub,
) -> None:
    raw_state = json.dumps(
        {
            "allowed_tools": ["classify_cxr"],
            "allowed_tools_this_step": ["search_tb_knowledge"],
            "case_state": {"case_id": "case-secret"},
            "observations": [{"status": "internal"}],
            "tool_calls": [{"name": "search_tb_knowledge"}],
        },
        ensure_ascii=False,
    )
    requests_stub.agent_execution_receipts = []
    requests_stub.agent_execution_plan = {
        "source": "plan_react",
        "initial_plan": {
            "steps": [
                {"id": "p1", "objective": raw_state, "status": "pending"},
                {"id": "p2", "objective": "概括当前问题", "status": "pending"},
            ]
        },
    }
    requests_stub.agent_response_overrides = {
        "summary": raw_state,
        "visual_evidence_notes": [],
    }

    app = _run_app()
    app.chat_input[0].set_value("请回答当前问题").run()

    assert not app.exception
    rendered_surface = _visible_text(app)
    for internal_field in (
        "allowed_tools",
        "allowed_tools_this_step",
        "case_state",
        "observations",
        "tool_calls",
        "case-secret",
    ):
        assert internal_field not in rendered_surface
    assert "概括当前问题" in rendered_surface
    assert "本轮回答未形成可安全展示的文本，请重试。" in rendered_surface


def test_chat_replaces_stale_plan_instead_of_replaying_plan_history(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_execution_receipts = []
    requests_stub.agent_execution_plan = {
        "source": "plan_react",
        "initial_plan": {
            "steps": [
                {"id": "first", "objective": "核对第一轮问题", "status": "pending"},
                {"id": "first-duplicate", "objective": "核对第一轮问题", "status": "pending"},
            ]
        },
    }
    requests_stub.agent_response_overrides = {"summary": "第一轮回答。"}

    app = _run_app()
    app.chat_input[0].set_value("第一轮").run()
    first_plan_markup = "\n".join(
        item.value
        for item in app.markdown
        if "tbx-turn-plan" in item.value and "<style>" not in item.value
    )
    assert first_plan_markup.count("核对第一轮问题") == 1

    requests_stub.agent_execution_plan = {
        "source": "plan_react",
        "initial_plan": {
            "steps": [{"id": "second", "objective": "处理第二轮追问", "status": "pending"}]
        },
    }
    requests_stub.agent_response_overrides = {"summary": "第二轮回答。"}
    app.chat_input[0].set_value("第二轮").run()

    assert not app.exception
    current_plan_markup = "\n".join(
        item.value
        for item in app.markdown
        if "tbx-turn-plan" in item.value and "<style>" not in item.value
    )
    assert "处理第二轮追问" in current_plan_markup
    assert "核对第一轮问题" not in current_plan_markup
    assert current_plan_markup.count("tbx-turn-plan") == 1


def test_tool_badge_prefers_public_model_tool_name(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_execution_receipts = [
        {
            "model_tool_name": "localize_cxr",
            "tool_name": "classify_current_cxr",
            "status": "succeeded",
            "attempt": 1,
        }
    ]

    app = _run_app()
    app.chat_input[0].set_value("显示本轮工具").run()

    assert not app.exception
    tool_markup = "\n".join(
        item.value for item in app.markdown if "tbx-tool-strip" in item.value
    )
    assert "候选区域定位" in tool_markup
    assert "胸片分类" not in tool_markup


def test_guideline_evidence_gap_receipt_is_shown_as_warning_not_success(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "search_tb_guidance"
    requests_stub.agent_execution_receipts = [
        {
            "tool_name": "search_tb_guidance",
            "status": "succeeded",
            "outcome": "evidence_gap",
            "attempt": 1,
        }
    ]
    requests_stub.agent_response_overrides = {
        "summary": "当前受审核知识库没有标准治疗方案与疗程的可引用条款。",
        "answer_status": "INSUFFICIENT_EVIDENCE",
        "guideline_subtopic": "standard_regimen_duration",
        "claims": [],
    }

    app = _run_app()
    app.chat_input[0].set_value("标准疗程大概是什么？").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "指南证据检索 · 标准疗程 · 证据不足" in visible_text
    tool_markup = "\n".join(
        element.value
        for element in app.markdown
        if "tbx-plan-step" in element.value and "<style>" not in element.value
    )
    assert "tbx-plan-step--evidence-gap" in tool_markup
    assert "tbx-plan-step--completed" not in tool_markup


def test_applied_common_knowledge_fallback_is_shown_as_successful_tool_use(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "search_tb_guidance"
    requests_stub.agent_execution_receipts = [
        {
            "tool_name": "search_tb_guidance",
            "status": "succeeded",
            "outcome": "evidence_gap",
            "attempt": 1,
        }
    ]
    requests_stub.agent_response_overrides = {
        "summary": (
            "肺结核主要通过空气传播。\n\n"
            "注：本轮未检索到可引用指南依据，以上为通用医学信息。"
        ),
        "answer_status": "INSUFFICIENT_EVIDENCE",
        "guideline_subtopic": "respiratory_protection",
        "claims": [],
        "citations": [],
        "narrator_policy_id": "tbx-medical-common-knowledge-card-v2",
        "narration_status": "applied",
        "narrator_generation_invoked": True,
    }

    app = _run_app()
    app.chat_input[0].set_value("结核病会传染吗？").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "指南检索未命中 · 通用医学信息" in visible_text
    assert "指南证据检索 · 呼吸防护 · 证据不足" not in visible_text
    assert "注：本轮未检索到可引用指南依据，以上为通用医学信息。" in visible_text
    tool_markup = "\n".join(
        element.value
        for element in app.markdown
        if "tbx-plan-step" in element.value and "<style>" not in element.value
    )
    assert "tbx-plan-step--completed" in tool_markup
    assert "tbx-plan-step--evidence-gap" not in tool_markup


def test_provider_failure_with_reviewed_card_remains_successful_tool_use(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "search_tb_guidance"
    requests_stub.agent_execution_receipts = [
        {
            "tool_name": "search_tb_guidance",
            "status": "succeeded",
            "outcome": "evidence_gap",
            "attempt": 1,
        }
    ]
    requests_stub.agent_response_overrides = {
        "summary": (
            "怀疑肺结核时可佩戴贴合良好的口罩并尽快接受评估。\n\n"
            "注：本轮未检索到可引用指南依据，以上为通用医学信息。"
        ),
        "answer_status": "INSUFFICIENT_EVIDENCE",
        "guideline_subtopic": "respiratory_protection",
        "claims": [],
        "citations": [],
        "narrator_policy_id": "tbx-medical-common-knowledge-card-v2",
        "narration_status": "fallback_error",
        "narrator_generation_invoked": True,
    }

    app = _run_app()
    app.chat_input[0].set_value("怀疑肺结核时需要戴口罩吗？").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert "指南检索未命中 · 通用医学信息" in visible_text
    assert "指南证据检索 · 呼吸防护 · 证据不足" not in visible_text


def test_chat_uses_selected_guideline_claims_instead_of_stale_expanded_lists(
    requests_stub: RequestsStub,
) -> None:
    selected = "病原学确认后可进行利福平耐药的快速分子检测。"
    requests_stub.agent_tool_name = "search_tb_guidance"
    requests_stub.agent_execution_receipts = [
        {
            "tool_name": "search_tb_guidance",
            "status": "succeeded",
            "outcome": "answer_ready",
            "attempt": 1,
        }
    ]
    requests_stub.agent_response_overrides = {
        "summary": selected,
        "answer_status": "ANSWERED",
        "guideline_subtopic": "rapid_molecular_diagnostics",
        "claims": [{"text": selected, "chunk_ids": ["selected-chunk"]}],
        "visual_evidence_notes": [],
        "diagnostic_information": ["未选择的检查信息。"],
        "next_step_information": [
            "未选择的初始检查。",
            selected,
            "未选择的特殊人群信息。",
        ],
        "treatment_education": ["未选择的治疗信息。"],
    }

    app = _run_app()
    app.chat_input[0].set_value("快速分子检测怎么用？").run()

    assert not app.exception
    visible_text = _visible_text(app)
    assert visible_text.count(selected) == 1
    for unselected in (
        "未选择的检查信息。",
        "未选择的初始检查。",
        "未选择的特殊人群信息。",
        "未选择的治疗信息。",
    ):
        assert unselected not in visible_text


def test_retrieved_passages_are_kept_in_a_closed_chat_popover(
    requests_stub: RequestsStub,
) -> None:
    evidence_text = "痰抗酸杆菌涂片阴性不能排除肺结核。"
    requests_stub.agent_tool_name = "search_tb_guidance"
    requests_stub.agent_response_overrides = {
        "summary": "一次痰涂片阴性不能排除肺结核。",
        "answer_status": "ANSWERED",
        "guideline_subtopic": "negative_test_interpretation",
        "claims": [{"text": evidence_text, "chunk_ids": ["smear-limit"]}],
        "retrieved_evidence": [
            {
                "chunk_id": "smear-limit",
                "text": evidence_text,
                "source_id": "ats-cdc-idsa-2017",
                "source": "ATS/CDC/IDSA 结核病诊断指南",
                "section": "Testing for TB disease",
                "locator": "AFB smear microscopy recommendation",
                "url": "https://www.idsociety.org/practice-guideline/diagnosis-of-tb/",
            }
        ],
        "visual_evidence_notes": [],
        "diagnostic_information": ["不应在主回答展开的诊断列表。"],
        "next_step_information": ["不应在主回答展开的下一步列表。"],
        "treatment_education": ["不应在主回答展开的治疗列表。"],
    }

    app = _run_app()
    app.chat_input[0].set_value("痰涂片阴性是否排除肺结核？").run()

    assert not app.exception
    evidence_popovers = [
        item
        for item in app.get("popover")
        if item.proto.popover.label == "查看检索内容"
    ]
    assert len(evidence_popovers) == 1
    assert evidence_popovers[0].proto.popover.open is False
    popover_markup = "\n".join(item.value for item in evidence_popovers[0].markdown)
    assert evidence_text in popover_markup
    assert "来源：ATS/CDC/IDSA 结核病诊断指南" in popover_markup
    assert "章节：Testing for TB disease" in popover_markup
    assert "位置：AFB smear microscopy recommendation" in popover_markup
    assert 'href="https://www.idsociety.org/practice-guideline/diagnosis-of-tb/"' in (
        popover_markup
    )
    assert sum(evidence_text in item.value for item in app.markdown) == 1
    visible_text = _visible_text(app)
    assert "一次痰涂片阴性不能排除肺结核。" in visible_text
    assert "不应在主回答展开的诊断列表。" not in visible_text
    assert "不应在主回答展开的下一步列表。" not in visible_text
    assert "不应在主回答展开的治疗列表。" not in visible_text


@pytest.mark.parametrize(
    "localization_status",
    ["not_requested", "completed_no_detection", "failed"],
)
def test_non_completed_localization_never_draws_legacy_detection_boxes(
    requests_stub: RequestsStub,
    localization_status: str,
) -> None:
    assessment = _assessment_payload()
    case = assessment["case"]
    # A stale legacy value must never be interpreted as current localization evidence.
    case["vision_evidence"]["detections"] = [
        {
            "bbox_xyxy": [4.0, 4.0, 20.0, 20.0],
            "score": 0.99,
            "label": "legacy_candidate",
        }
    ]
    case["localization_evidence"] = {
        "status": localization_status,
        "detections": [],
    }
    requests_stub.assessment_payload = assessment
    requests_stub.agent_execution_receipts = []

    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    app.chat_input[0].set_value("分析胸片").run()

    assert not app.exception
    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图"]
    assert layer_control.value == "原图"


def test_completed_localization_without_a_tool_receipt_stays_hidden(
    requests_stub: RequestsStub,
) -> None:
    assessment = _assessment_payload()
    assessment["case"]["localization_evidence"] = {
        "status": "completed",
        "run_id": "localization-contract",
        "detections": [
            {
                "bbox_xyxy": [4.0, 4.0, 20.0, 20.0],
                "score": 0.88,
                "label": "tb_suspicious_region",
            }
        ],
    }
    requests_stub.assessment_payload = assessment
    requests_stub.agent_execution_receipts = []

    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    app.chat_input[0].set_value("分析胸片").run()

    assert not app.exception
    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图"]
    assert layer_control.value == "原图"


def test_anatomy_artifact_without_a_tool_receipt_stays_hidden(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    _seed_matching_assessment(app)
    app.session_state["anatomy_boundary_bytes"] = _transparent_boundary()
    app.session_state["anatomy_case_id"] = "case-contract"
    app.session_state["anatomy_image_sha256"] = app.session_state[
        "current_image_sha256"
    ]
    app.run()

    assert not app.exception
    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图"]
    assert layer_control.value == "原图"


def test_uploaded_image_is_registered_then_sent_to_agent_with_complete_context(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "localize_current_cxr"
    refreshed_case = json.loads(json.dumps(requests_stub.assessment_payload["case"]))
    refreshed_case["localization_evidence"] = {
        "status": "completed",
        "run_id": "localization-contract",
        "detections": [
            {
                "bbox_xyxy": [4.0, 4.0, 20.0, 20.0],
                "score": 0.88,
                "label": "tb_suspicious_region",
            }
        ],
    }
    requests_stub.case_read_payload = refreshed_case
    app = _run_app()
    expected_user_id = app.session_state["user_id"]
    expected_owner_scope = app.session_state["owner_scope"]
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    assert app.session_state["assessment_image_bytes"] == _png_bytes()
    assert app.session_state["assessment_image_name"] == "cxr.png"
    assert app.session_state["current_image_sha256"]
    expected_thread_id = app.session_state["thread_id"]
    assert app.chat_input[0].disabled is False
    _confirm_uploaded_image(app)

    app.chat_input[0].set_value("病灶在哪里？").run()
    assert not app.exception
    visible_text = _visible_text(app)
    assert "本轮工具链" in visible_text
    assert "候选区域定位" in visible_text
    assert "工具：候选区域定位" not in visible_text
    assert "工具：能力说明" not in visible_text
    assert "影像依据" in visible_text
    assert "候选框 1 与左侧肺野的上肺野二维投影相交。" in visible_text
    evidence_sections = [
        element.value
        for element in app.markdown
        if "**影像依据**" in element.value
    ]
    assert len(evidence_sections) == 1
    assert "候选框 1 与左侧肺野的上肺野二维投影相交。" in evidence_sections[0]
    for internal_step in (
        "执行计划",
        "构建最小必要上下文",
        "检索诊断与进一步检查依据",
        "校验工具契约、证据与安全边界",
        "生成受证据约束的回答",
        "规划器本轮降级",
        "接下来",
        "将结合当前胸片模型结果进行问询",
        "完整证据见病例详情",
        "internal fallback reflection",
    ):
        assert internal_step not in visible_text
    assert "胸片输入与质量检查" not in visible_text
    assert "TBX-CXR Vision v1 三分类" not in visible_text
    assert "三分类 argmax 分流；D-FINE 仅定位" not in visible_text
    assert "rank03" not in visible_text.casefold()
    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图", "候选框"]
    assert layer_control.value == "候选框"
    assert not [call for call in requests_stub.calls if "/anatomy-runs" in call.path]

    workflow_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path in {"/v1/assessments/cxr", "/v1/agent/respond"}
    ]
    assert [call.path for call in workflow_calls] == [
        "/v1/assessments/cxr",
        "/v1/agent/respond",
    ]
    case_refresh_calls = [
        call
        for call in requests_stub.calls
        if call.method == "GET" and call.path == "/v1/cases/case-contract"
    ]
    assert len(case_refresh_calls) == 1
    assert case_refresh_calls[0].kwargs["params"] == {
        "owner_scope": expected_owner_scope,
        "user_id": expected_user_id,
    }
    assert app.session_state["assessment_result"]["case"]["localization_evidence"][
        "detections"
    ] == refreshed_case["localization_evidence"]["detections"]

    assessment_call, agent_call = workflow_calls
    assert assessment_call.kwargs["data"] == {
        "user_id": expected_user_id,
        "owner_scope": expected_owner_scope,
        "consent_to_process": "true",
        "attested_chest_radiograph": "true",
    }
    assert assessment_call.kwargs["data"]["consent_to_process"] == "true"
    assert assessment_call.kwargs["data"]["attested_chest_radiograph"] == "true"
    uploaded_name, uploaded_bytes, uploaded_mime = assessment_call.kwargs["files"]["file"]
    assert (uploaded_name, uploaded_bytes, uploaded_mime) == (
        "cxr.png",
        _png_bytes(),
        "image/png",
    )

    assert agent_call.kwargs["json"] == {
        "thread_id": expected_thread_id,
        "user_id": expected_user_id,
        "owner_scope": expected_owner_scope,
        "message": "病灶在哪里？",
        "case_id": "case-contract",
        "llm_provider": "local_medgemma",
    }
    assert app.session_state["case_id"] == "case-contract"
    assert (
        app.session_state["assessment_upload_sha256"] == app.session_state["current_image_sha256"]
    )


def test_mock_evidence_from_real_claiming_api_is_labeled_and_blocked(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.assessment_payload = _assessment_payload(synthetic=True)
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析胸片").run()

    workflow_calls = [
        call
        for call in requests_stub.calls
        if call.path in {"/v1/assessments/cxr", "/v1/agent/respond"}
    ]
    assert [call.path for call in workflow_calls] == ["/v1/assessments/cxr"]
    assert "case_id" not in app.session_state
    assert "assessment_confirmation_sha256" not in app.session_state
    visible_text = _visible_text(app)
    assert "本次模型未识别为结核类。" in visible_text
    assert "本次没有调用后续 Agent" in visible_text

    result_markup = "\n".join(
        element.value
        for element in app.markdown
        if "tbx-compact-result" in element.value
        and "<style>" not in element.value
    )
    assert "本次模型未识别为结核类。" in result_markup
    assert "rank03" not in result_markup
    assert "D-FINE" not in result_markup


def test_anatomy_receipt_loads_optional_layer_without_selecting_it(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "inspect_anatomical_context"
    app = _run_app()
    expected_user_id = app.session_state["user_id"]
    expected_owner_scope = app.session_state["owner_scope"]
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析肺野结构").run()
    original_assessment = app.session_state["assessment_result"]

    assert not app.exception
    assert app.session_state["anatomy_boundary_bytes"] == _transparent_boundary()
    assert app.session_state["anatomy_case_id"] == "case-contract"
    assert app.session_state["anatomy_technical_failure"] is False
    assert app.session_state["assessment_result"] == original_assessment
    assert "肺野结构分析" in _visible_text(app)

    layer_control = _element_by_label(app.radio, "影像显示")
    assert layer_control.options == ["原图", "肺野"]
    assert layer_control.value == "原图"
    assert not [item for item in app.expander if item.label == "候选框空间关系"]
    _element_by_label(app.radio, "导航").set_value("病例详情").run()
    assert "肺野分割与空间关系" in _visible_text(app)

    anatomy_calls = [call for call in requests_stub.calls if "/anatomy-runs" in call.path]
    assert [call.method for call in anatomy_calls] == ["POST", "GET", "GET"]
    expected_identity = {
        "owner_scope": expected_owner_scope,
        "user_id": expected_user_id,
    }
    assert anatomy_calls[0].kwargs["params"] == expected_identity
    assert anatomy_calls[0].kwargs["files"] == {
        "file": ("cxr.png", _png_bytes(), "image/png")
    }
    assert anatomy_calls[1].kwargs["params"] == expected_identity
    assert anatomy_calls[2].kwargs["params"] == {
        **expected_identity,
        "structure": "combined",
    }

    assessment_calls = [
        call for call in requests_stub.calls if call.path == "/v1/assessments/cxr"
    ]
    assert len(assessment_calls) == 1


def test_completed_refinement_adds_real_contour_layer_and_capacity_note(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "inspect_anatomical_context"
    requests_stub.anatomy_refinement_status = "completed"
    requests_stub.anatomy_capacity_abstained_count = 3
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析肺野结构").run()

    assert not app.exception
    assert app.session_state["anatomy_boundary_bytes"] == _transparent_boundary()
    assert app.session_state["anatomy_contour_bytes"] == _transparent_contours()
    assert _element_by_label(app.radio, "影像显示").options == [
        "原图",
        "肺野",
        "轮廓",
        "组合",
    ]
    assert _element_by_label(app.radio, "影像显示").value == "原图"
    assert "未经病灶分割临床验证" in _visible_text(app)
    assert "3 个候选框因单次处理上限未生成轮廓" in _visible_text(app)
    assert any(call.path.endswith("/contours.png") for call in requests_stub.calls)


def test_refinement_failure_keeps_lung_layer_as_partial_success(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.agent_tool_name = "inspect_anatomical_context"
    requests_stub.anatomy_terminal_status = "completed_with_refinement_failure"
    requests_stub.anatomy_refinement_status = "technical_failure"
    app = _run_app()
    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析肺野结构").run()

    assert not app.exception
    assert app.session_state["anatomy_boundary_bytes"] == _transparent_boundary()
    assert "anatomy_contour_bytes" not in app.session_state
    assert app.session_state["anatomy_technical_failure"] is False
    assert "肺野已完成；可选轮廓细化暂不可用" in _visible_text(app)
    assert not any(call.path.endswith("/contours.png") for call in requests_stub.calls)


def test_new_case_clears_uploader_and_current_image(requests_stub: RequestsStub) -> None:
    app = _run_app()
    original_generation = app.session_state["uploader_generation"]
    original_thread_id = app.session_state["thread_id"]

    app.file_uploader[0].upload("cxr.png", _png_bytes(), "image/png").run()
    assert app.file_uploader[0].value is not None
    assert "assessment_image_bytes" in app.session_state
    assert "current_image_sha256" in app.session_state

    _element_by_label(app.button, "新建病例").click().run()
    assert not app.exception
    assert app.session_state["uploader_generation"] == original_generation + 1
    assert app.session_state["thread_id"] != original_thread_id
    assert app.file_uploader[0].value is None
    for key in (
        "assessment_image_bytes",
        "assessment_image_name",
        "current_image_sha256",
        "assessment_result",
        "assessment_upload_sha256",
        "case_id",
    ):
        assert key not in app.session_state


def test_replacing_image_starts_fresh_case_and_resets_confirmations(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    app.file_uploader[0].upload("first.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析第一张胸片").run()
    original_thread_id = app.session_state["thread_id"]
    assert app.session_state["case_id"] == "case-contract"
    assert app.session_state["chat_history"]

    second_image = _png_bytes((120, 80, 40))
    app.file_uploader[0].upload("second.png", second_image, "image/png").run()

    assert app.session_state["assessment_image_bytes"] == second_image
    assert app.session_state["thread_id"] != original_thread_id
    assert app.session_state["chat_history"] == []
    assert "image_deidentified_attestation" not in app.session_state
    assert "image_processing_consent" not in app.session_state
    assert app.chat_input[0].disabled is False
    for key in (
        "case_id",
        "assessment_result",
        "assessment_upload_sha256",
        "assessment_confirmation_sha256",
        "generated_report",
        "anatomy_case_id",
    ):
        assert key not in app.session_state


def test_failed_reassessment_cannot_fall_back_to_previous_case(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    app.file_uploader[0].upload("first.png", _png_bytes(), "image/png").run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析第一张胸片").run()
    first_case_calls = len(
        [call for call in requests_stub.calls if call.path == "/v1/agent/respond"]
    )

    requests_stub.assessment_status_code = 503
    app.file_uploader[0].upload(
        "second.png",
        _png_bytes((120, 80, 40)),
        "image/png",
    ).run()
    _confirm_uploaded_image(app)
    app.chat_input[0].set_value("分析第二张胸片").run()

    assert "case_id" not in app.session_state
    assert "assessment_result" not in app.session_state
    assert "assessment_confirmation_sha256" not in app.session_state
    assert len([call for call in requests_stub.calls if call.path == "/v1/agent/respond"]) == (
        first_case_calls
    )
    assert "胸片载入未完成，本次没有调用后续 Agent" in _visible_text(app)


def test_rebuilding_identity_clears_case_image_and_review_state(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    _seed_matching_assessment(app)
    app.session_state["pending_reviews"] = [{"review_id": "review-old"}]
    previous_user_id = app.session_state["user_id"]

    _element_by_label(app.button, "重建演示身份").click().run()

    assert app.session_state["user_id"] != previous_user_id
    assert app.session_state["owner_scope"].endswith(app.session_state["user_id"])
    assert app.session_state["chat_history"] == []
    for key in (
        "case_id",
        "assessment_result",
        "assessment_image_bytes",
        "assessment_upload_sha256",
        "assessment_confirmation_sha256",
        "pending_reviews",
    ):
        assert key not in app.session_state


def test_local_demo_identity_fields_are_read_only(requests_stub: RequestsStub) -> None:
    app = _run_app()

    user_id = _element_by_label(app.text_input, "用户 ID")
    owner_scope = _element_by_label(app.text_input, "Owner scope")
    assert user_id.disabled is True
    assert owner_scope.disabled is True
    assert "普通用户不可编辑" in _visible_text(app)


def test_report_and_artifact_requests_include_complete_identity(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    expected_user_id = app.session_state["user_id"]
    expected_owner_scope = app.session_state["owner_scope"]
    _seed_matching_assessment(app)

    _element_by_label(app.radio, "导航").set_value("病例详情").run()

    _element_by_label(app.button, "生成报告文件").click().run()
    assert not app.exception

    report_calls = [
        call
        for call in requests_stub.calls
        if call.method == "POST" and call.path == "/v1/cases/case-contract/reports"
    ]
    assert len(report_calls) == 1
    assert report_calls[0].kwargs["json"] == {
        "owner_scope": expected_owner_scope,
        "user_id": expected_user_id,
        "actor_id": expected_user_id,
    }

    artifact_calls = [
        call for call in requests_stub.calls if call.path.startswith("/v1/reports/report-contract/")
    ]
    assert {call.path for call in artifact_calls} == {
        "/v1/reports/report-contract/markdown",
        "/v1/reports/report-contract/json",
    }
    for call in artifact_calls:
        assert call.kwargs["params"] == {
            "owner_scope": expected_owner_scope,
            "user_id": expected_user_id,
        }


def test_only_an_active_batch_exposes_review_workspace_with_complete_identity(
    requests_stub: RequestsStub,
) -> None:
    app = _run_app()
    expected_user_id = app.session_state["user_id"]
    expected_owner_scope = app.session_state["owner_scope"]
    assert "复核工作台" not in _element_by_label(app.radio, "导航").options

    requests_stub.pending_reviews = [
        {
            "review_id": "review-batch-ui",
            "case_id": "case-contract",
            "owner_scope": expected_owner_scope,
            "trigger_reasons": ["batch_non_tb_abnormal"],
            "origin": "batch_screening",
            "batch_id": "batch-ui",
            "batch_item_id": "item-ui",
            "status": "pending",
            "version": 1,
        }
    ]
    app.session_state["active_batch_id"] = "batch-ui"
    app.run()

    navigation = _element_by_label(app.radio, "导航")
    navigation.set_value("复核工作台").run()
    assert not app.exception
    assert not app.checkbox
    assert "批量筛查复核工作台" in _visible_text(app)
    assert any("item-ui" in expander.label for expander in app.expander)

    review_calls = [
        call
        for call in requests_stub.calls
        if call.method == "GET" and call.path == "/v1/reviews/pending"
    ]
    assert review_calls
    assert review_calls[-1].kwargs["params"] == {
        "owner_scope": expected_owner_scope,
        "user_id": expected_user_id,
    }


def test_real_mode_keeps_chat_disabled_when_language_model_is_unavailable(
    requests_stub: RequestsStub,
) -> None:
    requests_stub.llm_component_state = "unavailable"
    app = _run_app()
    assert not app.exception
    assert app.chat_input[0].disabled is True
    assert "DEMO / MOCK" not in _visible_text(app)
    assert "本地 MedGemma 未就绪" in _visible_text(app)
