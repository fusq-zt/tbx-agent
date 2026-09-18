from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from anatomy_client import AnatomyClientError, request_and_poll_anatomy
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from tbx_agent.product_identity import VISION_MODEL_DISPLAY_NAME
from tbx_agent.vision.display import DetectionDisplayPolicy, select_display_detections
from tbx_agent.vision.image_validator import ImageValidationError, validate_image

API_URL = os.getenv("TBX_AGENT_API_URL", "http://127.0.0.1:8000").rstrip("/")
UI_ROOT = Path(__file__).resolve().parent
UI_MAX_UPLOAD_BYTES = int(os.getenv("TBX_AGENT_UI_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
DETECTION_DISPLAY_POLICY = DetectionDisplayPolicy()
_MEDICAL_COMMON_KNOWLEDGE_POLICY_ID = "tbx-medical-common-knowledge-card-v2"

CLASS_LABELS = {
    "healthy": "健康样本训练类别",
    "sick_non_tb": "非结核病变样本训练类别",
    "tb": "结核样本训练类别",
}
DECISION_RULE_LABELS = {
    "native_three_class_argmax": "三分类最高类别分流",
    "legacy_p_tb_threshold": "旧版固定筛查规则",
    "p_tb_threshold": "固定筛查规则",
    "p_tb_gte_threshold": "固定筛查规则",
}
DETECTOR_ROLE_LABELS = {
    "advisory_localization_only": "仅作候选区域定位，不参与分类分流",
    "legacy_vote": "旧版投票策略（仅为兼容历史记录）",
}
COMPONENT_LABELS = {
    "rank03_image_assessment": VISION_MODEL_DISPLAY_NAME,
    "guideline_retrieval": "指南证据检索",
    "active_screening": "主动筛查问询",
    "case_storage": "病例与审计存储",
    "optional_narrator": "可选语言整理",
    "anatomy_segmentation": "可选肺野分割",
    "contour_refinement": "可选轮廓细化",
}

TOOL_COMPONENT_LABELS = {
    # Public Plan + ReAct capability names. Prefer these labels whenever a
    # receipt carries ``model_tool_name``; internal handler names below remain
    # only for compatibility and detail-page receipts.
    "classify_cxr": "胸片分类",
    "localize_cxr": "候选区域定位",
    "analyze_lung_anatomy": "肺野结构分析",
    "search_tb_knowledge": "结核知识检索",
    "emergency_triage": "急症升级",
    "get_exact_case_and_explain": "当前病例证据",
    "classify_current_cxr": "胸片分类",
    "localize_current_cxr": "候选区域定位",
    "inspect_anatomical_context": "肺野结构分析",
    "inspect_image_quality": "基础输入可用性检查",
    "compare_with_prior_cxr": "纵向对比可用性检查",
    "search_tb_guidance": "结核指南检索",
    "retrieve_guideline": "结核指南检索",
    "retrieve_treatment_education": "治疗教育边界",
    "retrieve_diagnostic_guidance": "检查与诊断信息",
}
VISUAL_RESULT_LABELS = {
    "model_flagged": "模型辅助筛查标记",
    "model_not_flagged": "未识别为结核类",
    "non_tb_abnormal": "非结核异常类别",
    "indeterminate": "结果不确定",
    "technical_failure": "技术处理失败",
    "pending_human_review": "结果不确定",
}
VISUAL_RESULT_PRESENTATION = {
    "model_flagged": {
        "class_name": "flagged",
        "icon": "!",
        "message": "模型发现需要进一步评估的信号，建议结合病原学检查和专业人员判读。",
    },
    "model_not_flagged": {
        "class_name": "not-flagged",
        "icon": "—",
        "message": "分类器最高分对应非结核训练类别。",
    },
    "non_tb_abnormal": {
        "class_name": "not-flagged",
        "icon": "—",
        "message": "分类器最高分对应非结核病变训练类别。",
    },
    "pending_human_review": {
        "class_name": "review",
        "icon": "?",
        "message": "请在当前对话中重新上传或补充信息；单病例不会进入复核工作台。",
    },
    "technical_failure": {
        "class_name": "failure",
        "icon": "×",
        "message": "本次影像处理未形成可用模型结果，请排查运行时或输入后重试。",
    },
    "indeterminate": {
        "class_name": "review",
        "icon": "?",
        "message": "当前证据不足以形成稳定分流结果，请在对话中补充信息或重试。",
    },
}
_UNSAFE_RESULT_MARKER = re.compile(r"\bTB\s+" + r"DETECTED\b", re.IGNORECASE)
_SYNTHETIC_EVIDENCE_MARKER = re.compile(r"(?:^|[^a-z0-9])(mock|synthetic)(?:[^a-z0-9]|$)")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _localization_evidence(case: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the explicit localization state for this exact case.

    Legacy ``vision_evidence.detections`` values are deliberately ignored.  They
    cannot distinguish a detector that was not requested from a detector that
    completed without a candidate, and therefore are not safe display evidence.
    """

    return _mapping(case.get("localization_evidence"))


def _completed_localization_view(case: Mapping[str, Any]) -> Mapping[str, Any]:
    """Build overlay input only for a completed localization execution."""

    localization = _localization_evidence(case)
    if str(localization.get("status") or "not_requested") != "completed":
        return {}
    detections = localization.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return {}
    vision = _mapping(case.get("vision_evidence"))
    return {
        "image_width": vision.get("image_width"),
        "image_height": vision.get("image_height"),
        "detections": detections,
    }


def _safe_text(value: Any) -> str:
    """Prevent a legacy upstream label from becoming a diagnosis-like UI statement."""

    text = str(value or "")
    return _UNSAFE_RESULT_MARKER.sub("模型辅助筛查提示需进一步评估", text)


_INTERNAL_AGENT_STATE_FIELD = re.compile(
    r"(?i)(?:[\"']?)(?:allowed_tools(?:_this_step)?|case_state|observations|"
    r"tool_calls)(?:[\"']?)\s*[:=]"
)


def _public_agent_text(value: Any) -> str:
    """Return model prose only, never a serialized internal Agent state.

    The full API payload remains in session history for the existing audit/detail
    paths.  This boundary is intentionally render-only: mappings, lists and JSON
    strings are not conversational copy, and state-shaped provider echoes must
    never be sent to Markdown.
    """

    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or _INTERNAL_AGENT_STATE_FIELD.search(text):
        return ""
    candidate = text
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 2:
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        structured = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        structured = None
    if isinstance(structured, (Mapping, list)):
        return ""
    return _safe_text(text)


def _short_hash(value: Any) -> str:
    text = str(value or "")
    return f"{text[:12]}…" if len(text) > 12 else text


def _visual_result_presentation(visual_result: Any) -> dict[str, str]:
    key = str(visual_result or "").strip().lower()
    configured = VISUAL_RESULT_PRESENTATION.get(key)
    if configured:
        return {"label": VISUAL_RESULT_LABELS[key], **configured}
    return {
        "label": VISUAL_RESULT_LABELS.get(key, "结果状态未知"),
        "class_name": "failure",
        "icon": "?",
        "message": "模型返回了未识别的状态；本次输出不得用于临床判断。",
    }


def _assessment_visual_result(assessment: Mapping[str, Any]) -> str:
    case = _mapping(assessment.get("case"))
    response = _mapping(assessment.get("response"))
    fusion = _mapping(case.get("fusion_decision"))
    return str(response.get("visual_result") or fusion.get("visual_result") or "")


def _has_classification_evidence(case: Mapping[str, Any]) -> bool:
    return bool(_mapping(case.get("vision_evidence")))


def _is_explicit_demo(health: Mapping[str, Any], capabilities: Mapping[str, Any]) -> bool:
    """Permit deterministic demo UX only for an explicitly synthetic healthy server."""
    return bool(
        health.get("status") == "ok"
        and health.get("vision_backend") == "mock"
        and health.get("mode") == "synthetic_demo"
        and health.get("deployment_profile") != "production"
        and capabilities.get("status") == "ok"
        and capabilities.get("mode") == "synthetic_demo"
        and any(
            _mapping(item).get("component_id") == "rank03_image_assessment"
            and _mapping(item).get("state") == "ready"
            and _mapping(item).get("synthetic") is True
            for item in capabilities.get("components") or []
        )
    )


def _demo_session_active() -> bool:
    snapshot = _mapping(st.session_state.get("system_snapshot"))
    return _is_explicit_demo(
        _mapping(snapshot.get("health")), _mapping(snapshot.get("capabilities"))
    )


def _assessment_runtime_kind(
    case: Mapping[str, Any],
    *,
    capability_synthetic: bool,
    runtime_verified: bool,
) -> str:
    """Resolve result provenance from both the capability contract and returned evidence.

    A service claiming a real backend is not sufficient: a mock run/model identifier always
    wins. Conversely, real product names are only shown when the capability snapshot is
    verified and the returned classifier identifier matches the deployed contract.
    Localization has its own execution identity and is not required for initial screening.
    """

    evidence = _mapping(case.get("vision_evidence"))
    evidence_ids = (
        str(evidence.get("run_id") or ""),
        str(evidence.get("classifier_model_id") or ""),
    )
    normalized_ids = tuple(value.strip().lower() for value in evidence_ids)
    if capability_synthetic or any(
        _SYNTHETIC_EVIDENCE_MARKER.search(value) for value in normalized_ids if value
    ):
        return "synthetic"

    classifier_id = normalized_ids[1]
    evidence_matches_real_contract = bool(
        normalized_ids[0]
        and "rank03" in classifier_id
    )
    if runtime_verified and evidence_matches_real_contract:
        return "real"
    return "unverified"


def _assessment_matches_workspace(
    assessment: Any,
    *,
    current_image_sha256: Any,
    assessment_image_sha256: Any,
    confirmation_image_sha256: Any,
    assessment_user_id: Any,
    assessment_owner_scope: Any,
    current_user_id: Any,
    current_owner_scope: Any,
) -> bool:
    """Return true only when result, image, confirmation and identity are the same scope."""

    current_hash = str(current_image_sha256 or "")
    return bool(
        isinstance(assessment, Mapping)
        and current_hash
        and current_hash == str(assessment_image_sha256 or "")
        and current_hash == str(confirmation_image_sha256 or "")
        and str(assessment_user_id or "") == str(current_user_id or "")
        and str(assessment_owner_scope or "") == str(current_owner_scope or "")
    )


def _status_label(value: Any) -> str:
    labels = {
        "ok": "可用",
        "completed": "已完成",
        "completed_no_detection": "已完成，无候选框",
        "complete": "已完成",
        "not_requested": "未调用",
        "unsupported": "当前不支持",
        "stale": "证据已过期",
        "collecting": "问询中",
        "pending": "待处理",
        "pending_human_review": "结果不确定",
        "running": "运行中",
        "technical_failure": "技术失败",
        "not_required": "未进入批量复核",
        "not_configured": "未启用",
        "disabled": "未启用",
        "fallback": "未完整执行",
        "fallback_error": "未完整执行",
        "rejected_by_safety": "未采用语言整理",
        "error": "失败",
        "failed": "失败",
        "warning": "需注意",
        "transport_valid": "传输与解码校验通过",
        "ready": "就绪",
        "configured_not_probed": "已配置（未探测）",
        "degraded": "降级可用",
        "unavailable": "不可用",
        "succeeded": "执行成功",
        "timed_out": "执行超时",
        "step_limit_exceeded": "超出步骤上限",
    }
    key = str(value or "").lower()
    return labels.get(key, str(value or "未知"))


def _escape(value: Any) -> str:
    return html.escape(_safe_text(value), quote=True)


def load_ui_styles() -> None:
    stylesheet = UI_ROOT / "assets" / "tbx_agent.css"
    try:
        css = stylesheet.read_text(encoding="utf-8")
    except OSError:
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_section_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="tbx-section">
          <div class="tbx-section__eyebrow">{_escape(eyebrow)}</div>
          <h2>{_escape(title)}</h2>
          <p>{_escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(_health: Mapping[str, Any], _manifest: Mapping[str, Any]) -> None:
    st.markdown(
        """
        <section class="tbx-minimal-header">
          <div class="tbx-brandline">
            <div class="tbx-brandmark">T</div>
            <div>
              <h1>TBX-Agent</h1>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _clear_case_state(*, keep_identity: bool = True) -> None:
    _cancel_current_screening(show_error=False)
    uploader_generation = int(st.session_state.get("uploader_generation", 0)) + 1
    keys = (
        "case_id",
        "assessment_result",
        "assessment_image_bytes",
        "assessment_preview_bytes",
        "assessment_image_name",
        "current_image_sha256",
        "assessment_upload_sha256",
        "assessment_confirmation_sha256",
        "assessment_user_id",
        "assessment_owner_scope",
        "image_processing_consent",
        "image_deidentified_attestation",
        "generated_report",
        "generated_report_markdown",
        "generated_report_json",
        "screening_session",
        "screening_response",
        "pending_reviews",
        "chat_history",
        "anatomy_run",
        "anatomy_boundary_bytes",
        "anatomy_contour_bytes",
        "anatomy_case_id",
        "anatomy_image_sha256",
        "anatomy_technical_failure",
        "anatomy_display_mode",
        "pending_anatomy_display_mode",
        "localization_receipt_case_id",
        "localization_receipt_image_sha256",
        "anatomy_receipt_case_id",
        "anatomy_receipt_image_sha256",
    )
    for key in keys:
        st.session_state.pop(key, None)
    st.session_state.uploader_generation = uploader_generation
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex}"
    st.session_state.chat_history = []
    if not keep_identity:
        user_id = f"local-user-{uuid.uuid4().hex[:8]}"
        st.session_state.user_id = user_id
        st.session_state.owner_scope = f"local-demo:{user_id}"


def _activate_uploaded_image(
    *,
    image_bytes: bytes,
    preview_bytes: bytes,
    image_name: str,
    image_sha256: str,
) -> None:
    """Bind a new upload to a fresh case/thread and remove every stale case output."""

    _cancel_current_screening(show_error=False)
    case_keys = (
        "case_id",
        "assessment_result",
        "assessment_upload_sha256",
        "assessment_confirmation_sha256",
        "assessment_user_id",
        "assessment_owner_scope",
        "generated_report",
        "generated_report_markdown",
        "generated_report_json",
        "screening_session",
        "screening_response",
        "anatomy_run",
        "anatomy_boundary_bytes",
        "anatomy_contour_bytes",
        "anatomy_case_id",
        "anatomy_image_sha256",
        "anatomy_technical_failure",
        "anatomy_display_mode",
        "pending_anatomy_display_mode",
        "localization_receipt_case_id",
        "localization_receipt_image_sha256",
        "anatomy_receipt_case_id",
        "anatomy_receipt_image_sha256",
    )
    for key in case_keys:
        st.session_state.pop(key, None)
    st.session_state.assessment_image_bytes = image_bytes
    st.session_state.assessment_preview_bytes = preview_bytes
    st.session_state.assessment_image_name = image_name
    st.session_state.current_image_sha256 = image_sha256
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex}"
    st.session_state.chat_history = []


def _clear_chat_state() -> None:
    _cancel_current_screening(show_error=False)
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex}"
    st.session_state.chat_history = []


def _assessment_matches_current_session(assessment: Any) -> bool:
    return _assessment_matches_workspace(
        assessment,
        current_image_sha256=st.session_state.get("current_image_sha256"),
        assessment_image_sha256=st.session_state.get("assessment_upload_sha256"),
        confirmation_image_sha256=st.session_state.get("assessment_confirmation_sha256"),
        assessment_user_id=st.session_state.get("assessment_user_id"),
        assessment_owner_scope=st.session_state.get("assessment_owner_scope"),
        current_user_id=st.session_state.get("user_id"),
        current_owner_scope=st.session_state.get("owner_scope"),
    )


def call(method: str, path: str, *, show_error: bool = True, **kwargs) -> Any:
    timeout = kwargs.pop("timeout", (5, 120))
    try:
        response = requests.request(method, API_URL + path, timeout=timeout, **kwargs)
    except requests.Timeout:
        if show_error:
            st.error("请求超时。后端可能仍在处理，请稍后查看病例状态，避免连续重复提交。")
        return None
    except requests.RequestException as exc:
        if show_error:
            st.error(f"无法连接 API：{exc}")
        return None
    if not response.ok:
        try:
            error_payload = response.json()
            detail = (
                error_payload.get("detail", response.text)
                if isinstance(error_payload, Mapping)
                else response.text
            )
        except ValueError:
            detail = response.text
        if show_error:
            request_id = response.headers.get("X-Request-ID")
            suffix = f" · 请求标识 {request_id}" if request_id else ""
            if response.status_code in {401, 403}:
                st.error(f"当前身份无权执行此操作{suffix}。")
            elif response.status_code == 422:
                st.error(f"提交信息不完整或格式不正确：{detail}{suffix}")
            elif response.status_code == 429:
                st.warning(f"请求过于频繁，请稍后重试{suffix}。")
            elif response.status_code >= 500:
                st.error(f"服务暂时不可用（{response.status_code}）{suffix}。")
            else:
                st.error(f"请求失败（{response.status_code}）：{detail}{suffix}")
        return None
    try:
        return response.json()
    except ValueError:
        if show_error:
            st.error("API 返回了无法解析的响应；本次结果未写入当前工作区。")
        return None


def _system_snapshot(*, max_age_seconds: float = 4.0) -> tuple[dict[str, Any], ...]:
    """Reuse short-lived read-only status calls across Streamlit widget reruns."""

    now = time.monotonic()
    cached = _mapping(st.session_state.get("system_snapshot"))
    cached_at = st.session_state.get("system_snapshot_cached_at")
    try:
        cache_age = now - float(cached_at)
    except (TypeError, ValueError):
        cache_age = max_age_seconds + 1.0
    if cached and 0.0 <= cache_age < max_age_seconds:
        return (
            dict(_mapping(cached.get("health"))),
            dict(_mapping(cached.get("capabilities"))),
            dict(_mapping(cached.get("manifest"))),
        )

    health = _mapping(call("GET", "/healthz", show_error=False, timeout=(2, 5)))
    capabilities = _mapping(
        call("GET", "/v1/system/capabilities", show_error=False, timeout=(2, 8))
        if health
        else None
    )
    manifest = _mapping(
        call("GET", "/v1/system/manifest", show_error=False, timeout=(2, 8))
        if health
        else None
    )
    snapshot = {
        "health": dict(health),
        "capabilities": dict(capabilities),
        "manifest": dict(manifest),
    }
    st.session_state.system_snapshot = snapshot
    st.session_state.system_snapshot_cached_at = now
    return snapshot["health"], snapshot["capabilities"], snapshot["manifest"]


def _remote_llm_connection_ready() -> bool:
    return bool(
        st.session_state.get("llm_connection_id")
        and st.session_state.get("llm_connection_thread_id")
        == st.session_state.get("thread_id")
    )


def _agent_llm_payload() -> dict[str, str]:
    provider = str(st.session_state.get("llm_provider") or "local_medgemma")
    if provider == "local_qwen":
        provider = "local_medgemma"
    payload = {"llm_provider": provider}
    if provider == "openai_compatible":
        connection_id = str(st.session_state.get("llm_connection_id") or "")
        if connection_id:
            payload["llm_connection_id"] = connection_id
    return payload


def _forget_remote_llm_connection(*, revoke: bool) -> None:
    connection_id = str(st.session_state.get("llm_connection_id") or "")
    connection_thread_id = str(st.session_state.get("llm_connection_thread_id") or "")
    if revoke and connection_id and connection_thread_id:
        call(
            "DELETE",
            f"/v1/llm/connections/{connection_id}",
            show_error=False,
            params={
                "thread_id": connection_thread_id,
                "user_id": st.session_state.get("user_id"),
                "owner_scope": st.session_state.get("owner_scope"),
            },
            timeout=(2, 8),
        )
    for key in (
        "llm_connection_id",
        "llm_connection_thread_id",
        "llm_connection_model",
        "llm_connection_base_url",
        "llm_connection_expires_at",
    ):
        st.session_state.pop(key, None)


def _render_llm_settings_body(*, local_ready: bool, service_available: bool) -> None:
    """Configure the answer model inside the lightweight settings popover."""

    labels = {
        "local_medgemma": "本地 MedGemma 1.5 4B · Q4_K_M",
        "openai_compatible": "OpenAI 兼容 API",
    }
    selected = st.radio(
        "回答模型",
        list(labels),
        format_func=labels.get,
        key="llm-provider-selection",
        horizontal=True,
    )
    active = str(st.session_state.get("llm_provider") or "local_medgemma")
    if active == "local_qwen":
        active = "local_medgemma"

    if selected == "local_medgemma":
        if _demo_session_active() and not local_ready and active == "local_medgemma":
            st.info("演示使用确定性回答，无需连接语言模型。")
            return
        if local_ready:
            st.success("本地模型已就绪")
        else:
            st.warning("本地模型未就绪")
        if active == "local_medgemma":
            st.caption("当前由本地 MedGemma 1.5 4B · Q4_K_M 生成回答。")
            return
        if st.button(
            "切换到本地模型",
            key="activate-local-medgemma",
            type="primary",
            use_container_width=True,
            disabled=not local_ready,
        ):
            _forget_remote_llm_connection(revoke=True)
            st.session_state.llm_provider = "local_medgemma"
            st.rerun(scope="app")
        return

    if active == "openai_compatible" and _remote_llm_connection_ready():
        model = str(st.session_state.get("llm_connection_model") or "已连接模型")
        st.success(f"{model} 已连接")
        st.caption("密钥仅保存在 API 进程内存中，到期或重启后需重连。")
        if st.button(
            "断开并切回本地模型",
            key="disconnect-openai-compatible",
            use_container_width=True,
        ):
            _forget_remote_llm_connection(revoke=True)
            st.session_state.llm_provider = "local_medgemma"
            st.rerun(scope="app")
        return

    if st.session_state.get("llm_connection_id"):
        _forget_remote_llm_connection(revoke=True)
    st.caption("连接测试通过后，Agent 才会切换到该接口。")
    with st.form("openai-compatible-connection", clear_on_submit=True, border=False):
        base_url = st.text_input(
            "API 地址",
            value="https://api.openai.com/v1",
            placeholder="https://host.example/v1",
        )
        model = st.text_input("模型", placeholder="例如 gpt-4.1-mini")
        credential_input = st.text_input("API Key", type="password")
        submitted = st.form_submit_button(
            "连接并测试",
            use_container_width=True,
            disabled=not service_available,
        )
    if submitted:
        connection = call(
            "POST",
            "/v1/llm/connections",
            json={
                "thread_id": st.session_state.thread_id,
                "user_id": st.session_state.user_id,
                "owner_scope": st.session_state.owner_scope,
                "base_url": base_url,
                "model": model,
                "api_key": credential_input,
            },
            timeout=(5, 120),
        )
        connection_payload = _mapping(connection)
        if connection_payload.get("connection_id"):
            st.session_state.llm_connection_id = str(connection_payload["connection_id"])
            st.session_state.llm_connection_thread_id = st.session_state.thread_id
            st.session_state.llm_connection_model = str(
                connection_payload.get("model") or model
            )
            st.session_state.llm_connection_base_url = str(
                connection_payload.get("base_url") or base_url
            )
            st.session_state.llm_connection_expires_at = str(
                connection_payload.get("expires_at") or ""
            )
            st.session_state.llm_provider = "openai_compatible"
            st.rerun(scope="app")


def render_llm_settings(*, local_ready: bool, service_available: bool) -> None:
    """Render a persistent overlay controlled by one state callback."""

    if "llm_settings_open" not in st.session_state:
        st.session_state.llm_settings_open = False

    def _toggle_settings() -> None:
        st.session_state.llm_settings_open = not bool(
            st.session_state.llm_settings_open
        )

    def _close_settings() -> None:
        st.session_state.llm_settings_open = False

    st.button(
        "⚙ 设置",
        key="open-llm-settings",
        use_container_width=True,
        on_click=_toggle_settings,
    )
    if not st.session_state.llm_settings_open:
        return

    with st.container(border=True, key="llm-settings-drawer"):
        title_column, close_column = st.columns([5, 1], vertical_alignment="center")
        title_column.markdown("**模型设置**")
        close_column.button(
            "关闭",
            key="close-llm-settings",
            on_click=_close_settings,
        )
        _render_llm_settings_body(
            local_ready=local_ready,
            service_available=service_available,
        )


def _cancel_current_screening(*, show_error: bool) -> None:
    """Close a collecting server-side questionnaire before discarding local state."""

    session = _mapping(st.session_state.get("screening_session"))
    session_id = str(session.get("session_id") or "")
    if not session_id or session.get("status") != "collecting":
        return
    call(
        "POST",
        f"/v1/screening/sessions/{session_id}/cancel",
        show_error=show_error,
        json={
            "user_id": st.session_state.get("user_id"),
            "owner_scope": st.session_state.get("owner_scope"),
        },
    )


def fetch_artifact(path: str) -> bytes | None:
    try:
        response = requests.get(
            API_URL + path,
            params={
                "owner_scope": st.session_state.owner_scope,
                "user_id": st.session_state.user_id,
            },
            timeout=(5, 30),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"报告文件读取失败：{exc}")
        return None
    return response.content


def show_response(payload: dict | None, *, show_summary: bool = True) -> None:
    if not payload:
        return
    response = _mapping(payload.get("response", payload))
    receipts = [
        receipt
        for receipt in _execution_receipts(payload)
        if (
            _receipt_public_tool_name(receipt) in TOOL_COMPONENT_LABELS
            or str(receipt.get("tool_name") or "") in TOOL_COMPONENT_LABELS
        )
    ]
    urgency = response.get("urgency")
    raw_summary = response.get("summary", "")
    summary = _public_agent_text(raw_summary)
    visual_result = response.get("visual_result")
    if show_summary and summary:
        if urgency == "emergency" or visual_result == "technical_failure":
            st.error(summary)
        elif urgency in {"prompt_evaluation", "priority_screening"} or visual_result in {
            "model_flagged",
            "pending_human_review",
            "indeterminate",
        }:
            st.warning(summary)
        else:
            st.info(summary)
    elif show_summary and raw_summary:
        st.warning("本轮回答未形成可安全展示的文本，请重试。")

    execution_items = []
    if response.get("request_id"):
        execution_items.append(f"请求 {response['request_id']}")
    if response.get("narration_status"):
        execution_items.append(f"语言整理：{_status_label(response['narration_status'])}")
    for receipt in receipts:
        public_tool_name = _receipt_public_tool_name(receipt)
        internal_tool_name = str(receipt.get("tool_name") or "unknown")
        label = TOOL_COMPONENT_LABELS.get(public_tool_name) or TOOL_COMPONENT_LABELS.get(
            internal_tool_name,
            "工具",
        )
        execution_items.append(
            f"{label}：{_status_label(receipt.get('status'))}"
        )
    if execution_items:
        st.caption(" · ".join(execution_items))

    if receipts:
        with st.expander("工具执行回执"):
            st.json(
                [
                    {
                        "call_id": receipt.get("call_id"),
                        "model_tool_name": receipt.get("model_tool_name"),
                        "tool_name": receipt.get("tool_name"),
                        "routing_policy_id": receipt.get("routing_policy_id"),
                        "status": receipt.get("status"),
                        "runtime_ms": receipt.get("runtime_ms"),
                        "timeout_ms": receipt.get("timeout_ms"),
                        "step_index": receipt.get("step_index"),
                        "max_steps": receipt.get("max_steps"),
                        "citation_count": receipt.get("citation_count"),
                        "input_sha256": receipt.get("input_sha256"),
                        "deterministic_router": receipt.get("deterministic_router"),
                        "visual_policy_mutation_allowed": receipt.get(
                            "visual_policy_mutation_allowed"
                        ),
                        "medical_route_mutation_allowed": receipt.get(
                            "medical_route_mutation_allowed"
                        ),
                        "error_code": receipt.get("error_code"),
                    }
                    for receipt in receipts
                ],
                expanded=False,
            )

    for key, title in (
        ("diagnostic_information", "诊断信息"),
        ("next_step_information", "下一步信息"),
        ("treatment_education", "治疗教育"),
        ("limitations", "限制"),
    ):
        values = response.get(key) or []
        if isinstance(values, str):
            values = [values]
        if values:
            visible_values = [
                rendered for value in values if (rendered := _public_agent_text(value))
            ]
            if visible_values:
                st.markdown(f"**{title}**")
                for rendered in visible_values:
                    st.markdown(f"- {rendered}")
    citations = response.get("citations") or []
    if citations:
        with st.expander("证据来源（指南与审定资料）"):
            for item in citations:
                citation = _mapping(item)
                title = _safe_text(citation.get("title") or "未命名来源")
                url = str(citation.get("url") or "").strip()
                locator = _safe_text(citation.get("locator") or citation.get("section"))
                support = _safe_text(citation.get("support_text"))
                organization = _safe_text(citation.get("organization"))
                year = _safe_text(citation.get("publication_year"))
                meta = " · ".join(part for part in (organization, year, locator) if part)
                link = (
                    f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">'
                    "查看来源 ↗</a>"
                    if url.startswith(("https://", "http://"))
                    else ""
                )
                st.markdown(
                    f"""
                    <div class="tbx-citation">
                      <div class="tbx-citation__title">{_escape(title)}</div>
                      <div class="tbx-citation__meta">{_escape(meta)}</div>
                      <div class="tbx-citation__support">{_escape(support)}</div>
                      {link}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_module_status(
    health: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    capabilities: Mapping[str, Any] | None = None,
) -> None:
    health = health or {}
    manifest = manifest or {}
    capabilities = capabilities or {}
    backend = VISION_MODEL_DISPLAY_NAME
    narrator = _mapping(manifest.get("narrator"))
    narrator_backend = str(narrator.get("backend") or "none")

    rows: list[dict[str, str]] = [
        {
            "模块": "API",
            "状态": (
                "可用"
                if health.get("status") == "ok"
                else "可用（组件降级）"
                if health
                else "不可用"
            ),
            "角色": "必需",
            "说明": str(health.get("service") or API_URL),
        }
    ]

    components = capabilities.get("components")
    if isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        for raw_component in components:
            component = _mapping(raw_component)
            component_id = str(component.get("component_id") or "unknown")
            if component_id.startswith("agent_tool:"):
                tool_id = component_id.split(":", 1)[1]
                component_label = TOOL_COMPONENT_LABELS.get(tool_id)
                if not component_label:
                    continue
            else:
                component_label = COMPONENT_LABELS.get(component_id, component_id)
                if "rank03" in component_label.casefold():
                    component_label = VISION_MODEL_DISPLAY_NAME
            state = component.get("state")
            state_label = "合成演示" if component.get("synthetic") else _status_label(state)
            load_state = component.get("loaded")
            load_note = (
                "已加载"
                if load_state is True
                else "懒加载/未探测"
                if load_state is False
                else "未主动探测"
            )
            rows.append(
                {
                    "模块": component_label,
                    "状态": state_label,
                    "角色": "必需" if component.get("required") else "可选",
                    "说明": " · ".join(
                        part
                        for part in (
                            (
                                VISION_MODEL_DISPLAY_NAME
                                if component_id == "rank03_image_assessment"
                                else _safe_text(component.get("implementation"))
                            ),
                            load_note,
                            (
                                "真实影像运行时"
                                if component_id == "rank03_image_assessment"
                                else _safe_text(component.get("detail"))
                            ),
                        )
                        if part
                    ),
                }
            )
    else:
        # Backward-compatible inference for servers predating /v1/system/capabilities.
        rows.extend(
            [
                {
                    "模块": "影像推理",
                    "状态": "演示" if backend == "mock" else "已配置",
                    "角色": "必需",
                    "说明": backend,
                },
                {
                    "模块": "指南知识库",
                    "状态": "已加载" if manifest.get("knowledge_snapshot_id") else "状态未知",
                    "角色": "必需",
                    "说明": str(manifest.get("knowledge_snapshot_id") or "API 未提供"),
                },
                {
                    "模块": "语言整理",
                    "状态": "未启用" if narrator_backend == "none" else "已配置",
                    "角色": "可选",
                    "说明": narrator_backend,
                },
            ]
        )

    rows.extend(
        [
            {
                "模块": "分类决策策略",
                "状态": "已配置" if manifest.get("classifier_rule") else "状态未知",
                "角色": "必需",
                "说明": str(manifest.get("classifier_rule") or "API 未提供"),
            },
            {
                "模块": "定位证据角色",
                "状态": "已配置" if manifest.get("detector_role") else "状态未知",
                "角色": "辅助",
                "说明": DETECTOR_ROLE_LABELS.get(
                    str(manifest.get("detector_role")),
                    str(manifest.get("detector_role") or "API 未提供"),
                ),
            },
        ]
    )

    st.dataframe(rows, hide_index=True, width="stretch")
    if capabilities.get("mode"):
        st.caption("运行模式：真实推理能力快照（可选语言模型按连接状态单独检查）")
    if health and health.get("clinical_validation") is False:
        st.caption("临床验证状态：未完成。本系统输出仅用于辅助筛查和进一步评估导航。")


def show_classifier_evidence(case: Mapping[str, Any], manifest: Mapping[str, Any] | None) -> None:
    """Show the class decision without exposing raw class scores or thresholds."""

    evidence = _mapping(case.get("vision_evidence"))
    fusion = _mapping(case.get("fusion_decision"))
    manifest = manifest or {}

    st.markdown("#### 胸片分类")

    rule = str(
        evidence.get("classifier_decision_rule")
        or fusion.get("classifier_decision_rule")
        or manifest.get("classifier_rule")
        or "unknown"
    )
    rule_description = DECISION_RULE_LABELS.get(rule, rule)

    predicted_class = evidence.get("predicted_class", fusion.get("predicted_class"))
    if predicted_class:
        predicted_label = CLASS_LABELS.get(str(predicted_class), str(predicted_class))
        st.write(f"分类结果：{predicted_label}")
        st.caption(f"决策方式：{rule_description}")
    else:
        st.warning("本次没有形成单一分类结果。")
    if rule in {"p_tb_gte_threshold", "legacy_p_tb_threshold", "p_tb_threshold"}:
        if evidence.get("classifier_flagged") is True:
            st.warning("固定筛查规则已触发辅助筛查标记。")
        elif evidence.get("classifier_flagged") is False:
            st.info("固定筛查规则本次未触发。")
    if evidence.get("classifier_argmax_tied"):
        st.warning("三个训练类别出现最大值并列，请在当前对话中重新上传或补充信息。")


def _render_detection_overlay(
    image_bytes: bytes,
    evidence: Mapping[str, Any],
) -> tuple[Image.Image, list[dict[str, str]], int]:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    api_width = float(evidence.get("image_width") or image.width)
    api_height = float(evidence.get("image_height") or image.height)
    if api_width <= 0 or api_height <= 0:
        api_width, api_height = float(image.width), float(image.height)
    scale_x = image.width / api_width
    scale_y = image.height / api_height
    line_width = max(2, round(min(image.size) / 180))
    valid_rows: list[dict[str, str]] = []
    skipped = 0

    detections = evidence.get("detections") or []
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        detections = []
    selected_detections = select_display_detections(
        detections,
        policy=DETECTION_DISPLAY_POLICY,
        image_width=api_width,
    )
    skipped = max(0, len(detections) - len(selected_detections))
    for index, selected in enumerate(selected_detections, start=1):
        bbox = selected.bbox_xyxy
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
            score = selected.score
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
            skipped += 1
            continue
        x1 = min(max(x1, 0.0), api_width) * scale_x
        x2 = min(max(x2, 0.0), api_width) * scale_x
        y1 = min(max(y1, 0.0), api_height) * scale_y
        y2 = min(max(y2, 0.0), api_height) * scale_y
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(255, 176, 0), width=line_width)
        label = f"Candidate {index}"
        label_left = int(x1)
        label_top = max(0, int(y1) - 16)
        label_right = min(image.width, label_left + max(100, len(label) * 7))
        draw.rectangle(
            (label_left, label_top, label_right, min(image.height, label_top + 16)),
            fill=(20, 20, 20),
        )
        draw.text((label_left + 2, label_top + 2), label, fill=(255, 220, 120))
        valid_rows.append(
            {
                "候选区域": f"ROI {index}",
                "坐标 xyxy": f"({x1 / scale_x:.1f}, {y1 / scale_y:.1f}, "
                f"{x2 / scale_x:.1f}, {y2 / scale_y:.1f})",
            }
        )
    return overlay, valid_rows, skipped


def _compose_anatomy_view(
    image_bytes: bytes,
    boundary_bytes: bytes,
    contour_bytes: bytes | None,
    evidence: Mapping[str, Any],
    *,
    display_mode: str,
) -> Image.Image:
    """Compose source-coordinate lung boundaries without changing model evidence."""

    with Image.open(io.BytesIO(image_bytes)) as source:
        original = ImageOps.exif_transpose(source).convert("RGBA")
    with Image.open(io.BytesIO(boundary_bytes)) as source_boundary:
        boundary = source_boundary.convert("RGBA")
    if boundary.size != original.size:
        raise ValueError("anatomy boundary dimensions do not match the assessed image")
    contours = None
    if contour_bytes is not None:
        with Image.open(io.BytesIO(contour_bytes)) as source_contours:
            contours = source_contours.convert("RGBA")
        if contours.size != original.size:
            raise ValueError("refinement contour dimensions do not match the assessed image")

    canvas = original
    if display_mode in {"候选框", "组合"}:
        detected, _, _ = _render_detection_overlay(image_bytes, evidence)
        canvas = detected.convert("RGBA")
    if display_mode in {"肺野", "组合"}:
        canvas = Image.alpha_composite(canvas, boundary)
    if display_mode in {"轮廓", "组合"} and contours is not None:
        canvas = Image.alpha_composite(canvas, contours)
    return canvas.convert("RGB")


def _anatomy_location_rows(run: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return compact rows from server-authored source-coordinate relationships."""

    side_labels = {"left_lung": "左", "right_lung": "右"}
    zone_labels = {
        "upper_lung_field": "上肺野",
        "middle_lung_field": "中肺野",
        "lower_lung_field": "下肺野",
    }
    status_labels = {
        "localized": "已定位",
        "outside_lungs": "肺野外/未定位",
        "invalid_anatomy": "分割 QC 未通过",
    }
    raw_locations = run.get("detector_locations") or []
    if not isinstance(raw_locations, Sequence) or isinstance(raw_locations, (str, bytes)):
        return []
    rows: list[dict[str, str]] = []
    for index, raw_location in enumerate(raw_locations, start=1):
        location = _mapping(raw_location)
        status = str(location.get("status") or "")
        raw_assignments = location.get("assignments") or []
        assignments = (
            raw_assignments
            if isinstance(raw_assignments, Sequence)
            and not isinstance(raw_assignments, (str, bytes))
            else []
        )
        if not assignments:
            rows.append(
                {
                    "候选框": f"ROI {index}",
                    "空间状态": status_labels.get(status, status or "—"),
                    "二维肺野": "—",
                    "框内交叠": "—",
                }
            )
            continue
        for raw_assignment in assignments:
            assignment = _mapping(raw_assignment)
            try:
                overlap = float(assignment.get("box_overlap_fraction"))
            except (TypeError, ValueError):
                overlap_text = "—"
            else:
                overlap_text = f"{overlap * 100:.1f}%" if math.isfinite(overlap) else "—"
            side = side_labels.get(str(assignment.get("lung") or ""), "—")
            zone = zone_labels.get(str(assignment.get("primary_zone") or ""), "—")
            rows.append(
                {
                    "候选框": f"ROI {index}",
                    "空间状态": status_labels.get(status, status or "—"),
                    "二维肺野": f"{side}侧 {zone}",
                    "框内交叠": overlap_text,
                }
            )
    return rows


def show_anatomy_spatial_evidence(run: Mapping[str, Any]) -> None:
    summary = _mapping(run.get("spatial_summary"))
    if not summary:
        st.caption("空间关系摘要不可用；肺野边界仍可单独显示。")
        return
    st.caption(
        "肺野 QC："
        f"{_status_label(summary.get('anatomy_qc_status'))} · "
        f"候选框 {summary.get('candidate_count', 0)} · "
        f"已定位 {summary.get('localized_count', 0)} · "
        f"肺野外/未定位 {summary.get('outside_lungs_count', 0)}"
    )
    rows = _anatomy_location_rows(run)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.caption("当前没有可列出的候选框空间关系。")
    statements = summary.get("statements") or []
    if isinstance(statements, Sequence) and not isinstance(statements, (str, bytes)):
        for statement in statements:
            public_statement = _safe_text(statement)
            if "rank03" in public_statement.casefold() or "D-FINE" in public_statement:
                continue
            st.markdown(f"- {public_statement}")
    st.caption(
        f"上/中/下仅为二维肺野分区；该证据不参与 {VISION_MODEL_DISPLAY_NAME} 的三分类分流。"
    )


def show_detection_evidence(case: Mapping[str, Any], image_bytes: bytes | None) -> None:
    localization = _localization_evidence(case)
    status = str(localization.get("status") or "not_requested")
    st.markdown("#### 定位检测器证据")
    if status == "not_requested":
        st.info("本病例尚未调用定位工具。")
        return
    if status == "completed_no_detection":
        st.info("定位工具已完成，本次没有返回候选区域。")
        return
    if status == "failed":
        st.warning("定位工具执行失败；本次没有可显示的候选区域。")
        return
    if status != "completed":
        st.info(f"定位证据当前状态：{_status_label(status)}。")
        return

    evidence = _completed_localization_view(case)
    if not evidence:
        st.warning("定位状态与候选区域数据不一致；已停止绘制。")
        return
    if not image_bytes:
        st.warning("当前会话没有保留原始上传图像，无法绘制候选框；结构化坐标仍保留在病例证据中。")
        return
    try:
        overlay, rows, skipped = _render_detection_overlay(image_bytes, evidence)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        st.warning(f"候选框预览生成失败，但不影响后端已保存的结构化结果：{exc}")
        return

    if rows:
        original_column, overlay_column = st.columns(2)
        with original_column:
            st.image(image_bytes, caption="原始上传图像", width="stretch")
        with overlay_column:
            st.image(
                overlay,
                caption="候选区域叠加图（橙框，仅供定位参考）",
                width="stretch",
            )
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.image(image_bytes, caption="原始上传图像", width="stretch")
        st.warning("定位结果中没有可安全绘制的候选区域。")
    st.caption(f"当前显示 {len(rows)} 个主要候选区域；重叠候选已合并。")
    if skipped:
        with st.expander("查看未展示的原始候选说明"):
            st.write(
                f"{skipped} 个原始候选因重叠去重、数量上限或坐标无效未画在主图上。"
            )


def show_assessment_summary(
    result: Mapping[str, Any],
    *,
    synthetic: bool,
    runtime_verified: bool = False,
) -> None:
    case = _mapping(result.get("case"))
    response = _mapping(result.get("response"))
    evidence = _mapping(case.get("vision_evidence"))
    fusion = _mapping(case.get("fusion_decision"))
    visual_result = str(response.get("visual_result") or fusion.get("visual_result") or "")
    presentation = _visual_result_presentation(visual_result)
    result_label = presentation["label"]
    runtime_kind = _assessment_runtime_kind(
        case,
        capability_synthetic=synthetic,
        runtime_verified=runtime_verified,
    )
    runtime_label = {
        "real": "真实运行时证据",
        "synthetic": "合成演示证据",
        "unverified": "来源未验证",
    }[runtime_kind]
    st.markdown(
        f"""
        <div class="tbx-result-banner tbx-result-banner--{presentation['class_name']}">
          <div class="tbx-result-banner__icon">{presentation['icon']}</div>
          <div>
            <strong>{_escape(runtime_label + ' · ' + result_label)}</strong>
            <span>{_escape(presentation['message'])}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predicted_class = response.get("predicted_class") or fusion.get("predicted_class")
    predicted_label = CLASS_LABELS.get(str(predicted_class), str(predicted_class or "—"))
    runtime_ms = evidence.get("runtime_ms")
    metric_columns = st.columns(2)
    metric_columns[0].metric("系统分流", result_label)
    metric_columns[1].metric("原生训练类别", predicted_label)
    if runtime_ms is not None:
        st.caption(f"结构化影像处理耗时 {runtime_ms} ms · 病例 {case.get('case_id', '—')}")
    if response.get("reused_existing_assessment"):
        st.info("检测到同一工作区内已有匹配评估，本次返回复用结果，未重复执行影像推理。")


def show_compact_assessment(
    result: Mapping[str, Any],
    *,
    synthetic: bool,
    runtime_verified: bool = False,
) -> None:
    case = _mapping(result.get("case"))
    response = _mapping(result.get("response"))
    fusion = _mapping(case.get("fusion_decision"))
    visual_result = str(response.get("visual_result") or fusion.get("visual_result") or "")
    presentation = _visual_result_presentation(visual_result)
    predicted_class = str(response.get("predicted_class") or fusion.get("predicted_class") or "")
    if visual_result == "non_tb_abnormal" or (
        visual_result == "model_not_flagged" and predicted_class == "sick_non_tb"
    ):
        conclusion = "本次模型未识别为结核类，但发现非结核异常倾向。"
    elif visual_result == "model_not_flagged":
        conclusion = "本次模型未识别为结核类。"
    elif visual_result == "model_flagged":
        conclusion = "三分类模型结果为结核类，建议进一步检查。"
    elif visual_result == "pending_human_review":
        conclusion = "本次结果不确定，请在当前对话中重试或补充信息。"
    elif visual_result == "technical_failure":
        conclusion = "本次分析失败，请重试。"
    else:
        conclusion = "本次结果暂时无法判断。"
    st.markdown(
        f"""
        <div class="tbx-compact-result tbx-compact-result--{presentation['class_name']}">
          <strong>{_escape(conclusion)}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_workspace_image(
    image_bytes: bytes,
    assessment: Mapping[str, Any] | None,
    *,
    result_matches_image: bool,
    anatomy_boundary_bytes: bytes | None = None,
    anatomy_contour_bytes: bytes | None = None,
    display_mode: str = "原图",
) -> None:
    """Keep the active image canvas stable while surfacing the latest visual tool output."""

    if result_matches_image and assessment:
        case = _mapping(assessment.get("case"))
        evidence = _completed_localization_view(case)
        if anatomy_boundary_bytes and display_mode in {"肺野", "轮廓", "组合"}:
            try:
                anatomy_view = _compose_anatomy_view(
                    image_bytes,
                    anatomy_boundary_bytes,
                    anatomy_contour_bytes,
                    evidence,
                    display_mode=display_mode,
                )
            except (OSError, UnidentifiedImageError, ValueError):
                st.image(image_bytes, width="stretch")
                st.caption("空间图层合成失败，已回退到原图；分类结果不受影响。")
                return
            else:
                st.image(anatomy_view, width="stretch")
                return
        if display_mode == "原图":
            st.image(image_bytes, width="stretch")
            return
        try:
            overlay, rows, _ = _render_detection_overlay(image_bytes, evidence)
        except (OSError, UnidentifiedImageError, ValueError):
            rows = []
        if rows:
            st.image(overlay, width="stretch")
            return
    st.image(image_bytes, width="stretch")


def show_chat_response(payload: Mapping[str, Any]) -> None:
    """Render the useful answer only; provenance lives on the detail page."""

    response = _mapping(payload.get("response", payload))
    raw_summary = response.get("summary")
    summary = _public_agent_text(raw_summary)
    if summary:
        st.markdown(summary)
    elif raw_summary:
        st.warning("本轮回答未形成可安全展示的文本，请重试。")

    # A grounded retrieval answer is already integrated into ``summary`` by
    # the backend. Keep the conversational surface to that answer only; the
    # verbatim passages and their provenance remain one click away below.
    # This also prevents legacy diagnostic/next-step arrays from re-expanding
    # the retrieved claims in the main chat area.
    if response.get("retrieved_evidence"):
        show_retrieved_evidence(response)
        return

    # Tool responses keep the full structured evidence lists for audit and the
    # detail page.  The conversational view must not repeat the leading claim
    # when that same claim is already the summary (including PARTIAL answers
    # whose summary appends an evidence-gap sentence).
    seen = {summary.strip()} if summary.strip() else set()
    grounded_answer = response.get("answer_status") is not None
    for key, title in (
        ("visual_evidence_notes", "影像依据"),
        ("diagnostic_information", "检查信息"),
        ("next_step_information", "下一步"),
        ("treatment_education", "治疗教育"),
    ):
        values = response.get(key) or []
        if isinstance(values, str):
            values = [values]
        if grounded_answer and key != "visual_evidence_notes":
            # The grounded answer is a synthesis. Exact retrieved passages and
            # extractive claims belong behind the evidence control below, not
            # as a second copy of the answer in the chat transcript.
            values = []
        visible_values = []
        for value in values:
            rendered = _public_agent_text(value).strip()
            if not rendered or rendered in seen or summary.strip().startswith(rendered):
                continue
            seen.add(rendered)
            visible_values.append(rendered)
        if visible_values:
            items = "\n".join(f"- {value}" for value in visible_values)
            st.markdown(f"**{title}**\n\n{items}")


def show_retrieved_evidence(response: Mapping[str, Any]) -> None:
    """Keep retrieved source passages available without crowding the answer."""

    raw_items = response.get("retrieved_evidence") or []
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
        return
    items = [
        _mapping(item)
        for item in raw_items
        if _safe_text(_mapping(item).get("text")).strip()
    ]
    if not items:
        return

    with st.popover("查看检索内容"):
        for index, item in enumerate(items, start=1):
            source = _safe_text(
                item.get("source")
                or item.get("title")
                or item.get("source_id")
                or "未命名来源"
            ).strip()
            section = _safe_text(item.get("section")).strip()
            locator = _safe_text(item.get("locator")).strip()
            text = _safe_text(item.get("text")).strip()
            url = str(item.get("url") or "").strip()
            meta_rows = [f"来源：{source}"]
            if section:
                meta_rows.append(f"章节：{section}")
            if locator:
                meta_rows.append(f"位置：{locator}")
            link = (
                f'<a href="{_escape(url)}" target="_blank" rel="noopener noreferrer">'
                "查看来源 ↗</a>"
                if url.startswith(("https://", "http://"))
                else ""
            )
            st.markdown(
                f"""
                <div class="tbx-citation tbx-retrieved-evidence">
                  <div class="tbx-citation__title">检索片段 {index}</div>
                  <div class="tbx-citation__meta">{_escape(' · '.join(meta_rows))}</div>
                  <div class="tbx-citation__support">{_escape(text)}</div>
                  {link}
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_tool_chain(
    payload: Mapping[str, Any] | None,
    *,
    show_plan: bool = True,
) -> None:
    """Render the public Plan and actual model-facing tool badges.

    Only the initial plan's concise objectives are shown. Plan metadata,
    revisions, ReAct observations and hidden reasoning never enter the chat
    surface. Tool badges come from real receipts, preferring
    ``model_tool_name`` over the internal handler name.
    """

    if show_plan:
        show_initial_plan(payload)

    response = _mapping(_mapping(payload).get("response", _mapping(payload)))
    guideline_subtopic_labels = {
        "risk_groups": "高风险人群",
        "active_screening_population": "主动筛查人群",
        "rapid_molecular_diagnostics": "快速分子检测",
        "negative_test_interpretation": "阴性结果解释",
        "diagnostic_pathway": "诊断检查",
        "treatment_principles": "治疗原则",
        "standard_regimen_duration": "标准疗程",
        "respiratory_protection": "呼吸防护",
        "cad_result_interpretation": "影像结果解释",
        "special_population_guidance": "特殊人群",
        "special_population_testing": "特殊人群检查",
    }
    guideline_label = guideline_subtopic_labels.get(
        str(response.get("guideline_subtopic") or "")
    )
    visible_receipts: list[tuple[Mapping[str, Any], str, str]] = []
    receipt_index: dict[str, int] = {}
    guideline_tools = {
        "search_tb_knowledge",
        "search_tb_guidance",
        "retrieve_guideline",
    }
    for receipt in _execution_receipts(payload):
        public_tool_name = _receipt_public_tool_name(receipt)
        internal_tool_name = str(receipt.get("tool_name") or "").strip()
        label = TOOL_COMPONENT_LABELS.get(public_tool_name) or TOOL_COMPONENT_LABELS.get(
            internal_tool_name
        )
        if not label:
            continue
        if public_tool_name in guideline_tools or internal_tool_name in guideline_tools:
            public_knowledge_call = (
                str(receipt.get("model_tool_name") or "") == "search_tb_knowledge"
                or (
                    not receipt.get("model_tool_name")
                    and internal_tool_name == "search_tb_knowledge"
                )
            )
            knowledge_label = "结核知识检索" if public_knowledge_call else "指南证据检索"
            label = f"{knowledge_label} · {guideline_label}" if guideline_label else label
        key = public_tool_name or internal_tool_name
        item = (receipt, label, key)
        if key in receipt_index:
            # Keep one compact badge per public tool while reflecting the final
            # retry status for that tool.
            visible_receipts[receipt_index[key]] = item
        else:
            receipt_index[key] = len(visible_receipts)
            visible_receipts.append(item)
    if not visible_receipts:
        return
    status_marks = {
        "succeeded": "✓",
        "failed": "!",
        "timed_out": "!",
        "unavailable": "!",
        "saturated": "!",
        "rejected": "!",
        "step_limit_exceeded": "!",
    }
    rows = []
    answer_status = str(response.get("answer_status") or "").upper()
    common_knowledge_fallback_applied = (
        answer_status == "INSUFFICIENT_EVIDENCE"
        and response.get("narrator_policy_id")
        == _MEDICAL_COMMON_KNOWLEDGE_POLICY_ID
        and str(response.get("narration_status") or "").lower()
        in {"applied", "fallback_error"}
        and not response.get("claims")
        and not response.get("citations")
    )
    for receipt, label, public_tool_name in visible_receipts:
        status = str(receipt.get("status") or "failed")
        internal_tool_name = str(receipt.get("tool_name") or "")
        is_knowledge_tool = (
            public_tool_name in guideline_tools or internal_tool_name in guideline_tools
        )
        common_knowledge_row = (
            is_knowledge_tool
            and status == "succeeded"
            and common_knowledge_fallback_applied
        )
        evidence_gap = str(receipt.get("outcome") or "").lower() == "evidence_gap" or (
            is_knowledge_tool and answer_status == "INSUFFICIENT_EVIDENCE"
        )
        if common_knowledge_row:
            style_status = "completed"
            status_mark = "✓"
            label = "指南检索未命中 · 通用医学信息"
        elif evidence_gap:
            style_status = "evidence-gap"
            status_mark = "!"
            label = f"{label} · 证据不足"
        else:
            style_status = "completed" if status == "succeeded" else "failed"
            status_mark = status_marks.get(status, "·")
        rows.append(
            f'<div class="tbx-tool-badge tbx-tool-badge--{style_status} '
            f'tbx-plan-step--{style_status}">'
            f'<span>{_escape(status_mark)}</span>'
            f'<strong>{_escape(label)}</strong>'
            "</div>"
        )
    st.markdown(
        '<div class="tbx-tool-strip"><div class="tbx-plan-title">本轮工具链</div>'
        '<div class="tbx-tool-badges">'
        + "".join(rows)
        + "</div></div>",
        unsafe_allow_html=True,
    )


_GENERIC_PLAN_OBJECTIVES = {
    "直接回答当前问题",
    "回答当前问题",
    "生成回答",
    "整理回答",
    "结合已有信息回答",
    "根据已有信息回答",
}
_NON_PUBLIC_PLAN_MARKERS = (
    "思维过程",
    "推理过程",
    "隐藏推理",
    "chain of thought",
    "reasoning",
    "分析用户意图",
    "校验安全边界",
    "构建最小必要上下文",
    "agent state",
    "allowed_tools",
    "case_state",
    "observations",
    "tool_calls",
)
_MAX_PUBLIC_PLAN_OBJECTIVE_CHARS = 96


def _visible_initial_plan_steps(
    payload: Mapping[str, Any] | None,
) -> list[str]:
    execution_plan = _mapping(_mapping(payload).get("execution_plan"))
    metadata = _mapping(execution_plan.get("plan_metadata"))
    if metadata.get("planning_used") is False:
        return []
    initial_plan = _mapping(execution_plan.get("initial_plan"))
    raw_steps = initial_plan.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        return []
    visible: list[str] = []
    seen: set[str] = set()
    for raw_step in raw_steps[:4]:
        step = _mapping(raw_step)
        objective = " ".join(_public_agent_text(step.get("objective")).split())
        if not objective or objective in _GENERIC_PLAN_OBJECTIVES:
            continue
        # Suppress generic fallback prose even if a provider added punctuation.
        compact = objective.rstrip("。.!！ ")
        normalized = objective.casefold()
        if (
            len(objective) > _MAX_PUBLIC_PLAN_OBJECTIVE_CHARS
            or compact in _GENERIC_PLAN_OBJECTIVES
            or any(marker in objective.casefold() for marker in _NON_PUBLIC_PLAN_MARKERS)
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        visible.append(objective)
    return visible


def show_initial_plan(payload: Mapping[str, Any] | None) -> None:
    """Show only short, user-auditable objectives from ``initial_plan``."""

    objectives = _visible_initial_plan_steps(payload)
    if not objectives:
        return
    rows = [
        '<div class="tbx-plan-step tbx-plan-step--planned">'
        f"<span>{index}</span><strong>{_escape(objective)}</strong></div>"
        for index, objective in enumerate(objectives, start=1)
    ]
    st.markdown(
        '<div class="tbx-plan tbx-turn-plan"><div class="tbx-plan-title">计划</div>'
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


def _latest_chat_plan_payload(
    history: Sequence[Mapping[str, Any]] | Any,
) -> Mapping[str, Any]:
    """Return a plan only when it belongs to the current completed chat turn."""

    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)) or not history:
        return {}
    latest = _mapping(history[-1])
    if str(latest.get("role") or "") != "assistant":
        return {}
    payload = _mapping(latest.get("payload"))
    return payload if _visible_initial_plan_steps(payload) else {}


def show_chat_message(
    message: Mapping[str, Any],
    *,
    synthetic: bool,
    runtime_verified: bool = False,
    show_plan: bool = False,
) -> None:
    role = str(message.get("role") or "assistant")
    with st.chat_message(role):
        if role == "user":
            image_bytes = message.get("image_bytes")
            if isinstance(image_bytes, bytes):
                st.image(
                    image_bytes,
                    caption=_safe_text(message.get("image_name") or "当前胸片"),
                    width=180,
                )
            st.markdown(_safe_text(message.get("content")))
            return

        payload = _mapping(message.get("payload"))
        show_tool_chain(payload, show_plan=show_plan)
        assessment = _mapping(message.get("assessment"))
        suppress_payload = bool(message.get("suppress_payload"))
        if assessment and _has_classification_evidence(
            _mapping(assessment.get("case"))
        ) and (not payload or suppress_payload):
            show_compact_assessment(
                assessment,
                synthetic=synthetic,
                runtime_verified=runtime_verified,
            )
        if payload and not suppress_payload:
            show_chat_response(payload)
        elif message.get("content"):
            content = _public_agent_text(message.get("content"))
            if content:
                st.markdown(content)
            else:
                st.warning("本轮回答未形成可安全展示的文本，请重试。")
        elif message.get("error"):
            st.error(_safe_text(message.get("error")))


def _should_hide_initial_agent_payload(prompt: str, payload: Mapping[str, Any]) -> bool:
    """Keep a first image-analysis turn focused on the image conclusion.

    The full response remains stored for audit/detail use. Explicit requests for tests,
    treatment, urgent help, or next steps still render immediately.
    """

    # A compact classification card must not hide unfinished work or tool failure.
    execution = _mapping(payload.get("execution_plan"))
    if execution.get("unfinished_evidence") or execution.get("finalization_recovery"):
        return False
    if any(
        _mapping(step).get("status") in {"pending", "failed"}
        for step in _mapping(execution.get("final_plan")).get("steps", [])
    ):
        return False
    response = _mapping(payload.get("response", payload))
    tool_names = {
        name
        for receipt in _execution_receipts(payload)
        for name in (
            str(receipt.get("model_tool_name") or ""),
            str(receipt.get("tool_name") or ""),
        )
        if name
    }
    if response.get("urgency") == "emergency" or tool_names.intersection(
        {
            "emergency_triage",
            "localize_cxr",
            "localize_current_cxr",
            "analyze_lung_anatomy",
            "inspect_anatomical_context",
            "search_tb_knowledge",
            "retrieve_treatment_education",
        }
    ):
        return False
    if not tool_names.intersection({"classify_cxr", "classify_current_cxr"}):
        return False
    explicit_followup_terms = (
        "下一步",
        "怎么办",
        "检查",
        "治疗",
        "用药",
        "药物",
        "症状",
        "接触史",
        "进一步",
        "建议",
        "如何",
        "病灶",
        "在哪",
        "哪里",
        "为什么",
        "依据",
        "概率",
        "分数",
    )
    normalized = prompt.strip().casefold()
    return not any(term in normalized for term in explicit_followup_terms)


def _screening_button_key(session_id: str, question_id: str, value: Any) -> str:
    digest = hashlib.sha256(f"{session_id}:{question_id}:{value!r}".encode()).hexdigest()[:12]
    return f"screening-answer-{digest}"


def _store_screening_payload(payload: Mapping[str, Any]) -> None:
    st.session_state.screening_session = dict(_mapping(payload.get("session")))
    st.session_state.screening_response = dict(_mapping(payload.get("response")))


def _submit_chat_screening_answer(
    *,
    session: Mapping[str, Any],
    question_id: str,
    answer: Any,
    display_value: str,
) -> None:
    updated = call(
        "POST",
        f"/v1/screening/sessions/{session['session_id']}/answers",
        json={
            "user_id": st.session_state.user_id,
            "owner_scope": st.session_state.owner_scope,
            "question_id": question_id,
            "answer": answer,
        },
    )
    if not updated:
        return
    st.session_state.chat_history.append(
        {"role": "user", "content": f"主动筛查：{display_value}"}
    )
    _store_screening_payload(_mapping(updated))
    st.rerun()


def _exit_chat_screening(session: Mapping[str, Any]) -> None:
    session_id = str(session.get("session_id") or "")
    if session_id and session.get("status") == "collecting":
        cancelled = call(
            "POST",
            f"/v1/screening/sessions/{session_id}/cancel",
            json={
                "user_id": st.session_state.user_id,
                "owner_scope": st.session_state.owner_scope,
            },
        )
        if not cancelled:
            return
    st.session_state.pop("screening_session", None)
    st.session_state.pop("screening_response", None)
    st.session_state.chat_history.append(
        {"role": "assistant", "content": "已退出主动筛查。"}
    )
    st.rerun()


def _open_case_details() -> None:
    st.session_state.navigation = "病例详情"


def _assess_uploaded_image_for_screening(
    *,
    image_bytes: bytes,
    image_name: str,
    image_sha256: str,
    capability_synthetic: bool,
    runtime_verified: bool,
) -> str | None:
    """Register the current upload so screening can bind the exact case."""

    assessment = call(
        "POST",
        "/v1/assessments/cxr",
        files={
            "file": (
                image_name,
                image_bytes,
                _image_mime_type(image_name),
            )
        },
        data={
            "user_id": st.session_state.user_id,
            "owner_scope": st.session_state.owner_scope,
            "consent_to_process": "true",
            "attested_chest_radiograph": "true",
        },
    )
    case = _mapping(_mapping(assessment).get("case"))
    case_id = str(case.get("case_id") or "")
    if not assessment or not case_id:
        return None
    has_classification = _has_classification_evidence(case)
    if has_classification and (
        _assessment_runtime_kind(
            case,
            capability_synthetic=capability_synthetic,
            runtime_verified=runtime_verified,
        )
        not in ({"real", "synthetic"} if _demo_session_active() else {"real"})
    ):
        st.error("当前胸片未形成已验证的真实模型结果，主动筛查暂不绑定该影像。")
        return None
    if (
        has_classification
        and _assessment_visual_result(_mapping(assessment)) == "technical_failure"
    ):
        st.error("当前胸片处理失败，请重新上传或稍后再试。")
        return None

    st.session_state.assessment_result = assessment
    st.session_state.assessment_upload_sha256 = image_sha256
    st.session_state.assessment_confirmation_sha256 = image_sha256
    st.session_state.assessment_user_id = st.session_state.user_id
    st.session_state.assessment_owner_scope = st.session_state.owner_scope
    st.session_state.case_id = case_id
    for key in (
        "generated_report",
        "generated_report_markdown",
        "generated_report_json",
        "anatomy_run",
        "anatomy_boundary_bytes",
        "anatomy_contour_bytes",
        "anatomy_case_id",
        "anatomy_image_sha256",
        "anatomy_technical_failure",
        "anatomy_display_mode",
        "pending_anatomy_display_mode",
        "localization_receipt_case_id",
        "localization_receipt_image_sha256",
        "anatomy_receipt_case_id",
        "anatomy_receipt_image_sha256",
    ):
        st.session_state.pop(key, None)
    return case_id


def render_chat_screening_controls(
    *,
    service_available: bool,
    assessment_ready: bool,
    image_bytes: bytes | None,
    image_name: str,
    image_sha256: str | None,
    capability_synthetic: bool,
    runtime_verified: bool,
) -> tuple[bool, str | None]:
    """Render the deterministic screening state machine inside the Agent conversation."""

    session = _mapping(st.session_state.get("screening_session"))
    response = _mapping(st.session_state.get("screening_response"))
    if not session:
        action_count = 3 if assessment_ready else 1
        action_columns = st.columns(action_count)
        if action_columns[0].button(
            "开始主动筛查",
            type="secondary",
            disabled=not service_available,
            use_container_width=True,
            key="chat-start-screening",
        ):
            screening_case_id = (
                str(st.session_state.get("case_id") or "") if assessment_ready else ""
            )
            if not screening_case_id and isinstance(image_bytes, bytes) and image_sha256:
                screening_case_id = str(
                    _assess_uploaded_image_for_screening(
                        image_bytes=image_bytes,
                        image_name=image_name,
                        image_sha256=image_sha256,
                        capability_synthetic=capability_synthetic,
                        runtime_verified=runtime_verified,
                    )
                    or ""
                )
                if not screening_case_id:
                    return False, None
            started = call(
                "POST",
                "/v1/screening/sessions",
                json={
                    "thread_id": st.session_state.thread_id,
                    "user_id": st.session_state.user_id,
                    "owner_scope": st.session_state.owner_scope,
                    "case_id": screening_case_id or None,
                    "consent": True,
                },
            )
            if started:
                st.session_state.chat_history.append(
                    {"role": "user", "content": "开始主动筛查"}
                )
                _store_screening_payload(_mapping(started))
                st.rerun()
        if assessment_ready:
            if action_columns[1].button(
                "下一步检查",
                use_container_width=True,
                key="chat-next-tests",
            ):
                return False, "根据当前病例，下一步建议做哪些检查？"
            action_columns[2].button(
                "查看详情",
                use_container_width=True,
                key="chat-open-details",
                on_click=_open_case_details,
            )
        return False, None

    status = str(session.get("status") or "")
    if status == "collecting":
        question = _mapping(response.get("next_question"))
        question_id = str(question.get("question_id") or "")
        with st.chat_message("assistant"):
            answered_count = len(_mapping(session.get("answers")))
            st.caption(f"主动筛查 · 已回答 {answered_count} 项")
            if session.get("case_id") and response.get("visual_result"):
                visual = _visual_result_presentation(response.get("visual_result"))
                st.caption(f"当前胸片 · {visual['label']}")
            st.markdown(f"**{_safe_text(question.get('text_zh') or '请选择一项')}**")
            answer_type = str(question.get("answer_type") or "")
            choices = [str(value) for value in question.get("choices") or []]

            if answer_type == "boolean":
                options: list[tuple[str, Any]] = [
                    ("是", True),
                    ("否", False),
                    ("不清楚", "不知道"),
                    ("跳过", "跳过"),
                ]
                columns = st.columns(4)
                for column, (label, value) in zip(columns, options, strict=True):
                    if column.button(
                        label,
                        use_container_width=True,
                        key=_screening_button_key(str(session["session_id"]), question_id, value),
                    ):
                        _submit_chat_screening_answer(
                            session=session,
                            question_id=question_id,
                            answer=value,
                            display_value=label,
                        )
            elif answer_type == "single_choice":
                options = [(choice, choice) for choice in choices] + [
                    ("不清楚", "不知道"),
                    ("跳过", "跳过"),
                ]
                for offset in range(0, len(options), 2):
                    columns = st.columns(2)
                    for column, (label, value) in zip(
                        columns, options[offset : offset + 2], strict=False
                    ):
                        if column.button(
                            label,
                            use_container_width=True,
                            key=_screening_button_key(
                                str(session["session_id"]), question_id, value
                            ),
                        ):
                            _submit_chat_screening_answer(
                                session=session,
                                question_id=question_id,
                                answer=value,
                                display_value=label,
                            )
            elif answer_type == "multi_choice":
                selected = st.multiselect(
                    "可多选",
                    choices,
                    label_visibility="collapsed",
                    placeholder="请选择，可多选",
                    key=f"screening-multi-{session['session_id']}-{question_id}",
                )
                submit_column, unknown_column, skip_column = st.columns(3)
                if submit_column.button(
                    "下一题",
                    type="primary",
                    disabled=not selected,
                    use_container_width=True,
                    key=f"screening-multi-submit-{session['session_id']}-{question_id}",
                ):
                    _submit_chat_screening_answer(
                        session=session,
                        question_id=question_id,
                        answer=selected,
                        display_value="、".join(selected),
                    )
                if unknown_column.button(
                    "不清楚",
                    use_container_width=True,
                    key=_screening_button_key(
                        str(session["session_id"]), question_id, "不知道"
                    ),
                ):
                    _submit_chat_screening_answer(
                        session=session,
                        question_id=question_id,
                        answer="不知道",
                        display_value="不清楚",
                    )
                if skip_column.button(
                    "跳过",
                    use_container_width=True,
                    key=_screening_button_key(str(session["session_id"]), question_id, "跳过"),
                ):
                    _submit_chat_screening_answer(
                        session=session,
                        question_id=question_id,
                        answer="跳过",
                        display_value="跳过",
                    )
            elif answer_type == "integer":
                value = st.number_input(
                    "请输入数值",
                    min_value=0,
                    max_value=130,
                    value=None,
                    step=1,
                    key=f"screening-integer-{session['session_id']}-{question_id}",
                )
                submit_column, unknown_column, skip_column = st.columns(3)
                if submit_column.button(
                    "下一题",
                    type="primary",
                    disabled=value is None,
                    use_container_width=True,
                    key=f"screening-integer-submit-{session['session_id']}-{question_id}",
                ):
                    _submit_chat_screening_answer(
                        session=session,
                        question_id=question_id,
                        answer=int(value),
                        display_value=str(value),
                    )
                for column, label, answer in (
                    (unknown_column, "不清楚", "不知道"),
                    (skip_column, "跳过", "跳过"),
                ):
                    if column.button(
                        label,
                        use_container_width=True,
                        key=_screening_button_key(
                            str(session["session_id"]), question_id, answer
                        ),
                    ):
                        _submit_chat_screening_answer(
                            session=session,
                            question_id=question_id,
                            answer=answer,
                            display_value=label,
                        )
            else:
                value = st.text_input(
                    "请输入回答",
                    key=f"screening-text-{session['session_id']}-{question_id}",
                )
                if st.button(
                    "下一题",
                    type="primary",
                    disabled=not value.strip(),
                    key=f"screening-text-submit-{session['session_id']}-{question_id}",
                ):
                    _submit_chat_screening_answer(
                        session=session,
                        question_id=question_id,
                        answer=value,
                        display_value=value,
                    )

            if st.button(
                "退出主动筛查",
                use_container_width=True,
                key=f"screening-exit-{session['session_id']}",
            ):
                _exit_chat_screening(session)
        return True, None

    if status == "complete":
        with st.chat_message("assistant"):
            st.markdown("**主动筛查已完成**")
            if session.get("case_id") and response.get("visual_result"):
                visual = _visual_result_presentation(response.get("visual_result"))
                st.caption(f"已结合当前胸片 · {visual['label']}")
            next_steps = response.get("next_step_information") or []
            if isinstance(next_steps, str):
                next_steps = [next_steps]
            for value in list(next_steps)[:2]:
                st.markdown(f"- {_safe_text(value)}")
            if st.button(
                "完成",
                type="primary",
                use_container_width=True,
                key=f"screening-finish-{session['session_id']}",
            ):
                summary = "主动筛查已完成。"
                if next_steps:
                    summary += f" {_safe_text(next_steps[0])}"
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": summary}
                )
                st.session_state.pop("screening_session", None)
                st.session_state.pop("screening_response", None)
                st.rerun()
        return False, None

    st.session_state.pop("screening_session", None)
    st.session_state.pop("screening_response", None)
    return False, None


def _image_mime_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        return "application/dicom"
    return "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"


def _upload_preview(payload: bytes) -> bytes:
    """Return a metadata-free display raster while preserving raw bytes for the API."""

    validated = validate_image(payload, max_bytes=UI_MAX_UPLOAD_BYTES)
    output = io.BytesIO()
    validated.image.save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue()


def _run_anatomy_layer(
    *,
    case_id: str,
    image_name: str,
    image_bytes: bytes,
    image_sha256: str,
) -> str:
    """Run the optional worker and persist only artifacts bound to this image."""

    try:
        anatomy_result = request_and_poll_anatomy(
            api_url=API_URL,
            case_id=case_id,
            owner_scope=st.session_state.owner_scope,
            user_id=st.session_state.user_id,
            image_name=image_name,
            image_bytes=image_bytes,
            image_mime_type=_image_mime_type(image_name),
        )
    except AnatomyClientError:
        st.session_state.anatomy_technical_failure = True
        return "failed"

    st.session_state.anatomy_run = dict(anatomy_result.run)
    st.session_state.anatomy_case_id = case_id
    st.session_state.anatomy_image_sha256 = image_sha256
    if (
        anatomy_result.status in {"completed", "completed_with_refinement_failure"}
        and anatomy_result.boundary_png
    ):
        st.session_state.anatomy_boundary_bytes = anatomy_result.boundary_png
        if anatomy_result.contours_png:
            st.session_state.anatomy_contour_bytes = anatomy_result.contours_png
        else:
            st.session_state.pop("anatomy_contour_bytes", None)
        st.session_state.anatomy_technical_failure = False
        return "completed"

    st.session_state.pop("anatomy_boundary_bytes", None)
    st.session_state.pop("anatomy_contour_bytes", None)
    st.session_state.anatomy_technical_failure = True
    return "failed"


def _execution_receipts(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Return actual receipts, including model-only forward-compatible records."""

    answer = _mapping(payload)
    response = _mapping(answer.get("response"))
    raw_receipts = (
        answer.get("execution_receipts")
        if "execution_receipts" in answer
        else response.get("execution_receipts")
    )
    if isinstance(raw_receipts, Sequence) and not isinstance(
        raw_receipts, (str, bytes)
    ):
        return [
            receipt
            for raw_receipt in raw_receipts
            if (receipt := _mapping(raw_receipt))
            and (receipt.get("model_tool_name") or receipt.get("tool_name"))
        ]
    return []


def _receipt_public_tool_name(receipt: Mapping[str, Any]) -> str:
    """Prefer the model-facing capability while retaining legacy receipts."""

    return str(receipt.get("model_tool_name") or receipt.get("tool_name") or "").strip()


_TOOL_NAME_ALIASES = {
    "classify_current_cxr": {"classify_current_cxr", "classify_cxr"},
    "localize_current_cxr": {"localize_current_cxr", "localize_cxr"},
    "inspect_anatomical_context": {
        "inspect_anatomical_context",
        "analyze_lung_anatomy",
    },
    "search_tb_knowledge": {
        "search_tb_knowledge",
        "search_tb_guidance",
        "retrieve_guideline",
    },
}


def _used_tool(payload: Mapping[str, Any] | None, tool_name: str) -> bool:
    accepted_names = _TOOL_NAME_ALIASES.get(tool_name, {tool_name})
    return any(
        {
            str(receipt.get("model_tool_name") or ""),
            str(receipt.get("tool_name") or ""),
        }.intersection(accepted_names)
        and str(receipt.get("status") or "") == "succeeded"
        for receipt in _execution_receipts(payload)
    )


def show_execution_trace(
    case: Mapping[str, Any],
    *,
    synthetic: bool,
    runtime_verified: bool = False,
) -> None:
    evidence = _mapping(case.get("vision_evidence"))
    localization = _localization_evidence(case)
    fusion = _mapping(case.get("fusion_decision"))
    predicted_class = evidence.get("predicted_class") or fusion.get("predicted_class")
    localization_status = str(localization.get("status") or "not_requested")
    detections = localization.get("detections") or []
    detector_returned = localization_status == "completed" and isinstance(
        detections, list
    )
    raw_quality_codes = evidence.get("image_quality_codes") or []
    quality_codes = (
        [str(code) for code in raw_quality_codes]
        if isinstance(raw_quality_codes, Sequence)
        and not isinstance(raw_quality_codes, (str, bytes))
        else []
    )
    input_provenance = " · ".join(
        value
        for value in (
            str(evidence.get("image_source_format") or "").strip(),
            str(evidence.get("input_transform_id") or "").strip(),
        )
        if value
    )
    runtime_kind = _assessment_runtime_kind(
        case,
        capability_synthetic=synthetic,
        runtime_verified=runtime_verified,
    )
    classifier_name = {
        "real": f"{VISION_MODEL_DISPLAY_NAME} 三分类器（真实运行时）",
        "synthetic": "合成三分类器（mock）",
        "unverified": "三分类器（来源未验证）",
    }[runtime_kind]
    detector_name = {
        "real": "D-FINE 定位检测器（真实运行时）",
        "synthetic": "合成定位器（mock）",
        "unverified": "定位器（来源未验证）",
    }[runtime_kind]
    rows = [
        {
            "步骤": "01",
            "执行模块": "图像传输与解码校验",
            "状态": _status_label(evidence.get("image_quality_status")),
            "说明": " · ".join(
                value
                for value in (
                    input_provenance,
                    "、".join(quality_codes),
                )
                if value
            )
            or "仅表示文件可被系统处理",
        },
        {
            "步骤": "02",
            "执行模块": classifier_name,
            "状态": "已完成" if predicted_class else "未返回分类结果",
            "说明": VISION_MODEL_DISPLAY_NAME,
        },
        {
            "步骤": "03",
            "执行模块": detector_name,
            "状态": _status_label(localization_status),
            "说明": (
                f"返回 {len(detections)} 个候选框"
                if detector_returned
                else "已执行，未返回候选框"
                if localization_status == "completed_no_detection"
                else "—"
            ),
        },
        {
            "步骤": "04",
            "执行模块": "确定性策略分流",
            "状态": "已完成" if fusion else "未返回决策证据",
            "说明": str(fusion.get("policy_id") or "—"),
        },
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        f"运行标识：{evidence.get('run_id', '—')} · "
        "界面只展示可审计的工具动作与结果，不展示模型内部思维过程。"
    )


def show_provenance(case: Mapping[str, Any]) -> None:
    evidence = _mapping(case.get("vision_evidence"))
    localization = _localization_evidence(case)
    fusion = _mapping(case.get("fusion_decision"))
    provenance = {
        "case_id": case.get("case_id"),
        "classification_run_id": evidence.get("run_id"),
        "localization_status": localization.get("status", "not_requested"),
        "localization_run_id": localization.get("run_id"),
        "image_source_format": evidence.get("image_source_format"),
        "input_transform_id": evidence.get("input_transform_id"),
        "classifier_model": VISION_MODEL_DISPLAY_NAME,
        "classifier_checkpoint_sha256": _short_hash(evidence.get("classifier_checkpoint_sha256")),
        "detector_model": localization.get("detector_model_id"),
        "detector_checkpoint_sha256": _short_hash(
            localization.get("detector_checkpoint_sha256")
        ),
        "localization_preprocessing_version": localization.get("preprocessing_version"),
        "policy_id": fusion.get("policy_id"),
        "clinical_validation": fusion.get("clinical_validation"),
    }
    st.json(provenance, expanded=False)


def show_assessment_evidence(
    result: Mapping[str, Any],
    image_bytes: bytes | None,
    manifest: Mapping[str, Any] | None,
    *,
    synthetic: bool = False,
    runtime_verified: bool = False,
) -> None:
    case = _mapping(result.get("case"))
    if not case:
        st.warning("API 未返回病例证据。")
        return
    show_assessment_summary(
        result,
        synthetic=synthetic,
        runtime_verified=runtime_verified,
    )
    show_classifier_evidence(case, manifest)
    show_detection_evidence(case, image_bytes)
    show_execution_trace(
        case,
        synthetic=synthetic,
        runtime_verified=runtime_verified,
    )
    show_provenance(case)


st.set_page_config(
    page_title="TBX-Agent · 病例工作台",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)
load_ui_styles()

if "user_id" not in st.session_state:
    st.session_state.user_id = f"local-user-{uuid.uuid4().hex[:8]}"
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread-{uuid.uuid4().hex}"
if "owner_scope" not in st.session_state:
    st.session_state.owner_scope = f"local-demo:{st.session_state.user_id}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploader_generation" not in st.session_state:
    st.session_state.uploader_generation = 0
if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "local_medgemma"
elif st.session_state.llm_provider == "local_qwen":
    # Migrate an existing browser session without breaking the old API alias.
    st.session_state.llm_provider = "local_medgemma"
if (
    "llm-provider-selection" not in st.session_state
    or st.session_state["llm-provider-selection"] == "local_qwen"
):
    st.session_state["llm-provider-selection"] = st.session_state.llm_provider
navigation_options = ["病例工作台", "病例详情", "批量筛查"]
if st.session_state.get("active_batch_id"):
    navigation_options.append("复核工作台")
if st.session_state.get("navigation") not in navigation_options:
    st.session_state.navigation = "病例工作台"

health_payload, capabilities_payload, manifest_payload = _system_snapshot()
required_components_ready = bool(
    health_payload.get("required_components_ready", health_payload.get("status") == "ok")
)
synthetic_vision = health_payload.get("vision_backend") == "mock"
vision_capability_seen = False
vision_capability_real = False
llm_capability_seen = False
llm_capability_ready = False
for raw_component in capabilities_payload.get("components") or []:
    component = _mapping(raw_component)
    if component.get("component_id") == "rank03_image_assessment":
        vision_capability_seen = True
        synthetic_vision = synthetic_vision or bool(component.get("synthetic"))
        vision_capability_real = bool(
            not component.get("synthetic") and component.get("state") == "ready"
        )
    elif component.get("component_id") == "llm_evidence_composer":
        llm_capability_seen = True
        llm_capability_ready = bool(
            component.get("state") == "ready" and component.get("loaded") is True
        )
real_vision_runtime_verified = bool(
    health_payload
    and required_components_ready
    and capabilities_payload.get("runtime_verified") is True
    and vision_capability_seen
    and vision_capability_real
    and not synthetic_vision
    and health_payload.get("vision_backend") == "rank03"
)
local_llm_ready = bool(
    health_payload
    and health_payload.get("narrator_backend") == "llama_cpp"
    and (
        llm_capability_ready
        if llm_capability_seen
        else required_components_ready
        and str(health_payload.get("mode", "")).startswith("real_rank03")
    )
)
selected_llm_provider = str(
    st.session_state.get("llm_provider") or "local_medgemma"
)
selected_llm_ready = (
    local_llm_ready
    if selected_llm_provider == "local_medgemma"
    else _remote_llm_connection_ready()
)
demo_runtime = _is_explicit_demo(health_payload, capabilities_payload)
deterministic_demo = bool(
    demo_runtime
    and health_payload.get("narrator_backend") == "none"
    and selected_llm_provider == "local_medgemma"
)
chat_ready = bool(health_payload and (selected_llm_ready or deterministic_demo))

with st.sidebar:
    st.markdown(
        """
        <div class="tbx-sidebar-brand">
          <strong>TBX-Agent 控制台</strong>
          <span>病例上下文、运行状态与本地会话</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    current_page = st.radio(
        "导航",
        navigation_options,
        label_visibility="collapsed",
        key="navigation",
    )
    st.markdown("##### 当前工作区")
    current_case = st.session_state.get("case_id")
    sidebar_api_status = "在线" if health_payload else "离线"
    sidebar_backend = (
        "合成演示"
        if synthetic_vision
        else VISION_MODEL_DISPLAY_NAME
    )
    current_image_hash = st.session_state.get("current_image_sha256")
    current_assessment = st.session_state.get("assessment_result")
    sidebar_case = (
        _short_hash(current_case)
        if current_case and _assessment_matches_current_session(current_assessment)
        else "待分析"
        if current_image_hash
        else "尚未建立"
    )
    sidebar_messages = len(st.session_state.chat_history)
    st.markdown(
        f"""
        <div class="tbx-sidebar-status"><span>API</span>
          <strong>{_escape(sidebar_api_status)}</strong></div>
        <div class="tbx-sidebar-status"><span>影像后端</span>
          <strong>{_escape(sidebar_backend)}</strong></div>
        <div class="tbx-sidebar-status"><span>病例</span>
          <strong>{_escape(sidebar_case)}</strong></div>
        <div class="tbx-sidebar-status"><span>会话消息</span>
          <strong>{sidebar_messages}</strong></div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("系统与模型状态"):
        show_module_status(health_payload, manifest_payload, capabilities_payload)
    with st.expander("本地会话身份"):
        st.text_input("用户 ID", value=st.session_state.user_id, disabled=True)
        st.text_input("Owner scope", value=st.session_state.owner_scope, disabled=True)
        st.caption("生产部署中这两个值必须由认证令牌派生；普通用户不可编辑。")
        if st.button("重建演示身份", use_container_width=True):
            _clear_case_state(keep_identity=False)
            st.rerun()
    st.caption("新建病例只清理当前浏览器工作区，不删除后端审计记录。")

render_hero(health_payload, manifest_payload)

if not health_payload:
    st.error("API 当前不可用。可以查看已保留的本地结果，但不能提交新的筛查或问询。")
elif deterministic_demo:
    st.info(
        "DEMO / MOCK · 合成演示：可体验指南问答、主动筛查和模拟影像工具。"
        "不加载视觉或语言模型，影像结果不是实际模型推理。"
    )
elif not selected_llm_ready:
    if selected_llm_provider == "openai_compatible":
        st.error("请在右上角设置中连接并测试 OpenAI 兼容 API。")
    else:
        st.error("本地 MedGemma 未就绪，请检查模型服务或切换 OpenAI 兼容 API。")
elif not real_vision_runtime_verified:
    st.warning(
        f"{VISION_MODEL_DISPLAY_NAME} 当前未就绪；通用问答仍可使用，胸片相关工具会明确返回不可用。"
    )

if current_page == "病例工作台":
    assistant_column, image_column = st.columns([1.2, 1], gap="medium")

    # Render the visual workspace first so a newly uploaded image is available to the
    # chat turn in the same Streamlit rerun. Uploading only changes the active canvas;
    # model execution begins when the user sends a message.
    with image_column:
        render_llm_settings(
            local_ready=local_llm_ready,
            service_available=bool(health_payload),
        )
        with st.container(border=True, key="image-panel"):
            st.markdown(
                '<div class="tbx-panel-label"><span>▧</span> Image</div>',
                unsafe_allow_html=True,
            )
            current_image_bytes = st.session_state.get("assessment_image_bytes")
            current_preview_bytes = st.session_state.get("assessment_preview_bytes")
            if isinstance(current_image_bytes, bytes) and not isinstance(
                current_preview_bytes, bytes
            ):
                try:
                    current_preview_bytes = _upload_preview(current_image_bytes)
                except ImageValidationError:
                    current_preview_bytes = None
                else:
                    st.session_state.assessment_preview_bytes = current_preview_bytes
            current_image_hash = st.session_state.get("current_image_sha256")
            assessment_result = st.session_state.get("assessment_result")
            result_matches_selection = _assessment_matches_current_session(assessment_result)
            selected_case = _mapping(_mapping(assessment_result).get("case"))
            selected_case_id = str(selected_case.get("case_id") or "")
            anatomy_boundary = st.session_state.get("anatomy_boundary_bytes")
            anatomy_contours = st.session_state.get("anatomy_contour_bytes")
            anatomy_run = _mapping(st.session_state.get("anatomy_run"))
            localization_receipt_matches = bool(
                result_matches_selection
                and st.session_state.get("localization_receipt_case_id")
                == selected_case_id
                and st.session_state.get("localization_receipt_image_sha256")
                == current_image_hash
            )
            anatomy_receipt_matches = bool(
                result_matches_selection
                and st.session_state.get("anatomy_receipt_case_id")
                == selected_case_id
                and st.session_state.get("anatomy_receipt_image_sha256")
                == current_image_hash
            )
            anatomy_matches_selection = bool(
                anatomy_receipt_matches
                and isinstance(anatomy_boundary, bytes)
                and st.session_state.get("anatomy_case_id") == selected_case_id
                and st.session_state.get("anatomy_image_sha256") == current_image_hash
            )
            display_mode = "原图"
            if result_matches_selection:
                raw_detections = (
                    _completed_localization_view(selected_case).get("detections")
                    if localization_receipt_matches
                    else []
                )
                has_detections = bool(
                    isinstance(raw_detections, Sequence)
                    and not isinstance(raw_detections, (str, bytes))
                    and raw_detections
                )
                display_choices = ["原图"]
                if has_detections:
                    display_choices.append("候选框")
                if anatomy_matches_selection:
                    display_choices.append("肺野")
                if anatomy_matches_selection and isinstance(anatomy_contours, bytes):
                    display_choices.append("轮廓")
                if anatomy_matches_selection and (
                    has_detections or isinstance(anatomy_contours, bytes)
                ):
                    display_choices.append("组合")
                pending_display_mode = st.session_state.pop(
                    "pending_anatomy_display_mode", None
                )
                if pending_display_mode in display_choices:
                    st.session_state.anatomy_display_mode = pending_display_mode
                elif st.session_state.get("anatomy_display_mode") not in display_choices:
                    st.session_state.anatomy_display_mode = "原图"
                display_mode = st.radio(
                    "影像显示",
                    display_choices,
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="anatomy_display_mode",
                )
            uploader_key = f"cxr-uploader-{st.session_state.uploader_generation}"
            uploaded = None

            if isinstance(current_image_bytes, bytes) and isinstance(current_preview_bytes, bytes):
                show_workspace_image(
                    current_preview_bytes,
                    _mapping(assessment_result),
                    result_matches_image=result_matches_selection,
                    anatomy_boundary_bytes=(
                        anatomy_boundary if anatomy_matches_selection else None
                    ),
                    anatomy_contour_bytes=(
                        anatomy_contours
                        if anatomy_matches_selection and isinstance(anatomy_contours, bytes)
                        else None
                    ),
                    display_mode=display_mode,
                )
                with st.popover("上传 / 更换胸片", use_container_width=True):
                    uploaded = st.file_uploader(
                        "上传胸片",
                        type=["png", "jpg", "jpeg", "dcm", "dicom"],
                        label_visibility="collapsed",
                        key=uploader_key,
                    )
            else:
                uploaded = st.file_uploader(
                    "上传胸片",
                    type=["png", "jpg", "jpeg", "dcm", "dicom"],
                    label_visibility="collapsed",
                    key=uploader_key,
                )

            if uploaded is not None:
                uploaded_bytes = uploaded.getvalue()
                uploaded_sha256 = hashlib.sha256(uploaded_bytes).hexdigest()
                image_valid = False
                preview_bytes = b""
                try:
                    preview_bytes = _upload_preview(uploaded_bytes)
                except ImageValidationError as exc:
                    st.error(f"文件未通过本地输入校验：{exc}")
                else:
                    image_valid = True
                if image_valid and uploaded_sha256 != current_image_hash:
                    _activate_uploaded_image(
                        image_bytes=uploaded_bytes,
                        preview_bytes=preview_bytes,
                        image_name=uploaded.name,
                        image_sha256=uploaded_sha256,
                    )
                    st.rerun()
            if not isinstance(current_image_bytes, bytes):
                st.caption(
                    "上传图像后，在左侧输入问题即可体验模拟影像工具。"
                    if demo_runtime
                    else "上传胸片后，在左侧输入问题即可开始真实推理。"
                )

            if anatomy_receipt_matches:
                if st.session_state.get("anatomy_technical_failure"):
                    st.caption("肺野边界：技术失败（不影响既有分类结果）")
                elif anatomy_matches_selection:
                    refinement_status = str(anatomy_run.get("refinement_status") or "disabled")
                    if refinement_status == "technical_failure":
                        st.caption("肺野已完成；可选轮廓细化暂不可用，不影响分类分流。")
                    elif refinement_status == "completed":
                        refinement_evidence = _mapping(
                            anatomy_run.get("refinement_evidence")
                        )
                        capacity_abstained = int(
                            refinement_evidence.get("capacity_abstained_count") or 0
                        )
                        capacity_note = (
                            f"另有 {capacity_abstained} 个候选框因单次处理上限未生成轮廓。"
                            if capacity_abstained
                            else ""
                        )
                        st.caption(
                            "轮廓仅供可视化复核，未经病灶分割临床验证。"
                            + capacity_note
                        )
                    # Structured spatial relationships are available on the
                    # second-level case detail page, not on the workbench.

        reset_left, reset_right = st.columns(2)
        reset_left.button(
            "清空对话",
            use_container_width=True,
            disabled=not bool(st.session_state.get("chat_history")),
            on_click=_clear_chat_state,
        )
        reset_right.button(
            "新建病例",
            use_container_width=True,
            on_click=_clear_case_state,
        )

    current_image_bytes = st.session_state.get("assessment_image_bytes")
    current_preview_bytes = st.session_state.get("assessment_preview_bytes")
    current_image_name = str(st.session_state.get("assessment_image_name") or "cxr.png")
    current_image_hash = st.session_state.get("current_image_sha256")
    assessment_result = st.session_state.get("assessment_result")
    result_matches_selection = _assessment_matches_current_session(assessment_result)
    current_case_payload = _mapping(_mapping(assessment_result).get("case"))
    stored_case_id = str(st.session_state.get("case_id") or "")
    evidence_case_id = str(current_case_payload.get("case_id") or "")
    current_assessment_runtime_kind = _assessment_runtime_kind(
        current_case_payload,
        capability_synthetic=synthetic_vision,
        runtime_verified=real_vision_runtime_verified,
    )
    current_case_has_classification = _has_classification_evidence(
        current_case_payload
    )
    assessment_ready_for_followup = bool(
        result_matches_selection
        and stored_case_id
        and stored_case_id == evidence_case_id
        and (
            not current_case_has_classification
            or current_assessment_runtime_kind == "real"
            or (demo_runtime and current_assessment_runtime_kind == "synthetic")
        )
    )
    pending_image_assessment = bool(
        isinstance(current_image_bytes, bytes) and not assessment_ready_for_followup
    )
    with assistant_column:
        with st.container(border=True, key="agent-panel"):
            st.markdown(
                '<div class="tbx-panel-label"><span>▤</span> Agent</div>',
                unsafe_allow_html=True,
            )
            chat_view = st.container(height=620, border=False, key="chat-view")
            with chat_view:
                chat_history = list(st.session_state.chat_history)
                for message in chat_history:
                    show_chat_message(
                        _mapping(message),
                        synthetic=synthetic_vision,
                        runtime_verified=real_vision_runtime_verified,
                        show_plan=False,
                    )

                # Plans are transient turn guidance, not transcript content.
                # Keep exactly one replaceable slot for the latest completed
                # assistant turn so reruns cannot replay every historical plan.
                current_plan_slot = st.empty()
                latest_plan_payload = _latest_chat_plan_payload(chat_history)
                if latest_plan_payload:
                    with current_plan_slot.container():
                        show_initial_plan(latest_plan_payload)

                screening_collecting, quick_prompt = render_chat_screening_controls(
                    service_available=bool(health_payload),
                    assessment_ready=assessment_ready_for_followup,
                    image_bytes=(
                        current_image_bytes
                        if isinstance(current_image_bytes, bytes)
                        else None
                    ),
                    image_name=current_image_name,
                    image_sha256=str(current_image_hash or "") or None,
                    capability_synthetic=synthetic_vision,
                    runtime_verified=real_vision_runtime_verified,
                )

            typed_prompt = st.chat_input(
                "输入问题，或询问当前胸片…",
                key="case-chat-input",
                disabled=not chat_ready or screening_collecting,
            )

        pending_prompt = quick_prompt or typed_prompt
        pending_prompt = pending_prompt.strip() if pending_prompt else ""
        if pending_prompt:
            # Remove the previous turn's plan before rendering the new one in
            # the live assistant message below.
            current_plan_slot.empty()
            matching_case_id = (
                st.session_state.get("case_id") if assessment_ready_for_followup else None
            )
            needs_assessment = pending_image_assessment
            user_message: dict[str, Any] = {
                "role": "user",
                "content": pending_prompt,
            }
            if needs_assessment:
                user_message.update(
                    {
                        "image_bytes": current_preview_bytes,
                        "image_name": current_image_name,
                        "image_sha256": current_image_hash,
                    }
                )
            st.session_state.chat_history.append(user_message)

            assessment_for_message: Mapping[str, Any] = {}
            answer_payload: Any = None
            turn_error = ""
            case_refreshed_after_tool = False
            with chat_view:
                show_chat_message(
                    user_message,
                    synthetic=synthetic_vision,
                    runtime_verified=real_vision_runtime_verified,
                )
                with st.chat_message("assistant"):
                    live_plan_slot = st.empty()
                    with live_plan_slot.status("正在处理", expanded=False) as live_plan:
                        if needs_assessment:
                            new_assessment = call(
                                "POST",
                                "/v1/assessments/cxr",
                                files={
                                    "file": (
                                        current_image_name,
                                        current_image_bytes,
                                        _image_mime_type(current_image_name),
                                    )
                                },
                                data={
                                    "user_id": st.session_state.user_id,
                                    "owner_scope": st.session_state.owner_scope,
                                    "consent_to_process": "true",
                                    "attested_chest_radiograph": "true",
                                },
                            )
                            new_case = _mapping(_mapping(new_assessment).get("case"))
                            new_case_id = new_case.get("case_id")
                            if new_assessment and new_case_id:
                                assessment_for_message = _mapping(new_assessment)
                                st.session_state.assessment_result = new_assessment
                                st.session_state.assessment_upload_sha256 = current_image_hash
                                st.session_state.assessment_user_id = st.session_state.user_id
                                st.session_state.assessment_owner_scope = (
                                    st.session_state.owner_scope
                                )
                                for report_key in (
                                    "generated_report",
                                    "generated_report_markdown",
                                    "generated_report_json",
                                ):
                                    st.session_state.pop(report_key, None)
                                for anatomy_key in (
                                    "anatomy_run",
                                    "anatomy_boundary_bytes",
                                    "anatomy_contour_bytes",
                                    "anatomy_case_id",
                                    "anatomy_image_sha256",
                                    "anatomy_technical_failure",
                                    "anatomy_display_mode",
                                    "pending_anatomy_display_mode",
                                    "localization_receipt_case_id",
                                    "localization_receipt_image_sha256",
                                    "anatomy_receipt_case_id",
                                    "anatomy_receipt_image_sha256",
                                ):
                                    st.session_state.pop(anatomy_key, None)
                                returned_runtime_kind = _assessment_runtime_kind(
                                    new_case,
                                    capability_synthetic=synthetic_vision,
                                    runtime_verified=real_vision_runtime_verified,
                                )
                                returned_visual_result = _assessment_visual_result(
                                    _mapping(new_assessment)
                                )
                                returned_has_classification = (
                                    _has_classification_evidence(new_case)
                                )
                                if (
                                    returned_has_classification
                                    and returned_runtime_kind not in (
                                        {"real", "synthetic"} if demo_runtime else {"real"}
                                    )
                                ):
                                    st.session_state.pop("case_id", None)
                                    st.session_state.pop("assessment_confirmation_sha256", None)
                                    turn_error = (
                                        f"影像 API 返回的模型证据不是已验证的真实 "
                                        f"{VISION_MODEL_DISPLAY_NAME} 运行结果；"
                                        "本次没有调用后续 Agent。"
                                    )
                                elif (
                                    returned_has_classification
                                    and returned_visual_result == "technical_failure"
                                ):
                                    st.session_state.pop("case_id", None)
                                    st.session_state.assessment_confirmation_sha256 = (
                                        current_image_hash
                                    )
                                    turn_error = (
                                        "影像处理发生技术失败，本次没有调用后续 Agent。"
                                        "请排查后重试。"
                                    )
                                else:
                                    matching_case_id = new_case_id
                                    st.session_state.case_id = new_case_id
                                    st.session_state.assessment_confirmation_sha256 = (
                                        current_image_hash
                                    )
                            else:
                                st.session_state.pop("case_id", None)
                                st.session_state.pop("assessment_result", None)
                                st.session_state.pop("assessment_upload_sha256", None)
                                st.session_state.pop("assessment_confirmation_sha256", None)
                                turn_error = (
                                    "胸片载入未完成，本次没有调用后续 Agent。请检查服务状态后重试。"
                                )

                        if not turn_error:
                            answer_payload = call(
                                "POST",
                                "/v1/agent/respond",
                                json={
                                    "thread_id": st.session_state.thread_id,
                                    "user_id": st.session_state.user_id,
                                    "owner_scope": st.session_state.owner_scope,
                                    "message": pending_prompt,
                                    "case_id": matching_case_id,
                                    **_agent_llm_payload(),
                                },
                            )
                            if not answer_payload:
                                turn_error = (
                                    "影像结果已保留，但当前无法取得 Agent 回答；可以直接再次提问。"
                                )
                            elif matching_case_id:
                                answer = _mapping(answer_payload)
                                classification_used = _used_tool(
                                    answer, "classify_current_cxr"
                                )
                                localization_used = _used_tool(
                                    answer, "localize_current_cxr"
                                )
                                if classification_used or localization_used:
                                    refreshed_case = call(
                                        "GET",
                                        f"/v1/cases/{matching_case_id}",
                                        params={
                                            "owner_scope": st.session_state.owner_scope,
                                            "user_id": st.session_state.user_id,
                                        },
                                    )
                                    if refreshed_case:
                                        refreshed_assessment = dict(
                                            _mapping(
                                                st.session_state.get("assessment_result")
                                            )
                                        )
                                        refreshed_assessment["case"] = dict(
                                            _mapping(refreshed_case)
                                        )
                                        st.session_state.assessment_result = (
                                            refreshed_assessment
                                        )
                                        if assessment_for_message:
                                            assessment_for_message = refreshed_assessment
                                        if localization_used:
                                            st.session_state.localization_receipt_case_id = (
                                                matching_case_id
                                            )
                                            st.session_state.localization_receipt_image_sha256 = (
                                                current_image_hash
                                            )
                                            refreshed_detections = (
                                                _completed_localization_view(
                                                    _mapping(refreshed_case)
                                                ).get("detections")
                                            )
                                            if (
                                                isinstance(
                                                    refreshed_detections, Sequence
                                                )
                                                and not isinstance(
                                                    refreshed_detections, (str, bytes)
                                                )
                                                and refreshed_detections
                                            ):
                                                st.session_state.pending_anatomy_display_mode = (
                                                    "候选框"
                                                )
                                        case_refreshed_after_tool = True
                                if (
                                    _used_tool(answer, "inspect_anatomical_context")
                                    and isinstance(current_image_bytes, bytes)
                                    and current_image_hash
                                ):
                                    st.session_state.anatomy_receipt_case_id = (
                                        matching_case_id
                                    )
                                    st.session_state.anatomy_receipt_image_sha256 = (
                                        current_image_hash
                                    )
                                    _run_anatomy_layer(
                                        case_id=str(matching_case_id),
                                        image_name=current_image_name,
                                        image_bytes=current_image_bytes,
                                        image_sha256=str(current_image_hash),
                                    )
                                    case_refreshed_after_tool = True
                        live_plan.update(
                            label=(
                                "处理完成"
                                if not turn_error
                                else "处理未完成"
                            ),
                            state="complete" if not turn_error else "error",
                            expanded=False,
                        )
                    live_plan_slot.empty()

                    suppress_payload = bool(
                        assessment_for_message
                        and answer_payload
                        and _should_hide_initial_agent_payload(
                            pending_prompt, _mapping(answer_payload)
                        )
                    )
                    show_tool_chain(_mapping(answer_payload))
                    if (
                        assessment_for_message
                        and _has_classification_evidence(
                            _mapping(assessment_for_message.get("case"))
                        )
                        and (not answer_payload or suppress_payload)
                    ):
                        show_compact_assessment(
                            assessment_for_message,
                            synthetic=synthetic_vision,
                            runtime_verified=real_vision_runtime_verified,
                        )
                    if answer_payload and not suppress_payload:
                        show_chat_response(_mapping(answer_payload))
                    elif turn_error:
                        st.error(turn_error)

            assistant_message: dict[str, Any] = {"role": "assistant"}
            if assessment_for_message:
                assistant_message["assessment"] = assessment_for_message
            if answer_payload:
                assistant_message["payload"] = answer_payload
                assistant_message["suppress_payload"] = suppress_payload
            if turn_error:
                assistant_message["error"] = turn_error
            st.session_state.chat_history.append(assistant_message)
            if assessment_for_message or case_refreshed_after_tool:
                # The image column rendered earlier in this run; refresh it once so the
                # new overlay and optional anatomy control appear immediately.
                st.rerun()

elif current_page == "批量筛查":
    render_section_header(
        "Batch screening",
        "批量筛查",
        "一次上传多张胸片；非健康类别与不确定结果可进入本批次复核队列。",
    )
    batch_files = st.file_uploader(
        "上传胸片或 DICOM",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        accept_multiple_files=True,
        key="batch-cxr-uploader",
        help="支持 PNG、JPEG 与受支持的单帧 CR/DX DICOM。",
    )
    batch_action, new_batch_action = st.columns([3, 1])
    run_batch = batch_action.button(
        "开始批量筛查",
        type="primary",
        use_container_width=True,
        disabled=not bool(batch_files) or not real_vision_runtime_verified,
    )
    if new_batch_action.button("新建批次", use_container_width=True):
        st.session_state.pop("active_batch_id", None)
        st.session_state.pop("batch_results", None)
        st.session_state.pop("pending_reviews", None)
        st.rerun()

    if run_batch:
        batch_id = str(
            st.session_state.get("active_batch_id")
            or f"batch-{uuid.uuid4().hex[:16]}"
        )
        st.session_state.active_batch_id = batch_id
        rows: list[dict[str, Any]] = []
        progress = st.progress(0, text="正在执行真实影像推理…")
        total = len(batch_files)
        for index, uploaded in enumerate(batch_files, start=1):
            payload = uploaded.getvalue()
            item_digest = hashlib.sha256(payload).hexdigest()[:12]
            batch_item_id = f"item-{index:04d}-{item_digest}"
            result = call(
                "POST",
                f"/v1/batches/{batch_id}/assessments/cxr",
                files={
                    "file": (
                        uploaded.name,
                        payload,
                        _image_mime_type(uploaded.name),
                    )
                },
                data={
                    "batch_item_id": batch_item_id,
                    "user_id": st.session_state.user_id,
                    "owner_scope": st.session_state.owner_scope,
                    "consent_to_process": "true",
                    "attested_chest_radiograph": "true",
                },
            )
            result_payload = _mapping(result)
            case = _mapping(result_payload.get("case"))
            response = _mapping(result_payload.get("response"))
            fusion = _mapping(case.get("fusion_decision"))
            visual_result = str(
                response.get("visual_result") or fusion.get("visual_result") or ""
            )
            predicted_class = str(
                response.get("predicted_class") or fusion.get("predicted_class") or ""
            )
            rows.append(
                {
                    "文件": uploaded.name,
                    "结果": VISUAL_RESULT_LABELS.get(
                        visual_result,
                        "处理失败" if not result_payload else visual_result,
                    ),
                    "训练类别": CLASS_LABELS.get(predicted_class, predicted_class or "—"),
                    "进入复核": "是" if result_payload.get("enqueued_for_review") else "否",
                    "病例": _short_hash(case.get("case_id")) if case else "—",
                    "batch_item_id": batch_item_id,
                }
            )
            progress.progress(index / total, text=f"已完成 {index}/{total}")
        progress.empty()
        st.session_state.batch_results = rows
        st.session_state.pop("pending_reviews", None)
        st.rerun()

    batch_rows = st.session_state.get("batch_results") or []
    active_batch_id = str(st.session_state.get("active_batch_id") or "")
    if batch_rows:
        queued_count = sum(row.get("进入复核") == "是" for row in batch_rows)
        completed_count = len(batch_rows)
        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("已处理", completed_count)
        metric_b.metric("进入复核", queued_count)
        metric_c.metric("无需复核", completed_count - queued_count)
        st.dataframe(batch_rows, hide_index=True, width="stretch")
        st.caption(f"批次 {active_batch_id} · 复核入队与影像推理相互独立，可安全重试。")
        if queued_count and st.button(
            "打开本批次复核工作台",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.navigation = "复核工作台"
            st.rerun()
    else:
        st.info("选择多张胸片后即可开始；单病例分析仍在首页对话框中完成。")

elif current_page == "病例详情":
    render_section_header(
        "Case details",
        "病例详情",
        "模型原始证据、执行回执、指南引用与报告下载集中在这里。",
    )
    detail_assessment = _mapping(st.session_state.get("assessment_result"))
    if not _assessment_matches_current_session(detail_assessment):
        st.info("当前还没有可查看的病例。请先回到病例工作台上传胸片并发起分析。")
    else:
        detail_case = _mapping(detail_assessment.get("case"))
        detail_preview = st.session_state.get("assessment_preview_bytes")
        show_assessment_evidence(
            detail_assessment,
            detail_preview if isinstance(detail_preview, bytes) else None,
            manifest_payload,
            synthetic=synthetic_vision,
            runtime_verified=real_vision_runtime_verified,
        )
        anatomy_detail = _mapping(st.session_state.get("anatomy_run"))
        if anatomy_detail:
            st.markdown("#### 肺野分割与空间关系")
            show_anatomy_spatial_evidence(anatomy_detail)

        latest_agent_payload: Mapping[str, Any] = {}
        for chat_item in reversed(st.session_state.chat_history):
            candidate = _mapping(_mapping(chat_item).get("payload"))
            if candidate:
                latest_agent_payload = candidate
                break
        if latest_agent_payload:
            st.markdown("#### 最近一次 Agent 回答的工具与依据")
            show_response(dict(latest_agent_payload), show_summary=False)

        detail_runtime_kind = _assessment_runtime_kind(
            detail_case,
            capability_synthetic=synthetic_vision,
            runtime_verified=real_vision_runtime_verified,
        )
        report_ready = bool(
            detail_runtime_kind == "real"
            and _assessment_visual_result(detail_assessment) != "technical_failure"
            and st.session_state.get("case_id")
        )
        st.markdown("#### 结构化报告")
        if not report_ready:
            st.warning("当前结果不满足真实运行时报告生成条件。")
        elif st.button("生成报告文件", key="generate-report"):
            report = call(
                "POST",
                f"/v1/cases/{st.session_state.case_id}/reports",
                json={
                    "owner_scope": st.session_state.owner_scope,
                    "user_id": st.session_state.user_id,
                    "actor_id": st.session_state.user_id,
                },
            )
            if report:
                st.session_state.generated_report = report
                st.session_state.generated_report_markdown = fetch_artifact(
                    report["markdown_download_url"]
                )
                st.session_state.generated_report_json = fetch_artifact(
                    report["json_download_url"]
                )
        generated_report = st.session_state.get("generated_report") if report_ready else None
        if isinstance(generated_report, Mapping):
            st.success(f"报告已生成：{generated_report.get('report_id', '—')}")
            markdown_column, json_column = st.columns(2)
            markdown_bytes = st.session_state.get("generated_report_markdown")
            json_bytes = st.session_state.get("generated_report_json")
            if markdown_bytes:
                markdown_column.download_button(
                    "下载 Markdown 报告",
                    data=markdown_bytes,
                    file_name="tbx-agent-report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            if json_bytes:
                json_column.download_button(
                    "下载 JSON 报告",
                    data=json_bytes,
                    file_name="tbx-agent-report.json",
                    mime="application/json",
                    use_container_width=True,
                )

elif current_page == "复核工作台":
    active_batch_id = str(st.session_state.get("active_batch_id") or "")
    render_section_header(
        "Reviewer workspace",
        "批量筛查复核工作台",
        f"仅显示当前批次 {active_batch_id} 自动入队的项目。",
    )
    if not active_batch_id:
        st.info("请先完成一个批量筛查任务。")
        st.stop()
    if st.button("刷新本批次队列", disabled=not bool(health_payload)):
        pending = call(
            "GET",
            "/v1/reviews/pending",
            params={
                "owner_scope": st.session_state.owner_scope,
                "user_id": st.session_state.user_id,
            },
        )
        st.session_state.pending_reviews = [
            item
            for item in (pending or [])
            if _mapping(item).get("batch_id") == active_batch_id
        ]
    if "pending_reviews" not in st.session_state:
        pending = call(
            "GET",
            "/v1/reviews/pending",
            show_error=False,
            params={
                "owner_scope": st.session_state.owner_scope,
                "user_id": st.session_state.user_id,
            },
        )
        st.session_state.pending_reviews = [
            item
            for item in (pending or [])
            if _mapping(item).get("batch_id") == active_batch_id
        ]
    pending_reviews = st.session_state.get("pending_reviews", [])
    st.metric("当前待复核", f"{len(pending_reviews)} 项")
    if not pending_reviews:
        st.info("当前批次没有待复核项目。")
    decision_labels = {
        "keep_model_flagged": "保留筛查标记",
        "keep_model_not_flagged": "改为不标记",
        "indeterminate": "仍不确定",
        "technical_repeat_required": "需要重新采集/处理",
    }
    for review in pending_reviews:
        review_payload = _mapping(review)
        with st.expander(
            f"{review_payload.get('batch_item_id', '批次项目')} · "
            f"病例 {_short_hash(review_payload.get('case_id'))}"
        ):
            st.caption("入队原因：" + "、".join(review_payload.get("trigger_reasons") or []))
            decision = st.selectbox(
                "复核结论",
                list(decision_labels),
                format_func=decision_labels.get,
                key=f"review-decision-{review_payload.get('review_id')}",
            )
            note = st.text_area(
                "备注（可选）",
                key=f"review-note-{review_payload.get('review_id')}",
                max_chars=2_000,
            )
            if st.button(
                "完成复核",
                key=f"complete-review-{review_payload.get('review_id')}",
            ):
                completed = call(
                    "PATCH",
                    f"/v1/reviews/{review_payload.get('review_id')}",
                    json={
                        "owner_scope": st.session_state.owner_scope,
                        "user_id": st.session_state.user_id,
                        "reviewer_id": st.session_state.user_id,
                        "expected_version": review_payload.get("version"),
                        "decision": decision,
                        "note": note or None,
                    },
                )
                if completed:
                    st.session_state.pending_reviews = [
                        item
                        for item in pending_reviews
                        if _mapping(item).get("review_id")
                        != review_payload.get("review_id")
                    ]
                    st.rerun()
            with st.expander("技术记录"):
                st.caption(
                    f"复核记录 {review_payload.get('review_id', '—')} · "
                    f"版本 {review_payload.get('version', '—')}"
                )
                st.json(review_payload, expanded=False)
