from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tbx_agent.api.main import create_app
from tbx_agent.capability_answer import TBX_CAPABILITY_ANSWER
from tbx_agent.config import Settings
from tbx_agent.narrator import (
    NARRATOR_POLICY_ID,
    NarrationError,
    OllamaNarrator,
    SafeGuidelineNarration,
    _approved_payload,
    complete_general_chat,
    validate_grounded_narration,
)
from tbx_agent.schemas import (
    AgentResponse,
    Citation,
    GuidelineAnswerStatus,
    NarrationStatus,
    ResponseKind,
    Urgency,
)
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemma4:latest"
MODEL_DIGEST = "c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb"


def _png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (512, 512), color=(40, 60, 80)).save(output, format="PNG")
    return output.getvalue()


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
        openai_enabled=False,
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        retain_uploaded_image=False,
    )


def _ollama(*, base_url: str = "http://127.0.0.1:11434", **kwargs: Any) -> OllamaNarrator:
    return OllamaNarrator(
        model=MODEL,
        expected_digest=MODEL_DIGEST,
        base_url=base_url,
        timeout_seconds=2,
        max_response_bytes=4096,
        **kwargs,
    )


def _authoritative_response() -> AgentResponse:
    return AgentResponse(
        request_id="request-secret",
        trace_id="trace-secret",
        thread_id="thread-secret",
        case_id="case-secret",
        response_kind=ResponseKind.NEXT_TEST_INFORMATION,
        summary="肺结核判断需要完整证据链，本系统不能确诊或排除肺结核。",
        limitations=["本系统不能确诊或排除肺结核。"],
        citations=[
            Citation(
                chunk_id="private-chunk",
                source_id="private-source",
                title="测试来源",
                organization="测试机构",
                publication_year=2024,
                section="测试章节",
                locator="private-locator",
                url="https://example.invalid/private",
                support_text="private-support-text",
            )
        ],
        urgency=Urgency.PROMPT_EVALUATION,
    )


def _domain_fields(response: AgentResponse) -> dict[str, Any]:
    return response.model_dump(
        exclude={
            "summary",
            "narrator_backend",
            "narrator_model",
            "narrator_model_digest",
            "narrator_policy_id",
            "narration_status",
            "narrator_generation_invoked",
            "narrator_prompt_tokens",
            "narrator_completion_tokens",
        }
    )


class _GeneralChatGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    def complete_structured(self, **kwargs):
        self.requests.append(kwargs)
        return json.dumps({"answer": self.answer}, ensure_ascii=False), {
            "prompt_tokens": 9,
            "completion_tokens": 5,
        }


def test_general_chat_keeps_final_answer_without_reasoning_or_case_state() -> None:
    generator = _GeneralChatGenerator(
        "Analysis: The user wants a short answer.\n"
        "当前病例：分类未运行；定位未运行。\n"
        "Final Answer: 2"
    )

    answer, usage = complete_general_chat(generator, query="1+1=?")

    assert answer.answer == "2"
    assert usage == {"prompt_tokens": 9, "completion_tokens": 5}
    system_prompt = generator.requests[0]["messages"][0]["content"]
    assert "病例状态" in system_prompt
    assert "不要输出思考、分析、推理" in system_prompt


def test_general_chat_rejects_raw_internal_runtime_state() -> None:
    generator = _GeneralChatGenerator(
        'case_state={"image_loaded":true,"classification":{"status":"not_run"}}'
    )

    with pytest.raises(NarrationError, match="generation failed"):
        complete_general_chat(generator, query="1+1=?")


def test_ollama_accepts_loopback_and_rejects_remote_endpoint():
    narrator = _ollama()
    assert narrator.base_url == "http://127.0.0.1:11434"

    with pytest.raises(ValueError, match="remote Ollama"):
        _ollama(base_url="https://ollama.example.com")

    remote = _ollama(base_url="https://ollama.example.com", allow_remote=True)
    assert remote.base_url == "https://ollama.example.com"


@pytest.mark.parametrize("model", ["", "gemma4:cloud", "gemma4:latest:cloud"])
def test_ollama_rejects_empty_and_cloud_model_tags(model: str):
    with pytest.raises(ValueError):
        OllamaNarrator(
            model=model,
            expected_digest=MODEL_DIGEST,
            max_response_bytes=4096,
        )


def test_ollama_rejects_model_digest_mismatch(monkeypatch):
    narrator = OllamaNarrator(
        model=MODEL,
        expected_digest="a" * 64,
        max_response_bytes=4096,
    )

    def fake_request(path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        assert payload is None
        if path == "/api/version":
            return {"version": "0.31.2"}
        assert path == "/api/tags"
        return {"models": [{"name": MODEL, "digest": MODEL_DIGEST}]}

    monkeypatch.setattr(narrator, "_request", fake_request)
    with pytest.raises(NarrationError, match="digest"):
        narrator.provenance()


def test_manifest_exposes_only_non_sensitive_narrator_identity(tmp_path):
    settings = replace(
        _settings(tmp_path),
        narrator_backend="ollama",
        ollama_model=MODEL,
        ollama_model_digest=MODEL_DIGEST,
    )
    payload = TestClient(create_app(TBXAgentService(settings))).get("/v1/system/manifest")

    assert payload.status_code == 200
    narrator = payload.json()["narrator"]
    assert narrator == {
        "backend": "ollama",
        "model": MODEL,
        "expected_model_digest": MODEL_DIGEST,
        "policy_id": "tbx-grounded-evidence-synthesis-v2",
        "local_only_client_policy": True,
    }
    assert "127.0.0.1:11434" not in payload.text


def test_structured_ollama_narration_changes_only_summary_and_excludes_secrets(monkeypatch):
    narrator = _ollama()
    captured_payload: dict[str, Any] = {}

    def fake_request(path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if path == "/api/version":
            return {"version": "0.31.2"}
        if path == "/api/tags":
            return {
                "models": [
                    {
                        "name": MODEL,
                        "digest": MODEL_DIGEST,
                        "details": {"family": "gemma4", "parameter_size": "8.0B"},
                    }
                ]
            }
        assert path == "/api/chat"
        assert payload is not None
        captured_payload.update(payload)
        return {
            "message": {
                "content": json.dumps(
                    {"summary": ("请注意：肺结核判断需要完整证据链，本系统不能确诊或排除肺结核。")},
                    ensure_ascii=False,
                )
            }
        }

    monkeypatch.setattr(narrator, "_request", fake_request)
    original = _authoritative_response()
    rendered = narrator.narrate(original)

    assert rendered.summary.startswith("请注意：")
    assert rendered.narration_status == NarrationStatus.APPLIED
    assert rendered.narrator_backend == "ollama"
    assert rendered.narrator_model == MODEL
    assert rendered.narrator_model_digest == MODEL_DIGEST
    assert rendered.narrator_policy_id
    assert _domain_fields(rendered) == _domain_fields(original)

    request_text = json.dumps(captured_payload, ensure_ascii=False)
    assert captured_payload["stream"] is False
    assert captured_payload["think"] is False
    assert isinstance(captured_payload["format"], dict)
    for secret in (
        "request-secret",
        "trace-secret",
        "thread-secret",
        "case-secret",
        "private-chunk",
        "private-source",
        "private-locator",
        "private-support-text",
        "https://example.invalid/private",
    ):
        assert secret not in request_text


def test_visual_spatial_evidence_reaches_narrator_only_as_approved_fact() -> None:
    statement = "候选框 1 与左侧肺野的上肺野二维投影相交；这不代表肺叶或病灶诊断。"
    response = AgentResponse(
        request_id="request-1",
        trace_id="trace-1",
        thread_id="thread-1",
        case_id="case-1",
        response_kind=ResponseKind.VISUAL_SCREENING_RESULT,
        summary="模型证据仅用于辅助筛查，不能确诊或排除肺结核。",
        visual_result="pending_human_review",
        visual_evidence_notes=[statement],
        limitations=["本系统不能确诊或排除肺结核。"],
        review_status="pending",
    )

    payload = _approved_payload(response)

    assert statement in payload["approved_fact_options"]
    assert "本系统不能确诊或排除肺结核。" not in payload["approved_fact_options"]
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "counts_b64" not in encoded
    assert "bbox_xyxy" not in encoded
    assert "image_sha256" not in encoded


def test_general_capability_question_is_projected_to_tbx_agent_features() -> None:
    class _GenericCapabilityGenerator:
        def complete_structured(self, **kwargs):
            system_prompt = kwargs["messages"][0]["content"]
            assert "TBX-Agent" in system_prompt
            assert "不得泛化" in system_prompt
            return (
                json.dumps({"answer": "我是一个通用人工智能助手。"}, ensure_ascii=False),
                {"prompt_tokens": 12, "completion_tokens": 8},
            )

    answer, usage = complete_general_chat(
        _GenericCapabilityGenerator(),
        query="你会干什么？",
    )

    assert answer.answer == TBX_CAPABILITY_ANSWER
    assert "胸片" in answer.answer
    assert "指南" in answer.answer
    assert "通用人工智能" not in answer.answer
    assert usage == {"prompt_tokens": 12, "completion_tokens": 8}


class _SpyNarrator:
    backend_id = "ollama"
    model = MODEL
    model_digest = MODEL_DIGEST
    policy_id = "tbx-narrator-style-only-v1"

    def __init__(
        self,
        *,
        error: bool = False,
        unsafe: bool = False,
        mutate_protected: bool = False,
        inject_information: bool = False,
    ):
        self.calls = 0
        self.error = error
        self.unsafe = unsafe
        self.mutate_protected = mutate_protected
        self.inject_information = inject_information
        self.source_queries: list[str | None] = []

    def narrate(self, response: AgentResponse) -> AgentResponse:
        self.calls += 1
        self.source_queries.append(response.source_query)
        if self.error:
            raise NarrationError("simulated local runtime failure")
        summary = "已经确诊肺结核。" if self.unsafe else f"请注意：{response.summary}"
        updates: dict[str, Any] = {
            "summary": summary,
            "narrator_backend": self.backend_id,
            "narrator_model": self.model,
            "narrator_model_digest": self.model_digest,
            "narrator_policy_id": self.policy_id,
            "narration_status": NarrationStatus.APPLIED,
        }
        if self.mutate_protected:
            updates["limitations"] = []
        if self.inject_information:
            updates["diagnostic_information"] = ["模型擅自添加的信息。"]
        return response.model_copy(
            update={
                **updates,
            }
        )


def test_service_failure_falls_back_and_api_still_returns_200(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    narrator = _SpyNarrator(error=True)
    service.narrator = narrator

    response = TestClient(create_app(service)).post(
        "/v1/agent/respond",
        json={
            "thread_id": "thread-fallback",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "胸片异常后应该做什么检查？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert narrator.calls == 1
    assert payload["answer_status"] == GuidelineAnswerStatus.ANSWERED.value
    assert payload["claims"]
    assert payload["summary"] == payload["claims"][0]["text"]
    assert payload["claims"][0]["chunk_ids"][0] in {
        item["chunk_id"] for item in payload["retrieved_evidence"]
    }
    assert payload["narration_status"] == NarrationStatus.FALLBACK_ERROR
    assert payload["narrator_backend"] == "ollama"


def test_required_llm_failure_preserves_authoritative_tool_evidence(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    service.settings = replace(service.settings, require_llm_inference=True)
    narrator = _SpyNarrator(error=True)
    service.narrator = narrator

    response = service.respond(
        message="胸片异常后应该做什么检查？",
        thread_id="thread-required-llm",
        user_id="user",
        owner_scope="tenant:user",
    )

    assert narrator.calls == 1
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.claims
    assert response.retrieved_evidence
    assert response.narration_status == NarrationStatus.FALLBACK_ERROR
    assert response.narrator_generation_invoked is True


def test_service_rejects_unsafe_narration_and_keeps_authoritative_summary(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    service.narrator = _SpyNarrator(unsafe=True)

    response = service.respond(
        message="胸片异常后应该做什么检查？",
        thread_id="thread-unsafe",
        user_id="user",
        owner_scope="tenant:user",
    )

    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.claims
    assert response.summary == response.claims[0].text
    assert response.narration_status == NarrationStatus.REJECTED_BY_SAFETY


def test_service_rejects_any_change_to_protected_response_fields(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    service.narrator = _SpyNarrator(mutate_protected=True)

    response = service.respond(
        message="胸片异常后应该做什么检查？",
        thread_id="thread-invariants",
        user_id="user",
        owner_scope="tenant:user",
    )

    assert response.limitations == [
        "本系统不用于确诊或排除肺结核。",
        "回答仅使用本轮列出的受审核知识块。",
    ]
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.claims
    assert response.narration_status == NarrationStatus.REJECTED_BY_SAFETY


def test_service_rejects_information_not_derived_from_selected_claims(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    service.narrator = _SpyNarrator(inject_information=True)

    response = service.respond(
        message="胸片异常后应该做什么检查？",
        thread_id="thread-information-injection",
        user_id="user",
        owner_scope="tenant:user",
    )

    assert "模型擅自添加的信息。" not in response.diagnostic_information
    assert response.answer_status == GuidelineAnswerStatus.ANSWERED
    assert response.narration_status == NarrationStatus.REJECTED_BY_SAFETY


def test_service_accepts_integrated_summary_with_all_grounded_claims(tmp_path):
    service = TBXAgentService(_settings(tmp_path))

    class _SelectingGroundedNarrator:
        backend_id = "test-grounded"
        model = "test-model"
        model_digest = None
        policy_id = NARRATOR_POLICY_ID

        def narrate(self, response: AgentResponse) -> AgentResponse:
            summary = (
                "建议优先关注两类人群：一类是HIV感染者、肺结核患者密切接触者及"
                "免疫抑制相关人群；另一类是糖尿病患者、65岁及以上老年人和学校等"
                "人员密集机构人群。"
            )
            chunk_ids = list(
                dict.fromkeys(
                    chunk_id for claim in response.claims for chunk_id in claim.chunk_ids
                )
            )
            rendered = validate_grounded_narration(
                response,
                SafeGuidelineNarration(
                    answer_status=response.answer_status,
                    summary=summary,
                    summary_chunk_ids=chunk_ids,
                ),
            )
            return rendered.model_copy(
                update={
                    "narrator_backend": self.backend_id,
                    "narrator_model": self.model,
                    "narrator_policy_id": self.policy_id,
                    "narration_status": NarrationStatus.APPLIED,
                    "narrator_generation_invoked": True,
                }
            )

    service.narrator = _SelectingGroundedNarrator()
    response = service.respond(
        message="请整合主动筛查高风险人群和重点人群。",
        thread_id="thread-grounded-selection",
        user_id="user",
        owner_scope="tenant:user",
    )

    selected_texts = {claim.text for claim in response.claims}
    assert len(selected_texts) == 2
    assert response.narration_status == NarrationStatus.APPLIED
    assert response.summary.startswith("建议优先关注两类人群")
    for values in (
        response.diagnostic_information,
        response.next_step_information,
        response.treatment_education,
    ):
        assert set(values) <= selected_texts


def test_emergency_skips_but_visual_evidence_is_composed_by_narrator(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    narrator = _SpyNarrator()
    service.narrator = narrator

    emergency = service.respond(
        message="我正在大量咯血并且严重呼吸困难",
        thread_id="thread-emergency",
        user_id="user",
        owner_scope="tenant:user",
    )
    assert narrator.calls == 0
    assert emergency.narration_status == NarrationStatus.SKIPPED_EMERGENCY

    payload = _png()
    case, _ = service.assess_cxr(
        payload,
        user_id="user",
        owner_scope="tenant:user",
        consent_to_process=True,
        attested_chest_radiograph=True,
    )
    case, _ = service.classify_cxr_case(
        case_id=case.case_id,
        owner_scope="tenant:user",
        user_id="user",
        payload=payload,
    )
    visual = service.respond(
        message="解释这张胸片结果",
        thread_id="thread-visual",
        user_id="user",
        owner_scope="tenant:user",
        case_id=case.case_id,
    )
    assert narrator.calls == 1
    assert narrator.source_queries == ["解释这张胸片结果"]
    assert visual.response_kind == ResponseKind.VISUAL_SCREENING_RESULT
    assert visual.narration_status == NarrationStatus.APPLIED
