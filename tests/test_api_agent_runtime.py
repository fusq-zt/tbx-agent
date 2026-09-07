from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import tbx_agent.api.main as api_main
from tbx_agent.config import Settings
from tbx_agent.schemas import AgentResponse, ResponseKind
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _Dumpable(dict):
    def model_dump(self, *, mode: str):
        assert mode == "json"
        return dict(self)

    @property
    def fallback_used(self) -> bool:
        return bool(self.get("fallback_used", False))


def _response(*, thread_id: str, case_id: str | None, summary: str) -> AgentResponse:
    return AgentResponse(
        request_id="request-1",
        trace_id="trace-1",
        thread_id=thread_id,
        case_id=case_id,
        response_kind=ResponseKind.SAFE_ABSTENTION,
        summary=summary,
    )


class _StubControllerAdapter:
    def __init__(self, _service):
        pass

    def invoke_with_receipt(self, request, *, narrator_override=None):
        assert narrator_override is None
        if request["message"] == "你好":
            return SimpleNamespace(
                response=_response(
                    thread_id=request["thread_id"],
                    case_id=request.get("case_id"),
                    summary="你好。",
                ),
                tool_results=[],
                execution_plan={
                    "plan_id": "run-no-tool",
                    "policy_id": "tbx-plan-react-v1",
                    "source": "plan_react",
                    "framework": "langgraph",
                    "graph_node_trace": [
                        "load_context",
                        "plan",
                        "decide",
                        "finalize",
                    ],
                    "tool_names": [],
                    "steps": [],
                    "hidden_reasoning_persisted": False,
                },
                trace={
                    "trace_version": "tbx-agent-trace-v2",
                    "decisions": [],
                    "state_transitions": [],
                    "terminal": {"action": "stop", "reason_code": "react_answered"},
                },
                receipt=None,
                reflection=None,
            )

        receipt = _Dumpable(
            tool_name="classify_current_cxr",
            model_tool_name="classify_cxr",
            status="succeeded",
            fallback_used=False,
        )
        return SimpleNamespace(
            response=_response(
                thread_id=request["thread_id"],
                case_id=request.get("case_id"),
                summary="分类完成。",
            ),
            tool_results=[SimpleNamespace(receipt=receipt)],
            execution_plan={
                "plan_id": "run-classify",
                "policy_id": "tbx-plan-react-v1",
                "source": "plan_react",
                "framework": "langgraph",
                "graph_node_trace": [
                    "load_context",
                    "plan",
                    "decide",
                    "execute_tool",
                    "observe",
                    "decide",
                    "finalize",
                ],
                "tool_names": ["classify_cxr"],
                "steps": [{
                    "id": "s1",
                    "phase": "tool",
                    "label": "胸片分类",
                    "tool_name": "classify_cxr",
                    "internal_tool_name": "classify_current_cxr",
                    "status": "completed",
                }],
                "hidden_reasoning_persisted": False,
            },
            trace={
                "trace_version": "tbx-agent-trace-v2",
                "decisions": [
                    {
                        "step_index": 0,
                        "action": "classify_cxr",
                        "reason_code": "classification_required",
                    }
                ],
                "state_transitions": [
                    {
                        "step_index": 0,
                        "tool_name": "classify_cxr",
                        "tool_status": "succeeded",
                    }
                ],
                "terminal": {"action": "stop", "reason_code": "task_already_complete"},
            },
            receipt=receipt,
            reflection=None,
        )

    def close(self) -> None:
        return None


def _client(tmp_path: Path, monkeypatch) -> TestClient:
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
    )
    monkeypatch.setattr(api_main, "TBXAgentGraph", _StubControllerAdapter)
    return TestClient(api_main.create_app(TBXAgentService(settings)))


def test_agent_api_serializes_tool_free_stop_without_fake_receipt(tmp_path, monkeypatch):
    response = _client(tmp_path, monkeypatch).post(
        "/v1/agent/respond",
        json={
            "thread_id": "thread-no-tool",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "message": "你好",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_receipt"] is None
    assert payload["execution_receipts"] == []
    assert payload["execution_plan"]["tool_names"] == []
    assert payload["execution_plan"]["steps"] == []
    assert payload["execution_plan"]["framework"] == "langgraph"
    assert payload["execution_plan"]["graph_node_trace"] == [
        "load_context",
        "plan",
        "decide",
        "finalize",
    ]
    assert payload["reflection"] is None
    assert payload["agent_trace"]["trace_version"] == "tbx-agent-trace-v2"
    assert payload["agent_trace"]["terminal"]["reason_code"] == "react_answered"
    assert "plan_source" not in payload["agent_trace"]


def test_agent_api_exposes_on_demand_classification_as_first_executed_step(
    tmp_path,
    monkeypatch,
):
    response = _client(tmp_path, monkeypatch).post(
        "/v1/agent/respond",
        json={
            "thread_id": "thread-case",
            "user_id": "user",
            "owner_scope": "tenant:user",
            "case_id": "case-1",
            "message": "这张胸片有没有结核病？",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["execution_receipt"]["tool_name"] == "classify_current_cxr"
    assert payload["execution_receipt"]["model_tool_name"] == "classify_cxr"
    assert payload["execution_plan"]["tool_names"] == ["classify_cxr"]
    assert payload["execution_plan"]["steps"][0] == {
        "id": "s1",
        "phase": "tool",
        "label": "胸片分类",
        "tool_name": "classify_cxr",
        "internal_tool_name": "classify_current_cxr",
        "status": "completed",
    }
    assert payload["agent_trace"]["trace_version"] == "tbx-agent-trace-v2"
    assert payload["agent_trace"]["decisions"][0]["action"] == "classify_cxr"
