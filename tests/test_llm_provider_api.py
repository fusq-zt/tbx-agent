from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

import tbx_agent.api.main as api_main
from tbx_agent.api.main import create_app
from tbx_agent.api.models import AgentQuery
from tbx_agent.config import Settings
from tbx_agent.narrator import NARRATOR_POLICY_ID
from tbx_agent.schemas import NarrationStatus
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_local_provider_defaults_to_medgemma_and_accepts_legacy_wire_alias() -> None:
    fields = {
        "thread_id": "thread-provider",
        "message": "你好",
    }
    assert AgentQuery(**fields).llm_provider == "local_medgemma"
    assert AgentQuery(**fields, llm_provider="local_qwen").llm_provider == "local_qwen"


class _OpenAICompatibleTestDouble:
    backend_id = "openai_compatible"
    runtime_contract = "openai-compatible-grounded-generation-v1"
    model_digest = None
    synthetic = False
    instances = []

    def __init__(self, *, base_url, model, api_key, **_kwargs):
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self._api_key = api_key
        self.closed = False
        type(self).instances.append(self)

    def narrate(self, response):
        return response.model_copy(
            update={
                "summary": (
                    f"综合来看：{response.summary}"
                    if response.answer_status is not None and len(response.claims) > 1
                    else response.summary
                ),
                "narrator_backend": self.backend_id,
                "narrator_model": self.model,
                "narrator_model_digest": None,
                "narrator_policy_id": NARRATOR_POLICY_ID,
                "narration_status": NarrationStatus.APPLIED,
                "narrator_generation_invoked": True,
                "narrator_prompt_tokens": 7,
                "narrator_completion_tokens": 3,
            }
        )

    def complete_structured(self, **kwargs):
        schema_name = kwargs["schema_name"]
        if schema_name == "tbx_plan_react_plan":
            query = kwargs["messages"][-1]["content"]
            needs = ["tb_knowledge"] if "胸片异常" in query else ["none"]
            return (
                json.dumps(
                    {
                        "goal": "回答当前问题",
                        "steps": [
                            {
                                "objective": f"处理 {need} 信息",
                                "evidence_need": need,
                            }
                            for need in needs
                        ],
                    },
                    ensure_ascii=False,
                ),
                {"prompt_tokens": 11, "completion_tokens": 5},
            )
        if schema_name == "tbx_agent_tool_selection":
            messages = kwargs["messages"]
            query = messages[-1]["content"]
            internal_message = next(
                message
                for message in messages
                if message["role"] == "system"
                and "TBX_INTERNAL_CONTEXT_JSON=" in message["content"]
            )
            marker = "TBX_INTERNAL_CONTEXT_JSON="
            content = internal_message["content"]
            internal = json.loads(content[content.index(marker) + len(marker) :])
            if "胸片异常" in query and not internal["observations"]:
                selection = {
                    "tool": "search_tb_knowledge",
                    "direct_answer": None,
                }
            else:
                selection = {
                    "tool": None,
                    "direct_answer": (
                        "已根据检索结果回答。"
                        if internal["observations"]
                        else "远端模型回答：2"
                    ),
                }
            return (
                json.dumps(selection, ensure_ascii=False),
                {"prompt_tokens": 9, "completion_tokens": 3},
            )
        raise AssertionError(f"unexpected schema: {schema_name}")

    def close(self):
        self.closed = True


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    _OpenAICompatibleTestDouble.instances.clear()
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
        narrator_backend="none",
        openai_enabled=False,
        require_real_inference=False,
        require_llm_inference=False,
    )
    monkeypatch.setattr(
        api_main,
        "OpenAICompatibleNarrator",
        _OpenAICompatibleTestDouble,
    )
    return TestClient(create_app(TBXAgentService(settings)))


def test_remote_provider_handles_general_question_without_case_or_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    identity = {
        "thread_id": "thread-general-provider",
        "user_id": "user-general-provider",
        "owner_scope": "tenant:general-provider",
    }
    connection = client.post(
        "/v1/llm/connections",
        json={
            **identity,
            "base_url": "https://llm.example.test/v1",
            "model": "example-instruct",
            "api_key": "test-key-general-provider",
        },
    ).json()

    response = client.post(
        "/v1/agent/respond",
        json={
            **identity,
            "message": "1+1 = ？",
            "llm_provider": "openai_compatible",
            "llm_connection_id": connection["connection_id"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["response_kind"] == "general_answer"
    assert payload["summary"] == "远端模型回答：2"
    assert payload["execution_receipts"] == []
    assert payload["execution_plan"]["tool_names"] == []
    assert payload["narrator_backend"] == "openai_compatible"
    assert payload["narrator_model"] == "example-instruct"
    assert payload["narrator_policy_id"] == "tbx-plan-react-direct-v1"
    assert payload["narrator_generation_invoked"] is True
    assert payload["narrator_prompt_tokens"] == 9
    assert _OpenAICompatibleTestDouble.instances[-1].closed is True

def test_remote_provider_changes_real_turn_without_persisting_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    secret = "test-key-never-persist-this-value"
    identity = {
        "thread_id": "thread-provider-switch",
        "user_id": "user-provider-switch",
        "owner_scope": "tenant:provider-switch",
    }
    connection = client.post(
        "/v1/llm/connections",
        json={
            **identity,
            "base_url": "https://llm.example.test/v1",
            "model": "example-instruct",
            "api_key": secret,
        },
    )
    assert connection.status_code == 200
    connection_payload = connection.json()
    assert connection_payload["provider"] == "openai_compatible"
    assert connection_payload["credential_persistence"] == "process_memory_only"
    assert secret not in connection.text

    response = client.post(
        "/v1/agent/respond",
        json={
            **identity,
            "message": "胸片异常后还需要做什么检查？",
            "llm_provider": "openai_compatible",
            "llm_connection_id": connection_payload["connection_id"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["narrator_backend"] == "openai_compatible"
    assert payload["narrator_model"] == "example-instruct"
    assert payload["narrator_generation_invoked"] is True
    assert payload["narrator_prompt_tokens"] == 7
    assert secret not in response.text

    persisted = b"".join(
        path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file() and path.stat().st_size < 10_000_000
    )
    assert secret.encode() not in persisted


def test_remote_connection_is_thread_bound_and_revocable(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    identity = {
        "thread_id": "thread-a",
        "user_id": "user-a",
        "owner_scope": "tenant:a",
    }
    connection = client.post(
        "/v1/llm/connections",
        json={
            **identity,
            "base_url": "https://llm.example.test/v1",
            "model": "example-instruct",
            "api_key": "test-key-thread-bound",
        },
    ).json()
    wrong_thread = client.post(
        "/v1/agent/respond",
        json={
            **identity,
            "thread_id": "thread-b",
            "message": "需要什么检查？",
            "llm_provider": "openai_compatible",
            "llm_connection_id": connection["connection_id"],
        },
    )
    assert wrong_thread.status_code == 404

    revoked = client.delete(
        f"/v1/llm/connections/{connection['connection_id']}",
        params=identity,
    )
    assert revoked.status_code == 200
    assert revoked.json() == {"revoked": True}
    unavailable = client.post(
        "/v1/agent/respond",
        json={
            **identity,
            "message": "需要什么检查？",
            "llm_provider": "openai_compatible",
            "llm_connection_id": connection["connection_id"],
        },
    )
    assert unavailable.status_code == 409


def test_provider_contract_rejects_mismatched_connection_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    client = _client(tmp_path, monkeypatch)
    base = {
        "thread_id": "thread-contract",
        "user_id": "user-contract",
        "owner_scope": "tenant:contract",
        "message": "解释检查",
    }
    missing = client.post(
        "/v1/agent/respond",
        json={**base, "llm_provider": "openai_compatible"},
    )
    assert missing.status_code == 422
    local_with_handle = client.post(
        "/v1/agent/respond",
        json={**base, "llm_provider": "local_qwen", "llm_connection_id": "llmc_bad"},
    )
    assert local_with_handle.status_code == 422
