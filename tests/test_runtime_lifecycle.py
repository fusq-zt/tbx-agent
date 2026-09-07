from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tbx_agent.api import main as api
from tbx_agent.config import Settings
from tbx_agent.schemas import AgentResponse, ResponseKind
from tbx_agent.service import TBXAgentService
from tbx_agent.storage import SQLiteStore
from tbx_agent.tools.contracts import ToolCallStatus, ToolInvocation
from tbx_agent.tools.registry import ToolDefinition


def _settings(tmp_path):
    return replace(
        Settings.from_env(),
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        anatomy_backend="none",
        anatomy_required=False,
    )


def _response(invocation):
    return AgentResponse(
        request_id=invocation.request_id,
        trace_id=invocation.trace_id,
        thread_id=invocation.thread_id,
        response_kind=ResponseKind.SAFE_ABSTENTION,
        summary="synthetic lifecycle response",
    )


def test_shutdown_drains_timed_out_tool_before_closing_database(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    started, release, persisted = Event(), Event(), Event()

    def handler(invocation):
        started.set()
        assert release.wait(timeout=3)
        service.store.get_or_create_thread("worker", "user", "tenant")
        persisted.set()
        return _response(invocation)

    service.tool_registry.register(
        ToolDefinition(
            name="lifecycle_probe",
            audit_action="lifecycle_probe",
            handler=handler,
            timeout_seconds=0.02,
        )
    )
    invocation = ToolInvocation(
        tool_name="lifecycle_probe",
        message="synthetic",
        request_id="request",
        trace_id="trace",
        thread_id="thread",
        owner_scope="tenant",
        user_id="user",
        routing_policy_id="test",
        max_steps=service.tool_registry.max_steps,
    )
    try:
        result = service.tool_registry.execute(
            invocation, fallback_factory=lambda **kwargs: _response(kwargs["invocation"])
        )
        assert started.is_set()
        assert result.receipt.status == ToolCallStatus.TIMED_OUT
        with ThreadPoolExecutor(max_workers=1) as worker:
            closing = worker.submit(service.close)
            release.set()
            closing.result(timeout=3)
        assert persisted.is_set()
        with sqlite3.connect(service.settings.db_path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM threads").fetchone()[0] == 1
        with pytest.raises(sqlite3.ProgrammingError):
            service.store.integrity_check()
        service.close()
    finally:
        release.set()
        service.close()


def test_service_does_not_close_a_caller_owned_store(tmp_path):
    store = SQLiteStore(tmp_path / "injected.sqlite3")
    try:
        service = TBXAgentService(_settings(tmp_path), store=store)
        service.close()
        service.close()
        assert store.get_or_create_thread("thread", "user", "tenant").thread_id == "thread"
    finally:
        store.close()


def test_app_closes_owned_service_clears_connections_and_can_restart(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(Settings, "from_env", staticmethod(lambda: settings))
    created = []

    def create_service(config):
        instance = TBXAgentService(config)
        created.append(instance)
        return instance

    monkeypatch.setattr(api, "TBXAgentService", create_service)
    app = api.create_app()
    for iteration in range(2):
        with TestClient(app) as client:
            result = client.get("/v1/cases/missing?owner_scope=tenant&user_id=user")
            assert result.status_code == 404
            app.state.llm_connections.create(
                owner_scope="tenant", user_id="user", thread_id="thread",
                base_url="http://127.0.0.1:9999/v1", model="synthetic", api_key="test-only-key",
            )
            assert len(app.state.llm_connections) == 1
        assert len(created) == iteration + 1
        assert len(app.state.llm_connections) == 0
        with pytest.raises(sqlite3.ProgrammingError):
            created[-1].store.integrity_check()


def test_failed_startup_also_closes_owned_service(tmp_path, monkeypatch):
    settings = replace(_settings(tmp_path), anatomy_backend="xrv_pspnet")
    monkeypatch.setattr(Settings, "from_env", staticmethod(lambda: settings))
    service = TBXAgentService(replace(settings, anatomy_backend="none"))
    monkeypatch.setattr(api, "TBXAgentService", lambda config: service)
    monkeypatch.setattr(
        api, "build_capability_snapshot",
        lambda service: SimpleNamespace(status="failed", runtime_verified=False),
    )
    with pytest.raises(RuntimeError, match="startup warmup"), TestClient(api.create_app()):
        pytest.fail("startup must fail before accepting requests")
    with pytest.raises(sqlite3.ProgrammingError):
        service.store.integrity_check()


def test_app_leaves_injected_service_to_its_caller(tmp_path):
    service = TBXAgentService(_settings(tmp_path))
    try:
        with TestClient(api.create_app(service)) as client:
            assert client.get("/livez").status_code == 200
        assert service.store.get_or_create_thread("thread", "user", "tenant").thread_id == "thread"
    finally:
        service.close()
