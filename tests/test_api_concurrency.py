from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event

import httpx
import pytest

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.service import ConsentRequiredError, TBXAgentService


@pytest.mark.parametrize(
    ("route", "method", "error", "extra_form"),
    [
        ("/v1/assessments/cxr", "assess_cxr", ConsentRequiredError, {}),
        (
            "/v1/batches/batch-1/assessments/cxr",
            "assess_cxr_for_batch",
            ConsentRequiredError,
            {"batch_item_id": "item-1"},
        ),
        (
            "/v1/cases/case-1/anatomy-runs?user_id=user&owner_scope=tenant:user",
            "request_anatomy_run",
            ValueError,
            {},
        ),
    ],
)
def test_liveness_remains_responsive_during_upload_processing(
    tmp_path, monkeypatch, route, method, error, extra_form
):
    settings = replace(
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
    service = TBXAgentService(settings)
    started, release = Event(), Event()

    def process_upload(*args, **kwargs):
        started.set()
        if not release.wait(timeout=3):
            raise RuntimeError("upload processing blocked the ASGI event loop")
        raise error("synthetic input rejected after processing")

    monkeypatch.setattr(service, method, process_upload)

    async def exercise():
        transport = httpx.ASGITransport(app=create_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            upload = asyncio.create_task(
                client.post(
                    route,
                    data={
                        "user_id": "user",
                        "owner_scope": "tenant:user",
                        "consent_to_process": "true",
                        "attested_chest_radiograph": "true",
                        **extra_form,
                    },
                    files={"file": ("synthetic.png", b"synthetic", "image/png")},
                )
            )
            try:
                assert await asyncio.to_thread(started.wait, 2)
                health = await asyncio.wait_for(client.get("/livez"), timeout=2)
                assert health.status_code == 200
                assert not upload.done(), "health must complete while upload work is pending"
            finally:
                release.set()
                response = await upload
            assert response.status_code == 422
            assert response.json()["detail"] == "synthetic input rejected after processing"

    try:
        asyncio.run(exercise())
    finally:
        release.set()
        service.tool_registry.close()
        service.store.close()
