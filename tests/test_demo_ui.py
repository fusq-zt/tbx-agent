"""Exercise the shipped UI against the actual in-process, model-free API."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

from tbx_agent.api.main import create_app
from tbx_agent.config import Settings
from tbx_agent.service import TBXAgentService

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_api(tmp_path, monkeypatch):
    settings = replace(
        Settings.from_env(),
        project_root=PROJECT_ROOT,
        config_dir=PROJECT_ROOT / "configs",
        knowledge_dir=PROJECT_ROOT / "knowledge",
        retrieval_config_path=PROJECT_ROOT / "configs/retrieval.yaml",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "cases",
        deployment_profile="development",
        vision_backend="mock",
        narrator_backend="none",
        anatomy_backend="none",
        contour_refinement_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        anatomy_required=False,
        openai_enabled=False,
    )
    service = TBXAgentService(settings)
    calls = []
    try:
        with TestClient(create_app(service)) as client:
            def request(method, url, **kwargs):
                kwargs.pop("timeout", None)
                path = urlsplit(url).path
                calls.append((method.upper(), path))
                actual = client.request(method, path, **kwargs)
                response = requests.Response()
                response.status_code = actual.status_code
                response.headers.update(actual.headers)
                response._content = actual.content
                response.encoding = "utf-8"
                return response

            monkeypatch.setattr(requests, "request", request)
            monkeypatch.setattr(requests, "get", lambda url, **kw: request("GET", url, **kw))
            yield service, calls
    finally:
        service.close()


def _app():
    return AppTest.from_file(str(PROJECT_ROOT / "ui/streamlit_app.py"), default_timeout=20).run()


def _button(app, label):
    return next(item for item in app.button if item.label == label)


def _text(app):
    return "\n".join(str(item.value) for kind in (
        app.markdown, app.info, app.warning, app.error, app.caption,
    ) for item in kind)


def test_demo_ui_runs_capabilities_guidance_and_screening_without_models(demo_api):
    service, calls = demo_api
    app = _app()
    assert not app.exception
    assert app.chat_input[0].disabled is False
    assert "DEMO / MOCK" in _text(app)
    assert "本地 MedGemma 未就绪" not in _text(app)

    app.chat_input[0].set_value("你能做什么？").run()
    assert not app.exception
    assert "三分类" in _text(app)
    app.chat_input[0].set_value("结核病会传染吗？").run()
    assert not app.exception
    assert "气溶胶" in _text(app)
    assert app.session_state["chat_history"][-1]["payload"]["execution_receipts"]

    _button(app, "开始主动筛查").click().run()
    assert not app.exception
    assert app.session_state["screening_session"]["status"] == "collecting"
    app.multiselect[0].set_value(["以上均无"]).run()
    _button(app, "下一题").click().run()
    assert not app.exception
    assert app.session_state["screening_session"]["answers"]
    _button(app, "退出主动筛查").click().run()
    assert app.chat_input[0].disabled is False
    assert service.narrator is None
    assert service.vision.call_count == 0
    assert ("POST", "/v1/agent/respond") in calls


def test_demo_ui_reuses_simulated_case_for_followup_and_questionnaire(demo_api):
    service, calls = demo_api
    app = _app()
    output = io.BytesIO()
    Image.new("RGB", (512, 512), (48, 68, 88)).save(output, format="PNG")
    app.file_uploader[0].upload("synthetic-fixture.png", output.getvalue(), "image/png").run()
    app.chat_input[0].set_value("请分析这张胸片").run()
    assert not app.exception
    case_id = app.session_state["case_id"]
    assert service.vision.call_count == 1
    app.chat_input[0].set_value("请汇总已完成的分析").run()
    assert not app.exception
    assert app.session_state["case_id"] == case_id
    assert service.vision.call_count == 1
    assert calls.count(("POST", "/v1/assessments/cxr")) == 1
    _button(app, "开始主动筛查").click().run()
    assert not app.exception
    assert app.session_state["screening_session"]["case_id"] == case_id
