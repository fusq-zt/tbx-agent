from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_ENTRYPOINT = PROJECT_ROOT / "ui" / "streamlit_app.py"


@pytest.mark.integration
def test_streamlit_real_rank03_and_medgemma_end_to_end() -> None:
    if os.getenv("TBX_RUN_REAL_UI_SMOKE") != "1":
        pytest.skip("set TBX_RUN_REAL_UI_SMOKE=1 for the local real-model smoke")
    image_path = Path(os.environ["TBX_REAL_UI_IMAGE"]).resolve()
    health = requests.get("http://127.0.0.1:8000/healthz", timeout=30).json()
    assert health == {
        **health,
        "status": "ok",
        "vision_backend": "rank03",
        "narrator_backend": "llama_cpp",
        "mode": "real_rank03_with_local_llm",
        "required_components_ready": True,
    }

    app = AppTest.from_file(str(UI_ENTRYPOINT), default_timeout=180).run()
    app.file_uploader[0].upload(
        image_path.name,
        image_path.read_bytes(),
        "image/png",
    ).run(timeout=180)
    labels = {item.label for item in app.checkbox}
    assert "我确认该胸片已去除姓名、证件号等可识别信息" not in labels
    assert "我同意将该胸片发送给当前本地服务进行辅助筛查" not in labels
    assert app.chat_input[0].disabled is False
    app.chat_input[0].set_value("请解释这张胸片结果，并说明下一步应该做什么。").run(timeout=180)

    assert not app.exception
    assessment = app.session_state["assessment_result"]
    assert assessment["response"]["reused_existing_assessment"] is False
    evidence = assessment["case"]["vision_evidence"]
    assert not evidence["run_id"].startswith("mock-")
    assert evidence["classifier_model_id"].startswith("tbx11k_rank03_")
    assert "MOCK_ONLY" not in evidence["classifier_model_id"]
    assert evidence["detector_model_id"].endswith(":dfine_l")
    assert any(item in evidence["artifact_refs"] for item in ("device:cuda", "device:cpu"))

    answer = app.session_state["chat_history"][-1]["payload"]
    assert answer["narrator_backend"] == "llama_cpp"
    assert answer["narrator_model"] == "tbx-medgemma-1.5-4b-it-q4-k-m"
    assert answer["narration_status"] == "applied"
    assert answer["narrator_policy_id"] == "tbx-grounded-evidence-synthesis-v2"
    assert answer["narrator_generation_invoked"] is True
    assert answer["narrator_prompt_tokens"] > 0
    assert answer["narrator_completion_tokens"] > 0
