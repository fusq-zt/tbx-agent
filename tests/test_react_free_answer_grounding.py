"""The production semantic general-chat path cannot publish invented findings."""
from __future__ import annotations

import pytest
from test_plan_react_runtime import _upload
from test_react_first_runtime import Decisions, run
from test_react_first_runtime import services as services

from tbx_agent.schemas import NarrationStatus


@pytest.mark.parametrize("has_image", [False, True])
@pytest.mark.parametrize("claim", [
    "左上肺存在一个约2厘米的结节。",
    "双肺未见明显异常。",
    "The left upper lung contains a 2 cm nodule.",
])
def test_semantic_chat_without_evidence_cannot_publish_image_claim(
    services, has_image, claim,
):
    service = services()
    case = _upload(service) if has_image else None
    generator = Decisions({
        "tasks": [{"task": "general_chat", "when": "always"}],
        "answer": claim,
    })
    result = run(service, generator, "请描述影像发现", case)
    assert result.response.narration_status == NarrationStatus.REJECTED_BY_SAFETY
    assert claim not in result.response.summary
    assert "证据校验" in result.response.summary
    assert "模型连接" not in result.response.summary
    assert result.execution_plan["tool_names"] == []
    assert service.vision.call_count == 0
    assert service.vision.localization_call_count == 0
