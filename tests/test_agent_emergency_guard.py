from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from tbx_agent.config import Settings
from tbx_agent.orchestration import TBXAgentGraph
from tbx_agent.schemas import NarrationStatus, ResponseKind, Urgency
from tbx_agent.service import TBXAgentService


class _EmergencyGeneratorSpy:
    backend_id = "emergency-guard-spy"
    model = "synthetic"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete_structured(self, *, schema_name, **kwargs):
        self.calls.append(schema_name)
        return json.dumps(
            {
                "goal": "解释当前胸片分类",
                "steps": [
                    {"objective": "胸片分类", "evidence_need": "classification"}
                ],
            }
        ), {}

    def narrate(self, response, **kwargs):
        self.calls.append("narrate")
        return response


@pytest.mark.parametrize("cached_classification", [False, True])
def test_emergency_preempts_models_and_cached_visual_evidence(tmp_path, cached_classification):
    project_root = Path(__file__).resolve().parents[1]
    settings = replace(
        Settings.from_env(),
        project_root=project_root,
        config_dir=project_root / "configs",
        knowledge_dir=project_root / "knowledge",
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
    )
    service = TBXAgentService(settings)
    case_id = None
    if cached_classification:
        image = io.BytesIO()
        Image.linear_gradient("L").resize((512, 512)).convert("RGB").save(image, format="PNG")
        payload = image.getvalue()
        case, _ = service.assess_cxr(
            payload,
            user_id="emergency-user",
            owner_scope="tenant:emergency",
            consent_to_process=True,
            attested_chest_radiograph=True,
        )
        case, _ = service.classify_cxr_case(
            case_id=case.case_id,
            user_id="emergency-user",
            owner_scope="tenant:emergency",
            payload=payload,
        )
        case_id = case.case_id

    generator = _EmergencyGeneratorSpy()
    result = TBXAgentGraph(service).invoke_with_receipt(
        {
            "message": "我正在大量咯血并且严重呼吸困难，分析这张胸片的分类。",
            "thread_id": "emergency-thread",
            "user_id": "emergency-user",
            "owner_scope": "tenant:emergency",
            "case_id": case_id,
        },
        narrator_override=generator,
    )

    assert result.response.response_kind == ResponseKind.EMERGENCY_ESCALATION
    assert result.response.urgency == Urgency.EMERGENCY
    assert "120" in result.response.summary
    assert result.response.narration_status == NarrationStatus.SKIPPED_EMERGENCY
    assert result.response.narrator_generation_invoked is False
    assert generator.calls == []
    assert result.tool_results == []
    assert result.trace.terminal.reason_code == "emergency_guard"
    assert result.execution_plan["cached_evidence"] == []
    assert result.execution_plan["graph_node_trace"] == ["load_context", "finalize"]
