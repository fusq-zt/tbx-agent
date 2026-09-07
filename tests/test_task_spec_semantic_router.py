"""Compatibility tests for deterministic TaskSpec interpretation.

The live action planner is Plan + ReAct and is covered by
``test_llm_tool_calling.py`` and ``test_plan_react.py``.  TaskSpec remains the
availability fallback and context projection; it is no longer the model-facing
tool catalog.
"""

from __future__ import annotations

from tbx_agent.task_spec import (
    TaskGoal,
    TaskSpecSource,
    interpret_task_spec,
    parse_task_spec,
)


class _UnavailableGenerator:
    backend_id = "unavailable-test"
    model = "unavailable-test-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete_structured(self, **_kwargs):
        self.calls += 1
        raise RuntimeError("planner unavailable")


def test_missing_model_uses_deterministic_task_spec_fallback() -> None:
    query = "痰片没查到菌是不是就能排除结核？"

    interpreted = interpret_task_spec(query, generator=None)

    assert interpreted.source == TaskSpecSource.RULE_FALLBACK
    assert interpreted.schema_validated is False
    assert interpreted.task_spec == parse_task_spec(query)


def test_provider_failure_cannot_remove_availability_fallback() -> None:
    query = "孕妇怀疑肺结核时应该做什么检查？"
    generator = _UnavailableGenerator()

    interpreted = interpret_task_spec(query, generator=generator)

    assert generator.calls == 1
    assert interpreted.source == TaskSpecSource.RULE_FALLBACK
    assert interpreted.schema_validated is False
    assert interpreted.authorization_validated is False
    assert interpreted.task_spec == parse_task_spec(query)


def test_non_tb_fallback_does_not_invent_a_medical_tool_goal() -> None:
    for query in ("1+1等于多少？", "人每天应该摄入多少糖分？"):
        interpreted = interpret_task_spec(query, generator=None)

        assert interpreted.task_spec.task_goals == [TaskGoal.GENERAL_CHAT]
        assert interpreted.task_spec.required_evidence == []
        assert interpreted.task_spec.no_tool_only is True
