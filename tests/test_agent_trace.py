from tbx_agent.agent_state import AgentAction
from tbx_agent.agent_trace import AgentRunTrace, TerminalRecord
from tbx_agent.controller import CONTROLLER_POLICY_ID, AgentBudget
from tbx_agent.task_spec import parse_task_spec


def test_public_trace_has_terminal_action_and_never_claims_hidden_reasoning() -> None:
    trace = AgentRunTrace(
        controller_policy_id=CONTROLLER_POLICY_ID,
        task_spec=parse_task_spec("你好"),
        terminal=TerminalRecord(
            action=AgentAction.STOP,
            reason_code="no_tool_task",
        ),
        budget=AgentBudget(steps_used=1),
    )

    payload = trace.model_dump(mode="json")
    assert payload["trace_version"] == "tbx-agent-trace-v2"
    assert payload["terminal"]["action"] == "stop"
    assert payload["hidden_reasoning_persisted"] is False
