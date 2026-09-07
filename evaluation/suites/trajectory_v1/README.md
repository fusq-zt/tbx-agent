# Historical trajectory fixture

This directory preserves the pre-controller synthetic trajectory fixture for provenance. Its cases
target the retired multi-step plan/conditional-recovery schema and do **not** validate the current
`TaskSpec -> CaseState -> one action -> Observation -> replanning -> STOP/REFER_TO_HUMAN` runtime.

Do not publish a result from this fixture as current Agent evidence. A replacement suite must first:

- use `AgentRunTrace` v2, including `ControllerDecision`, `StateTransition`, `TerminalRecord`, and
  `AgentBudget`;
- assert one action per decision and a fresh state hash after each tool Observation;
- cover success, completed-with-no-detection, failure, unavailable capability, evidence conflict,
  missing prior study, and budget exhaustion branches;
- verify the default 5-step / 4-tool budget and absence of autonomous retry;
- retain only structured hashes and reason codes, never model chain-of-thought or patient data.

Until that migration is complete, use the controller unit and orchestration tests named in
[`docs/agent_trajectory_evaluation.md`](../../../docs/agent_trajectory_evaluation.md). The retained
JSONL/manifest files are historical fixtures, not a release gate.
