# TBX-Agent trajectory suite v2

This fixed suite evaluates the current bounded controller and `AgentRunTrace`
v2 contract by executing real `TBXAgentService.respond_with_controller` turns.
It covers on-demand classification, on-demand localization, compound
observe-and-replan behavior, cached no-tool rationale, a missing-prior evidence
gap, one retryable capacity recovery, and tool/prompt-injection rejection.

The fixtures describe synthetic software behavior only. They are not clinical
examples, are not used for model or threshold selection, and never use the
locked TBX11K test split. The runtime evaluator must use actual execution plans,
state transitions, and tool receipts; it must not manufacture legacy planner,
checkpoint, or Reflection events.

Regeneration policy: preserve case IDs, seed, semantic intent, and the checked-in
`cases_sha256`; any semantic change requires a new suite version and manifest
hash. Runtime observations and reports remain generated artifacts and are not
checked in here.
