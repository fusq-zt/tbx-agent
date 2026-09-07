# TBX-Agent trajectory suite v3

This fixed suite evaluates the current Plan+ReAct controller, `AgentRunTrace`
v2 trace contract, and `tbx-tool-contract-v7` receipt contract by executing real
`TBXAgentService.respond_with_controller` turns. Expected trajectories use the
four public model tools (`classify_cxr`, `localize_cxr`,
`analyze_lung_anatomy`, and `search_tb_knowledge`); internal adapter names remain
auditable in receipts. It covers on-demand classification and localization,
compound observe-then-replan behavior with at most one action per ReAct step,
cached no-tool rationale, a missing-prior zero-tool evidence gap, one retryable
capacity recovery, and tool/prompt-injection rejection. v2 remains unchanged
for historical replay.

Suite `3.2.0` records the missing-prior response as `safe_abstention`, matching
the explicit evidence-gap semantics of the production graph. The case IDs and
synthetic inputs are unchanged; the new manifest hash records this contract
revision rather than silently rewriting the prior split identity.

The fixtures describe synthetic software behavior only. They are not clinical
examples, are not used for model or threshold selection, and never use the
locked TBX11K test split. The runtime evaluator must use actual execution plans,
state transitions, and tool receipts; it must not manufacture legacy planner,
checkpoint, or Reflection events.

Regeneration policy: preserve case IDs, seed, semantic intent, and the checked-in
`cases_sha256`; any semantic change requires a manifest hash update. Runtime
observations and reports remain generated artifacts and are not checked in here.
