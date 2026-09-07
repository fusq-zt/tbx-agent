# TBX-Agent system suite v1.5

This synthetic, non-clinical suite supersedes v1.4 for the active fusion
contract. It preserves the v1.4 Agent and guideline-tool cases, while replacing
the obsolete quality-warning routing case with a new immutable case ID:
`sysv15.vision.quality-warning-advisory-preserves-healthy.001`.

Planner-facing expectations use the public `search_tb_knowledge` name. Runtime
receipts separately retain the internal adapter name for auditability; the two
names are not treated as two tool calls.

Under the active `native_three_class_argmax + advisory_localization_only` policy,
a non-technical image-quality warning remains internal evidence and does not
override the classifier route or create a single-case review. Exact argmax ties
and technical failures retain their existing fail-closed behavior. Published
v1.1-v1.4 suites remain unchanged for historical replay.

The suite does not use TBX11K, a locked split, a hidden test set, patient data,
or model weights. It is a software and Agent contract regression suite only.
