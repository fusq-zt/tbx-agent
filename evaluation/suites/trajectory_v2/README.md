# trajectory_v2

Synthetic fixtures for tool trajectory schema and evidence binding. Tests load `cases.jsonl` and its matching
`manifest.json`; keep their IDs, expectations and content hashes together. These assets contain
no patient images, model weights or execution results.

The versioned directories support compatibility and regression tests. They are not independent
clinical datasets or a measure of real-model accuracy. Current test scope and results are in
[the evaluation overview](../../../docs/evaluation.md).
