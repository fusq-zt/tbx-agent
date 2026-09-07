# TBX-Agent system suite v1.6

This synthetic, non-clinical suite makes the compiled LangGraph Plan+ReAct
runtime the prompt-driven evaluation path. New observations may contain only
the four public tools: `classify_cxr`, `localize_cxr`,
`analyze_lung_anatomy`, and `search_tb_knowledge`.

Capability answers and deterministic emergency guards are explicit no-tool
routes. The exact-tie fixture now follows the deployed, fixed-order native
`argmax` contract instead of adding an undeployed review policy. The historical
v1.1-v1.5 suites remain unchanged and readable through compatibility aliases.

The suite contains no patient data, clinical images, model weights, TBX11K
split, locked test set, or hidden test set. Passing it is software regression
evidence, not clinical validation.
