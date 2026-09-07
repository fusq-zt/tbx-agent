# Synthetic imaging software conformance v1

This immutable suite covers new software contracts that are intentionally absent
from `system_v1_2`: restricted DICOM conversion/rejection, PSPNet-shaped paired-lung
mask evidence and QC, deterministic detector-to-lung-field geometry, MedSAM-shaped
optional contour evidence, transparent API layers, and fail-closed UI consumption.

Every input is generated in memory and is explicitly non-patient. The runner does
not access TBX11K, a locked split, a hidden test set, the network, or model weights.
It does not run rank03, D-FINE, PSPNet, MedSAM, or Qwen. Passing this suite means
only that the checked software contracts behaved as declared; it says nothing
about classifier/detector accuracy, segmentation quality, diagnosis, treatment,
clinical safety, generalization, or medical-device readiness.

`system_v1_2` remains byte-for-byte independent and keeps its original purpose.
This suite is not an additive rewrite of that Agent evaluation and its results must
not be compared as if both suites shared a case set.

Run from a source checkout with the `dev`, `dicom`, and `ui` extras installed:

```bash
tbx-agent-software-conformance \
  --config evaluation/software_conformance_config_v1.json \
  --output-dir /external/runtime/evaluation_runs/software-conformance-example
```

The output directory must not exist. `report.json` records the hypothesis, complete
configuration, seed, suite hash (there is no train/validation/test split), source
revision and source-tree hash, metrics, runtime, and an explicit null/reason for
peak VRAM. A hash-chained ledger beside the output directory retains passing,
regressed, and failed attempts without overwriting earlier receipts.

Full asynchronous worker persistence and HTTP endpoint headers remain covered by
`tests/test_anatomy_integration.py`; this small formal suite exercises the core
production DTO, rendering, redaction, and UI-client boundaries without starting a
server. Real pinned-model smokes and target-population validation are separate work.
