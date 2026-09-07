# TBX-Agent synthetic system suite v1.2

This directory is the current, additive, non-clinical software-regression suite.
Its first 24 records are a byte-for-byte prefix copy of v1.1.0
(`cases_sha256=a32916159e9c570e843cd13f51c3f7444677fb38f349f714fcfbc1990cd10fef`).
The six added records cover the active `native_three_class_argmax` visual contract:

- `healthy` routes to `model_not_flagged`;
- `sick_non_tb` routes to `pending_human_review`;
- `tb` routes to `model_flagged`;
- a high detector score cannot override a `healthy` classifier route;
- an empty detector result cannot exclude a `tb` classifier route;
- an image-quality warning overrides `healthy` and requires human review.

All probabilities, detections and quality states are synthetic fixture evidence. The
deterministic adapter does not load rank03, D-FINE, PSPNet, Qwen, a patient image or a
clinical dataset. Passing this suite demonstrates only the checked software contracts;
it is not model-performance or clinical validation.

Use `evaluation/system_bench_config.json` or the identical version-pinned
`evaluation/system_bench_config_v1_2.json` for the current suite. Historical v1.1.0
remains reproducible with `evaluation/system_bench_config_v1_1.json` and the immutable
`evaluation/suites/system_v1` files. Reports from different suite versions must not be
treated as an exact paired comparison.

The adapter binds every observation to the suite version, cases SHA-256, candidate
configuration SHA-256 and adapter identity. It refuses an existing output directory and
retains failures in the hash-chained ledger. The suite is never used for model or
threshold selection and never reads a locked or hidden test split.
