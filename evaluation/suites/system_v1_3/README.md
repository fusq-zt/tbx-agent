# TBX-Agent system suite v1.3

This synthetic, non-clinical suite preserves every v1.2 contract except the
explicitly versioned three-class routing change: `sick_non_tb` now maps directly
to `non_tb_abnormal` and does not enter the batch-only review queue. The v1.2
suite remains unchanged for historical replay.

The suite does not use TBX11K, a locked split, a hidden test set, patient data,
or model weights. It is a software and Agent contract regression suite only.
