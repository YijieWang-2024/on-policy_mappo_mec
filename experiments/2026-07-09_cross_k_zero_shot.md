# 2026-07-09 Cross-K Zero-Shot Evaluation

## Hypothesis

A permutation-invariant empirical-measure descriptor should permit a policy
trained at fleet size K=16 to execute without retraining at unseen finite fleet
sizes. Adding fleet mean and standard deviation to the anchor descriptor may
improve this transfer.

## Protocol

- Training scenario: `v7_random_split_hotspots`, K=16
- Evaluation fleet sizes: K in `{8, 16, 24, 32}`
- Methods: `anchor_pool`, `csd_pool`, `mean`
- Training seeds: `{1, 2, 3}`
- Held-out episodes per cell: 24
- Evaluation seeds: `100000 + 13i`
- Concurrent evaluation processes: at most 3
- K=16 values reuse the matched held-out evaluations from Batch A.

Run:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\2026-07-09_cross_k_zero_shot\run.ps1
```

Generated outputs:

`artifacts/evaluations/v7_random_split_hotspots/cross_k_zero_shot_20260709/`

## Decision Rule

Compare methods within each K. Treat cross-K reward levels as environment-scale
changes, not as a learning curve. Promote CSD as the main descriptor only if its
paired improvement over anchor is consistent across training seeds and at more
than one unseen K.

## Results

Values are means over three independently trained policies; reward reports
mean +/- sample standard deviation across training seeds.

| K | Method | Reward | Cost | Accept | Overflow Mbit | W1 |
|---:|---|---:|---:|---:|---:|---:|
| 8 | mean | -791.473 +/- 8.577 | 3.957 | 0.426 | 4.291 | 1192.252 |
| 8 | anchor_pool | -792.361 +/- 35.872 | 3.962 | 0.503 | 10.136 | 1139.098 |
| 8 | csd_pool | -849.344 +/- 57.003 | 4.247 | 0.437 | 12.927 | 1292.365 |
| 16 | mean | -606.658 +/- 30.557 | 3.033 | 0.520 | 1.990 | 862.249 |
| 16 | anchor_pool | -497.362 +/- 16.717 | 2.487 | 0.673 | 0.691 | 798.013 |
| 16 | csd_pool | -490.097 +/- 29.663 | 2.450 | 0.686 | 1.057 | 755.853 |
| 24 | mean | -614.223 +/- 74.770 | 3.071 | 0.621 | 0.297 | 699.851 |
| 24 | anchor_pool | -749.541 +/- 134.717 | 3.748 | 0.438 | 0.000 | 1061.598 |
| 24 | csd_pool | -625.222 +/- 85.571 | 3.126 | 0.548 | 0.408 | 822.147 |
| 32 | mean | -721.015 +/- 147.691 | 3.605 | 0.589 | 0.011 | 808.831 |
| 32 | anchor_pool | -992.681 +/- 109.459 | 4.963 | 0.266 | 0.000 | 1457.911 |
| 32 | csd_pool | -936.638 +/- 136.564 | 4.683 | 0.302 | 0.050 | 1447.688 |

Paired CSD-minus-anchor reward differences:

| K | Mean delta | Seed deltas |
|---:|---:|---|
| 8 | -56.984 | -26.097, -75.084, -69.770 |
| 16 | +7.265 | -3.526, +24.720, +0.601 |
| 24 | +124.319 | -2.268, +116.753, +258.472 |
| 32 | +56.043 | -65.759, +3.509, +230.379 |

The promotion rule is not met. CSD is competitive at training K=16 and helps
some seeds for larger K, but it consistently loses at K=8 and is highly
seed-dependent at K=24/32. Mean pooling is the strongest large-K zero-shot
baseline despite its weaker K=16 control.
