# 2026-07-09 Fixed-K Spatial Resolution Experiment

## Hypothesis

At fixed K=16, the gain from `hotspot_pool` to `anchor_pool` indicates that
spatially distributed local empirical-measure statistics matter more than global
moments. A denser fixed anchor grid may reduce spatial quantization ambiguity
without introducing a learned attention encoder or reconstruction network.

## Compared Descriptors

- `hotspot_pool`: active workload centers only
- `anchor_pool`: workload centers plus five fixed cross anchors
- `grid_pool`: workload centers plus a fixed 3x3 uniform grid

Each anchor records soft local mass, mean position offset, and mean queue. The
new method changes only the fixed spatial resolution.

## Protocol

- Scenario and fleet: `v7_random_split_hotspots`, K=16
- Training seeds: `{1, 2, 3}`
- Budget: 3.5M environment steps
- Hidden size: 144
- PPO epochs: 15
- Validation: 8 episodes every 50 updates
- Final heldout: 100 episodes at `200000 + 13i`
- Concurrency: 3

Parameter counts at hidden size 144:

| Architecture | Actor + critic parameters |
|---|---:|
| anchor_pool | 94,727 |
| grid_pool | 101,735 |
| csd_pool | 97,355 |
| mhd_deepsets | 318,863 |
| slot_query | 1,935,041 |

Outputs:

`artifacts/runs/v7_random_split_hotspots/v7_grid_pool_3500k_20260709/`

## Results

All three runs and their 100-episode heldout evaluations completed.

| Method | Reward | Cost | Accept | Accepted Mbit | Overflow Mbit | W1 |
|---|---:|---:|---:|---:|---:|---:|
| hotspot_pool | -532.692 +/- 27.473 | 2.663 | 0.626 | 93.971 | 1.621 | 814.739 |
| anchor_pool | -478.846 +/- 12.990 | 2.394 | 0.692 | 103.754 | 0.662 | 769.559 |
| csd_pool | -471.863 +/- 26.627 | 2.359 | 0.708 | 106.175 | 0.706 | 732.751 |
| grid_pool | -450.890 +/- 22.913 | 2.254 | 0.720 | 107.959 | 1.127 | 742.562 |

Paired hierarchical-bootstrap reward comparisons:

| Comparison | Mean delta | Per-seed deltas | 95% CI |
|---|---:|---|---|
| grid - anchor | +27.957 | -0.831, +29.671, +55.030 | [+0.229, +54.746] |
| grid - CSD | +20.974 | -6.802, +8.411, +61.311 | [-9.122, +59.070] |

Increasing the fixed spatial resolution from five cross anchors to a 3x3 grid
improves the heldout team objective relative to `anchor_pool`, although one
training seed is effectively tied. The result establishes a useful spatial
inductive bias; it does not yet establish a learned population descriptor.
