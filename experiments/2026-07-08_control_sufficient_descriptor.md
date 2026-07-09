# 2026-07-08 Control-Sufficient Descriptor Experiments

## Purpose

Test whether a low-complexity control-sufficient population descriptor can match or improve the existing v7 finite-K MEC baselines without slot attention, full set reconstruction, or learned world-model rollouts.

## New Architecture

`csd_pool` is a deterministic extension of `anchor_pool`:

- anchor/hotspot radial geometry statistics from `anchor_pool`
- UAV fleet mean over `(x, y, queue)`
- UAV fleet standard deviation over `(x, y, queue)`

This keeps the descriptor permutation-invariant and adds only global fleet dispersion/load statistics.

## Batch A: v7 3500k Multi-Seed

Comparable baseline family:

- `v7_3500k_anchor_pool_seed1-3`
- `v7_3500k_hotspot_pool_seed1-3`
- `v7_3500k_mean_seed1-3`
- `v7_3500k_flat_seed1-3`

Launched runs:

| Run | Seed | Status | PID |
|---|---:|---|---:|
| `v7_3500k_csd_pool_seed1` | 1 | running | 13436 |
| `v7_3500k_csd_pool_seed2` | 2 | running | 7936 |
| `v7_3500k_csd_pool_seed3` | 3 | running | 3176 |

Shared command parameters:

```text
python onpolicy/scripts/train/train_mec.py
  --env_name MEC
  --algorithm_name mappo
  --mec_scenario v7_random_split_hotspots
  --mec_policy_arch csd_pool
  --mec_critic_arch csd_pool
  --n_training_threads 1
  --n_rollout_threads 8
  --n_eval_rollout_threads 1
  --num_mini_batch 1
  --ppo_epoch 15
  --hidden_size 144
  --layer_N 1
  --episode_length 200
  --num_env_steps 3500000
  --use_eval
  --eval_interval 50
  --eval_episodes 8
  --test_episodes 24
  --use_wandb
  --log_interval 10
  --save_interval 25
```

Launcher logs:

`artifacts/runs/v7_random_split_hotspots/v7_csd_pool_3500k_20260708/`

Training outputs:

`onpolicy/scripts/results/MEC/v7_random_split_hotspots/mappo/v7_3500k_csd_pool_seed*/run1/`

## Decision Gate

After Batch A completes:

1. Run held-out evaluation on each best checkpoint.
2. Compare mean held-out reward, cost, acceptance, overflow, and W1 against existing v7 3500k baselines.
3. If `csd_pool` beats or matches `anchor_pool`, promote it as the main control-sufficient descriptor.
4. If it loses to `anchor_pool`, keep `anchor_pool` as the main architecture and use `csd_pool` as evidence that extra global dispersion statistics are not enough.
