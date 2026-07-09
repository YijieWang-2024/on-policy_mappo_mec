# 2026-07-02 migration resetperm1500 seed1 validation

## Purpose

Validate that the first MEC core refactor in `on-policy_mappo_mec` preserves
the main training behavior from the previous `on-policy` implementation. This
refactor restarts the project around a cleaner MEC environment and MAPPO
integration after earlier experiments could not make Flat reliably learn a
good descriptor.

## Compared runs

Old `on-policy` baselines:

- `resetperm1500_mean_baseline_mean_mean_seed1`
- `resetperm1500_actor_flat_flat_seed1`

New `on-policy_mappo_mec` runs:

- `port_resetperm1500_mean_seed1`
- `port_resetperm1500_flat_seed1`

## Shared command template

```powershell
conda run -n marl python -m onpolicy.scripts.train.train_mec `
  --env_name MEC `
  --algorithm_name mappo `
  --experiment_name <experiment_name> `
  --mec_scenario v6_hap_loadbearing `
  --mec_policy_arch <mean|flat> `
  --seed 1 `
  --n_rollout_threads 16 `
  --n_eval_rollout_threads 8 `
  --episode_length 350 `
  --mec_episode_horizon 350 `
  --num_env_steps 1500000 `
  --ppo_epoch 5 `
  --num_mini_batch 1 `
  --hidden_size 128 `
  --layer_N 2 `
  --mec_logstd_init -1.2 `
  --entropy_coef 0.003 `
  --use_entropy_anneal `
  --mec_rolewise_loss `
  --use_eval `
  --eval_interval 18 `
  --eval_episodes 24 `
  --eval_seed 1000 `
  --save_step_checkpoints `
  --use_wandb
```

`--use_wandb` follows the inherited codebase convention where the local
TensorBoard path is used when WandB is disabled by the flag logic.

## Key parameters

| Field | Value |
| --- | --- |
| Scenario | `v6_hap_loadbearing` |
| Seed | `1` |
| Agents | `17` |
| Horizon | `350` |
| Environment steps | `1,500,000` |
| Rollout threads | `16` |
| Eval threads | `8` |
| PPO epochs | `5` |
| Mini-batches | `1` |
| Hidden size / layers | `128 / 2` |
| LR / critic LR | `5e-4 / 5e-4` |
| Entropy coef | `0.003` with annealing |
| MEC action std init | `-1.2` |
| Beta init | `alpha=2.0`, `eta=2.0` |
| Role-wise PPO loss | enabled |
| Fixed validation | `24` episodes, seed `1000` |

## Results

Source summary:

- `experiments/archive/generated_outputs/v6_hap_loadbearing/migration_compare_resetperm1500_seed1/summary.json`
- `experiments/archive/generated_outputs/v6_hap_loadbearing/migration_compare_resetperm1500_seed1/mean_old_vs_new_episode_reward.png`
- `experiments/archive/generated_outputs/v6_hap_loadbearing/migration_compare_resetperm1500_seed1/flat_old_vs_new_episode_reward.png`

| Architecture | Old best eval reward | New best eval reward | Old last eval reward | New last eval reward |
| --- | ---: | ---: | ---: | ---: |
| Mean | `-720.1961` | `-704.1445` | `-775.3277` | `-754.5024` |
| Flat | `-910.7025` | `-970.4603` | `-910.7025` | `-970.4603` |

Best fixed-validation checkpoints:

| Run | Best reward | Step | Validation episodes | Validation seed |
| --- | ---: | ---: | ---: | ---: |
| `port_resetperm1500_mean_seed1` | `-704.1445` | `1,209,600` | `24` | `1000` |
| `port_resetperm1500_flat_seed1` | `-970.4603` | `1,495,200` | `24` | `1000` |

Training throughput was about `244-245 FPS` near the end of both runs.

## Interpretation boundary

This is a migration validation, not a final algorithm claim. The Mean run
matches or slightly improves over the previous implementation. The Flat run is
worse by about `60` reward on the fixed validation curve, so it is acceptable
as a first-port sanity check but should remain on the follow-up list if Flat
equivalence matters for later descriptor diagnostics.
