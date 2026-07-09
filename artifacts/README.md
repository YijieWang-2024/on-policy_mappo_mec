# Local artifact layout

This directory is for local generated experiment artifacts. It is intentionally
ignored by Git except for this README. Keep durable, human-reviewed experiment
records in `experiments/` or `docs/`.

## Directory contract

| Path | Contents |
| --- | --- |
| `analysis/` | Generated plots, summary CSVs, and one-off analysis scripts. |
| `analysis/scripts/` | Local analysis helpers that produced artifact tables or figures. |
| `analysis/v6_hap_loadbearing/` | Generated v6/reset-permutation analysis outputs. |
| `analysis/v7_random_split_hotspots/` | Generated v7 analysis dashboards, plots, and summary tables. |
| `evaluations/` | Ad hoc held-out evaluation outputs, diagnostic summaries, and evaluation plots. |
| `evaluations/v6_hap_loadbearing/` | v6 held-out and normalization diagnostic outputs. |
| `runs/` | Raw launch logs, controller scripts, queue status files, and training/eval batch logs. |
| `runs/v6_hap_loadbearing_resetperm/` | Older v6 reset-permutation logs, including former `training_logs/`. |
| `runs/v7_random_split_hotspots/` | v7 batch run folders and their raw logs. |

## Promotion rule

When an artifact becomes part of the project record, copy or distill it into a
tracked file under `experiments/` or `docs/`. Do not commit raw TensorBoard
events, checkpoints, large logs, or transient monitor outputs.

Small generated outputs that were already part of the Git history were promoted
to `experiments/archive/generated_outputs/` during the 2026-07-08 cleanup.

## Retired top-level paths

The old top-level `analysis_outputs/`, `eval_outputs/`, `run_logs/`, and
`training_logs/` directories were consolidated here on 2026-07-08. They remain
ignored in `.gitignore` only so old scripts do not accidentally create tracked
files if rerun.
