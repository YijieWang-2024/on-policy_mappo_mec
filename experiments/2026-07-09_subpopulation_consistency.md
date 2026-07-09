# 2026-07-09 Subpopulation Consistency Gate

## Status

Stopped as out of scope after 193.6k of 1.5M steps. The paper studies a fixed
finite fleet at K=16; cross-K transfer and smaller-subpopulation feedback are not
the target control problem. The implementation was removed after stopping the
processes.

## Hypothesis

The full empirical-measure actor feature should be recoverable from a smaller
random subpopulation estimate. Enforcing feature consistency may reduce the
cross-K fragility observed for `csd_pool` without reconstructing the full state
or learning a rollout model.

For a rollout observation, the full-population actor feature is used as a
stop-gradient target. A uniformly sampled half-population is pooled through the
same CSD and actor MLP, and mean squared feature error is added to the actor
objective.

## Gate

- Scenario and training fleet: `v7_random_split_hotspots`, K=16
- Architecture: actor and critic `csd_pool`
- Training seed: 1
- Budget: 1.5M environment steps
- Hidden size: 144
- PPO epochs: 15
- Sample ratio: 0.5
- Consistency coefficients: `{0.1, 1, 10}`
- Validation: 8 episodes every 50 updates
- Heldout: 24 shared episodes at K=16 and K=8

The three coefficients run concurrently. Promote one coefficient only if it
keeps the K=16 result competitive with the unregularized seed1 checkpoint and
materially improves K=8. Otherwise stop this auxiliary direction.

Generated outputs:

`artifacts/runs/v7_random_split_hotspots/v7_csd_subcons_gate_20260709/`

## Early Diagnostic Only

At 160k validation steps, coefficients `{0.1, 1, 10}` obtained rewards
`{-779.601, -837.513, -852.215}`. These incomplete runs are not formal evidence
and must not appear in the main result tables.
