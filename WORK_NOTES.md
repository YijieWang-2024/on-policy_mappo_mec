# Work Notes

## 2026-06-10 - MPE rendering setup corrections

### Summary

Improved the retained MPE rendering path so shared-policy rendering initializes
its output directory correctly and reports dependency or OpenGL setup failures
clearly.

### Key changes

- Initialized the shared runner's render-mode `run_dir` and `gifs` directory
  without depending on training or Weights & Biases setup.
- Added the compatible `pyglet>=1.5,<2` rendering dependency explicitly.
- Replaced invalid fallback print calls during rendering imports with actionable
  exceptions for missing pyglet and unavailable OpenGL.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
conda run -n marl python -c "import pyglet; print(pyglet.version)"
git diff --check
```

All 10 tests passed. The active `marl` environment reports pyglet version
`1.5.31`, which satisfies the pinned compatibility range.

## 2026-06-10 - Replay buffer indexing and sampling corrections

### Summary

Audited the shared and separated replay buffers after the Gymnasium migration
and corrected several indexing and sampling issues that could misalign rollout
transitions, omit training samples, or form invalid recurrent sequences.

### Key changes

- Standardized replay buffer indexing so transition data is stored at time
  `t`, while observations, recurrent states, and masks for the resulting state
  are stored at `t + 1`.
- Corrected return calculation to use each transition's actual next-value
  prediction and the mask belonging to its next state.
- Populated shared-buffer advantages for non-GAE return calculation.
- Changed feed-forward mini-batch sampling so remainder samples are retained
  instead of silently dropped.
- Added divisibility and size checks for naive-recurrent and recurrent
  mini-batches.
- Prevented recurrent chunks from crossing agent or environment trajectory
  boundaries.
- Corrected separated recurrent batches to flatten in time-major order expected
  by the recurrent layer.
- Removed the obsolete sequential-agent factor update path from the separated
  runner and added visible average-reward logging.
- Added focused regression tests for buffer insertion, return calculation,
  remainder sampling, recurrent ordering, and chunk-boundary validation.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
git diff --check
```

All 10 tests passed, including the 5 new replay-buffer indexing and sampling
regression tests and the existing 5 Gymnasium migration tests.

## 2026-06-10 - MPE Gymnasium migration and rollout semantics

### Summary

Completed the first substantial update after the repository was reduced to the
MPE-focused version. The work modernizes the retained MPE stack for Gymnasium
and NumPy 2, and corrects how episode termination and time-limit truncation are
handled during rollout and return calculation.

### Key changes

- Migrated MPE environments, spaces, wrappers, training, evaluation, and
  rendering flows from the legacy Gym API to Gymnasium.
- Replaced global scenario randomness with seeded per-environment generators,
  making seeded resets reproducible.
- Rebuilt the retained vector environment wrappers with Gymnasium-style reset,
  step, autoreset, seed, and final-observation handling.
- Separated true task termination from time-limit truncation:
  - recurrent state and GAE continuation stop at either boundary;
  - critic bootstrapping remains enabled for truncation;
  - critic bootstrapping is disabled for true termination.
- Updated shared and separated MPE runners and replay buffers to preserve and
  evaluate the actual final observation before autoreset.
- Added PPO diagnostics for approximate KL divergence, clipping fraction, and
  explained variance, plus optional `--target_kl` early stopping.
- Updated dependencies for Gymnasium and NumPy 2 compatibility.
- Added focused migration tests and documented the new semantics in `README.md`.

### Verification

Executed successfully on 2026-06-10:

```bash
conda run -n marl python -m unittest discover -s tests -v
conda run -n marl python -m compileall -q onpolicy tests
git diff --check
```

All 5 migration tests passed. The tests cover all retained MPE scenarios,
reproducible seeded resets, time-limit truncation, vector autoreset final
observations, and truncation-versus-termination bootstrapping.

### Notes

- `origin` is the private archive repository `YijieWang-2024/on-policy`.
- `upstream` remains the original `marlbenchmark/on-policy` repository.
- MPE environments should report external time limits as truncation and reserve
  termination for true task-ending conditions.
