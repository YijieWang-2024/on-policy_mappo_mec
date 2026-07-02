# Experiment tracking policy

This repository tracks reproducible experiment evidence, not full training
artifacts.

Commit these files:

- launch scripts or exact commands;
- distilled run configs and key hyperparameters;
- numerical summaries, comparison tables, and plots;
- short notes that state what was verified and what remains uncertain.

Do not commit these files directly:

- model checkpoints, optimizer states, and TensorBoard event files;
- large raw logs from long training jobs;
- local IDE state or cache directories.

Use `onpolicy/scripts/results/` as the local artifact store for full runs. If a
model must be archived long term, publish it as a release artifact or move the
project to Git LFS/DVC instead of storing large binaries in normal Git history.
