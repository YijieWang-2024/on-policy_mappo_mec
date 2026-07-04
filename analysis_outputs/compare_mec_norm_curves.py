"""Compare MEC normalization ablations from TensorBoard event files.

This script is intentionally experiment-specific: it compares the reset-perm
mean/seed1 2x2 normalization runs plus the component-RMS extension.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path("onpolicy/scripts/results/MEC/v6_hap_loadbearing/mappo")


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    feature_norm: str
    rms: str
    path: Path


RUNS = [
    RunSpec(
        "feat_on_no_rms",
        "No RMS",
        "FeatureNorm on",
        "none",
        ROOT / "postchange_resetperm1500_mean_seed1/run1",
    ),
    RunSpec(
        "feat_on_flat_rms",
        "Flat RMS",
        "FeatureNorm on",
        "flat",
        ROOT / "postchange_rms_resetperm1500_mean_seed1/run2",
    ),
    RunSpec(
        "feat_on_component_rms",
        "Component RMS",
        "FeatureNorm on",
        "component",
        ROOT / "postchange_component_rms_resetperm1500_mean_seed1/run1",
    ),
    RunSpec(
        "feat_off_no_rms",
        "No RMS",
        "FeatureNorm off",
        "none",
        ROOT / "postchange_nofeat_resetperm1500_mean_seed1/run1",
    ),
    RunSpec(
        "feat_off_flat_rms",
        "Flat RMS",
        "FeatureNorm off",
        "flat",
        ROOT / "postchange_rms_nofeat_resetperm1500_mean_seed1/run1",
    ),
    RunSpec(
        "feat_off_component_rms",
        "Component RMS",
        "FeatureNorm off",
        "component",
        ROOT / "postchange_component_rms_nofeat_resetperm1500_mean_seed1/run1",
    ),
]


TAGS = {
    "train": "average_episode_rewards",
    "eval": "eval_average_episode_rewards",
}


def read_scalar(run: RunSpec, tag: str) -> list[dict[str, float | int | str]]:
    log_dir = run.path / "logs"
    if not log_dir.exists():
        return []
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    try:
        accumulator.Reload()
    except Exception as exc:  # pragma: no cover - diagnostic output path
        print(f"warning: failed to load {log_dir}: {exc}")
        return []
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    rows = []
    for event in accumulator.Scalars(tag):
        rows.append(
            {
                "run": run.key,
                "label": run.label,
                "feature_norm": run.feature_norm,
                "rms": run.rms,
                "step": int(event.step),
                "value": float(event.value),
            }
        )
    return rows


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0 or window <= 1:
        return arr
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i + 1 - window)
        out[i] = arr[lo : i + 1].mean()
    return out


def summarize(rows: list[dict[str, float | int | str]], metric: str) -> list[dict[str, float | int | str | None]]:
    by_run: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)

    summaries = []
    for spec in RUNS:
        run_rows = sorted(by_run.get(spec.key, []), key=lambda item: int(item["step"]))
        if not run_rows:
            summaries.append(
                {
                    "metric": metric,
                    "run": spec.key,
                    "feature_norm": spec.feature_norm,
                    "rms": spec.rms,
                    "points": 0,
                    "first_step": None,
                    "last_step": None,
                    "last_reward": None,
                    "best_step": None,
                    "best_reward": None,
                    "last5_mean": None,
                    "early_500k_mean": None,
                    "auc_mean": None,
                }
            )
            continue
        steps = np.asarray([int(row["step"]) for row in run_rows], dtype=np.float64)
        values = np.asarray([float(row["value"]) for row in run_rows], dtype=np.float64)
        best_idx = int(values.argmax())
        early = values[steps <= 500_000]
        if len(steps) > 1:
            auc_mean = float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))
        else:
            auc_mean = float(values[-1])
        summaries.append(
            {
                "metric": metric,
                "run": spec.key,
                "feature_norm": spec.feature_norm,
                "rms": spec.rms,
                "points": int(len(run_rows)),
                "first_step": int(steps[0]),
                "last_step": int(steps[-1]),
                "last_reward": float(values[-1]),
                "best_step": int(steps[best_idx]),
                "best_reward": float(values[best_idx]),
                "last5_mean": float(values[-min(5, len(values)) :].mean()),
                "early_500k_mean": float(early.mean()) if len(early) else None,
                "auc_mean": auc_mean,
            }
        )
    return summaries


def common_last_step(rows: list[dict[str, float | int | str]]) -> int | None:
    by_run: dict[str, list[int]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(int(row["step"]))
    last_steps = [max(by_run[run.key]) for run in RUNS if by_run.get(run.key)]
    if len(last_steps) != len(RUNS):
        return None
    return min(last_steps)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, float | int | str]],
    metric: str,
    output: Path,
    smooth_window: int,
) -> None:
    by_run: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = {"none": "#333333", "flat": "#1f77b4", "component": "#d62728"}
    linestyles = {"none": "-", "flat": "--", "component": "-"}

    for ax, feature_norm in zip(axes, ["FeatureNorm on", "FeatureNorm off"]):
        for spec in [run for run in RUNS if run.feature_norm == feature_norm]:
            run_rows = sorted(by_run.get(spec.key, []), key=lambda item: int(item["step"]))
            if not run_rows:
                continue
            steps = [int(row["step"]) for row in run_rows]
            values = [float(row["value"]) for row in run_rows]
            plotted_values = rolling_mean(values, smooth_window)
            ax.plot(
                steps,
                plotted_values,
                label=spec.label,
                color=colors[spec.rms],
                linestyle=linestyles[spec.rms],
                linewidth=2.0,
            )
        ax.set_title(feature_norm)
        ax.set_xlabel("env steps")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes[0].set_ylabel("episode reward (higher is better)")
    fig.suptitle(f"MEC mean seed1 normalization comparison: {metric}")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def add_deltas(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    by_metric_feature = {}
    for row in summaries:
        if row["rms"] == "none":
            by_metric_feature[(row["metric"], row["feature_norm"])] = row

    enriched = []
    for row in summaries:
        item = dict(row)
        base = by_metric_feature.get((row["metric"], row["feature_norm"]))
        for field in ["best_reward", "last_reward", "last5_mean", "early_500k_mean", "auc_mean"]:
            value = row.get(field)
            base_value = base.get(field) if base else None
            item[f"{field}_delta_vs_no_rms"] = (
                float(value) - float(base_value)
                if value is not None and base_value is not None
                else None
            )
        enriched.append(item)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="eval_outputs/norm_2x3_component_mean_seed1")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_summaries = []
    all_prefix_summaries = []
    common_steps = {}
    for metric, tag in TAGS.items():
        metric_rows = []
        for run in RUNS:
            metric_rows.extend(read_scalar(run, tag))
        metric_rows = sorted(metric_rows, key=lambda row: (str(row["run"]), int(row["step"])))
        write_csv(output_dir / f"episode_reward_{metric}_2x3.csv", metric_rows)
        plot_metric(
            metric_rows,
            metric,
            output_dir / f"episode_reward_{metric}_2x3_smoothed.png",
            args.smooth_window,
        )
        all_rows.extend(metric_rows)
        all_summaries.extend(summarize(metric_rows, metric))
        metric_common_step = common_last_step(metric_rows)
        common_steps[metric] = metric_common_step
        if metric_common_step is not None:
            prefix_rows = [
                row for row in metric_rows if int(row["step"]) <= int(metric_common_step)
            ]
            all_prefix_summaries.extend(summarize(prefix_rows, f"{metric}_prefix"))

    summary_rows = add_deltas(all_summaries)
    prefix_summary_rows = add_deltas(all_prefix_summaries)
    write_csv(output_dir / "summary_table.csv", summary_rows)
    write_csv(output_dir / "summary_prefix_common_step_table.csv", prefix_summary_rows)
    summary = {
        "protocol": {
            "scenario": "v6_hap_loadbearing",
            "arch": "mean",
            "seed": 1,
            "metric_direction": "higher episode_reward is better because rewards are negative costs",
            "smooth_window": args.smooth_window,
            "note": "FeatureNorm off is produced by passing --use_feature_normalization because the upstream flag is store_false.",
        },
        "common_steps": common_steps,
        "runs": [
            {
                "key": run.key,
                "feature_norm": run.feature_norm,
                "rms": run.rms,
                "path": str(run.path),
            }
            for run in RUNS
        ],
        "summary": summary_rows,
        "summary_prefix_common_step": prefix_summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
