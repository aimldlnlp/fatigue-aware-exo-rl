from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exo_rl.utils.io import ensure_dir, load_json
from exo_rl.viz.style import POLICY_COLORS, POLICY_LABELS, POLICY_ORDER, apply_plot_style, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="outputs_sweep_default/summary.json")
    parser.add_argument("--outdir", type=str, default="outputs_sweep_default/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(args.summary)
    outdir = ensure_dir(args.outdir)
    apply_plot_style()
    aggregate = summary["aggregate"]
    policies = POLICY_ORDER
    labels = [POLICY_LABELS[p] for p in policies]
    colors = [POLICY_COLORS[p] for p in policies]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), constrained_layout=True)
    fig.suptitle("Multi-Seed Performance Summary", fontsize=18, fontweight="normal", y=1.02)

    success_means = [aggregate[p]["success_rate"] for p in policies]
    success_stds = [aggregate[p]["success_rate_std"] for p in policies]
    axes[0, 0].bar(labels, success_means, yerr=success_stds, color=colors, capsize=6)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Success Rate Across Seeds")
    style_axis(axes[0, 0])
    axes[0, 0].tick_params(axis="x", rotation=18)

    assist_means = [aggregate[p]["avg_assist_ratio"] for p in policies]
    distance_means = [aggregate[p]["avg_distance"] for p in policies]
    for label, color, assist, distance in zip(labels, colors, assist_means, distance_means, strict=True):
        axes[0, 1].scatter(assist, distance, s=150, color=color, alpha=0.9)
        axes[0, 1].annotate(label, (assist, distance), xytext=(8, 6), textcoords="offset points", fontsize=10)
    axes[0, 1].set_xlabel("Average Assist Ratio")
    axes[0, 1].set_ylabel("Average Distance")
    axes[0, 1].set_title("Support Efficiency Trade-off")
    style_axis(axes[0, 1])

    time_means = [aggregate[p]["time_to_target"] for p in policies]
    time_stds = [aggregate[p]["time_to_target_std"] for p in policies]
    axes[1, 0].bar(labels, time_means, yerr=time_stds, color=colors, capsize=6)
    axes[1, 0].set_title("Time to First Success")
    axes[1, 0].set_ylabel("Steps")
    style_axis(axes[1, 0])
    axes[1, 0].tick_params(axis="x", rotation=18)

    seed_runs = summary["seed_runs"]
    seed_labels = [str(run["seed"]) for run in seed_runs]
    ppo_success = [run["metrics"]["ppo_rl"]["success_rate"] for run in seed_runs]
    ppo_assist = [run["metrics"]["ppo_rl"]["avg_assist_ratio"] for run in seed_runs]
    ppo_distance = [run["metrics"]["ppo_rl"]["avg_distance"] for run in seed_runs]
    axes[1, 1].plot(seed_labels, ppo_success, marker="o", linewidth=2.5, color="#0f766e", label="success")
    axes[1, 1].plot(seed_labels, ppo_assist, marker="s", linewidth=2.0, color="#7c3aed", label="assist")
    axes[1, 1].plot(seed_labels, ppo_distance, marker="^", linewidth=2.0, color="#b45309", label="distance")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("PPO Variation Across Seeds")
    style_axis(axes[1, 1])
    axes[1, 1].legend(frameon=False, ncol=3, loc="upper center", fontsize=9, handlelength=2.2)
    fig.savefig(Path(outdir) / "multiseed_summary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
