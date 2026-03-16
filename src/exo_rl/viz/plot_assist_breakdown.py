from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exo_rl.utils.io import ensure_dir
from exo_rl.viz.policy_rollouts import rollout_all_policies
from exo_rl.viz.style import METRIC_COLORS, POLICY_COLORS, apply_plot_style, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best.pt")
    parser.add_argument("--outdir", type=str, default="outputs/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    apply_plot_style()
    _config, rollouts = rollout_all_policies(args.model)
    ppo_rollout = next(item for item in rollouts if item["policy"] == "ppo_rl")
    history = ppo_rollout["history"]
    steps = np.arange(1, len(history) + 1)
    human_joint_1 = np.array([abs(step["human_torque"][0]) for step in history], dtype=np.float64)
    human_joint_2 = np.array([abs(step["human_torque"][1]) for step in history], dtype=np.float64)
    assist_joint_1 = np.array([abs(step["assist_torque"][0]) for step in history], dtype=np.float64)
    assist_joint_2 = np.array([abs(step["assist_torque"][1]) for step in history], dtype=np.float64)
    fatigue = np.array([step["fatigue"] for step in history], dtype=np.float64)
    assist_ratio = np.array([step["assist_ratio"] for step in history], dtype=np.float64)
    distance = np.array([step["distance"] for step in history], dtype=np.float64)

    fig = plt.figure(figsize=(13.5, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    fig.suptitle("Assist Breakdown for Best PPO Rollout", fontsize=18, fontweight="normal")

    ax1.stackplot(
        steps,
        human_joint_1,
        assist_joint_1,
        labels=["human torque | joint 1", "exo torque | joint 1"],
        colors=[METRIC_COLORS["human"], METRIC_COLORS["exo"]],
        alpha=0.88,
    )
    ax1.set_title("Joint 1 Effort Sharing")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Abs torque")
    style_axis(ax1)
    ax1.legend(frameon=False, loc="upper right", fontsize=9)

    ax2.stackplot(
        steps,
        human_joint_2,
        assist_joint_2,
        labels=["human torque | joint 2", "exo torque | joint 2"],
        colors=["#8aa6ca", "#67958e"],
        alpha=0.88,
    )
    ax2.set_title("Joint 2 Effort Sharing")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Abs torque")
    style_axis(ax2)
    ax2.legend(frameon=False, loc="upper right", fontsize=9)

    ax3.plot(steps, fatigue, color=METRIC_COLORS["fatigue"], linewidth=2.5, label="fatigue")
    ax3.plot(steps, assist_ratio, color=METRIC_COLORS["assist"], linewidth=2.2, label="assist ratio")
    ax3.fill_between(steps, 0, fatigue, color=METRIC_COLORS["fatigue"], alpha=0.07)
    ax3.set_title("Fatigue and Assistance Over Time")
    ax3.set_xlabel("Step")
    ax3.set_ylim(0, 1.0)
    style_axis(ax3)
    ax3.legend(frameon=False, loc="upper left", fontsize=9)

    cumulative_human = np.cumsum((human_joint_1 + human_joint_2) * 0.5)
    cumulative_exo = np.cumsum((assist_joint_1 + assist_joint_2) * 0.5)
    ax4.plot(steps, cumulative_human, color=METRIC_COLORS["human"], linewidth=2.5, label="cumulative human effort")
    ax4.plot(steps, cumulative_exo, color=POLICY_COLORS["ppo_rl"], linewidth=2.5, label="cumulative exo effort")
    ax4.plot(steps, distance, color="#5c5c5c", linewidth=1.6, linestyle="--", label="distance")
    ax4.set_title("Effort Budget vs Distance")
    ax4.set_xlabel("Step")
    style_axis(ax4)
    ax4.legend(frameon=False, loc="upper right", fontsize=9)

    fig.savefig(Path(outdir) / "assist_breakdown.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
