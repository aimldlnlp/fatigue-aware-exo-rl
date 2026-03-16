from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exo_rl.utils.io import ensure_dir
from exo_rl.viz.policy_rollouts import rollout_all_policies
from exo_rl.viz.style import METRIC_COLORS, POLICY_COLORS, POLICY_LABELS, apply_plot_style, style_axis


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
    target_xy = rollouts[0]["target_xy"]

    fig = plt.figure(figsize=(13.5, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.24)
    ax_workspace = fig.add_subplot(grid[:, 0])
    ax_distance = fig.add_subplot(grid[0, 1])
    ax_fatigue = fig.add_subplot(grid[1, 1])
    fig.suptitle("Trajectory Comparison on a Shared Reaching Task", fontsize=18, fontweight="normal")

    for rollout in rollouts:
        policy = rollout["policy"]
        history = rollout["history"]
        traj = np.array([step["end_effector_xy"] for step in history], dtype=np.float64)
        distance = np.array([step["distance"] for step in history], dtype=np.float64)
        fatigue = np.array([step["fatigue"] for step in history], dtype=np.float64)
        assist = np.array([step["assist_ratio"] for step in history], dtype=np.float64)
        steps = np.arange(1, len(history) + 1)
        color = POLICY_COLORS[policy]
        label = POLICY_LABELS[policy]

        linewidth = 3.2 if policy == "ppo_rl" else 2.4
        alpha = 0.95 if policy == "ppo_rl" else 0.8
        ax_workspace.plot(traj[:, 0], traj[:, 1], linewidth=linewidth, color=color, alpha=alpha, label=label)
        ax_workspace.scatter([traj[0, 0]], [traj[0, 1]], color=color, s=55, alpha=0.55)
        ax_workspace.scatter([traj[-1, 0]], [traj[-1, 1]], color=color, s=90, edgecolor="#111827", linewidth=0.5)

        ax_distance.plot(steps, distance, linewidth=2.2 if policy == "ppo_rl" else 1.9, color=color, label=label)
        ax_fatigue.plot(steps, fatigue, linewidth=2.0, color=color, label=f"{label} fatigue")
        ax_fatigue.plot(steps, assist, linewidth=1.7, linestyle="--", color=color, alpha=0.7, label=f"{label} assist")

    ax_workspace.scatter([target_xy[0]], [target_xy[1]], marker="X", s=170, color="#b91c1c", label="target")
    ax_workspace.set_xlim(-1.05, 1.05)
    ax_workspace.set_ylim(-1.05, 1.05)
    ax_workspace.set_aspect("equal")
    ax_workspace.set_title("End-Effector Trajectories")
    ax_workspace.set_xlabel("x")
    ax_workspace.set_ylabel("y")
    style_axis(ax_workspace)
    ax_workspace.legend(frameon=False, loc="lower left", fontsize=9)

    ax_distance.set_title("Distance-to-Target")
    ax_distance.set_xlabel("Step")
    ax_distance.set_ylabel("Distance")
    style_axis(ax_distance)
    ax_distance.legend(frameon=False, loc="upper right", fontsize=9)

    ax_fatigue.set_title("Fatigue and Assistance")
    ax_fatigue.set_xlabel("Step")
    ax_fatigue.set_ylim(0, 1.0)
    style_axis(ax_fatigue)
    ax_fatigue.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.5, handlelength=2.0)

    fig.savefig(Path(outdir) / "trajectory_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
