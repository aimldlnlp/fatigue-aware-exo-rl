from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exo_rl.agents.policy import ActorCritic
from exo_rl.baselines.fixed_assist import FixedAssistPolicy
from exo_rl.baselines.rule_based_fatigue import RuleBasedFatiguePolicy
from exo_rl.baselines.zero_assist import ZeroAssistPolicy
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.viz.common import arm_points, figure_to_rgb, save_frames_mp4_gif
from exo_rl.viz.style import METRIC_COLORS, POLICY_COLORS, POLICY_LABELS, apply_plot_style, style_axis, style_minimal_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best.pt")
    parser.add_argument("--mp4", type=str, default="outputs/videos/policy_comparison.mp4")
    parser.add_argument("--gif", type=str, default="outputs/videos/policy_comparison.gif")
    parser.add_argument("--seed-offset", type=int, default=1200)
    return parser.parse_args()


class RLPolicy:
    name = "ppo_rl"

    def __init__(self, model_path: str | Path) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint["config"]
        self.model = ActorCritic(
            checkpoint["obs_dim"],
            checkpoint["act_dim"],
            config["train"]["hidden_sizes"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.model.distribution(obs_tensor)
            action = torch.sigmoid(mean)
        return action.squeeze(0).numpy()


def load_config(model_path: str | Path) -> dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint["config"]


def draw_panel(
    ax: plt.Axes,
    name: str,
    q: np.ndarray,
    link_lengths: np.ndarray,
    target_xy: np.ndarray,
    trail: np.ndarray,
    info: dict,
    color: str,
) -> None:
    points = arm_points(q, link_lengths)
    if len(trail) > 1:
        ax.plot(trail[:, 0], trail[:, 1], color=color, alpha=0.22, linewidth=1.8)
    ax.plot(points[:, 0], points[:, 1], marker="o", linewidth=3.6, color=color)
    ax.scatter([target_xy[0]], [target_xy[1]], s=110, c="#b91c1c", marker="x", linewidths=2.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    style_minimal_axis(ax)
    ax.set_title(POLICY_LABELS[name], fontsize=11, color="#1d1d1d")
    info_text = (
        f"s {int(bool(info['success']))}   "
        f"d {info['distance']:.2f}   "
        f"a {info['assist_ratio']:.2f}   "
        f"f {info['fatigue']:.2f}"
    )
    ax.text(
        0.03,
        0.05,
        info_text,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
        bbox={"facecolor": "#ffffff", "edgecolor": "none", "alpha": 0.75, "pad": 1.8},
    )


def main() -> None:
    args = parse_args()
    apply_plot_style()
    config = load_config(args.model)
    policies = [
        (RLPolicy(args.model), POLICY_COLORS["ppo_rl"]),
        (ZeroAssistPolicy(), POLICY_COLORS["zero_assist"]),
        (FixedAssistPolicy(), POLICY_COLORS["fixed_assist"]),
        (RuleBasedFatiguePolicy(), POLICY_COLORS["rule_based_fatigue"]),
    ]
    envs = []
    observations = []
    seed = int(config["seed"]) + int(args.seed_offset)
    target_xy = None
    for _, _color in policies:
        env = Arm2DEnv(config["env"], seed=seed)
        obs, info = env.reset()
        envs.append(env)
        observations.append(obs)
        target_xy = info["target_xy"].copy()
    assert target_xy is not None

    done_flags = [False] * len(policies)
    latest_infos = [
        {
            "success": False,
            "distance": float(np.linalg.norm(target_xy - env.target_xy)),
            "assist_ratio": 0.0,
            "fatigue": env.fatigue,
            "human_effort": 0.0,
        }
        for env in envs
    ]
    trails: list[list[np.ndarray]] = [[] for _ in policies]
    frames: list[np.ndarray] = []

    for step in range(envs[0].episode_steps):
        for idx, ((policy, _color), env) in enumerate(zip(policies, envs, strict=True)):
            ee = np.array(env.history[-1]["end_effector_xy"]) if env.history else env.target_xy * 0.0
            if not done_flags[idx]:
                action = policy.act(observations[idx])
                result = env.step(action)
                observations[idx] = result.obs
                latest_infos[idx] = result.info
                done_flags[idx] = bool(result.terminated or result.truncated)
                ee = np.array(result.info["end_effector_xy"])
            trails[idx].append(ee)

        fig = plt.figure(figsize=(13.5, 8.0))
        grid = fig.add_gridspec(3, 4, height_ratios=[4.0, 4.0, 1.8], hspace=0.20, wspace=0.18)
        panel_axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 0:2]),
            fig.add_subplot(grid[1, 2:4]),
        ]
        metric_ax = fig.add_subplot(grid[2, :])
        fig.suptitle(
            "Policy Comparison: Fatigue-Aware Reaching Assistance",
            fontsize=16,
            fontweight="normal",
            y=0.98,
        )

        for ax, (policy, color), env, info, trail in zip(
            panel_axes,
            policies,
            envs,
            latest_infos,
            trails,
            strict=True,
        ):
            draw_panel(
                ax=ax,
                name=policy.name,
                q=env.q,
                link_lengths=env.link_lengths,
                target_xy=target_xy,
                trail=np.asarray(trail, dtype=np.float64),
                info=info,
                color=color,
            )

        policy_names = [policy.name.replace("_", "\n") for policy, _ in policies]
        assist = [info["assist_ratio"] for info in latest_infos]
        fatigue = [info["fatigue"] for info in latest_infos]
        distance = [info["distance"] for info in latest_infos]
        x = np.arange(len(policy_names))
        width = 0.23
        metric_ax.bar(x - width, assist, width=width, color=METRIC_COLORS["assist"], label="assist")
        metric_ax.bar(x, fatigue, width=width, color=METRIC_COLORS["fatigue"], label="fatigue")
        metric_ax.bar(x + width, np.clip(distance, 0.0, 1.0), width=width, color=METRIC_COLORS["distance"], label="distance")
        metric_ax.set_xticks(x, policy_names)
        metric_ax.set_ylim(0, 1.0)
        metric_ax.set_title(f"Step {step + 1}", fontsize=11)
        style_axis(metric_ax)
        metric_ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.3)

        frames.append(figure_to_rgb(fig))
        plt.close(fig)
        if all(done_flags):
            break

    save_frames_mp4_gif(frames, Path(args.mp4), Path(args.gif), fps=18)
    print(f"saved {args.mp4}")
    print(f"saved {args.gif}")


if __name__ == "__main__":
    main()
