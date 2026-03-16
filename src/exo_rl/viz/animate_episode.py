from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from exo_rl.agents.policy import ActorCritic
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.viz.common import arm_points, figure_to_rgb, save_frames_mp4_gif
from exo_rl.viz.style import METRIC_COLORS, POLICY_COLORS, apply_plot_style, style_axis

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best.pt")
    parser.add_argument("--mp4", type=str, default="outputs/videos/policy_episode.mp4")
    parser.add_argument("--gif", type=str, default="outputs/videos/policy_episode.gif")
    return parser.parse_args()


def load_policy(model_path: str | Path) -> tuple[ActorCritic, dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"], config["train"]["hidden_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def main() -> None:
    args = parse_args()
    apply_plot_style()
    model, config = load_policy(args.model)
    env = Arm2DEnv(config["env"], seed=int(config["seed"]) + 999)
    obs, info = env.reset()
    frames: list[np.ndarray] = []
    target = info["target_xy"]
    for _ in range(env.episode_steps):
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
        with torch.no_grad():
            mean, _ = model.distribution(obs_tensor)
            action = torch.sigmoid(mean).squeeze(0).numpy()
        result = env.step(action)
        points = arm_points(env.q, env.link_lengths)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        axes[0].plot(points[:, 0], points[:, 1], marker="o", linewidth=3.8, color=POLICY_COLORS["ppo_rl"])
        axes[0].scatter([target[0]], [target[1]], s=120, c="#b91c1c", marker="x")
        axes[0].set_xlim(-1.05, 1.05)
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].set_aspect("equal")
        axes[0].set_title("Fatigue-Aware Exoskeleton")
        style_axis(axes[0])
        axes[1].bar(
            ["fatigue", "human effort", "assist ratio"],
            [result.info["fatigue"], result.info["human_effort"], result.info["assist_ratio"]],
            color=[METRIC_COLORS["fatigue"], METRIC_COLORS["human"], METRIC_COLORS["assist"]],
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"distance {result.info['distance']:.3f}")
        style_axis(axes[1])
        fig.tight_layout()
        frames.append(figure_to_rgb(fig))
        plt.close(fig)
        obs = result.obs
        if result.terminated or result.truncated:
            break
    save_frames_mp4_gif(frames, Path(args.mp4), Path(args.gif), fps=20)
    print(f"saved {args.mp4}")
    print(f"saved {args.gif}")


if __name__ == "__main__":
    main()
