from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exo_rl.agents.policy import ActorCritic
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/models/best.pt")
    parser.add_argument("--outdir", type=str, default="outputs/figures")
    parser.add_argument("--grid-points", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    config = checkpoint["config"]
    model = ActorCritic(checkpoint["obs_dim"], checkpoint["act_dim"], config["train"]["hidden_sizes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    outdir = ensure_dir(args.outdir)

    env = Arm2DEnv(config["env"], seed=int(config["seed"]) + 1234)
    radius = sum(config["env"]["link_lengths"]) * 0.8
    grid = np.linspace(-radius, radius, int(args.grid_points))
    score = np.full((len(grid), len(grid)), np.nan, dtype=np.float64)

    for ix, x in enumerate(grid):
        for iy, y in enumerate(grid):
            if np.sqrt(x * x + y * y) > radius or np.sqrt(x * x + y * y) < 0.2:
                continue
            obs, _ = env.reset()
            env.target_xy = np.array([x, y], dtype=np.float64)
            obs = env._get_obs()
            successes = []
            for _ in range(env.episode_steps):
                obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32)
                with torch.no_grad():
                    mean, _ = model.distribution(obs_tensor)
                    action = torch.sigmoid(mean).squeeze(0).numpy()
                result = env.step(action)
                successes.append(float(result.info["success"]))
                obs = result.obs
                if result.truncated or result.terminated:
                    break
            score[iy, ix] = np.mean(successes)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(
        score,
        origin="lower",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title("Workspace Success Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="success rate")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "workspace_heatmap.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
