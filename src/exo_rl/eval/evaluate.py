from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from exo_rl.agents.policy import ActorCritic
from exo_rl.baselines.fixed_assist import FixedAssistPolicy
from exo_rl.baselines.rule_based_fatigue import RuleBasedFatiguePolicy
from exo_rl.baselines.zero_assist import ZeroAssistPolicy
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.eval.metrics import aggregate_episode_infos
from exo_rl.utils.device import cuda_available
from exo_rl.utils.io import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--model", type=str, default="outputs/models/best.pt")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outputs-dir", type=str, default=None)
    return parser.parse_args()


class RLPolicy:
    name = "ppo_rl"

    def __init__(self, model_path: str | Path, device: str | torch.device = "cpu") -> None:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]
        self.model = ActorCritic(
            checkpoint["obs_dim"],
            checkpoint["act_dim"],
            config["train"]["hidden_sizes"],
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = torch.device(device)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, _ = self.model.distribution(obs_tensor)
            action = torch.sigmoid(mean)
        return action.squeeze(0).cpu().numpy()


def run_episode(env: Arm2DEnv, policy) -> tuple[dict[str, float], list[dict], np.ndarray]:
    obs, info = env.reset()
    target = info["target_xy"].copy()
    episode_infos: list[dict] = []
    while True:
        action = policy.act(obs)
        result = env.step(action)
        episode_infos.append(result.info)
        obs = result.obs
        if result.terminated or result.truncated:
            break
    return aggregate_episode_infos(episode_infos), episode_infos, target


def evaluate_policy(env_config: dict, seed: int, policy, episodes: int) -> tuple[dict[str, float], list[dict]]:
    env = Arm2DEnv(env_config, seed=seed)
    aggregate = []
    episode_logs = []
    for idx in range(episodes):
        metrics, infos, target = run_episode(env, policy)
        aggregate.append(metrics)
        episode_logs.append(
            {
                "episode": idx,
                "policy": policy.name,
                "metrics": metrics,
                "target_xy": target.tolist(),
                "trajectory": [info["end_effector_xy"].tolist() for info in infos],
                "fatigue": [float(info["fatigue"]) for info in infos],
                "human_effort": [float(info["human_effort"]) for info in infos],
                "assist_ratio": [float(info["assist_ratio"]) for info in infos],
            }
        )
    keys = aggregate[0].keys()
    summary = {key: float(np.mean([entry[key] for entry in aggregate])) for key in keys}
    return summary, episode_logs


def main() -> None:
    args = parse_args()
    config = deepcopy(load_json(args.config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.outputs_dir is not None:
        config["outputs_dir"] = args.outputs_dir
    episodes = int(args.episodes or config["eval"]["episodes"])
    outputs_dir = ensure_dir(config["outputs_dir"])
    eval_dir = ensure_dir(outputs_dir / "eval")
    rl_policy = RLPolicy(args.model, device="cuda" if cuda_available() else "cpu")
    policies = [
        rl_policy,
        ZeroAssistPolicy(),
        FixedAssistPolicy(),
        RuleBasedFatiguePolicy(),
    ]
    summaries: dict[str, dict[str, float]] = {}
    logs: dict[str, list[dict]] = {}
    for idx, policy in enumerate(policies):
        summary, episode_logs = evaluate_policy(config["env"], int(config["seed"]) + idx * 100, policy, episodes)
        summaries[policy.name] = summary
        logs[policy.name] = episode_logs
        print(
            f"{policy.name:18s} success={summary['success_rate']:.3f} "
            f"distance={summary['avg_distance']:.3f} effort={summary['avg_human_effort']:.3f} "
            f"assist={summary['avg_assist_ratio']:.3f} fatigue={summary['final_fatigue']:.3f}"
        )
    save_json(eval_dir / "metrics_summary.json", summaries)
    save_json(eval_dir / "episode_logs.json", logs)


if __name__ == "__main__":
    main()
