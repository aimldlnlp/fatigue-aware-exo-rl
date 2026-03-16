from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from exo_rl.agents.policy import ActorCritic
from exo_rl.baselines.fixed_assist import FixedAssistPolicy
from exo_rl.baselines.rule_based_fatigue import RuleBasedFatiguePolicy
from exo_rl.baselines.zero_assist import ZeroAssistPolicy
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.envs.dynamics import forward_kinematics


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


def build_policies(model_path: str | Path) -> list:
    return [
        RLPolicy(model_path),
        ZeroAssistPolicy(),
        FixedAssistPolicy(),
        RuleBasedFatiguePolicy(),
    ]


def make_shared_initial_state(config: dict, seed: int) -> dict:
    env = Arm2DEnv(config["env"], seed=seed)
    env.reset()
    return {
        "q": env.q.copy(),
        "dq": env.dq.copy(),
        "fatigue": float(env.fatigue),
        "target_xy": env.target_xy.copy(),
    }


def rollout_policy(env: Arm2DEnv, policy, initial_state: dict) -> dict:
    env.reset()
    env.q = initial_state["q"].copy()
    env.dq = initial_state["dq"].copy()
    env.fatigue = float(initial_state["fatigue"])
    env.target_xy = initial_state["target_xy"].copy()
    env.prev_action = np.zeros(env.action_size, dtype=np.float64)
    env.last_human_effort = 0.0
    env.last_assist_ratio = 0.0
    env.success_streak = 0
    env.history = []
    ee = forward_kinematics(env.q, env.link_lengths)
    env.prev_distance = float(np.linalg.norm(env.target_xy - ee))
    obs = env._get_obs()
    while True:
        action = policy.act(obs)
        result = env.step(action)
        obs = result.obs
        if result.terminated or result.truncated:
            break
    return {
        "policy": policy.name,
        "target_xy": env.target_xy.copy(),
        "history": env.history.copy(),
    }


def rollout_all_policies(model_path: str | Path, seed_offset: int = 1400) -> tuple[dict, list[dict]]:
    config = load_config(model_path)
    shared_seed = int(config["seed"]) + seed_offset
    initial_state = make_shared_initial_state(config, shared_seed)
    rollouts = []
    for policy in build_policies(model_path):
        env = Arm2DEnv(config["env"], seed=shared_seed)
        rollouts.append(rollout_policy(env, policy, initial_state))
    return config, rollouts
