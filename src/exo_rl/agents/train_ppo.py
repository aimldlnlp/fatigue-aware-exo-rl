from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import optim

from exo_rl.agents.policy import ActorCritic
from exo_rl.envs.arm2d_env import Arm2DEnv
from exo_rl.envs.rewards import summarize_episode
from exo_rl.utils.device import resolve_device
from exo_rl.utils.io import ensure_dir, load_json, save_json
from exo_rl.utils.logging import MetricTracker
from exo_rl.utils.seeding import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outputs-dir", type=str, default=None)
    return parser.parse_args()


def make_envs(env_config: dict, num_envs: int, seed: int) -> list[Arm2DEnv]:
    return [Arm2DEnv(env_config, seed=seed + idx) for idx in range(num_envs)]


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def main() -> None:
    args = parse_args()
    config = deepcopy(load_json(args.config))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.outputs_dir is not None:
        config["outputs_dir"] = args.outputs_dir
    set_seed(int(config["seed"]))
    device = resolve_device(config["device"])
    outputs_dir = ensure_dir(config["outputs_dir"])
    models_dir = ensure_dir(outputs_dir / "models")
    logs_dir = ensure_dir(outputs_dir / "logs")
    figures_dir = ensure_dir(outputs_dir / "figures")
    ensure_dir(outputs_dir / "videos")
    save_json(outputs_dir / "resolved_config.json", config)

    train_cfg = config["train"]
    envs = make_envs(config["env"], int(train_cfg["num_envs"]), int(config["seed"]))
    obs, _ = zip(*(env.reset() for env in envs))
    obs_array = np.stack(obs)
    obs_dim = envs[0].obs_size
    act_dim = envs[0].action_size

    agent = ActorCritic(obs_dim, act_dim, train_cfg["hidden_sizes"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(train_cfg["learning_rate"]))
    tracker = MetricTracker()
    episode_summaries: list[dict] = []
    best_success = float("-inf")

    steps_per_env = int(train_cfg["steps_per_env"])
    num_envs = int(train_cfg["num_envs"])
    batch_size = steps_per_env * num_envs
    minibatch_size = batch_size // int(train_cfg["minibatches"])
    gamma = float(train_cfg["gamma"])
    gae_lambda = float(train_cfg["gae_lambda"])

    for update in range(1, int(train_cfg["total_updates"]) + 1):
        obs_buffer = torch.zeros((steps_per_env, num_envs, obs_dim), dtype=torch.float32, device=device)
        act_buffer = torch.zeros((steps_per_env, num_envs, act_dim), dtype=torch.float32, device=device)
        logp_buffer = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=device)
        rew_buffer = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=device)
        done_buffer = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=device)
        val_buffer = torch.zeros((steps_per_env, num_envs), dtype=torch.float32, device=device)

        for step in range(steps_per_env):
            obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=device)
            with torch.no_grad():
                action_tensor, log_prob, _, value = agent.sample_action(obs_tensor)
            obs_buffer[step] = obs_tensor
            act_buffer[step] = action_tensor
            logp_buffer[step] = log_prob
            val_buffer[step] = value

            next_obs = []
            dones = []
            rewards = []
            for env_idx, env in enumerate(envs):
                result = env.step(action_tensor[env_idx].cpu().numpy())
                next_obs.append(result.obs)
                rewards.append(result.reward)
                dones.append(float(result.terminated or result.truncated))
                if result.truncated or result.terminated:
                    summary = summarize_episode(env.history)
                    summary["episode_length"] = len(env.history)
                    episode_summaries.append(summary)
                    reset_obs, _ = env.reset()
                    next_obs[-1] = reset_obs
            obs_array = np.stack(next_obs)
            rew_buffer[step] = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            done_buffer[step] = torch.as_tensor(dones, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_value = agent.value(torch.as_tensor(obs_array, dtype=torch.float32, device=device))

        advantages, returns = compute_gae(
            rewards=rew_buffer,
            dones=done_buffer,
            values=val_buffer,
            next_value=next_value,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        flat_obs = obs_buffer.reshape(batch_size, obs_dim)
        flat_act = act_buffer.reshape(batch_size, act_dim)
        flat_logp = logp_buffer.reshape(batch_size)
        flat_adv = advantages.reshape(batch_size)
        flat_ret = returns.reshape(batch_size)
        flat_val = val_buffer.reshape(batch_size)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std(unbiased=False) + 1e-8)
        indices = np.arange(batch_size)
        clip_coef = float(train_cfg["clip_coef"])
        vf_coef = float(train_cfg["vf_coef"])
        entropy_coef = float(train_cfg["entropy_coef"])

        for _ in range(int(train_cfg["update_epochs"])):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_obs = flat_obs[mb_idx]
                mb_act = flat_act[mb_idx]
                mb_old_logp = flat_logp[mb_idx]
                mb_adv = flat_adv[mb_idx]
                mb_ret = flat_ret[mb_idx]
                mb_old_val = flat_val[mb_idx]

                new_logp, entropy, new_value = agent.evaluate_actions(mb_obs, mb_act)
                log_ratio = new_logp - mb_old_logp
                ratio = log_ratio.exp()

                pg_loss_1 = -mb_adv * ratio
                pg_loss_2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(pg_loss_1, pg_loss_2).mean()

                value_clipped = mb_old_val + torch.clamp(new_value - mb_old_val, -clip_coef, clip_coef)
                value_loss_unclipped = (new_value - mb_ret).pow(2)
                value_loss_clipped = (value_clipped - mb_ret).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                entropy_loss = entropy.mean()

                loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()

        recent = episode_summaries[-max(1, num_envs * 2) :]
        if recent:
            avg_reward = float(torch.mean(rew_buffer).item())
            avg_distance = float(np.mean([entry["avg_distance"] for entry in recent]))
            avg_effort = float(np.mean([entry["avg_human_effort"] for entry in recent]))
            avg_assist = float(np.mean([entry["avg_assist_ratio"] for entry in recent]))
            avg_success = float(np.mean([entry["success_rate"] for entry in recent]))
            avg_fatigue = float(np.mean([entry["final_fatigue"] for entry in recent]))
            tracker.add(
                update=update,
                reward=avg_reward,
                distance=avg_distance,
                human_effort=avg_effort,
                assist_ratio=avg_assist,
                success_rate=avg_success,
                final_fatigue=avg_fatigue,
            )

        if update % int(train_cfg["log_interval"]) == 0 or update == 1:
            print(
                f"update={update:03d} reward={tracker.mean('reward', 1):+.3f} "
                f"success={tracker.mean('success_rate', 1):.3f} "
                f"distance={tracker.mean('distance', 1):.3f} "
                f"fatigue={tracker.mean('final_fatigue', 1):.3f}"
            )

        if update % int(train_cfg["save_interval"]) == 0 or update == int(train_cfg["total_updates"]):
            checkpoint = {
                "model_state_dict": agent.state_dict(),
                "config": config,
                "update": update,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
            }
            torch.save(checkpoint, models_dir / "latest.pt")
            current_success = tracker.mean("success_rate", 1)
            if current_success >= best_success:
                best_success = current_success
                torch.save(checkpoint, models_dir / "best.pt")

    history_payload = {
        "updates": tracker.values.get("update", []),
        "reward": tracker.values.get("reward", []),
        "distance": tracker.values.get("distance", []),
        "human_effort": tracker.values.get("human_effort", []),
        "assist_ratio": tracker.values.get("assist_ratio", []),
        "success_rate": tracker.values.get("success_rate", []),
        "final_fatigue": tracker.values.get("final_fatigue", []),
    }
    save_json(logs_dir / "training_history.json", history_payload)
    save_json(logs_dir / "episode_summaries.json", {"episodes": episode_summaries})
    print(f"saved model to {models_dir / 'best.pt'}")
    print(f"training history at {logs_dir / 'training_history.json'}")
    print(f"figures directory ready at {figures_dir}")


if __name__ == "__main__":
    main()
