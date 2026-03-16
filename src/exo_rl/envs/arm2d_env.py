from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exo_rl.envs.dynamics import forward_kinematics, integrate_dynamics, wrap_angle
from exo_rl.envs.fatigue import update_fatigue
from exo_rl.envs.rewards import compute_reward
from exo_rl.envs.user_model import compute_human_torque


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class Arm2DEnv:
    def __init__(self, config: dict, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.dt = float(config["dt"])
        self.episode_steps = int(config["episode_steps"])
        self.link_lengths = np.asarray(config["link_lengths"], dtype=np.float64)
        self.inertia = np.asarray(config["inertia"], dtype=np.float64)
        self.damping = np.asarray(config["damping"], dtype=np.float64)
        self.human_max_torque = np.asarray(config["human_max_torque"], dtype=np.float64)
        self.exo_max_torque = np.asarray(config["exo_max_torque"], dtype=np.float64)
        self.kp = np.asarray(config["kp"], dtype=np.float64)
        self.kd = np.asarray(config["kd"], dtype=np.float64)
        self.assist_kp = np.asarray(config["assist_kp"], dtype=np.float64)
        self.assist_kd = np.asarray(config["assist_kd"], dtype=np.float64)
        self.action_smoothing = float(config["action_smoothing"])
        self.success_threshold = float(config["success_threshold"])
        self.success_hold_steps = int(config["success_hold_steps"])
        self.fatigue_gain = float(config["fatigue_gain"])
        self.fatigue_recovery = float(config["fatigue_recovery"])
        self.target_radius_range = config["target_radius_range"]
        self.initial_fatigue_range = config["initial_fatigue_range"]
        self.max_reach = float(np.sum(self.link_lengths))
        self.obs_size = 15
        self.action_size = 2
        self.history: list[dict] = []
        self.reset()

    def _sample_target(self) -> np.ndarray:
        r_low, r_high = self.target_radius_range
        radius = self.rng.uniform(r_low, r_high)
        theta = self.rng.uniform(-0.95 * np.pi, 0.95 * np.pi)
        return np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        ee = forward_kinematics(self.q, self.link_lengths)
        error = self.target_xy - ee
        obs = np.array(
            [
                np.sin(self.q[0]),
                np.cos(self.q[0]),
                np.sin(self.q[1]),
                np.cos(self.q[1]),
                self.dq[0],
                self.dq[1],
                self.target_xy[0] / self.max_reach,
                self.target_xy[1] / self.max_reach,
                error[0] / self.max_reach,
                error[1] / self.max_reach,
                self.fatigue,
                self.last_human_effort,
                self.last_assist_ratio,
                self.prev_action[0],
                self.prev_action[1],
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self) -> tuple[np.ndarray, dict]:
        self.steps = 0
        self.q = self.rng.uniform(low=-0.3, high=0.3, size=2)
        self.dq = np.zeros(2, dtype=np.float64)
        self.fatigue = float(self.rng.uniform(*self.initial_fatigue_range))
        self.target_xy = self._sample_target()
        self.prev_action = np.zeros(2, dtype=np.float64)
        self.last_human_effort = 0.0
        self.last_assist_ratio = 0.0
        self.success_streak = 0
        ee = forward_kinematics(self.q, self.link_lengths)
        self.prev_distance = float(np.linalg.norm(self.target_xy - ee))
        self.history = []
        return self._get_obs(), {"target_xy": self.target_xy.copy()}

    def step(self, action: np.ndarray) -> StepResult:
        self.steps += 1
        clipped = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        smoothed_action = (
            self.action_smoothing * self.prev_action + (1.0 - self.action_smoothing) * clipped
        )
        human_torque, q_target, human_effort = compute_human_torque(
            q=self.q,
            dq=self.dq,
            target_xy=self.target_xy,
            link_lengths=self.link_lengths,
            kp=self.kp,
            kd=self.kd,
            max_torque=self.human_max_torque,
            fatigue=self.fatigue,
        )
        q_error = wrap_angle(q_target - self.q)
        desired_assist = self.assist_kp * q_error - self.assist_kd * self.dq
        desired_assist = np.clip(desired_assist, -self.exo_max_torque, self.exo_max_torque)
        assist_torque = smoothed_action * desired_assist
        self.q, self.dq, ddq = integrate_dynamics(
            q=self.q,
            dq=self.dq,
            human_torque=human_torque,
            assist_torque=assist_torque,
            inertia=self.inertia,
            damping=self.damping,
            dt=self.dt,
        )
        ee = forward_kinematics(self.q, self.link_lengths)
        distance = float(np.linalg.norm(self.target_xy - ee))
        progress = self.prev_distance - distance
        action_delta = float(np.linalg.norm(smoothed_action - self.prev_action))
        assist_ratio = float(np.mean(np.abs(assist_torque) / np.maximum(self.exo_max_torque, 1e-6)))
        self.fatigue = update_fatigue(
            fatigue=self.fatigue,
            effort=human_effort,
            gain=self.fatigue_gain,
            recovery=self.fatigue_recovery,
        )
        success = distance < self.success_threshold
        self.success_streak = self.success_streak + 1 if success else 0
        terminated = self.success_streak >= self.success_hold_steps
        reward, success = compute_reward(
            progress=progress,
            distance=distance,
            success_threshold=self.success_threshold,
            fatigue=self.fatigue,
            human_effort=human_effort,
            assist_ratio=assist_ratio,
            action_delta=action_delta,
            success_streak=self.success_streak,
            terminated=terminated,
            config=self.config,
        )
        truncated = self.steps >= self.episode_steps
        info = {
            "distance": distance,
            "progress": progress,
            "end_effector_xy": ee.copy(),
            "target_xy": self.target_xy.copy(),
            "q_target": q_target.copy(),
            "assist_gain": smoothed_action.copy(),
            "human_torque": human_torque.copy(),
            "assist_torque": assist_torque.copy(),
            "human_effort": human_effort,
            "assist_ratio": assist_ratio,
            "fatigue": self.fatigue,
            "success": success,
            "ddq": ddq.copy(),
        }
        self.history.append(info)
        self.prev_action = smoothed_action
        self.prev_distance = distance
        self.last_human_effort = human_effort
        self.last_assist_ratio = assist_ratio
        return StepResult(self._get_obs(), reward, terminated, truncated, info)
