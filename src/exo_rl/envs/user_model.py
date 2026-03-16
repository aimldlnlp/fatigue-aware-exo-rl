from __future__ import annotations

import numpy as np

from exo_rl.envs.dynamics import inverse_kinematics, wrap_angle
from exo_rl.envs.fatigue import capacity_scale


def compute_human_torque(
    q: np.ndarray,
    dq: np.ndarray,
    target_xy: np.ndarray,
    link_lengths: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    max_torque: np.ndarray,
    fatigue: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    q_target = inverse_kinematics(target_xy, link_lengths)
    q_error = wrap_angle(q_target - q)
    raw_torque = kp * q_error - kd * dq
    limit = max_torque * capacity_scale(fatigue)
    human_torque = np.clip(raw_torque, -limit, limit)
    effort = float(np.mean(np.abs(human_torque) / np.maximum(max_torque, 1e-6)))
    return human_torque, q_target, effort

