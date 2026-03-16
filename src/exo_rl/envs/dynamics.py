from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def forward_kinematics(q: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
    q1, q2 = q
    l1, l2 = link_lengths
    elbow = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    wrist = elbow + np.array([l2 * np.cos(q1 + q2), l2 * np.sin(q1 + q2)])
    return wrist


def inverse_kinematics(target: np.ndarray, link_lengths: np.ndarray) -> np.ndarray:
    x, y = target
    l1, l2 = link_lengths
    radius_sq = x * x + y * y
    cos_q2 = (radius_sq - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    q2 = math.acos(cos_q2)
    k1 = l1 + l2 * math.cos(q2)
    k2 = l2 * math.sin(q2)
    q1 = math.atan2(y, x) - math.atan2(k2, k1)
    return wrap_angle(np.array([q1, q2], dtype=np.float64))


def integrate_dynamics(
    q: np.ndarray,
    dq: np.ndarray,
    human_torque: np.ndarray,
    assist_torque: np.ndarray,
    inertia: np.ndarray,
    damping: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_torque = human_torque + assist_torque - damping * dq
    ddq = total_torque / inertia
    dq_next = dq + ddq * dt
    q_next = wrap_angle(q + dq_next * dt)
    return q_next, dq_next, ddq

