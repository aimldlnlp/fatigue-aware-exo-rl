from __future__ import annotations

import numpy as np


def compute_reward(
    progress: float,
    distance: float,
    success_threshold: float,
    fatigue: float,
    human_effort: float,
    assist_ratio: float,
    action_delta: float,
    success_streak: int,
    terminated: bool,
    config: dict,
) -> tuple[float, bool]:
    success = distance < success_threshold
    reward = config["progress_scale"] * progress
    reward -= config["distance_scale"] * distance
    reward -= config["fatigue_penalty_scale"] * fatigue
    reward -= config["human_effort_scale"] * human_effort
    reward -= config["assist_penalty_scale"] * assist_ratio
    reward -= config["smoothness_scale"] * action_delta
    if success:
        reward += config["success_bonus"] + 0.2 * min(success_streak, 5)
    if terminated and success:
        reward += config["terminal_success_bonus"]
    return float(reward), success


def summarize_episode(history: list[dict]) -> dict[str, float]:
    distances = np.array([entry["distance"] for entry in history], dtype=np.float64)
    human_efforts = np.array([entry["human_effort"] for entry in history], dtype=np.float64)
    assist_ratios = np.array([entry["assist_ratio"] for entry in history], dtype=np.float64)
    fatigues = np.array([entry["fatigue"] for entry in history], dtype=np.float64)
    successes = np.array([entry["success"] for entry in history], dtype=np.float64)
    return {
        "avg_distance": float(np.mean(distances)),
        "avg_human_effort": float(np.mean(human_efforts)),
        "avg_assist_ratio": float(np.mean(assist_ratios)),
        "final_fatigue": float(fatigues[-1]),
        "success_rate": float(np.max(successes)),
        "success_fraction": float(np.mean(successes)),
    }
