from __future__ import annotations

import numpy as np


def aggregate_episode_infos(infos: list[dict]) -> dict[str, float]:
    distances = np.array([info["distance"] for info in infos], dtype=np.float64)
    efforts = np.array([info["human_effort"] for info in infos], dtype=np.float64)
    assist = np.array([info["assist_ratio"] for info in infos], dtype=np.float64)
    fatigue = np.array([info["fatigue"] for info in infos], dtype=np.float64)
    success = np.array([info["success"] for info in infos], dtype=np.float64)
    return {
        "success_rate": float(np.max(success)),
        "success_fraction": float(np.mean(success)),
        "avg_distance": float(np.mean(distances)),
        "avg_human_effort": float(np.mean(efforts)),
        "avg_assist_ratio": float(np.mean(assist)),
        "final_fatigue": float(fatigue[-1]),
        "time_to_target": float(np.argmax(success > 0.0) + 1 if np.any(success > 0.0) else len(success)),
    }
