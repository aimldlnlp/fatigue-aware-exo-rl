from __future__ import annotations

import numpy as np


def update_fatigue(
    fatigue: float,
    effort: float,
    gain: float,
    recovery: float,
) -> float:
    fatigue_delta = gain * effort - recovery * max(0.0, 1.0 - effort)
    return float(np.clip(fatigue + fatigue_delta, 0.0, 1.0))


def capacity_scale(fatigue: float) -> float:
    return float(np.clip((1.0 - fatigue) ** 1.4, 0.15, 1.0))

