from __future__ import annotations

import numpy as np


class RuleBasedFatiguePolicy:
    name = "rule_based_fatigue"

    def act(self, obs: np.ndarray) -> np.ndarray:
        fatigue = float(obs[10])
        effort = float(obs[11])
        gain = 0.08
        if fatigue > 0.5 or effort > 0.4:
            gain = 0.38
        if fatigue > 0.68 or effort > 0.52:
            gain = 0.55
        return np.full(2, gain, dtype=np.float32)
