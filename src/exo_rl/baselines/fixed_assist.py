from __future__ import annotations

import numpy as np


class FixedAssistPolicy:
    name = "fixed_assist"

    def __init__(self, gain: float = 0.16) -> None:
        self.gain = gain

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.full(2, self.gain, dtype=np.float32)
