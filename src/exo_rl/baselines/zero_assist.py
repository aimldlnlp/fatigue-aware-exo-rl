from __future__ import annotations

import numpy as np


class ZeroAssistPolicy:
    name = "zero_assist"

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)

