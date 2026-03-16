import random

import numpy as np
import torch

from exo_rl.utils.device import cuda_available


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available():
        torch.cuda.manual_seed_all(seed)
