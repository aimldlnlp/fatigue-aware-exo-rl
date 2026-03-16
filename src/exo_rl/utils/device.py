from __future__ import annotations

import warnings

import torch


def cuda_available() -> bool:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization: .*", category=UserWarning)
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False


def resolve_device(config_value: str) -> torch.device:
    if config_value == "auto":
        return torch.device("cuda" if cuda_available() else "cpu")
    return torch.device(config_value)
