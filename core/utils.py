from __future__ import annotations

import os

import torch


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_reproducibility(seed: int) -> None:
    # Reproducibility controls; may reduce throughput on some GPUs.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
