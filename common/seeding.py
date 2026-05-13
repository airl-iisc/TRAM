"""Deterministic-training helpers.

Mask R-CNN is non-deterministic by default — the RPN sampler, ROI Align, and
data-loader workers all introduce randomness. Seeding everything below makes
the same seed reproduce the same run on the same hardware.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_all(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch (CPU + CUDA + cuDNN).

    With `deterministic=True`, cuDNN picks deterministic algorithms (slower
    but reproducible). Set environment variables before model creation so
    that any cudnn calls inside fresh modules also pick up the setting.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeder — pass as `worker_init_fn` to DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
