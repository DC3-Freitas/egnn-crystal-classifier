"""
Utility functions shared across the entire package.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure
    reproducible results.

    Args:
        seed: Seed value to apply across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
