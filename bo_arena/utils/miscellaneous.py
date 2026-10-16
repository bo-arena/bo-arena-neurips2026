"""A collection of miscellaneous utility functions."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all relevant random number generators for reproducibility.

    Args:
        seed: The random seed to use for all generators.
    """
    random.seed(seed)
    np.random.seed(seed)  # ruff: ignore[numpy-legacy-random]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_seed(seed_seq: np.random.SeedSequence) -> int:
    """Generate a random integer seed from a NumPy SeedSequence.

    Args:
        seed_seq: The SeedSequence to generate a random seed from.

    Returns:
        A random integer seed.
    """
    return int(seed_seq.generate_state(1)[0])
