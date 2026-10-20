"""Random agent."""

import numpy as np
from jaxtyping import Float

from bo_arena.utils.dataset import Dataset

from .ask_tell_agent import AskTellAgent


class Random(AskTellAgent):
    """Random agent that samples points uniformly at random from the input space."""

    def __init__(self, input_dim: int, **kwargs: object) -> None:  # ruff: ignore[unused-method-argument]
        """Initialize the random agent.

        Args:
            input_dim: Dimensionality of the input space.
            **kwargs: Additional keyword arguments (not used by the random agent).
        """
        self.input_dim = input_dim

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "random"

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "n {self.input_dim}"]:
        """Select next points to evaluate by sampling uniformly at random.

        Note: Assumes that the input space is the unit hypercube [0, 1]^d.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility.

        Returns:
            n points randomly sampled from the input space.
        """
        return np.random.default_rng(seed).uniform(size=(n, self.input_dim))

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument, no-self-use]
        """Tell the agent about new observations.

        Note: The random agent does not use observations to inform its sampling
            strategy, so this method is a no-op.

        Args:
            data: Dataset containing new observations.
            seed: Random seed for reproducibility.

        Returns:
            None
        """
        return

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument, no-self-use]
        """Provide initial observations to the agent.

        Note: The random agent does not use observations to inform its sampling
            strategy, so this method is a no-op.

        Args:
            data: Dataset containing initial observations.
            seed: Random seed for reproducibility.

        Returns:
            None
        """
        return
