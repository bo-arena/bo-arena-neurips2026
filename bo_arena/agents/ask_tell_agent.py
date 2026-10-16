"""Base class for ask-tell agents."""

from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float

from bo_arena.utils.dataset import Dataset


class AskTellAgent(ABC):
    """Base class for ask-tell agents.

    Ask-tell agents must implement three main methods:
    - ask: called to obtain new points to evaluate.
    - tell: called to provide observations for previously requested points.
    - tell_initial_data: called to provide initial observations to the agent.

    In addition, a string representation method (__str__) is required, which should
    include useful information about the agent for experiment management.

    These methods are used in the Bayesian optimization loop to iteratively
    select points to evaluate and update the agent's internal model with new
    observations.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        raise NotImplementedError

    @abstractmethod
    def ask(self, n: int, seed: int) -> Float[np.ndarray, "M D"]:
        """Return points to evaluate next.

        Note:
            Agents may return fewer than `n` points (e.g. during initialization phases
            where random exploration points are used before fitting a surrogate model).

        Args:
            n: Maximum number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation, with shape (M, D) where
            1 <= M <= n.
        """
        raise NotImplementedError

    @abstractmethod
    def tell(self, data: Dataset, seed: int) -> None:
        """Provide observations for previously requested points.

        Args:
            data: Observations corresponding to the points requested
                in the previous `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        raise NotImplementedError

    @abstractmethod
    def tell_initial_data(self, data: Dataset, seed: int) -> None:
        """Provide initial observations to the agent.

        This method is used to provide the agent with an initial dataset
        before any `ask` calls are made. It is typically called once at the
        start of the optimization process.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        raise NotImplementedError

    def generate_initial_points(  # ruff: ignore[no-self-use]
        self,
        n: int,  # ruff: ignore[unused-method-argument]
        seed: int,  # ruff: ignore[unused-method-argument]
    ) -> Float[np.ndarray, "N D"] | None:
        """Optionally generate custom initial points for evaluation.

        Override this method when the agent needs to control how initial points are
        generated. For the majority of methods, this method should be left
        unimplemented, and the Arena will handle initialization for fair comparison of
        algorithms.

        By default returns None, which signals the Arena to use its standard
        initialization method.

        Args:
            n: Number of initial points to generate.
            seed: Random seed for reproducibility.

        Returns:
            An array of initial points in [0, 1]^D, or None to use the default
            initialization.
        """
        return None
