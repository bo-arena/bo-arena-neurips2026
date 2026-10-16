"""Input space definitions."""

from abc import abstractmethod

import numpy as np
from jaxtyping import Float
from scipy.stats import qmc


class InputSpace:
    """Base class for input spaces."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample(self, n_points: int, seed: int) -> Float[np.ndarray, "N D"]:
        """Sample points from the input space.

        Args:
            n_points: Number of points to sample.
            seed: Random seed for reproducibility.

        Returns:
            Sampled points.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class ContinuousInputSpace(InputSpace):
    """Continuous input space defined by lower and upper bounds."""

    def __init__(
        self,
        lower_bounds: Float[np.ndarray, " D"],
        upper_bounds: Float[np.ndarray, " D"],
    ) -> None:
        super().__init__()
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.input_dim = lower_bounds.shape[0]

    def sample(
        self, n_points: int, seed: int
    ) -> Float[np.ndarray, "N {self.input_dim}"]:
        """Sample points from the continuous input space via Sobol indices.

        Args:
            n_points: Number of points to sample.
            seed: Random seed for reproducibility.

        Returns:
            Sampled points of shape (n_points, input_dim).
        """
        rng = np.random.default_rng(seed)
        sampler = qmc.Sobol(d=self.input_dim, scramble=True, rng=rng)
        unit_samples = sampler.random(n=n_points)
        samples = qmc.scale(unit_samples, self.lower_bounds, self.upper_bounds)
        return samples
