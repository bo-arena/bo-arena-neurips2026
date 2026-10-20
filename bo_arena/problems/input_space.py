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
        if np.any(upper_bounds <= lower_bounds):
            msg = "Upper bounds must be strictly greater than lower bounds."
            raise ValueError(msg)
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

    def to_unit(self, X: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "N D"]:
        """Affinely map points from [lower_bounds, upper_bounds] to [0, 1]^D.

        Args:
            X: Points in the input space.

        Returns:
            Points in [0, 1]^D.
        """
        return (X - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def from_unit(self, U: Float[np.ndarray, "N D"]) -> Float[np.ndarray, "N D"]:
        """Affinely map points from [0, 1]^D to [lower_bounds, upper_bounds].

        Args:
            U: Points in [0, 1]^D.

        Returns:
            Points in the input space.
        """
        X = self.lower_bounds + U * (self.upper_bounds - self.lower_bounds)
        # Clip to guard against floating-point error.
        return np.clip(X, self.lower_bounds, self.upper_bounds)
