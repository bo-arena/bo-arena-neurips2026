"""Class for storing datasets of observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jaxtyping import Float


@dataclass
class Dataset:
    """Base class for datasets.

    Args:
        X: input data.
        objectives: objective values, with one row per input and one column per
            objective.
    """

    X: Float[np.ndarray, "N ..."]
    objectives: Float[np.ndarray, "N n_objectives"]

    def __post_init__(self) -> None:
        """Checks that the shapes of $X$ and objectives are compatible."""
        if self.objectives.ndim != 2:
            msg = f"Objectives must be a 2-dimensional array with shape (N, n_objectives). Got objectives.shape={self.objectives.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

        if self.X.shape[0] != self.objectives.shape[0]:
            msg = f"Inputs, X, and objectives must have the same number of elements. Got X.shape={self.X.shape} and objectives.shape={self.objectives.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

    def __add__(self, other: Dataset) -> Dataset:
        """Combine two datasets.

        Order of left and right dataset is preserved.

        Args:
            other: Dataset to concatenate with this dataset.

        Returns:
            Concatenated dataset.

        Raises:
            ValueError: If the datasets have a different number of objectives.
        """
        if self.n_objectives != other.n_objectives:
            msg = f"Cannot combine datasets with {self.n_objectives} and {other.n_objectives} objectives."  # ruff: ignore[line-too-long]
            raise ValueError(msg)
        X = np.concatenate((self.X, other.X))
        objectives = np.concatenate((self.objectives, other.objectives))
        return Dataset(X=X, objectives=objectives)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]

    @property
    def n_objectives(self) -> int:
        r"""Number of objectives."""
        return self.objectives.shape[1]
