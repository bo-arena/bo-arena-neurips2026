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
        y: output data.
    """

    X: Float[np.ndarray, "N ..."]
    y: Float[np.ndarray, " N"]

    def __post_init__(self) -> None:
        """Checks that the shapes of $X$ and $y$ are compatible."""
        if self.X.shape[0] != self.y.shape[0]:
            msg = f"Inputs, X, and outputs, y, must have the same number of elements. Got X.shape={self.X.shape} and y.shape={self.y.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

        if self.y.ndim != 1:
            msg = f"Outputs, y, must be a 1-dimensional array with shape (N,). Got y.shape={self.y.shape}."  # ruff: ignore[line-too-long]
            raise ValueError(msg)

    def __add__(self, other: Dataset) -> Dataset:
        """Combine two datasets.

        Order of left and right dataset is preserved.

        Args:
            other: Dataset to concatenate with this dataset.

        Returns:
            Concatenated dataset.
        """
        X = np.concatenate((self.X, other.X))
        y = np.concatenate((self.y, other.y))
        return Dataset(X=X, y=y)

    @property
    def n(self) -> int:
        r"""Number of observations."""
        return self.X.shape[0]
