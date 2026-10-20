"""Tests for Dataset."""

import numpy as np
import pytest

from bo_arena.utils.dataset import Dataset


def _dataset(n: int, d: int = 3, m: int = 1, start: int = 0) -> Dataset:
    return Dataset(
        X=np.arange(start, start + n * d, dtype=np.float64).reshape(n, d),
        objectives=np.arange(start, start + n * m, dtype=np.float64).reshape(n, m),
    )


def test_properties() -> None:
    """The n and n_objectives properties should reflect the array shapes."""
    data = _dataset(n=4, m=2)
    assert data.n == 4
    assert data.n_objectives == 2


@pytest.mark.parametrize(
    ("X", "objectives"),
    [
        (np.zeros((4, 3)), np.zeros(4)),  # 1D objectives
        (np.zeros((4, 3)), np.zeros((5, 1))),  # mismatched N
    ],
)
def test_invalid_shapes_raise(X: np.ndarray, objectives: np.ndarray) -> None:
    """Invalid shapes should be rejected on construction."""
    with pytest.raises(ValueError):  # ruff: ignore[pytest-raises-too-broad]
        Dataset(X=X, objectives=objectives)


def test_add_concatenates_in_order() -> None:
    """Adding datasets should concatenate X and objectives, preserving order."""
    left, right = _dataset(n=2), _dataset(n=3, start=100)
    combined = left + right
    np.testing.assert_array_equal(combined.X, np.concatenate((left.X, right.X)))
    np.testing.assert_array_equal(
        combined.objectives, np.concatenate((left.objectives, right.objectives))
    )


def test_add_to_empty() -> None:
    """Adding to an empty dataset (as the Arena does) should return the other data."""
    empty = Dataset(X=np.empty((0, 3)), objectives=np.empty((0, 1)))
    data = _dataset(n=2)
    combined = empty + data
    np.testing.assert_array_equal(combined.X, data.X)
    np.testing.assert_array_equal(combined.objectives, data.objectives)


def test_add_mismatched_objectives_raises() -> None:
    """Datasets with different numbers of objectives cannot be combined."""
    with pytest.raises(ValueError, match="objectives"):
        _ = _dataset(n=2, m=1) + _dataset(n=2, m=2)
