"""Base class for problems."""

from abc import abstractmethod

import numpy as np
from bencherscaffold.client import BencherClient
from bencherscaffold.protoclasses.bencher_pb2 import Value
from jaxtyping import Float, UInt32

from bo_arena.problems.metadata import PROBLEM_METADATA, ProblemMetadata
from bo_arena.utils.dataset import Dataset


class Problem:
    """Base class for problems.

    Note that problems should be written as maximization problems (following the same
    convention as in BoTorch).

    Args:
        name: Name of the problem.
        metadata: Problem metadata.
    """

    def __init__(self, name: str, metadata: ProblemMetadata) -> None:
        self.name = name
        self.metadata = metadata

    def __str__(self) -> str:
        """Return a string representation of the problem."""
        return self.name

    @abstractmethod
    def evaluate_points(
        self,
        points: Float[np.ndarray, "N D"],
        seeds: UInt32[np.ndarray, " N"] | None = None,
    ) -> Dataset:
        """Evaluate the problem at the given points.

        Args:
            points: Points to evaluate.
            seeds: (Optional) seeds, one per point, for seeding stochastic problems.

        Returns:
            Dataset containing the evaluated points and their objective values.
        """
        msg = "This method should be implemented by subclasses."
        raise NotImplementedError(msg)


class BencherProblem(Problem):
    """Problem implementation using Bencher.

    Args:
        name: Name of the problem.
        metadata: Problem metadata. If None, will be looked up in PROBLEM_METADATA
          using the name.
        address: Address of the Bencher server. Can be a hostname (e.g. "127.0.0.1")
          or a Unix socket path (e.g. "unix:///path/to/bencher.sock").
        port: Port of the Bencher server. Ignored when address is a Unix socket.
    """

    def __init__(
        self,
        name: str,
        metadata: ProblemMetadata | None = None,
        address: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        if metadata is None:
            metadata = PROBLEM_METADATA.get(name)
            if metadata is None:
                msg = f"Metadata not found for problem '{name}'."
                raise ValueError(msg)
        super().__init__(name, metadata)
        self.bencher_client = BencherClient(address=address, port=port)

    def evaluate_points(
        self,
        points: Float[np.ndarray, "N D"],
        seeds: UInt32[np.ndarray, " N"] | None = None,
    ) -> Dataset:
        """Evaluate the problem at the given points.

        Bencher problems are minimization problems, so the objective values are
        negated to follow the maximization convention.

        Args:
            points: Points to evaluate.
            seeds: (Optional) seeds, one per point, for seeding stochastic problems.

        Returns:
            Dataset containing the evaluated points and their objective values.
        """
        objectives = []
        for i, point in enumerate(points):
            bencher_point = [
                Value(type=self.metadata.input_type, value=float(d)) for d in point
            ]
            result = self.bencher_client.evaluate_point(
                benchmark_name=self.name,
                point=bencher_point,
                random_seed=int(seeds[i]) if seeds is not None else None,
            )
            # Turn minimization into maximization for BoTorch.
            objectives.append([-objective.value for objective in result.objectives])
        return Dataset(X=points, objectives=np.array(objectives))
