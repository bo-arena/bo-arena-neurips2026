"""Problem metadata."""

from dataclasses import dataclass

import numpy as np
from bencherscaffold.protoclasses.bencher_pb2 import ValueType

from bo_arena.problems.input_space import ContinuousInputSpace, InputSpace


@dataclass(frozen=True)  # frozen=True makes them immutable
class ProblemMetadata:
    """Metadata that problems must provide.

    Args:
        input_dim: Dimensionality of the problem's input space.
        input_type: Type of the input variables, as a protobuf `ValueType` enum
            value (e.g. `ValueType.CONTINUOUS`).
        default_n_initial_points: Default number of initial points to evaluate
            before the optimization loop starts.
        default_n_total_evaluations: Default total evaluation budget, including
            the initial points.
        input_space: The problem's input space.
        stochastic: Whether the (unseeded) problem is stochastic, i.e. it can
            return different values when evaluated at the same point.
    """

    input_dim: int
    input_type: int  # protobuf ValueType enum (not a PEP-compliant type)
    default_n_initial_points: int
    default_n_total_evaluations: int
    input_space: InputSpace
    stochastic: bool = False


PROBLEM_METADATA: dict[str, ProblemMetadata] = {
    "lunarlander": ProblemMetadata(
        input_dim=12,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=500,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(12),
            upper_bounds=np.ones(12),
        ),
        stochastic=True,
    ),
    "robotpushing": ProblemMetadata(
        input_dim=14,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=500,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(14),
            upper_bounds=np.ones(14),
        ),
        stochastic=True,
    ),
    "mujoco-swimmer": ProblemMetadata(
        input_dim=16,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=500,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(16),
            upper_bounds=np.ones(16),
        ),
        stochastic=True,
    ),
    "mujoco-hopper": ProblemMetadata(
        input_dim=33,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=500,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(33),
            upper_bounds=np.ones(33),
        ),
        stochastic=True,
    ),
    "rover": ProblemMetadata(
        input_dim=60,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(60),
            upper_bounds=np.ones(60),
        ),
        stochastic=True,
    ),
    "mujoco-walker": ProblemMetadata(
        input_dim=102,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=2000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(102),
            upper_bounds=np.ones(102),
        ),
        stochastic=True,
    ),
    "mujoco-halfcheetah": ProblemMetadata(
        input_dim=102,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=2000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(102),
            upper_bounds=np.ones(102),
        ),
        stochastic=True,
    ),
    "mopta08": ProblemMetadata(
        input_dim=124,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(124),
            upper_bounds=np.ones(124),
        ),
    ),
    "lasso-dna": ProblemMetadata(
        input_dim=180,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(180),
            upper_bounds=np.ones(180),
        ),
    ),
    "svm": ProblemMetadata(
        input_dim=388,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(388),
            upper_bounds=np.ones(388),
        ),
    ),
    "mujoco-ant": ProblemMetadata(
        input_dim=888,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(888),
            upper_bounds=np.ones(888),
        ),
        stochastic=True,
    ),
    "mujoco-humanoid": ProblemMetadata(
        input_dim=6392,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(6392),
            upper_bounds=np.ones(6392),
        ),
        stochastic=True,
    ),
    "bbob-attractivesector": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "bbob-ellipsoidrotated": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "bbob-sharpridge": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "bbob-rastriginrotated": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "bbob-griewankrosenbrock": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "bbob-schwefel": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
}


SYNTHETIC_PROBLEM_METADATA: dict[str, ProblemMetadata] = {
    "hartmann6d_embed_dim_100": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=20,
        default_n_total_evaluations=200,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "hartmann6d_embed_dim_300": ProblemMetadata(
        input_dim=300,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=20,
        default_n_total_evaluations=500,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(300),
            upper_bounds=np.ones(300),
        ),
    ),
    "ackley_dim_100": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "michalewicz_dim_100": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
    "styblinski_tang_dim_100": ProblemMetadata(
        input_dim=100,
        input_type=ValueType.CONTINUOUS,
        default_n_initial_points=30,
        default_n_total_evaluations=1000,
        input_space=ContinuousInputSpace(
            lower_bounds=np.zeros(100),
            upper_bounds=np.ones(100),
        ),
    ),
}
