"""Tests for randomness."""

import jax
import numpy as np
import pytest
from jaxtyping import Float, UInt32

from bo_arena.agents.random import Random
from bo_arena.agents.registry import AGENTS
from bo_arena.agents.saasbo import SAASBO
from bo_arena.arena import Arena, RunResult
from bo_arena.problems.synthetic_problems import Hartmann6D
from bo_arena.utils.dataset import Dataset
from bo_arena.utils.miscellaneous import generate_seed

SKIP_REASONS = {
    "bounce": "intricate setup is incompatible with this harness",
    "bounce_log_ei": "intricate setup is incompatible with this harness",
    "bounce_dsp": "intricate setup is incompatible with this harness",
    "bounce_log_ei_dsp": "intricate setup is incompatible with this harness",
}
AGENT_PARAMS = [
    pytest.param(name, marks=pytest.mark.skip(reason=SKIP_REASONS[name]))
    if name in SKIP_REASONS
    else name
    for name in AGENTS
]


@pytest.mark.parametrize("agent_name", AGENT_PARAMS)
def test_seed_determinism(agent_name: str) -> None:
    """Running the Arena twice with the same seed should give identical results."""
    problem = Hartmann6D(embed_dim=100)
    agent_cls = AGENTS[agent_name]

    def run_arena(seed: int) -> RunResult:
        agent = agent_cls(input_dim=problem.metadata.input_dim, n_total_evaluations=12)
        arena = Arena(
            problem=problem,
            agent=agent,
            seed=seed,
            batch_size=1,
            n_initial_points=10,
            n_total_evaluations=12,
        )
        return arena.run()

    result_1 = run_arena(seed=42)
    result_2 = run_arena(seed=42)

    np.testing.assert_array_equal(result_1.observed_data.X, result_2.observed_data.X)
    np.testing.assert_array_equal(
        result_1.observed_data.objectives, result_2.observed_data.objectives
    )


@pytest.mark.parametrize("agent_name", AGENT_PARAMS)
def test_different_seeds_give_different_results(agent_name: str) -> None:
    """Running the Arena with different seeds should give different results."""
    problem = Hartmann6D(embed_dim=100)
    agent_cls = AGENTS[agent_name]

    def run_arena(seed: int) -> RunResult:
        agent = agent_cls(input_dim=problem.metadata.input_dim, n_total_evaluations=12)
        arena = Arena(
            problem=problem,
            agent=agent,
            seed=seed,
            batch_size=1,
            n_initial_points=10,
            n_total_evaluations=12,
        )
        return arena.run()

    result_1 = run_arena(seed=42)
    result_2 = run_arena(seed=123)

    assert not np.array_equal(result_1.observed_data.X, result_2.observed_data.X)


def test_observed_data_is_float64() -> None:
    """Arena.run() should return float64 arrays for both X and y."""
    problem = Hartmann6D(embed_dim=100)
    agent = Random(input_dim=problem.metadata.input_dim)
    arena = Arena(
        problem=problem,
        agent=agent,
        seed=42,
        batch_size=1,
        n_initial_points=5,
        n_total_evaluations=7,
    )
    result = arena.run()

    assert result.observed_data.X.dtype == np.float64
    assert result.observed_data.objectives.dtype == np.float64


def test_evaluation_seeds_independent_of_batch_size() -> None:
    """The seed for evaluation i should not depend on the batch size."""

    class SeedRecordingHartmann6D(Hartmann6D):
        def __init__(self) -> None:
            super().__init__(embed_dim=100)
            self.recorded_seeds: list[int] = []

        def evaluate_points(
            self,
            points: Float[np.ndarray, "N D"],
            seeds: UInt32[np.ndarray, " N"] | None = None,
        ) -> Dataset:
            assert seeds is not None
            assert seeds.dtype == np.uint32
            self.recorded_seeds.extend(int(s) for s in seeds)
            return super().evaluate_points(points, seeds)

    def run_arena(batch_size: int) -> list[int]:
        problem = SeedRecordingHartmann6D()
        arena = Arena(
            problem=problem,
            agent=Random(input_dim=problem.metadata.input_dim),
            seed=42,
            batch_size=batch_size,
            n_initial_points=5,
            n_total_evaluations=14,
        )
        arena.run()
        return problem.recorded_seeds

    seeds_batch_1 = run_arena(batch_size=1)
    seeds_batch_4 = run_arena(batch_size=4)

    assert len(seeds_batch_1) == 14
    assert len(set(seeds_batch_1)) == 14
    assert seeds_batch_1 == seeds_batch_4


def test_seed_spawning_is_deterministic() -> None:
    """Spawning from a SeedSequence must return the same values every time."""
    for _ in range(5):
        key = np.random.SeedSequence(42)
        seed_a = generate_seed(key.spawn(2)[0])

        key = np.random.SeedSequence(42)
        seed_b = generate_seed(key.spawn(2)[0])

        assert seed_a == seed_b


def test_spawn_is_append_stable() -> None:
    """spawn(n+1)[:n] must produce the same children as spawn(n).

    This is the property that makes the tree-based seeding strategy in Arena
    forward-compatible: adding a new method only increases the spawn count,
    and existing children keep their spawn indices.
    """
    key = np.random.SeedSequence(42)
    a, b = key.spawn(2)

    key = np.random.SeedSequence(42)
    a3, b3, _c3 = key.spawn(3)

    assert generate_seed(a) == generate_seed(a3)
    assert generate_seed(b) == generate_seed(b3)


@pytest.mark.parametrize(
    ("master_seed", "expected_seeds"),
    [
        (0, [4088532484, 3581274545, 476252562]),
        (42, [3142992634, 3908812709, 1436910793]),
        (123456, [419425804, 3878087563, 2422650640]),
    ],
)
def test_seed_spawning_hard_coded_values(
    master_seed: int, expected_seeds: list[int]
) -> None:
    """Guard against SeedSequence.spawn changing across NumPy versions.

    Exercises the same tree structure used by Arena.run(): master key spawns
    init_key and loop_key, init_key spawns two children, loop_key spawns one
    iteration key which spawns one child.
    """
    key = np.random.SeedSequence(master_seed)
    init_key, loop_key = key.spawn(2)
    gen_init_key, tell_init_key = init_key.spawn(2)
    iter_key = loop_key.spawn(1)[0]
    ask_key = iter_key.spawn(1)[0]

    assert generate_seed(gen_init_key) == expected_seeds[0]
    assert generate_seed(tell_init_key) == expected_seeds[1]
    assert generate_seed(ask_key) == expected_seeds[2]


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="needs a CPU-only JAX to trip the guard"
)
def test_saasbo_cuda_without_gpu_jax_raises() -> None:
    """Asking for CUDA while JAX is on the CPU should fail explicitly."""
    with pytest.raises(RuntimeError, match="JAX is running on the CPU"):
        SAASBO(input_dim=4, device="cuda")
