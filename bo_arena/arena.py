"""Arena for running BO experiments."""

import time
from dataclasses import dataclass
from logging import Logger

import numpy as np
from jaxtyping import Float
from wandb import Run

from bo_arena.agents import AskTellAgent
from bo_arena.problems import Problem
from bo_arena.problems.input_space import ContinuousInputSpace
from bo_arena.utils.dataset import Dataset
from bo_arena.utils.miscellaneous import (
    generate_seed,
    seed_everything,
)


@dataclass
class RunResult:
    """Result of an Arena run.

    Args:
        observed_data: The dataset of all observed points.
        ask_times: Wall-clock time (seconds) for each agent.ask() call.
        tell_times: Wall-clock time (seconds) for each agent.tell() call.
        eval_times: Wall-clock time (seconds) for each problem.evaluate_points() call.
    """

    observed_data: Dataset
    ask_times: list[float]
    tell_times: list[float]
    eval_times: list[float]


class Arena:
    """Arena for running BO experiments.

    Note that problems are assumed to be maximization problems, following the same
    convention as in BoTorch.

    Args:
        problem: The problem to run.
        agent: The BO agent to use for optimization.
        seed: Random seed for reproducibility.
        batch_size: Number of points to evaluate in each BO iteration.
        n_initial_points: (Optional) number of initial points to sample - if None, uses
            the problem's default.
        n_total_evaluations: (Optional) number of total evaluations - if None, uses the
            problem's default.
        logger: (Optional) logger for logging information.
        wandb_run: (Optional) WandB run to use for logging.
    """

    def __init__(
        self,
        problem: Problem,
        agent: AskTellAgent,
        seed: int,
        batch_size: int,
        n_initial_points: int | None = None,
        n_total_evaluations: int | None = None,
        logger: Logger | None = None,
        wandb_run: Run | None = None,
    ) -> None:
        self.problem = problem
        self.agent = agent
        self._key = np.random.SeedSequence(seed)
        self._has_run = False

        if n_initial_points is None:
            self.n_initial_points = problem.metadata.default_n_initial_points
        else:
            self.n_initial_points = n_initial_points

        if n_total_evaluations is None:
            self.n_total_evaluations = problem.metadata.default_n_total_evaluations
        else:
            self.n_total_evaluations = n_total_evaluations

        self.batch_size = batch_size
        self.logger = logger
        self.wandb_run = wandb_run
        self.observed_data = Dataset(
            np.empty((0, self.problem.metadata.input_dim), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
        )

    def _evaluate(self, points: Float[np.ndarray, "N D"]) -> Dataset:
        """Evaluate points on the problem, returning float64 data.

        Args:
            points: Points to evaluate.

        Returns:
            Dataset containing the evaluated points and their objective values.

        Raises:
            ValueError: If the problem returns more than one objective, as
                multi-objective problems are not yet supported.
        """
        data = self.problem.evaluate_points(points)
        if data.n_objectives != 1:
            msg = (
                f"Problem '{self.problem}' returned {data.n_objectives} objectives, "
                "but only single-objective problems are currently supported."
            )
            raise ValueError(msg)
        return Dataset(
            X=data.X.astype(np.float64),
            objectives=data.objectives.astype(np.float64),
        )

    def run(self) -> RunResult:  # ruff: ignore[too-many-statements]
        """Run the main BO loop.

        Runs the main Bayesian optimization loop, which consists of:
        1. Sampling initial points from the problem's input space and evaluating them.
        2. Iteratively asking the agent for new points to evaluate, evaluating them on
           the problem, and providing the observations back to the agent.

        If a logger is provided, logs information about each iteration, including
        the number of evaluations, times taken for each step, and the best observation
        so far. If a WandB run is provided, logs detailed metrics for each evaluation.

        Seed derivation uses a tree structure rather than a chain. The master key is
        split into an init branch and a loop branch. Since ``SeedSequence.spawn(n+1)[:n]
        `` produces the same children as ``spawn(n)``, a new method can be added by
        increasing the spawn count without changing any previously derived seeds.

        Returns:
            A RunResult containing the observed data and per-iteration timing info.

        Raises:
            RuntimeError: If the method is called more than once on the same instance.
                This is to prevent non-deterministic behavior due to the internal state
                of the Arena (e.g. seed counter, agent) being modified across runs.
        """
        if self._has_run:
            msg = (
                "Arena.run() has already been called. Create a new Arena instance for "
                "each run."
            )
            raise RuntimeError(msg)
        self._has_run = True

        # Split master key into independent branches for init and loop phases.
        # Init-phase keys are spawned upfront so that each method gets a fixed
        # spawn index. Adding a new method means increasing the spawn count —
        # spawn(n+1)[:n] produces the same children as spawn(n).
        init_key, loop_key = self._key.spawn(2)
        gen_init_key, tell_init_key = init_key.spawn(2)

        # Sample initial points from the problem's input space and evaluate them.
        # Agents may override generate_initial_points to provide custom initialization.
        seed = generate_seed(gen_init_key)
        seed_everything(seed)
        custom_points = self.agent.generate_initial_points(self.n_initial_points, seed)
        if custom_points is not None:
            initial_points = custom_points
        else:
            initial_points = self.problem.metadata.input_space.sample(
                self.n_initial_points,
                seed,
            )
        initial_points = initial_points.astype(np.float64)
        input_space = self.problem.metadata.input_space
        if isinstance(input_space, ContinuousInputSpace):
            initial_points = np.clip(
                initial_points,
                input_space.lower_bounds,
                input_space.upper_bounds,
            )
        initial_data = self._evaluate(initial_points)
        self.observed_data = self.observed_data + initial_data
        initial_observations = initial_data.objectives[:, 0]
        seed = generate_seed(tell_init_key)
        seed_everything(seed)
        self.agent.tell_initial_data(self.observed_data, seed)

        # Log initial points info.
        if self.logger is not None:
            self.logger.info(
                f"Initial points evaluated: {self.n_initial_points}\n"
                f"Initial best observation: {float(np.max(initial_observations)):.4f}"
            )

        if self.wandb_run is not None:
            for iter in range(self.n_initial_points):
                observation_val = float(initial_observations[iter])
                best_val_so_far = float(np.max(initial_observations[: iter + 1]))
                is_best_observation = observation_val == best_val_so_far
                self.wandb_run.log({
                    "batch_index": 0,
                    "iteration": iter,
                    "observation": observation_val,
                    "best_observation": best_val_so_far,
                    "is_best_observation": is_best_observation,
                })

        # Run core BO loop. Uses a while loop rather than a fixed iteration count
        # because agents may return fewer points than requested (e.g. during
        # initialization phases).
        ask_times: list[float] = []
        tell_times: list[float] = []
        eval_times: list[float] = []
        batch_idx = 0
        while self.observed_data.n < self.n_total_evaluations:
            remaining = self.n_total_evaluations - self.observed_data.n
            n_ask = min(self.batch_size, remaining)

            # Each iteration gets its own branch so adding iterations can't
            # shift other iterations' seeds. Keys within the iteration are
            # spawned upfront so new per-iteration methods can be added by
            # increasing the spawn count.
            iter_key = loop_key.spawn(1)[0]
            ask_key, tell_key = iter_key.spawn(2)

            # Ask the agent for new points to evaluate.
            seed = generate_seed(ask_key)
            seed_everything(seed)
            start = time.perf_counter()
            new_points = self.agent.ask(n_ask, seed)
            ask_time = time.perf_counter() - start
            ask_times.append(ask_time)
            new_points = new_points.astype(np.float64)
            if isinstance(input_space, ContinuousInputSpace):
                new_points = np.clip(
                    new_points,
                    input_space.lower_bounds,
                    input_space.upper_bounds,
                )

            # Evaluate the problem at the new points.
            start = time.perf_counter()
            new_points_dataset = self._evaluate(new_points)
            eval_time = time.perf_counter() - start
            eval_times.append(eval_time)

            # Update the observed data with the new points and observations.
            self.observed_data = self.observed_data + new_points_dataset

            # Provide the new observations to the agent.
            seed = generate_seed(tell_key)
            seed_everything(seed)
            start = time.perf_counter()
            self.agent.tell(new_points_dataset, seed)
            tell_time = time.perf_counter() - start
            tell_times.append(tell_time)

            # Log iteration info (note, only logs info about the entire batch, not
            # individual points).
            actual_batch_size = new_points.shape[0]
            observations = self.observed_data.objectives[:, 0]
            if self.logger is not None:
                new_best_in_batch = np.max(observations[-actual_batch_size:]) > np.max(
                    observations[:-actual_batch_size]
                )
                self.logger.info(
                    f"BO loop iteration {batch_idx + 1}:\n"
                    f"  Total evaluations: {self.observed_data.n}\n"
                    f"  Ask time: {ask_time:.4f}s\n"
                    f"  Eval time: {eval_time:.4f}s\n"
                    f"  Tell time: {tell_time:.4f}s\n"
                    f"  Best observation: {float(np.max(observations)):.4f}\n"
                    f"  New best observation in current batch: {new_best_in_batch}"
                )

            # Log detailed metrics to WandB for each point in the batch.
            if self.wandb_run is not None:
                batch_start = self.observed_data.n - actual_batch_size
                for iter in range(batch_start, self.observed_data.n):
                    observation_val = float(observations[iter])
                    best_val_so_far = float(np.max(observations[: iter + 1]))
                    is_best_observation = observation_val == best_val_so_far
                    self.wandb_run.log({
                        "batch_index": batch_idx + 1,
                        "iteration": iter,
                        "observation": observation_val,
                        "best_observation": best_val_so_far,
                        "is_best_observation": is_best_observation,
                        "ask_time": ask_time,
                        "eval_time": eval_time,
                        "tell_time": tell_time,
                    })

            batch_idx += 1

        return RunResult(
            observed_data=self.observed_data,
            ask_times=ask_times,
            tell_times=tell_times,
            eval_times=eval_times,
        )
