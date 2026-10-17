"""Vanilla BO agent."""

import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.acqf_opt import get_log_ei_opt_kwargs
from bo_arena.utils.dataset import Dataset


class VanillaBO(AskTellAgent):
    """Hvarfner's Vanilla BO.

    This agent implements the method described in "Vanilla Bayesian Optimization
    Performs Great in High Dimensions" by Hvarfner et al. (2024), which relies on an RBF
    kernel with a dimensionality-scaled prior (DSP) applied to the lengthscales.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the VanillaBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        if acqf_opt_strategy == "default":
            self.opt_kwargs = {
                "raw_samples": 512,
                "num_restarts": 4,
                "retry_on_optimization_warning": False,
                "options": {
                    "nonnegative": False,
                    "sample_around_best": True,
                    "sample_around_best_sigma": 0.1,
                    "maxiter": 300,
                    "batch_limit": 64,
                },
            }
        elif acqf_opt_strategy in {"no_raasp", "botorch_raasp"}:
            self.opt_kwargs = get_log_ei_opt_kwargs(acqf_opt_strategy)
        else:
            msg = f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}"
            raise ValueError(msg)

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None

        # Bounds are always [0,1]^d since the Arena normalizes inputs
        self.bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "vanilla_bo"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model (with DSP) using ExactMarginalLogLikelihood."""
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
        )
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:  # ruff: ignore[unused-method-argument]
        """Select the next points to evaluate using qLogNEI.

        This method optimizes the q-batch Log-Noisy-Expected-Improvement acquisition
        function, matching the original code from Hvarfner et al. (2024).

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Points selected by the agent for evaluation.

        Raises:
            RuntimeError: If the model has not been initialized with
                `tell_initial_data`.
        """
        if self.model is None:
            msg = "Model has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Hvarfner et al. use the MC-based qLogNEI
        acq_func = qLogNoisyExpectedImprovement(
            model=self.model,
            X_baseline=self.train_X,
        )

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()


class VanillaBOYu2026(AskTellAgent):
    """Hvarfner's "Vanilla BO", as implemented by Yu et al. (2026).

    This agent implements the "Vanilla BO" baseline as implemented by Yu et al. (2026)
    in their GIT-BO paper, which can be found in their Github repository:
    https://github.com/rosenyu304/GITBO. Note: This implementation is not identical to
    that of the original paper by Hvarfner et al. (2024), which is implemented above in
    the `VanillaBO` agent.

    Reference:
        Yu, R. T. Y., Picard, C., & Ahmed, F. (2026). GIT-BO: High-dimensional Bayesian
        optimization using tabular foundation models. International Conference on
        Learning Representations (ICLR) 2026.
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the VanillaBOYu2026 agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        if acqf_opt_strategy == "default":
            self.opt_kwargs = {
                "raw_samples": 3,
                "num_restarts": 1,
            }
        else:
            msg = (
                f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}. "
                "Note that this agent only supports 'default' settings, "
                "which match the implementation used by Yu et al. (2026)."
            )
            raise ValueError(msg)

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None

        # Bounds are always [0,1]^d since the Arena normalizes inputs
        self.bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "vanilla_bo_yu_2026"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model (with DSP) using ExactMarginalLogLikelihood."""
        normed_y = self.train_y / (torch.max(self.train_y) - torch.min(self.train_y))
        self.model = SingleTaskGP(
            self.train_X.double(),
            normed_y.double(),
            input_transform=Normalize(d=self.train_X.shape[1]),
            outcome_transform=Standardize(m=1),
            likelihood=None,
        )
        mll = ExactMarginalLogLikelihood(
            model=self.model, likelihood=self.model.likelihood
        )
        fit_gpytorch_mll(mll)

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:  # ruff: ignore[unused-method-argument]
        """Select the next points to evaluate using Expected Improvement (EI).

        This method optimizes the EI acquisition function.

        Args:
            n: Number of points to select (used for batch BO, which is not supported by
                this agent and will raise an error).
            seed: Random seed for reproducibility.

        Returns:
            Points selected by the agent for evaluation.

        Raises:
            RuntimeError: If the model has not been initialized with
                `tell_initial_data`.
            ValueError: If `n > 1`, since batch BO is not supported by this agent.
        """
        if self.model is None:
            msg = "Model has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        best_f = self.train_y.max()

        if n == 1:
            acq_func = ExpectedImprovement(
                model=self.model,
                best_f=best_f,
            )
        else:
            msg = "Batch BO is not supported by this agent."
            raise ValueError(msg)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()


class WindowVanillaBO(VanillaBO):
    """Modification of Hvarfner's Vanilla BO to scale to high N.

    Adopts the same approach as Hvarfner et al. but uses a simple approach to scale to
    high numbers of observations N by solely retaining a window of the most recent
    `window_size` observations, and discarding the rest.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        window_size: int = 100,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,
    ) -> None:
        """Initialize the VanillaBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            window_size: The number of most recent observations to retain when fitting
              the model. Older observations will be discarded.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.window_size = window_size
        super().__init__(
            input_dim=input_dim,
            acqf_opt_strategy=acqf_opt_strategy,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"window_{self.window_size}_vanilla_bo"

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Note that this method retains only the most recent `window_size` observations
        from the initial data.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )
        # Retain only the most recent `window_size` observations
        if self.train_X.shape[0] > self.window_size:
            self.train_X = self.train_X[-self.window_size :]
            self.train_y = self.train_y[-self.window_size :]
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the SingleTaskGP model.

        Note that this method retains only the most recent `window_size` observations
        when fitting the model, discarding older observations.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        # Retain only the most recent `window_size` observations
        if self.train_X.shape[0] > self.window_size:
            self.train_X = self.train_X[-self.window_size :]
            self.train_y = self.train_y[-self.window_size :]
        self._fit_model()


class BestWindowVanillaBO(VanillaBO):
    """Modification of Hvarfner's Vanilla BO to scale to high N.

    Adopts the same approach as WindowVanillaBO but ensures that the best `n_best`
    observations (by y-value) are always retained in the window, even if they fall
    outside the most recent `window_size` observations.

    The window is constructed by taking the union of:
    1. The top `n_best` points by y-value from the current data.
    2. The most recent observations, filling the remaining window capacity.

    Duplicates are avoided: if a top-n point is already in the recency window, it
    doesn't consume an extra slot.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        window_size: int = 100,
        n_best: int = 1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,
    ) -> None:
        """Initialize the BestWindowVanillaBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            window_size: The maximum number of observations to retain when fitting
              the model.
            n_best: The number of top-performing observations to always retain.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given, or if n_best >
              window_size.
        """
        if n_best > window_size:
            msg = f"n_best ({n_best}) must be <= window_size ({window_size})"
            raise ValueError(msg)
        self.window_size = window_size
        self.n_best = n_best
        super().__init__(
            input_dim=input_dim,
            acqf_opt_strategy=acqf_opt_strategy,
            dtype=dtype,
            device=device,
            **kwargs,
        )

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"best_{self.n_best}_window_{self.window_size}_vanilla_bo"

    def _apply_window(self) -> None:
        """Select the training window: top n_best + most recent to fill capacity."""
        N = self.train_X.shape[0]
        if self.window_size >= N:
            return

        # Indices of the top n_best points by y-value
        top_indices = torch.topk(self.train_y.squeeze(-1), min(self.n_best, N)).indices

        # Indices of the most recent observations
        recent_indices = torch.arange(N - self.window_size, N, device=self.device)

        # Union of both sets (unique indices), preserving at most window_size
        all_indices = torch.cat([top_indices, recent_indices])
        unique_indices = torch.unique(all_indices)

        # If union exceeds window_size, prioritize top-n and trim recency
        if unique_indices.shape[0] > self.window_size:
            top_set = set(top_indices.tolist())
            recent_list = recent_indices.tolist()
            # Keep all top-n, fill remaining from recent (most recent first)
            remaining = self.window_size - len(top_set)
            recent_only = [i for i in reversed(recent_list) if i not in top_set]
            keep = sorted(top_set | set(recent_only[:remaining]))
            unique_indices = torch.tensor(keep, device=self.device, dtype=torch.long)

        self.train_X = self.train_X[unique_indices]
        self.train_y = self.train_y[unique_indices]

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )
        self._apply_window()
        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Update the agent with new observations and refit the SingleTaskGP model.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._apply_window()
        self._fit_model()
