"""SAASBO agent."""

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.generation import gen_candidates_scipy
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.acqf_opt import get_log_ei_opt_kwargs
from bo_arena.utils.dataset import Dataset


def check_jax_sees_gpu() -> None:
    """Raise error if JAX runs the sampler on the CPU while CUDA was asked for.

    Raises:
        RuntimeError: If JAX is on the CPU.
    """
    import jax  # ruff: ignore[import-outside-top-level]

    if jax.default_backend() == "cpu":
        msg = (
            "SAASBO was submitted with device='cuda', but JAX is running on the CPU. "
            "jax[cuda12] is a dependency on Linux, so either the environment is "
            "stale (re-run uv sync) or the plugin failed to initialise here."
        )
        raise RuntimeError(msg)


class SAASBO(AskTellAgent):
    """Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO).

    This agent implements the method described in "High-Dimensional Bayesian
    Optimization with Sparse Axis-Aligned Subspaces" by Eriksson & Jankowiak (2021),
    which places strong priors on the lengthscales to avoid overfitting.

    Reference:
        Eriksson, D., & Jankowiak, M. (2021). High-Dimensional Bayesian Optimization
        with Sparse Axis-Aligned Subspaces. Conference on Uncertainty in Artificial
        Intelligence (UAI).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        warmup_steps: int = 512,
        num_samples: int = 256,
        thinning: int = 16,
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the SAASBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            warmup_steps: Number of warmup steps for HMC sampling.
            num_samples: Number of HMC samples to draw.
            thinning: Thinning factor for HMC samples (keeps every Nth sample).
            kwargs: Additional arguments.

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
            RuntimeError: If ``device`` is CUDA but JAX wants to sample on the CPU.
        """
        if str(device).startswith("cuda"):
            check_jax_sees_gpu()

        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        if acqf_opt_strategy == "default":
            self.opt_kwargs = {
                "raw_samples": 5000,
                "num_restarts": 3,
                "options": {
                    "batch_limit": 64,
                },
            }
        elif acqf_opt_strategy in {"no_raasp", "botorch_raasp"}:
            self.opt_kwargs = get_log_ei_opt_kwargs(acqf_opt_strategy)
        else:
            msg = f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}"
            raise ValueError(msg)
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning

        self.train_X = torch.empty(0, self.input_dim, dtype=dtype, device=device)
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model = None

        # Bounds are always [0,1]^d since Bencher normalizes inputs
        self.bounds = torch.stack([
            torch.zeros(self.input_dim, dtype=self.dtype, device=self.device),
            torch.ones(self.input_dim, dtype=self.dtype, device=self.device),
        ])

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return "saasbo"

    def _fit_model(self, seed: int) -> None:
        """Fit a SAAS GP model using HMC (NUTS) sampling.

        Args:
            seed: Random seed for reproducibility.
        """
        self.model = SaasFullyBayesianSingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_y,
            outcome_transform=Standardize(m=1),
        )
        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=self.warmup_steps,
            num_samples=self.num_samples,
            thinning=self.thinning,
            disable_progbar=True,
            seed=seed,
        )
        # Every fit is a new shape, so the cache is never used and keeps growing on CPU.
        import jax  # ruff: ignore[import-outside-top-level]

        jax.clear_caches()
        torch.cuda.empty_cache()

    def tell_initial_data(self, data: Dataset, seed: int) -> None:
        """Provide initial observations to the agent and fit the SAAS GP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        self.train_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_y = torch.tensor(
            data.y, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)
        self._fit_model(seed)

    def tell(self, data: Dataset, seed: int) -> None:
        """Update the agent with new observations and refit the SAAS GP model.

        Args:
            data: Observations corresponding to the points requested
                in the previous `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.
        """
        new_X = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_y = torch.tensor(data.y, dtype=self.dtype, device=self.device).unsqueeze(-1)

        self.train_X = torch.cat([self.train_X, new_X])
        self.train_y = torch.cat([self.train_y, new_y])
        self._fit_model(seed)

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:  # ruff: ignore[unused-method-argument]
        """Select the next points to evaluate using Log Expected Improvement (qLogEI).

        This method optimizes the qLogEI acquisition function using MCMC samples from
        the SAAS model.

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

        best_f = self.train_y.max()

        # Use qLogEI for both single and batch cases since SAAS model uses MCMC samples
        acq_func = qLogExpectedImprovement(
            model=self.model,
            best_f=best_f,
        )
        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()
