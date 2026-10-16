"""Infinite width BNN agent."""

import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.models.kernels import InfiniteWidthBNNKernel
from botorch.models.utils.gpytorch_modules import (
    get_gaussian_likelihood_with_gamma_prior,
)
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.acqf_opt import get_log_ei_opt_kwargs
from bo_arena.utils.dataset import Dataset


class IBNN(AskTellAgent):
    """Infinite-width BNN agent.

    Fits a GP whose covariance is the kernel of an infinite-width fully-connected ReLU
    network (Lee et al., 2018), as used for BO by Li et al. (2024), and selects points
    using the LogEI acquisition function.

    The kernel's weight and bias variances are fitted by marginal likelihood alongside
    the mean and noise. See `IBNNLi2024` for the paper-faithful variant, which holds
    them fixed.

    References:
        Lee, J., Bahri, Y., Novak, R., Schoenholz, S., Pennington, J., &
        Sohl-Dickstein, J. (2018). Deep Neural Networks as Gaussian Processes.
        International Conference on Learning Representations (ICLR).

        Li, Y. L., Rudner, T. G. J., & Wilson, A. G. (2024). A Study of Bayesian
        Neural Network Surrogates for Bayesian Optimization. International Conference
        on Learning Representations (ICLR).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        depth: int = 3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the IBNN agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp" or "botorch_raasp".
            depth: Depth of the network whose infinite-width limit defines the kernel.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments.

        Raises:
            ValueError: If `depth` is negative or `acqf_opt_strategy` is invalid.
        """
        if depth < 0:
            msg = f"depth must be non-negative, got {depth}."
            raise ValueError(msg)

        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        if acqf_opt_strategy == "default":
            # Use the same opt kwargs as Vanilla BO
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

        self.depth = depth
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
        return f"ibnn_depth_{self.depth}"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP with the I-BNN kernel."""
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
            covar_module=InfiniteWidthBNNKernel(depth=self.depth),
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
        """Select the next points to evaluate using Log Expected Improvement (LogEI).

        This method optimizes the LogEI (or qLogEI for batch size > 1) acquisition
        function.

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

        if n == 1:
            acq_func = LogExpectedImprovement(model=self.model, best_f=best_f)
        else:
            acq_func = qLogExpectedImprovement(model=self.model, best_f=best_f)

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()


class IBNNLi2024(IBNN):
    """I-BNN agent using the surrogate configuration of Li et al. (2024).

    Reproduces the surrogate as implemented in the original repository
    (https://github.com/yucenli/bnn-bo), which differs from `IBNN` in two ways:

    - The kernel's weight and bias variances are fixed at 10.0 and 1.6 rather than
      fitted.
    - The likelihood uses a Gamma(1.1, 0.05) noise prior, which was the SingleTaskGP
      default in the BoTorch version they pinned (0.8.5). Current BoTorch defaults to a
      LogNormal prior instead.

    Note that the acquisition function is *not* matched: Li et al. use qEI whereas this
    agent utilises the more numerically stable LogEI.

    Reference:
        Li, Y. L., Rudner, T. G. J., & Wilson, A. G. (2024). A Study of Bayesian
        Neural Network Surrogates for Bayesian Optimization. International Conference
        on Learning Representations (ICLR).
    """

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"ibnn_li2024_depth_{self.depth}"

    def _build_kernel(self) -> InfiniteWidthBNNKernel:
        """Construct the I-BNN kernel, with variances pinned to Li et al.'s values."""
        covar_module = InfiniteWidthBNNKernel(depth=self.depth)
        covar_module = covar_module.to(dtype=self.dtype, device=self.device)
        covar_module.weight_var = torch.tensor(
            10.0, dtype=self.dtype, device=self.device
        )
        covar_module.bias_var = torch.tensor(1.6, dtype=self.dtype, device=self.device)

        # keep variances fixed, as in Li et al.
        covar_module.raw_weight_var.requires_grad_(False)
        covar_module.raw_bias_var.requires_grad_(False)

        return covar_module

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP with the I-BNN kernel, as configured by Li et al.

        Only the mean constant and the noise are fitted; the kernel variances are held
        fixed. `get_gaussian_likelihood_with_gamma_prior` supplies the Gamma(1.1, 0.05)
        noise prior that was the SingleTaskGP default in BoTorch 0.8.5.
        """
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
            likelihood=get_gaussian_likelihood_with_gamma_prior(),
            covar_module=self._build_kernel(),
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_mll(mll)
