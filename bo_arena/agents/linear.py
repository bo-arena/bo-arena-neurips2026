"""Linear BO agent."""

import math
from typing import override

import gpytorch
import numpy as np
import torch
from botorch.acquisition import LogExpectedImprovement, qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models.exact_prediction_strategies import (
    DefaultPredictionStrategy,
    LinearPredictionStrategy,
)
from gpytorch.priors import LogNormalPrior
from jaxtyping import Float
from linear_operator.operators import (
    LowRankRootLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)
from torch import nn

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.acqf_opt import get_log_ei_opt_kwargs
from bo_arena.utils.dataset import Dataset


def _root_linear_operator(
    root: Float[torch.Tensor, "... N D"],
) -> RootLinearOperator | LowRankRootLinearOperator:
    """Return the most efficient root linear operator for the given feature matrix.

    Uses ``LowRankRootLinearOperator`` when N > D (more data than features), which
    enables Woodbury-based inference in D-dimensional feature space. Falls back to
    ``RootLinearOperator`` otherwise.
    """
    n, d = root.shape[-2:]
    if d >= n:
        return RootLinearOperator(root)
    return LowRankRootLinearOperator(root)


def project_onto_unit_sphere(
    x: Float[torch.Tensor, "... D"],
) -> Float[torch.Tensor, "... D+1"]:
    """Project points onto a unit sphere via inverse stereographic projection.

    Maps points from R^d onto S^d (the d-dimensional unit sphere embedded in R^(d+1)).

    Args:
        x: Input points in R^d.

    Returns:
        Points on the (d+1)-dimensional unit sphere.
    """
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    return torch.cat([2 * x, x_sq_norm - 1.0], dim=-1).mul(1.0 / (1.0 + x_sq_norm))


class SphericalLinearKernel(Kernel):
    """Linear kernel applied after inverse stereographic projection with ARD.

    This kernel first scales inputs by ARD lengthscales and a global lengthscale,
    projects them onto a unit sphere via inverse stereographic projection, then computes
    a linear (dot-product) kernel. A learned constant/linear mixture allows
    interpolation between a constant kernel and the pure spherical linear kernel.

    The kernel is:
        k(x, x') = c_0 + c_1 * <phi(s(x)), phi(s(x'))>

    where s(x) is the lengthscale-scaled input, phi is the inverse stereographic
    projection, and (c_0, c_1) are learned mixture weights.
    """

    has_lengthscale = True

    def __init__(
        self,
        ard_num_dims: int,
        normalize_lengthscale: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
        **kwargs: object,
    ) -> None:
        """Initialize the SphericalLinearKernel.

        Args:
            ard_num_dims: Number of input dimensions (enables ARD lengthscales).
            normalize_lengthscale: If True, use ``softmax(ls).sqrt()`` as the effective
                lengthscale, constraining the ARD vector to unit L2 norm.
            kwargs: Additional arguments passed to the base Kernel class.
        """
        super().__init__(ard_num_dims=ard_num_dims, **kwargs)

        self.normalize_lengthscale = normalize_lengthscale

        # Learned mixture coefficients: softmax([raw_coeffs]) -> [constant, linear]
        self.raw_coeffs = nn.Parameter(torch.zeros(*self.batch_shape, 2))

        # Global lengthscale: sigmoid(raw_glob_ls) * max_sq_norm
        self.raw_glob_ls = nn.Parameter(torch.zeros(*self.batch_shape, 1))

        # ARD lengthscale prior and positivity constraint
        ls_prior = LogNormalPrior(
            loc=math.sqrt(2),
            scale=math.sqrt(3),
        )
        ls_constraint = GreaterThan(0.025, transform=None, initial_value=ls_prior.mode)
        self.register_prior(
            "lengthscale_prior",
            ls_prior,
            lambda m: m.lengthscale,
            lambda m, v: m._set_lengthscale(v),  # ruff: ignore[private-member-access]
        )
        self.register_constraint("raw_lengthscale", ls_constraint)

        # Bounds for the input space (set during forward, used for global lengthscale)
        self._mins: torch.Tensor | None = None
        self._maxs: torch.Tensor | None = None
        self._centers: torch.Tensor | None = None

    def set_bounds(
        self,
        bounds: Float[torch.Tensor, "2 D"],
    ) -> None:
        """Set the input space bounds, used for centering and global lengthscale.

        Args:
            bounds: Tensor of shape (2, d) with bounds[0] = lower, bounds[1] = upper.
        """
        self._mins = bounds[0]
        self._maxs = bounds[1]
        self._centers = (bounds[0] + bounds[1]) / 2.0

    @override
    def prediction_strategy(
        self,
        train_inputs: list[Float[torch.Tensor, "N D"]],
        train_prior_dist: gpytorch.distributions.MultivariateNormal,
        train_labels: Float[torch.Tensor, " N"],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> LinearPredictionStrategy | DefaultPredictionStrategy:
        """Choose prediction strategy based on N vs D.

        LinearPredictionStrategy (D-space Woodbury) is efficient when N > D.
        DefaultPredictionStrategy (N-space Cholesky) is efficient when D >= N,
        which is the common case for high-dimensional problems (e.g. mujoco-humanoid).
        """
        n = train_inputs[0].shape[-2]
        d = train_inputs[0].shape[-1] + 2  # +1 stereographic projection, +1 constant
        if d >= n:
            return DefaultPredictionStrategy(
                train_inputs, train_prior_dist, train_labels, likelihood
            )
        return LinearPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )

    def forward(
        self,
        x1: Float[torch.Tensor, "... N D"],
        x2: Float[torch.Tensor, "... M D"],
        diag: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
        **params: object,  # ruff: ignore[unused-method-argument]
    ) -> (
        RootLinearOperator
        | LowRankRootLinearOperator
        | MatmulLinearOperator
        | Float[torch.Tensor, "... N"]
    ):
        """Compute the spherical linear kernel.

        Args:
            x1: First set of input points.
            x2: Second set of input points.
            diag: If True, return only the diagonal of the kernel matrix.
            params: Additional parameters (unused).

        Returns:
            Kernel matrix or diagonal.
        """
        if diag:
            # The diagonal is always 1
            return torch.ones(x1.shape[:-1], dtype=x1.dtype, device=x1.device)

        x1_equal_x2 = torch.equal(x1, x2)

        if self.normalize_lengthscale:
            # softmax(ls).sqrt() has L2 norm = 1 since sum(softmax) = 1
            lengthscale = torch.softmax(self.lengthscale, dim=-1).sqrt()
        else:
            lengthscale = self.lengthscale  # (1, D) with ARD

        # Global lengthscale based on max possible squared norm
        max_sq_norm = (
            ((self._maxs - self._mins) / (2 * lengthscale))
            .square()
            .sum(dim=-1, keepdim=True)
        )
        glob_ls = torch.sqrt(torch.sigmoid(self.raw_glob_ls[..., None]) * max_sq_norm)

        # Mixture weights via softmax; use dim=-1 to support batch dimensions
        coeffs = torch.softmax(self.raw_coeffs, dim=-1)
        sqrt_const = torch.sqrt(coeffs[..., 0])
        sqrt_linear = torch.sqrt(coeffs[..., 1])

        def _featurize(x: torch.Tensor) -> torch.Tensor:
            x_ = (x - self._centers).div(lengthscale).div(glob_ls)
            x_ = project_onto_unit_sphere(x_)
            return torch.cat(
                [
                    x_ * sqrt_linear[..., None, None],
                    sqrt_const[..., None, None].expand(*x_.shape[:-1], 1),
                ],
                dim=-1,
            )

        x1_ = _featurize(x1)

        if x1_equal_x2:
            return _root_linear_operator(x1_)

        x2_ = _featurize(x2)
        return MatmulLinearOperator(x1_, x2_.mT)


class LinearBO(AskTellAgent):
    """Linear BO agent using a spherical linear kernel.

    This agent implements the method from "We Still Don't Understand High-Dimensional
    Bayesian Optimization" by Doumont et al. (2025), which uses a spherical linear
    kernel (linear kernel after inverse stereographic projection) with ARD lengthscales.

    Reference:
        Doumont, C., Fan, D., Maus, N., Gardner, J. R., Moss, H., & Pleiss, G. (2025).
        We Still Don't Understand High-Dimensional Bayesian Optimization. International
        Conference on Artificial Intelligence and Statistics (AISTATS).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        normalize_lengthscale: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialize the LinearBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "botorch_raasp". Setting "default" uses the same
              settings as in the original paper.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            normalize_lengthscale: If True, use ``softmax(ls).sqrt()`` as the effective
                lengthscale, constraining the ARD vector to unit L2 norm.
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
        self.normalize_lengthscale = normalize_lengthscale

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
        suffix = "_normalized_ls" if self.normalize_lengthscale else ""
        return f"linear_bo{suffix}"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP with the SphericalLinearKernel."""
        covar_module = SphericalLinearKernel(
            ard_num_dims=self.input_dim,
            normalize_lengthscale=self.normalize_lengthscale,
        )
        covar_module.set_bounds(self.bounds)

        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
            covar_module=covar_module,
        )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Always use Cholesky (for Woodbury) + fast log-prob (for scaling to large N)
        with (
            gpytorch.settings.max_cholesky_size(float("inf")),
            gpytorch.settings.cholesky_max_tries(10),
            gpytorch.settings.fast_computations(
                log_prob=True, covar_root_decomposition=False, solves=False
            ),
        ):
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
        function using `sample_around_best=True` for better candidate generation.

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
            acq_func = LogExpectedImprovement(
                model=self.model,
                best_f=best_f,
            )
        else:
            # Use MC version for batch
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
