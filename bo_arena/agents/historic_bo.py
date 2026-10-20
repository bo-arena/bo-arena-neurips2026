"""GP-based BO agent prior to logEI and DSP."""

import numpy as np
import torch
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import gen_candidates_scipy
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from jaxtyping import Float

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.dataset import Dataset


class HistoricBO(AskTellAgent):
    """Historic GP-based BO agent prior to logEI and DSP.

    This agent implements a typical GP-based BO approach prior to the introduction of
    DSP in Hvarfner et al. (2024), and logEI in Ament et al. (2023). In particular, it
    uses the standard Expected Improvement (EI) acquisition function (specifically the
    qNoisy variant, for consistency with VanillaBO and Ax, which use qLogNEI), and a
    GP with a Matérn-5/2 kernel with a lengthscale prior of Gamma(3, 6) and an
    outputscale prior of Gamma(2, 0.15). It also uses a Gamma(1.1, 0.05) prior on the
    noise, with a minimum noise value of 1e-4. These were  the defaults in BoTorch
    prior to the 0.12 release: https://github.com/meta-pytorch/botorch/releases/tag/v0.12.0.

    Reference:
        Hvarfner, C., Hellsten, E. O., & Nardi, L. (2024). Vanilla Bayesian Optimization
        Performs Great in High Dimensions. International Conference on Machine Learning
        (ICML).
        Ament, S., Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2023).
        Unexpected Improvements to Expected Improvement for Bayesian Optimization.
        Neural Information Processing Systems (NeurIPS).
    """

    def __init__(
        self,
        input_dim: int,
        acqf_opt_strategy: str = "default",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Initialise the HistoricBO agent.

        Args:
            input_dim: Dimensionality of the input space (assumed to be [0,1]^d).
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can
              only be "default", and uses the same optimisation settings as the vanilla
              BO implementation for a fair comparison.
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
                "raw_samples": 1024,
                "num_restarts": 4,
                "retry_on_optimization_warning": False,
                "options": {
                    "maxiter": 300,
                    "batch_limit": 64,
                },
            }
        else:
            msg = (
                f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}. This agent is "
                "solely intended for use with the default strategy, to compare with "
                "Vanilla BO."
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
        return "historic_bo"

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model using ExactMarginalLogLikelihood.

        Uses the same hyperpriors as BoTorch < 0.12. These can be found here:
            https://github.com/meta-pytorch/botorch/blob/v0.11.3/botorch/models/utils/gpytorch_modules.py.
        """
        covar_module = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5,
                ard_num_dims=self.train_X.shape[-1],
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        noise_prior = GammaPrior(1.1, 0.05)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                1e-4, transform=None, initial_value=noise_prior_mode
            ),
            # transform=None is deliberate, and was present in BoTorch <= 0.12:
            # https://github.com/meta-pytorch/botorch/blob/v0.11.3/botorch/models/utils/gpytorch_modules.py#L52
        )
        self.model = SingleTaskGP(
            self.train_X,
            self.train_y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
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
        """Select the next points to evaluate using Expected Improvement (EI).

        This method optimizes the EI (or qEI for batch size > 1) acquisition
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

        # Use qNoisy variant for consistency with Vanilla BO and Ax
        acq_func = qNoisyExpectedImprovement(model=self.model, X_baseline=self.train_X)

        candidates, _ = optimize_acqf(
            acq_func,
            bounds=self.bounds,
            q=n,
            gen_candidates=gen_candidates_scipy,
            **self.opt_kwargs,
        )

        return candidates.detach().cpu().numpy()
