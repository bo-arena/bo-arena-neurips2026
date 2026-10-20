"""Bounce (BO Using iNcreasingly high-dim Combinatorial/continuous Embeddings) agent."""

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import torch
from botorch import gen_candidates_scipy
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from jaxtyping import Float
from torch.quasirandom import SobolEngine

from bo_arena.agents.ask_tell_agent import AskTellAgent
from bo_arena.utils.acqf_opt import get_candidates
from bo_arena.utils.dataset import Dataset
from bo_arena.utils.miscellaneous import generate_seed


@dataclass
class BounceState:
    """Bounce state used to track the adaptive subspace expansion (continuous-only)."""

    # Problem and embedding configuration
    input_dim: int
    n_initial_points: int
    eval_budget: int  # Total number of evaluations (including initial points)
    # allocated to the entire run
    new_bins_on_split: int = 2
    initial_target_dim: int = 2
    eval_budget_until_input_dim: int = 250  # TODO: a sensible way of  # ruff: ignore[missing-todo-author]
    # setting this (and some of the properties above) should be considered when we
    # release bo-arena.

    # Computed embedding and dimensionality state (post-initialized)
    d_init: int = dataclasses.field(init=False)
    target_dim: int = dataclasses.field(init=False)
    n_splits: int = dataclasses.field(init=False)

    # Trust region parameters
    length_continuous: float = 0.8
    length_init_continuous: float = 0.8
    length_min_continuous: float = 0.5**7
    length_max_continuous: float = 1.6
    # Retained from the mixed-space Bounce paper despite the continuous-only setting:
    # drives the geometric adjustment factor via the ratio
    # (length_min_discrete / length_discrete_continuous).
    length_discrete_continuous: float = dataclasses.field(init=False)
    length_init_discrete: int = 10
    length_min_discrete: int = 1

    # Optimization state
    split_budgets: dict[int, int] = dataclasses.field(init=False)
    n_evals: int = 0  # Stores total number of evaluations (including initial points),
    # managed by Bounce agent
    terminated: bool = False

    def __post_init__(self) -> None:
        """Compute split schedule and per-tier budgets."""
        if self.eval_budget_until_input_dim > self.eval_budget:
            msg = (
                f"eval_budget_until_input_dim ({self.eval_budget_until_input_dim}) "
                f"cannot exceed eval_budget ({self.eval_budget})"
            )
            raise ValueError(msg)
        if self.eval_budget_until_input_dim <= self.n_initial_points:
            msg = (
                f"eval_budget_until_input_dim ({self.eval_budget_until_input_dim}) "
                f"must exceed n_initial_points ({self.n_initial_points}) to leave "
                f"at least one evaluation for the adaptive phase"
            )
            raise ValueError(msg)

        self.d_init = min(self.initial_target_dim, self.input_dim)
        self.target_dim = self.d_init

        # Compute number of splits and optimize bin count
        self.n_splits = math.ceil(
            math.log(
                self.input_dim / self.d_init,
                self.new_bins_on_split + 1,
            )
        )
        self._adjust_number_bins_on_split()

        # Initialize trust region and compute per-tier budgets
        self.length_discrete_continuous = float(
            min(self.length_init_discrete, self.target_dim)
        )

        # Allocate evaluation budget to each dimensionality tier
        tiers = [
            int(self.d_init * (self.new_bins_on_split + 1) ** k)
            for k in range(self.n_splits + 1)
        ]
        self.split_budgets = {t: self.split_budget_for(t) for t in tiers}

    def _adjust_number_bins_on_split(self) -> None:
        """Pick ``new_bins_on_split`` that best reaches input_dim after n_splits splits.

        The default value may not evenly reach input_dim via
        ``d_init * (new_bins_on_split + 1)^n_splits``. This corrects for that by
        searching over candidate bin counts and selecting the one with the smallest
        overshoot/undershoot.
        """
        if self.d_init >= self.input_dim:
            return
        upper = max(2, math.floor(math.log2(max(self.input_dim, 2))))
        possible = np.arange(1, upper)
        if possible.size == 0:
            possible = np.array([1])
        errs = np.abs(self.d_init * (1 + possible) ** self.n_splits - self.input_dim)
        best = int(possible[int(np.argmin(errs))])
        if best != self.new_bins_on_split:
            self.new_bins_on_split = best
            self.n_splits = math.ceil(
                math.log(self.input_dim / self.d_init, self.new_bins_on_split + 1)
            )

    def split_budget_for(self, target_dim: int) -> int:
        """Calculate per-tier evaluation budget.

        Args:
            target_dim: Target dimensionality for this tier.

        Returns:
            Evaluation budget allocated to this tier.
        """
        total_budget = self.eval_budget_until_input_dim - self.n_initial_points
        if target_dim >= self.input_dim:
            return min(10 * target_dim, self.eval_budget - self.n_evals)
        num = self.new_bins_on_split * total_budget * target_dim
        den = self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1))
        split_budget = round(-num / den)
        return min(2**target_dim, split_budget)

    def reset_trust_region(self) -> None:
        """Reset trust region lengths after split or restart."""
        self.length_discrete_continuous = float(
            min(self.length_init_discrete, self.target_dim)
        )
        self.length_continuous = self.length_init_continuous
        self.terminated = False


def compute_adjustment_factor(state: BounceState, batch_size: int) -> float:
    """Calculate the geometric shrink factor for the trust region.

    Args:
        state: Current Bounce state.
        batch_size: Size of the current batch.

    Returns:
        Geometric adjustment factor for the trust region length.
    """
    # Remaining evaluation budget for the current dimensionality tier
    remaining = state.split_budgets[state.target_dim]
    remaining = min(remaining, state.eval_budget - state.n_evals)
    remaining = max(remaining, 1)

    # Geometric factor chosen so that length_discrete_continuous reaches
    # length_min_discrete in exactly `remaining` single-point steps
    factor = (state.length_min_discrete / state.length_discrete_continuous) ** (
        1.0 / remaining
    )
    factor = factor**batch_size
    return float(np.clip(factor, a_min=1e-10, a_max=None))


def update_bounce_trust_region_state(
    state: BounceState,
    y_next: Float[torch.Tensor, "n 1"],
    y_best_prev: float,
    adjustment_factor: float,
) -> BounceState:
    """Update the trust region based on observed improvement.

    Shrinks or expands the trust region length based on whether improvement
    was observed, and marks termination when the TR shrinks below minimum length.

    Args:
        state: Current Bounce state.
        y_next: New function values from the latest batch.
        y_best_prev: Best observed value prior to this batch.
        adjustment_factor: Geometric factor for TR adjustment.

    Returns:
        Updated Bounce state.
    """
    # Determine if improvement was observed
    y_best_new = float(torch.max(y_next).item())
    improved = y_best_new > y_best_prev + 1e-3 * math.fabs(y_best_prev)

    # Shrink or expand trust region lengths geometrically
    if not improved:
        state.length_discrete_continuous *= adjustment_factor
        state.length_continuous *= adjustment_factor
    else:
        state.length_discrete_continuous = min(
            state.length_discrete_continuous / adjustment_factor,
            float(state.target_dim),
        )
        state.length_continuous = min(
            state.length_continuous / adjustment_factor,
            state.length_max_continuous,
        )

    # Check termination: TR has shrunk to minimum size
    discrete_done = state.length_discrete_continuous - 1e-4 <= state.length_min_discrete
    continuous_done = state.length_continuous - 1e-4 <= state.length_min_continuous
    if discrete_done or continuous_done:
        state.terminated = True

    return state


def embedding_matrix(
    input_dim: int,
    target_dim: int,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Float[torch.Tensor, "target_dim input_dim"]:
    """Create a sparse embedding matrix for dimensionality reduction.

    This matrix projects points from a high-dimensional space to a lower-dimensional
    subspace using a hashing-based approach.

    Args:
        input_dim: The dimensionality of the original input space.
        target_dim: The dimensionality of the target subspace.
        generator: Optional random number generator for reproducibility.
        dtype: The torch dtype for the output tensor.
        device: The torch device for the output tensor.

    Returns:
        A (target_dim, input_dim) tensor representing the embedding matrix.
    """
    # Return identity matrix if target size greater than input size
    if target_dim >= input_dim:
        return torch.eye(input_dim, dtype=dtype, device=device)

    # Add 1 to indices for padding column in matrix
    input_dims_perm = torch.randperm(input_dim, generator=generator, device=device) + 1

    # Split dims into almost equally-sized bins
    bins = torch.tensor_split(input_dims_perm, target_dim)

    # Zero pad bins, the index 0 will be cut off later
    bins = torch.nn.utils.rnn.pad_sequence(bins, batch_first=True)

    # Add one extra column for padding
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)

    # Fill mask with random +/- 1 at indices
    mtrx = mtrx.scatter_(
        1,
        bins,
        (
            2
            * torch.randint(
                2, (target_dim, input_dim), generator=generator, device=device
            )
            - 1
        ).to(mtrx.dtype),
    )

    # Cut off index zero as this corresponds to zero padding
    return mtrx[:, 1:]


def project_up(
    X_target_01: Float[torch.Tensor, "N target_dim"],
    S: Float[torch.Tensor, "target_dim input_dim"],
) -> Float[torch.Tensor, "N input_dim"]:
    """Project low-dim points up to the full input space via the embedding ``S``.

    The projection is performed in the ``[-1, 1]`` space where ``S`` is defined: input
    points are first rescaled from ``[0, 1]^target_dim`` to ``[-1, 1]^target_dim``,
    multiplied by ``S``, then rescaled back to ``[0, 1]^input_dim``.

    Args:
        X_target_01: A batch of points in ``[0, 1]^target_dim``.
        S: The count-sketch embedding matrix.

    Returns:
        A tensor of projected points in ``[0, 1]^input_dim``.
    """
    X_target = (2.0 * X_target_01) - 1.0
    X_full = X_target @ S
    return (X_full + 1.0) / 2.0


def project_down(
    X_full_01: Float[torch.Tensor, "N input_dim"],
    S: Float[torch.Tensor, "target_dim input_dim"],
) -> Float[torch.Tensor, "N target_dim"]:
    """Project full-dim points down to the target subspace via least-squares.

    Inverse of :func:`project_up`. Points are rescaled to ``[-1, 1]^input_dim``,
    solved against ``S.T`` in the least-squares sense, and rescaled back to
    ``[0, 1]^target_dim``. Recovery is exact (up to floating-point noise) when the
    input originated from ``project_up`` because ``S`` has full row rank; for
    arbitrary full-dim inputs, ``lstsq`` silently returns a min-norm pseudo-inverse
    solution rather than an exact recovery.

    Args:
        X_full_01: A batch of points in ``[0, 1]^input_dim``,
            assumed to have originated from :func:`project_up` with the same ``S``.
        S: The count-sketch embedding matrix.

    Returns:
        A tensor of projected points in ``[0, 1]^target_dim``.
    """
    X_full = (2.0 * X_full_01) - 1.0
    X_target = torch.linalg.lstsq(S.t(), X_full.t()).solution.t()
    return (X_target + 1.0) / 2.0


def increase_embedding_and_observations(
    S: Float[torch.Tensor, "target_dim input_dim"],
    X: Float[torch.Tensor, "n_obs target_dim"],
    n_new_bins: int,
    generator: torch.Generator | None = None,
) -> tuple[
    Float[torch.Tensor, "new_target_dim input_dim"],
    Float[torch.Tensor, "n_obs new_target_dim"],
]:
    """Increase the subspace dimensionality by splitting existing bins.

    This function implements the adaptive subspace expansion by splitting each row of
    the embedding matrix into multiple new rows, effectively increasing the target
    dimensionality while preserving the mapping for existing observations.

    Args:
        S: The current embedding matrix of shape (target_dim, input_dim).
        X: The current observations in the subspace.
        n_new_bins: The number of new bins to create from each existing bin.
        generator: Optional random number generator for reproducibility.

    Returns:
        A tuple (S_update, X_update) where:
            S_update: The expanded embedding matrix.
            X_update: The updated observations in the new subspace.

    Raises:
        ValueError: If the number of columns in X does not match the number of rows in
            S, indicating that the observations do not lie in the current subspace.
    """
    if X.size(1) != S.size(0):
        msg = (
            f"Observations don't lie in row space of S: "
            f"X.size(1)={X.size(1)}, S.size(0)={S.size(0)}"
        )
        raise ValueError(msg)

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[
            torch.randperm(len(idxs_non_zero), generator=generator, device=S.device)
        ].reshape(-1)

        if len(idxs_non_zero) <= 1:
            continue

        non_zero_elements = row[idxs_non_zero].reshape(-1)

        # Number of new bins is always less or equal than the contributing input dims in
        # the row minus one
        n_row_bins = min(n_new_bins + 1, len(idxs_non_zero))

        # The dims in the first bin won't be moved
        new_bins = torch.tensor_split(idxs_non_zero + 1, n_row_bins)[1:]
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        # Pad the tuples of bins with zeros to apply _scatter
        new_bins_padded = torch.nn.utils.rnn.pad_sequence(new_bins, batch_first=True)
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        # Submatrix to stack on S_update
        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), dtype=S.dtype, device=S.device
        )

        # Fill with old values (add 1 to indices for padding column)
        S_stack = S_stack.scatter_(
            1, new_bins_padded, els_to_move_padded.to(S_stack.dtype)
        )

        # Set values that were moved to zero in current row
        S_update[row_idx, torch.hstack(new_bins) - 1] = 0

        # Repeat observations for row at the end of X (column-wise)
        X_update = torch.hstack((
            X_update,
            X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)),
        ))

        # Stack onto S_update except for padding column
        S_update = torch.vstack((
            S_update,
            S_stack[:, 1:],
        ))

    return S_update, X_update


class Bounce(AskTellAgent):
    """Continuous-only Bounce agent (no binary/categorical inputs).

    This agent implements the continuous-only variant of Bounce as described in
    "Bounce: Reliable High-Dimensional Bayesian Optimization for Combinatorial and
    Mixed Spaces" by Papenmeier et al. (2023). Bounce starts optimization in a
    low-dimensional random subspace and adaptively expands the embedding until the
    full input dimensionality is reached.

    Reference:
        Papenmeier, L., Nardi, L., Poloczek, M. (2023). Bounce: Reliable
        High-Dimensional Bayesian Optimization for Combinatorial and Mixed
        Spaces. Neural Information Processing Systems (NeurIPS).
    """

    def __init__(
        self,
        input_dim: int,
        n_total_evaluations: int,
        acqf_opt_strategy: str = "default",
        use_dsp: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
        use_log_ei: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        **kwargs: Any,  # ruff: ignore[any-type, unused-method-argument]
    ) -> None:
        """Initialize the Bounce agent.

        Args:
            input_dim: Dimensionality of the input space (assumed [0,1]^d).
            n_total_evaluations: Total evaluation budget.
            acqf_opt_strategy: Strategy for optimizing the acquisition function. Can be
              "default", "no_raasp", or "raasp". Setting "default" preserves
              the original Bounce optimizer settings (raw_samples=512, num_restarts=10).
            use_dsp: If True, use BoTorch's default SingleTaskGP (Hvarfner's
              Dimensionality-Scaled Prior) instead of the Matern-2.5 + ARD + Gamma
              priors used in the original Bounce paper.
            use_log_ei: If True, use qLogNEI instead of standard EI for improved
              numerical stability. Note that the original paper used standard EI, so
              the default is False to preserve that behavior.
            dtype: The torch dtype to use for the model and data.
            device: The torch device to use for the model and data.
            kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If an unknown acqf_opt_strategy is given.
        """
        self.dtype = dtype
        self.device = device
        self.input_dim = input_dim
        self.use_dsp = use_dsp
        self.use_log_ei = use_log_ei
        self.n_total_evaluations = n_total_evaluations
        if self.n_total_evaluations is None:
            msg = (
                "n_total_evaluations is required for Bounce. "
                "Pass it as a keyword argument to the constructor."
            )
            raise RuntimeError(msg)
        if acqf_opt_strategy in {"default", "no_raasp", "raasp"}:
            self.acqf_opt_strategy = acqf_opt_strategy
        else:
            msg = (
                f"Unknown acqf_opt_strategy: {acqf_opt_strategy!r}. "
                "Note that the bounce agent doesn't support botorch_raasp."
            )
            raise ValueError(msg)

        self.train_X: torch.Tensor | None = None  # stored in [0, 1]^target_dim
        self.train_y = torch.empty(0, 1, dtype=dtype, device=device)
        self.model: SingleTaskGP | None = None
        self.state: BounceState | None = None
        self._restart_buffer = torch.empty(
            0, self.input_dim, dtype=dtype, device=device
        )

        # Projection matrix. Coordinate conventions throughout this class:
        #   Arena / external   : [0, 1]^input_dim
        #   Subspace / train_X : [0, 1]^target_dim  (GP is trained in this space)
        # Conversion between the two goes via [-1, 1] where S is defined; see
        # project_up / project_down in baxus_bounce_utils.
        self.S: Float[torch.Tensor, "target_dim input_dim"] | None = None

    def __str__(self) -> str:
        """Return a string representation of the agent."""
        base_str = "bounce"
        if self.use_log_ei:
            base_str += "_log_ei"
        if self.use_dsp:
            base_str += "_dsp"
        return base_str

    def _fit_model(self) -> None:
        """Fit a SingleTaskGP model in the low-dimensional subspace.

        When use_dsp=False, uses Matern-2.5 with ARD and Gamma priors on train_X
        which is stored in [0,1]^target_dim. When use_dsp=True, uses BoTorch's
        default SingleTaskGP (Hvarfner's DSP). Falls back to Adam if standard
        fitting fails.
        """
        if self.use_dsp:
            self.model = SingleTaskGP(self.train_X, self.train_y)
        else:
            target_dim = self.train_X.shape[-1]
            likelihood = GaussianLikelihood(noise_prior=GammaPrior(1.1, 0.05))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=target_dim,
                    lengthscale_prior=GammaPrior(1.5, 0.1),
                ),
                outputscale_prior=GammaPrior(1.5, 0.5),
            )
            self.model = SingleTaskGP(
                self.train_X,
                self.train_y,
                covar_module=covar_module,
                likelihood=likelihood,
            )

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        with gpytorch.settings.max_cholesky_size(float("inf")):  # Always use Cholesky
            try:
                fit_gpytorch_mll(mll)
            except ModelFittingError:
                # Right after increasing the target dim, the covariance matrix may
                # become indefinite, and thus Cholesky decomposition might fail.
                self.model.train()
                optimizer = torch.optim.Adam(
                    [{"params": self.model.parameters()}], lr=0.1
                )
                for _ in range(100):
                    optimizer.zero_grad()
                    output = self.model(self.train_X)
                    loss = -mll(output, self.model.train_targets.flatten())
                    loss.backward()
                    optimizer.step()
                self.model.eval()

    def _get_trust_region_bounds(
        self,
        center: Float[torch.Tensor, " target_dim"],
        length: float,
    ) -> tuple[Float[torch.Tensor, " target_dim"], Float[torch.Tensor, " target_dim"]]:
        """Get the trust region bounds centered around the best point.

        Computes bounds as ``center ± length * weights / 2`` clipped to [0, 1],
        where weights are ARD lengthscales normalized to unit geometric mean.

        Args:
            center: Center point of the trust region in [0,1]^target_dim.
            length: Length parameter of the trust region.

        Returns:
            A tuple of (tr_lb, tr_ub) tensors in [0,1]^target_dim.
        """
        # Scale the TR to be proportional to the lengthscales
        if self.use_dsp:
            # With DSP enabled, there is no ScaleKernel wrapping the base kernel
            lengthscale = self.model.covar_module.lengthscale
        else:
            lengthscale = self.model.covar_module.base_kernel.lengthscale
        weights = lengthscale.squeeze().detach()
        if weights.ndim > 0:
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        # Create trust region bounds in [0,1]^d space with lengthscale scaling
        tr_lb = torch.clamp(center - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(center + weights * length / 2.0, 0.0, 1.0)

        return tr_lb, tr_ub

    def generate_initial_points(self, n: int, seed: int) -> Float[np.ndarray, "N D"]:
        """Generate initial points in the low-dimensional subspace and project up.

        Initializes state and embedding matrix, generates Sobol points in the
        low-dimensional subspace, and projects them to [0,1]^d.

        Args:
            n: Number of initial points to generate.
            seed: Random seed for reproducibility.

        Returns:
            Initial points in [0,1]^d for the Arena to evaluate.
        """
        # Initialize state and embedding matrix early
        self.state = BounceState(
            input_dim=self.input_dim,
            n_initial_points=n,
            eval_budget=self.n_total_evaluations,
        )
        embedding_ss, sobol_ss = np.random.SeedSequence(seed).spawn(2)
        generator = torch.Generator(device=self.device).manual_seed(
            generate_seed(embedding_ss)
        )
        self.S = embedding_matrix(
            self.state.input_dim,
            self.state.d_init,
            generator=generator,
            dtype=self.dtype,
            device=self.device,
        )

        # Generate Sobol points in [0,1]^target_dim and project up to [0,1]^input_dim
        sobol = SobolEngine(
            self.state.target_dim, scramble=True, seed=generate_seed(sobol_ss)
        )
        X_target = sobol.draw(n).to(dtype=self.dtype, device=self.device)
        X_full_01 = project_up(X_target, self.S)

        return X_full_01.detach().cpu().numpy()

    def tell_initial_data(self, data: Dataset, seed: int) -> None:  # ruff: ignore[unused-method-argument]
        """Provide initial observations to the agent and fit the SingleTaskGP model.

        Args:
            data: Initial observations to provide to the agent.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If `generate_initial_points` has not been called first.
        """
        if self.state is None:
            msg = (
                "Bounce state has not been initialized. "
                "generate_initial_points must be called before tell_initial_data."
            )
            raise RuntimeError(msg)

        X_full_01 = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        self.train_X = project_down(X_full_01, self.S)
        self.train_y = torch.tensor(
            data.objectives, dtype=self.dtype, device=self.device
        )

        self.n_init = self.train_X.shape[0]  # Used for Bounce restarts
        self.state.n_evals = self.train_X.shape[0]

        self._fit_model()

    def tell(self, data: Dataset, seed: int) -> None:
        """Update the agent with new observations and manage trust region geometry.

        Projects new data into the subspace, updates the trust region based on
        improvement, and handles embedding splits or full-dimensional restarts
        when the trust region terminates.

        Args:
            data: Observations corresponding to the points requested in the previous
                `ask` call.
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Raises:
            RuntimeError: If the Bounce state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "Bounce state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        X_full_01 = torch.tensor(data.X, dtype=self.dtype, device=self.device)
        new_X = project_down(X_full_01, self.S)
        new_y = torch.tensor(data.objectives, dtype=self.dtype, device=self.device)

        batch_size = new_X.shape[0]

        # Don't update the trust region if new data is still from the restart buffer
        if self.model is not None:
            y_best_prev = float(self.train_y.max().item())

            factor = compute_adjustment_factor(self.state, batch_size)
            self.state = update_bounce_trust_region_state(
                self.state, new_y, y_best_prev, factor
            )

            # Update per-tier budget tracking
            tr_dim = self.state.target_dim
            self.state.split_budgets[tr_dim] = (
                self.state.split_budgets[tr_dim] - batch_size
            )

        self.train_X = torch.cat((self.train_X, new_X), dim=0)
        self.train_y = torch.cat([self.train_y, new_y])
        self.state.n_evals += batch_size

        if self.state.terminated:
            if self.state.target_dim < self.state.input_dim:
                # Increase target space: split each bin into new_bins_on_split+1 bins,
                # growing the embedding from target_dim to a higher dimensionality.
                generator = torch.Generator(device=self.device).manual_seed(seed)
                self.S, self.train_X = increase_embedding_and_observations(
                    self.S,
                    self.train_X,
                    self.state.new_bins_on_split,
                    generator=generator,
                )
                self.state.target_dim = len(self.S)

                # Allocate budget for the new tier if not already tracked
                if self.state.target_dim not in self.state.split_budgets:
                    self.state.split_budgets[self.state.target_dim] = (
                        self.state.split_budget_for(self.state.target_dim)
                    )

                # Partially reset trust region
                self.state.reset_trust_region()

            elif self.state.target_dim == self.state.input_dim:
                # Completely reset trust region and resample a fresh embedding
                self.state.reset_trust_region()
                self.state.split_budgets[self.state.target_dim] = (
                    self.state.split_budget_for(self.state.target_dim)
                )
                embedding_ss, sobol_ss = np.random.SeedSequence(seed).spawn(2)
                generator = torch.Generator(device=self.device).manual_seed(
                    generate_seed(embedding_ss)
                )
                self.S = embedding_matrix(
                    input_dim=self.state.input_dim,
                    target_dim=self.state.input_dim,
                    generator=generator,
                    dtype=self.dtype,
                    device=self.device,
                )

                # Purge local data
                self.train_X = torch.empty(
                    0, self.state.target_dim, dtype=self.dtype, device=self.device
                )
                self.train_y = torch.empty(0, 1, dtype=self.dtype, device=self.device)
                self.model = None

                # Re-seed the restart buffer with fresh Sobol points in [0,1]^d
                sobol = SobolEngine(
                    self.state.target_dim,
                    scramble=True,
                    seed=generate_seed(sobol_ss),
                )
                X_target = sobol.draw(self.n_init).to(
                    dtype=self.dtype, device=self.device
                )
                self._restart_buffer = project_up(X_target, self.S)

            else:
                msg = (
                    f"Impossible state: target_dim={self.state.target_dim} "
                    f"> dim={self.state.input_dim}"
                )
                raise RuntimeError(msg)

        # No need to fit model when restart buffer is still being unrolled
        if self._restart_buffer.shape[0] == 0:
            self._fit_model()

    def ask(self, n: int, seed: int) -> Float[np.ndarray, "{n} D"]:
        """Select candidate points via (q)(log)(N)EI optimization in the trust region.

        Optimizes (q)(log)(N)EI over the trust region in the low-dimensional subspace,
        then projects candidates back to the full space.

        Args:
            n: Number of points to select (used for batch BO).
            seed: Random seed for reproducibility. Implementations should use this
                seed for any stochastic operations to ensure deterministic behavior.

        Returns:
            Candidate points in [0,1]^d for evaluation.

        Raises:
            RuntimeError: If the Bounce state has not been initialized with
                `tell_initial_data`.
        """
        if self.state is None:
            msg = "Bounce state has not been initialized. Call tell_initial_data first."
            raise RuntimeError(msg)

        # Consume buffer: Feed pre-generated points sequentially
        if self._restart_buffer.shape[0] > 0:
            batch_n = min(n, self._restart_buffer.shape[0])
            samples = self._restart_buffer[:batch_n]
            self._restart_buffer = self._restart_buffer[batch_n:]
            return samples.cpu().numpy()

        # Find the best point (center of trust region) in [0,1]^target_dim
        best_idx = self.train_y.argmax()
        x_center = self.train_X[best_idx].clone()

        # Get trust region bounds (in [0,1]^target_dim)
        tr_lb, tr_ub = self._get_trust_region_bounds(
            x_center, self.state.length_continuous
        )
        bounds = torch.stack([tr_lb, tr_ub], dim=0)

        # Optimize (q)(log)(N)EI over the trust region
        best_f = float(self.train_y.max().item())
        with gpytorch.settings.max_cholesky_size(float("inf")):  # Always use Cholesky
            if n == 1:
                if self.use_log_ei:
                    acqf = qLogNoisyExpectedImprovement(
                        model=self.model, X_baseline=self.train_X
                    )
                else:
                    acqf = ExpectedImprovement(model=self.model, best_f=best_f)
                if self.acqf_opt_strategy == "default":
                    candidates, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        raw_samples=512,
                        num_restarts=10,
                        options={"batch_limit": 64},
                    )
                elif self.acqf_opt_strategy == "no_raasp":
                    candidates, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        gen_candidates=gen_candidates_scipy,
                        raw_samples=5000,
                        num_restarts=5,
                        retry_on_optimization_warning=False,
                        options={"batch_limit": 64},
                    )
                elif self.acqf_opt_strategy == "raasp":
                    # Perform RAASP initialisation ourself due to issues with
                    # sample_around_best=True and trust regions in the current BoTorch
                    # implementation.
                    X_cand = get_candidates(
                        n_candidates=5000,
                        lb=tr_lb,
                        ub=tr_ub,
                        best_x=x_center,
                        seed=seed,
                        method="raasp",
                        dtype=self.dtype,
                        device=self.device,
                    )

                    # Score all candidates and keep the top 5 as initial
                    # points for the gradient-based optimizer.
                    with torch.no_grad():
                        acq_vals = torch.cat([
                            acqf(batch.unsqueeze(1)) for batch in X_cand.split(64)
                        ])
                    top_idx = acq_vals.topk(5).indices
                    X_init = X_cand[top_idx].unsqueeze(1)

                    tr_bounds = torch.stack([tr_lb, tr_ub], dim=0)
                    candidates, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=tr_bounds,
                        q=1,
                        gen_candidates=gen_candidates_scipy,
                        batch_initial_conditions=X_init,
                        num_restarts=5,
                        retry_on_optimization_warning=False,
                        options={
                            "batch_limit": 64,
                        },
                    )
                else:
                    msg = f"Unknown acqf_opt_strategy: {self.acqf_opt_strategy!r}"
                    raise ValueError(msg)

            else:
                if self.acqf_opt_strategy != "default":
                    msg = (
                        "The 'default' acqf_opt_strategy must be used for batch "
                        "optimization. "
                        f"Received: {self.acqf_opt_strategy!r}"
                    )
                    raise ValueError(msg)
                # Sequential greedy with X_pending
                sampler = SobolQMCNormalSampler(
                    sample_shape=torch.Size([1024]), seed=seed
                )
                cand_list: list[torch.Tensor] = []
                for _ in range(n):
                    x_pending = torch.cat(cand_list, dim=0) if cand_list else None
                    if self.use_log_ei:
                        acqf = qLogNoisyExpectedImprovement(
                            model=self.model,
                            X_baseline=self.train_X,
                            sampler=sampler,
                            X_pending=x_pending,
                        )
                    else:
                        acqf = qExpectedImprovement(
                            model=self.model,
                            best_f=best_f,
                            sampler=sampler,
                            X_pending=x_pending,
                        )
                    x_new, _ = optimize_acqf(
                        acq_function=acqf,
                        bounds=bounds,
                        q=1,
                        raw_samples=512,
                        num_restarts=10,
                        options={"batch_limit": 64},
                    )
                    cand_list.append(x_new)
                candidates = torch.cat(cand_list, dim=0)

        X_next_01 = project_up(candidates, self.S)
        return X_next_01.detach().cpu().numpy()
