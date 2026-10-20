"""Script for testing out agents locally, without full WandB and logging set up."""

import argparse
import logging

import torch

from bo_arena.agents import AGENTS
from bo_arena.agents.bounce import Bounce
from bo_arena.arena import Arena
from bo_arena.problems import BencherProblem
from bo_arena.problems.synthetic_problems import Ackley, Michalewicz, StyblinskiTang

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization Problem")

    parser.add_argument(
        "--problem",
        type=str,
        default="lasso-dna",
        help="Name of the problem to run (e.g., 'svm')",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="random",
        choices=list(AGENTS.keys()),
        help=f"Agent to use. Available: {', '.join(AGENTS.keys())}",
    )
    parser.add_argument(
        "--acqf-opt-strategy",
        type=str,
        default="default",
        choices=["default", "no_raasp", "raasp"],
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        dest="n_initial_points",
        help="Number of initial points",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=None,
        dest="n_total_evaluations",
        help="Total number of evaluations",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="Address of the Bencher server (hostname or unix:///path/to/socket)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port of the Bencher server",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run agent computations on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=["float32", "float64"],
        help="Torch dtype for agent computations",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark."""
    args = parse_args()

    logger.info(f"Starting run with Problem: {args.problem}, Agent: {args.agent}")
    logger.info(f"Num threads: {torch.get_num_threads()}")

    # Initialize Problem
    if args.problem == "ackley_dim_100":
        problem = Ackley(dim=100)
    elif args.problem == "michalewicz_dim_100":
        problem = Michalewicz(dim=100)
    elif args.problem == "styblinski_tang_dim_100":
        problem = StyblinskiTang(dim=100)
    else:
        problem = BencherProblem(
            name=args.problem, address=args.address, port=args.port
        )

    metadata = problem.metadata
    input_dim = metadata.input_dim

    # Resolve n_init and n_tot
    n_init = (
        args.n_initial_points
        if args.n_initial_points is not None
        else metadata.default_n_initial_points
    )
    n_tot = (
        args.n_total_evaluations
        if args.n_total_evaluations is not None
        else metadata.default_n_total_evaluations
    )

    # Initialize Agent
    agent_class = AGENTS[args.agent]
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    agent = agent_class(
        input_dim=input_dim,
        acqf_opt_strategy=args.acqf_opt_strategy,
        n_total_evaluations=n_tot,
        device=args.device,
        dtype=dtype,
    )
    if isinstance(agent, Bounce) and n_init != 5:
        msg = "Bounce agent expected to be run with 5 initial points."
        raise ValueError(msg)

    # Run Arena
    bo_arena = Arena(
        problem=problem,
        agent=agent,
        seed=args.seed,
        n_initial_points=n_init,
        n_total_evaluations=n_tot,
        batch_size=args.batch_size,
        logger=logger,
        wandb_run=None,
    )

    bo_arena.run()


if __name__ == "__main__":
    main()
