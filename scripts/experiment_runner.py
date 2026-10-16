"""Main entry point for running BO arena experiments."""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

import botorch
import cpuinfo
import gpytorch
import numpy as np
import torch
import wandb

from bo_arena.agents import AGENTS
from bo_arena.agents.bounce import Bounce
from bo_arena.arena import Arena
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS
from bo_arena.problems import BencherProblem
from bo_arena.problems.synthetic_problems import Ackley, Michalewicz, StyblinskiTang

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization Benchmark")

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
        choices=["default", "no_raasp", "raasp", "botorch_raasp"],
    )
    parser.add_argument(
        "--seed-idx",
        type=int,
        required=True,
        dest="seed_idx",
        help=f"Index into EXPERIMENT_SEEDS (0-{len(EXPERIMENT_SEEDS) - 1})",
    )
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
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    return parser.parse_args()


def main() -> None:  # ruff: ignore[too-many-statements]
    """Run the benchmark."""
    # Ensure working tree is clean for reproducibility
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],  # ruff: ignore[start-process-with-partial-path]
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT_DIR,
    ).stdout.strip()
    if git_status:
        msg = (
            "Uncommitted changes detected. Commit or stash them before running "
            "experiments to ensure reproducibility.\n"
            f"Dirty files:\n{git_status}"
        )
        raise RuntimeError(msg)

    args = parse_args()
    seed = EXPERIMENT_SEEDS[args.seed_idx]

    # wandb.init() reads the entity and project from these environment variables.
    if not args.no_wandb:
        missing = [
            v for v in ("WANDB_ENTITY", "WANDB_PROJECT") if not os.environ.get(v)
        ]
        if missing:
            msg = (
                f"Environment variables {', '.join(missing)} must be set for WandB "
                "logging (see slurm/config.sh.example), or pass --no-wandb."
            )
            raise RuntimeError(msg)

    # Initialize Problem and Agent
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

    # Build experiment directory and file paths
    # Structure: results/{problem}/{agent}/acqf_opt_{strategy}/[batch_{N}/]
    problem_name = args.problem
    agent_name = str(agent)
    acqf_opt_dir = f"acqf_opt_{args.acqf_opt_strategy}"
    experiment_dir = RESULTS_DIR / problem_name / agent_name / acqf_opt_dir
    if args.batch_size > 1:
        experiment_dir = experiment_dir / f"batch_{args.batch_size}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    xs_file = experiment_dir / f"seed_{seed}_xs.npy"
    ys_file = experiment_dir / f"seed_{seed}_ys.npy"
    times_file = experiment_dir / f"seed_{seed}_times.npz"
    config_file = experiment_dir / f"seed_{seed}_config.json"
    log_file = experiment_dir / f"seed_{seed}.log"

    # Check for existing results/logs to avoid overwriting previous runs
    if (
        xs_file.exists()
        or ys_file.exists()
        or times_file.exists()
        or config_file.exists()
        or log_file.exists()
    ):
        msg = (
            f"Results or log file already exists for seed {seed} in "
            f"{experiment_dir}. Refusing to overwrite."
        )
        raise FileExistsError(msg)

    # Build per-seed config capturing experiment parameters and environment
    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT_DIR,
    ).stdout.strip()

    config = {
        "problem": problem_name,
        "agent": agent_name,
        "acqf_opt_strategy": args.acqf_opt_strategy,
        "n_initial_points": n_init,
        "n_total_evaluations": n_tot,
        "batch_size": args.batch_size,
        "git_commit": git_hash or "unknown",
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpytorch.__version__,
        "cpu": cpuinfo.get_cpu_info()["brand_raw"],
        "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "torch_num_threads": torch.get_num_threads(),
        "device": args.device,
        "dtype": args.dtype,
    }
    with config_file.open("x", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2) + "\n")

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )
    wandb_group = f"{problem_name}/{agent_name}/{acqf_opt_dir}"
    if args.batch_size > 1:
        wandb_group += f"/batch_{args.batch_size}"
    logger.info(f"Experiment: {wandb_group}, seed={seed}")
    logger.info(
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')},"
        f" SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID')}"
    )
    logger.info(f"Num threads: {torch.get_num_threads()} ")
    logger.info(f"Config: {config}")

    # Initialize WandB in offline mode to avoid API throttling with many
    # concurrent jobs. Upload results later with: wandb sync <wandb-dir>
    wandb_run = None
    if not args.no_wandb:
        os.environ["WANDB_MODE"] = "offline"
        wandb_run = wandb.init(
            name=f"seed_{seed}",
            group=wandb_group,
            config={**config, "seed": seed},
        )
        logger.info(f"WandB run ID: {wandb_run.id}")

    # Run Arena
    bo_arena = Arena(
        problem=problem,
        agent=agent,
        seed=seed,
        n_initial_points=n_init,
        n_total_evaluations=n_tot,
        batch_size=args.batch_size,
        logger=logger,
        wandb_run=wandb_run,
    )
    result = bo_arena.run()

    # Save results
    np.save(xs_file, result.observed_data.X)
    np.save(ys_file, result.observed_data.objectives[:, 0])
    np.savez(
        times_file,
        ask=np.array(result.ask_times),
        tell=np.array(result.tell_times),
        eval=np.array(result.eval_times),
    )
    logger.info(f"Results saved to {experiment_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
