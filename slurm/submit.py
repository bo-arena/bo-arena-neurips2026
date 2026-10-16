"""Submit experiment grid to SLURM.

Computes the cross product of benchmarks, agents, and strategies, looks up
the right SLURM resources for each combination, and submits one job array
per combination (one array task per seed).

Usage:
    # Preview what would be submitted:
    python slurm/submit.py \
        --benchmark rover,svm \
        --agent vanilla_bo,saasbo \
        --acqf-opt-strategy default,raasp \
        --seed-range 0-9 \
        --dry-run

    # Actually submit:
    python slurm/submit.py \
        --benchmark rover,svm \
        --agent vanilla_bo,saasbo \
        --acqf-opt-strategy default,raasp \
        --seed-range 0-9
"""

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SBATCH_TEMPLATE = SCRIPT_DIR / "run_experiment.sbatch"

# ---------------------------------------------------------------------------
# Resource profiles keyed by agent name. Falls back to "default".
# Add new agents here as needed.
# ---------------------------------------------------------------------------
RESOURCE_PROFILES = {
    "saasbo": {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 8,
        "mem": "32G",
        "time": "72:00:00",
        "n_evals": 500,
    },
    "default": {
        "device": "cpu",
        "cpus": 4,
        "mem": "6G",
        "time": "48:00:00",
    },
}

# ---------------------------------------------------------------------------
# Minimum memory per benchmark. The final allocation is
# max(agent profile mem, benchmark mem).
# Benchmarks not listed here use 0 (i.e., agent profile decides).
# ---------------------------------------------------------------------------
BENCHMARK_MEM = {
    "mujoco-ant": "12G",
    "mujoco-humanoid": "16G",
}

# ---------------------------------------------------------------------------
# Minimum time per benchmark. The final allocation is
# max(agent profile time, benchmark time).
# Benchmarks not listed here use "00:00:00" (i.e., agent profile decides).
# ---------------------------------------------------------------------------
BENCHMARK_TIME = {
    "mujoco-ant": "72:00:00",
    "mujoco-humanoid": "72:00:00",
}


# ---------------------------------------------------------------------------
# Per-(benchmark, agent) overrides. These are merged on top of the resolved
# profile, so you can selectively override any field (device, gres, mem, …).
# ---------------------------------------------------------------------------
COMBO_OVERRIDES: dict[tuple[str, str], dict] = {
    ("mujoco-humanoid", "turbo"): {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 4,
        "time": "48:00:00",
    },
    ("mujoco-humanoid", "turbo_dsp"): {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 4,
        "time": "48:00:00",
    },
    ("mujoco-humanoid", "turbo_log_ei"): {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 4,
        "time": "48:00:00",
    },
    ("mujoco-ant", "saasbo"): {
        "time": "96:00:00",
    },
    ("mujoco-humanoid", "saasbo"): {
        "gres": "gpu:1,gpumem_per_gpu:120G",  # SAASBO quickly OOMs
        "time": "96:00:00",
        "n_evals": 350,  # perhaps more is possible, has not been tested
    },
    ("mujoco-humanoid", "linear_bo"): {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 4,
        "mem": "32G",
        "time": "48:00:00",
    },
    ("mujoco-humanoid", "bounce"): {
        "device": "cuda",
        "gres": "gpu:1",
        "cpus": 4,
        "time": "120:00:00",  # time needed exhibits high variance across seeds
    },
}


def _parse_time(time_str: str) -> int:
    """Parse a SLURM time string like '48:00:00' to total seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    msg = f"Unrecognized time string '{time_str}'. Use HH:MM:SS or MM:SS format."
    raise ValueError(msg)


def _format_time(seconds: int) -> str:
    """Format seconds back to HH:MM:SS."""
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_mem(mem_str: str) -> int:
    """Parse a memory string like '16G' or '512M' to megabytes."""
    mem_str = mem_str.strip().upper()
    if mem_str.endswith("G"):
        return int(mem_str[:-1]) * 1024
    if mem_str.endswith("M"):
        return int(mem_str[:-1])
    if mem_str.endswith("T"):
        return int(mem_str[:-1]) * 1024 * 1024
    msg = (
        f"Unrecognized memory string '{mem_str}'. "
        "Use a suffix of 'M', 'G', or 'T' (e.g. '16G', '512M')."
    )
    raise ValueError(msg)


def _format_mem(mb: int) -> str:
    """Format megabytes back to a human-readable string."""
    if mb >= 1024 and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def parse_config_sh() -> dict[str, str]:
    """Read key=value pairs from slurm/config.sh."""
    config_path = SCRIPT_DIR / "config.sh"
    if not config_path.exists():
        print(
            f"Error: {config_path} not found.\n"
            "Copy slurm/config.sh.example to slurm/config.sh and fill in your paths.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip().strip('"').strip("'")

    required_keys = ["REPO", "LOGS", "CPU_PARTITION", "GPU_PARTITION"]
    missing = [k for k in required_keys if not config.get(k)]
    if missing:
        print(
            f"Error: config.sh is missing keys: {', '.join(missing)}.\n"
            "See slurm/config.sh.example for the expected format.",
            file=sys.stderr,
        )
        sys.exit(1)

    return config


def get_profile(agent: str, benchmark: str) -> dict:
    """Return the resource profile for the given agent, with combo overrides."""
    profile = {**RESOURCE_PROFILES.get(agent, RESOURCE_PROFILES["default"])}
    if (benchmark, agent) in COMBO_OVERRIDES:
        profile.update(COMBO_OVERRIDES[benchmark, agent])
    return profile


def resolve_mem(agent: str, benchmark: str) -> str:
    """Return the memory allocation as max(agent profile, benchmark minimum).

    If "mem" is explicitly set in COMBO_OVERRIDES, that value is used
    directly (benchmark floor is skipped).
    """
    profile = get_profile(agent, benchmark)
    agent_mem = _parse_mem(profile["mem"])
    override = COMBO_OVERRIDES.get((benchmark, agent), {})
    if "mem" in override:
        return _format_mem(agent_mem)
    bench_mem = _parse_mem(BENCHMARK_MEM.get(benchmark, "0G"))
    return _format_mem(max(agent_mem, bench_mem))


def resolve_time(agent: str, benchmark: str) -> str:
    """Return the time limit as max(agent profile, benchmark minimum).

    If "time" is explicitly set in COMBO_OVERRIDES, that value is used
    directly (benchmark floor is skipped).
    """
    profile = get_profile(agent, benchmark)
    agent_time = _parse_time(profile["time"])
    override = COMBO_OVERRIDES.get((benchmark, agent), {})
    if "time" in override:
        return _format_time(agent_time)
    bench_time = _parse_time(BENCHMARK_TIME.get(benchmark, "00:00:00"))
    return _format_time(max(agent_time, bench_time))


def build_sbatch_cmd(
    benchmark: str,
    agent: str,
    acqf_opt_strategy: str,
    seed_range: str,
    config: dict[str, str],
    batch_size: int | None = None,
    n_evals: int | None = None,
    n_init: int | None = None,
) -> list[str]:
    """Build the sbatch command for a single experiment combination."""
    profile = get_profile(agent, benchmark)
    mem = resolve_mem(agent, benchmark)
    time = resolve_time(agent, benchmark)
    partition_key = "GPU_PARTITION" if profile["device"] == "cuda" else "CPU_PARTITION"
    partition = config[partition_key]

    export_parts = [
        "ALL",
        f"BENCHMARK={benchmark}",
        f"AGENT={agent}",
        f"ACQF_OPT_STRATEGY={acqf_opt_strategy}",
        f"DEVICE={profile['device']}",
    ]
    if n_evals is not None:
        export_parts.append(f"N_EVALS={n_evals}")
    elif "n_evals" in profile:
        export_parts.append(f"N_EVALS={profile['n_evals']}")
    if batch_size is not None:
        export_parts.append(f"BATCH_SIZE={batch_size}")
    if n_init is not None:
        export_parts.append(f"N_INIT={n_init}")
    export_vars = ",".join(export_parts)

    cmd = [
        "sbatch",
        f"--chdir={config['REPO']}",
        f"--partition={partition}",
        "--ntasks-per-node=1",
        f"--cpus-per-task={profile['cpus']}",
        f"--mem={mem}",
        f"--time={time}",
        f"--array={seed_range}",
        f"--export={export_vars}",
        "-o",
        f"{config['LOGS']}/%j_%x_%a.out",
    ]

    if "gres" in profile:
        cmd.append(f"--gres={profile['gres']}")

    cmd.append(str(SBATCH_TEMPLATE))
    return cmd


def main() -> None:
    """Parse arguments and submit experiment grid to SLURM."""
    parser = argparse.ArgumentParser(
        description="Submit experiment grid to SLURM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        help="Comma-separated benchmark names (e.g., rover,svm,lasso-dna)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Comma-separated agent names (e.g., vanilla_bo,saasbo)",
    )
    parser.add_argument(
        "--acqf-opt-strategy",
        type=str,
        required=True,
        help="Comma-separated strategies (e.g., default,raasp)",
    )
    parser.add_argument(
        "--seed-range",
        type=str,
        default="0-9",
        help="SLURM array range for seeds (default: 0-9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for candidate generation (overrides default of 1)",
    )
    parser.add_argument(
        "--n-evals",
        type=int,
        default=None,
        help="Total number of evaluations (overrides agent profile default)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Number of initial points (overrides benchmark default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sbatch commands without submitting",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"\d+(-\d+)?(%\d+)?(,\d+(-\d+)?(%\d+)?)*", args.seed_range):
        parser.error(
            f"Invalid --seed-range format: '{args.seed_range}'. "
            "Expected a SLURM array spec like '0-9', '0,2,4', or '0-9%3'."
        )

    config = parse_config_sh()
    Path(config["LOGS"]).mkdir(parents=True, exist_ok=True)

    benchmarks = [s.strip() for s in args.benchmark.split(",")]
    agents = [s.strip() for s in args.agent.split(",")]
    strategies = [s.strip() for s in args.acqf_opt_strategy.split(",")]

    grid = list(itertools.product(benchmarks, agents, strategies))

    print(
        f"{'[DRY RUN] ' if args.dry_run else ''}Submitting {len(grid)} job array(s) "
        f"(seeds {args.seed_range} each):\n"
    )

    submitted = 0
    failed = 0

    for benchmark, agent, strategy in grid:
        cmd = build_sbatch_cmd(
            benchmark,
            agent,
            strategy,
            args.seed_range,
            config,
            batch_size=args.batch_size,
            n_evals=args.n_evals,
            n_init=args.n_init,
        )
        profile = get_profile(agent, benchmark)
        mem = resolve_mem(agent, benchmark)
        time = resolve_time(agent, benchmark)
        label = (
            f"  {benchmark} / {agent} / {strategy}"
            f"  [{profile['device']}, {mem}, {time}]"
        )

        if args.dry_run:
            print(label)
            print(f"    {' '.join(cmd)}\n")
        else:
            print(label)
            result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
                cmd, capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                print(f"    {result.stdout.strip()}")
                submitted += 1
            else:
                print(f"    FAILED: {result.stderr.strip()}")
                failed += 1

    if not args.dry_run:
        print(f"\nDone. Submitted: {submitted}, Failed: {failed}")


if __name__ == "__main__":
    main()
