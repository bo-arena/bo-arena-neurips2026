# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**bo-arena** is a Python package for simple, reliable, and efficient benchmarking of Bayesian Optimization (BO) algorithms. It provides an ask-tell framework to run BO agents on standardized problems with a strong emphasis on reproducibility and deterministic behavior.

## Commands

```bash
# Install all dependencies (uses uv)
uv sync --all-extras

# Run all tests
uv run pytest tests

# Run a single test
uv run pytest tests/test_seed_determinism.py::test_seed_determinism

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Architecture

The core loop is in `bo_arena/arena.py` (`Arena` class). It orchestrates the ask-tell loop between an agent and a problem, managing seed derivation for each step. `Arena.run()` can only be called once per instance to prevent state corruption.

**Agents** (`bo_arena/agents/`) implement the `AskTellAgent` ABC with three methods:
- `ask(n, seed)` — request `n` candidate points
- `tell(data, seed)` — receive observations
- `tell_initial_data(data, seed)` — receive initial dataset before the loop starts

New agents must be registered in `bo_arena/agents/registry.py`.

**Problems** (`bo_arena/problems/`) implement the `Problem` ABC with `evaluate_points()`. `BencherProblem` wraps real-world problems from the external Bencher service. Available problems (rover, mopta08, lasso-dna, svm, mujoco-ant, mujoco-humanoid) are defined in `problem.py` metadata. All problems use the **maximization convention** (following BoTorch).

**Seed management** (`bo_arena/utils/miscellaneous.py`) uses NumPy's `SeedSequence.spawn()` for hierarchical deterministic seeding. `seed_everything()` seeds random, numpy, torch, and CUDA. Each component (initial data generation, each ask/tell call) gets its own derived seed.

**Dataset** (`bo_arena/utils/dataset.py`) is a simple dataclass holding `(X, y)` pairs and supports concatenation via `+`.

## Key Conventions

- **Reproducibility is critical**: all randomness must flow through the `SeedSequence` hierarchy. Tests in `tests/test_seed_determinism.py` guard against regressions.
- **jaxtyping** is used for shape-annotated array types (e.g., `Float[np.ndarray, "N D"]`). Runtime checking is enabled via beartype in tests (`--jaxtyping-packages` pytest option).
- **Google-style docstrings**. Flexible naming conventions are intentional for research clarity (several PEP8/N-rules are disabled in Ruff config).
- Magic numbers and higher McCabe complexity are allowed (research code).
- `__init__` methods are not annotated; argument types are documented in the class docstring instead.
