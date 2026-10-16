"""Utilities for plotting cumulative-max BO results in NeurIPS style."""

import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator
from tueplots import figsizes, fonts, fontsizes
from tueplots.constants.color import palettes

from bo_arena.problems.metadata import PROBLEM_METADATA, SYNTHETIC_PROBLEM_METADATA
from bo_arena.experiment_seeds import EXPERIMENT_SEEDS
from bo_arena.utils.data_processing import get_cum_max_y_stats

PROBLEM_METADATA = {**PROBLEM_METADATA, **SYNTHETIC_PROBLEM_METADATA}

plt.rcParams.update({"figure.dpi": 200})

AGENT_NAME_MAPPING = {
    "best_1_window_100_vanilla_bo": "Vanilla BO (window 100 with incumbent)",
    "bounce": "Bounce",
    "ei_bo": "EI-BO (Cheon et al., 2025)",
    "eulbo": "EULBO",
    "historic_bo": "BoTorch v0.11.3 (2024)",
    "linear_bo": "Linear BO",
    "random": "Random",
    "saasbo": "SAASBO",
    "turbo": "TuRBO",
    "turbo_dsp": "TuRBO (DSP)",
    "turbo_log_ei": "TuRBO (LogEI)",
    "turbo_log_ei_dsp": "TuRBO (LogEI + DSP)",
    "vanilla_bo": "Vanilla BO",
    "vanilla_bo_yu_2026": "``Vanilla BO'' (Yu et al., 2026)",
    "window_100_vanilla_bo": "Vanilla BO (window 100)",
}

_PAULTOL = palettes.paultol_bright
# Colors are intentionally reused across agents that are never co-plotted
# (e.g. turbo / turbo_dsp share a color, as do the eulbo / linear_bo / turbo_log_ei
# group). Double-check the palette before adding a new co-plot combination.
AGENT_COLOR_MAPPING = {
    "best_1_window_100_vanilla_bo": f"#{_PAULTOL[4]}",
    "bounce": f"#{_PAULTOL[3]}",
    "ei_bo": f"#{_PAULTOL[3]}",
    "eulbo": f"#{_PAULTOL[1]}",
    "historic_bo": f"#{_PAULTOL[2]}",
    "linear_bo": f"#{_PAULTOL[1]}",
    "random": "#555555",
    "saasbo": f"#{_PAULTOL[5]}",
    "turbo": f"#{_PAULTOL[2]}",
    "turbo_dsp": f"#{_PAULTOL[2]}",
    "turbo_log_ei": f"#{_PAULTOL[1]}",
    "turbo_log_ei_dsp": f"#{_PAULTOL[1]}",
    "vanilla_bo": f"#{_PAULTOL[0]}",
    "vanilla_bo_yu_2026": f"#{_PAULTOL[1]}",
    "window_100_vanilla_bo": f"#{_PAULTOL[0]}",
}

# Per-agent linestyles. Only consulted when `show_strategy=False`; when comparing
# strategies within an agent, STRATEGY_LINESTYLE_MAPPING takes over so each line
# can be distinguished by strategy instead.
AGENT_LINESTYLE_MAPPING = {
    "ei_bo": "dashed",
    "historic_bo": "dashed",
    "random": "dashed",
    "turbo_dsp": "dashed",
    "turbo_log_ei_dsp": "dashed",
    "vanilla_bo_yu_2026": "dashed",
}

STRATEGY_NAME_MAPPING = {
    "default": "RAASP",
    "botorch_raasp": "BoTorch RAASP",
    "raasp": "RAASP",
    "no_raasp": "no RAASP",
}

# Per-strategy linestyles. Only consulted when `show_strategy=True`.
STRATEGY_LINESTYLE_MAPPING = {
    "default": "solid",
    "botorch_raasp": "solid",
    "no_raasp": "dashed",
}

PROBLEM_NAME_MAPPING = {
    "ackley_dim_100": "Ackley (100D)",
    "michalewicz_dim_100": "Michalewicz (100D)",
    "styblinski_tang_dim_100": "Styblinski-Tang (100D)",
    "bbob-attractivesector": "Attractive Sector (100D)",
    "bbob-ellipsoidrotated": "Ellipsoid (100D)",
    "bbob-griewankrosenbrock": "Griewank-Rosenbrock (100D)",
    "bbob-rastriginrotated": "Rastrigin (100D)",
    "bbob-schwefel": "Schwefel (100D)",
    "rover": "Rover (60D)",
    "mopta08": "MOPTA08 (124D)",
    "lasso-dna": "Lasso-DNA (180D)",
    "svm": "SVM (388D)",
    "mujoco-ant": "MuJoCo Ant (888D)",
    "mujoco-humanoid": "MuJoCo Humanoid (6392D)",
}

N_SEEDS = 20

_NOTEBOOKS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _NOTEBOOKS_DIR.parent / "results"
EULBO_RESULTS_DIR = _NOTEBOOKS_DIR.parent / "eulbo_results"
EI_BO_RESULTS_DIR = _NOTEBOOKS_DIR.parent / "ei_bo_results"
SAVEFIG_DIR = _NOTEBOOKS_DIR.parent.parent.parent / "latex/neurips-bo-arena/figures"


def _results_dir(agent_name: str, problem: str, strategy: str) -> Path:
    """Return the base directory for result files for this combination.

    Most agents live under the standard
    `RESULTS_DIR/{problem}/{agent}/acqf_opt_{strategy}/` tree; a few have their
    own conventions and are special-cased here.
    """
    if agent_name == "eulbo":
        return EULBO_RESULTS_DIR / problem
    if agent_name == "ei_bo":
        return EI_BO_RESULTS_DIR / problem
    base = RESULTS_DIR / problem / agent_name / f"acqf_opt_{strategy}"
    if agent_name in ["window_100_vanilla_bo", "best_1_window_100_vanilla_bo"]:
        return base / "batch_20"
    return base


def _load_cum_max_stats(
    res_path: Path,
    seeds: list[int],
    context: str,
    allow_partial_seeds: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load seed files and return (mean, sem) of the cumulative max.

    Emits a warning and returns None if the line should be skipped (either no
    seed files found, or seed files missing and partial seeds not allowed).

    Args:
        res_path: Directory containing seed_{seed}_ys.npy files.
        seeds: Seeds to look for.
        context: Human-readable identifier for warning messages
            (e.g. "vanilla_bo/default on svm").
        allow_partial_seeds: If True, proceed with whatever seeds are available.
            If False, any missing seed causes the whole line to be skipped.
    """
    available = [s for s in seeds if (res_path / f"seed_{s}_ys.npy").exists()]
    missing = len(seeds) - len(available)

    if not available:
        detail = (
            f"no seed files in {res_path}"
            if allow_partial_seeds
            else (f"{missing}/{len(seeds)} seeds missing")
        )
        warnings.warn(f"Skipping {context}: {detail}", stacklevel=3)
        return None
    if missing and not allow_partial_seeds:
        warnings.warn(
            f"Skipping {context}: {missing}/{len(seeds)} seed files missing",
            stacklevel=3,
        )
        return None
    if missing:
        warnings.warn(
            f"{context}: using {len(available)}/{len(seeds)} seeds ({missing} missing)",
            stacklevel=3,
        )

    ys = np.array([np.load(res_path / f"seed_{s}_ys.npy") for s in available])
    if ys.ndim == 3 and ys.shape[-1] == 1:
        ys = ys.squeeze(-1)
    return get_cum_max_y_stats(ys)


def _load_cum_max_stats_ei_bo(
    res_path: Path,
    seeds: list[int],
    context: str,
    allow_partial_seeds: bool,  # noqa: FBT001
    drop_k_slowest: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load ei_bo results and return (mean, sem) of the cumulative max.

    ei_bo stores each seed under its own subdirectory with checkpoint files
    (seed_{s}_iter_{n}_ys.npy) and, once complete, a final seed_{s}_ys.npy.
    All seeds are truncated to the shortest common run length so the array is
    rectangular.

    Args:
        drop_k_slowest: Drop the k seeds that have completed the fewest iterations
            before computing the common truncation point. Useful when a small number
            of slow seeds would otherwise cut the plot short for everyone.
    """

    def _latest(seed: int) -> tuple[int, Path] | None:
        seed_dir = res_path / f"seed_{seed}"
        if not seed_dir.exists():
            return None
        final = seed_dir / f"seed_{seed}_ys.npy"
        if final.exists():
            return len(np.load(final)), final
        best_n, best_path = -1, None
        for cp in seed_dir.glob(f"seed_{seed}_iter_*_ys.npy"):
            m = re.search(r"_iter_(\d+)_ys\.npy$", cp.name)
            if m:
                n = int(m.group(1))
                if n > best_n:
                    best_n, best_path = n, cp
        return (best_n, best_path) if best_n >= 0 and best_path is not None else None

    seed_latest = {s: d for s in seeds if (d := _latest(s)) is not None}
    available = list(seed_latest.keys())
    missing = len(seeds) - len(available)

    if not available:
        warnings.warn(f"Skipping {context}: no seed files in {res_path}", stacklevel=3)
        return None
    if missing and not allow_partial_seeds:
        warnings.warn(
            f"Skipping {context}: {missing}/{len(seeds)} seeds missing", stacklevel=3
        )
        return None
    if missing:
        warnings.warn(
            f"{context}: using {len(available)}/{len(seeds)} seeds ({missing} missing)",
            stacklevel=3,
        )

    if drop_k_slowest > 0:
        available = sorted(available, key=lambda s: seed_latest[s][0])[drop_k_slowest:]
        if not available:
            warnings.warn(
                f"Skipping {context}: all seeds dropped (drop_k_slowest={drop_k_slowest})",
                stacklevel=3,
            )
            return None
        warnings.warn(
            f"{context}: dropped {drop_k_slowest} slowest seeds, "
            f"using {len(available)} remaining",
            stacklevel=3,
        )

    min_iter = min(seed_latest[s][0] for s in available)
    ys = np.array([np.load(seed_latest[s][1])[:min_iter] for s in available])
    return get_cum_max_y_stats(ys)


def _plot_cum_max_y_single(
    agents: dict[str, list[str]],
    problems: list[str],
    n_seeds: int,
    ncols: int | None,
    height_to_width_ratio: float,
    show_strategy: bool,  # noqa: FBT001
    agent_name_mapping: dict[str, str] | None,
    legend_ncols: int | None,
    filename: str | None,
    allow_partial_seeds: bool,  # noqa: FBT001
    drop_k_slowest: int,
    overlay_last: list[str] | None,
) -> None:
    """Draw a single cum-max-y figure. See `plot_cum_max_y_custom` for details."""
    # Merge user-provided name overrides on top of the defaults
    names = {**AGENT_NAME_MAPPING, **(agent_name_mapping or {})}
    seeds = EXPERIMENT_SEEDS[:n_seeds]
    # Sort problems by input dimensionality (ascending) so the grid reads left-to-right,
    # low-to-high regardless of the order callers pass them in
    problems = sorted(problems, key=lambda p: PROBLEM_METADATA[p].input_dim)

    # Compute grid shape (default: single row with one column per problem)
    n_problems = len(problems)
    ncols = ncols or n_problems
    nrows = math.ceil(n_problems / ncols)

    # Compose tueplots style: figure sizing + NeurIPS fonts and font sizes
    size = figsizes.neurips2024(
        ncols=ncols, nrows=nrows, height_to_width_ratio=height_to_width_ratio
    )
    style = {**size, **fonts.neurips2024_tex(), **fontsizes.neurips2024()}

    with plt.rc_context(style):
        fig, axes = plt.subplots(nrows, ncols, sharey=False, squeeze=False)
        axes_flat = axes.flatten()

        for i, (ax, problem) in enumerate(zip(axes_flat, problems)):
            n_init = PROBLEM_METADATA[problem].default_n_initial_points
            # Track post-init y-bounds across lines to auto-zoom the y-axis later
            post_init_mins = []
            post_init_maxs = []

            for agent_name, strategies in agents.items():
                for strategy in strategies:
                    res_path = _results_dir(agent_name, problem, strategy)
                    context = f"{agent_name}/{strategy} on {problem}"
                    if agent_name == "ei_bo":
                        stats = _load_cum_max_stats_ei_bo(
                            res_path, seeds, context, allow_partial_seeds, drop_k_slowest
                        )
                    else:
                        stats = _load_cum_max_stats(
                            res_path, seeds, context, allow_partial_seeds
                        )
                    if stats is None:
                        continue
                    cum_max_y_mean, cum_max_y_sem = stats
                    iters = range(1, len(cum_max_y_mean) + 1)

                    # Build legend label and pick color/linestyle for this line
                    agent_label = names[agent_name]
                    strategy_label = STRATEGY_NAME_MAPPING.get(strategy, strategy)
                    label = (
                        f"{agent_label} ({strategy_label})"
                        if show_strategy
                        else agent_label
                    )
                    color = AGENT_COLOR_MAPPING[agent_name]
                    if show_strategy:
                        ls = STRATEGY_LINESTYLE_MAPPING.get(strategy, "solid")
                    else:
                        ls = AGENT_LINESTYLE_MAPPING.get(agent_name, "solid")
                    on_top = overlay_last and agent_name in overlay_last

                    # Mean line + SEM band
                    ax.plot(
                        iters, cum_max_y_mean, label=label, color=color, linestyle=ls,
                        **({"zorder": 3} if on_top else {}),
                    )
                    ax.fill_between(
                        iters,
                        cum_max_y_mean - cum_max_y_sem,
                        cum_max_y_mean + cum_max_y_sem,
                        alpha=0.2,
                        color=color,
                        **({"zorder": 2.5} if on_top else {}),
                    )

                    # Record post-init bounds so the y-axis can ignore init-phase spikes
                    post_init_data = slice(n_init, None)
                    post_init_mins.append(
                        min((cum_max_y_mean - cum_max_y_sem)[post_init_data])
                    )
                    post_init_maxs.append(
                        max((cum_max_y_mean + cum_max_y_sem)[post_init_data])
                    )

            # Dotted vertical line marking the end of the init phase
            ax.axvline(n_init + 0.5, color="gray", linestyle=(0, (1, 1)), alpha=0.5)

            # Zoom y-axis to the post-init range with a small margin
            if post_init_mins:
                y_min = min(post_init_mins)
                y_max = max(post_init_maxs)
                if y_max > y_min:
                    margin = 0.05 * (y_max - y_min)
                    ax.set_ylim(y_min - margin, y_max + margin)

            # Axis labels only on the outer edges of the grid
            ax.set_title(f"{PROBLEM_NAME_MAPPING[problem]}")
            row = i // ncols
            if row == nrows - 1:
                ax.set_xlabel("Iteration")
            if i % ncols == 0:
                ax.set_ylabel("Max. Observed Value")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            if ax.get_xlim()[1] >= 5000:
                ax.xaxis.set_major_formatter(EngFormatter(sep=""))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.grid(True, alpha=0.2)

        # Hide any leftover axes in the last (partially filled) row
        for j in range(n_problems, nrows * ncols):
            axes_flat[j].set_visible(False)

        fig.align_ylabels(axes[:, 0])

        # Single shared legend below the figure (dedupe across all axes so agents
        # absent from the first subplot still appear)
        seen = {}
        for ax in axes_flat[:n_problems]:
            for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
                if label not in seen:
                    seen[label] = handle
        fig.legend(
            list(seen.values()),
            list(seen.keys()),
            loc="outside lower center",
            ncols=legend_ncols or len(seen),
        )

        if filename is not None:
            plt.savefig(SAVEFIG_DIR / f"{filename}.pdf")
        plt.show()


def plot_cum_max_y_custom(
    agents: dict[str, list[str]],
    problems: list[str],
    n_seeds: int = N_SEEDS,
    ncols: int | None = None,
    height_to_width_ratio: float = 0.95,
    show_strategy: bool = False,
    agent_name_mapping: dict[str, str] | None = None,
    legend_ncols: int | None = None,
    filename: str | None = None,
    allow_partial_seeds: bool = True,
    drop_k_slowest: int = 5,
    appendix: list[str] | None = None,
    appendix_ncols: int | None = None,
    appendix_height_to_width_ratio: float | None = None,
    overlay_last: list[str] | None = None,
) -> None:
    """Plot cumulative maximum y values (mean +- 1SEM) for custom agent/strategy combos.

    If `appendix` is provided, a second figure is drawn with `problems + appendix`
    and saved as `{filename}_appendix.pdf`. Useful for producing a concise main-paper
    figure plus a fuller appendix variant in a single call.

    Args:
        agents: Mapping from agent name to list of acqf_opt strategies.
            e.g. {"vanilla_bo": ["default", "botorch_raasp", "no_raasp"], "turbo": ["default"]}
        problems: Problems shown in the main figure.
        n_seeds: Number of seeds to consider.
        ncols: Number of columns in the subplot grid. Defaults to len(problems) (single row).
        height_to_width_ratio: Height-to-width ratio of each subplot.
        show_strategy: Whether to include the strategy name in the legend label.
        agent_name_mapping: Optional overrides for agent display names.
            Only needs to contain the names you want to change.
        legend_ncols: Number of columns in the legend. Defaults to one row (all entries).
        filename: Optional filename (without extension) to save the figure as PDF under SAVEFIG_DIR.
        allow_partial_seeds: If True, plot lines even when some seed files are missing
            (uses only the available seeds). If False, skip any agent/strategy/problem
            combination missing seeds.
        drop_k_slowest: For ei_bo, drop the k seeds with the fewest completed iterations
            before computing the common truncation point. Ignored for other agents.
        appendix: Extra problems appended to `problems` to produce a second
            (fuller) figure. No second figure is drawn if None or empty.
        appendix_ncols: `ncols` override for the appendix figure. Defaults to `ncols`.
        appendix_height_to_width_ratio: `height_to_width_ratio` override for the
            appendix figure. Defaults to `height_to_width_ratio`.
        overlay_last: Agent names that should be rendered on top of all other lines
            (via higher zorder). Legend order is unaffected.
    """
    _plot_cum_max_y_single(
        agents=agents,
        problems=problems,
        n_seeds=n_seeds,
        ncols=ncols,
        height_to_width_ratio=height_to_width_ratio,
        show_strategy=show_strategy,
        agent_name_mapping=agent_name_mapping,
        legend_ncols=legend_ncols,
        filename=filename,
        allow_partial_seeds=allow_partial_seeds,
        drop_k_slowest=drop_k_slowest,
        overlay_last=overlay_last,
    )
    if appendix:
        _plot_cum_max_y_single(
            agents=agents,
            problems=list(problems) + list(appendix),
            n_seeds=n_seeds,
            ncols=appendix_ncols if appendix_ncols is not None else ncols,
            height_to_width_ratio=(
                appendix_height_to_width_ratio
                if appendix_height_to_width_ratio is not None
                else height_to_width_ratio
            ),
            show_strategy=show_strategy,
            agent_name_mapping=agent_name_mapping,
            legend_ncols=legend_ncols,
            filename=f"{filename}_appendix" if filename is not None else None,
            allow_partial_seeds=allow_partial_seeds,
            drop_k_slowest=drop_k_slowest,
            overlay_last=overlay_last,
        )
