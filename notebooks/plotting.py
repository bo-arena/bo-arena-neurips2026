# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from plotting_utils import plot_cum_max_y_custom

# %%
plot_cum_max_y_custom(
    agents={
        "random": ["default"],
        "vanilla_bo_yu_2026": ["default"],
        "historic_bo": ["default"],
        "vanilla_bo": ["default"],
        "ei_bo": ["default"],
    },
    problems=["mopta08", "lasso-dna", "svm"],
    legend_ncols=3,
    height_to_width_ratio=1.02,
    appendix=["rover", "mujoco-ant", "mujoco-humanoid"],
    appendix_ncols=3,
    appendix_height_to_width_ratio=0.83,
    agent_name_mapping={"vanilla_bo": "Vanilla BO (Hvarfner et al., 2024)"},
    filename="yu_2026",
)

# %%
plot_cum_max_y_custom(
    agents={
        "random": ["default"],
        "turbo": ["default"],
        "saasbo": ["default"],
        "bounce": ["default"],
        "vanilla_bo": ["default"],
        "linear_bo": ["default"],
    },
    problems=["rover", "mopta08", "lasso-dna", "svm", "mujoco-ant", "mujoco-humanoid"],
    ncols=3,
    height_to_width_ratio=0.83,
    filename="fig7_vanilla_bo",
    overlay_last=["saasbo"],
)

# %%
plot_cum_max_y_custom(
    agents={
        "turbo": ["raasp", "no_raasp"],
        "vanilla_bo": ["botorch_raasp", "no_raasp"],
    },
    problems=["mopta08", "lasso-dna", "svm"],
    show_strategy=True,
    legend_ncols=2,
    height_to_width_ratio=1.02,
    appendix=["rover", "mujoco-ant", "mujoco-humanoid"],
    appendix_ncols=3,
    appendix_height_to_width_ratio=0.87,
    filename="raasp_vs_no_raasp",
)

# %%
plot_cum_max_y_custom(
    agents={
        "vanilla_bo": ["botorch_raasp"],
        "turbo": ["raasp"],
        "turbo_dsp": ["raasp"],
        "turbo_log_ei": ["raasp"],
        "turbo_log_ei_dsp": ["raasp"],
    },
    problems=["mopta08", "lasso-dna", "svm"],
    appendix=["rover", "mujoco-ant", "mujoco-humanoid"],
    appendix_ncols=3,
    appendix_height_to_width_ratio=0.83,
    filename="turbo_logei_dsp",
)

# %%
plot_cum_max_y_custom(
    agents={"window_100_vanilla_bo": ["default"], "best_1_window_100_vanilla_bo": ["default"], "eulbo": ["default"]},
    problems=["mopta08", "lasso-dna", "svm"],
    filename="eulbo",
)

# %%
plot_cum_max_y_custom(
    agents={
        "random": ["default"],
        "turbo": ["default"],
        # "saasbo": ["default"],
        "bounce": ["default"],
        "vanilla_bo": ["default"],
        "linear_bo": ["default"],
    },
    problems=["bbob-griewankrosenbrock", "bbob-rastriginrotated", "bbob-schwefel"],
    filename="bbob",
)

# %%
plot_cum_max_y_custom(
    agents={
        "random": ["default"],
        "turbo": ["default"],
        # "saasbo": ["default"],
        "bounce": ["default"],
        "vanilla_bo": ["default"],
        "linear_bo": ["default"],
    },
    problems=["ackley_dim_100", "michalewicz_dim_100", "styblinski_tang_dim_100"],
    filename="sfu",
)
# %%
