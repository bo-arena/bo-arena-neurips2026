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
from bo_arena.agents import VanillaBO
from bo_arena.arena import Arena
from bo_arena.problems import BencherProblem

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# %%
# Initialise problem and agent
problem = BencherProblem(name="svm")
agent = VanillaBO(input_dim=problem.metadata.input_dim)

# Run the experiment
bo_arena = Arena(
    problem=problem,
    agent=agent,
    seed=43,
    batch_size=1,
    n_initial_points=30,
    n_total_evaluations=100,
    logger=logger,
)

# Result containing xs, ys and timings of key parts of the loop
result = bo_arena.run()
