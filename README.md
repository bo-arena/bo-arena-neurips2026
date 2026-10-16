# bo-arena
Arena for simple, reliable, and efficient benchmarking of Bayesian optimisation algorithms. This repository contains the code used to run the experiments in the paper: `BO-Arena: An Evolving Benchmark for High-Dimensional Bayesian Optimisation`. For commands used to run the experiments for the paper, see the "Experiment Commands" section below. Note that this repository contains additional code for running ablations in the paper. As such, a separate repository will be released upon publication of the paper, containing only the state-of-the-art algorithm implementations used to generate results for our leaderboard. Contributions from the community should take place in this separate repository, whose sole purpose is to act as an evolving collection of state-of-the-art high-dimensional Bayesian optimisation algorithms.

### Contributing

Please refer to the [contributing guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

### Quickstart

In order to run a quick experiment with `bo-arena`, using similar code to the snippet presented in the appendix of our paper, follow these steps:

1. Install dependencies using `uv` (see the "Install Dependencies" section of the [contributing guidelines](CONTRIBUTING.md) for details).
2. Start up the `bencher` server by following the instructions in the `bencher` documentation [here](https://github.com/lpapenme/bencher). To summarise, you will need to first pull the docker image with:

    ```
    docker pull gaunab/bencher:latest
    ```

    Then, you can start the server with:

    ```
    docker run -p 50051:50051 --restart always -d gaunab/bencher:latest
    ```
3. Run the `quickstart` notebook in the `notebooks/` directory, which demonstrates how to run a simple experiment with `bo-arena` using the `bencher` server you just started. For instructions on converting between `.py` and `.ipynb` formats, see the "Notebooks" section of the [contributing guidelines](CONTRIBUTING.md).

### Running Experiments with bencher on a SLURM Cluster
The `slurm` directory contains SLURM batch scripts for building Singularity containers and running `bencher` and `bo-arena` experiments on a SLURM cluster. These scripts may require modifications to paths, resource specifications, and other parameters to fit your specific cluster setup.

Each experiment job is self-contained: it starts a `bencher` server as a sidecar process, runs the experiment, then shuts the server down on exit. This means no shared `bencher` server is needed, jobs scale independently, and there is no coordination between jobs. Each job gives its sidecar a private set of Unix domain sockets under `$TMPDIR/bencher-$SLURM_JOB_ID`, so jobs co-scheduled on the same node cannot reach each other's server. There are two containers involved per job:

1. **`bencher` container**: Runs the `bencher` server on the job's own sockets. Started as a background sidecar at the beginning of each experiment job and stopped automatically when the job exits.
2. **`bo-arena` experiment container**: Runs the `bo-arena` experiment, connecting to the `bencher` server over that job's router socket.

We use Unix sockets rather than fixed TCP ports to prevent different SLURM jobs from using each other's sidecars. TCP ports live in the node's network namespace, so jobs on the same node bind to the same TCP ports (50051, 50053-50060). This is problematic: when one of those job exits, its cleanup script kills the shared sidecar, and the other jobs that relied on that sidecar might fail. In contrast, sockets are addressed by filesystem path, so putting the job id in the path means different jobs can't share each other's sidecar, avoiding this problem.

> **⚠️ Warning**
>
> Important - The run_experiment.sbatch script creates socket files for communicating with bencher under the directory $TMPDIR/bencher-$SLURM_JOB_ID (falling back to /tmp/bencher-$SLURM_JOB_ID if $TMPDIR is not set). The script generally removes this directory (and the created sockets) once it has finished running, but some cases (e.g. SIGKILL) it will not. Ideally $TMPDIR should be a job-specific temporary directory, so it will get deleted once the job finishes running in any case (on many clusters this is the case). If not, be careful, as $TMPDIR could get cluttered with directories containing socket files.

In order to run experiments, follow these steps:
1. Copy `slurm/config.sh.example` to `slurm/config.sh` and fill in your
cluster-specific paths, partition names and WandB credentials (if running with WandB,
as is the default behaviour). This file is gitignored and must be created by each user.
2. Obtain the Singularity images, which are expected to be in an `images/` directory.
    - **`bencher`**: build with `sbatch build_bencher_singularity.sbatch` or just pull the published image (easier than having to build it yourself). The socket support above requires `bencherscaffold >= 0.5.2` (previous versions bind to TCP ports instead of Unix sockets), which the latest published image satisfies:
      ```
      singularity pull images/bencher.sif docker://gaunab/bencher:latest
      ```
      Override the path with the `BENCHER_IMAGE` environment variable if you keep it elsewhere.
    - **`bo-arena`**: build with `sbatch build_bo_arena_singularity.sbatch`. Note that this needs `--fakeroot`; on clusters without such access, build the image elsewhere and copy it in.

   The exact images used for running the experiments in the paper will be released upon publication.
3. Run experiments with `submit.py`. Commands used to run the experiments can be found below. Each job starts its own `bencher` sidecar automatically — no need to start a separate `bencher` job first. Note that `submit.py` performs operations such as setting time and memory limits, and deciding whether to run on CPU or GPU depending on the algorithm/task being run, so you may wish to edit this depending on your specific cluster setup. The main code for actually running the experiments, which is executed via `submit.py`, can be found in `scripts/experiment_runner.py`. The core logic is very similar to the code snippet provided in our paper, with additional code added for performing tasks such as (optionally) initialising a WandB run, setting up a logger, ensuring the working tree is clean for reproducibility, and saving results.
4. (Optional, if using WandB) - Sync WandB results after experiments finish using `bash submit.sh sync_wandb.sbatch`. Experiments log to WandB in offline mode to avoid API rate limits with many concurrent jobs.

Note that if you wish to run experiments without using the SLURM scripts, or building the Singularity image for `bo-arena`, you can do so by creating a virtual environment with the dependencies specified in `uv.lock` (which were used for running experiments in the paper), and running the code in `scripts/experiment_runner.py` directly, ensuring you have a `bencher` server running and accessible at `localhost:50051`. The correct arguments to provide to `experiment_runner.py` can be inferred from the "Experiment Commands" section below.

### Experiment Commands

Below are the commands used to run the experiments in the paper:

1. Main results (Figure 3). For this plot, we ran algorithms using the default settings from their respective papers via the following command:
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent vanilla_bo,linear_bo,random,turbo \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    The bounce results are obtained via the following command (as the number of initial points used in the paper is 5, due to a low-dimensional subspace being used at the start):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent bounce \
    --acqf-opt-strategy default \
    --n-init 5 \
    --seed-range 0-19
    ```
    The SAASBO results are obtained via the following command, as we limited the total number of evaluations to 500 due to the expense of running the algorithm (and didn't run on mujoco-humanoid as it was too expensive to run with SAASBO):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant\
    --agent saasbo \
    --acqf-opt-strategy default \
    --n-evals 500 \
    --seed-range 0-19
    ```

2. Figure 1 (and Figure 8) results. The "Vanilla BO (Hvarfner et al., 2024)" and "Random" results were obtained using the command above. The "Vanilla BO (Yu et al., 2026)" results were obtained using the following command:
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent vanilla_bo_yu_2026 \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    The "BoTorch v0.11.3 (2024)" results were obtained using the following command:
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent historic_bo \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    The "EI-BO (Cheon et al., 2025)" results were obtained using the package referenced in the original paper, modified to use `bencher` and the same initial points as used in the rest of the experiments.

3. Synthetic problem results (Figure 4). These results were obtained with the following commands:
    ```
    python slurm/submit.py \
    --problem ackley_dim_100,michalewicz_dim_100,styblinski_tang_dim_100 \
    --agent vanilla_bo,linear_bo,random,turbo \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    And the results for bounce were obtained with:
    ```
    python slurm/submit.py \
    --problem ackley_dim_100,michalewicz_dim_100,styblinski_tang_dim_100 \
    --agent bounce \
    --acqf-opt-strategy default \
    --n-init 5 \
    --seed-range 0-19
    ```

4. EULBO comparison (Figure 5). The Vanilla BO (window 100) and Vanilla BO (window 100 with incumbent) results were obtained with the following commands:
    ```
    python slurm/submit.py \
    --problem mopta08,lasso-dna,svm \
    --agent window_vanilla_bo,best_window_vanilla_bo \
    --acqf-opt-strategy default \
    --batch-size 20 \
    --n-evals 20000 \
    --seed-range 0-19
    ```
    The EULBO results were obtained using the public repository of the original paper, modified to use `bencher` and the same initial points as used in the rest of the experiments.

5. RAASP ablation (Figures 6 and 9). These results were obtained with the following commands.

    TuRBO (RAASP):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent turbo \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    Vanilla BO (BoTorch RAASP):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent vanilla_bo \
    --acqf-opt-strategy botorch_raasp \
    --seed-range 0-19
    ```
    Vanilla BO and TuRBO (no RAASP):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent vanilla_bo,turbo \
    --acqf-opt-strategy no_raasp \
    --seed-range 0-19
    ```

6. Trust-regions ablation (Figures 7 and 10). These results were obtained with the following commands.

    Vanilla BO:
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent vanilla_bo \
    --acqf-opt-strategy botorch_raasp \
    --seed-range 0-19
    ```
    TuRBO:
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent turbo \
    --acqf-opt-strategy default \
    --seed-range 0-19
    ```
    TuRBO (DSP):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent turbo_dsp \
    --acqf-opt-strategy raasp \
    --seed-range 0-19
    ```
    TuRBO (LogEI):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent turbo_log_ei \
    --acqf-opt-strategy raasp \
    --seed-range 0-19
    ```
    TuRBO (LogEI + DSP):
    ```
    python slurm/submit.py \
    --problem rover,mopta08,lasso-dna,svm,mujoco-ant,mujoco-humanoid \
    --agent turbo_log_ei_dsp \
    --acqf-opt-strategy raasp \
    --seed-range 0-19
    ```

Code used for plotting the figures can be found in `notebooks/plotting.py`.
