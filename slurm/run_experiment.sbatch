#!/bin/bash
#SBATCH --job-name=run_experiment
#SBATCH --nodes=1

# Unified experiment script for CPU and GPU jobs. All resource parameters
# (--mem, --time, --partition, --gres, --array) are set by submit.py at
# submission time, not hardcoded here.
#
# Required environment variables (set via submit.py --export):
#   PROBLEM            - problem name (e.g., "lasso-dna")
#   AGENT              - agent name (e.g., "vanilla_bo")
#   ACQF_OPT_STRATEGY  - acquisition function optimization strategy (e.g., "default")
#   DEVICE             - "cpu" or "cuda"
#
# Optional environment variables:
#   N_EVALS            - total number of evaluations (overrides problem default)
#   BATCH_SIZE         - batch size for candidate generation (default: 1)
#   N_INIT             - number of initial points (overrides problem default)
#   NO_WANDB           - if non-empty, disable WandB logging

source "$(pwd)/slurm/preamble.sh"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [ -z "$PROBLEM" ] || [ -z "$AGENT" ] || [ -z "$ACQF_OPT_STRATEGY" ] || [ -z "$DEVICE" ]; then
    echo "Error: PROBLEM, AGENT, ACQF_OPT_STRATEGY, and DEVICE must be set."
    echo "Use submit.py to launch experiments."
    exit 1
fi

REPO=$(pwd)

# Enable GPU passthrough into the container when running on CUDA
NV_FLAG=""
if [ "$DEVICE" = "cuda" ]; then
    NV_FLAG="--nv"
fi

# Synthetic problems are evaluated in-process and don't need the Bencher server.
case "$PROBLEM" in
    ackley_*|michalewicz_*|styblinski_tang_*)
        IS_SYNTHETIC=1
        ;;
    *)
        IS_SYNTHETIC=0
        ;;
esac

# A socket is addressed by filesystem path, so putting the job id in the path makes
# collisions impossible by construction. Requires an image with bencherscaffold >= 0.5.2.
BENCHER_IMAGE="${BENCHER_IMAGE:-images/bencher.sif}"

# bencher is nine processes: the client-facing router plus eight benchmark services.
BENCHER_SERVICES=(SERVER LASSO NODEP MAXSAT EBO MUJOCO SVM IOH BO4MOB)

# A bencher older than scaffold 0.5.2 silently ignores BENCHER_*_HOST and falls back to
# TCP ports, so the sockets never appear at all. Without a limit the job would wait out
# its full walltime and produce nothing.
BENCHER_SOCKET_TIMEOUT=600   # seconds for all nine sockets to appear
BENCHER_ACCEPT_TIMEOUT=120   # seconds for the router to start accepting

if [ "$IS_SYNTHETIC" -eq 0 ]; then
    BENCHER_SOCKET_DIR="${TMPDIR:-/tmp}/bencher-$SLURM_JOB_ID"

    # AF_UNIX caps the whole path at 108 bytes (sun_path).
    if [ $(( ${#BENCHER_SOCKET_DIR} + 16 )) -ge 108 ]; then
        echo "Error: socket dir '$BENCHER_SOCKET_DIR' is too long for AF_UNIX (108-byte limit)."
        echo "Set TMPDIR to something shorter."
        exit 1
    fi

    mkdir -p "$BENCHER_SOCKET_DIR"

    # Not every filesystem supports AF_UNIX and $TMPDIR is not guaranteed to be
    # node-local storage.
    probe_sock="$BENCHER_SOCKET_DIR/.probe.sock"
    nc -lU "$probe_sock" >/dev/null 2>&1 &
    probe_pid=$!
    probe_ok=0
    for _ in $(seq 1 10); do
        if nc -U -z "$probe_sock" 2>/dev/null; then probe_ok=1; break; fi
        sleep 0.2
    done
    kill "$probe_pid" 2>/dev/null
    rm -f "$probe_sock"
    if [ "$probe_ok" -ne 1 ]; then
        echo "Error: cannot create a unix socket in $BENCHER_SOCKET_DIR."
        echo "That filesystem may not support AF_UNIX."
        echo "Set TMPDIR to node-local storage."
        rm -rf "$BENCHER_SOCKET_DIR"
        exit 1
    fi


    # Build singularity's --env list: a socket per service, named from the variable stem.
    BENCHER_ENV=""
    for svc in "${BENCHER_SERVICES[@]}"; do
        sock="$BENCHER_SOCKET_DIR/$(echo "$svc" | tr '[:upper:]' '[:lower:]').sock"
        BENCHER_ENV="${BENCHER_ENV:+$BENCHER_ENV,}BENCHER_${svc}_HOST=unix:$sock"
        if [ "$svc" = SERVER ]; then BENCHER_ROUTER_SOCK="$sock"; fi
    done

    BENCHER_ADDRESS="unix:$BENCHER_ROUTER_SOCK"
    echo "Bencher sockets: $BENCHER_SOCKET_DIR"

    # Start bencher as a background sidecar.
    singularity run --bind "$BENCHER_SOCKET_DIR" --env "$BENCHER_ENV" "$BENCHER_IMAGE" &
    BENCHER_PID=$!
    trap 'kill $BENCHER_PID 2>/dev/null; rm -rf "$BENCHER_SOCKET_DIR"; exit' TERM
    trap 'kill $BENCHER_PID 2>/dev/null; rm -rf "$BENCHER_SOCKET_DIR"' EXIT

    # Wait for Bencher services to be ready.
    echo "Starting up bencher server"
    n_sock=0
    for _ in $(seq 1 $BENCHER_SOCKET_TIMEOUT); do
        n_sock=$(( $(find "$BENCHER_SOCKET_DIR" -type s -name '*.sock' | wc -l) ))
        kill -0 "$BENCHER_PID" 2>/dev/null || { echo "bencher sidecar died"; exit 1; }
        if [ "$n_sock" -eq ${#BENCHER_SERVICES[@]} ]; then break; fi
        sleep 1
    done
    if [ "$n_sock" -ne ${#BENCHER_SERVICES[@]} ]; then
        echo "Error: only $n_sock/${#BENCHER_SERVICES[@]} bencher sockets appeared in ${BENCHER_SOCKET_TIMEOUT}s."
        echo "Is $BENCHER_IMAGE built with bencherscaffold >= 0.5.2? Older images ignore"
        echo "BENCHER_*_HOST and bind TCP ports instead, so no socket is ever created."
        exit 1
    fi

    # Check whether connect() is successful.
    router_ready=0
    for _ in $(seq 1 $BENCHER_ACCEPT_TIMEOUT); do
        if nc -U -z "$BENCHER_ROUTER_SOCK" 2>/dev/null; then router_ready=1; break; fi
        sleep 1
    done
    if [ "$router_ready" -ne 1 ]; then
        echo "Error: bencher router bound $BENCHER_ROUTER_SOCK but never accepted a connection in ${BENCHER_ACCEPT_TIMEOUT}s."
        exit 1
    fi
    echo "Bencher server started"
else
    echo "Synthetic problem '$PROBLEM' — skipping Bencher server startup"
    BENCHER_SOCKET_DIR="${TMPDIR:-/tmp}"   # nothing is bound; keeps the bind flag valid
    BENCHER_ADDRESS="127.0.0.1"            # unused, but the runner still takes an address
fi

# Run the experiment (SLURM_ARRAY_TASK_ID indexes into EXPERIMENT_SEEDS)
echo "Running: problem=$PROBLEM, agent=$AGENT, acqf_opt_strategy=$ACQF_OPT_STRATEGY, device=$DEVICE, seed_idx=$SLURM_ARRAY_TASK_ID"
singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --bind "${TMPDIR:-/tmp}" \
    $NV_FLAG \
    --env TMPDIR=${TMPDIR:-/tmp} \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env PYTHONPATH=/workspace/bo-arena \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        /opt/venv/bin/python scripts/experiment_runner.py \
            --problem $PROBLEM \
            --agent $AGENT \
            --acqf-opt-strategy $ACQF_OPT_STRATEGY \
            --seed-idx $SLURM_ARRAY_TASK_ID \
            --address $BENCHER_ADDRESS \
            --device $DEVICE \
            --dtype float64 \
            ${N_EVALS:+--n-evals $N_EVALS} \
            ${BATCH_SIZE:+--batch-size $BATCH_SIZE} \
            ${N_INIT:+--n-init $N_INIT} \
            ${NO_WANDB:+--no-wandb}
    "
