#!/bin/bash
# Usage: slurm/submit.sh [--gpu] <script_name> [extra sbatch args...]
# Example: slurm/submit.sh sync_wandb.sbatch
#
# Jobs run on CPU_PARTITION by default. Pass --gpu to run on GPU_PARTITION with
# one GPU instead (e.g. on clusters where CPU nodes are rarely available).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/config.sh"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found."
    echo "Copy slurm/config.sh.example to slurm/config.sh and fill in your paths."
    exit 1
fi
set -a
source "$CONFIG"
set +a

RESOURCE_ARGS=(--partition="$CPU_PARTITION")
if [ "$1" = "--gpu" ]; then
    # Many clusters reject jobs on a GPU partition that don't request a GPU.
    RESOURCE_ARGS=(--partition="$GPU_PARTITION" --gres=gpu:1)
    shift
fi

if [ -z "$1" ]; then
    echo "Usage: slurm/submit.sh [--gpu] <script_name> [extra sbatch args...]"
    exit 1
fi

SCRIPT="$1"
shift

mkdir -p "$LOGS"

sbatch \
    --chdir="$REPO" \
    -o "$LOGS/%j_%x_%a.out" \
    "${RESOURCE_ARGS[@]}" \
    "$@" \
    "$SCRIPT_DIR/$SCRIPT"
