#!/bin/bash
# Usage: slurm/submit.sh <script_name> [extra sbatch args...]
# Example: slurm/submit.sh sync_wandb.sbatch

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

SCRIPT="$1"
shift

if [[ "$SCRIPT" == *gpu* ]]; then
    PARTITION="$GPU_PARTITION"
else
    PARTITION="$CPU_PARTITION"
fi

mkdir -p "$LOGS"

sbatch \
    --chdir="$REPO" \
    -o "$LOGS/%j_%x_%a.out" \
    --partition="$PARTITION" \
    "$@" \
    "$SCRIPT_DIR/$SCRIPT"
