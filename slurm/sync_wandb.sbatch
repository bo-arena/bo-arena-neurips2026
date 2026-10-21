#!/bin/bash
#SBATCH --job-name=sync_wandb
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --time=24:00:00

# Syncs all unsynced offline WandB runs to the server. Only run this once all
# experiments have finished: a run synced while still in progress is marked as
# failed and as synced, so later syncs will skip it unless the INCLUDE_SYNCED=1
# flag is also passed (see below).
#
# Usage:
#   bash slurm/submit.sh sync_wandb.sbatch
#
# Optional environment variables:
#   INCLUDE_SYNCED     - if non-empty, also re-sync runs that are already marked
#                        as synced (e.g. to repair runs synced while in progress)
#
# e.g. bash slurm/submit.sh sync_wandb.sbatch --export=ALL,INCLUDE_SYNCED=1

source "$(pwd)/slurm/preamble.sh"

REPO=$(pwd)

singularity exec \
    --bind "$REPO:/workspace/bo-arena" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    images/bo_arena.sif \
    bash -c "
        cd /workspace/bo-arena
        /opt/venv/bin/python -m wandb sync --yes ${INCLUDE_SYNCED:+--include-synced} wandb/
    "
