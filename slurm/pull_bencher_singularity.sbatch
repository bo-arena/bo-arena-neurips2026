#!/bin/bash
#SBATCH --job-name=pull_bencher_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=04:00:00

source "$(pwd)/slurm/preamble.sh"

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Pull the Singularity container
mkdir -p images
singularity pull images/bencher.sif docker://gaunab/bencher:latest
