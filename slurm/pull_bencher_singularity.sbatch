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

# Pull the Singularity image
# Note - this image was built from bencher at commit ba9ada07a7f3eaa5efd691200ed4866ff4ca6e1b
mkdir -p images
singularity pull --force images/bencher.sif docker://tchristie/bencher@sha256:a7ca83f6d0b3810ed8dc03bb210aa35c46e65d5d19e4371aa7dbf038be3b6467
