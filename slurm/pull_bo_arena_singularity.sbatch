#!/bin/bash
#SBATCH --job-name=pull_bo_arena_singularity
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=04:00:00

source "$(pwd)/slurm/preamble.sh"

# Set cache and tmp directories for Singularity
export SINGULARITY_CACHEDIR=$SCRATCH
export SINGULARITY_TMPDIR=$SCRATCH
export APPTAINER_CACHEDIR=$SCRATCH
export APPTAINER_TMPDIR=$SCRATCH

# Pull the bo-arena Singularity image
# Note - this image was built from official bo-arena repository at commit d28bbe77727b41c44ae3a464376441124f4fc3b1
mkdir -p images
singularity pull --force images/bo_arena.sif docker://tchristie/bo-arena@sha256:e1eb20f52187709dd3f325e103cf48805fd48b18449366c91d112bdf02a403a1
