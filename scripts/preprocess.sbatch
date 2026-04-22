#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

# Usage:
#   sbatch scripts/preprocess.sbatch <dataset> [extra pipeline flags]
#
# Examples:
#   sbatch scripts/preprocess.sbatch mydataset
#   sbatch scripts/preprocess.sbatch mydataset --synthseg
#   sbatch scripts/preprocess.sbatch mydataset --subject sub-001

set -euo pipefail

DATASET="${1:?Usage: sbatch scripts/preprocess.sbatch <dataset> [extra pipeline flags]}"
shift

mkdir -p slurms
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

N_WORKERS="${SLURM_CPUS_PER_TASK:-8}"

echo "Dataset:   $DATASET"
echo "Workers:   $N_WORKERS"
echo "GPUs:      ${SLURM_GPUS_PER_TASK:-0}"
echo "Extra args: $*"
echo "==============================="

uv run --extra preprocessing preprocessing/pipeline.py \
    --dataset "$DATASET" \
    --n_workers "$N_WORKERS" \
    "$@"
