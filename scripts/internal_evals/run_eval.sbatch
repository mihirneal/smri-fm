#!/usr/bin/env bash
#SBATCH --job-name=smri_fm_internal_eval
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=slurms/%x-%j.out
#SBATCH --error=slurms/%x-%j.err
#SBATCH --time=04:00:00


set -euo pipefail

echo "Setting up environment variables."
source scripts/internal_evals/load_env.sh
echo "Done setting up environment variables."

echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not_set}"
echo "Working directory: $(pwd)"

# Just specify your config
# uv run python -m evaluation.main --config <config_path>
# Example:
# uv run python -m evaluation.main --config src/evaluation/config/fomo_brain_age_gap_probe.yaml
uv run python -m evaluation.main --config src/evaluation/config/fomo_brain_age_gap_scikit_probe.yaml