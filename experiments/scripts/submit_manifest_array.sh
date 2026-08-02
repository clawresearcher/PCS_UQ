#!/bin/bash
set -euo pipefail
#SBATCH --job-name=pcs_uq_%A_%a
#SBATCH --output=logs/manifest-%A_%a.out
#SBATCH --error=logs/manifest-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=15G

: "${PCS_UQ_TASKS:?Set PCS_UQ_TASKS to a frozen inventory CSV}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"

mkdir -p logs
python -m experiments.scripts.run_manifest_task \
  --inventory "$PCS_UQ_TASKS" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --results-root "${PCS_UQ_RESULTS_ROOT:-experiments/results/reproduction_main}"
