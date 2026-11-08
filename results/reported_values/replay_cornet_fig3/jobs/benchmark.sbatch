#!/bin/bash
#SBATCH --job-name=cnr-bench
#SBATCH --partition=P2
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9%10
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/logs/benchmark-%A_%a.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/jobs/env.sh
ms=$((2025 + SLURM_ARRAY_TASK_ID))
echo "node=$SLURMD_NODENAME job=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID mechanism_seed=$ms start=$(date -Is)"
"$PY" "$W/drivers/benchmark_grid.py" --code "$CODE" --cornet "$CORNET" --mechanism-seed "$ms" --out "$W/results/benchmark/mech-$ms"
echo "end=$(date -Is)"
