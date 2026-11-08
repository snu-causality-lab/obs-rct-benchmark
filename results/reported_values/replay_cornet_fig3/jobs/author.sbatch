#!/bin/bash
#SBATCH --job-name=cnr-author
#SBATCH --partition=P2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-9%10
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/logs/author-%A_%a.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/jobs/env.sh
seed=$SLURM_ARRAY_TASK_ID
echo "node=$SLURMD_NODENAME job=$SLURM_ARRAY_JOB_ID task=$seed start=$(date -Is)"
"$PY" "$W/drivers/author_replay.py" --cornet "$CORNET" --stubs "$W/stubs" --seed "$seed" --out "$W/results/author/seed-$(printf %02d $seed)"
echo "end=$(date -Is)"
