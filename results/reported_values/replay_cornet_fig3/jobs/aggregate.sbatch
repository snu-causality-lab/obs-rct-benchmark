#!/bin/bash
#SBATCH --job-name=cnr-aggregate
#SBATCH --partition=P2
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/logs/aggregate-%j.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/jobs/env.sh
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date -Is)"
"$PY" "$W/drivers/aggregate.py" --results "$W/results" --reported "$W/inputs/reported_values.csv" --out "$W/results/analysis"
echo "end=$(date -Is)"
