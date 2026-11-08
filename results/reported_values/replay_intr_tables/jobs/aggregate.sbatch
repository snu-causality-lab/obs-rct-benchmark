#!/bin/bash
#SBATCH --job-name=irr-aggregate
#SBATCH --partition=P2
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/logs/aggregate-%j.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/jobs/env.sh
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date -Is)"
"$PY" "$W/drivers/aggregate.py" --results "$W/results" --preflight "$W/preflight" \
  --reported "$W/inputs/reported_values.csv" --native "$W/inputs/native_kps_code_scalar_records.csv" \
  --out "$W/results/analysis"
echo "end=$(date -Is)"
