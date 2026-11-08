#!/bin/bash
#SBATCH --job-name=irr-bench
#SBATCH --partition=P2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-35%12
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/logs/benchmark-%A_%a.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/jobs/env.sh
CELLS=(t1_n200 t1_n500 t1_n1000 t2_J1_b0 t2_J1_b25 t2_J9_b0 t2_J9_b25 bridge_r128 bridge_r512)
t=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((t / 4))]}
from=$(((t % 4) * 25))
to=$((from + 24))
out="$W/results/benchmark/$cell/seeds-$(printf %02d "$from")-$(printf %02d "$to")"
echo "node=$SLURMD_NODENAME job=$SLURM_ARRAY_JOB_ID task=$t cell=$cell seeds=$from-$to start=$(date -Is)"
"$PY" "$W/drivers/benchmark_grid.py" --code "$CODE" --cell "$cell" --seed-from "$from" --seed-to "$to" --out "$out"
echo "end=$(date -Is)"
