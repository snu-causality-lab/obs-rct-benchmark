#!/bin/bash
#SBATCH --job-name=irr-author
#SBATCH --partition=P2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-27%12
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/logs/author-%A_%a.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/jobs/env.sh
CELLS=(t1_n200 t1_n500 t1_n1000 t2_J1_b0 t2_J1_b25 t2_J9_b0 t2_J9_b25)
t=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((t / 4))]}
from=$(((t % 4) * 25))
to=$((from + 24))
out="$W/results/author/$cell/seeds-$(printf %02d "$from")-$(printf %02d "$to")"
echo "node=$SLURMD_NODENAME job=$SLURM_ARRAY_JOB_ID task=$t cell=$cell seeds=$from-$to start=$(date -Is)"
mkdir -p "$out"
sha256sum "$INTEGRATIVE_R_SOURCE/R/utils.R" "$INTEGRATIVE_R_SOURCE/R/intRlearner.R" > "$out/sources.sha256"
"$RSCRIPT" --vanilla "$W/drivers/author_replay.R" "$INTEGRATIVE_R_SOURCE" "$out" "$cell" "$from" "$to"
echo "end=$(date -Is)"
