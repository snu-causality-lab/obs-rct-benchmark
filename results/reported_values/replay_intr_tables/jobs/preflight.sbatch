#!/bin/bash
#SBATCH --job-name=irr-preflight
#SBATCH --partition=P2
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/logs/preflight-%j.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-intr-replay-20261108-r01/jobs/env.sh
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date -Is)"
"$PY" "$W/drivers/preflight.py" --code "$CODE" --driver "$W/drivers/author_replay.R" --out "$W/preflight"
echo "end=$(date -Is)"
