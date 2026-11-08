#!/bin/bash
#SBATCH --job-name=cnr-preflight
#SBATCH --partition=P2
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/logs/preflight-%j.out
set -euo pipefail
source /shared/s2/lab05/sanghack/work/obsrct-cornet-replay-20261108-r02/jobs/env.sh
echo "node=$SLURMD_NODENAME job=$SLURM_JOB_ID start=$(date -Is)"
"$PY" "$W/drivers/preflight.py" --code "$CODE" --cornet "$CORNET" --stubs "$W/stubs"
echo "end=$(date -Is)"
