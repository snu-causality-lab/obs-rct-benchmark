#!/bin/bash
#SBATCH --job-name=v1core-main
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-127%64
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01/logs/main-%A_%a.out
# Usage: CODE_ID=<commit> sbatch --export=ALL jobs/main.sbatch <outputs dir name>
source jobs/common.sh
OUT=${1:?outputs dir required}
/usr/bin/time -v "$PY" jobs/run_shard.py "$SLURM_ARRAY_TASK_ID" "$OUT"
