#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/%x-%A_%a.out
# Usage: CODE_ID=<id> GROUP=<native_r|gp> sbatch --export=ALL --array=0-N --mem=.. --time=.. -J <name> \
#   jobs/cpu_array.sbatch <shards dir> <outputs dir> <index offset> [--discard-metrics]
# With INDEX_FILE set, positions index its lines (one shard index per line) instead of the shards.
source jobs/common.sh
i=$(shard_index $(( SLURM_ARRAY_TASK_ID + ${3:?offset} )))
/usr/bin/time -v "$PY" jobs/run_shard.py "$1" "$i" "$2" ${4:-}
