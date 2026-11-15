#!/bin/bash
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/%x-%A_%a.out
# Usage: CODE_ID=<id> GROUP=tabpfn sbatch --export=ALL --array=0-N --time=.. -J <name> \
#   jobs/gpu_array.sbatch <shards dir> <outputs dir> <index offset> <total shards> [--discard-metrics]
# Each task runs PER_GPU (default 4) consecutive shards concurrently on one GPU, one CPU thread each.
# With INDEX_FILE set, positions index its lines (one shard index per line) instead of the shards.
source jobs/common.sh
PER=${PER_GPU:-4}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=(); idx=()
for k in $(seq 0 $(( PER - 1 ))); do
  p=$(( ${3:?offset} + SLURM_ARRAY_TASK_ID * PER + k ))
  [ "$p" -lt "${4:?total}" ] || continue
  i=$(shard_index "$p")
  "$PY" jobs/run_shard.py "$1" "$i" "$2" ${5:-} >> "$RUN/logs/${SLURM_JOB_NAME}-shard-$i.log" 2>&1 &
  pids+=($!); idx+=($i)
done
rc=0
for n in "${!pids[@]}"; do wait "${pids[$n]}" || { echo "shard ${idx[$n]} failed"; rc=1; }; done
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
exit $rc
