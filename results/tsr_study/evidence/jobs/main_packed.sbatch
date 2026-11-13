#!/bin/bash
#SBATCH --job-name=tsr-main-packed
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/main-packed-%A_%a.out
# Usage: sbatch --array=0-4 --time=HH:MM:SS jobs/main_packed.sbatch <first-shard> <shards-per-task>
# Array task i runs shards first + i*per ... first + (i+1)*per - 1 concurrently on one GPU (one CPU thread each).
set -uo pipefail
source jobs/common.sh
FIRST=$1; PER=$2
"$PY" -c "import obs_rct_benchmark; print('package', obs_rct_benchmark.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=(); shards=()
for k in $(seq 0 $((PER - 1))); do
  s=$((FIRST + SLURM_ARRAY_TASK_ID * PER + k)); [ "$s" -le 63 ] || continue
  sh=$(printf "%02d" "$s")
  extra=""
  if [ -s "$RUN/outputs/shard-$sh/records.jsonl" ]; then
    extra="--resume"; echo "shard-$sh resumes"
  elif [ -e "$RUN/outputs/shard-$sh" ]; then
    mkdir -p "$RUN/preserved/empty-starts"; mv "$RUN/outputs/shard-$sh" "$RUN/preserved/empty-starts/shard-$sh-$(date +%s)"
    echo "shard-$sh had no records; empty start preserved, running fresh"
  fi
  "$PY" -m obs_rct_benchmark.cli run --config "$RUN/shards/shard-$sh.toml" --output "$RUN/outputs/shard-$sh" $extra \
    >> "$RUN/logs/shard-$sh.log" 2>&1 &
  pids+=($!); shards+=("$sh")
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; code=$?
  echo "shard-${shards[$i]} exit $code"
  [ "$code" -eq 0 ] || rc=1
done
exit $rc
