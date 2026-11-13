#!/bin/bash
#SBATCH --job-name=tsr-main
#SBATCH --partition=P2
#SBATCH --gres=gpu:1
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/main-%A_%a.out
# Usage: sbatch --array=0-15 --time=HH:MM:SS jobs/main.sbatch <code-dir-name> <outputs-dir-name>
# Each array task runs four seed shards concurrently on one GPU (one CPU thread each).
set -uo pipefail
source jobs/common.sh
OUT=$RUN/${2:?outputs dir required}
mkdir -p "$OUT" "$RUN/logs/$2"
"$PY" -c "import obs_rct_benchmark, importlib.metadata as m; print('package', obs_rct_benchmark.__file__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=(); shards=()
for k in 0 1 2 3; do
  s=$((SLURM_ARRAY_TASK_ID * 4 + k)); sh=$(printf "%02d" "$s")
  "$PY" -m obs_rct_benchmark.cli run --config "$RUN/shards/shard-$sh.toml" --output "$OUT/shard-$sh" \
    > "$RUN/logs/$2/shard-$sh.log" 2>&1 &
  pids+=($!); shards+=("$sh")
done
rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; code=$?
  echo "shard-${shards[$i]} exit $code"
  [ "$code" -eq 0 ] || rc=1
done
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true
exit $rc
