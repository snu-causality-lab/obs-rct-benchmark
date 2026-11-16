#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
# One condition of a block's reproducibility check per array task (jobs/reproduce_part.py), with the
# block's frozen code; then one merge job. Runs are the active shard directories only.
# Usage: CODE_ID=<freeze> RUN_DIR=<run> sbatch --export=ALL --chdir=<run> --output=<run>/logs/%x-%A_%a.out \
#   --array=0-<n-1> -J icmsel-repro-<block>-part jobs/reproduce_part.sbatch core|stress <tag>
# then (afterany): ... -J icmsel-repro-<block>-merge jobs/reproduce_part.sbatch core|stress <tag> merge
set -uo pipefail
BLOCK=${1:?core|stress}; TAG=${2:?tag}; STEP=${3:-run}
case "$BLOCK" in core) P=out-core; N=1984 ;; stress) P=out-stress; N=1024 ;; *) echo "unknown block $BLOCK"; exit 2 ;; esac
GROUP=gp source jobs/common.sh
CFG=$CODE/configs/v1_icm_selected_$BLOCK.toml
LIMIT=(); [ -f "joblimit-$BLOCK.json" ] && LIMIT=(--job-limit-shards "joblimit-$BLOCK.json")
PARTS=checks-$BLOCK-$TAG-reproduce-parts
if [ "$STEP" = merge ]; then
  "$PY" jobs/reproduce_part.py merge --code "$CODE" --config "$CFG" --parts "$PARTS"/part-*.json \
    --output "checks-$BLOCK-$TAG-reproduce.json" > "logs/checks-$BLOCK-$TAG-reproduce.stdout" 2>&1
  rc=$?; echo "merge exit $rc"; exit $rc
fi
mapfile -t RUNS < <(find "$P" -mindepth 1 -maxdepth 1 -type d ! -name '*.*' | sort)
[ "${#RUNS[@]}" -eq "$N" ] || { echo "$P: ${#RUNS[@]} active shard directories, expected $N"; exit 3; }
mapfile -t KEYS < <("$PY" -c "import sys; sys.path.insert(0, 'jobs'); import reproduce_part as R
from obs_rct_benchmark.runner import load_config; print('\n'.join(R.selected(load_config(sys.argv[1]))))" "$CFG")
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]:?no condition for this task}
I=$(printf %02d "$SLURM_ARRAY_TASK_ID")
mkdir -p "$PARTS"
"$PY" jobs/reproduce_part.py run --code "$CODE" --config "$CFG" --scenario "$KEY" --runs "${RUNS[@]}" \
  --output "$PARTS/part-$I.json" "${LIMIT[@]}" > "logs/$PARTS-$I.stdout" 2>&1
rc=$?; echo "$KEY exit $rc"; exit $rc
