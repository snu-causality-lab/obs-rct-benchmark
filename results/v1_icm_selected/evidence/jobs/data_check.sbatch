#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
# Data check of the selected-rho addendum: regenerate every condition at each block's first seed with this
# addendum's code in the gp environment and compare every sample role with wave 2's gp dump.
# Usage: CODE_ID=<id> RUN_DIR=<run> sbatch --export=ALL --chdir=<run> --output=<run>/logs/%x-%j.out jobs/data_check.sbatch <output dir>
set -uo pipefail
GROUP=gp source jobs/common.sh
OUT=${1:?output dir}
mkdir -p "$OUT" && [ -z "$(ls -A "$OUT")" ] || { echo "$OUT is not empty"; exit 1; }
W2=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
rc=0
for b in core:70000:data-check-wave2 stress:80000:data-check-stress; do
  IFS=: read -r block seed ref <<< "$b"
  "$PY" "$CODE/scripts/check_data_identity.py" dump --config "$CODE/configs/v1_icm_selected_$block.toml" \
    --seed "$seed" --output "$OUT/data-$block.npz" || exit 1
  "$PY" "$CODE/scripts/check_data_identity.py" compare --reference "$W2/$ref/data-gp.npz" \
    --others "$OUT/data-$block.npz" --output "$OUT/data-check-$block.json"
  r=$?; echo "compare $block exit $r"; [ $r -eq 0 ] || rc=$r
done
sha256sum "$OUT"/* "$W2"/data-check-wave2/data-gp.npz "$W2"/data-check-stress/data-gp.npz
exit $rc
