#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/data-check-%j.out
# Regenerate every condition at one seed in each group's environment and compare every
# sample role with the native_r environment, which is the wave-1 and stress-study-1 interpreter.
# Usage: CODE_ID=<id> sbatch --export=ALL jobs/data_check.sbatch <config prefix> <seed> <output dir>
set -uo pipefail
mkdir -p "$3" && [ -z "$(ls -A "$3")" ] || { echo "$3 is not empty"; exit 1; }
for g in native_r gp tabpfn tsr_legacy; do
  ( GROUP=$g; source jobs/common.sh; echo "== $g $PY"
    "$PY" "$CODE/scripts/check_data_identity.py" dump --config "$CODE/configs/$1_$g.toml" --seed "$2" --output "$3/data-$g.npz" ) || exit 1
done
( GROUP=native_r; source jobs/common.sh
  "$PY" "$CODE/scripts/check_data_identity.py" compare --reference "$3/data-native_r.npz" \
    --others "$3/data-gp.npz" "$3/data-tabpfn.npz" "$3/data-tsr_legacy.npz" --output "$3/data-check.json" )
echo "compare exit $?"
sha256sum "$3"/*
