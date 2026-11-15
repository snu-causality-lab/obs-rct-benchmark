#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/write-shards-%j.out
# Usage: CODE_ID=<id> sbatch --export=ALL jobs/write_shards.sbatch <prefix> [--preflight]
GROUP=native_r source jobs/common.sh
set -e
for g in ${GROUPS_TO_WRITE:-native_r gp tabpfn tsr_legacy}; do
  [ -e "$1-$g" ] && { echo "$1-$g exists; refusing"; exit 1; }
  "$PY" "$CODE/scripts/make_v1_wave2_configs.py" --config "$CODE/configs/v1_wave2_$g.toml" --output "$1-$g" ${2:-}
done
sha256sum "$1"-*/study_manifest.json
