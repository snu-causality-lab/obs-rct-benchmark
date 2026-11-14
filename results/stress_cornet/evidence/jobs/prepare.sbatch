#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-stress-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-stress-20261114-r01/logs/prepare-%j.out
# Usage: RUN_DIR=<run> CODE_ID=<commit> GROUP=stress sbatch --export=ALL jobs/prepare.sbatch
source jobs/common.sh
set -e
"$PY" -m pip freeze > "environment-freeze-$CODE_ID.txt"
cd "$CODE" && "$PY" -m pytest -q -p no:cacheprovider tests/test_stress.py tests/test_rct_selected.py tests/test_methods.py tests/test_runner.py 2>&1 | tail -3
"$PY" scripts/check_release.py | tail -1
cd "$RUN"
[ -e shards ] && { echo "shards exist; refusing"; exit 1; }
"$PY" "$CODE/scripts/make_stress_configs.py" --config "$CODE/configs/v1_stress_cornet.toml" --output shards
sha256sum "$CODE/configs/v1_stress_cornet.toml" shards/study_manifest.json | tee shards/SHA256SUMS
"$PY" jobs/run_shard.py shards 0 check-out --check-only
echo PREPARE_OK
