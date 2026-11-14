#!/bin/bash
#SBATCH --job-name=v1core-prepare
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01/logs/prepare-%j.out
# Usage: CODE_ID=<commit> sbatch --export=ALL jobs/prepare.sbatch
source jobs/common.sh
set -e
"$PY" -c "import obs_rct_benchmark, sys; assert obs_rct_benchmark.__file__.startswith('$CODE/src'), obs_rct_benchmark.__file__; print(obs_rct_benchmark.__file__)"
"$PY" -m pip freeze > "environment-freeze-$CODE_ID.txt"
cd "$CODE"
"$PY" -m pytest -q -p no:cacheprovider tests/test_rct_selected.py tests/test_v1_core_analysis.py \
    tests/test_methods.py tests/test_runner.py tests/test_cornet.py tests/test_catalog_study.py 2>&1 | tail -3
"$PY" scripts/check_release.py | tail -1
cd "$RUN"
[ -e shards ] && { echo "shards exist; refusing to overwrite"; exit 1; }
"$PY" "$CODE/scripts/make_v1_core_configs.py" --config "$CODE/configs/v1_core_study.toml" --output shards
sha256sum "$CODE/configs/v1_core_study.toml" shards/study_manifest.json | tee shards/SHA256SUMS
echo PREPARE_OK
