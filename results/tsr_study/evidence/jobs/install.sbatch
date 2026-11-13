#!/bin/bash
#SBATCH --job-name=tsr-install
#SBATCH --partition=P2
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/install-%j.out
# Usage: sbatch jobs/install.sbatch <code-dir-name>
set -euo pipefail
source jobs/common.sh
CODE=$RUN/code/$1
"$PY" -m pip install --quiet --no-deps --force-reinstall "$CODE"
"$PY" -m pip freeze > "environment-v1-freeze-$1.txt"
cd "$CODE"
"$PY" -m pytest -q -p no:cacheprovider tests/test_tsr.py tests/test_tsr_analysis.py tests/test_methods.py tests/test_runner.py 2>&1 | tail -3
"$PY" scripts/check_release.py | tail -1
cd "$RUN"
[ -d shards ] && { echo "shards exist; refusing to overwrite"; exit 1; }
"$PY" "$CODE/scripts/make_reference_configs.py" --config "$CODE/configs/tsr_study.toml" --output shards --shards 64
sha256sum "$CODE/configs/tsr_study.toml" shards/shard-00.toml | tee shards/SHA256SUMS
mkdir -p outputs
echo INSTALL_OK
