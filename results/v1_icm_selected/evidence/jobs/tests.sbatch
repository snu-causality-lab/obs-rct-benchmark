#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01/logs/tests-%x-%j.out
# Usage: CODE_ID=<id> GROUP=<native_r|gp> RUN_DIR=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01 sbatch --export=ALL jobs/tests.sbatch <pytest args...>
source jobs/common.sh
cd "$CODE" && "$PY" -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -15
