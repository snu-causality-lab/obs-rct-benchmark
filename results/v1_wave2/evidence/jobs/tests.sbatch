#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/tests-%j.out
# Usage: CODE_ID=<id> sbatch --export=ALL jobs/tests.sbatch <pytest args...>
GROUP=native_r source jobs/common.sh
cd "$CODE" && "$PY" -m pytest -q -p no:cacheprovider "$@" 2>&1 | tail -15
