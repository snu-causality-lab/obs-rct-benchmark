#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01/logs/native-check-%j.out
# Usage: CODE_ID=<id> GROUP=gp RUN_DIR=/shared/s2/lab05/sanghack/work/obsrct-causal-icm-selected-20261115-r01 sbatch --export=ALL jobs/native_check.sbatch
source jobs/common.sh
echo "node $(hostname)"
"$PY" -m obs_rct_benchmark.cli run --config jobs/runner_check.toml --output check-run-$SLURM_JOB_ID && echo "runner ok"
"$PY" jobs/native_check.py jobs/check.toml
