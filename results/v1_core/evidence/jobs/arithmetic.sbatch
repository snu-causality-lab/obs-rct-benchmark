#!/bin/bash
#SBATCH --job-name=v1core-arith
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01/logs/arithmetic-%j.out
# Rerun of the arithmetic check only: job 819601 called it without --config/--runs.
# Usage: CODE_ID=<commit> sbatch --export=ALL jobs/arithmetic.sbatch <outputs dir> <analysis dir>
source jobs/common.sh
OUT=${1:?}; AN=${2:?}
"$PY" "$CODE/scripts/verify_v1_core.py" arithmetic --analysis "$AN" --config "$CODE/configs/v1_core_study.toml" --runs "$OUT"/group-* --output "checks-$AN/arithmetic.json"
