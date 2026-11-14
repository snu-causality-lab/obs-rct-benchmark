#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-stress-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-stress-20261114-r01/logs/analysis-%j.out
# Usage: RUN_DIR=<run> CODE_ID=<commit> GROUP=stress sbatch --export=ALL jobs/analysis.sbatch <outputs> <analysis>
source jobs/common.sh
OUT=${1:?}; AN=${2:?}; C="$CODE/configs/v1_stress_cornet.toml"
mkdir -p "checks-$AN"
"$PY" "$CODE/scripts/analyze_stress.py" --config "$C" --runs "$OUT"/stress-seed-* --output "$AN"; a=$?
"$PY" "$CODE/scripts/verify_stress.py" targets --config "$C" --output "checks-$AN/targets.json"; b=$?
"$PY" "$CODE/scripts/verify_stress.py" arithmetic --config "$C" --analysis "$AN" --runs "$OUT"/stress-seed-* --output "checks-$AN/arithmetic.json"; c=$?
"$PY" "$CODE/scripts/verify_stress.py" reproduce --config "$C" --runs "$OUT"/stress-seed-* --output "checks-$AN/reproduce.json"; d=$?
echo "analyze=$a targets=$b arithmetic=$c reproduce=$d"
exit $(( a || b || c || d ))
