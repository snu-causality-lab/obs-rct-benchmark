#!/bin/bash
#SBATCH --job-name=v1core-analysis
#SBATCH --partition=P2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-core-20261114-r01/logs/analysis-%j.out
# Usage: CODE_ID=<commit> sbatch --export=ALL jobs/analysis.sbatch <outputs dir name> <analysis dir name>
source jobs/common.sh
OUT=${1:?outputs dir required}; AN=${2:?analysis dir required}
mkdir -p "checks-$AN"
"$PY" "$CODE/scripts/analyze_v1_core.py" --config "$CODE/configs/v1_core_study.toml" --runs "$OUT"/group-* --output "$AN"; a=$?
"$PY" "$CODE/scripts/verify_v1_core.py" arithmetic --analysis "$AN" --config "$CODE/configs/v1_core_study.toml" --runs "$OUT"/group-* --output "checks-$AN/arithmetic.json"; b=$?
"$PY" "$CODE/scripts/verify_v1_core.py" selector --config "$CODE/configs/v1_core_study.toml" --runs "$OUT"/group-* --output "checks-$AN/selector.json"; c=$?
"$PY" "$CODE/scripts/verify_v1_core.py" reproduce --config "$CODE/configs/v1_core_study.toml" --runs "$OUT"/group-* --output "checks-$AN/reproduce.json"; d=$?
echo "analyze=$a arithmetic=$b selector=$c reproduce=$d"
exit $(( a || b || c || d ))
