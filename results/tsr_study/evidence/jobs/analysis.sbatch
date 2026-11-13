#!/bin/bash
#SBATCH --job-name=tsr-analysis
#SBATCH --partition=P2
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-tsr-study-20261113-r01/logs/analysis-%j.out
# Usage: sbatch jobs/analysis.sbatch <code-dir-name> <analysis-dir-name> <outputs-dir-name>
set -euo pipefail
source jobs/common.sh
CODE=$RUN/code/$1
cd "$CODE"
sha256sum scripts/analyze_tsr_study.py
"$PY" scripts/analyze_tsr_study.py --config configs/tsr_study.toml --runs "$RUN"/${3:?outputs dir required}/shard-* \
  --catalog results/catalog_study --output "$RUN/$2"
echo ANALYSIS_DONE
