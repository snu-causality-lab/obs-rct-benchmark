#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
source jobs/common.sh
"$PY" jobs/memprobe.py "$CODE" "$1" ${2:-}
