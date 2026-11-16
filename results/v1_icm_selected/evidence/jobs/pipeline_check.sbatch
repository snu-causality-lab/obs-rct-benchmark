#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# CODE_ID=<id> GROUP=gp RUN_DIR=<run> sbatch --export=ALL --chdir=<run> --output=<run>/logs/%x-%j.out jobs/pipeline_check.sbatch
source jobs/common.sh
mkdir -p "$RUN/pipeline-check-$SLURM_JOB_ID"
"$PY" jobs/pipeline_check.py "$CODE" "$RUN/pipeline-check-$SLURM_JOB_ID"
