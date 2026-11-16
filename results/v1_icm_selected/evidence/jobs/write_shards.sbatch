#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
# Usage: CODE_ID=<id> GROUP=gp RUN_DIR=<run> sbatch --export=ALL --chdir=<run> --output=<run>/logs/%x-%j.out \
#   -J <name> jobs/write_shards.sbatch <config, relative to the code snapshot> <output dir, relative to RUN> [--preflight]
source jobs/common.sh
cd "$RUN" || exit 2
"$PY" "$CODE/scripts/make_icm_selected_configs.py" --config "$CODE/$1" --output "$RUN/$2" ${3:-}
