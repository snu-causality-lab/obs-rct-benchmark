#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/%x-%j.out
# Reproducibility check for one group of one study, in that group's environment, with the
# study's frozen code. GPU groups are submitted with --gres=gpu:1 --cpus-per-task=4 --mem=16G.
# Usage: sbatch -J repro-<study>-<group> --time=.. jobs/reproduce.sbatch w2|sw2 <group> <tag>
set -uo pipefail
STUDY=${1:?w2|sw2}; G=${2:?group}; TAG=${3:?tag}
case "$STUDY" in
  w2)  export CODE_ID=80c9129; P=out;  CFG=v1_wave2;        VF=verify_v1_wave2.py ;;
  sw2) export CODE_ID=61cb584; P=sout; CFG=v1_stress_wave2; VF=verify_stress_wave2.py ;;
  *) echo "unknown study $STUDY"; exit 2 ;;
esac
GROUP=$G source jobs/common.sh
case "$G" in tabpfn|tsr_legacy) nvidia-smi --query-gpu=name,memory.total --format=csv,noheader ;; esac
mapfile -t RUNS < <(find "$P-$G" -mindepth 1 -maxdepth 1 -type d ! -name '*.*' | sort)
echo "$P-$G: ${#RUNS[@]} active shard directories"
OUT=checks-$STUDY-$TAG-reproduce-$G
"$PY" "$CODE/scripts/$VF" reproduce --group "$G" --config "$CODE/configs/${CFG}_$G.toml" \
  --runs "${RUNS[@]}" --output "$OUT.json" > "logs/$OUT.stdout" 2>&1
rc=$?; echo "reproduce exit $rc"
exit $rc
