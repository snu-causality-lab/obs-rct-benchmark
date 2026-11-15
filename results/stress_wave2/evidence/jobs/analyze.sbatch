#!/bin/bash
#SBATCH --partition=P2
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --exclude=b03,b14,b15,b24,b25,b26,b27,b28,b30,b31
#SBATCH --chdir=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01
#SBATCH --output=/shared/s2/lab05/sanghack/work/obsrct-v1-wave2-20261114-r01/logs/%x-%j.out
# Analysis and independent arithmetic for one study, with that study's frozen code.
# Usage: sbatch -J analyze-<study> jobs/analyze.sbatch w2|sw2 <tag>
# Runs are the active shard directories only: moved-aside attempts (*.interrupted-N,
# *.provenance-unavailable-N) are kept on disk and never passed.
set -uo pipefail
STUDY=${1:?w2|sw2}; TAG=${2:?tag}
case "$STUDY" in
  w2)  export CODE_ID=80c9129; P=out;  CFG=v1_wave2;        AN=analyze_v1_wave2.py;    VF=verify_v1_wave2.py;    N=1984 ;;
  sw2) export CODE_ID=61cb584; P=sout; CFG=v1_stress_wave2; AN=analyze_stress_wave2.py; VF=verify_stress_wave2.py; N=512 ;;
  *) echo "unknown study $STUDY"; exit 2 ;;
esac
GROUP=native_r source jobs/common.sh
case "$STUDY" in
  w2)  REF=(--wave1 "$CODE/results/v1_core"); VREF=() ;;
  sw2) REF=(--stress1 "$CODE/results/stress_cornet"); VREF=("${REF[@]}") ;;
esac
GROUPS_=(native_r gp tabpfn tsr_legacy)
RUNS=()
for g in "${GROUPS_[@]}"; do
  mapfile -t r < <(find "$P-$g" -mindepth 1 -maxdepth 1 -type d ! -name '*.*' | sort)
  echo "$P-$g: ${#r[@]} active shard directories"
  [ "${#r[@]}" -eq "$N" ] || { echo "expected $N"; exit 3; }
  RUNS+=("${r[@]}")
done
CONFIGS=(); for g in "${GROUPS_[@]}"; do CONFIGS+=("$CODE/configs/${CFG}_$g.toml"); done
OUT=analysis-$STUDY-$TAG
sha256sum "$CODE/scripts/$AN" "$CODE/scripts/$VF" "${CONFIGS[@]}"
"$PY" "$CODE/scripts/$AN" --configs "${CONFIGS[@]}" --runs "${RUNS[@]}" "${REF[@]}" --output "$OUT" > "logs/$OUT.stdout" 2>&1
rc=$?; echo "analyzer exit $rc"; tail -3 "logs/$OUT.stdout"
[ $rc -eq 0 ] || exit $rc
"$PY" "$CODE/scripts/$VF" arithmetic --analysis "$OUT" --runs "${RUNS[@]}" "${VREF[@]}" \
  --output "checks-$STUDY-$TAG-arithmetic.json" > "logs/checks-$STUDY-$TAG-arithmetic.stdout" 2>&1
rc=$?; echo "arithmetic exit $rc"
exit $rc
