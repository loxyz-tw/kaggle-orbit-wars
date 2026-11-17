#!/usr/bin/env bash
# 2p strength verdict for a checkpoint via the SHIPPING numpy submission path
# (lean_submission_numpy: current-position lead-aim + sun guard), so the number reflects what
# actually ships. Two reads: vs a heuristic opponent you provide, and head-to-head vs a reference
# checkpoint (~50% = ties the reference).
#
# Usage: OW_OPP=<your opponent agent .py> [OW_REF_CK=<reference ckpt>] \
#        ./eval_lean_verdict.sh <ckpt> <tag> [N=120] [CORES=0-11]
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
CK="${1:?ckpt}"; TAG="${2:?tag}"; N="${3:-120}"; CORES="${4:-0-11}"; ML="${5:-}"
OPP="${OW_OPP:?set OW_OPP=<opponent agent .py>}"
REF_CK="${OW_REF_CK:-}"
TMPD="$(mktemp -d)"; trap 'rm -rf "$TMPD"' EXIT
SUB="$TMPD/verdict_${TAG}.py"
REF_SUB="$TMPD/verdict_ref.py"
OUT="eval_${TAG}.txt"; : > "$OUT"

echo "=== build $CK -> $SUB (shipping numpy path)${ML:+ min_launch=$ML} ===" | tee -a "$OUT"
OW_BUILD_MINLAUNCH="$ML" "$PY" submission/_build_lean_submission.py "$CK" "$SUB" > "$TMPD/build.log" 2>&1 \
  || { echo "build FAILED:" >&2; cat "$TMPD/build.log" >&2; exit 1; }
grep -E "min_launch|wrote" "$TMPD/build.log" | tee -a "$OUT"

echo "=== VERDICT $TAG  ckpt=$CK  N=$N  $(date) ===" | tee -a "$OUT"
r=$(taskset -c "$CORES" env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OW_WORKERS=4 \
    OW_AGENT="$SUB" OW_OPP="$OPP" \
    "$PY" eval_2p_full.py "$N" 2>"$TMPD/eval_opp.err" | grep -aoE "win = [0-9]+/[0-9]+ = [0-9.]+%") || r=""
if [ -z "$r" ]; then echo "eval vs_opp FAILED:" >&2; tail -n 30 "$TMPD/eval_opp.err" >&2; exit 1; fi
echo "vs_opp: $r" | tee -a "$OUT"
if [ -n "$REF_CK" ] && [ "$CK" != "$REF_CK" ]; then
  "$PY" submission/_build_lean_submission.py "$REF_CK" "$REF_SUB" > "$TMPD/build_ref.log" 2>&1 \
    || { echo "build ref FAILED:" >&2; cat "$TMPD/build_ref.log" >&2; exit 1; }
  r=$(taskset -c "$CORES" env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OW_WORKERS=8 \
      OW_AGENT="$SUB" OW_OPP="$REF_SUB" \
      "$PY" eval_2p_full.py "$N" 2>"$TMPD/eval_ref.err" | grep -aoE "win = [0-9]+/[0-9]+ = [0-9.]+%") || r=""
  if [ -z "$r" ]; then echo "eval h2h_ref FAILED:" >&2; tail -n 30 "$TMPD/eval_ref.err" >&2; exit 1; fi
  echo "h2h_ref: $r" | tee -a "$OUT"
else
  echo "h2h_ref: skipped (set OW_REF_CK)" | tee -a "$OUT"
fi
echo "=== VERDICT DONE $TAG $(date) ===" | tee -a "$OUT"
