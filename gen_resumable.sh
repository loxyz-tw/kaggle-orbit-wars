#!/usr/bin/env bash
# Resumable, chunked BC dataset generation from official replays (one date-dir at a time).
# Each day is processed by NSHARDS parallel gen_bc_demos_replay.py shards; a per-day .done
# marker is written on success, so a crash loses only the current day and re-running the SAME
# command skips completed days. Markers/chunks are keyed by date (YYYYMMDD), so adding new
# day-dirs later only generates the new dates.
#
# Expected replay layout: $RD/YYYYMMDD/**/*.json  (one dir per day)
# Output: bc_chunks/<TAG>_<date>_<shard>_c*.npz  (streamed by bc_train_clone.py --stream)
#
#   Usage: LB=<leaderboard.csv> [RD=replays] ./gen_resumable.sh <TAG> [MIN_SCORE=0] [NSHARDS=4] [CHUNK=200000]
#   LB  = public leaderboard CSV (columns Rank,TeamName,Score) used to pick strong demonstrators.
#   Optional GEN_CORES=0-15 pins shards with taskset (default: no pinning).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
mkdir -p logs bc_chunks
TAG="${1:?give a TAG}"
MINSC="${2:-0}"; NSH="${3:-4}"; CHUNK="${4:-200000}"
LB="${LB:?set LB=<leaderboard CSV with Rank,TeamName,Score>}"
RD="${RD:-replays}"

PIN=()
[ -n "${GEN_CORES:-}" ] && PIN=(taskset -c "$GEN_CORES")

# the sorted date-dirs present (gen's --day-slice is a python slice over THIS same sorted list)
mapfile -t DAYS < <(ls -1 "$RD" | grep -E '^[0-9]{8}$' | sort)
ND=${#DAYS[@]}
[ "$ND" -gt 0 ] || { echo "no YYYYMMDD day-dirs under $RD"; exit 1; }
echo "=== resumable gen TAG=$TAG min_score=$MINSC shards=$NSH : $ND days (${DAYS[0]}..${DAYS[$((ND-1))]}) $(date) ==="

for ((i=0; i<ND; i++)); do
  d="${DAYS[$i]}"; mark="bc_chunks/${TAG}_${d}.done"
  if [ -f "$mark" ]; then echo "  [day $i $d] SKIP (done)"; continue; fi
  rm -f bc_chunks/${TAG}_${d}_*.npz                      # clean THIS day's partial only (never other days)
  echo "  [day $i $d] gen $NSH shards $(date +%H:%M:%S)"
  pids=()
  for ((k=0; k<NSH; k++)); do
    XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 \
      "${PIN[@]}" "$PY" gen_bc_demos_replay.py \
        --replays-dir "$RD" --recursive --day-slice="${i}:$((i+1))" --shard "$k/$NSH" \
        --leaderboard "$LB" --min-score "$MINSC" --idle-frac 1.0 \
        --chunk-samples "$CHUNK" --out "bc_chunks/${TAG}_${d}_${k}.npz" \
        >"logs/gen_${TAG}_${d}_${k}.log" 2>&1 &
    pids+=($!)
  done
  fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
  nch=$(ls bc_chunks/${TAG}_${d}_*_c*.npz 2>/dev/null | wc -l)
  if [ "$fail" -eq 0 ] && [ "$nch" -gt 0 ]; then
    touch "$mark"; echo "  [day $i $d] DONE ($nch chunks)"
  else
    echo "  [day $i $d] FAILED (fail=$fail nch=$nch) -> NO marker; rerun resumes here"; exit 1
  fi
done

TOT=$(ls bc_chunks/${TAG}_*_c*.npz 2>/dev/null | wc -l)
echo "=== resumable gen COMPLETE TAG=$TAG : $TOT chunks across $ND days $(date) ==="
