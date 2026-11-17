#!/usr/bin/env bash
# 4-player dual-trunk PPO fine-tune ("melee") starting from the 4p BC model (L3-pack, h768/L3, 27-dim).
#   - reward-mode winner (1st = +1, everyone else -1) instead of graded placement: graded reward
#     rewards "not finishing last" and produces turtling.
#   - PBRS expansion shaping (--expansion-shaping --expansion-weight): potential = controlled production.
#   - long horizon (gamma 0.998); actor KL-anchored to the BC model; separate smaller critic trunk.
#   - default = pure N-way self-play. Optional heuristic-opponent mix: set OPP_PROD_FRAC>0 AND
#     PRODUCER_MODULE=<your_heuristic.py> (not shipped; see jax_lean_selfplay.py header).
# Self-contained leg chain: each leg = LEG_M million env-steps; RESUME after a crash by re-running.
#   Usage: BC=<4p BC ckpt, e.g. ckpt/bc_4p_l3pack.ep1> [MAP_BANK=kaggle_map_bank_4p.npz] \
#          ./run_ppo_dual_4p_melee.sh [target_M=10]
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

TARGET_M="${1:-10}"
LEG_M="${LEG_M:-5}"
UPDATES=$(( LEG_M * 1000000 / 8192 ))
BC="${BC:?set BC=<4p BC checkpoint (27-dim, h768/L3)>}"
MAP_BANK="${MAP_BANK:-kaggle_map_bank_4p.npz}"   # build with: python build_map_bank.py --num-agents 4 --out kaggle_map_bank_4p.npz
OPP_PROD_FRAC="${OPP_PROD_FRAC:-0}"
PROD_LEVELS="${PROD_LEVELS:-12,14,18}"
PRODUCER_MODULE="${PRODUCER_MODULE:-}"
EXP_W="${EXP_W:-0.03}"                                # PBRS expansion weight
OUT_TAG="${OUT_TAG:-melee}"
OUT="${OUT:-ckpt/dual_ppo_4p_${OUT_TAG}}"
LOGDIR=logs; mkdir -p "$LOGDIR" "$(dirname "$OUT")"

OPP_ARGS=()
if [ "$OPP_PROD_FRAC" != "0" ]; then
  [ -n "$PRODUCER_MODULE" ] || { echo "OPP_PROD_FRAC>0 needs PRODUCER_MODULE=<your heuristic .py>"; exit 1; }
  OPP_ARGS=(--opponent-producer-frac "$OPP_PROD_FRAC" --producer-levels "$PROD_LEVELS" --producer-module "$PRODUCER_MODULE")
fi

FB="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0"
COMMON=(
  --map-bank "$MAP_BANK" --num-players 4
  "${OPP_ARGS[@]}"
  --expansion-shaping --expansion-weight "$EXP_W"
  --reward-mode winner
  --num-envs 256 --rollout-steps 32 --updates "$UPDATES" --epochs 1 --minibatch-size 256
  --hidden-size 768 --num-layers 3 --num-heads 8
  --rich-features --orbit-features --is-4p-feature --rank-feature --frac-bins "$FB"
  --dual-trunk --critic-hidden 384 --critic-layers 4 --critic-heads 4
  --actor-lr 5e-6 --critic-lr 3e-4 --gamma 0.998 --gae-lambda 0.95
  --kl-coef "${KL_COEF:-0.3}" --bc-ref-ckpt "$BC" --weight-decay 1e-4 --grad-clip 1.0
  --checkpoint-interval "$UPDATES" --seed 20260623
)
WARMUP_LEG1=320

run_leg () {
  echo "[chain] leg $1 -> $2 (warmup=$3) $(date)"
  "$PY" jax_lean_selfplay.py "${COMMON[@]}" --warmup-critic-updates "$3" \
    --init-checkpoint "$1" --checkpoint "$2" > "$4" 2>&1
}

echo "=== dual PPO 4p MELEE start: ${LEG_M}M/leg to ${TARGET_M}M, base=$BC, PBRS w=$EXP_W, opp_frac=${OPP_PROD_FRAC} $(date) ==="
prev="$BC"
for (( M=LEG_M; M<=TARGET_M; M+=LEG_M )); do
  out="${OUT}_${M}m"
  if [ "$M" -eq "$LEG_M" ]; then warmup="$WARMUP_LEG1"; else warmup=0; fi
  until [ -f "$out" ]; do
    if ! pgrep -f "[j]ax_lean_selfplay.*${out}" >/dev/null; then
      echo "[chain] leg ${M}M (re)launch from $prev $(date)"
      run_leg "$prev" "$out" "$warmup" "$LOGDIR/dual4p_${OUT_TAG}_${M}m.log"
      sleep 5
    fi
    sleep 30
  done
  echo "[chain] leg ${M}M DONE ($out) $(date)"
  prev="$out"
done
echo "=== dual PPO 4p MELEE COMPLETE: ${prev} $(date) ==="
