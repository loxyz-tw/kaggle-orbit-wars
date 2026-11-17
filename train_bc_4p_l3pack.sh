#!/usr/bin/env bash
# 4-player BC model ("L3-pack") of the final dispatch submission: entity transformer h768 / 3 layers /
# 8 heads with the two extra 4p columns (is_4p flag + my garrison-rank fraction) -> 27-dim input.
# Trained on MIXED 2p+4p games (2p skill transfers; 4p-only training was clearly worse),
# demonstrators with leaderboard score >= 1520, outcome weighting with linear rank weight, won games
# up-weighted x2, constant lr, no warmup / grad-clip. Operating point = <out>.ep1 (first epoch;
# the 2nd epoch overfit for this model size). RESUME-safe: re-run the same command.
#   Usage: [STREAM='bc_chunks/b15_*_c*.npz'] [OUT=ckpt/bc_4p_l3pack] ./train_bc_4p_l3pack.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
STREAM="${STREAM:-bc_chunks/b15_*_c*.npz}"
OUT="${OUT:-ckpt/bc_4p_l3pack}"
mkdir -p "$(dirname "$OUT")" logs
echo "=== 4p L3-pack train start $(date) ==="
"$PY" bc_train_clone.py --hidden 768 --layers 3 --heads 8 --no-focus --per-step \
    --is-4p-feature --rank-feature \
    --min-score 1520 --stream "$STREAM" \
    --outcome-weight --rank-weight linear --win-weight 2.0 \
    --batch 256 --lr 4e-4 --epochs 2 \
    --ckpt-every-chunks 30 --resume --out "$OUT" > logs/train_bc_4p_l3pack.log 2>&1 \
    || { echo "=== train FAILED (exit $?), see logs/train_bc_4p_l3pack.log ===" >&2; exit 1; }
echo "=== 4p L3-pack train DONE $(date) ==="
